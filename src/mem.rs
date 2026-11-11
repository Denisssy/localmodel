//! Where every byte of a model is going to sit.
//!
//! `fit::for_gguf` answers "does it fit". This module answers the next
//! question: *which* part sits on which card, and which part stays in host
//! RAM. It reads the per-block tensor sizes from the GGUF table (see
//! `gguf::Meta::block_bytes`), applies the same rules llama.cpp uses:
//!
//!   * `-ngl N` offloads the **last** N blocks;
//!   * `--n-cpu-moe N` keeps the routed experts (`*_exps`) of the **first**
//!     N blocks in host RAM;
//!   * `token_embd` stays in host memory, the output tensor follows the GPU
//!     layers;
//!   * `--no-kv-offload` keeps the KV cache in host RAM;
//!   * `--split-mode layer` splits contiguous block ranges between cards,
//!     `row`/`tensor` split every block (KV stays on the main card for
//!     `row`), `none` uses the first card only.
//!
//! Everything is an estimate, and the card labels it as such. The dashboard
//! replaces it with llama.cpp's own memory-breakdown log once the model has
//! loaded (`telemetry::LoadInfo`).

use crate::fit;
use crate::gguf;
use crate::models::Model;
use crate::Cfg;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KvHome {
    Vram,
    Ram,
}

impl KvHome {
    pub fn label(self) -> &'static str {
        match self {
            KvHome::Vram => "VRAM",
            KvHome::Ram => "host RAM",
        }
    }
}

/// One destination: a GPU or the host.
#[derive(Clone, Default)]
pub struct Bucket {
    pub weights_gb: f64,
    /// Main model KV.
    pub kv_gb: f64,
    /// Draft/MTP KV, kept separate only on the host row.
    pub draft_kv_gb: f64,
    pub overhead_gb: f64,
    /// Whole blocks that live here (inclusive ranges).
    pub layers: Vec<(u32, u32)>,
    /// Blocks whose routed experts live here (`--n-cpu-moe`).
    pub expert_layers: Vec<(u32, u32)>,
}

impl Bucket {
    pub fn total_gb(&self) -> f64 {
        self.weights_gb + self.kv_gb + self.draft_kv_gb + self.overhead_gb
    }

    /// `0-24, 31` or `--`.
    pub fn layers_text(&self) -> String {
        ranges_text(&self.layers)
    }

    pub fn experts_text(&self) -> String {
        // Expert blocks are usually scattered singletons (Mamba/MoE hybrids
        // have only a few expert layers among many blocks). A full list is
        // longer than any card row, so show the first few and a count.
        ranges_text_capped(&self.expert_layers, 6)
    }
}

fn ranges_text(v: &[(u32, u32)]) -> String {
    if v.is_empty() {
        return "--".to_string();
    }
    v.iter()
        .map(|(a, b)| if a == b { a.to_string() } else { format!("{a}-{b}") })
        .collect::<Vec<_>>()
        .join(", ")
}

fn ranges_text_capped(v: &[(u32, u32)], max: usize) -> String {
    if v.is_empty() {
        return "--".to_string();
    }
    if v.len() <= max {
        return ranges_text(v);
    }
    let head = v[..max]
        .iter()
        .map(|(a, b)| if a == b { a.to_string() } else { format!("{a}-{b}") })
        .collect::<Vec<_>>()
        .join(", ");
    format!("{head}, … +{}", v.len() - max)
}

fn push_range(v: &mut Vec<(u32, u32)>, a: u32, b: u32) {
    if a > b {
        return;
    }
    if let Some(last) = v.last_mut() {
        if last.1 + 1 == a {
            last.1 = b;
            return;
        }
    }
    v.push((a, b));
}

/// A draft model on disk (external MTP/DFlash head). The built-in nextn head
/// needs no entry here: it is already inside the main file.
pub struct Draft<'a> {
    pub meta: &'a gguf::Meta,
    pub weights: u64,
}

pub struct Plan {
    pub block_count: u32,
    /// All weights incl. input embedding and output tensor, load factor applied.
    pub total_weights_gb: f64,
    pub input_gb: f64,
    pub output_gb: f64,
    pub kv_main_gb: f64,
    pub kv_draft_gb: f64,
    /// Draft cache types (`f16/f16` unless `extra` overrides them).
    pub draft_types: String,
    pub overhead_gb: f64,
    pub gpus: Vec<Bucket>,
    pub host: Bucket,
    pub kv_home: KvHome,
    pub split_mode: String,
    pub ngl: u32,
    pub ncmoe: u32,
    /// True when the per-tensor table was missing and the block sizes had to
    /// be divided evenly. The card then says "approx".
    pub approx: bool,
    pub note: Option<String>,
}

impl Plan {
    /// Whole model footprint on the cards.
    pub fn vram_gb(&self) -> f64 {
        self.gpus.iter().map(Bucket::total_gb).sum()
    }

    /// One row per card and one for the host.
    pub fn place_rows(&self) -> Vec<(String, String)> {
        let mut v = Vec::new();
        for (i, g) in self.gpus.iter().enumerate() {
            let mut parts = vec![format!("weights {:.2}", g.weights_gb)];
            let kv = g.kv_gb + g.draft_kv_gb;
            if kv > 0.01 {
                parts.push(format!("KV {kv:.2}"));
            }
            if g.overhead_gb > 0.01 {
                parts.push(format!("ovh {:.2}", g.overhead_gb));
            }
            if !g.layers.is_empty() {
                parts.push(format!("layers {}", g.layers_text()));
            }
            v.push((format!("GPU{i}"), format!("{} GiB", parts.join(" · "))));
        }
        if self.host.total_gb() > 0.01 {
            let mut parts = vec![format!("weights {:.2} GiB", self.host.weights_gb)];
            let kv = self.host.kv_gb + self.host.draft_kv_gb;
            if kv > 0.01 {
                parts.push(format!("KV {kv:.2}"));
            }
            if !self.host.layers.is_empty() {
                parts.push(format!("layers {}", self.host.layers_text()));
            }
            if !self.host.expert_layers.is_empty() {
                let n: u32 = self
                    .host
                    .expert_layers
                    .iter()
                    .map(|(a, b)| b - a + 1)
                    .sum();
                parts.push(format!("experts {n}: {}", self.host.experts_text()));
            }
            v.push(("Host".into(), parts.join(" · ")));
        }
        v
    }
}

/// Parse `--tensor-split 0.6,0.4`, normalized. Empty means equal shares.
fn fractions(spec: Option<&str>, n: usize) -> Vec<f64> {
    let mut v: Vec<f64> = spec
        .unwrap_or("")
        .split(',')
        .filter_map(|x| x.trim().parse::<f64>().ok())
        .filter(|x| *x >= 0.0)
        .collect();
    if v.len() != n || v.iter().sum::<f64>() <= 0.0 {
        return vec![1.0 / n as f64; n];
    }
    let sum: f64 = v.iter().sum();
    for x in &mut v {
        *x /= sum;
    }
    v
}

/// Plan the placement of `m` with the effective `cfg`.
///
/// `gpus` is the number of cards and `room` their usable memory; both come
/// from `fit::Hardware`. Pass `Draft::None` for a model without a draft head.
pub fn plan(m: &Model, draft: Option<Draft>, cfg: &Cfg, hw: &fit::Hardware) -> Plan {
    let n = m.meta.block_count.max(m.layers).max(1);
    let n_usize = n as usize;
    let meta = &m.meta;

    // Per-block bytes. Fall back to an even split when the tensor table is
    // empty (a file written by an older parser, or a test model).
    let mut approx = false;
    let block_bytes: Vec<f64> = if meta.block_bytes.len() >= n_usize
        && meta.block_bytes.iter().any(|b| *b > 0)
    {
        meta.block_bytes.iter().take(n_usize).map(|b| *b as f64).collect()
    } else {
        approx = true;
        let input = meta.input_bytes as f64;
        let output = meta.output_bytes as f64;
        let rest = (m.size as f64 - input - output).max(0.0);
        vec![rest / n as f64; n_usize]
    };
    let expert_bytes: Vec<f64> = (0..n_usize)
        .map(|i| meta.block_expert_bytes.get(i).copied().unwrap_or(0) as f64)
        .collect();

    let load = fit::WEIGHT_LOAD_FACTOR;
    let input_gb = meta.input_bytes as f64 * load / GIB;
    let output_gb = if meta.output_bytes > 0 {
        meta.output_bytes as f64 * load / GIB
    } else {
        0.0
    };
    let total_weights_gb: f64 = (block_bytes.iter().sum::<f64>()
        + meta.input_bytes as f64
        + meta.output_bytes as f64)
        * load
        / GIB;

    // -ngl offloads from the end of the model.
    let want_ngl = cfg.ngl.unwrap_or_else(|| {
        // Matches `build_args`: Nemotron defaults to 999, everything else 99.
        if m.name.to_ascii_lowercase().contains("nemotron") { 999 } else { 99 }
    }) as u32;
    let ngl = want_ngl.min(n);
    let gpu_start = n.saturating_sub(ngl);
    let ncmoe = if m.moe {
        (cfg.ncmoe.unwrap_or(58) as u32).min(n)
    } else {
        0
    };

    // KV cache. `fit::kv_usage_gb` already knows about sliding windows and
    // asymmetric K/V types; reuse it for the main model. The draft has its
    // own types (`-ctkd/-ctvd`, default f16) — resolved from the knobs, the
    // `extra` flags, or the engine default — so `fit::draft_kv_types` takes
    // the whole `cfg`.
    let (draft_ctk, draft_ctv) = fit::draft_kv_types(cfg);
    let builtin = m.nextn.then(|| fit::DraftKv {
        layers: &meta.kv.draft,
        swa_window: meta.kv.swa_window,
        weights_bytes: 0,
        ctk: &draft_ctk,
        ctv: &draft_ctv,
    });
    let external = draft.as_ref().map(|d| fit::DraftKv {
        layers: &d.meta.kv.main,
        swa_window: d.meta.kv.swa_window,
        weights_bytes: d.weights,
        ctk: &draft_ctk,
        ctv: &draft_ctv,
    });
    let (kv_main_gb, kv_builtin_draft_gb) =
        fit::kv_usage_gb(meta, builtin.as_ref(), cfg.ctx, &cfg.ctk, &cfg.ctv);
    let (_, kv_external_draft_gb) = match external.as_ref() {
        Some(d) => fit::kv_usage_gb(meta, Some(d), cfg.ctx, &cfg.ctk, &cfg.ctv),
        None => (0.0, 0.0),
    };
    let kv_draft_gb = kv_builtin_draft_gb + kv_external_draft_gb;
    let draft_types = format!("{draft_ctk}/{draft_ctv}");
    let draft_weights_gb = draft
        .as_ref()
        .map(|d| d.weights as f64 * load / GIB)
        .unwrap_or(0.0);

    let kv_home = if cfg.kv_ram { KvHome::Ram } else { KvHome::Vram };
    let nb = hw.gpus.len();
    let split_mode = cfg.split_mode.clone();

    // Build a bucket per card. With no card everything lands in host RAM, so
    // the screen still explains why nothing is on a GPU.
    let mut buckets: Vec<Bucket> = (0..nb).map(|_| Bucket::default()).collect();
    let mut host = Bucket::default();

    if nb == 0 {
        // No GPU: the whole thing is host memory.
        host.weights_gb = total_weights_gb + draft_weights_gb;
        host.kv_gb = kv_main_gb;
        host.draft_kv_gb = kv_draft_gb;
        host.layers = vec![(0, n.saturating_sub(1))];
        return Plan {
            block_count: n,
            total_weights_gb,
            input_gb,
            output_gb,
            kv_main_gb,
            kv_draft_gb,
            draft_types,
            overhead_gb: 0.0,
            gpus: buckets,
            host,
            kv_home,
            split_mode,
            ngl,
            ncmoe,
            approx,
            note: Some("no CUDA device: everything runs from host RAM".into()),
        };
    }

    // Which blocks are wholly on a GPU, and which of their experts stay home.
    let mut gpu_blocks: Vec<(u32, f64, f64)> = Vec::new(); // (block, vram_gb, host_expert_gb)
    let mut host_weights = input_gb;
    for b in 0..n {
        let w = block_bytes[b as usize] * load / GIB;
        let e = expert_bytes[b as usize] * load / GIB;
        if b < gpu_start {
            host_weights += w;
            push_range(&mut host.layers, b, b);
            continue;
        }
        let mut gpu_w = w;
        if b < ncmoe && e > 0.0 {
            gpu_w -= e;
            host_weights += e;
            push_range(&mut host.expert_layers, b, b);
        }
        gpu_blocks.push((b, gpu_w, 0.0));
    }
    // The output tensor rides with the last GPU layer.
    let gpu_output = if ngl > 0 { output_gb } else { 0.0 };
    if ngl == 0 {
        host_weights += output_gb;
    }

    // External draft placement follows `build_args`: dense models keep it on
    // the CPU unless `draft_gpu`, while the Nemotron and MoE branches always
    // offload it (`--spec-draft-ngl 999`).
    let draft_on_gpu = draft.is_some() && (cfg.draft_gpu || m.moe);
    let mut host_draft_weights = 0.0;
    if external.is_some() && !draft_on_gpu {
        host_draft_weights = draft_weights_gb;
        host_weights += draft_weights_gb;
    }

    let kv_gpu_main = if kv_home == KvHome::Ram {
        0.0
    } else if ngl >= n {
        kv_main_gb
    } else {
        kv_main_gb * ngl as f64 / n as f64
    };
    let kv_host_main = kv_main_gb - kv_gpu_main;
    // Built-in MTP KV is part of the target context; an external draft's KV
    // follows the draft itself.
    let mut kv_gpu_draft = 0.0;
    if kv_home == KvHome::Vram {
        if m.nextn {
            kv_gpu_draft += kv_builtin_draft_gb;
        }
        if draft_on_gpu {
            kv_gpu_draft += kv_external_draft_gb;
        }
    }
    let kv_host_draft = kv_draft_gb - kv_gpu_draft;

    let gpu_weight_total: f64 = gpu_blocks.iter().map(|(_, w, _)| *w).sum::<f64>()
        + gpu_output
        + if draft_on_gpu { draft_weights_gb } else { 0.0 };
    let frac = fractions(cfg.tensor_split.as_deref(), nb);
    let total_gpu_kv = kv_gpu_main + kv_gpu_draft;

    match split_mode.as_str() {
        "none" => {
            let g = &mut buckets[0];
            g.weights_gb = gpu_weight_total;
            g.kv_gb = kv_gpu_main;
            g.draft_kv_gb = kv_gpu_draft;
            for (b, _, _) in &gpu_blocks {
                push_range(&mut g.layers, *b, *b);
            }
            for g in buckets.iter_mut().skip(1) {
                g.overhead_gb = 0.0;
            }
        }
        "row" | "tensor" => {
            for (i, g) in buckets.iter_mut().enumerate() {
                g.weights_gb = gpu_weight_total * frac[i];
                // Row split keeps the KV on the main card; tensor split divides it.
                if split_mode == "row" {
                    if i == 0 {
                        g.kv_gb = kv_gpu_main;
                        g.draft_kv_gb = kv_gpu_draft;
                    }
                } else {
                    g.kv_gb = kv_gpu_main * frac[i];
                    g.draft_kv_gb = kv_gpu_draft * frac[i];
                }
                // Every block is touched by every card in these modes.
                if !gpu_blocks.is_empty() {
                    let a = gpu_blocks.first().map(|x| x.0).unwrap_or(0);
                    let b = gpu_blocks.last().map(|x| x.0).unwrap_or(0);
                    push_range(&mut g.layers, a, b);
                }
            }
        }
        _ => {
            // layer split: contiguous ranges, proportional to the target share.
            let targets: Vec<f64> = frac.iter().map(|f| gpu_weight_total * f).collect();
            let mut gi = 0usize;
            let mut acc = 0.0;
            for (b, w, _) in &gpu_blocks {
                // Move to the next card once this one has reached its share,
                // but never leave the last card empty when there are blocks.
                while gi + 1 < nb && acc >= targets[gi] && acc > 0.0 {
                    gi += 1;
                    acc = 0.0;
                }
                buckets[gi].weights_gb += *w;
                push_range(&mut buckets[gi].layers, *b, *b);
                acc += *w;
            }
            // Output tensor and a GPU draft go to the last card that has work.
            let last = buckets
                .iter()
                .rposition(|g| !g.layers.is_empty())
                .unwrap_or(0);
            buckets[last].weights_gb += gpu_output;
            if draft_on_gpu {
                buckets[last].weights_gb += draft_weights_gb;
            }
            // KV follows the blocks; use the same share as the weights.
            if total_gpu_kv > 0.0 {
                let share: Vec<f64> = buckets
                    .iter()
                    .map(|g| if gpu_weight_total > 0.0 { g.weights_gb / gpu_weight_total } else { 0.0 })
                    .collect();
                for (i, g) in buckets.iter_mut().enumerate() {
                    g.kv_gb = kv_gpu_main * share[i];
                    g.draft_kv_gb = kv_gpu_draft * share[i];
                }
            }
        }
    }

    // Overhead: one CUDA context per card that holds anything, compute buffer
    // proportional to that card's weights.
    for (i, g) in buckets.iter_mut().enumerate() {
        let used = g.weights_gb > 0.0 || g.kv_gb > 0.0;
        if used {
            let share = if gpu_weight_total > 0.0 { g.weights_gb / gpu_weight_total } else { frac[i] };
            g.overhead_gb = fit::CUDA_CTX_GB + fit::COMPUTE_GB * share;
        }
    }

    host.weights_gb = host_weights + host_draft_weights;
    host.kv_gb = kv_host_main;
    host.draft_kv_gb = kv_host_draft;

    let mut note = None;
    if approx {
        note = Some("per-tensor table missing: block sizes are an even split".into());
    }
    if nb > 1 && !matches!(split_mode.as_str(), "layer" | "row" | "tensor" | "none") {
        note = Some(format!("unknown split mode '{split_mode}', using layer"));
    }

    let overhead_gb = buckets.iter().map(|g| g.overhead_gb).sum();
    Plan {
        block_count: n,
        total_weights_gb,
        input_gb,
        output_gb,
        kv_main_gb,
        kv_draft_gb,
        draft_types,
        overhead_gb,
        gpus: buckets,
        host,
        kv_home,
        split_mode,
        ngl,
        ncmoe,
        approx,
        note,
    }
}

/// Largest standard context that still fits once `-ngl` / `--n-cpu-moe`
/// placement is applied.
///
/// `fit::suggest_ctx` answers "does the whole model fit"; for a MoE model with
/// experts deliberately kept in host RAM that says no even when the actual
/// placement fits comfortably. This walks the same context ladder using the
/// real per-block placement instead.
pub fn max_ctx(m: &Model, draft: Option<Draft>, cfg: &Cfg, hw: &fit::Hardware) -> u32 {
    const STEPS: &[u32] = &[4096, 8192, 16384, 32768, 65536, 131072, 196608, 262144];
    if hw.vram_total_gb <= 0.0 {
        return 0;
    }
    let cap = if m.meta.ctx_train > 0 { m.meta.ctx_train } else { u32::MAX };
    let room = hw.vram_room_gb();
    for &step in STEPS.iter().rev() {
        if step > cap {
            continue;
        }
        let mut c = cfg.clone();
        c.ctx = step;
        let d = draft.as_ref().map(|d| Draft { meta: d.meta, weights: d.weights });
        if plan(m, d, &c, hw).vram_gb() <= room {
            return step;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{KvGeometry, KvLayer};
    use crate::models::Model;
    use std::path::PathBuf;

    fn meta(blocks: usize) -> gguf::Meta {
        gguf::Meta {
            layers: blocks as u32,
            block_count: blocks as u32,
            nextn: true,
            arch: "qwen35".into(),
            ctx_train: 262144,
            nextn_layers: 1,
            kv: KvGeometry {
                main: vec![KvLayer { heads: 4, k: 256, v: 256, swa: false }; 8],
                draft: vec![KvLayer { heads: 4, k: 256, v: 256, swa: false }],
                swa_window: 0,
            },
            block_bytes: vec![200 << 20; blocks],
            block_expert_bytes: vec![0; blocks],
            input_bytes: 150 << 20,
            output_bytes: 150 << 20,
            ..Default::default()
        }
    }

    fn model(blocks: usize) -> Model {
        let size = (blocks as u64 * 200 + 300) << 20;
        Model {
            path: PathBuf::from("/nonexistent/m.gguf"),
            name: "m.gguf".into(),
            size,
            quant: "Q6_K".into(),
            layers: blocks as u32,
            moe: false,
            experts_total: 0,
            experts_active: 0,
            nextn: true,
            meta: meta(blocks),
        }
    }

    fn hw(cards: usize, gb: f64) -> fit::Hardware {
        fit::Hardware {
            ram_total_gb: 64.0,
            ram_avail_gb: 32.0,
            vram_total_gb: gb * cards as f64,
            disk_free_gb: 2000.0,
            gpus: (0..cards).map(|i| (format!("GPU{i}"), gb)).collect(),
        }
    }

    fn cfg() -> Cfg {
        Cfg {
            ctx: 32768,
            port: None,
            ngl: None,
            ncmoe: None,
            ctk: "q8_0".into(),
            ctv: "q8_0".into(),
            draft_ctk: String::new(),
            draft_ctv: String::new(),
            threads: None,
            reasoning: None,
            effort: None,
            draft_tokens: None,
            extra: None,
            tools: None,
            draft_gpu: false,
            vision: false,
            mmproj: None,
            alias: None,
            flash_attn: true,
            temp: 1.0,
            top_p: 0.95,
            host: "127.0.0.1".into(),
            kv_ram: false,
            split_mode: "layer".into(),
            tensor_split: None,
            log_verbosity: 4,
            ..Cfg::default()
        }
    }

    #[test]
    fn one_card_gets_all_layers_and_kv() {
        let p = plan(&model(8), None, &cfg(), &hw(1, 24.0));
        assert_eq!(p.gpus.len(), 1);
        assert_eq!(p.gpus[0].layers_text(), "0-7");
        assert!(p.host.weights_gb > 0.0, "input embedding stays in RAM");
        assert!(p.host.kv_gb < 0.001);
        assert!(p.vram_gb() > 1.0);
    }

    #[test]
    fn ngl_offloads_the_last_layers() {
        let mut c = cfg();
        c.ngl = Some(3);
        let p = plan(&model(8), None, &c, &hw(1, 24.0));
        assert_eq!(p.gpus[0].layers_text(), "5-7");
        assert_eq!(p.host.layers_text(), "0-4");
    }

    #[test]
    fn kv_in_ram_moves_the_cache_off_the_card() {
        let mut c = cfg();
        c.kv_ram = true;
        let p = plan(&model(8), None, &c, &hw(1, 24.0));
        assert_eq!(p.kv_home, KvHome::Ram);
        assert!(p.gpus[0].kv_gb < 0.001);
        assert!(p.host.kv_gb > 0.0);
    }

    #[test]
    fn two_cards_split_layers_and_kv() {
        let mut c = cfg();
        c.tensor_split = Some("0.5,0.5".into());
        let p = plan(&model(8), None, &c, &hw(2, 24.0));
        assert_eq!(p.gpus.len(), 2);
        assert!(p.gpus[0].weights_gb > 0.0 && p.gpus[1].weights_gb > 0.0);
        assert_eq!(p.gpus.iter().map(|g| g.layers.len()).sum::<usize>(), 2);
        assert!(p.gpus[0].overhead_gb > 0.0 && p.gpus[1].overhead_gb > 0.0);
    }

    #[test]
    fn moe_external_draft_rides_the_gpu() {
        let mut m = model(8);
        m.moe = true;
        m.meta.block_expert_bytes = vec![150 << 20; 8];
        let without = plan(&m, None, &cfg(), &hw(1, 24.0));
        let meta = m.meta.clone();
        let with = plan(
            &m,
            Some(Draft { meta: &meta, weights: 200 << 20 }),
            &cfg(),
            &hw(1, 24.0),
        );
        let dg = with.gpus[0].weights_gb - without.gpus[0].weights_gb;
        assert!(dg > 0.15, "draft weights on the card: {dg}");
        let dh = with.host.weights_gb - without.host.weights_gb;
        assert!(dh < 0.01, "no draft weights in RAM: {dh}");
    }

    #[test]
    fn long_expert_lists_are_capped_for_the_card() {
        let b = Bucket {
            expert_layers: (0..20).map(|i| (i * 2, i * 2)).collect(),
            ..Bucket::default()
        };
        let t = b.experts_text();
        assert!(t.ends_with("+14"), "{t}");
        assert!(t.chars().count() < 48, "{t} is too long for a card row");
        assert!(!t.contains("18"), "the tail must be hidden: {t}");
    }

    #[test]
    fn ncmoe_keeps_first_experts_in_ram() {
        let mut m = model(8);
        m.moe = true;
        m.meta.block_expert_bytes = vec![150 << 20; 8];
        let mut c = cfg();
        c.ncmoe = Some(2);
        let p = plan(&m, None, &c, &hw(1, 24.0));
        assert_eq!(p.host.experts_text(), "0-1");
        assert!(p.host.weights_gb > 0.0);
        assert_eq!(p.gpus[0].layers_text(), "0-7");
    }

    #[test]
    fn max_ctx_accounts_for_experts_kept_in_ram() {
        let mut m = model(8);
        m.moe = true;
        m.meta.block_expert_bytes = vec![150 << 20; 8];
        let h = hw(1, 4.0);
        let mut c = cfg();
        c.ncmoe = Some(8);
        let with = max_ctx(&m, None, &c, &h);
        let mut c0 = cfg();
        c0.ncmoe = Some(0);
        let without = max_ctx(&m, None, &c0, &h);
        assert!(with > 0, "placement fits with experts in RAM");
        assert!(
            without < with,
            "whole model on the card fits less context: {without} vs {with}"
        );
    }
}
