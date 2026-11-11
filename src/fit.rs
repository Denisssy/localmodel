//! "Will it fit?" estimates for models on this machine.
//!
//! Two calculators live here:
//!
//!   * `estimate` reads the published Strata tables and models the hybrid
//!     VRAM/RAM split those packs use:
//!
//!       * Qwen3.6-35B-A3B runs in hybrid mode: the engine keeps every expert
//!         it can in VRAM and computes the rest from RAM.
//!       * Qwen3.8-Flash-Next keeps shard 1's experts in RAM; the GPU holds
//!         the dense part. Without enough RAM it only runs in low-RAM mode,
//!         which the docs say needs 32 GB of RAM and a 24 GB card (or the
//!         Coder pack).
//!       * Gyro is quoted as VRAM + RAM together.
//!
//!   * `for_gguf` reads the file's own GGUF metadata: block count, attention
//!     layer layout, KV heads, head dimensions and sliding windows. That is
//!     what the hub uses before a model has ever been launched.
//!
//! The Strata numbers are published ones from docs/MODELS.md and
//! docs/QWEN36.md. The GGUF constants are calibrated on this box (see
//! `COMPUTE_GB`).
//!
//! Everything here is an estimate. It is deliberately conservative: a "fits"
//! means the model should load with the desktop running, not that it will be
//! fast.
use std::path::Path;

use crate::Cfg;
use crate::gguf;
use crate::server;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// GGUF weights grow a little when loaded (alignment, duplicated tied
/// embeddings, the output tensor). Measured 7.04 GiB file → ~7.11 GiB resident.
pub(crate) const WEIGHT_LOAD_FACTOR: f64 = 1.01;
/// Compute buffers (graph, logits, batch) and the CUDA context, together.
/// Calibrated 2026-11-08 on this box against Ornith-1.5-9B-Q6_K at 262144
/// tokens, q8_0/q8_0 with the built-in MTP head: weights 7.11 + KV 4.78 +
/// 1.55 = 13.44 GiB, while llama-server reported 14009 MiB (13.68 GiB).
/// The 0.24 GiB gap is inside the "fits" margin on purpose; the verdict is
/// meant to be conservative, not exact to the last frame.
pub(crate) const COMPUTE_GB: f64 = 1.25;
pub(crate) const CUDA_CTX_GB: f64 = 0.30;/// Less than this much free VRAM after loading counts as "tight".
const TIGHT_GB: f64 = 1.0;

/// Bytes per KV-cache element for a llama.cpp `--cache-type-*` name.
///
/// Block formats from ggml: q8_0 packs 32 values in 34 bytes, q5_1 in 24,
/// q5_0 in 22, q4_1 in 20, q4_0/iq4_nl in 18; q6_k is 210 bytes per 256.
/// f16/bf16 are plain 2-byte elements.
pub fn kv_type_bytes(name: &str) -> f64 {
    match name.trim().to_ascii_lowercase().as_str() {
        "f16" | "fp16" | "bf16" => 2.0,
        "q8_0" => 1.0625,
        "q6_k" => 0.820_3125,
        "q5_1" => 0.75,
        "q5_0" => 0.6875,
        "q4_1" => 0.625,
        "q4_0" | "iq4_nl" => 0.5625,
        _ => 2.0,
    }
}

/// A draft context to add on top of the main model.
///
/// For a built-in MTP head pass `meta.kv.draft` with `weights_bytes = 0`
/// (the head is already inside the main file). For an external draft model
/// pass that file's `kv.main` and its file size.
#[derive(Clone, Copy)]
pub struct DraftKv<'a> {
    pub layers: &'a [gguf::KvLayer],
    pub swa_window: u32,
    pub weights_bytes: u64,
    pub ctk: &'a str,
    pub ctv: &'a str,
}

/// Scan a raw `extra` string for the draft cache-type override flags.
fn scan_extra(extra: &str, keys: &[&str]) -> Option<String> {
    let toks: Vec<&str> = extra.split_whitespace().collect();
    toks.iter()
        .enumerate()
        .find_map(|(i, t)| {
            if let Some((k, v)) = t.split_once('=') {
                if keys.contains(&k) && !v.is_empty() {
                    return Some(v.to_string());
                }
            }
            if keys.contains(t) {
                if let Some(v) = toks.get(i + 1) {
                    return Some((*v).to_string());
                }
            }
            None
        })
}

/// Resolve the draft / MTP context cache types.
///
/// The draft has its own flags (`-ctkd` / `-ctvd`, long forms
/// `--cache-type-k-draft` / `--cache-type-v-draft`) and they default to
/// `f16`; the draft does **not** inherit the target's `--cache-type-k/v`.
/// Priority: the explicit `draft_ctk`/`draft_ctv` knobs > a matching flag in
/// the profile's `extra` > llama.cpp's own default (`f16`). The draft cache
/// types only matter for a model that actually has a draft head, so the hub
/// keeps calling this for every model — an empty result simply renders as
/// `f16/f16`.
pub fn draft_kv_types(c: &Cfg) -> (String, String) {
    // Priority: the explicit knob > a matching `extra` flag > the f16 engine
    // default. `engine` resolves a single flag pair from `extra`.
    let engine = |extra: Option<&str>, flag: &[&str]| -> String {
        scan_extra(extra.unwrap_or(""), flag).unwrap_or_else(|| "f16".into())
    };
    let k_flags = ["-ctkd", "--cache-type-k-draft", "--spec-draft-type-k"];
    let v_flags = ["-ctvd", "--cache-type-v-draft", "--spec-draft-type-v"];
    let k = if c.draft_ctk.is_empty() {
        engine(c.extra.as_deref(), &k_flags)
    } else {
        c.draft_ctk.clone()
    };
    let v = if c.draft_ctv.is_empty() {
        engine(c.extra.as_deref(), &v_flags)
    } else {
        c.draft_ctv.clone()
    };
    (k, v)
}

/// KV bytes at `ctx` for one set of layers. Sliding-window layers stop
/// growing once the window is full; that is why a 262K Gemma context does not
/// cost 256× the KV of a 1K one.
fn layer_bytes(
    layers: &[gguf::KvLayer],
    swa_window: u32,
    ctx: u32,
    k_bpe: f64,
    v_bpe: f64,
) -> f64 {
    layers
        .iter()
        .map(|l| {
            let tokens = if l.swa && swa_window > 0 {
                swa_window.min(ctx) as f64
            } else {
                ctx as f64
            };
            let heads = l.heads as f64;
            tokens * heads * (l.k as f64 * k_bpe + l.v as f64 * v_bpe)
        })
        .sum()
}

pub(crate) fn kv_usage_gb(
    meta: &gguf::Meta,
    draft: Option<&DraftKv>,
    ctx: u32,
    ctk: &str,
    ctv: &str,
) -> (f64, f64) {
    let main = layer_bytes(
        &meta.kv.main,
        meta.kv.swa_window,
        ctx,
        kv_type_bytes(ctk),
        kv_type_bytes(ctv),
    ) / GIB;
    let draft = draft
        .map(|d| {
            layer_bytes(
                d.layers,
                d.swa_window,
                ctx,
                kv_type_bytes(d.ctk),
                kv_type_bytes(d.ctv),
            ) / GIB
        })
        .unwrap_or(0.0);
    (main, draft)
}

/// VRAM estimate computed from the GGUF metadata itself.
pub struct GgufEstimate {
    pub verdict: Verdict,
    pub ctx: u32,
    pub weights_gb: f64,
    pub kv_main_gb: f64,
    pub kv_draft_gb: f64,
    pub overhead_gb: f64,
    pub need_vram_gb: f64,
    /// VRAM the card may use for planning (total minus the desktop baseline).
    pub room_gb: f64,
    /// Average KV cost of one token at this context, for display.
    pub kv_gb_per_token: f64,
    /// One-line explanation; the hub renders its own breakdown and the tests
    /// assert on this wording.
    #[allow(dead_code)]
    pub why: String,
}

/// Estimate VRAM for a GGUF file from its own metadata.
///
/// `weights_bytes` is the file size (for a sharded model, the sum of parts).
/// `draft` accounts for the MTP/nextn context; pass `None` when speculative
/// decoding is off.
///
/// Assumes every layer is on the GPU (`-ngl 99`). For an MoE model running
/// with `--n-cpu-moe`, subtract the offloaded expert bytes from
/// `weights_bytes` before calling; the per-tensor split belongs to the hub.
pub fn for_gguf(
    meta: &gguf::Meta,
    weights_bytes: u64,
    hw: &Hardware,
    ctx: u32,
    ctk: &str,
    ctv: &str,
    draft: Option<DraftKv<'_>>,
    kv_ram: bool,
) -> GgufEstimate {
    let draft_weights = draft.map(|d| d.weights_bytes).unwrap_or(0);
    let weights_gb = (weights_bytes.saturating_add(draft_weights)) as f64 / GIB * WEIGHT_LOAD_FACTOR;
    let (kv_main_gb, kv_draft_gb) = kv_usage_gb(meta, draft.as_ref(), ctx, ctk, ctv);
    let overhead_gb = COMPUTE_GB + CUDA_CTX_GB;
    // `--no-kv-offload` keeps the cache in host memory, so it must not count
    // against the card even though it still costs the same bytes.
    let kv_vram = if kv_ram { 0.0 } else { kv_main_gb + kv_draft_gb };
    let need_vram_gb = weights_gb + kv_vram + overhead_gb;
    let room_gb = hw.vram_room_gb();

    let (verdict, why) = if hw.vram_total_gb <= 0.0 {
        (Verdict::No, "no GPU memory detected".to_string())
    } else if !meta.kv.has_kv() {
        (
            Verdict::No,
            "KV geometry is missing from the GGUF metadata".to_string(),
        )
    } else {
        let slack = room_gb - need_vram_gb;
        let verdict = if slack >= TIGHT_GB {
            Verdict::Fits
        } else if slack >= 0.0 {
            Verdict::Tight
        } else {
            Verdict::No
        };
        let mut why = format!(
            "{:.1} GB weights + {:.1} GB KV + {:.1} GB overhead = {:.1} GB of {:.1} GB usable",
            weights_gb,
            kv_main_gb + kv_draft_gb,
            overhead_gb,
            need_vram_gb,
            room_gb
        );
        if kv_ram {
            why.push_str("; KV in host RAM (--no-kv-offload)");
        }
        if slack >= 0.0 {
            why.push_str(&format!("; {:.1} GB spare", slack));
        } else {
            why.push_str(&format!("; {:.1} GB short", -slack));
        }
        (verdict, why)
    };

    GgufEstimate {
        verdict,
        ctx,
        weights_gb,
        kv_main_gb,
        kv_draft_gb,
        overhead_gb,
        need_vram_gb,
        room_gb,
        kv_gb_per_token: (kv_main_gb + kv_draft_gb) * GIB / ctx.max(1) as f64,
        why,
    }
}

/// Largest standard context size that still fits, capped by the model's
/// training context. 0 means even the smallest step does not fit.
/// Context sizes worth suggesting. llama.cpp handles the 2^n sizes best;
/// 196608 is the 3×65536 step that Qwen-family models use.
const CTX_STEPS: &[u32] = &[4096, 8192, 16384, 32768, 65536, 131072, 196608, 262144];

/// Largest context that fits if every weight is on the card.
///
/// Kept for the "what would fit without expert offload" comparison; the hub
/// uses `mem::max_ctx`, which accounts for `--n-cpu-moe` placement.
#[allow(dead_code)]
pub fn suggest_ctx(
    meta: &gguf::Meta,
    weights_bytes: u64,
    hw: &Hardware,
    ctk: &str,
    ctv: &str,
    draft: Option<DraftKv<'_>>,
    kv_ram: bool,
) -> u32 {
    let draft_weights = draft.map(|d| d.weights_bytes).unwrap_or(0);
    let weights_gb = (weights_bytes.saturating_add(draft_weights)) as f64 / GIB * WEIGHT_LOAD_FACTOR;
    let overhead_gb = COMPUTE_GB + CUDA_CTX_GB;
    let room_gb = hw.vram_room_gb();
    let cap = if meta.ctx_train > 0 { meta.ctx_train } else { u32::MAX };

    // Steps are few, so brute-force the descending list instead of inventing
    // an inverse formula that would have to special-case sliding windows.
    for &step in CTX_STEPS.iter().rev() {
        if step > cap {
            continue;
        }
        let (main, draft_gb) = kv_usage_gb(meta, draft.as_ref(), step, ctk, ctv);
        let kv = if kv_ram { 0.0 } else { main + draft_gb };
        if weights_gb + overhead_gb + kv <= room_gb {
            return step;
        }
    }
    0
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Fits,
    Tight,
    No,
}

impl Verdict {
    /// Verdict for an already-computed placement (`need` vs usable `room`).
    pub fn for_slack(need_gb: f64, room_gb: f64) -> Self {
        if room_gb <= 0.0 {
            Verdict::No
        } else {
            let slack = room_gb - need_gb;
            if slack >= TIGHT_GB {
                Verdict::Fits
            } else if slack >= 0.0 {
                Verdict::Tight
            } else {
                Verdict::No
            }
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Verdict::Fits => "fits",
            Verdict::Tight => "tight",
            Verdict::No => "no",
        }
    }

    pub fn color(self) -> &'static str {
        match self {
            Verdict::Fits => crate::GREEN,
            Verdict::Tight => crate::YELLOW,
            Verdict::No => crate::RED,
        }
    }
}

pub struct Hardware {
    pub ram_total_gb: f64,
    pub ram_avail_gb: f64,
    pub vram_total_gb: f64,
    pub disk_free_gb: f64,
    /// `(name, total_gb)` for each CUDA device. Empty when nvidia-smi is not
    /// available, in which case `vram_total_gb` is authoritative.
    pub gpus: Vec<(String, f64)>,
}

impl Hardware {
    pub fn probe(models_dir: &Path) -> Self {
        let (ram_total_gb, ram_avail_gb) = match server::sys_mem() {
            Some(m) => (
                m.total_kb as f64 / 1024.0 / 1024.0,
                m.avail_kb as f64 / 1024.0 / 1024.0,
            ),
            None => (0.0, 0.0),
        };
        let gpus: Vec<(String, f64)> = server::gpu_infos()
            .into_iter()
            .map(|g| (g.name, g.mem_total / 1024.0))
            .collect();
        let vram_total_gb = if gpus.is_empty() {
            match server::gpu_mem() {
                Some((_used, total)) => total / 1024.0,
                None => 0.0,
            }
        } else {
            gpus.iter().map(|(_, gb)| gb).sum()
        };
        Self {
            ram_total_gb,
            ram_avail_gb,
            vram_total_gb,
            disk_free_gb: disk_free_gb(models_dir),
            gpus,
        }
    }

    /// VRAM the model may actually use for planning: every card minus a small
    /// desktop baseline. Deliberately not `total - used`, because when a model
    /// is already serving, its own weights are the "used" part.
    pub fn vram_room_gb(&self) -> f64 {
        if self.gpus.is_empty() {
            (self.vram_total_gb - 0.9).max(0.0)
        } else {
            self.gpus.iter().map(|(_, gb)| (gb - 0.9).max(0.0)).sum()
        }
    }
}

fn disk_free_gb(path: &Path) -> f64 {
    let out = std::process::Command::new("df")
        .args(["-B1", "--output=avail"])
        .arg(path)
        .output()
        .ok();
    let Some(out) = out else { return 0.0 };
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines()
        .nth(1)
        .and_then(|l| l.trim().parse::<f64>().ok())
        .map(|b| b / 1e9)
        .unwrap_or(0.0)
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PackKind {
    /// Qwen3.6-35B-A3B: experts split between VRAM and RAM.
    Hybrid,
    /// Qwen3.8-Flash-Next: shard 1's experts must sit in RAM.
    FlashNext,
    /// Gyro: one memory figure for VRAM + RAM.
    Gyro,
}

pub struct CatalogEntry {
    pub family: &'static str,
    pub size: &'static str,
    pub download_gb: f64,
    /// The part of the weights that is routed experts.
    pub experts_gb: f64,
    /// Strata's own "setup plans for" RAM figure, where it publishes one.
    pub plan_ram_gb: f64,
    pub kind: PackKind,
    pub note: &'static str,
}

/// Published sizes. Keep in sync with Strata docs/MODELS.md.
pub const CATALOG: &[CatalogEntry] = &[
    // Qwen3.6-35B-A3B (Unsloth, MTP) - docs/MODELS.md#qwen36-35b-a3b
    CatalogEntry {
        family: "qwen36",
        size: "ud-q4_k_xl",
        download_gb: 22.9,
        experts_gb: 18.0,
        plan_ram_gb: 32.0,
        kind: PackKind::Hybrid,
        note: "smallest; fits 30 GB RAM with room to spare",
    },
    CatalogEntry {
        family: "qwen36",
        size: "ud-q6_k_xl",
        download_gb: 32.6,
        experts_gb: 27.0,
        plan_ram_gb: 44.0,
        kind: PackKind::Hybrid,
        note: "your stated minimum; needs the GPU to take ~9-10 GB of experts",
    },
    CatalogEntry {
        family: "qwen36",
        size: "q8_0",
        download_gb: 37.8,
        experts_gb: 32.0,
        plan_ram_gb: 52.0,
        kind: PackKind::Hybrid,
        note: "closest to full quality; needs 64 GB RAM to be comfortable",
    },
    // Qwen3.8-Flash-Next, ISTA-DASLab originals - docs/MODELS.md#pick-by-memory
    CatalogEntry {
        family: "qwen",
        size: "q2_0",
        download_gb: 76.0,
        experts_gb: 34.0,
        plan_ram_gb: 44.0,
        kind: PackKind::FlashNext,
        note: "smallest full-expert pack",
    },
    CatalogEntry {
        family: "qwen",
        size: "iq2_xs",
        download_gb: 76.0,
        experts_gb: 35.5,
        plan_ram_gb: 45.5,
        kind: PackKind::FlashNext,
        note: "better quality than Q2_0, same class of memory",
    },
    CatalogEntry {
        family: "qwen",
        size: "iq3_xxs",
        download_gb: 76.0,
        experts_gb: 43.0,
        plan_ram_gb: 53.0,
        kind: PackKind::FlashNext,
        note: "great quality; 48 GB RAM and up",
    },
    CatalogEntry {
        family: "qwen",
        size: "iq3_s",
        download_gb: 76.0,
        experts_gb: 50.0,
        plan_ram_gb: 60.0,
        kind: PackKind::FlashNext,
        note: "matches the full model on the published tests",
    },
    CatalogEntry {
        family: "coder",
        size: "iq1_m",
        download_gb: 76.0,
        experts_gb: 29.6,
        plan_ram_gb: 39.6,
        kind: PackKind::FlashNext,
        note: "half the experts, chosen on code; weak outside code",
    },
    CatalogEntry {
        family: "gyro",
        size: "gyro-s",
        download_gb: 58.5,
        experts_gb: 20.0,
        plan_ram_gb: 27.6,
        kind: PackKind::Gyro,
        note: "smallest Flash-Next; lower quality, ~20% slower decode",
    },
    CatalogEntry {
        family: "gyro",
        size: "gyro-m",
        download_gb: 92.0,
        experts_gb: 27.0,
        plan_ram_gb: 35.0,
        kind: PackKind::Gyro,
        note: "IQ3_XXS-class quality, ~20% less GPU memory",
    },
];

pub struct Estimate {
    pub verdict: Verdict,
    /// RAM/VRAM only; `verdict` also counts free disk.
    pub mem_verdict: Verdict,
    pub need_ram_gb: f64,
    pub need_vram_gb: f64,
    pub need_disk_gb: f64,
    pub why: String,
}

/// KV cache bytes per token for Qwen3.6-35B-A3B.
///
/// docs/QWEN36.md: 45 KB per token, fp32 only (10 attention layers and the
/// MTP block). The qwen36 engine path has no `--kv` switch, so int8/q4_0
/// sizes do not apply here.
fn kv_gb(ctx: u32) -> f64 {
    const KB_PER_TOKEN: f64 = 45.0;
    (ctx as f64 * KB_PER_TOKEN) / 1e6
}

pub fn estimate(e: &CatalogEntry, h: &Hardware, ctx: u32) -> Estimate {
    // Qwen3.6 keeps its cache in fp32 regardless of --kv.
    let kv = kv_gb(ctx);
    // Engine reserve + compute buffers, from docs/QWEN36.md.
    const RESERVE_GB: f64 = 0.77;
    const BUFFERS_GB: f64 = 1.0;
    // Leave the OS, the desktop and a browser something to live on. Measured
    // on this box: the engine's host arena plus ~6 GB of everything else.
    const OS_GB: f64 = 6.0;

    let (need_ram_gb, need_vram_gb, why) = match e.kind {
        PackKind::Hybrid => {
            // Dense part (attention, embeddings, MTP) rides on the GPU too.
            let dense = (e.download_gb - e.experts_gb).max(0.5);
            let room = (h.vram_room_gb() - dense - kv - RESERVE_GB - BUFFERS_GB).max(0.0);
            let on_gpu = room.min(e.experts_gb);
            // Hybrid mode allocates a host arena for *every* routed expert
            // (q36_engine.cu: cudaHostAlloc, malloc fallback), not just the
            // ones the card cannot hold. That arena is the RAM floor.
            let host = e.experts_gb;
            (
                host + OS_GB,
                (dense + kv + RESERVE_GB + BUFFERS_GB).min(h.vram_total_gb),
                format!(
                    "{:.1} GB of experts on the GPU, all {:.1} GB also in host RAM (KV {:.1} GB at {}K)",
                    on_gpu,
                    host,
                    kv,
                    ctx / 1024
                ),
            )
        }
        PackKind::FlashNext => {
            let need = e.experts_gb + 10.0;
            let dense = (e.download_gb - e.experts_gb).max(0.5);
            (
                need,
                (dense + kv + RESERVE_GB + BUFFERS_GB).min(h.vram_total_gb),
                format!(
                    "shard 1 needs {:.1} GB of experts in RAM + 10 GB for the system",
                    e.experts_gb
                ),
            )
        }
        PackKind::Gyro => {
            let room = (h.vram_room_gb() - RESERVE_GB - BUFFERS_GB).max(0.0);
            let on_ram = (e.plan_ram_gb - room).max(0.0);
            (
                on_ram + OS_GB,
                room.min(e.plan_ram_gb),
                format!(
                    "{:.1} GB total for the model, {:.1} GB of it in VRAM",
                    e.plan_ram_gb, room
                ),
            )
        }
    };

    let need_disk_gb = e.download_gb + 5.0;

    // Verdict. RAM is the binding constraint on this class of machine.
    let mut mem_verdict = if need_ram_gb <= h.ram_total_gb * 0.85 {
        Verdict::Fits
    } else if need_ram_gb <= h.ram_total_gb * 0.97 {
        Verdict::Tight
    } else {
        Verdict::No
    };
    let mut why = why;

    if h.vram_total_gb > 0.0 && need_vram_gb > h.vram_total_gb {
        mem_verdict = Verdict::No;
        why.push_str(&format!(
            "; needs {:.1} GB VRAM, card has {:.1}",
            need_vram_gb, h.vram_total_gb
        ));
    }
    // Low-RAM mode is the one exception for Flash-Next, and only from 24 GB VRAM.
    if e.kind == PackKind::FlashNext
        && h.ram_total_gb < e.experts_gb + 10.0
        && h.vram_total_gb >= 24.0
    {
        mem_verdict = Verdict::Tight;
        why.push_str("; only via low-RAM mode (experts mmap'ed from the SSD)");
    }

    let mut verdict = mem_verdict;
    if h.disk_free_gb < need_disk_gb {
        verdict = Verdict::No;
        why.push_str(&format!(
            "; needs {:.0} GB free disk, {:.1} GB available",
            need_disk_gb, h.disk_free_gb
        ));
    }

    Estimate {
        verdict,
        mem_verdict,
        need_ram_gb,
        need_vram_gb,
        need_disk_gb,
        why,
    }
}

/// Command a user would run for this pack on this machine.
pub fn command(e: &CatalogEntry, ctx: u32) -> String {
    let family = e.family;
    if family == "qwen36" {
        // No --kv: this engine path keeps its cache in fp32.
        return format!(
            "strata launch --family {family} --model {} --context {} --vision off --tune skip --yes --no-tui",
            e.size, ctx
        );
    }
    format!(
        "strata launch --family {family} --model {} --context {} --kv int8 --vision off --tune skip --yes --no-tui",
        e.size, ctx
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{KvGeometry, KvLayer, Meta};

    fn hw(vram_total_gb: f64) -> Hardware {
        Hardware {
            ram_total_gb: 64.0,
            ram_avail_gb: 32.0,
            vram_total_gb,
            disk_free_gb: 2000.0,
            gpus: if vram_total_gb > 0.0 { vec![("test".into(), vram_total_gb)] } else { Vec::new() },
        }
    }

    fn attn(heads: u32, k: u32, v: u32, swa: bool) -> KvLayer {
        KvLayer { heads, k, v, swa }
    }

    /// The real Ornith-1.5-9B layout: 33 blocks, every 4th is full attention
    /// (8), one built-in MTP block (1), 4 KV heads of 256 elements.
    fn ornith_9b() -> Meta {
        Meta {
            layers: 33,
            quant: "Q6_K".into(),
            nextn: true,
            arch: "qwen35".into(),
            block_count: 33,
            ctx_train: 262144,
            nextn_layers: 1,
            kv: KvGeometry {
                main: vec![attn(4, 256, 256, false); 8],
                draft: vec![attn(4, 256, 256, false)],
                swa_window: 0,
            },
            ..Default::default()
        }
    }

    fn builtin_draft<'a>(meta: &'a Meta, ctk: &'a str, ctv: &'a str) -> DraftKv<'a> {
        DraftKv {
            layers: &meta.kv.draft,
            swa_window: meta.kv.swa_window,
            weights_bytes: 0,
            ctk,
            ctv,
        }
    }

    #[test]
    fn kv_type_sizes_match_ggml_blocks() {
        assert_eq!(kv_type_bytes("f16"), 2.0);
        assert_eq!(kv_type_bytes("bf16"), 2.0);
        assert_eq!(kv_type_bytes("q8_0"), 1.0625);
        assert!((kv_type_bytes("q6_k") - 0.8203125).abs() < 1e-9);
        assert_eq!(kv_type_bytes("q4_0"), 0.5625);
        assert_eq!(kv_type_bytes("IQ4_NL"), 0.5625);
        // Unknown names fall back to the safe (largest) assumption.
        assert_eq!(kv_type_bytes("nope"), 2.0);
    }

    #[test]
    fn ornith_9b_kv_matches_the_measurement() {
        // 7558901696 bytes = 7.04 GiB Q6_K file; measured 14009 MiB VRAM.
        let meta = ornith_9b();
        let e = for_gguf(
            &meta,
            7_558_901_696,
            &hw(15.5),
            262144,
            "q8_0",
            "q8_0",
            Some(builtin_draft(&meta, "q8_0", "q8_0")),
            false,
        );
        // The plan's check: 9 attention layers (8 main + 1 MTP) at q8_0.
        assert!((e.kv_main_gb - 4.25).abs() < 0.02, "main KV {}", e.kv_main_gb);
        assert!((e.kv_draft_gb - 0.53125).abs() < 0.02, "draft KV {}", e.kv_draft_gb);
        assert!((e.kv_main_gb + e.kv_draft_gb - 4.78).abs() < 0.06);
        assert_eq!(e.verdict, Verdict::Fits);
        assert!(e.need_vram_gb > 13.0 && e.need_vram_gb < 14.0, "{}", e.need_vram_gb);
    }

    #[test]
    fn ornith_9b_ctx_suggestion_respects_the_card() {
        let meta = ornith_9b();
        let draft = builtin_draft(&meta, "q8_0", "q8_0");
        assert_eq!(suggest_ctx(&meta, 7_558_901_696, &hw(15.5), "q8_0", "q8_0", Some(draft), false), 262144);
        // A 12 GB card leaves ~11.1 GiB; after 7.1 weights and 1.55 overhead
        // only ~2.4 GiB is left, which is the 131072 step, not 262144.
        assert_eq!(suggest_ctx(&meta, 7_558_901_696, &hw(12.0), "q8_0", "q8_0", Some(draft), false), 131072);
        // An 8 GB card cannot hold the weights at all.
        assert_eq!(suggest_ctx(&meta, 7_558_901_696, &hw(8.0), "q8_0", "q8_0", Some(draft), false), 0);
    }

    #[test]
    fn sliding_window_layers_stop_growing() {
        // Gemma-ish: 5 sliding-window blocks and one full-attention block.
        let mut layers = vec![attn(8, 256, 256, true); 5];
        layers.push(attn(1, 512, 512, false));
        let bytes = layer_bytes(&layers, 1024, 262144, 1.0625, 1.0625) / GIB;
        // Full block: 1 GiB * 262144/1024 / 512 ≈ 0.266 GiB;
        // window blocks: 5 * 8 / 1... ≈ 0.021 GiB. Total ≈ 0.286 GiB.
        assert!((bytes - 0.2864).abs() < 0.01, "gemma KV {bytes}");
    }

    #[test]
    fn asymmetric_kv_types_are_priced_separately() {
        let meta = Meta {
            layers: 1,
            quant: "Q8_0".into(),
            nextn: false,
            arch: "test".into(),
            block_count: 1,
            ctx_train: 131072,
            nextn_layers: 0,
            kv: KvGeometry { main: vec![attn(8, 128, 128, false)], draft: vec![], swa_window: 0 },
            ..Default::default()
        };
        let mixed = for_gguf(&meta, 1 << 30, &hw(24.0), 131072, "q8_0", "f16", None, false);
        let q8_both = for_gguf(&meta, 1 << 30, &hw(24.0), 131072, "q8_0", "q8_0", None, false);
        // V in f16 is almost twice the size of V in q8_0, so the mixed cache
        // must be bigger than q8_0/q8_0, and smaller than f16/f16.
        assert!(mixed.kv_main_gb > q8_both.kv_main_gb);
        let f16_both = for_gguf(&meta, 1 << 30, &hw(24.0), 131072, "f16", "f16", None, false);
        assert!(mixed.kv_main_gb < f16_both.kv_main_gb);
    }

    #[test]
    fn missing_kv_metadata_is_not_a_false_fits() {
        let meta = Meta {
            layers: 32,
            quant: "Q4_K".into(),
            nextn: false,
            arch: "unknown".into(),
            block_count: 32,
            ctx_train: 4096,
            nextn_layers: 0,
            kv: KvGeometry::default(),
            ..Default::default()
        };
        let e = for_gguf(&meta, 4 << 30, &hw(24.0), 4096, "q8_0", "q8_0", None, false);
        assert_eq!(e.verdict, Verdict::No);
        assert!(e.why.contains("KV geometry"));
    }
    #[test]
    fn draft_kv_defaults_to_f16_and_reads_extra() {
        assert_eq!(draft_kv_types(&Cfg::default()), ("f16".into(), "f16".into()));
    }

    #[test]
    fn draft_kv_prefers_the_knob_over_extra_then_f16() {
        // The explicit knob wins even when extra also names a draft flag.
        let with_knob = Cfg {
            draft_ctk: "q8_0".into(),
            draft_ctv: "q8_0".into(),
            ..Cfg::default()
        };
        assert_eq!(draft_kv_types(&with_knob), ("q8_0".into(), "q8_0".into()));

        // Without the knob, `extra` overrides win.
        let with_extra = Cfg {
            extra: Some("--batch-size 1024 -ctkd q4_0 -ctvd q8_0".into()),
            ..Cfg::default()
        };
        assert_eq!(draft_kv_types(&with_extra), ("q4_0".into(), "q8_0".into()));

        let eq = Cfg {
            extra: Some("--cache-type-k-draft=q5_1".into()),
            ..Cfg::default()
        };
        assert_eq!(draft_kv_types(&eq), ("q5_1".into(), "f16".into()));
    }

}
