use std::fs;
use std::path::{Path, PathBuf};

use crate::gguf;

#[derive(Clone)]
pub struct Model {
    pub path: PathBuf,
    pub name: String,
    pub size: u64,
    /// Quantization guess (from file name), e.g. "Q4_K_M", "Q8_0".
    pub quant: String,
    /// Total transformer layers parsed from GGUF.
    pub layers: u32,
    /// Dense or MoE.
    pub moe: bool,
    /// Total experts (MoE only).
    pub experts_total: u32,
    /// Active experts per token (MoE only).
    pub experts_active: u32,
    /// Built-in MTP head (`blk.N.nextn.*`) inside this GGUF.
    pub nextn: bool,
    /// Full header metadata; the hub uses the KV geometry and context length
    /// without reading the file again.
    pub meta: gguf::Meta,
}

pub fn is_draft(name: &str) -> bool {
    let n = name.to_lowercase();
    // "mtp-" is the drafter prefix (mtp-Base-...); a full model build that
    // merely ends in "-mtp.gguf" (ISTA-DASLab keeps the head inside the main
    // file) must stay visible.
    n.contains("lilicorr") || n.contains("draft") || n.contains("mmproj") || n.contains("mtp-")
}

pub fn scan(dir: &Path) -> Vec<Model> {
    let mut out = Vec::new();
    collect(dir, &mut out);
    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    out
}

fn collect(dir: &Path, out: &mut Vec<Model>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect(&p, out);
            continue;
        }
        match p.extension() {
            Some(x) if x.eq_ignore_ascii_case("gguf") => {}
            _ => continue,
        }
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        if is_draft(&name) {
            continue;
        }
        let size = e.metadata().map(|m| m.len()).unwrap_or(0);
        let meta = gguf::parse(&p);
        let name_quant = gguf::quant_from_name(&name);
        let quant = if name_quant == "unknown" { meta.quant.clone() } else { name_quant };
        let quant = if quant.is_empty() { "unknown".to_string() } else { quant };
        let layers = meta.layers;
        let (moe, experts_total, experts_active) = detect_moe(&name, &meta);
        out.push(Model {
            path: p,
            name,
            size,
            quant,
            layers,
            moe,
            experts_total,
            experts_active,
            nextn: meta.nextn,
            meta,
        });
    }
}

fn detect_moe(name: &str, meta: &gguf::Meta) -> (bool, u32, u32) {
    let n = name.to_lowercase();
    // Authoritative: `llama.model_type` (2 = MoE) when the file declares it.
    if meta.model_type == 2 {
        return (true, meta.experts_total, meta.experts_active);
    }
    // Architecture suffix marks MoE (qwen35moe, qwen3_moe) even when
    // `llama.model_type` is absent — that is how Ornith-1.5-35B-A3B is told.
    if meta.arch.contains("_moe") {
        return (true, meta.experts_total, meta.experts_active);
    }
    // The file explicitly lists routed experts; a tensor table with
    // `*_exps` tensors is MoE by definition.
    if meta.experts_total > 0 || meta.experts_active > 0 {
        return (true, meta.experts_total, meta.experts_active);
    }
    // Active-param notation (e.g. 74B-A9B), classic MoE names, and Nemotron.
    let active_notation = ["a3b", "a9b", "a12b", "a22b", "a35b"]
        .iter()
        .any(|t| n.contains(t));
    if n.contains("moe") || n.contains("mixtral") || n.contains("nemotron") || active_notation {
        // mixtral-8x7b -> 8 experts, 2 active per token; others unknown.
        let total = if n.contains("mixtral") {
            n.split('x')
                .next()
                .and_then(|s| s.chars().filter(|c| c.is_ascii_digit()).collect::<String>().parse::<u32>().ok())
                .unwrap_or(8)
        } else {
            0
        };
        return (true, total, if total > 0 { 2 } else { 0 });
    }
    (false, 0, 0)
}

pub fn find_draft(model: &Path) -> Option<PathBuf> {
    let dir = model.parent()?;
    let stem = model.file_stem()?.to_string_lossy().to_string();
    // Unsloth names the base model "<Base>-UD-<quant>"; draft files are
    // "<Base>-LiLiCorr", "draft-<Base>" or Unsloth's "mtp-<Base>-<quant>".
    let base = stem.split("-UD-").next().unwrap_or(&stem).to_string();
    let rd = fs::read_dir(dir).ok()?;
    for e in rd.flatten() {
        let p = e.path();
        let Some(name) = p.file_name().map(|s| s.to_string_lossy().to_lowercase()) else {
            continue;
        };
        if !name.contains("lilicorr") && !name.contains("draft") && !name.contains("mtp-") {
            continue;
        }
        let dstem = p.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        if dstem.starts_with(&stem) || stem.starts_with(&dstem) || dstem.contains(&base) {
            return Some(p);
        }
    }
    None
}

/// The vision projector that sits next to `model`, if any.
///
/// Unsloth and llama.cpp repos name these `mmproj-<base>-<quant>.gguf` or
/// `mmproj-<base>-f16.gguf`; there is usually exactly one per model folder, so
/// the first match wins. `scan` hides mmproj files (`is_draft`), so this is the
/// only way the hub can offer `vision = true` without a typed path.
pub fn find_mmproj(model: &Path) -> Option<PathBuf> {
    let dir = model.parent()?;
    let rd = fs::read_dir(dir).ok()?;
    for e in rd.flatten() {
        let p = e.path();
        let Some(name) = p.file_name().map(|s| s.to_string_lossy().to_lowercase()) else {
            continue;
        };
        if name.contains("mmproj") && name.ends_with(".gguf") {
            return Some(p);
        }
    }
    None
}

pub fn human(bytes: u64) -> String {
    const U: [&str; 5] = ["B", "K", "M", "G", "T"];
    let mut v = bytes as f64;
    let mut i = 0;
    while v >= 1024.0 && i < U.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{}{}", bytes, U[i])
    } else {
        format!("{:.1}{}", v, U[i])
    }
}

pub fn stem(name: &str) -> String {
    name.trim_end_matches(".gguf").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::Meta;

    fn dense_meta() -> Meta {
        Meta::default()
    }

    #[test]
    fn moe_from_arch_suffix_without_model_type() {
        // Ornith-1.5-35B-A3B: name has no "a3b"/"moe" marker, arch is qwen35moe.
        let (moe, total, active) = detect_moe("Ornith-1.5-35B-Q4_K_M.gguf", &Meta {
            arch: "qwen35moe".into(),
            experts_total: 256,
            experts_active: 8,
            ..Meta::default()
        });
        assert!(moe, "qwen35moe must be detected as MoE");
        assert_eq!((total, active), (256, 8));
    }

    #[test]
    fn moe_from_llama_model_type() {
        let (moe, total, active) = detect_moe("whatever.gguf", &Meta {
            model_type: 2,
            experts_total: 16,
            experts_active: 4,
            ..Meta::default()
        });
        assert!(moe);
        assert_eq!((total, active), (16, 4));
    }

    #[test]
    fn dense_when_no_marker_and_no_metadata() {
        let (moe, total, active) = detect_moe("Llama-3.1-8B-Q4_K_M.gguf", &dense_meta());
        assert!(!moe);
        assert_eq!((total, active), (0, 0));
    }

    #[test]
    fn active_param_notation_still_works() {
        let (moe, total, active) = detect_moe("NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf", &dense_meta());
        assert!(moe);
        // Active-param notation flags MoE; the total is unknown unless the
        // name is mixtral-style (8x7b).
        assert_eq!((total, active), (0, 0));
    }
}
