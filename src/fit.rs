//! "Will it fit?" estimates for the Strata model families.
//!
//! The numbers are the published ones from Strata's docs/MODELS.md and
//! docs/QWEN36.md, turned into arithmetic for *this* machine:
//!
//!   * Qwen3.6-35B-A3B runs in hybrid mode: the engine keeps every expert it
//!     can in VRAM and computes the rest from RAM.
//!   * Qwen3.8-Flash-Next keeps shard 1's experts in RAM; the GPU holds the
//!     dense part. Without enough RAM it only runs in low-RAM mode, which the
//!     docs say needs 32 GB of RAM and a 24 GB card (or the Coder pack).
//!   * Gyro is quoted as VRAM + RAM together.
//!
//! Everything here is an estimate. It is deliberately conservative: a "fits"
//! means the model should load with the desktop running, not that it will be
//! fast.

use std::path::Path;

use crate::server;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Fits,
    Tight,
    No,
}

impl Verdict {
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
        let (vram_total_gb, _vram_used_gb) = match server::gpu_mem() {
            Some((used, total)) => (total / 1024.0, used / 1024.0),
            None => (0.0, 0.0),
        };
        Self {
            ram_total_gb,
            ram_avail_gb,
            vram_total_gb,
            disk_free_gb: disk_free_gb(models_dir),
        }
    }

    /// VRAM the model may actually use for planning: the card minus a small
    /// desktop baseline. Deliberately not `total - used`, because when a model
    /// is already serving, its own weights are the "used" part.
    pub fn vram_room_gb(&self) -> f64 {
        (self.vram_total_gb - 0.9).max(0.0)
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
