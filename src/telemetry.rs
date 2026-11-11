//! The *actual* memory split, read back from llama.cpp's startup log.
//!
//! At `--log-verbosity 4` llama.cpp prints two things the planner cannot
//! know on its own:
//!
//! ```text
//! | memory breakdown [MiB] | total  free   self   model  context  compute |
//! |   - CUDA0 (RTX 4060 Ti) | 15946 = 14818 + (6402 = 6402 + 696 + 224) ...|
//! load_tensors:   CUDA0 model buffer size =  6402.57 MiB
//! llama_kv_cache:      CUDA0 KV buffer size =   544.00 MiB
//! sched_reserve:       CUDA0 compute buffer size =   224.02 MiB
//! ```
//!
//! The breakdown table is preferred: it is already per device, and it is
//! printed separately for the target model and for an external draft model.
//! The buffer lines are the fallback for builds that do not print the table.
//! Either way this is a report of the allocation llama.cpp made, not a live
//! measurement; the card labels it "actual (log)".

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::Path;

const MIB: f64 = 1024.0;

#[derive(Clone, Default)]
pub struct DevMem {
    /// Model weights on this device, MiB.
    pub model: f64,
    /// KV cache (context column), MiB.
    pub context: f64,
    /// Compute buffers, MiB.
    pub compute: f64,
}

#[derive(Clone, Default)]
pub struct Section {
    pub devices: BTreeMap<String, DevMem>,
    pub host_model: f64,
    /// Portion of `host_model` that is `CPU_Mapped` (file-backed mmap).
    pub host_model_mapped: f64,
    /// Portion of `host_model` that is anonymous host RAM (`CPU`).
    pub host_model_anon: f64,
    pub host_context: f64,
    pub host_compute: f64,
}

impl Section {
    pub fn is_empty(&self) -> bool {
        self.devices.is_empty() && self.host_model + self.host_context + self.host_compute < 0.01
    }

    fn dev(&mut self, name: &str) -> &mut DevMem {
        self.devices
            .entry(name.to_string())
            .or_insert_with(DevMem::default)
    }
}

#[derive(Clone, Default)]
pub struct LoadInfo {
    pub main: Section,
    pub draft: Section,
    /// `offloaded X/Y layers to GPU`.
    pub offloaded: Option<(u32, u32)>,
    /// True when the full breakdown table was found, false when only the
    /// individual buffer lines were.
    pub breakdown: bool,
    pub n_ctx: Option<u32>,
    /// llama.cpp warned that `--override-tensor` moved tensors to the CPU
    /// while mmap is enabled. The CPU weights are then file-backed pages and
    /// get re-read from disk on every expert lookup.
    pub mmap_cpu_experts: bool,
}

impl LoadInfo {
    pub fn is_empty(&self) -> bool {
        self.main.is_empty() && self.draft.is_empty()
    }

    /// Rows for the PLACEMENT block: `(label, "weights · KV · compute")`.
    pub fn rows(&self) -> Vec<(String, String)> {
        let mut v = Vec::new();
        for (name, d) in &self.main.devices {
            v.push((device_label(name), dev_text(d, true)));
        }
        if self.main.host_model + self.main.host_context + self.main.host_compute > 0.01 {
            let mut parts = Vec::new();
            if self.main.host_model > 0.01 {
                let mapped = self.main.host_model_mapped;
                let anon = self.main.host_model_anon;
                if mapped > 0.01 && anon > 0.01 {
                    parts.push(format!(
                        "weights {:.2} (mmap {:.2} + RAM {:.2})",
                        self.main.host_model / MIB,
                        mapped / MIB,
                        anon / MIB
                    ));
                } else {
                    parts.push(format!("weights {:.2}", self.main.host_model / MIB));
                }
            }
            if self.main.host_context > 0.01 {
                parts.push(format!("KV {:.2}", self.main.host_context / MIB));
            }
            if self.main.host_compute > 0.01 {
                parts.push(format!("ovh {:.2}", self.main.host_compute / MIB));
            }
            let label = if self.main.host_model_mapped > 0.01 && self.main.host_model_anon <= 0.01 {
                "Host (mmap)"
            } else {
                "Host"
            };
            v.push((label.into(), format!("{} GiB", parts.join(" · "))));
        }
        // The draft table repeats the target model's weights for a built-in
        // MTP head (the head lives inside the main file). Only print weights
        // when the draft really owns a separate file.
        for (name, d) in &self.draft.devices {
            let dup = self
                .main
                .devices
                .get(name)
                .map(|m| (m.model - d.model).abs() < 2.0)
                .unwrap_or(false);
            v.push((format!("Draft {}", device_label(name)), dev_text(d, !dup)));
        }
        if self.draft.host_model + self.draft.host_context > 0.01 {
            let dup = (self.main.host_model - self.draft.host_model).abs() < 2.0;
            let mut parts = Vec::new();
            if !dup {
                parts.push(format!("weights {:.2}", self.draft.host_model / MIB));
            }
            if self.draft.host_context > 0.01 {
                parts.push(format!("KV {:.2}", self.draft.host_context / MIB));
            }
            if !parts.is_empty() {
                v.push(("Draft host".into(), format!("{} GiB", parts.join(" · "))));
            }
        }
        v
    }
}

/// `CUDA0 (RTX 4060 Ti)` → `GPU0`, so live rows line up with the plan.
fn device_label(name: &str) -> String {
    let head = name.split_whitespace().next().unwrap_or(name);
    match head.strip_prefix("CUDA") {
        Some(n) => format!("GPU{n}"),
        None => head.to_string(),
    }
}

fn dev_text(d: &DevMem, weights: bool) -> String {
    let mut parts = Vec::new();
    if weights {
        parts.push(format!("weights {:.2}", d.model / MIB));
    }
    parts.push(format!("KV {:.2}", d.context / MIB));
    if d.compute > 0.01 {
        parts.push(format!("ovh {:.2}", d.compute / MIB));
    }
    format!("{} GiB", parts.join(" · "))
}

fn mib_after(s: &str, needle: &str) -> Option<f64> {
    let idx = s.find(needle)? + needle.len();
    s[idx..]
        .trim_start()
        .split_whitespace()
        .next()?
        .trim_end_matches(',')
        .parse()
        .ok()
}

/// The device name in `load_tensors:   CPU_Mapped model buffer size = ...` is
/// the token right before "model buffer size".
fn device_before<'a>(s: &'a str, needle: &str) -> Option<&'a str> {
    let idx = s.find(needle)?;
    s[..idx].split_whitespace().last()
}

fn parse_breakdown_line(line: &str) -> Option<(String, f64, f64, f64)> {
    let fields: Vec<&str> = line.split('|').collect();
    if fields.len() < 3 {
        return None;
    }
    let name = fields[1].trim().trim_start_matches('-').trim();
    if name.is_empty() || !fields[2].contains('=') {
        return None;
    }
    // "15946 = 14818 + (6402 = 6402 + 696 + 224) + -6195"  (device row)
    // "859 = 795 + 0 + 64"                                  (host row)
    let parts: Vec<&str> = fields[2].split('=').collect();
    let nums: Vec<f64> = if parts.len() >= 3 {
        parts[2]
            .split(')')
            .next()?
            .split('+')
            .filter_map(|x| x.trim().parse().ok())
            .collect()
    } else if parts.len() == 2 {
        parts[1]
            .split('+')
            .filter_map(|x| x.trim().parse().ok())
            .collect()
    } else {
        return None;
    };
    if nums.len() < 3 {
        return None;
    }
    Some((name.to_string(), nums[0], nums[1], nums[2]))
}

/// Parse a whole startup log. Safe to call with only the log head: all the
/// load lines are written before "listening on http".
pub fn parse(text: &str) -> LoadInfo {
    let mut info = LoadInfo::default();
    let mut target = 0u8; // 0 = main, 1 = draft

    for line in text.lines() {
        if line.contains("tensor overrides to CPU are used with mmap enabled") {
            info.mmap_cpu_experts = true;
        }
        if line.contains("getting device memory data for the extra model") {
            target = 1;
        }
        if line.contains("offloaded ") && line.contains(" layers to GPU") {
            let t = line.split("offloaded ").nth(1).unwrap_or("");
            if let Some((a, b)) = t.split_once('/') {
                let a = a.trim().parse().ok();
                let b = b.split_whitespace().next().and_then(|x| x.parse().ok());
                if let (Some(a), Some(b)) = (a, b) {
                    info.offloaded = Some((a, b));
                }
            }
        }
        if line.contains("llama_context: n_ctx ") {
            if let Some(v) = line
                .split('=')
                .nth(1)
                .and_then(|x| x.trim().parse::<u32>().ok())
            {
                info.n_ctx = Some(v);
            }
        }
        // The per-device table.
        if line.contains("memory breakdown") {
            info.breakdown = true;
            continue;
        }
        if info.breakdown && line.contains('|') {
            let Some((name, model, context, compute)) = parse_breakdown_line(line) else {
                continue;
            };
            let sec = if target == 1 { &mut info.draft } else { &mut info.main };
            if name.eq_ignore_ascii_case("Host") {
                sec.host_model = model;
                sec.host_context = context;
                sec.host_compute = compute;
            } else {
                let d = sec.dev(&name);
                d.model = model;
                d.context = context;
                d.compute = compute;
            }
            continue;
        }

        // Buffer lines. They are printed at load time and are the only source
        // of the `CPU_Mapped` / `CPU` split: the breakdown table is printed on
        // shutdown and lumps both into one "Host" row. Keep reading the split
        // even when the table is present.
        if let Some(v) = mib_after(line, "model buffer size =") {
            let Some(dev) = device_before(line, "model buffer size") else { continue };
            let sec = if target == 1 { &mut info.draft } else { &mut info.main };
            if dev.contains("CPU") || dev.starts_with("Host") {
                // `CPU_Mapped` is file-backed mmap, plain `CPU` is anonymous RAM.
                if dev.contains("Mapped") {
                    sec.host_model_mapped += v;
                } else {
                    sec.host_model_anon += v;
                }
                if !info.breakdown {
                    sec.host_model += v;
                }
            } else if !info.breakdown {
                sec.dev(dev).model += v;
            }
            continue;
        }
        // Everything below is already in the breakdown table, so do not add
        // it twice.
        if info.breakdown {
            continue;
        }
        if let Some(v) = mib_after(line, "KV buffer size =") {
            let Some(dev) = device_before(line, "KV buffer size") else { continue };
            let sec = if target == 1 { &mut info.draft } else { &mut info.main };
            if dev.contains("CPU") || dev.starts_with("Host") {
                sec.host_context += v;
            } else {
                sec.dev(dev).context += v;
            }
        } else if let Some(v) = mib_after(line, "compute buffer size =") {
            let Some(dev) = device_before(line, "compute buffer size") else { continue };
            let sec = if target == 1 { &mut info.draft } else { &mut info.main };
            if dev.contains("CPU") || dev.starts_with("Host") {
                sec.host_compute += v;
            } else {
                sec.dev(dev).compute += v;
            }
        } else if let Some(v) = mib_after(line, "RS buffer size =") {
            // Recurrent state of hybrid SSM blocks. It plays the role of KV.
            let Some(dev) = device_before(line, "RS buffer size") else { continue };
            let sec = if target == 1 { &mut info.draft } else { &mut info.main };
            if dev.contains("CPU") || dev.starts_with("Host") {
                sec.host_context += v;
            } else {
                sec.dev(dev).context += v;
            }
        }
    }
    info
}

/// Read the head of a log file. Load lines are written before the server
/// starts listening, so a few hundred KiB is always enough even after a long
/// session has appended request timings.
pub fn read(path: &Path) -> Option<LoadInfo> {
    let mut f = fs::File::open(path).ok()?;
    let mut buf = Vec::with_capacity(256 * 1024);
    f.by_ref().take(256 * 1024).read_to_end(&mut buf).ok()?;
    let text = String::from_utf8_lossy(&buf);
    let info = parse(&text);
    (!info.is_empty()).then_some(info)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOG: &str = "\
0.00.504.179 I common_memory_breakdown_print: | memory breakdown [MiB]  | total    free    self   model   context   compute    unaccounted |
0.00.504.186 I common_memory_breakdown_print: |   - CUDA0 (RTX 4060 Ti) | 15946 = 14818 + (7323 =  6402 +     696 +     224) +       -6195 |
0.00.504.186 I common_memory_breakdown_print: |   - Host                |                   859 =   795 +       0 +      64                |
0.00.531.653 I operator(): getting device memory data for the extra model at a context size of 32768:
0.00.729.910 I common_memory_breakdown_print: | memory breakdown [MiB]  | total    free    self   model   context   compute    unaccounted |
0.00.729.914 I common_memory_breakdown_print: |   - CUDA0 (RTX 4060 Ti) | 15946 = 14818 + (6626 =  6402 +     128 +      96) +       -5498 |
0.00.729.914 I common_memory_breakdown_print: |   - Host                |                   843 =   795 +       0 +      48                |
0.01.271.312 I load_tensors: offloaded 34/34 layers to GPU
0.01.271.316 I load_tensors:   CPU_Mapped model buffer size =   795.70 MiB
0.01.271.317 I load_tensors:        CUDA0 model buffer size =  6402.57 MiB
0.01.908.246 I llama_kv_cache:      CUDA0 KV buffer size =   544.00 MiB
0.01.958.806 I sched_reserve:      CUDA0 compute buffer size =   224.02 MiB
";

    #[test]
    fn breakdown_is_split_into_main_and_draft() {
        let i = parse(LOG);
        assert!(i.breakdown);
        assert_eq!(i.offloaded, Some((34, 34)));
        let d = i.main.devices.get("CUDA0 (RTX 4060 Ti)").expect("main device");
        assert!((d.model - 6402.0).abs() < 0.1);
        assert!((d.context - 696.0).abs() < 0.1);
        assert!((d.compute - 224.0).abs() < 0.1);
        assert!((i.main.host_model - 795.0).abs() < 0.1);
        let dd = i.draft.devices.get("CUDA0 (RTX 4060 Ti)").expect("draft device");
        assert!((dd.context - 128.0).abs() < 0.1);
        // The buffer lines must not double-count the table.
        assert!((d.model - 6402.0).abs() < 0.1);
    }

    #[test]
    fn buffer_lines_work_without_the_table() {
        let text = "\
0.01.271.316 I load_tensors:   CPU_Mapped model buffer size =   795.70 MiB
0.01.271.317 I load_tensors:        CUDA0 model buffer size =  6402.57 MiB
0.01.908.246 I llama_kv_cache:      CUDA0 KV buffer size =   544.00 MiB
0.01.908.446 I llama_kv_cache:        CPU KV buffer size =   128.00 MiB
0.01.958.806 I sched_reserve:      CUDA0 compute buffer size =   224.02 MiB
0.01.909.877 I llama_memory_recurrent:      CUDA0 RS buffer size =   150.75 MiB
";
        let i = parse(text);
        assert!(!i.breakdown);
        let d = i.main.devices.get("CUDA0").expect("device");
        assert!((d.model - 6402.57).abs() < 0.1);
        assert!((d.context - 694.75).abs() < 0.1, "{}", d.context);
        assert!((i.main.host_model - 795.70).abs() < 0.1);
        assert!((i.main.host_context - 128.0).abs() < 0.1);
    }

    #[test]
    fn mapped_and_anonymous_host_weights_are_split() {
        let text = "\
0.01.271.316 I load_tensors:   CPU_Mapped model buffer size =  10800.00 MiB
0.01.271.317 I load_tensors:          CPU model buffer size =   512.00 MiB
0.01.271.318 I load_tensors:        CUDA0 model buffer size =  4096.00 MiB
0.01.271.319 W llama_model_loader: tensor overrides to CPU are used with mmap enabled - consider using --load-mode none for better performance
";
        let i = parse(text);
        assert!((i.main.host_model_mapped - 10800.0).abs() < 0.1);
        assert!((i.main.host_model_anon - 512.0).abs() < 0.1);
        assert!((i.main.host_model - 11312.0).abs() < 0.1);
        assert!(i.mmap_cpu_experts);
        let rows = i.rows();
        let host = rows.iter().find(|(l, _)| l.starts_with("Host")).expect("host row");
        assert_eq!(host.0, "Host");
        assert!(host.1.contains("mmap 10.55"), "{}", host.1);
    }

    #[test]
    fn mmap_only_host_is_labelled() {
        let text = "\
0.01.271.316 I load_tensors:   CPU_Mapped model buffer size =  10800.00 MiB
";
        let i = parse(text);
        let rows = i.rows();
        let host = rows.iter().find(|(l, _)| l.starts_with("Host")).expect("host row");
        assert_eq!(host.0, "Host (mmap)");
    }
}
