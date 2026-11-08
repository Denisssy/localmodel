//! Engine selection and the Strata command line.
//!
//! localmodel runs two backends:
//!
//!   * **llama.cpp** (`llama-server`) for dense models. One process per model,
//!     its stderr is the live log, `localmodel` owns the PID.
//!   * **Strata** for mixture-of-experts models. Strata's own engine keeps the
//!     routed experts on the GPU and computes the rest on the CPU, which is
//!     exactly what a 16 GB card with a 30 GB machine needs.
//!
//! Strata is a separate binary. `strata launch --no-tui` starts the server,
//! prints the address and returns; the server keeps running in the background.
//! So localmodel never owns the PID there: it starts the launch, then talks to
//! the OpenAI-compatible HTTP API like any other client, and stops it with
//! `strata launch --stop`.

use std::env;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

use crate::models::Model;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Engine {
    LlamaCpp,
    Strata,
}

impl Engine {
    pub fn label(self) -> &'static str {
        match self {
            Engine::LlamaCpp => "llama.cpp",
            Engine::Strata => "strata",
        }
    }

    pub fn parse(s: &str) -> Option<Engine> {
        match s.to_lowercase().as_str() {
            "llama" | "llama.cpp" | "llamacpp" => Some(Engine::LlamaCpp),
            "strata" => Some(Engine::Strata),
            _ => None,
        }
    }
}

/// Strata is not a general GGUF runner. Its public families are exactly three
/// (docs/MODELS.md, docs/AI_REFERENCE.md): DeepSeek V4.1-Flash, Qwen3.8-Flash-Next
/// and Qwen3.6-35B-A3B, plus their variants (Coder, Swift, Gyro, NVFP4, Orca).
/// Anything else - Mixtral, Nemotron, Qwen3-30B-A3B - has to stay on llama.cpp,
/// however MoE it is.
pub fn supports_strata(name: &str) -> bool {
    let n = name.to_lowercase();
    let qwen36 = n.contains("qwen3.6") && n.contains("a3b");
    let flash_next = n.contains("qwen3.8") && n.contains("flash");
    let deepseek = n.contains("deepseek") && n.contains("v4.1");
    let variants = n.contains("gyro-s") || n.contains("gyro-m") || n.contains("flash-next");
    qwen36 || flash_next || deepseek || variants
}

/// Strata for its own families, llama.cpp for everything else. An explicit
/// choice wins, even when it is a bad idea (the caller warns).
pub fn route(m: &Model, forced: Option<Engine>) -> Engine {
    if let Some(e) = forced {
        return e;
    }
    if supports_strata(&m.name) && strata_bin().is_some() {
        Engine::Strata
    } else {
        Engine::LlamaCpp
    }
}

pub fn strata_bin() -> Option<PathBuf> {
    if let Some(b) = env::var_os("STRATA_BIN") {
        let p = PathBuf::from(b);
        if p.is_file() {
            return Some(p);
        }
    }
    if let Some(path) = env::var_os("PATH") {
        for dir in env::split_paths(&path) {
            let p = dir.join("strata");
            if p.is_file() {
                return Some(p);
            }
        }
    }
    let home = crate::config::home();
    for p in [home.join(".local/bin/strata"), PathBuf::from("/usr/local/bin/strata")] {
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

pub fn install_hint() -> &'static str {
    "strata not found. install it with:\n  \
     curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh\n  \
     (or set STRATA_BIN=/path/to/strata)"
}

/// One `strata launch` invocation.
#[derive(Clone)]
pub struct StrataSpec {
    /// `--family` + `--model` for a catalog pack...
    pub family: Option<String>,
    pub size: Option<String>,
    /// ...or `--model-path` for a GGUF/Shard folder already on disk.
    pub model_path: Option<PathBuf>,
    pub ctx: u32,
    /// int8 | q4_0 | k8v4 | fp16
    pub kv: String,
    pub parallel: u8,
    /// off | on | cpu | gpu
    pub vision: String,
    /// local | lan | cloudflare | tailscale
    pub access: String,
    pub port: u16,
    /// skip | quick | thorough
    pub tune: String,
    pub restart: bool,
    pub data_dir: Option<PathBuf>,
    pub extra: Vec<String>,
}

impl StrataSpec {
    pub fn new(ctx: u32, port: u16) -> Self {
        Self {
            family: None,
            size: None,
            model_path: None,
            ctx,
            kv: "int8".into(),
            parallel: 1,
            vision: "off".into(),
            access: "local".into(),
            port,
            tune: "skip".into(),
            restart: false,
            data_dir: None,
            extra: Vec::new(),
        }
    }

    /// KV format the engine will actually use. Qwen3.6-35B-A3B has no
    /// `--kv` switch: its cache is fp32 (45 KB/token). Flash-Next honours
    /// the requested format.
    pub fn effective_kv(&self) -> &str {
        if self.family.as_deref() == Some("qwen36") {
            "fp32"
        } else {
            &self.kv
        }
    }

    pub fn args(&self, plan: bool) -> Vec<String> {
        let mut a: Vec<String> = vec!["launch".into()];
        if let Some(f) = &self.family {
            a.push("--family".into());
            a.push(f.clone());
        }
        if let Some(m) = &self.size {
            a.push("--model".into());
            a.push(m.clone());
        }
        if let Some(p) = &self.model_path {
            a.push("--model-path".into());
            a.push(p.display().to_string());
        }
        a.push("--context".into());
        a.push(self.ctx.to_string());
        if self.family.as_deref() != Some("qwen36") {
            // qwen36's cache is fp32 only; setup warns and ignores --kv.
            a.push("--kv".into());
            a.push(self.kv.clone());
        }
        a.push("--parallel".into());
        a.push(self.parallel.to_string());
        a.push("--vision".into());
        a.push(self.vision.clone());
        a.push("--access".into());
        a.push(self.access.clone());
        a.push("--port".into());
        a.push(self.port.to_string());
        a.push("--tune".into());
        a.push(self.tune.clone());
        if let Some(d) = &self.data_dir {
            a.push("--data-dir".into());
            a.push(d.display().to_string());
        }
        if self.restart {
            a.push("--restart".into());
        }
        a.extend(self.extra.iter().cloned());
        if plan {
            a.push("--plan".into());
        } else {
            a.push("--yes".into());
            a.push("--no-tui".into());
        }
        a.push("--output".into());
        a.push("json".into());
        a
    }

    pub fn display(&self) -> String {
        match (&self.family, &self.size) {
            (Some(f), Some(m)) => format!("{f}/{m}"),
            _ => self
                .model_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "?".into()),
        }
    }
}

pub struct RunOutput {
    pub code: i32,
    pub json: Option<Value>,
    pub text: String,
}

fn run(bin: &Path, args: &[String]) -> io::Result<RunOutput> {
    let out = Command::new(bin).args(args).output()?;
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    Ok(RunOutput {
        code: out.status.code().unwrap_or(-1),
        json: extract_json(&stdout).or_else(|| extract_json(&stderr)),
        text: format!("{stdout}{stderr}"),
    })
}

/// Strata may print progress lines around the JSON object; take the outermost
/// `{ ... }` that parses.
fn extract_json(s: &str) -> Option<Value> {
    let start = s.find('{')?;
    let end = s.rfind('}')?;
    if end <= start {
        return None;
    }
    serde_json::from_str(&s[start..=end]).ok()
}

/// Read-only: what `strata launch` would do, without downloading anything.
pub fn plan(spec: &StrataSpec) -> io::Result<RunOutput> {
    let bin = strata_bin().ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "strata"))?;
    run(&bin, &spec.args(true))
}

/// Same as `launch`, but stdout/stderr go to `log` so the dashboard can tail
/// the launch and the JSON is read back from the file afterwards.
pub fn launch_to_log(spec: &StrataSpec, log: &Path) -> io::Result<RunOutput> {
    use std::process::Stdio;
    let bin = strata_bin().ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "strata"))?;
    if let Some(dir) = log.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let file = std::fs::File::create(log)?;
    let err = file.try_clone()?;
    let mut cmd = Command::new(&bin);
    cmd.args(spec.args(false))
        .stdin(Stdio::null())
        .stdout(Stdio::from(file))
        .stderr(Stdio::from(err));
    if spec.family.as_deref() == Some("qwen36") {
        // 128K leaves too little VRAM for the default chunk 2048 on a 16 GB
        // card; 512 also gives the base engine's text bit for bit.
        cmd.env("STRATA_Q36_CHUNK", "512");
    }
    let status = cmd.status()?;
    let text = std::fs::read_to_string(log).unwrap_or_default();
    Ok(RunOutput {
        code: status.code().unwrap_or(-1),
        json: extract_json(&text),
        text,
    })
}

pub fn stop() -> io::Result<RunOutput> {
    let bin = strata_bin().ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "strata"))?;
    run(&bin, &["launch".into(), "--stop".into()])
}

/// `strata models list --installed` (human output, printed by `localmodel list`)
pub fn installed_text() -> io::Result<RunOutput> {
    let bin = strata_bin().ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "strata"))?;
    let args: Vec<String> = ["models", "list", "--installed"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    run(&bin, &args)
}

/// `strata models inspect <pack> --json` for a single installed pack.
pub fn inspect_model(id: &str) -> io::Result<RunOutput> {
    let bin = strata_bin().ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "strata"))?;
    let args: Vec<String> = ["models", "inspect", id, "--json"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    run(&bin, &args)
}

/// Walk a JSON value for the first key with one of `names` (case-insensitive)
/// whose value is a string or number. Strata's JSON schema is versioned, so we
/// look the fields up instead of pinning a shape.
pub fn find_field<'a>(v: &'a Value, names: &[&str]) -> Option<&'a Value> {
    match v {
        Value::Object(map) => {
            for (k, val) in map {
                if names.iter().any(|n| k.eq_ignore_ascii_case(n)) {
                    if val.is_string() || val.is_number() {
                        return Some(val);
                    }
                }
            }
            for val in map.values() {
                if let Some(found) = find_field(val, names) {
                    return Some(found);
                }
            }
            None
        }
        Value::Array(items) => items.iter().find_map(|v| find_field(v, names)),
        _ => None,
    }
}

pub fn find_str(v: &Value, names: &[&str]) -> Option<String> {
    find_field(v, names).map(|v| match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    })
}

/// The address the freshly started server answers on.
pub fn server_url(json: &Value, port: u16) -> String {
    for key in ["api_url", "base_url", "url", "address", "endpoint"] {
        if let Some(s) = find_str(json, &[key]) {
            if s.starts_with("http") {
                return s.trim_end_matches('/').to_string();
            }
        }
    }
    format!("http://127.0.0.1:{port}")
}

/// Best-effort model name the server reports.
pub fn served_model(json: &Value) -> Option<String> {
    find_str(json, &["model", "model_name", "model_id", "served_model"])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_are_complete() {
        let mut s = StrataSpec::new(131072, 8080);
        s.family = Some("qwen36".into());
        s.size = Some("ud-q6_k_xl".into());
        let a = s.args(false);
        assert!(a.contains(&"--family".to_string()));
        assert!(a.contains(&"qwen36".to_string()));
        assert!(a.contains(&"--yes".to_string()));
        assert!(a.contains(&"--no-tui".to_string()));
        assert_eq!(a.last().unwrap(), "json");
        // qwen36 has no --kv switch (fp32 cache); Flash-Next does.
        assert!(!a.contains(&"--kv".to_string()));
        assert_eq!(s.effective_kv(), "fp32");

        let mut f = StrataSpec::new(131072, 8080);
        f.family = Some("qwen".into());
        f.size = Some("iq3_s".into());
        assert!(f.args(false).contains(&"--kv".to_string()));
        assert_eq!(f.effective_kv(), "int8");
    }

    #[test]
    fn json_is_extracted_from_noise() {
        let v = extract_json("downloading...\n{\"api_url\":\"http://127.0.0.1:8080\"}\ndone\n").unwrap();
        assert_eq!(find_str(&v, &["api_url"]).unwrap(), "http://127.0.0.1:8080");
    }
}
