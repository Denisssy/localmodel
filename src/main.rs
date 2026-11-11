mod config;
mod dev;
mod download;

use crate::dev::Dev;
mod engine;
mod fit;
mod gguf;
mod hub;
mod jinja;
mod knobs;
mod mem;
mod models;
mod resolve;
mod server;
mod settings;
mod stats;
mod telemetry;
mod validate;

use std::fs::File;
use std::io::{BufRead, BufReader, IsTerminal, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crossterm::cursor::{Hide, Show};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::execute;
use dialoguer::theme::ColorfulTheme;
use dialoguer::Select;
use serde_json::{json, Value};

use models::Model;
use resolve::Origin;
use settings::Profile;
use stats::Stats;

#[derive(Clone)]
struct RunOpts {
    query: Vec<String>,
    ctx: Option<u32>,
    port: Option<u16>,
    ngl: Option<u16>,
    ncmoe: Option<u16>,
    load_mode: Option<String>,
    moe_cache_mib: Option<u32>,
    cpu_moe: Option<bool>,
    n_cpu_ffn: Option<u16>,
    override_tensor: Option<String>,
    lazy_mode: Option<String>,
    numa: Option<String>,
    no_op_offload: Option<bool>,
    batch_size: Option<u32>,
    ubatch_size: Option<u32>,
    parallel: Option<u32>,
    threads_batch: Option<u32>,
    prio: Option<i32>,
    cache_reuse: Option<u32>,
    draft_n_min: Option<u16>,
    draft_p_min: Option<f64>,
    top_k: Option<u32>,
    min_p: Option<f64>,
    repeat_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    frequency_penalty: Option<f64>,
    seed: Option<i32>,
    api_key: Option<String>,
    timeout: Option<u32>,
    threads_http: Option<u32>,
    ctk: Option<String>,
    ctv: Option<String>,
    /// draft/MTP context K/V cache type; empty keeps the f16 engine default
    draft_ctk: Option<String>,
    draft_ctv: Option<String>,
    reasoning: Option<String>,
    effort: Option<String>,
    draft_tokens: Option<u16>,
    threads: Option<u32>,
    extra: Option<String>,
    /// --tools value: all | read_file,exec_shell,... (experimental)
    tools: Option<String>,
    /// serve the named settings profile instead of the model's own entry
    profile: Option<String>,
    /// keep the MTP draft on the GPU (small ctx only)
    draft_gpu: Option<bool>,
    /// keep the KV cache in host RAM instead of VRAM
    kv_ram: Option<bool>,
    /// layer | row | tensor | none
    split_mode: Option<String>,
    /// per-card shares, e.g. "0.57,0.43"
    tensor_split: Option<String>,
    /// llama.cpp log level; 4 prints the memory breakdown
    log_verbosity: Option<u32>,
    /// enable multimodal vision input ("on" | "off")
    vision: Option<String>,
    /// path to multimodal projector file (--mmproj)
    mmproj: Option<String>,
    alias: Option<String>,
    repl: bool,
    dry_run: bool,
    /// show where every setting came from, then exit (implies --dry-run)
    explain: bool,
    /// fail instead of warning when `extra` names an unknown or shadowed flag
    strict: bool,
    llama_help: bool,
    /// auto | strata | llama
    engine: Option<String>,
    family: Option<String>,
    size: Option<String>,
    access: Option<String>,
    tune: Option<String>,
    /// Transient settings for `localmodel dev`: a full profile layered over
    /// `[defaults]`, applied in memory and never written back. Lets the
    /// playground override the on-disk model profile without touching
    /// settings.toml.
    dev_scratch: Option<knobs::Profile>,
}

impl Default for RunOpts {
    fn default() -> Self {
        Self {
            query: Vec::new(),
            ctx: None,
            port: None,
            ngl: None,
            ncmoe: None,
            load_mode: None,
            moe_cache_mib: None,
            cpu_moe: None,
            n_cpu_ffn: None,
            override_tensor: None,
            lazy_mode: None,
            numa: None,
            no_op_offload: None,
            batch_size: None,
            ubatch_size: None,
            parallel: None,
            threads_batch: None,
            prio: None,
            cache_reuse: None,
            draft_n_min: None,
            draft_p_min: None,
            top_k: None,
            min_p: None,
            repeat_penalty: None,
            presence_penalty: None,
            frequency_penalty: None,
            seed: None,
            api_key: None,
            timeout: None,
            threads_http: None,
            ctk: None,
            ctv: None,
            draft_ctk: None,
            draft_ctv: None,
            reasoning: None,
            effort: None,
            draft_tokens: None,
            threads: None,
            extra: None,
            tools: None,
            profile: None,
            draft_gpu: None,
            kv_ram: None,
            split_mode: None,
            tensor_split: None,
            log_verbosity: None,
            alias: None,
            repl: false,
            dry_run: false,
            explain: false,
            strict: false,
            llama_help: false,
            engine: None,
            family: None,
            size: None,
            vision: None,
            mmproj: None,
            access: None,
            tune: None,
            dev_scratch: None,
        }
    }
}

/// Fully resolved launch configuration: CLI flags > model profile > defaults.
#[derive(Clone, Default)]
struct Cfg {
    ctx: u32,
    port: Option<u16>,
    ngl: Option<u16>,
    ncmoe: Option<u16>,
    load_mode: Option<String>,
    moe_cache_mib: Option<u32>,
    cpu_moe: bool,
    n_cpu_ffn: Option<u16>,
    override_tensor: Option<String>,
    lazy_mode: Option<String>,
    numa: Option<String>,
    no_op_offload: bool,
    batch_size: Option<u32>,
    ubatch_size: Option<u32>,
    parallel: Option<u32>,
    threads_batch: Option<u32>,
    prio: Option<i32>,
    cache_reuse: Option<u32>,
    draft_n_min: Option<u16>,
    draft_p_min: Option<f64>,
    ctk: String,
    ctv: String,
    /// Draft/MTP context K cache type; empty means the f16 engine default.
    draft_ctk: String,
    /// Draft/MTP context V cache type; empty means the f16 engine default.
    draft_ctv: String,
    threads: Option<u32>,
    reasoning: Option<String>,
    effort: Option<String>,
    draft_tokens: Option<u16>,
    extra: Option<String>,
    /// --tools value: all | read_file,exec_shell,... (experimental)
    tools: Option<String>,
    /// keep the MTP draft on the GPU (small ctx only)
    draft_gpu: bool,
    /// keep the KV cache in host RAM (`--no-kv-offload`)
    kv_ram: bool,
    split_mode: String,
    tensor_split: Option<String>,
    log_verbosity: u32,
    /// multimodal vision input enabled
    vision: bool,
    /// path to multimodal projector file (--mmproj)
    mmproj: Option<String>,
    alias: Option<String>,
    flash_attn: bool,
    temp: f64,
    top_p: f64,
    top_k: Option<u32>,
    min_p: Option<f64>,
    repeat_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    frequency_penalty: Option<f64>,
    seed: Option<i32>,
    host: String,
    api_key: Option<String>,
    timeout: Option<u32>,
    threads_http: Option<u32>,
}

impl Cfg {
    /// Kept for callers that already merged defaults with a model profile.
    /// `explain` is the source of truth; this is the no-provenance wrapper.
    #[cfg(test)]
    fn resolve(o: &RunOpts, p: &Profile) -> Self {
        Self::explain(o, p, None).0
    }
}

#[derive(Clone)]
struct Info {
    alias: String,
    port: u16,
    pid: u32,
    ctx: u32,
    ctk: String,
    ctv: String,
}

/// Leave the alternate screen / raw mode even if something panics, so the
/// user's shell is never left with a hidden cursor.
fn install_panic_hook() {
    let original = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = terminal::disable_raw_mode();
        let _ = execute!(std::io::stdout(), LeaveAlternateScreen, Show);
        original(info);
    }));
}

fn main() {
    install_panic_hook();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(|s| s.as_str()) {
        Some("list") | Some("ls") => cmd_list(),
        Some("add") | Some("get") | Some("pull") => cmd_add(&args[1..]),
        Some("stop") | Some("kill") => cmd_stop(),
        Some("run") => match strata_target(&args[1..]) {
            Some((family, size, rest)) => {
                let mut v = vec![family, size];
                v.extend(rest);
                cmd_strata(&v)
            }
            None => cmd_run(&args[1..]),
        },
        Some("hub") => hub::cmd_hub(&args[1..]),
        Some("dev") => Dev::cmd_dev(&args[1..]),
        Some("strata") => cmd_strata(&args[1..]),
        Some("doctor") | Some("fit") => cmd_doctor(&args[1..]),
        Some("info") | Some("show") => cmd_info(&args[1..]),
        Some("params") | Some("param") => cmd_params(&args[1..]),
        Some("settings") | Some("config") | Some("cfg") => cmd_settings(&args[1..]),
        Some("help") | Some("-h") | Some("--help") => {
            help();
            0
        }
        Some("version") | Some("-V") | Some("--version") => {
            println!("localmodel {}", env!("CARGO_PKG_VERSION"));
            0
        }
        None => {
            if std::io::stdin().is_terminal() {
                cmd_pick()
            } else {
                cmd_list()
            }
        }
        Some(_) => cmd_run(&args),
    };
    std::process::exit(code);
}

fn help() {
    println!(
        "localmodel {} - keep GGUF models in one folder, launch them fast

USAGE:
  localmodel                      pick a model from the list and launch it
  localmodel hub                  models + their launch settings on one screen
  localmodel dev                  playground: tweak every setting, launch, reset on stop
  localmodel run <model> [opts]   launch a model by name substring or index
  localmodel run <family>/<size>  same hub, Strata engine (e.g. qwen36/ud-q4_k_xl)
  localmodel strata <family> <size>  launch a Strata catalog model
  localmodel doctor               what fits this machine, and the commands
  localmodel info <model>         print the model card without launching
  localmodel params [pattern]     llama.cpp parameters cheat sheet / grep --help
  localmodel settings [--show]    edit saved per-model / per-pack / global launch settings
  localmodel settings --model M --set ctx=262144 [--unset ctk]   non-interactive edit
  localmodel settings --model M --show --origin                  where each value comes from
  localmodel add <url> [--name F] download a .gguf into the models folder
  localmodel list                 list downloaded models
  localmodel stop                 stop running localmodel servers

ENGINES:
  Strata runs only its own families: Qwen3.6-35B-A3B, Qwen3.8-Flash-Next
  (and Coder/Swift/Gyro/NVFP4 variants) and DeepSeek V4.1-Flash. Every other
  model, dense or MoE, stays on llama.cpp. `--engine llama|strata|auto`
  overrides the choice (default: auto).

RUN OPTIONS:
  --ctx N        context size          (default 131072)
  --port N       port                  (default: first free from 8080)
  --kv T         KV cache type for K and V   (default q8_0)
  --ctk T        KV cache type for K         (default q8_0)
  --ctv T        KV cache type for V         (default q8_0)
--draft-ctk T  draft/MTP context K for models with a draft head (default f16)
--draft-ctv T  draft/MTP context V for models with a draft head (default f16)
  --ncmoe N      experts kept on CPU   (big MoE models only, default 58)
  --ngl N        layers on GPU         (default 99 for dense, 999 for MoE)
  --engine E     auto | strata | llama
  --family F     Strata family (qwen36, qwen, coder, gyro, unsloth)
  --size S       Strata pack  (ud-q6_k_xl, iq2_xs, gyro-s, ...)
  --vision V     Strata pictures: off | on | cpu | gpu  (default off)
  --access A     Strata access: local | lan            (default local)
  --tune T       Strata tuning: skip | quick | thorough (default skip)
  --reasoning K  on | off
  --effort L     thinking level: high | xhigh | medium | low | off
  --draft-tokens N  speculative draft tokens per step (built-in MTP: 3)
  --tools L      built-in agent tools: all | off | read_file,exec_shell
  --profile NAME use a named settings profile instead of the model's own
  --draft-gpu    external draft on GPU (built-in MTP ignores this)
  --kv-ram       keep the KV cache in host RAM (frees VRAM, costs speed)
  --kv-vram      keep the KV cache in VRAM (default)
  --split-mode M two-card split: layer | row | tensor | none (default layer)
  --tensor-split S  per-card share, e.g. 0.57,0.43 (default even)
  --log-verbosity N llama.cpp log level 0..4 (default 4: memory breakdown)
  --threads N    CPU threads
  --alias NAME   served model name
  --extra \"...\"  extra engine flags (after saved ones, so they win)
  --repl         chat in the terminal instead of the dashboard
  --dry-run      print the exact command and exit
  --explain      print where each setting came from, then exit (implies --dry-run)
  --strict       treat unknown/shadowed `extra` flags as errors
  --llama-help   print `llama-server --help` and exit

The dashboard is the default on a TTY: the card stays put, stats update in
place, and these keys work without leaving it:
  ?  h    help            s    hint to use `localmodel hub`
  l       toggle the log  r    restart         q  Esc  Ctrl-C   stop and quit
`localmodel hub` is the one place to see a model, edit only the knobs that
apply to it, preview the argv and launch it. `localmodel params` lists the
flags you can pass via --extra without digging through llama.cpp docs.
`localmodel settings` saves the same values non-interactively: global
defaults, per-model overrides and per-pack overrides
for Strata hub entries ([model.\"qwen36/ud_q4_k_xl\"]), applied automatically by
`run` and the picker. CLI flags always win over saved settings.

ENV:
  LOCALMODEL_DIR  models folder    (default ~/models)
  LLAMA_SERVER    llama-server bin (default ~/llama.cpp/build/bin/llama-server)",
        env!("CARGO_PKG_VERSION")
    );
}

fn cmd_list() -> i32 {
    let dir = config::models_dir();
    let ms = models::scan(&dir);
    if ms.is_empty() {
        println!("no models in {}", dir.display());
        return 0;
    }
    println!("models in {}:", dir.display());
    for (i, m) in ms.iter().enumerate() {
        let draft = if m.nextn {
            "  +mtp"
        } else if models::find_draft(&m.path).is_some() {
            "  +draft"
        } else {
            ""
        };
        let kind = if m.moe { "MoE" } else { "Dense" };
        let eng = engine::route(m, None).label();
        println!(
            "  {:>2}.  {:<44} {:<5} {:>7}  {:<9}{}",
            i + 1,
            m.name,
            kind,
            models::human(m.size),
            eng,
            draft
        );
    }
    for s in server::list_servers(&dir.to_string_lossy()) {
        println!("  running: pid {} - {}", s.pid, truncate(&s.cmd, 80));
    }
    if engine::strata_bin().is_some() {
        if let Ok(o) = engine::installed_text() {
            let text = o.text.trim();
            if !text.is_empty() {
                println!("\nstrata packs:");
                for l in text.lines() {
                    println!("  {l}");
                }
            }
        }
    }
    0
}

fn cmd_add(args: &[String]) -> i32 {
    let mut dir = config::models_dir();
    let mut name: Option<String> = None;
    let mut urls: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--name" | "-n" => {
                i += 1;
                name = args.get(i).cloned();
            }
            "--dir" => {
                i += 1;
                if let Some(d) = args.get(i) {
                    dir = PathBuf::from(d);
                }
            }
            u if !u.starts_with('-') => urls.push(u.to_string()),
            _ => {}
        }
        i += 1;
    }
    if urls.is_empty() {
        eprintln!("usage: localmodel add <url> [--name file.gguf]");
        return 2;
    }
    let single = urls.len() == 1;
    for u in &urls {
        let n = if single { name.as_deref() } else { None };
        if let Err(e) = download::add(u, &dir, n) {
            eprintln!("error: {e}");
            return 1;
        }
    }
    0
}

fn cmd_stop() -> i32 {
    let dir = config::models_dir().to_string_lossy().to_string();
    let servers = server::list_servers(&dir);
    for s in &servers {
        println!("stopping pid {} - {}", s.pid, truncate(&s.cmd, 70));
        let _ = Command::new("kill").args(["-TERM", &s.pid.to_string()]).status();
    }
    if servers.is_empty() {
        println!("no running llama.cpp servers");
    }
    // Strata keeps its own server; `--stop` is a no-op when nothing serves.
    if engine::strata_bin().is_some() {
        match engine::stop() {
            Ok(o) if o.code == 0 => {
                let text = o.text.trim();
                if !text.is_empty() {
                    println!("strata: {text}");
                }
            }
            Ok(o) => println!("strata stop exited with {}: {}", o.code, o.text.trim()),
            Err(e) => println!("strata: {e}"),
        }
    }
    0
}

/// Parse `strata models list --installed` output into (family, size) pairs.
/// Only lines whose second column is `present` are installed.
fn strata_packs_from(text: &str) -> Vec<(String, String)> {
    text.lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            let id = it.next()?;
            if it.next() != Some("present") || !id.contains('-') {
                return None;
            }
            let (fam, size) = id.split_once('-')?;
            Some((fam.to_string(), size.to_string()))
        })
        .collect()
}

/// Installed Strata packs as (family, size) for the picker.
fn strata_packs() -> Vec<(String, String)> {
    match engine::installed_text() {
        Ok(o) => strata_packs_from(&o.text),
        Err(_) => Vec::new(),
    }
}

/// A named side profile (e.g. an agent variant) attached to a model, for the
/// picker: (profile key, model file stem, profile).
fn side_profiles(saved: &settings::Settings, ms: &[Model]) -> Vec<(String, String)> {
    let stems: Vec<String> = ms.iter().map(|m| models::stem(&m.name)).collect();
    saved
        .model
        .iter()
        .filter_map(|(key, p)| {
            let target = p.applies_to.as_ref()?;
            if key == target || !stems.iter().any(|s| s == target) {
                return None;
            }
            Some((key.clone(), target.clone()))
        })
        .collect()
}

fn cmd_pick() -> i32 {
    let dir = config::models_dir();
    let ms = models::scan(&dir);
    let packs = strata_packs();
    let saved = settings::load();
    let profiles = side_profiles(&saved, &ms);
    if ms.is_empty() && packs.is_empty() && profiles.is_empty() {
        eprintln!("no models in {}. add one: localmodel add <url>", dir.display());
        return 1;
    }
    let mut items: Vec<String> = ms
        .iter()
        .map(|m| {
            let draft = if m.nextn {
                "  +mtp"
            } else if models::find_draft(&m.path).is_some() {
                "  +draft"
            } else {
                ""
            };
            format!("{:<48} {:>7}{}", m.name, models::human(m.size), draft)
        })
        .collect();
    for (key, target) in &profiles {
        items.push(format!("{target} ({key}){:<27} profile", ""));
    }
    for (fam, size) in &packs {
        let tuned = if settings::find_key(&saved, &format!("{fam}/{size}")).is_some() {
            "  +settings"
        } else {
            ""
        };
        items.push(format!("{fam}/{size}{:<40} strata{tuned}", ""));
    }
    let idx = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select a model")
        .items(&items)
        .default(0)
        .interact()
        .unwrap_or(0);
    if idx < ms.len() {
        return cmd_run(&[ms[idx].name.clone()]);
    }
    let mut rest = idx - ms.len();
    if rest < profiles.len() {
        let (key, target) = &profiles[rest];
        return cmd_run(&["--profile".to_string(), key.clone(), target.clone()]);
    }
    rest -= profiles.len();
    let Some((fam, size)) = packs.get(rest) else {
        return 1;
    };
    cmd_strata(&[fam.clone(), size.clone()])
}

/// Print the model card without launching anything.
fn cmd_info(args: &[String]) -> i32 {
    let mut ctx: Option<u32> = None;
    let mut ctk: Option<String> = None;
    let mut ctv: Option<String> = None;
    let mut port: Option<u16> = None;
    let mut kv_ram: Option<bool> = None;
    let mut split_mode: Option<String> = None;
    let mut tensor_split: Option<String> = None;
    let mut draft_ctk: Option<String> = None;
    let mut draft_ctv: Option<String> = None;
    let mut query: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let val = |i: &mut usize| -> Option<String> {
            *i += 1;
            args.get(*i).cloned()
        };
        match a {
            "--ctx" | "-c" => ctx = val(&mut i).and_then(|v| v.parse().ok()),
            "--port" | "-p" => port = val(&mut i).and_then(|v| v.parse().ok()),
            "--kv" => {
                if let Some(v) = val(&mut i) {
                    ctk = Some(v.clone());
                    ctv = Some(v);
                }
            }
            "--kv-ram" => kv_ram = Some(true),
            "--kv-vram" => kv_ram = Some(false),
            "--split-mode" => split_mode = val(&mut i),
            "--tensor-split" => tensor_split = val(&mut i),
            "--draft-ctk" => draft_ctk = Some(val(&mut i).unwrap_or_default()),
            "--draft-ctv" => draft_ctv = Some(val(&mut i).unwrap_or_default()),
            s if s.starts_with('-') => {}
            s => query.push(s.to_string()),
        }
        i += 1;
    }
    let dir = config::models_dir();
    let ms = models::scan(&dir);
    if query.is_empty() {
        eprintln!("usage: localmodel info <model>");
        return 2;
    }
    let q = query.join(" ").to_lowercase();

    // GGUF models come first: a numeric index or a substring match.
    let gguf_match = q.parse::<usize>().map(|n| n >= 1 && n <= ms.len()).unwrap_or(false)
        || ms.iter().any(|m| m.name.to_lowercase().contains(&q));

    if gguf_match {
        let model = match pick(&ms, &query) {
            Some(m) => m,
            None => return 1,
        };
        let stem = models::stem(&model.name);
        let draft = if model.nextn {
            None
        } else {
            models::find_draft(&model.path)
        };
        let saved = settings::load();
        let model_profile = saved.model.get(&stem);
        let mut o = RunOpts::default();
        o.query = query.clone();
        o.ctx = ctx;
        o.ctk = ctk;
        o.ctv = ctv;
        o.port = port;
        o.kv_ram = kv_ram;
        o.split_mode = split_mode;
        o.tensor_split = tensor_split;
        o.draft_ctk = draft_ctk;
        o.draft_ctv = draft_ctv;
        let (cfg, _) = Cfg::explain(&o, &saved.defaults, model_profile);
        let alias = cfg.alias.clone().unwrap_or_else(|| stem.clone());
        let port = cfg.port.unwrap_or(8080);
        let logpath = dir.join(format!(".localmodel-{port}.log"));
        let launch = format!("localmodel run {alias}");
        if std::io::stdout().is_terminal() {
            println!(
                "{}",
                render_info(&model, &alias, port, &cfg, &logpath, "llama.cpp", &launch, draft.as_deref())
            );
        } else {
            println!(
                "{}",
                model_card(&model, &alias, port, &cfg, &logpath, "llama.cpp", draft.as_deref())
            );
        }
        return 0;
    }

    // Strata packs are virtual, so inspect the installed pack instead of a file.
    let strata = match find_strata_pack(&query) {
        Ok(found) => found,
        Err(message) => {
            eprintln!("{message}");
            return 1;
        }
    };
    if let Some(info) = strata {
        let saved = settings::load();
        let prof = saved.for_model(&info.served);
        let ctx = ctx.or(prof.ctx).unwrap_or(info.context.max(1));
        let default_kv = if info.family == "qwen36" { "fp32" } else { "int8" };
        let ctk = ctk.or(prof.ctk).unwrap_or_else(|| default_kv.into());
        let ctv = ctv.or(prof.ctv).unwrap_or_else(|| ctk.clone());
        let port = port.or(prof.port).unwrap_or(8080);
        let logpath = dir.join(format!(".localmodel-strata-{port}.log"));
        let model = Model {
            path: info.native_path.clone().unwrap_or_default(),
            name: info.id.clone(),
            size: info.size_bytes,
            quant: info.quant.clone(),
            layers: info.layers,
            moe: info.moe,
            experts_total: info.experts_total,
            experts_active: info.experts_active,
            nextn: false,
            meta: gguf::Meta::default(),
        };
        let engine = info.engine_label();
        let launch = format!("localmodel strata {} {}", info.family, info.size);
        if std::io::stdout().is_terminal() {
            println!(
                "{}",
                render_info(&model, &info.served, port, &card_cfg(ctx, &ctk, &ctv, "", ""), &logpath, &engine, &launch, None)
            );
        } else {
            println!(
                "{}",
                model_card(&model, &info.served, port, &card_cfg(ctx, &ctk, &ctv, "", ""), &logpath, &engine, None)
            );
        }
        return 0;
    }

    if ms.is_empty() {
        eprintln!("no models in {}", dir.display());
    } else {
        eprintln!("no model matches '{q}'");
    }
    1
}

/// Metadata for an installed Strata pack, used by `localmodel info`.
struct StrataPackInfo {
    id: String,
    family: String,
    size: String,
    served: String,
    context: u32,
    engine_version: Option<String>,
    backend: Option<String>,
    native_path: Option<PathBuf>,
    size_bytes: u64,
    layers: u32,
    quant: String,
    moe: bool,
    experts_total: u32,
    experts_active: u32,
}

impl StrataPackInfo {
    fn engine_label(&self) -> String {
        let version = self.engine_version.as_deref().unwrap_or("strata");
        let backend = match self.backend.as_deref() {
            Some(b) if b.contains("cuda") => "CUDA",
            Some(b) if b.contains("cpu") => "CPU",
            Some(b) => b,
            None => "unknown",
        };
        format!("strata {version} ({backend})")
    }
}

fn find_strata_pack(query: &[String]) -> Result<Option<StrataPackInfo>, String> {
    if query.is_empty() {
        return Ok(None);
    }
    let q = query.join(" ").to_lowercase();
    let q_alt = q.replace('/', "-");
    let mut matches: Vec<StrataPackInfo> = Vec::new();
    for (family, size) in strata_packs() {
        let id = format!("{family}-{size}");
        let info = strata_pack_info(&id);
        let served = info.as_ref().map(|i| i.served.as_str()).unwrap_or("");
        let hay = format!("{id} {family}/{size} {served}").to_lowercase();
        if hay.contains(&q) || hay.replace('/', "-").contains(&q_alt) {
            if let Some(info) = info {
                matches.push(info);
            }
        }
    }
    match matches.len() {
        0 => Ok(None),
        1 => Ok(matches.pop()),
        _ => Err(format!(
            "multiple strata packs match '{q}'; specify one of: {}",
            matches.iter().map(|m| m.id.as_str()).collect::<Vec<_>>().join(", ")
        )),
    }
}

fn strata_pack_info(pack_id: &str) -> Option<StrataPackInfo> {
    let out = engine::inspect_model(pack_id).ok()?;
    if out.code != 0 {
        return None;
    }
    let json = out.json?;
    let model = json.get("model")?;
    let id = model.get("id").and_then(Value::as_str).unwrap_or(pack_id).to_string();
    let served = model.get("name").and_then(Value::as_str).unwrap_or(&id).to_string();
    let context = model.get("context_tokens").and_then(Value::as_u64).unwrap_or(0) as u32;
    let backend = json
        .pointer("/components/backend")
        .and_then(Value::as_str)
        .or_else(|| model.get("backend").and_then(Value::as_str))
        .map(str::to_string);

    let mut engine_version = None;
    let mut native_path = None;
    if let Some(components) = json.pointer("/components/components").and_then(Value::as_array) {
        for component in components {
            match component.get("name").and_then(Value::as_str).unwrap_or("") {
                "engine" => {
                    engine_version = component.get("version").and_then(Value::as_str).map(str::to_string)
                }
                "native" => {
                    native_path = component.get("path").and_then(Value::as_str).map(PathBuf::from)
                }
                _ => {}
            }
        }
    }

    let size_bytes = native_path
        .as_ref()
        .and_then(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .unwrap_or(0);
    let (layers, meta_quant) = native_path
        .as_ref()
        .map(|p| {
            let meta = gguf::parse(p);
            (meta.layers, meta.quant)
        })
        .unwrap_or((0, String::new()));
    // The file name carries the exact Unsloth quant (Q4_K_XL); the GGUF tensor
    // type only says Q4_K, so prefer the name like `models::scan` does.
    let name_quant = gguf::quant_from_name(&served);
    let quant = if name_quant == "unknown" {
        meta_quant
    } else {
        name_quant
    };
    let lower = served.to_lowercase();
    let moe = lower.contains("a3b")
        || lower.contains("a9b")
        || lower.contains("moe")
        || lower.contains("nemotron");
    let (experts_total, experts_active) = if lower.contains("qwen3.6") && lower.contains("a3b") {
        (256, 8)
    } else {
        (0, 0)
    };
    let (family, size) = id
        .split_once('-')
        .map(|(f, s)| (f.to_string(), s.to_string()))
        .unwrap_or_default();

    Some(StrataPackInfo {
        id,
        family,
        size,
        served,
        context,
        engine_version,
        backend,
        native_path,
        size_bytes,
        layers,
        quant,
        moe,
        experts_total,
        experts_active,
    })
}

/// Pick a model by substring (case-insensitive) or 1-based index. No dialog.
fn pick(ms: &[Model], query: &[String]) -> Option<Model> {
    if query.is_empty() {
        return None;
    }
    let q = query.join(" ").to_lowercase();
    if let Ok(n) = q.parse::<usize>() {
        if n >= 1 && n <= ms.len() {
            return Some(ms[n - 1].clone());
        }
    }
    let matches: Vec<&Model> = ms.iter().filter(|m| m.name.to_lowercase().contains(&q)).collect();
    match matches.len() {
        0 => {
            eprintln!("no model matches '{q}'");
            None
        }
        1 => Some(matches[0].clone()),
        _ => {
            eprintln!(
                "multiple models match '{q}'; specify one of: {}",
                matches.iter().map(|m| m.name.as_str()).collect::<Vec<_>>().join(", ")
            );
            None
        }
    }
}

/// House default for models whose template exposes effort knobs: medium for
/// Qwen-style (xhigh overthinks for chat), full thinking for Nemotron-style.
fn default_effort(caps: &jinja::TemplateCaps) -> Option<String> {
    if caps.has_medium_effort {
        Some("high".to_string())
    } else if !caps.effort_values.is_empty() {
        Some("medium".to_string())
    } else {
        None
    }
}

fn build_args(m: &Model, draft: Option<&Path>, alias: &str, port: u16, c: &Cfg) -> Vec<String> {
    let n = m.name.to_lowercase();
    let mut a: Vec<String> = Vec::new();
    let push = |a: &mut Vec<String>, k: &str, v: &str| {
        a.push(k.to_string());
        a.push(v.to_string());
    };
    // Flag spelling comes from the registry, so `validate::check` (which
    // reads `knob.argv` to spot shadowing `extra`) and the hub's argv panel
    // agree with what is actually launched.
    let f = |key: &str| knobs::knob(key).map(|k| k.argv).unwrap_or("");
    let extra_flags = c.extra.as_deref().unwrap_or("");
    let shadowed_flag = |flag: &str| {
        extra_flags
            .split_whitespace()
            .any(|p| p == flag || p.starts_with(&format!("{flag}=")))
    };
    push(&mut a, "-m", &m.path.display().to_string());
    push(&mut a, f("alias"), alias);
    push(&mut a, f("ctx"), &c.ctx.to_string());
    push(&mut a, f("flash_attn"), if c.flash_attn { "on" } else { "off" });
    push(&mut a, f("ctk"), &c.ctk);
    push(&mut a, f("ctv"), &c.ctv);
    // Draft/MTP cache types. Emitted only when the user set a type and no
    // `extra` flag already names the same option (checked by both the long and
    // short spelling, so `-ctkd q4_0` in extra is not duplicated); `fit::
    // draft_kv_types` resolves the effective type for the card display.
    let extra = c.extra.as_deref().unwrap_or("");
    let shadowed = |key: &str| -> bool {
        let aliases = match key {
            "draft_ctk" => ["--cache-type-k-draft", "-ctkd"],
            "draft_ctv" => ["--cache-type-v-draft", "-ctvd"],
            _ => return false,
        };
        extra.split_whitespace().any(|p| {
            aliases.iter().any(|al| *al == p || p.starts_with(&format!("{al}=")))
        })
    };
    if !c.draft_ctk.is_empty() && !shadowed("draft_ctk") {
        push(&mut a, f("draft_ctk"), &c.draft_ctk);
    }
    if !c.draft_ctv.is_empty() && !shadowed("draft_ctv") {
        push(&mut a, f("draft_ctv"), &c.draft_ctv);
    }
    push(&mut a, f("temp"), &c.temp.to_string());
    push(&mut a, f("top_p"), &c.top_p.to_string());
    push(&mut a, f("host"), &c.host);
    push(&mut a, f("port"), &port.to_string());
    a.push("--jinja".to_string());
    // Multimodal vision: Gemma 4 Unified needs an external mmproj projector
    // file for llama.cpp. Pass --mmproj when configured, otherwise rely on
    // --mmproj-auto.
    if c.vision {
        if let Some(p) = &c.mmproj {
            a.push(f("mmproj").to_string());
            a.push(p.clone());
        } else {
            a.push("--mmproj-auto".to_string());
        }
    }
    // Prometheus counters at /metrics: the dashboard and benchmarks read
    // cumulative token totals from here.
    a.push("--metrics".to_string());
    // Reuse the KV cache across requests in >=256-token chunks: repeated
    // system prompts / long chat prefixes skip re-prefill instead of being
    // recomputed. 0 disables reuse entirely.
    if !shadowed_flag("--cache-reuse") {
        push(
            &mut a,
            f("cache_reuse"),
            &c.cache_reuse.unwrap_or(256).to_string(),
        );
    }
    // Batch sizes and request parallelism were hardcoded per model before;
    // they are real knobs now so a slow or memory-tight machine can tune them
    // without dropping into `extra`.
    if !shadowed_flag("-b") && !shadowed_flag("--batch-size") {
        push(&mut a, f("batch_size"), &c.batch_size.unwrap_or(2048).to_string());
    }
    if !shadowed_flag("-ub") && !shadowed_flag("--ubatch-size") {
        push(&mut a, f("ubatch_size"), &c.ubatch_size.unwrap_or(512).to_string());
    }
    if !shadowed_flag("--parallel") {
        push(&mut a, f("parallel"), &c.parallel.unwrap_or(1).to_string());
    }

    // Where the KV cache and the cards' share live. `kv_ram` is a valueless
    // flag (`--no-kv-offload`), so it is emitted only when on; the other two
    // only when they differ from llama.cpp's own defaults. That keeps the
    // common single-card argv unchanged.
    if c.kv_ram {
        a.push(f("kv_ram").to_string());
    }
    if c.split_mode != "layer" {
        push(&mut a, f("split_mode"), &c.split_mode);
    }
    if let Some(ts) = &c.tensor_split {
        push(&mut a, f("tensor_split"), ts);
    }
    // Level 4 is what prints llama.cpp's per-device memory breakdown; the
    // placement card parses that instead of guessing.
    push(&mut a, f("log_verbosity"), &c.log_verbosity.to_string());

    // Memory placement knobs that used to require hand-written `extra`.
    if let Some(v) = &c.load_mode {
        push(&mut a, f("load_mode"), v);
    }
    if let Some(v) = c.moe_cache_mib {
        push(&mut a, f("moe_cache_mib"), &v.to_string());
    }
    if c.cpu_moe {
        a.push(f("cpu_moe").to_string());
    }
    if let Some(v) = c.n_cpu_ffn {
        push(&mut a, f("n_cpu_ffn"), &v.to_string());
    }
    if let Some(v) = &c.override_tensor {
        push(&mut a, f("override_tensor"), v);
    }
    if let Some(v) = &c.lazy_mode {
        push(&mut a, f("lazy_mode"), v);
    }
    if let Some(v) = &c.numa {
        push(&mut a, f("numa"), v);
    }
    if c.no_op_offload {
        a.push(f("no_op_offload").to_string());
    }

    // Effort/thinking control, driven by what the template actually reads
    // (Qwen3.8: reasoning_effort; Nemotron: medium_effort). Explicit setting
    // wins; otherwise Qwen-style templates default to 'medium' (comfortable,
    // avoids the xhigh overthinking) and Nemotron-style to full thinking.
    let caps = jinja::caps_for_model(&m.path);
    if let Some(cp) = &caps {
        let effort = c
            .effort
            .clone()
            .or_else(|| default_effort(cp));
        if let Some(kw) = jinja::effort_kwargs(cp, effort.as_deref()) {
            push(&mut a, f("effort"), &kw.to_string());
        }
    }

    if n.contains("nemotron") {
        push(&mut a, f("ngl"), &c.ngl.unwrap_or(999).to_string());
        push(&mut a, f("ncmoe"), &c.ncmoe.unwrap_or(58).to_string());
        if let Some(k) = c.top_k {
            if !shadowed_flag("--top-k") {
                push(&mut a, f("top_k"), &k.to_string());
            }
        }
        if let Some(p) = c.min_p {
            push(&mut a, f("min_p"), &p.to_string());
        }
        if let Some(d) = draft {
            push(&mut a, "--spec-type", "draft-dflash");
            push(&mut a, "--spec-draft-model", &d.display().to_string());
            push(&mut a, f("draft_gpu"), "999");
            push(&mut a, f("draft_tokens"), &c.draft_tokens.unwrap_or(7).to_string());
        }
    } else if m.moe {
        push(&mut a, f("ngl"), &c.ngl.unwrap_or(99).to_string());
        push(&mut a, f("ncmoe"), &c.ncmoe.unwrap_or(58).to_string());
        if let Some(k) = c.top_k {
            if !shadowed_flag("--top-k") {
                push(&mut a, f("top_k"), &k.to_string());
            }
        }
        if let Some(p) = c.min_p {
            push(&mut a, f("min_p"), &p.to_string());
        }
        if m.nextn {
            // Built-in MTP head (blk.N.nextn.*): llama.cpp drafts from the
            // target model itself, so there is no --spec-draft-model to pass.
            // The nextn experts land where the CPU-MoE split puts them.
            push(&mut a, "--spec-type", "draft-mtp");
            push(&mut a, f("draft_tokens"), &c.draft_tokens.unwrap_or(3).to_string());
        } else if let Some(d) = draft {
            push(&mut a, "--spec-type", "draft-dflash");
            push(&mut a, "--spec-draft-model", &d.display().to_string());
            push(&mut a, f("draft_gpu"), "999");
            push(&mut a, f("draft_tokens"), &c.draft_tokens.unwrap_or(7).to_string());
        }
    } else {
        push(&mut a, f("ngl"), &c.ngl.unwrap_or(99).to_string());
        // A profile's `extra` may set --top-k (e.g. the model card's
        // recommended sampling), so only default otherwise.
        if !shadowed_flag("--top-k") {
            let k = c.top_k.unwrap_or(if n.contains("qwen") || n.contains("ornith") {
                // Unsloth/Qwen best practice: top_k 20 for thinking models.
                20
            } else {
                64
            });
            push(&mut a, f("top_k"), &k.to_string());
        }
        push(&mut a, f("min_p"), &c.min_p.unwrap_or(0.0).to_string());
        if m.nextn {
            // Unsloth UD quants already contain the MTP head (blk.N.nextn.*).
            // llama.cpp builds the draft context from the target model itself,
            // so there is no --spec-draft-model, -ngl or -device to pass.
            // n-max 3 measured 40-44 tok/s on a 4060 Ti 16 GB at 32K (n=1: ~30).
            push(&mut a, "--spec-type", "draft-mtp");
            push(&mut a, f("draft_tokens"), &c.draft_tokens.unwrap_or(3).to_string());
        } else if let Some(d) = draft {
            // External draft head (older models without a built-in MTP layer).
            // Qwen3.8 MTP head on the CPU, the same split Strata does in
            // hybrid mode. --spec-draft-device none is the important part:
            // without it llama.cpp still reserves ~700 MiB of draft compute
            // buffers on the GPU even with --spec-draft-ngl 0, which does not
            // fit beside a 12+ GB model and 128K of KV.
            let is_mtp = d.to_string_lossy().to_lowercase().contains("mtp");
            let spec = if is_mtp { "draft-mtp" } else { "draft-dflash" };
            push(&mut a, "--spec-type", spec);
            push(&mut a, "--spec-draft-model", &d.display().to_string());
            if c.draft_gpu {
                // Draft on the GPU: only fits with a small context (32-48K
                // with q4 KV). Decode gains ~30% since the CPU draft no
                // longer competes with prefill for threads.
                push(&mut a, f("draft_gpu"), "999");
            } else {
                push(&mut a, f("draft_gpu"), "0");
                push(&mut a, "--spec-draft-device", "none");
            }
            // A standalone MTP head predicts exactly one token ahead; asking
            // for more means chaining the head through itself. One draft
            // token per step is the sweet spot there.
            let n = c.draft_tokens.unwrap_or(if is_mtp { 1 } else { 7 });
            push(&mut a, f("draft_tokens"), &n.to_string());
        }
    }

    if let Some(r) = &c.reasoning {
        push(&mut a, f("reasoning"), r);
    }
    if let Some(t) = c.threads {
        push(&mut a, f("threads"), &t.to_string());
    }
    if let Some(t) = c.threads_batch {
        push(&mut a, f("threads_batch"), &t.to_string());
    }
    if let Some(p) = c.prio {
        push(&mut a, f("prio"), &p.to_string());
    }
    if let Some(v) = c.draft_n_min {
        push(&mut a, f("draft_n_min"), &v.to_string());
    }
    if let Some(v) = c.draft_p_min {
        push(&mut a, f("draft_p_min"), &v.to_string());
    }
    if let Some(v) = c.repeat_penalty {
        push(&mut a, f("repeat_penalty"), &v.to_string());
    }
    if let Some(v) = c.presence_penalty {
        push(&mut a, f("presence_penalty"), &v.to_string());
    }
    if let Some(v) = c.frequency_penalty {
        push(&mut a, f("frequency_penalty"), &v.to_string());
    }
    if let Some(v) = c.seed {
        push(&mut a, f("seed"), &v.to_string());
    }
    if let Some(v) = &c.api_key {
        push(&mut a, f("api_key"), v);
    }
    if let Some(v) = c.timeout {
        push(&mut a, f("timeout"), &v.to_string());
    }
    if let Some(v) = c.threads_http {
        push(&mut a, f("threads_http"), &v.to_string());
    }
    if let Some(t) = &c.tools {
        // llama.cpp built-in agent tools (file/shell access). Off by default:
        // the server is unauthenticated and bound to localhost only.
        push(&mut a, f("tools"), t);
    }
    if let Some(e) = &c.extra {
        for p in e.split_whitespace() {
            a.push(p.to_string());
        }
    }
    a
}

/// "128k" / "131072" / "32K" -> tokens.
fn parse_ctx(s: &str) -> Option<u32> {
    let t = s.trim().to_lowercase();
    let (num, mult) = if let Some(v) = t.strip_suffix('k') {
        (v, 1024u32)
    } else if let Some(v) = t.strip_suffix('m') {
        (v, 1024u32 * 1024)
    } else {
        (t.as_str(), 1u32)
    };
    num.trim().parse::<u32>().ok().map(|n| n.saturating_mul(mult))
}

/// llama.cpp KV type -> Strata `--kv` format.
fn strata_kv(ctk: &str) -> String {
    match ctk.to_lowercase().as_str() {
        "f16" | "fp16" | "bf16" | "f32" => "fp16".into(),
        "q4_0" | "q4_1" | "iq4_nl" => "q4_0".into(),
        "k8v4" => "k8v4".into(),
        _ => "int8".into(),
    }
}

fn strata_spec_from(o: &RunOpts, cfg: &Cfg, port: u16) -> engine::StrataSpec {
    let mut s = engine::StrataSpec::new(cfg.ctx, port);
    s.kv = strata_kv(&cfg.ctk);
    s.vision = o.vision.clone().unwrap_or_else(|| "off".into());
    s.access = o.access.clone().unwrap_or_else(|| "local".into());
    s.tune = o.tune.clone().unwrap_or_else(|| "skip".into());
    if let Some(e) = &o.extra {
        s.extra = e.split_whitespace().map(String::from).collect();
    }
    s
}

/// Run a model on Strata and watch it over HTTP. Strata owns the process, so
/// there is no child to reap: stopping means `strata launch --stop`.
#[allow(clippy::too_many_arguments)]
fn run_strata(
    spec: engine::StrataSpec,
    model: Option<&Model>,
    alias: &str,
    ctx: u32,
    port: u16,
    repl: bool,
    dry_run: bool,
    dir: &Path,
) -> i32 {
    let bin = engine::strata_bin();

    if dry_run {
        let mut parts = vec![shell_arg(
            &bin
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "strata".into()),
        )];
        parts.extend(spec.args(false).iter().map(|a| shell_arg(a)));
        println!("{}", parts.join(" "));
        return 0;
    }

    if bin.is_none() {
        eprintln!("{}", engine::install_hint());
        return 1;
    }

    let logpath = dir.join(format!(".localmodel-strata-{port}.log"));
    println!("engine  strata  ({})", spec.display());
    println!("log     {}", logpath.display());
    println!();

    let out = match engine::launch_to_log(&spec, &logpath) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("failed to run strata: {e}");
            return 1;
        }
    };
    if out.code != 0 {
        eprintln!("strata launch exited with {}", out.code);
        let lines: Vec<&str> = out.text.lines().collect();
        for l in lines.iter().skip(lines.len().saturating_sub(15)) {
            eprintln!("  {l}");
        }
        return 1;
    }

    let json = out.json.clone().unwrap_or(serde_json::Value::Null);
    let url = engine::server_url(&json, port);
    let served = engine::served_model(&json).unwrap_or_else(|| alias.to_string());

    let display_model = match model {
        Some(m) => m.clone(),
        None => Model {
            path: PathBuf::new(),
            name: spec.display(),
            size: 0,
            quant: spec.size.clone().unwrap_or_default().to_uppercase(),
            layers: 0,
            moe: true,
            experts_total: 256,
            experts_active: 8,
            nextn: false,
            meta: gguf::Meta::default(),
        },
    };

    let info = Info {
        alias: served.clone(),
        port,
        pid: 0,
        ctx,
        ctk: spec.effective_kv().to_string(),
        ctv: spec.effective_kv().to_string(),
    };

    println!(
        "{}",
        model_card(
            &display_model,
            &served,
            port,
            &card_cfg(ctx, spec.effective_kv(), spec.effective_kv(), "", ""),
            &logpath,
            "strata",
            None,
        )
    );
    println!();
    println!("url    {url}/v1");
    println!("stop   localmodel stop   (or `strata launch --stop`)");
    println!();

    let mut poller = stats::MetricsPoller::new(ctx as u64);

    if repl {
        loop {
            print!("\u{00bb} ");
            let _ = std::io::stdout().flush();
            let mut line = String::new();
            match std::io::stdin().read_line(&mut line) {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
            let t = line.trim();
            match t {
                "" => continue,
                "exit" | "quit" | "q" | "/quit" | "/exit" | "/stop" => break,
                "/help" => repl_help(),
                "/stats" => {
                    if let Some(m) = curl_get(&format!("{url}/metrics")) {
                        poller.observe(&m);
                    }
                    println!("{}", stats_line(&info, &poller.stats, &display_model));
                }
                other => {
                    print!("\u{2026} ");
                    let _ = std::io::stdout().flush();
                    match chat(port, &served, other, None, None) {
                        Some(r) => println!("\n{r}"),
                        None => println!("(no response)"),
                    }
                    if let Some(m) = curl_get(&format!("{url}/metrics")) {
                        poller.observe(&m);
                    }
                    println!("{}", stats_line(&info, &poller.stats, &display_model));
                }
            }
        }
        return 0;
    }

    if !std::io::stdout().is_terminal() {
        println!("serving in the background. stop with `localmodel stop`.");
        return 0;
    }

    strata_dashboard(&display_model, &info, &mut poller, &logpath, &url)
}

/// The same dashboard as llama.cpp, fed from `/metrics` instead of stderr.
fn strata_dashboard(
    model: &Model,
    info: &Info,
    poller: &mut stats::MetricsPoller,
    logpath: &Path,
    url: &str,
) -> i32 {
    let mut out = std::io::stdout();
    let _ = terminal::enable_raw_mode();
    let _ = execute!(out, EnterAlternateScreen, Hide);
    let _ = out.flush();

    let mut show_logs = true;
    let mut paused = false;
    let mut redraw = true;
    let mut frame = 0usize;
    let mut misses = 0u32;
    let hw = fit::Hardware::probe(&config::models_dir());
    let cfg = card_cfg(info.ctx, &info.ctk, &info.ctv, "", "");

    loop {
        // While paused, stop repainting so the terminal selection survives and
        // the card can be copied. One repaint is still done when the state
        // changes (p / o / l), so the PAUSED banner appears immediately.
        if !paused || redraw {
            let alive = match curl_get(&format!("{url}/metrics")) {
                Some(t) => poller.observe(&t),
                None => false,
            };
            misses = if alive { 0 } else { misses + 1 };

            let snapshot = poller.stats.clone();
            let gpus = server::gpu_infos();
            let ram = ram_text(0);
            let (cols, rows) = screen_size();
            let plan = mem::plan(model, None, &cfg, &hw);
            let screen = render_screen(
                model,
                info,
                &snapshot,
                model.nextn,
                &gpus,
                &ram,
                logpath,
                show_logs,
                frame,
                None,
                paused,
                cols,
                rows,
                &cfg,
                &hw,
                &plan,
                None,
            );
            print!("{}", paint_screen(&screen, cols));
            let _ = out.flush();
            redraw = false;

            // 30 seconds without /metrics: the server is gone.
            if misses > 30 {
                break;
            }
        }

        let mut quit = false;
        let timeout = if paused { Duration::from_millis(250) } else { Duration::from_millis(500) };
        if event::poll(timeout).unwrap_or(false) {
            loop {
                if let Ok(Event::Key(k)) = event::read() {
                    if k.modifiers.contains(KeyModifiers::CONTROL)
                        && matches!(k.code, KeyCode::Char('c') | KeyCode::Char('d'))
                    {
                        quit = true;
                    } else {
                        match k.code {
                            KeyCode::Char('q') | KeyCode::Esc => quit = true,
                            KeyCode::Char('l') => {
                                show_logs = !show_logs;
                                redraw = true;
                            }
                            KeyCode::Char('p') | KeyCode::Char('P') => {
                                paused = true;
                                redraw = true;
                            }
                            KeyCode::Char('o') | KeyCode::Char('O') => {
                                paused = false;
                                redraw = true;
                            }
                            _ => {}
                        }
                    }
                }
                if quit || !event::poll(Duration::from_millis(0)).unwrap_or(false) {
                    break;
                }
            }
        }
        if quit {
            break;
        }
        if !paused {
            frame += 1;
        }
    }

    let _ = execute!(out, LeaveAlternateScreen, Show);
    let _ = terminal::disable_raw_mode();
    let _ = out.flush();

    println!("stopping strata model...");
    match engine::stop() {
        Ok(o) if o.code == 0 => println!("stopped. log: {}", logpath.display()),
        Ok(o) => println!("strata stop exited with {}: {}", o.code, o.text.trim()),
        Err(e) => println!("could not stop strata: {e}"),
    }
    0
}

/// `localmodel strata <family> <size>` - launch a catalog pack.
fn cmd_strata(args: &[String]) -> i32 {
    let mut family: Option<String> = None;
    let mut size: Option<String> = None;
    let mut ctx: Option<u32> = None;
    let mut port: Option<u16> = None;
    let mut kv: Option<String> = None;
    let mut vision = "off".to_string();
    let mut access = "local".to_string();
    let mut tune = "skip".to_string();
    let mut parallel: u8 = 1;
    let mut restart = false;
    let mut dry_run = false;
    let mut repl = false;
    let mut data_dir: Option<PathBuf> = None;
    let mut extra: Option<String> = None;
    let mut profile: Option<String> = None;
    let mut positional: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let val = |i: &mut usize| -> Option<String> {
            *i += 1;
            args.get(*i).cloned()
        };
        match a {
            "--ctx" | "-c" => {
                if let Some(v) = val(&mut i).and_then(|v| parse_ctx(&v)) {
                    ctx = Some(v);
                }
            }
            "--port" | "-p" => port = val(&mut i).and_then(|v| v.parse().ok()),
            "--kv" => kv = val(&mut i),
            "--vision" => {
                if let Some(v) = val(&mut i) {
                    vision = v;
                }
            }
            "--access" => {
                if let Some(v) = val(&mut i) {
                    access = v;
                }
            }
            "--tune" => {
                if let Some(v) = val(&mut i) {
                    tune = v;
                }
            }
            "--parallel" => {
                if let Some(v) = val(&mut i) {
                    parallel = v.parse().unwrap_or(1).clamp(1, 8);
                }
            }
            "--family" => family = val(&mut i),
            "--model" | "--size" => size = val(&mut i),
            "--data-dir" => data_dir = val(&mut i).map(PathBuf::from),
            "--extra" => extra = val(&mut i),
            "--profile" => profile = val(&mut i),
            "--restart" => restart = true,
            "--dry-run" => dry_run = true,
            "--repl" => repl = true,
            "--plan" => {
                // A plan is read-only and prints what would download.
                let mut spec = engine::StrataSpec::new(ctx.unwrap_or(131072), port.unwrap_or(8080));
                spec.family = family.clone().or_else(|| positional.first().cloned());
                spec.size = size.clone().or_else(|| positional.get(1).cloned());
                spec.kv = kv.clone().unwrap_or_else(|| "int8".into());
                spec.vision = vision.clone();
                spec.access = access.clone();
                spec.tune = tune.clone();
                match engine::plan(&spec) {
                    Ok(o) => {
                        println!("{}", o.text.trim());
                        return if o.code == 0 { 0 } else { 1 };
                    }
                    Err(e) => {
                        eprintln!("{e}\n{}", engine::install_hint());
                        return 1;
                    }
                }
            }
            "--help" | "-h" => {
                help();
                return 0;
            }
            s if s.starts_with('-') => {}
            s => positional.push(s.to_string()),
        }
        i += 1;
    }

    if family.is_none() {
        family = positional.first().cloned();
    }
    if size.is_none() {
        size = positional.get(1).cloned();
    }
    let (Some(family), Some(size)) = (family, size) else {
        eprintln!("usage: localmodel strata <family> <size>   e.g. localmodel strata qwen36 ud-q6_k_xl");
        eprintln!("       localmodel doctor                  shows what fits this machine");
        return 2;
    };

    let dir = config::models_dir();
    // A hub pack keeps its launch settings under `family/size` (for example
    // [model."qwen36/ud_q4_k_xl"]), so the picker, `run family/size` and
    // `strata family size` all start the same way. CLI flags still win: the
    // saved extra flags come first, the command line's --extra after them.
    let saved = settings::load();
    let wanted = profile.clone().unwrap_or_else(|| format!("{family}/{size}"));
    let key = settings::find_key(&saved, &wanted)
        .unwrap_or(&wanted)
        .to_string();
    if profile.is_some() && !saved.model.contains_key(&key) {
        eprintln!("warning: no settings profile '{key}', using defaults for {family}/{size}");
    }
    let prof = saved.for_model(&key);
    let ctx = ctx.or(prof.ctx).unwrap_or(131072);
    let port = port.or(prof.port).unwrap_or_else(|| server::free_port(8080));

    let mut spec = engine::StrataSpec::new(ctx, port);
    spec.family = Some(family.clone());
    spec.size = Some(size.clone());
    spec.kv = kv.unwrap_or_else(|| "int8".into());
    spec.vision = vision;
    spec.access = access;
    spec.tune = tune;
    spec.parallel = parallel;
    spec.restart = restart;
    spec.data_dir = data_dir;
    let mut extras: Vec<String> = Vec::new();
    if let Some(e) = &prof.extra {
        extras.extend(e.split_whitespace().map(String::from));
    }
    if let Some(e) = &extra {
        extras.extend(e.split_whitespace().map(String::from));
    }
    spec.extra = extras;

    let alias = format!("{family}-{size}");
    let port = spec.port;
    run_strata(spec, None, &alias, ctx, port, repl, dry_run, &dir)
}

/// `localmodel doctor` - what this machine can actually run.
fn cmd_doctor(args: &[String]) -> i32 {
    let dir = config::models_dir();
    let hw = fit::Hardware::probe(&dir);

    let mut ctx = 131072u32;
    let mut i = 0;
    while i < args.len() {
        if matches!(args[i].as_str(), "--ctx" | "-c") {
            i += 1;
            if let Some(v) = args.get(i).and_then(|v| parse_ctx(v)) {
                ctx = v;
            }
        }
        i += 1;
    }

    println!("{BOLD}localmodel doctor{RESET} {DIM}estimates from the sizes Strata publishes{RESET}");
    println!();
    println!("{}", section("HARDWARE"));

    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|v| v.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into());
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
    println!("{}", kv("CPU", &format!("{cpu} ({threads} threads)")));

    if let Some(g) = server::gpu_info() {
        println!(
            "{}",
            kv(
                "GPU",
                &format!(
                    "{}  {:.1} GB VRAM ({:.1} GB in use)",
                    g.name,
                    g.mem_total / 1024.0,
                    g.mem_used / 1024.0
                )
            )
        );
    } else {
        println!("{}", kv("GPU", "none found"));
    }

    println!(
        "{}",
        kv(
            "RAM",
            &format!(
                "{:.1} GB total, {:.1} GB available",
                hw.ram_total_gb, hw.ram_avail_gb
            )
        )
    );
    println!(
        "{}",
        kv(
            "Disk",
            &format!("{:.1} GB free at {}", hw.disk_free_gb, dir.display())
        )
    );

    let llama = config::server_bin();
    let llama_state = if llama.exists() {
        format!("{GREEN}ok{RESET}      {}", llama.display())
    } else {
        format!("{RED}missing{RESET} {}", llama.display())
    };
    println!("{}", kv("llama.cpp", &llama_state));
    let strata_state = match engine::strata_bin() {
        Some(p) => format!("{GREEN}ok{RESET}      {}", p.display()),
        None => format!("{YELLOW}missing{RESET} run the one-line installer (see `localmodel strata --help`)"),
    };
    println!("{}", kv("strata", &strata_state));
    println!();

    println!(
        "{}",
        section(&format!("TARGET MODELS at {}K context", ctx / 1024))
    );
    println!(
        "  {DIM}{:<8} {:<12} {:>7} {:>7} {:<9} {}{RESET}",
        "family", "size", "disk GB", "RAM GB", "verdict", "why"
    );
    println!();
    let mut best: Option<(&fit::CatalogEntry, fit::Estimate)> = None;
    let mut disk_short: Vec<(&fit::CatalogEntry, fit::Estimate)> = Vec::new();
    let mut tight: Vec<(&fit::CatalogEntry, fit::Estimate)> = Vec::new();
    for e in fit::CATALOG {
        let est = fit::estimate(e, &hw, ctx);
        let vlabel = format!("{:<9}", est.verdict.label());
        println!(
            "  {:<8} {:<12} {:>7.1} {:>7.1} {}{}{RESET} {DIM}{}{RESET}",
            e.family,
            e.size,
            est.need_disk_gb,
            est.need_ram_gb,
            est.verdict.color(),
            vlabel,
            est.why
        );
        if est.verdict == fit::Verdict::Fits {
            let better = best
                .as_ref()
                .map(|(be, b)| {
                    // Prefer the user's target family, then the larger (better) file.
                    let rank = |fam: &str, ram: f64| (fam == "qwen36", ram);
                    rank(e.family, est.need_ram_gb) > rank(be.family, b.need_ram_gb)
                })
                .unwrap_or(true);
            if better {
                best = Some((e, est));
            }
        } else if est.mem_verdict == fit::Verdict::No {
            // Does not fit memory at all: nothing to say about disk.
        } else if hw.disk_free_gb < est.need_disk_gb {
            disk_short.push((e, est));
        } else {
            tight.push((e, est));
        }
    }
    println!();

    println!("{}", section("RECOMMENDATION"));
    match &best {
        Some((e, est)) => {
            println!(
                "  {GREEN}{} {}{RESET} - {:.1} GB RAM, {:.1} GB VRAM, {}",
                e.family,
                e.size,
                est.need_ram_gb,
                est.need_vram_gb,
                e.note
            );
            println!();
            println!("  {BOLD}{}{RESET}", fit::command(e, ctx));
            println!();
            println!("  {DIM}or, once the files are here: localmodel run <file>{RESET}");
        }
        None => {
            println!(
                "  {RED}Nothing in the catalog fits right now at {}K.{RESET}",
                ctx / 1024
            );
        }
    }

    if !tight.is_empty() {
        println!();
        println!("  {YELLOW}Tight: they start, but expect paging or a slow first token:{RESET}");
        for (e, est) in &tight {
            println!(
                "    {:<8} {:<12} {:.1} GB RAM, {:.1} GB VRAM  {DIM}{}{RESET}",
                e.family, e.size, est.need_ram_gb, est.need_vram_gb, est.why
            );
        }
    }

    if !disk_short.is_empty() {
        println!();
        println!("  {YELLOW}Memory is enough for these, the disk is not:{RESET}");
        for (e, est) in &disk_short {
            println!(
                "    {:<8} {:<12} needs {:.0} GB free, {:.1} GB there  ({})",
                e.family,
                e.size,
                est.need_disk_gb,
                hw.disk_free_gb,
                if est.mem_verdict == fit::Verdict::Tight { "tight" } else { "ok" }
            );
        }
        println!("  {DIM}free space in {} or point LOCALMODEL_DIR at another disk{RESET}", dir.display());
    }

    // The question that started this: does Flash-Next run here?
    let flash: Vec<&fit::CatalogEntry> =
        fit::CATALOG.iter().filter(|e| e.family == "qwen" || e.family == "gyro").collect();
    if !flash.is_empty() {
        let any_mem = flash
            .iter()
            .any(|e| fit::estimate(e, &hw, ctx).mem_verdict != fit::Verdict::No);
        println!();
        if any_mem {
            println!(
                "  {DIM}Qwen3.8-Flash-Next: some pack fits memory at {}K - see the disk column.{RESET}",
                ctx / 1024
            );
        } else {
            println!(
                "  {DIM}Qwen3.8-Flash-Next: no pack fits memory at {}K. The smallest full pack{RESET}",
                ctx / 1024
            );
            println!(
                "  {DIM}wants ~44 GB RAM plus ~76 GB disk; the Coder pack wants ~32 GB RAM{RESET}"
            );
            println!("  {DIM}and a 24 GB card.{RESET}");
        }
    }
    0
}

fn curl_get(url: &str) -> Option<String> {
    let out = Command::new("curl")
        .args(["-s", "--max-time", "10"])
        .arg(url)
        .stderr(Stdio::null())
        .output()
        .ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).to_string())
}

fn curl_post(url: &str, body: &str) -> Option<String> {
    let out = Command::new("curl")
        .args(["-s", "--max-time", "1800", "-X", "POST", "-H", "Content-Type: application/json", "--data-binary"])
        .arg(body)
        .arg(url)
        .stderr(Stdio::null())
        .output()
        .ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).to_string())
}

fn props_ctx(port: u16) -> Option<u64> {
    let s = curl_get(&format!("http://127.0.0.1:{port}/props"))?;
    let v: serde_json::Value = serde_json::from_str(&s).ok()?;
    v.get("default_generation_settings")?.get("n_ctx")?.as_u64()
}

fn chat(port: u16, alias: &str, text: &str, images: Option<&[String]>, template_kwargs: Option<&Value>) -> Option<String> {
    let content = match images {
        Some(imgs) if !imgs.is_empty() => {
            let mut parts: Vec<Value> = vec![json!({ "type": "text", "text": text })];
            for img in imgs {
                parts.push(json!({
                    "type": "image_url",
                    "image_url": { "url": img },
                }));
            }
            Value::Array(parts)
        }
        _ => Value::String(text.to_string()),
    };
    let mut req = serde_json::json!({
        "model": alias,
        "messages": [{ "role": "user", "content": content }],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 1024,
    });
    if let Some(kw) = template_kwargs {
        req.as_object_mut()?
            .insert("chat_template_kwargs".to_string(), kw.clone());
    }
    let resp = curl_post(&format!("http://127.0.0.1:{port}/v1/chat/completions"), &req.to_string())?;
    let v: serde_json::Value = serde_json::from_str(&resp).ok()?;
    v.get("choices")?.get(0)?.get("message")?.get("content")?.as_str().map(|s| s.to_string())
}

fn gpu_str(_pid: u32) -> String {
    let (u, t) = server::gpu_mem().unwrap_or((0.0, 0.0));
    let util = server::gpu_util().unwrap_or(0.0);
    format!("{:.1}/{:.1}G ({:.0}%)", u / 1024.0, t / 1024.0, util)
}

/// RAM as the dashboard shows it: system "used" (total - available, page
/// cache excluded) plus the server's resident/anon split.
struct RamText {
    system: String,
    server: String,
    short: String,
}

fn gb(kb: u64) -> String {
    if kb >= 1024 * 1024 {
        format!("{:.1} GB", kb as f64 / 1024.0 / 1024.0)
    } else {
        format!("{:.0} MB", kb as f64 / 1024.0)
    }
}

fn ram_text(pid: u32) -> RamText {
    let system = match server::sys_mem() {
        Some(m) => {
            let used = m.total_kb.saturating_sub(m.avail_kb);
            let pct = if m.total_kb > 0 {
                used as f64 / m.total_kb as f64 * 100.0
            } else {
                0.0
            };
            let color = if pct >= 90.0 {
                RED
            } else if pct >= 75.0 {
                YELLOW
            } else {
                GREEN
            };
            format!(
                "{color}{}{RESET}{DIM}/{}{RESET} {DIM}({:.0}%){RESET}",
                gb(used),
                gb(m.total_kb),
                pct
            )
        }
        None => format!("{DIM}n/a{RESET}"),
    };
    let (server_line, short) = match server::proc_mem(pid) {
        Some(p) => (
            format!(
                "{} {DIM}rss{RESET} · {}{DIM} anon{RESET}",
                gb(p.rss_kb),
                gb(p.anon_kb)
            ),
            format!("{} (anon {})", gb(p.rss_kb), gb(p.anon_kb)),
        ),
        None => (format!("{DIM}n/a{RESET}"), "--".to_string()),
    };
    RamText {
        system,
        server: server_line,
        short,
    }
}

/// Static model card shown once when the server starts.
/// A Cfg with the built-in defaults, for the static cards (`info` and the
/// line-mode startup card) where no saved profile is resolved.
fn card_cfg(ctx: u32, ctk: &str, ctv: &str, draft_ctk: &str, draft_ctv: &str) -> Cfg {
    Cfg {
        ctx,
        port: None,
        ngl: None,
        ncmoe: None,
        ctk: ctk.to_string(),
        ctv: ctv.to_string(),
        draft_ctk: draft_ctk.to_string(),
        draft_ctv: draft_ctv.to_string(),
        threads: None,
        reasoning: None,
        effort: None,
        draft_tokens: None,
        extra: None,
        tools: None,
        draft_gpu: false,
        kv_ram: false,
        split_mode: "layer".into(),
        tensor_split: None,
        log_verbosity: 4,
        vision: false,
        mmproj: None,
        alias: None,
        flash_attn: true,
        temp: 1.0,
        top_p: 0.95,
        host: "127.0.0.1".into(),
        ..Cfg::default()
    }
}

/// MEMORY + PLACEMENT blocks, shared by every card (startup, `info`, the
/// dashboard and the hub's estimate area). `live` is llama.cpp's own log
/// breakdown once the model has loaded.
fn memory_lines(
    plan: &mem::Plan,
    live: Option<&telemetry::LoadInfo>,
    hw: &fit::Hardware,
    compact: bool,
) -> Vec<String> {
    let mut b = vec![section("MEMORY")];
    if compact {
        b.push(kv(
            "VRAM",
            &format!(
                "{} {:.2} / {:.2} GiB",
                vram_bar(plan.vram_gb(), hw.vram_room_gb().max(0.01), 16),
                plan.vram_gb(),
                hw.vram_room_gb()
            ),
        ));
        let rows = match live {
            Some(l) => l.rows(),
            None => plan.place_rows(),
        };
        for (name, val) in rows {
            b.push(kv(&name, &val));
        }
        if live.map(|l| l.mmap_cpu_experts).unwrap_or(false) {
            b.push(kv("mmap", &format!("{YELLOW}CPU experts read from SSD{RESET}")));
        }
        return b;
    }
    b.push(kv(
        "Weights",
        &format!(
            "{:.2} GiB  {DIM}incl. output tensor {:.2}{RESET}",
            plan.total_weights_gb, plan.output_gb
        ),
    ));
    b.push(kv(
        "KV main",
        &format!("{:.2} GiB  {DIM}{}{RESET}", plan.kv_main_gb, plan.kv_home.label()),
    ));
    if plan.kv_draft_gb > 0.01 {
        b.push(kv(
            "KV draft/MTP",
            &format!("{:.2} GiB  {DIM}{}{RESET}", plan.kv_draft_gb, plan.draft_types),
        ));
    }
    b.push(kv("Compute", &format!("{:.2} GiB {DIM}buffers + CUDA{RESET}", plan.overhead_gb)));
    let room = hw.vram_room_gb();
    b.push(kv(
        "VRAM",
        &format!(
            "{} {:.2} / {:.2} GiB {DIM}usable{RESET}",
            vram_bar(plan.vram_gb(), room.max(0.01), 20),
            plan.vram_gb(),
            room
        ),
    ));
    b.push(kv(
        "Offload",
        &format!(
            "{}/{} layers  {DIM}split {}{RESET}",
            plan.ngl, plan.block_count, plan.split_mode
        ),
    ));
    if plan.ncmoe > 0 {
        b.push(kv(
            "Experts in RAM",
            &format!("first {} blocks  {DIM}--n-cpu-moe{RESET}", plan.ncmoe),
        ));
    }
    if plan.host.total_gb() > 0.01 {
        b.push(kv(
            "Host RAM",
            &format!(
                "{:.2} GiB  {DIM}input emb. {:.2}{RESET}",
                plan.host.total_gb(),
                plan.input_gb
            ),
        ));
    }
    b.push(kv(
        "Placement",
        &format!(
            "{DIM}{}{RESET}",
            if live.is_some() {
                "actual · llama.cpp startup log"
            } else if plan.approx {
                "planned · block sizes approximated"
            } else {
                "planned · from GGUF tensor table"
            }
        ),
    ));
    if let Some(l) = live {
        for (name, val) in l.rows() {
            b.push(kv(&name, &val));
        }
        if let Some((a, t)) = l.offloaded {
            b.push(kv("Offloaded", &format!("{a}/{t} layers to GPU")));
        }
        if l.mmap_cpu_experts {
            b.push(kv(
                "Warning",
                &format!("{YELLOW}CPU experts are mmap'd: every token re-reads them from SSD{RESET}"),
            ));
            b.push(kv(
                "Fix",
                &format!("{YELLOW}--load-mode none (RAM) or --moe-cache-mib N (GPU cache){RESET}"),
            ));
        }
    } else {
        for (name, val) in plan.place_rows() {
            b.push(kv(&name, &val));
        }
    }
    if let Some(note) = &plan.note {
        b.push(kv("", &format!("{YELLOW}{note}{RESET}")));
    }
    b
}

fn gpu_lines() -> Vec<String> {
    let gpus = server::gpu_infos();
    let mut b = vec![section("GPU")];
    if gpus.is_empty() {
        b.push(kv("Device", &format!("{DIM}n/a{RESET}")));
        return b;
    }
    for g in &gpus {
        let pct = if g.mem_total > 0.0 { g.mem_used / g.mem_total * 100.0 } else { 0.0 };
        b.push(kv(
            &format!("GPU{}", g.index),
            &format!("{}  {DIM}{:.0}% util · {:.0}°C · {:.0}/{:.0} W{RESET}", g.name, g.util, g.temp, g.power_draw, g.power_limit),
        ));
        b.push(kv(
            "",
            &format!(
                "{} {:.1}/{:.1} GiB {DIM}({:.1}%){RESET}",
                vram_bar(g.mem_used, g.mem_total, 20),
                g.mem_used / 1024.0,
                g.mem_total / 1024.0,
                pct
            ),
        ));
    }
    if gpus.len() > 1 {
        let link = match server::has_nvlink() {
            Some(true) => format!("{GREEN}NVLink{RESET}"),
            Some(false) => format!("{YELLOW}PCIe, no NVLink{RESET}"),
            None => format!("{DIM}unknown{RESET}"),
        };
        b.push(kv("Interconnect", &link));
    }
    b.push(kv("Backend", "CUDA (cuBLAS)"));
    b
}

/// Plan for a static card: the built-in MTP head is already in `m`, an
/// external draft file is parsed here so its weights and KV land in the right
/// column.
fn card_plan(
    m: &Model,
    cfg: &Cfg,
    hw: &fit::Hardware,
    draft: Option<&Path>,
) -> mem::Plan {
    if m.nextn {
        return mem::plan(m, None, cfg, hw);
    }
    let parsed = draft.and_then(|p| {
        let meta = gguf::parse(p);
        let bytes = std::fs::metadata(p).ok()?.len();
        Some((meta, bytes))
    });
    let d = parsed
        .as_ref()
        .map(|(meta, bytes)| mem::Draft { meta, weights: *bytes });
    mem::plan(m, d, cfg, hw)
}

/// Static colored card for the startup line and `localmodel info`.
/// Extra MEMORY rows for placement knobs that are set. Kept separate from
/// `memory_lines` because that function works from the plan, while these are
/// engine switches that change *how* the plan is realised on disk/RAM.
fn placement_knob_lines(cfg: &Cfg) -> Vec<String> {
    let mut v = Vec::new();
    if let Some(mode) = &cfg.load_mode {
        let note = match mode.as_str() {
            "none" => "anonymous RAM, no SSD paging",
            "mmap" | "mmap+mlock" => "experts read from SSD on demand",
            "mlock" => "pinned in RAM, never swapped",
            "dio" => "direct I/O, bypasses page cache",
            _ => "engine chooses per device",
        };
        v.push(kv("Load mode", &format!("{mode}  {DIM}{note}{RESET}")));
    }
    if let Some(n) = cfg.moe_cache_mib {
        v.push(kv("MoE cache", &format!("{n} MiB on GPU  {DIM}hot experts stay on the card{RESET}")));
    }
    if cfg.cpu_moe {
        v.push(kv("CPU MoE", &format!("all experts  {DIM}--cpu-moe{RESET}")));
    }
    if let Some(n) = cfg.n_cpu_ffn {
        v.push(kv("CPU FFN", &format!("first {n} blocks  {DIM}--n-cpu-ffn{RESET}")));
    }
    if let Some(ot) = &cfg.override_tensor {
        v.push(kv("Overrides", &format!("{DIM}{ot}{RESET}")));
    }
    v
}

fn model_card(
    m: &Model,
    alias: &str,
    port: u16,
    cfg: &Cfg,
    logpath: &Path,
    engine: &str,
    draft: Option<&Path>,
) -> String {
    let hw = fit::Hardware::probe(&config::models_dir());
    let plan = card_plan(m, cfg, &hw, draft);
    let kind = if m.moe { "MoE" } else { "Dense" };
    let mut body: Vec<String> = vec![String::new(), section("MODEL")];
    body.push(kv("Engine", engine));
    body.push(kv("Architecture", &format!("{kind} · {}", if m.meta.arch.is_empty() { m.name.as_str() } else { m.meta.arch.as_str() })));
    body.push(kv("Size", &format!("{}  {DIM}quant{RESET} {MAGENTA}{}{RESET}", models::human(m.size), m.quant)));
    body.push(kv("Context", &format!("{}  {DIM}({}K){RESET}", cfg.ctx, cfg.ctx / 1024)));
    body.push(kv("Layers", &m.layers.to_string()));
    body.push(kv("KV cache", &format!("{}/{}{}", cfg.ctk, cfg.ctv, if cfg.kv_ram { " · RAM" } else { " · VRAM" })));
    if m.moe && m.experts_total > 0 {
        body.push(kv("Experts", &format!("{} total, {} active/token", m.experts_total, m.experts_active)));
    }
    body.push(String::new());
    body.extend(memory_lines(&plan, None, &hw, false));
    body.extend(placement_knob_lines(cfg));
    body.push(String::new());
    body.extend(gpu_lines());
    body.push(String::new());
    body.push(section("SERVER"));
    body.push(kv("URL", &format!("http://127.0.0.1:{port}/v1")));
    body.push(kv("Log", &logpath.display().to_string()));
    body.push(String::new());
    let title = format!("{CYAN}{BOLD}{alias}{RESET} {DIM}:{port}{RESET}");
    let footer = format!("{DIM}localmodel hub  ·  settings  ·  params{RESET}");
    boxed(&title, &body, &footer, screen_size().0.max(80))
}

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const CYAN: &str = "\x1b[36m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const MAGENTA: &str = "\x1b[35m";
const BLUE: &str = "\x1b[34m";

/// Visible width of a string, ignoring ANSI SGR escape sequences.
fn vlen(s: &str) -> usize {
    let mut n = 0usize;
    let mut esc = false;
    for c in s.chars() {
        if esc {
            if c == 'm' {
                esc = false;
            }
            continue;
        }
        if c == '\x1b' {
            esc = true;
            continue;
        }
        n += 1;
    }
    n
}

fn section(name: &str) -> String {
    format!("  {CYAN}{BOLD}{name}{RESET}")
}

/// Truncate to `max` visible cells, keeping ANSI escapes intact.
fn clip_visible(s: &str, max: usize) -> String {
    let mut out = String::new();
    let mut n = 0usize;
    let mut esc = false;
    for c in s.chars() {
        if esc {
            out.push(c);
            if c == 'm' {
                esc = false;
            }
            continue;
        }
        if c == '\x1b' {
            esc = true;
            out.push(c);
            continue;
        }
        if n == max {
            break;
        }
        out.push(c);
        n += 1;
    }
    if n == max {
        out.push_str(RESET);
    }
    out
}

/// Wrap a rendered screen for the terminal.
///
/// `\x1b[2K` before every line erases the whole line, so a shorter line never
/// leaves the tail of the previous frame on screen. The old renderer only
/// emitted `\x1b[J` once at the end, which cleared below the cursor but not
/// the right-hand side of earlier lines - that is where the stray characters
/// came from. Lines are clipped to `cols - 1` so a full-width line cannot
/// trigger the terminal's auto-wrap and shift every following row.
pub(crate) fn paint_screen(screen: &str, cols: usize) -> String {
    let mut out = String::with_capacity(screen.len() + 32);
    out.push_str("\x1b[H");
    for line in screen.lines() {
        out.push_str("\x1b[2K");
        out.push_str(&clip_visible(line, cols.saturating_sub(1)));
        out.push_str("\r\n");
    }
    out.push_str("\x1b[J");
    out
}

/// Terminal size, or a sane fallback when stdout is not a terminal.
fn screen_size() -> (usize, usize) {    terminal::size()
        .map(|(c, r)| (c as usize, r as usize))
        .unwrap_or((100, 40))
}

fn kv(label: &str, value: &str) -> String {
    format!("  {DIM}{label:<14}{RESET} {value}")
}

fn vram_bar(used: f64, total: f64, width: usize) -> String {
    let frac = if total > 0.0 { (used / total).clamp(0.0, 1.0) } else { 0.0 };
    let filled = (frac * width as f64).round() as usize;
    let color = if frac >= 0.95 { RED } else if frac >= 0.80 { YELLOW } else { GREEN };
    format!("{color}{}{RESET}{DIM}{}{RESET}", "█".repeat(filled), "░".repeat(width - filled))
}

const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Full dashboard screen: model card + live monitoring, redrawn in place.
/// `rows` is the terminal height; whole sections are dropped when the window
/// is too short so nothing ever scrolls or gets cut mid-row.
fn render_screen(
    m: &Model,
    info: &Info,
    s: &Stats,
    has_draft: bool,
    gpus: &[server::GpuInfo],
    ram: &RamText,
    logpath: &Path,
    show_logs: bool,
    frame: usize,
    hint: Option<&str>,
    paused: bool,
    cols: usize,
    rows: usize,
    cfg: &Cfg,
    hw: &fit::Hardware,
    plan: &mem::Plan,
    live: Option<&telemetry::LoadInfo>,
) -> String {
    let (dot, state) = if s.alive { (GREEN, "UP") } else { (YELLOW, "loading") };
    let spin = if s.alive { " ".to_string() } else { SPINNER[frame % SPINNER.len()].to_string() };
    let ctx = if s.ctx > 0 { s.ctx } else { info.ctx as u64 };
    let dec = s.last_dec;
    let avg = s.avg_dec();
    let pre = s.last_pre;
    let tpt = dec.filter(|d| *d > 0.0).map(|d| format!("{:.1} ms", 1000.0 / d)).unwrap_or_else(|| "--".into());

    let arch = kv("Architecture", if m.moe { "MoE" } else { "Dense" });
    let size = kv(
        "Size",
        &format!("{}  {DIM}quant{RESET} {MAGENTA}{}{RESET}", models::human(m.size), m.quant),
    );
    let ctx_row = kv("Context", &format!("{ctx}  {DIM}({}K){RESET}", ctx / 1024));
    let layers = kv("Layers", &m.layers.to_string());
    let kv_row = kv(
        "KV cache",
        &format!(
            "{}/{}  {DIM}in {}{RESET}",
            info.ctk,
            info.ctv,
            if cfg.kv_ram { "RAM" } else { "VRAM" }
        ),
    );

    let model_full = {
        let mut g = vec![section("MODEL"), arch.clone(), size.clone(), ctx_row.clone(), layers, kv_row.clone()];
        if m.moe && m.experts_total > 0 {
            g.push(kv("Experts", &format!("{} total, {} active/token", m.experts_total, m.experts_active)));
        }
        g.push(String::new());
        g
    };
    let model_short = vec![
        section("MODEL"),
        arch.clone(),
        size.clone(),
        ctx_row.clone(),
        kv_row.clone(),
        String::new(),
    ];

    let memory_full = {
        let mut g = memory_lines(plan, live, hw, false);
        g.push(String::new());
        g
    };
    // Compact: totals plus the per-device placement, so the answer to
    // "where did the layers go" is still on screen.
    let mut memory_short = memory_lines(plan, live, hw, true);
    memory_short.push(String::new());

    let mut gpu_full = vec![section("GPU")];
    let mut gpu_short = vec![section("GPU")];
    if gpus.is_empty() {
        gpu_full.push(kv("Device", &format!("{DIM}n/a{RESET}")));
        gpu_short.push(kv("Device", &format!("{DIM}n/a{RESET}")));
    } else {
        for g in gpus {
            let pct = if g.mem_total > 0.0 { g.mem_used / g.mem_total * 100.0 } else { 0.0 };
            gpu_full.push(kv(
                &format!("GPU{}", g.index),
                &format!("{}  {DIM}{:.0}% util · {:.0}°C · {:.0}/{:.0} W{RESET}", g.name, g.util, g.temp, g.power_draw, g.power_limit),
            ));
            let bar = format!(
                "{} {:.1}/{:.1} GiB {DIM}({:.1}%){RESET}",
                vram_bar(g.mem_used, g.mem_total, 24),
                g.mem_used / 1024.0,
                g.mem_total / 1024.0,
                pct
            );
            gpu_full.push(kv("", &bar));
            gpu_short.push(kv(&format!("GPU{}", g.index), &bar));
        }
        if gpus.len() > 1 {
            let link = match server::has_nvlink() {
                Some(true) => format!("{GREEN}NVLink{RESET}"),
                Some(false) => format!("{YELLOW}PCIe, no NVLink{RESET}"),
                None => format!("{DIM}unknown{RESET}"),
            };
            gpu_full.push(kv("Interconnect", &link));
        }
        gpu_full.push(kv("Backend", "CUDA (cuBLAS)"));
    }
    gpu_full.push(String::new());
    gpu_short.push(String::new());

    let speed = kv(
        "Speed",
        &format!(
            "{GREEN}dec {}{RESET}{DIM}/s{RESET} {DIM}(avg {}){RESET}   {BLUE}pre {}{RESET}{DIM}/s{RESET}",
            fmt_opt(dec),
            fmt_opt(avg),
            fmt_opt(pre)
        ),
    );
    let ram_row = kv("RAM", &ram.system);
    let srv_ram = kv("Server RAM", &ram.server);
    let draft_row = if let Some(acc) = s.draft_accept() {
        let mut txt = format!("{GREEN}{:.0}%{RESET}", acc * 100.0);
        if let Some(mean) = s.draft_mean() {
            txt.push_str(&format!("  {DIM}mean {mean:.2} tok/step{RESET}"));
        }
        if !s.last_draft_per_pos.is_empty() {
            let pos = s
                .last_draft_per_pos
                .iter()
                .map(|x| format!("{:.0}", x * 100.0))
                .collect::<Vec<_>>()
                .join("/");
            txt.push_str(&format!("  {DIM}pos {pos}%{RESET}"));
        }
        Some(kv("Draft accept", &txt))
    } else if has_draft {
        // The model has a draft/MTP head, but nothing has been generated yet:
        // keep the row visible so the number is easy to find once a request
        // completes.
        Some(kv(
            "Draft accept",
            &format!("{DIM}--  waiting for the first request{RESET}"),
        ))
    } else {
        None
    };
    let mut live_full = vec![section("LIVE"), speed.clone(), kv("Time/token", &tpt)];
    if let Some(row) = &draft_row {
        live_full.push(row.clone());
    }
    live_full.extend([
        kv("Requests", &s.reqs.to_string()),
        ram_row.clone(),
        srv_ram.clone(),
        String::new(),
    ]);
    let mut live_short = vec![section("LIVE"), speed.clone()];
    if let Some(row) = &draft_row {
        live_short.push(row.clone());
    }
    live_short.extend([ram_row.clone(), srv_ram.clone(), String::new()]);

    let server_group = vec![
        section("SERVER"),
        kv("URL", &format!("http://127.0.0.1:{}/v1", info.port)),
        kv("Status", &format!("{dot}{state}{RESET} {spin}")),
        String::new(),
    ];

    let mut log_group: Vec<String> = Vec::new();
    if show_logs {
        log_group.push(section("LOG"));
        let logs = tail_lines(logpath, 4);
        if logs.is_empty() {
            log_group.push(format!("  {DIM}waiting for output...{RESET}"));
        } else {
            for l in &logs {
                // Use the full card width: the box is capped at `cols - 5`, so
                // this keeps two spaces of prefix inside it.
                log_group.push(log_line_w(l, cols.saturating_sub(7).max(20)));
            }
        }
        log_group.push(String::new());
    }

    // Progressive fit: full card, then compact sections, then fewer sections.
    // Never let the terminal scroll or cut a row in half.
    let budget = rows.saturating_sub(4); // two borders + two footer lines
    let full = vec![model_full, memory_full, gpu_full, live_full, server_group.clone()];
    let compact = vec![
        model_short,
        memory_short.clone(),
        gpu_short,
        live_short.clone(),
        server_group.clone(),
    ];
    let minimal = vec![memory_short, live_short.clone(), server_group.clone()];
    let with_log = |g: &Vec<Vec<String>>| {
        let mut g = g.clone();
        if show_logs {
            g.push(log_group.clone());
        }
        g
    };
    let layouts: Vec<Vec<Vec<String>>> = vec![
        with_log(&full),
        with_log(&compact),
        full,
        compact,
        minimal,
        vec![live_short, server_group.clone()],
        vec![server_group],
    ];

    let fits = |l: &Vec<Vec<String>>| 1 + l.iter().map(|g| g.len()).sum::<usize>() <= budget;
    let mut body = match layouts.into_iter().find(|l| fits(l)) {
        Some(l) => {
            let mut b = vec![String::new()];
            for g in l {
                b.extend(g);
            }
            b
        }
        None => vec![String::new()],
    };
    if body.len() > budget {
        body.truncate(budget.saturating_sub(1));
        body.push(format!("  {DIM}window too small - resize{RESET}"));
    }
    if let Some(h) = hint {
        if body.len() >= budget {
            body.pop();
        }
        body.push(format!("  {GREEN}{h}{RESET}"));
    }

    let title = format!("{dot}●{RESET} {} :{}", info.alias, info.port);
    let paused_row = if paused {
        format!("\n  {YELLOW}PAUSED - the screen is frozen, select and copy freely; o resumes{RESET}")
    } else {
        String::new()
    };
    let footer = format!(
        "{DIM}? help · s settings · l log · r restart · p pause · o resume · q quit{RESET}\n  {DIM}API http://127.0.0.1:{}/v1  ·  log {}{RESET}{paused_row}",
        info.port,
        logpath.display()
    );
    boxed(&title, &body, &footer, cols)
}

/// Shared frame renderer: title bar + rows + footer. `cols` is the terminal
/// width, so the box never wraps (which is what leaves smears behind).
fn boxed(title: &str, body: &[String], footer: &str, cols: usize) -> String {
    // The outer frame is `w + 4` columns ("│ " + w + " │"). Leave five cells
    // for it plus one spare: a line that fills the terminal exactly makes the
    // terminal auto-wrap, which shifts every following row. Capping at
    // `cols - 2` used to make the card wider than the terminal.
    let cap = cols.saturating_sub(5).max(12);
    let mut w = body
        .iter()
        .map(|l| vlen(l))
        .max()
        .unwrap_or(0)
        .max(vlen(title) + 2)
        .max(40)
        .min(cap);
    if vlen(title) > w.saturating_sub(2) {
        w = w.max(20).min(cap);
        let t = clip_visible(title, w.saturating_sub(2));
        return boxed_body(&t, body, footer, w, cols);
    }
    boxed_body(title, body, footer, w, cols)
}

fn boxed_body(title: &str, body: &[String], footer: &str, w: usize, cols: usize) -> String {
    let tvis = vlen(title);
    // Every row is exactly this wide: "│ " + w + " │" = w + 4.
    let outer = w + 4;
    // "┌─ " + title + " " + dashes + "┐" must add up to `outer` too.
    let dashes = outer.saturating_sub(3 + tvis + 1 + 1);
    let mut out = String::from("┌─ ");
    out.push_str(title);
    out.push(' ');
    out.push_str(&"─".repeat(dashes));
    out.push_str("┐\n");
    for l in body {
        let l = clip_visible(l, w);
        let pad = w.saturating_sub(vlen(&l));
        out.push_str("│ ");
        out.push_str(&l);
        out.push_str(&" ".repeat(pad + 1));
        out.push_str("│\n");
    }
    out.push('└');
    out.push_str(&"─".repeat(outer.saturating_sub(2)));
    out.push('┘');
    if !footer.is_empty() {
        out.push('\n');
        for line in footer.split('\n') {
            out.push_str(&clip_visible(line, cols.saturating_sub(1)));
            out.push('\n');
        }
        out.pop();
    }
    out
}

/// Static colored card for `localmodel info`: no live/status rows, because
/// nothing is running.
fn render_info(
    m: &Model,
    alias: &str,
    port: u16,
    cfg: &Cfg,
    logpath: &Path,
    engine: &str,
    launch: &str,
    draft: Option<&Path>,
) -> String {
    let hw = fit::Hardware::probe(&config::models_dir());
    let plan = card_plan(m, cfg, &hw, draft);
    let mut body: Vec<String> = vec![String::new(), section("MODEL")];
    body.push(kv("Engine", engine));
    body.push(kv("Architecture", if m.moe { "MoE" } else { "Dense" }));
    body.push(kv(
        "Size",
        &format!("{}  {DIM}quant{RESET} {MAGENTA}{}{RESET}", models::human(m.size), m.quant),
    ));
    body.push(kv("Context", &format!("{}  {DIM}({}K){RESET}", cfg.ctx, cfg.ctx / 1024)));
    body.push(kv("Layers", &m.layers.to_string()));
    body.push(kv("KV cache", &format!("{}/{}{}", cfg.ctk, cfg.ctv, if cfg.kv_ram { " · RAM" } else { " · VRAM" })));
    body.push(kv("File", &m.name));
    if m.moe && m.experts_total > 0 {
        body.push(kv("Experts", &format!("{} total, {} active/token", m.experts_total, m.experts_active)));
    }
    body.push(String::new());
    body.extend(memory_lines(&plan, None, &hw, false));
    body.extend(placement_knob_lines(cfg));
    body.push(String::new());
    body.extend(gpu_lines());
    body.push(String::new());
    body.push(section("SERVER"));
    body.push(kv("URL", &format!("http://127.0.0.1:{port}/v1")));
    body.push(kv("Log", &logpath.display().to_string()));
    body.push(String::new());
    let title = format!("{CYAN}{BOLD}{alias}{RESET} {DIM}:{port}{RESET}");
    let footer = format!(
        "{DIM}{launch}  ·  localmodel hub{RESET}"
    );
    boxed(&title, &body, &footer, screen_size().0.max(80))
}

/// Dynamic live stats line, refreshed every second.
fn stats_line(info: &Info, s: &Stats, m: &Model) -> String {
    let dot = if s.alive { "\x1b[32m\u{25cf}\x1b[0m" } else { "\x1b[33m\u{25cb}\x1b[0m" };
    let state = if s.alive { "UP" } else { "loading" };
    let ctx = if s.ctx > 0 { s.ctx } else { info.ctx as u64 };
    let dec = fmt_opt(s.last_dec);
    let avg = fmt_opt(s.avg_dec());
    let pre = fmt_opt(s.last_pre);
    let layers = format!("{}/{}", m.layers, m.layers);
    let mut line = format!(
        "{} ● {} :{} {}  ctx {}  kv {}/{}  layers {}  dec {}/s (avg {})  pre {}/s  vram {}  ram {}  req {}  gpu {}",
        dot, info.alias, info.port, state, ctx, info.ctk, info.ctv, layers, dec, avg, pre,
        gpu_str(info.pid), ram_text(info.pid).short, s.reqs, server::gpu_util().unwrap_or(0.0)
    );
    if m.moe && m.experts_total > 0 {
        line.push_str(&format!("  experts {}/{} active", m.experts_active, m.experts_total));
    }
    if let Some(acc) = s.draft_accept() {
        line.push_str(&format!("  draft {:.0}%", acc * 100.0));
        if let Some(mean) = s.draft_mean() {
            line.push_str(&format!(" ({mean:.2} tok/step)"));
        }
    }
    line
}

fn fmt_opt(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.1}")).unwrap_or_else(|| "--".into())
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let t: String = s.chars().take(n).collect();
        format!("{t}...")
    }
}

/// llama.cpp prefixes its log lines with `<time> <level> <module>:`, where the
/// level is E, W, I or D. Read it so a W line whose text happens to contain
/// "failed" (the auto-fit warning is one) is not painted as a hard error.
fn llama_log_level(l: &str) -> Option<&str> {
    let mut it = l.split_whitespace();
    it.next()?;
    match it.next()? {
        lv @ ("E" | "W" | "I" | "D") => Some(lv),
        _ => None,
    }
}

/// Log row clipped to `max` visible cells.
///
/// llama.cpp prefixes every line with a `0.15.047.740` timestamp and pads the
/// module column. On a narrow terminal that prefix eats half the row and the
/// message is what gets truncated, so the displayed row drops the timestamp
/// (the log file keeps it) and collapses the padding.
fn log_line_w(l: &str, max: usize) -> String {
    let compact = l.split_whitespace().collect::<Vec<_>>().join(" ");
    let (level, text) = match llama_log_level(&compact) {
        Some(lv) => {
            let mut it = compact.splitn(3, ' ');
            let _ts = it.next();
            let _lv = it.next();
            let rest = it.next().unwrap_or("");
            (Some(lv), format!("{lv} {rest}"))
        }
        None => (None, compact.clone()),
    };
    let low = text.to_lowercase();
    let color = match level {
        Some("E") => RED,
        Some("W") => YELLOW,
        Some(_) => DIM,
        None if low.contains("error") || low.contains("failed") || low.contains("abort") => RED,
        None if low.contains("warn") => YELLOW,
        None => DIM,
    };
    format!("  {color}{}{RESET}", truncate(&text, max))
}

/// Last `n` non-empty lines of a file, reading only its tail.
fn tail_lines(path: &Path, n: usize) -> Vec<String> {
    let Ok(mut f) = File::open(path) else {
        return Vec::new();
    };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    let start = len.saturating_sub(16 * 1024);
    if f.seek(SeekFrom::Start(start)).is_err() {
        return Vec::new();
    }
    let mut buf = Vec::new();
    if f.read_to_end(&mut buf).is_err() {
        return Vec::new();
    }
    let text = String::from_utf8_lossy(&buf);
    let mut lines: Vec<String> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.to_string())
        .collect();
    if start > 0 && !lines.is_empty() {
        lines.remove(0); // the first line may be cut mid-way
    }
    let skip = lines.len().saturating_sub(n);
    lines.split_off(skip)
}

/// `localmodel run qwen36/ud-q4_k_xl` is the same hub entry as
/// `localmodel strata qwen36 ud-q4_k_xl`: a catalog target with a slash.
fn strata_target(args: &[String]) -> Option<(String, String, Vec<String>)> {
    const EXTRA: &[&str] = &["swift", "nvfp4", "deepseek", "unsloth"];
    for (i, a) in args.iter().enumerate() {
        let Some((fam, size)) = a.split_once('/') else {
            continue;
        };
        let known = fit::CATALOG.iter().any(|e| e.family == fam) || EXTRA.contains(&fam);
        if !known || size.is_empty() {
            continue;
        }
        let mut rest = args.to_vec();
        rest.remove(i);
        return Some((fam.to_string(), size.to_string(), rest));
    }
    None
}

/// Value that follows `flag` in an argv vector.
fn argv_value<'a>(argv: &'a [String], flag: &str) -> Option<&'a str> {
    argv.iter()
        .position(|a| a == flag)
        .and_then(|i| argv.get(i + 1))
        .map(String::as_str)
}

/// Shared by `run --explain` and `settings --show --origin`.
fn print_origin_rows(origins: &[Origin]) {
    println!("  {:<14} {:<26} {}", "SETTING", "VALUE", "SOURCE");
    for o in origins {
        let src = if o.note.is_empty() {
            o.from.label().to_string()
        } else {
            format!("{} ({})", o.from.label(), o.note)
        };
        println!("  {:<14} {:<26} {}", o.key, truncate(&o.value, 24), src);
    }
}

/// `run --explain`: the resolved knobs and their origin, then the per-model
/// values that `build_args` derives from the model itself.
fn print_explain(
    m: &Model,
    profile: &str,
    has_profile: bool,
    cfg: &Cfg,
    origins: &[Origin],
    port: u16,
    argv: Option<&[String]>,
) {
    let stem = models::stem(&m.name);
    println!("model    {stem}");
    if has_profile {
        println!("profile  [model.\"{profile}\"] over [defaults]");
    } else if profile == stem {
        println!("profile  [defaults] only (no [model.\"{profile}\"] entry)");
    } else {
        println!("profile  [model.\"{profile}\"] not found, using [defaults]");
    }
    println!("order    cli > model > defaults > builtin > derived");
    println!();
    print_origin_rows(origins);

    // Values no profile can set: hardcoded per model class, or constants.
    let mut rows: Vec<(&str, String, String)> = Vec::new();
    if cfg.port.is_none() {
        rows.push(("port", port.to_string(), "first free from 8080".into()));
    }
    if let Some(argv) = argv {
        let from_extra = |f: &str| {
            cfg.extra
                .as_deref()
                .map(|e| e.split_whitespace().any(|p| p == f || p.starts_with(&format!("{f}="))))
                .unwrap_or(false)
        };
        if cfg.ngl.is_none() {
            if let Some(v) = argv_value(argv, "-ngl") {
                let note = if m.moe { "MoE default (all layers)" } else { "dense default (all layers)" };
                rows.push(("ngl", v.to_string(), note.into()));
            }
        }
        if cfg.ncmoe.is_none() {
            if let Some(v) = argv_value(argv, "--n-cpu-moe") {
                rows.push(("ncmoe", v.to_string(), "MoE default".into()));
            }
        }
        if let Some(v) = argv_value(argv, "--batch-size") {
            let note = if from_extra("--batch-size") { "from extra" } else { "model class default" };
            rows.push(("batch-size", v.to_string(), note.into()));
        }
        if let Some(v) = argv_value(argv, "--ubatch-size") {
            let note = if from_extra("--ubatch-size") { "from extra" } else { "model class default" };
            rows.push(("ubatch-size", v.to_string(), note.into()));
        }
        if cfg.draft_tokens.is_none() {
            if let Some(v) = argv_value(argv, "--spec-draft-n-max") {
                rows.push(("draft_tokens", v.to_string(), "speculative default".into()));
            }
        }
        if let Some(v) = argv_value(argv, "--spec-type") {
            let note = if m.nextn { "built-in MTP head" } else { "external draft file" };
            rows.push(("spec-type", v.to_string(), note.into()));
        }
        if let Some(v) = argv_value(argv, "--top-k") {
            let note = if from_extra("--top-k") { "from extra" } else { "sampling default" };
            rows.push(("top-k", v.to_string(), note.into()));
        }
        if let Some(v) = argv_value(argv, "--cache-reuse") {
            rows.push(("cache-reuse", v.to_string(), "constant (off on hybrid SSM)".into()));
        }
        if argv.iter().any(|a| a == "--mmproj-auto") {
            rows.push(("mmproj", "auto".into(), "vision without a projector path".into()));
        }
        if cfg.alias.is_none() {
            rows.push(("alias", stem.clone(), "model file stem".into()));
        }
    }
    if !rows.is_empty() {
        println!();
        println!("  DERIVED (computed by localmodel)");
        for (k, v, note) in rows {
            println!("  {:<14} {:<26} {}", k, truncate(&v, 24), note);
        }
    }
    println!();
}

fn cmd_run(args: &[String]) -> i32 {
    let mut o = RunOpts::default();
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let val = |i: &mut usize| -> Option<String> {
            *i += 1;
            args.get(*i).cloned()
        };
        match a {
            "--ctx" | "-c" => o.ctx = val(&mut i).and_then(|v| parse_ctx(&v)),
            "--port" | "-p" => o.port = val(&mut i).and_then(|v| v.parse().ok()),
            "--ngl" => o.ngl = val(&mut i).and_then(|v| v.parse().ok()),
            "--ncmoe" => o.ncmoe = val(&mut i).and_then(|v| v.parse().ok()),
            "--load-mode" => o.load_mode = val(&mut i),
            "--moe-cache-mib" => o.moe_cache_mib = val(&mut i).and_then(|v| v.parse().ok()),
            "--cpu-moe" => o.cpu_moe = Some(true),
            "--no-cpu-moe" => o.cpu_moe = Some(false),
            "--n-cpu-ffn" => o.n_cpu_ffn = val(&mut i).and_then(|v| v.parse().ok()),
            "--override-tensor" => o.override_tensor = val(&mut i),
            "--lazy-mode" => o.lazy_mode = val(&mut i),
            "--numa" => o.numa = val(&mut i),
            "--no-op-offload" => o.no_op_offload = Some(true),
            "--batch-size" | "-b" => o.batch_size = val(&mut i).and_then(|v| v.parse().ok()),
            "--ubatch-size" | "-ub" => o.ubatch_size = val(&mut i).and_then(|v| v.parse().ok()),
            "--parallel" => o.parallel = val(&mut i).and_then(|v| v.parse().ok()),
            "--threads-batch" | "-tb" => o.threads_batch = val(&mut i).and_then(|v| v.parse().ok()),
            "--prio" => o.prio = val(&mut i).and_then(|v| v.parse().ok()),
            "--cache-reuse" => o.cache_reuse = val(&mut i).and_then(|v| v.parse().ok()),
            "--spec-draft-n-min" => o.draft_n_min = val(&mut i).and_then(|v| v.parse().ok()),
            "--spec-draft-p-min" => o.draft_p_min = val(&mut i).and_then(|v| v.parse().ok()),
            "--top-k" => o.top_k = val(&mut i).and_then(|v| v.parse().ok()),
            "--min-p" => o.min_p = val(&mut i).and_then(|v| v.parse().ok()),
            "--repeat-penalty" => o.repeat_penalty = val(&mut i).and_then(|v| v.parse().ok()),
            "--presence-penalty" => o.presence_penalty = val(&mut i).and_then(|v| v.parse().ok()),
            "--frequency-penalty" => o.frequency_penalty = val(&mut i).and_then(|v| v.parse().ok()),
            "--seed" => o.seed = val(&mut i).and_then(|v| v.parse().ok()),
            "--api-key" => o.api_key = val(&mut i),
            "--timeout" => o.timeout = val(&mut i).and_then(|v| v.parse().ok()),
            "--threads-http" => o.threads_http = val(&mut i).and_then(|v| v.parse().ok()),
            "--engine" => o.engine = val(&mut i),
            "--family" => o.family = val(&mut i),
            "--size" => o.size = val(&mut i),
            "--vision" => o.vision = val(&mut i),
            "--mmproj" => o.mmproj = val(&mut i),
            "--access" => o.access = val(&mut i),
            "--tune" => o.tune = val(&mut i),
            "--kv" => {
                if let Some(v) = val(&mut i) {
                    o.ctk = Some(v.clone());
                    o.ctv = Some(v);
                }
            }
            "--ctk" => {
                if let Some(v) = val(&mut i) {
                    o.ctk = Some(v);
                }
            }
            "--ctv" => {
                if let Some(v) = val(&mut i) {
                    o.ctv = Some(v);
                }
            }
            "--draft-ctk" => o.draft_ctk = val(&mut i),
            "--draft-ctv" => o.draft_ctv = val(&mut i),
            "--reasoning" => o.reasoning = val(&mut i),
            "--effort" => o.effort = val(&mut i),
            "--draft-tokens" | "--spec-n-max" => o.draft_tokens = val(&mut i).and_then(|v| v.parse().ok()),
            "--threads" | "-t" => o.threads = val(&mut i).and_then(|v| v.parse().ok()),
            "--extra" => o.extra = val(&mut i),
            "--tools" => o.tools = val(&mut i),
            "--profile" => o.profile = val(&mut i),
            "--dev-scratch" => {
                if let Some(v) = val(&mut i) {
                    let p = toml::from_str(&v).unwrap_or_default();
                    o.dev_scratch = Some(p);
                }
            }
            "--draft-gpu" => o.draft_gpu = Some(true),
            "--no-draft-gpu" => o.draft_gpu = Some(false),
            "--kv-ram" => o.kv_ram = Some(true),
            "--kv-vram" => o.kv_ram = Some(false),
            "--split-mode" => o.split_mode = val(&mut i),
            "--tensor-split" => o.tensor_split = val(&mut i),
            "--log-verbosity" | "-lv" => o.log_verbosity = val(&mut i).and_then(|v| v.parse().ok()),
            "--alias" => o.alias = val(&mut i),
            "--repl" => o.repl = true,
            "--dry-run" => o.dry_run = true,
            "--explain" => o.explain = true,
            "--strict" => o.strict = true,
            "--llama-help" => o.llama_help = true,
            "--help" | "-h" => {
                help();
                return 0;
            }
            s if s.starts_with('-') => {}
            s => o.query.push(s.to_string()),
        }
        i += 1;
    }

    let dir = config::models_dir();
    let bin = config::server_bin();
    if !bin.exists() {
        eprintln!("llama-server not found at {}", bin.display());
        return 1;
    }

    // No model needed: show the installed binary's own reference.
    if o.llama_help {
        return match Command::new(&bin).arg("--help").status() {
            Ok(s) if s.success() => 0,
            _ => 1,
        };
    }

    if o.query.is_empty() {
        eprintln!("usage: localmodel run <model>  (use `localmodel list` to see models)");
        return 1;
    }

    let ms = models::scan(&dir);
    if ms.is_empty() {
        eprintln!("no models in {}. add one: localmodel add <url>", dir.display());
        return 1;
    }

    let model = match pick(&ms, &o.query) {
        Some(m) => m,
        None => return 1,
    };

    let stem = models::stem(&model.name);
    let saved = settings::load();
    let key = o.profile.clone().unwrap_or_else(|| stem.clone());
    if o.profile.is_some() && !saved.model.contains_key(&key)
        && o.dev_scratch.is_none()
    {
        eprintln!("warning: no settings profile '{key}', using defaults for {stem}");
    }
    let model_profile = o.dev_scratch.clone()
        .or_else(|| saved.model.get(&key).map(|p| p.clone()));
    let (cfg, origins) = Cfg::explain(&o, &saved.defaults, model_profile.as_ref());
    let alias = cfg.alias.clone().unwrap_or_else(|| stem.clone());
    let port = cfg.port.unwrap_or_else(|| server::free_port(8080));

    // MoE -> Strata, dense -> llama.cpp. An explicit --engine wins.
    let forced = o.engine.as_deref().and_then(engine::Engine::parse);
    let chosen = engine::route(&model, forced);
    if chosen == engine::Engine::Strata {
        if !engine::supports_strata(&model.name) {
            eprintln!(
                "warning: strata runs only Qwen3.6-35B-A3B, Qwen3.8-Flash-Next and DeepSeek V4.1;"
            );
            eprintln!("         {} is not one of them. Use --engine llama.", model.name);
        }
        let mut spec = strata_spec_from(&o, &cfg, port);
        match (&o.family, &o.size) {
            (Some(f), Some(s)) => {
                spec.family = Some(f.clone());
                spec.size = Some(s.clone());
            }
            _ => spec.model_path = Some(model.path.clone()),
        }
        if o.explain {
            print_explain(&model, &key, model_profile.is_some(), &cfg, &origins, port, None);
        }
        return run_strata(
            spec,
            Some(&model),
            &alias,
            cfg.ctx,
            port,
            o.repl,
            o.dry_run || o.explain,
            &dir,
        );
    }

    let draft = models::find_draft(&model.path);
    let sv = build_args(&model, draft.as_deref(), &alias, port, &cfg);

    // `extra` is the one open-ended channel; check its flags against the
    // binary that will actually run, and say when it shadows a typed knob.
    let issues = if o.explain || o.strict {
        let active: Vec<&str> = origins.iter().map(|o| o.key).collect();
        validate::check(&sv, cfg.extra.as_deref().unwrap_or(""), &bin, &active)
    } else {
        Vec::new()
    };

    // Provenance is only useful with the exact result it explains, so show
    // the table first and the command after it.
    if o.explain {
        print_explain(&model, &key, model_profile.is_some(), &cfg, &origins, port, Some(&sv));
        if !issues.is_empty() {
            println!("  WARNINGS");
            for w in &issues {
                println!("  - {w}");
            }
            println!();
        }
    }
    if o.strict && !issues.is_empty() {
        eprintln!("settings check failed:");
        for w in &issues {
            eprintln!("  - {w}");
        }
        return 2;
    }

    // Show the exact command instead of launching.
    if o.dry_run || o.explain {
        println!("{}", shell_cmd(&bin, &sv));
        return 0;
    }

    let logpath = dir.join(format!(".localmodel-{port}.log"));

    let tty = std::io::stdout().is_terminal();
    let dashboard = tty && !o.repl;

    // SIGINT fallback (Ctrl-C is a normal key while the dashboard is in raw
    // mode): kill whatever server is current and restore the terminal.
    let pid_slot = Arc::new(AtomicU32::new(0));
    let slot = pid_slot.clone();
    let _ = ctrlc::set_handler(move || {
        let pid = slot.load(Ordering::Relaxed);
        if pid > 0 {
            let _ = Command::new("kill").args(["-TERM", &pid.to_string()]).status();
        }
        let _ = terminal::disable_raw_mode();
        let _ = execute!(std::io::stdout(), LeaveAlternateScreen, Show);
        let _ = std::io::stdout().flush();
        std::process::exit(0);
    });

    if dashboard {
        let session = match Session::new(
            bin.clone(),
            model.clone(),
            draft.clone(),
            dir.clone(),
            pid_slot.clone(),
            alias.clone(),
            port,
            cfg.clone(),
            logpath.clone(),
        ) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("failed to start llama-server: {e}");
                return 1;
            }
        };
        return run_dashboard(session);
    }

    // The dashboard paints the card itself; line mode prints it once.
    println!(
        "{}",
        model_card(&model, &alias, port, &cfg, &logpath, "llama.cpp", draft.as_deref())
    );
    println!();

    let (child, stats) = match spawn_server(&bin, &sv, &logpath, cfg.ctx) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("failed to start llama-server: {e}");
            return 1;
        }
    };
    pid_slot.store(child.id(), Ordering::Relaxed);

    let info = Info {
        alias: alias.clone(),
        port,
        pid: child.id(),
        ctx: cfg.ctx,
        ctk: cfg.ctk.clone(),
        ctv: cfg.ctv.clone(),
    };

    let child = Arc::new(Mutex::new(child));
    let running = Arc::new(AtomicBool::new(true));

    // Line mode: --repl chat, or headless wait when stdout is not a TTY.
    println!("url    http://127.0.0.1:{port}/v1");
    println!("log    {}", logpath.display());
    println!("settings: localmodel settings   flags: localmodel params");
    println!("type /help or exit;  stop from outside: localmodel stop");
    println!();

    {
        let stats = stats.clone();
        let running = running.clone();
        let info = info.clone();
        let model_for_monitor = model.clone();
        thread::spawn(move || monitor(stats, running, info, model_for_monitor));
    }

    let stdin_tty = std::io::stdin().is_terminal();
    if !stdin_tty || !o.repl {
        while running.load(Ordering::Relaxed) {
            let exited = {
                let mut c = child.lock().unwrap();
                matches!(c.try_wait(), Ok(Some(_)))
            };
            if exited {
                break;
            }
            thread::sleep(Duration::from_millis(400));
        }
        running.store(false, Ordering::Relaxed);
        return 0;
    }

    // Wait for server to be alive (first stats sample).
    while running.load(Ordering::Relaxed) {
        let alive = stats.lock().unwrap().alive;
        if alive {
            break;
        }
        thread::sleep(Duration::from_millis(300));
    }
    if let Some(c) = props_ctx(port) {
        stats.lock().unwrap().ctx = c;
    }

    let mut line = String::new();
    // /effort override for this session: built from the model's template.
    let caps = jinja::caps_for_model(&model.path);
    let mut effort: Option<Value> = None;
    loop {
        if !running.load(Ordering::Relaxed) {
            break;
        }
        print!("\u{00bb} ");
        let _ = std::io::stdout().flush();
        line.clear();
        match std::io::stdin().read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let t = line.trim();
        if let Some(level) = t.strip_prefix("/effort ") {
            let level = level.trim();
            if level.is_empty() || level == "auto" {
                effort = None;
                println!("effort: auto (template default)");
            } else if let Some(cp) = &caps {
                match jinja::effort_kwargs(cp, Some(level)) {
                    Some(kw) => {
                        effort = Some(kw);
                        println!("effort: {level}");
                    }
                    None => println!("template has no effort level '{level}'"),
                }
            } else {
                println!("no chat template in this model");
            }
            continue;
        }
        match t {
            "" => continue,
            "exit" | "quit" | "q" | "/quit" | "/exit" | "/stop" => break,
            "/help" => repl_help(),
            "/effort" => println!(
                "effort: {}",
                effort
                    .as_ref()
                    .map(|k| k.to_string())
                    .unwrap_or_else(|| "auto (template default)".to_string())
            ),
            "/stats" => {
                let snapshot = { stats.lock().unwrap().clone() };
                println!("{}", stats_line(&info, &snapshot, &model));
            }
            other => {
                print!("\u{2026} ");
                let _ = std::io::stdout().flush();
                match chat(port, &alias, other, None, effort.as_ref()) {
                    Some(r) => println!("\n{r}"),
                    None => println!("(no response)"),
                }
                let snapshot = { stats.lock().unwrap().clone() };
                println!("{}", stats_line(&info, &snapshot, &model));
            }
        }
    }

    running.store(false, Ordering::Relaxed);
    if let Ok(mut c) = child.lock() {
        let _ = c.kill();
    }
    println!("bye");
    0
}

/// Spawn llama-server, stream stderr into the log file, and keep Stats in sync.
fn spawn_server(
    bin: &Path,
    args: &[String],
    logpath: &Path,
    ctx: u32,
) -> std::io::Result<(Child, Arc<Mutex<Stats>>)> {
    let log = File::create(logpath).ok();
    let mut child = Command::new(bin)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()?;
    let stderr = child.stderr.take().unwrap();
    let stats = Arc::new(Mutex::new(Stats { ctx: ctx as u64, ..Default::default() }));
    let st = stats.clone();
    let mut logf = log;
    thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines().flatten() {
            if let Some(f) = logf.as_mut() {
                let _ = writeln!(f, "{line}");
            }
            if let Ok(mut s) = st.lock() {
                s.observe(&line);
            }
        }
    });
    Ok((child, stats))
}

/// One running server plus everything needed to restart it.
struct Session {
    bin: PathBuf,
    model: Model,
    draft: Option<PathBuf>,
    dir: PathBuf,
    pid_slot: Arc<AtomicU32>,
    child: Arc<Mutex<Child>>,
    stats: Arc<Mutex<Stats>>,
    info: Info,
    logpath: PathBuf,
    cfg: Cfg,
}

impl Session {
    #[allow(clippy::too_many_arguments)]
    fn new(
        bin: PathBuf,
        model: Model,
        draft: Option<PathBuf>,
        dir: PathBuf,
        pid_slot: Arc<AtomicU32>,
        alias: String,
        port: u16,
        cfg: Cfg,
        logpath: PathBuf,
    ) -> std::io::Result<Self> {
        let sv = build_args(&model, draft.as_deref(), &alias, port, &cfg);
        let (child, stats) = spawn_server(&bin, &sv, &logpath, cfg.ctx)?;
        let pid = child.id();
        pid_slot.store(pid, Ordering::Relaxed);
        Ok(Self {
            bin,
            model,
            draft,
            dir,
            pid_slot,
            child: Arc::new(Mutex::new(child)),
            stats,
            info: Info {
                alias,
                port,
                pid,
                ctx: cfg.ctx,
                ctk: cfg.ctk.clone(),
                ctv: cfg.ctv.clone(),
            },
            logpath,
            cfg,
        })
    }

    fn restart(&mut self, cfg: &Cfg) -> std::io::Result<()> {
        if let Ok(mut c) = self.child.lock() {
            let _ = c.kill();
            let _ = c.wait();
        }
        let alias = cfg
            .alias
            .clone()
            .unwrap_or_else(|| models::stem(&self.model.name));
        let port = cfg.port.unwrap_or_else(|| server::free_port(8080));
        let logpath = self.dir.join(format!(".localmodel-{port}.log"));
        let sv = build_args(&self.model, self.draft.as_deref(), &alias, port, cfg);
        let (child, stats) = spawn_server(&self.bin, &sv, &logpath, cfg.ctx)?;
        let pid = child.id();
        self.pid_slot.store(pid, Ordering::Relaxed);
        self.child = Arc::new(Mutex::new(child));
        self.stats = stats;
        self.logpath = logpath;
        self.cfg = cfg.clone();
        self.info = Info {
            alias,
            port,
            pid,
            ctx: cfg.ctx,
            ctk: cfg.ctk.clone(),
            ctv: cfg.ctv.clone(),
        };
        Ok(())
    }

    fn exited(&self) -> Option<std::process::ExitStatus> {
        let mut c = self.child.lock().unwrap();
        c.try_wait().ok().flatten()
    }

    fn stop(&self) {
        self.pid_slot.store(0, Ordering::Relaxed);
        if let Ok(mut c) = self.child.lock() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

/// Two states are enough now that the settings editor lives in `hub.rs`:
/// the live card and its key reference.
enum View {
    Main,
    Help,
}

fn render_help(info: &Info, cols: usize) -> String {
    let body = vec![
        String::new(),
        section("KEYS"),
        format!("  {GREEN}?  h{RESET}      this help"),
        format!("  {GREEN}s{RESET}         where settings moved (hub)"),
        format!("  {GREEN}l{RESET}         show / hide the log panel"),
        format!("  {GREEN}p{RESET}         pause: freeze the screen so text can be selected and copied"),
        format!("  {GREEN}o{RESET}         resume"),
        format!("  {GREEN}r{RESET}         restart the server"),
        format!("  {GREEN}q  Esc{RESET}    stop the server and quit"),
        format!("  {GREEN}Ctrl-C{RESET}    stop the server and quit"),
        String::new(),
        section("SETTINGS"),
        format!("  {GREEN}localmodel hub{RESET}        model list + its knobs, save & launch"),
        format!("  {GREEN}localmodel settings{RESET}   non-interactive --set / --show --origin"),
        format!("  {GREEN}localmodel params{RESET}     llama.cpp flag cheat sheet"),
        String::new(),
        section("NOTES"),
        format!("  {DIM}RAM  system used = total - available (page cache excluded)"),
        format!("  {DIM}     server rss includes the mmap'ed model; anon is what it holds{RESET}"),
        format!("  {DIM}Host (mmap) = CPU weights are file-backed; experts are read from disk{RESET}"),
        format!("  {DIM}-ngl moves whole blocks; --n-cpu-moe moves only expert tensors{RESET}"),
        String::new(),
    ];
    boxed(
        &format!("{CYAN}{BOLD}help{RESET} {DIM}{}{RESET}", info.alias),
        &body,
        &format!("{DIM}? or Esc closes this help{RESET}"),
        cols,
    )
}

/// Apply one key press to the current view. Returns true when the dashboard
/// should shut down. `hint` is a one-frame message under the card (used by
/// `s`, which no longer opens a settings editor: that is `localmodel hub`).
fn handle_key(
    s: &mut Session,
    view: &mut View,
    show_logs: &mut bool,
    hint: &mut Option<String>,
    paused: &mut bool,
    k: KeyEvent,
) -> bool {
    // Ctrl-C / Ctrl-D quit; Enter may arrive as \n when input is piped.
    if k.modifiers.contains(KeyModifiers::CONTROL)
        && matches!(k.code, KeyCode::Char('c') | KeyCode::Char('d'))
    {
        return true;
    }
    let k = if k.code == KeyCode::Char('\n') || k.code == KeyCode::Char('\r') {
        KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)
    } else {
        k
    };
    *hint = None;
    let mut next: Option<View> = None;
    match view {
        View::Main => match k.code {
            KeyCode::Char('?') | KeyCode::Char('h') => next = Some(View::Help),
            KeyCode::Char('s') => {
                *hint = Some(
                    "settings moved: run `localmodel hub`, edit the model's row, then `s` or `L`"
                        .to_string(),
                );
            }
            KeyCode::Char('l') => *show_logs = !*show_logs,
            KeyCode::Char('p') | KeyCode::Char('P') => {
                *paused = true;
                *hint = Some("paused - select and copy freely, o resumes".to_string());
            }
            KeyCode::Char('o') | KeyCode::Char('O') => {
                *paused = false;
                *hint = None;
            }
            KeyCode::Char('r') => {
                let cfg = s.cfg.clone();
                let _ = s.restart(&cfg);
            }
            KeyCode::Char('q') | KeyCode::Esc => return true,
            _ => {}
        },
        View::Help => match k.code {
            KeyCode::Esc | KeyCode::Char('?') | KeyCode::Char('h') => next = Some(View::Main),
            KeyCode::Char('q') => return true,
            _ => {}
        },
    }
    if let Some(v) = next {
        *view = v;
    }
    false
}

/// Full-screen interactive dashboard: card, live stats, logs, settings.
/// Raw mode is enabled so single keys work; Ctrl-C also stops the server.
fn run_dashboard(mut s: Session) -> i32 {
    let mut out = std::io::stdout();
    let _ = terminal::enable_raw_mode();
    let _ = execute!(out, EnterAlternateScreen, Hide);
    let _ = out.flush();

    let mut view = View::Main;
    let mut show_logs = true;
    let mut hint: Option<String> = None;
    let mut paused = false;
    let mut redraw = true;
    let mut frame = 0usize;
    let mut exit_status = None;
    let hw = fit::Hardware::probe(&s.dir);
    // Parse an external draft once; built-in MTP needs no file.
    let draft_meta = s.draft.as_ref().and_then(|p| {
        std::fs::metadata(p)
            .ok()
            .map(|m| (p.clone(), gguf::parse(p), m.len()))
    });

    loop {
        // While paused, stop repainting so the terminal selection survives and
        // the card can be copied. State changes (p / o / l / help) still cause
        // exactly one repaint so the PAUSED banner appears at once.
        if !paused || redraw {
            // Snapshot under a short lock, then do slow I/O (nvidia-smi, /proc)
            // outside it so the stderr reader is never starved.
            let snapshot = { s.stats.lock().unwrap().clone() };
            let gpus = server::gpu_infos();
            let ram = ram_text(s.info.pid);
            let (cols, rows) = screen_size();
            // Re-read every frame: the table is written in two parts (target,
            // then draft), so a one-shot read can catch only the first one. The
            // file head is small, so this is cheap.
            let live = telemetry::read(&s.logpath);
            let draft = draft_meta
                .as_ref()
                .map(|(_, meta, bytes)| mem::Draft { meta, weights: *bytes });
            let plan = mem::plan(&s.model, draft, &s.cfg, &hw);
            let screen = match &view {
                View::Main => render_screen(
                    &s.model,
                    &s.info,
                    &snapshot,
                    draft_meta.is_some() || s.model.nextn,
                    &gpus,
                    &ram,
                    &s.logpath,
                    show_logs,
                    frame,
                    hint.as_deref(),
                    paused,
                    cols,
                    rows,
                    &s.cfg,
                    &hw,
                    &plan,
                    live.as_ref(),
                ),
                View::Help => render_help(&s.info, cols),
            };
            // Every line is clipped to the terminal width, so nothing wraps and
            // there is no leftover text to smear; each line is erased before it
            // is redrawn.
            print!("{}", paint_screen(&screen, cols));
            let _ = out.flush();
            redraw = false;
        }

        if let Some(st) = s.exited() {
            exit_status = Some(st);
            break;
        }

        let mut quit = false;
        // Wait for input, then drain everything already queued: one render per
        // burst keeps a fast key sequence responsive even when the frame itself
        // (nvidia-smi) takes a while. Resize events just fall through to the
        // next render, which re-reads the terminal size.
        let timeout = if paused { Duration::from_millis(250) } else { Duration::from_millis(500) };
        if event::poll(timeout).unwrap_or(false) {
            loop {
                if let Ok(Event::Key(k)) = event::read() {
                    if handle_key(&mut s, &mut view, &mut show_logs, &mut hint, &mut paused, k) {
                        quit = true;
                    }
                    redraw = true;
                }
                if quit || !event::poll(Duration::from_millis(0)).unwrap_or(false) {
                    break;
                }
            }
        }
        if quit {
            break;
        }
        frame += 1;
    }

    s.stop();
    let _ = execute!(out, LeaveAlternateScreen, Show);
    let _ = terminal::disable_raw_mode();
    let _ = out.flush();

    match exit_status {
        Some(st) if st.success() => {
            println!("server stopped. log: {}", s.logpath.display());
        }
        Some(st) => {
            println!("server exited ({st}). last log lines:\n");
            let txt = std::fs::read_to_string(&s.logpath).unwrap_or_default();
            let lines: Vec<&str> = txt.lines().collect();
            let start = lines.len().saturating_sub(15);
            for l in &lines[start..] {
                println!("  {l}");
            }
        }
        None => println!("stopped. log: {}", s.logpath.display()),
    }
    0
}

fn monitor(stats: Arc<Mutex<Stats>>, running: Arc<AtomicBool>, info: Info, model: Model) {
    while running.load(Ordering::Relaxed) {
        // Clone the counters, then release the lock before doing slow I/O
        // (nvidia-smi, curl) so the log reader thread is never starved.
        let snapshot = { stats.lock().unwrap().clone() };
        println!("\r{}", stats_line(&info, &snapshot, &model));
        thread::sleep(Duration::from_secs(1));
    }
}

fn repl_help() {
    println!(
        "  <text>      send a chat message\n  /effort L   thinking level: high|xhigh|medium|low|off|auto\n  /stats      print status line\n  /help       this help\n  exit        stop server and quit"
    );
}

/// Quote a single argument for a readable shell command line.
fn shell_arg(s: &str) -> String {
    if s.is_empty() {
        "''".to_string()
    } else if s
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || "-_./:=@,+%{}[]".contains(c))
    {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', "'\\''"))
    }
}

fn shell_cmd(bin: &Path, args: &[String]) -> String {
    let mut parts = vec![shell_arg(&bin.display().to_string())];
    parts.extend(args.iter().map(|a| shell_arg(a)));
    parts.join(" ")
}

/// llama.cpp parameter reference: curated cheat sheet, or the installed
/// binary's own `--help` filtered by a pattern.
fn cmd_params(args: &[String]) -> i32 {
    let bin = config::server_bin();
    let all = args.iter().any(|a| a == "--all" || a == "-a");
    let pattern = args
        .iter()
        .find(|a| !a.starts_with('-'))
        .map(|s| s.to_lowercase());

    if all || pattern.is_some() {
        if !bin.exists() {
            eprintln!("llama-server not found at {}", bin.display());
            return 1;
        }
        let out = match Command::new(&bin).arg("--help").output() {
            Ok(o) => o,
            Err(e) => {
                eprintln!("failed to run {} --help: {e}", bin.display());
                return 1;
            }
        };
        let stdout = String::from_utf8_lossy(&out.stdout).to_string();
        let text = if stdout.trim().is_empty() {
            String::from_utf8_lossy(&out.stderr).to_string()
        } else {
            stdout
        };
        match pattern {
            None => print!("{text}"),
            Some(p) => {
                let mut hit = false;
                for line in text.lines() {
                    if line.to_lowercase().contains(&p) {
                        println!("{line}");
                        hit = true;
                    }
                }
                if !hit {
                    eprintln!("no parameter matching '{p}' (try `localmodel params --all`)");
                    return 1;
                }
            }
        }
        return 0;
    }

    println!("{BOLD}llama.cpp server parameters{RESET} {DIM}(most useful for localmodel){RESET}\n");
    let groups: [(&str, &[(&str, &str)]); 7] = [
        (
            "MODEL / GPU OFFLOAD",
            &[
                ("-ngl, --n-gpu-layers N", "layers offloaded to GPU (999 = all)"),
                ("--n-cpu-moe N", "keep N MoE expert layers on CPU (saves VRAM)"),
                ("--split-mode MODE", "none | layer | row - multi-GPU split"),
                ("--tensor-split LIST", "per-GPU VRAM share, e.g. 0.8,0.2"),
                ("--main-gpu N", "primary GPU id"),
                ("--flash-attn on|off|auto", "flash attention (on by default on CUDA)"),
                ("--mmproj FILE", "vision projector for multimodal models"),
            ],
        ),
        (
            "CONTEXT / BATCH",
            &[
                ("-c, --ctx-size N", "context length in tokens"),
                ("-b, --batch-size N", "logical batch size (default 2048)"),
                ("-ub, --ubatch-size N", "physical micro-batch (default 512)"),
                ("-np, --parallel N", "parallel slots (each has its own KV)"),
                ("--cache-type-k TYPE", "KV type for K: f16, q8_0, q4_0, ..."),
                ("--cache-type-v TYPE", "KV type for V: f16, q8_0, q4_0, ..."),
                ("--defrag-thold F", "KV cache defragmentation threshold"),
            ],
        ),
        (
            "SAMPLING",
            &[
                ("--temp F", "temperature (1.0 = model default scale)"),
                ("--top-k N", "top-k sampling cutoff"),
                ("--top-p F", "nucleus sampling (0.95 typical)"),
                ("--min-p F", "min-p sampling"),
                ("--repeat-penalty F", "repetition penalty (1.0 = off)"),
                ("--seed N", "RNG seed; -1 = random"),
            ],
        ),
        (
            "SPECULATIVE DECODING",
            &[
                ("--spec-type TYPE", "draft-dflash, draft-eagle3, draft-mtp, ngram-*"),
                ("--spec-draft-model FILE", "small draft model (.gguf)"),
                ("--spec-draft-n-max N", "max draft tokens per step"),
                ("--spec-draft-ngl N", "GPU layers for the draft model"),
            ],
        ),
        (
            "SERVER",
            &[
                ("--host HOST", "bind address (127.0.0.1 keeps it local)"),
                ("--port N", "HTTP port (default 8080)"),
                ("--alias NAME", "served model name used by /v1 APIs"),
                ("--jinja", "use the model's Jinja chat template"),
                ("--chat-template-kwargs JSON", "extra template args (reasoning effort)"),
                ("--api-key KEY", "require this bearer token"),
            ],
        ),
        (
            "REASONING",
            &[
                ("--reasoning on|off", "enable/disable thinking output"),
                ("--reasoning-format FMT", "none | deepseek | deepseek-legacy"),
            ],
        ),
        (
            "RUNTIME",
            &[
                ("-t, --threads N", "CPU threads (physical cores usually best)"),
                ("--threads-batch N", "threads used for prompt processing"),
                ("--no-mmap", "load fully into RAM instead of mmap"),
                ("--mlock", "lock model in RAM (prevents swapping)"),
                ("-v, -vv, -vvv", "log verbosity"),
            ],
        ),
    ];
    for (name, items) in groups {
        println!("{CYAN}{BOLD}{name}{RESET}");
        for (flag, desc) in items {
            println!("  {GREEN}{flag:<30}{RESET} {desc}");
        }
        println!();
    }
    println!("{DIM}Anything else: `localmodel params <pattern>` greps this build's `llama-server --help`,");
    println!("`localmodel params --all` prints it in full, and `--extra \"...\"` passes flags through.{RESET}");
    0
}

/// `localmodel settings` — edit and persist launch settings.
#[derive(Clone)]
enum SettingsScope {
    Defaults,
    Model(String),
}

fn edit_values(prof: &mut Profile, sets: &[(String, String)], unsets: &[String]) -> bool {
    for (k, v) in sets {
        match prof.set(k, v) {
            Ok(()) => println!("{k} = {}", prof.get(k).unwrap_or_default()),
            Err(e) => {
                eprintln!("{e}");
                return false;
            }
        }
    }
    for k in unsets {
        let had = prof.get(k).is_some();
        match prof.set(k, "-") {
            Ok(()) => println!("{k} unset{}", if had { "" } else { " (was not set)" }),
            Err(e) => {
                eprintln!("{e}");
                return false;
            }
        }
    }
    true
}

/// Knob keys that have a value; used to tell "extra overrides this setting"
/// from "extra duplicates an unset knob".
fn active_keys(p: &Profile) -> Vec<&'static str> {
    crate::knobs::KEYS
        .iter()
        .copied()
        .filter(|k| p.get(k).is_some())
        .collect()
}

fn warn_extra(p: &Profile) {
    let Some(extra) = p.extra.as_deref() else { return };
    let active = active_keys(p);
    for w in crate::validate::check(&[], extra, &crate::config::server_bin(), &active) {
        eprintln!("warning: {w}");
    }
}

fn apply_settings(
    st: &mut settings::Settings,
    scope: &SettingsScope,
    sets: &[(String, String)],
    unsets: &[String],
) -> i32 {
    match scope {
        SettingsScope::Defaults => {
            let mut prof = st.defaults.clone();
            if !edit_values(&mut prof, sets, unsets) {
                return 1;
            }
            warn_extra(&prof);
            st.defaults = prof;
        }
        SettingsScope::Model(name) => {
            // Keep the spelling already in the file; only a brand-new profile
            // takes the name as typed.
            let key = settings::find_key(st, name)
                .map(str::to_string)
                .unwrap_or_else(|| name.clone());
            let mut prof = st.model.get(&key).cloned().unwrap_or_default();
            if !edit_values(&mut prof, sets, unsets) {
                return 1;
            }
            if prof.is_empty() {
                st.model.remove(&key);
            } else {
                // Shadowing is about the merged profile: a flag in extra can
                // override a value inherited from [defaults] too.
                let merged = st.defaults.overlay(&prof);
                warn_extra(&merged);
                st.model.insert(key, prof);
            }
        }
    }
    match settings::save(st) {
        Ok(p) => {
            println!("saved {}", p.display());
            0
        }
        Err(e) => {
            eprintln!("failed to save settings: {e}");
            1
        }
    }
}

/// `settings --show --origin`: the same provenance table as `run --explain`,
/// but without CLI flags (there are none on this path).
fn show_origins(st: &settings::Settings, scope: &SettingsScope) -> i32 {
    match scope {
        SettingsScope::Defaults => {
            let (_, origins) = Cfg::explain(&RunOpts::default(), &st.defaults, None);
            println!("scope    [defaults]");
            println!("order    defaults > builtin > derived");
            println!();
            print_origin_rows(&origins);
        }
        SettingsScope::Model(name) => {
            let Some(key) = settings::find_key(st, name) else {
                eprintln!("no settings profile '{name}'");
                eprintln!("scopes: localmodel settings --show");
                return 1;
            };
            let (_, origins) =
                Cfg::explain(&RunOpts::default(), &st.defaults, Some(&st.model[key]));
            println!("scope    [model.\"{key}\"] over [defaults]");
            println!("order    model > defaults > builtin > derived");
            println!();
            print_origin_rows(&origins);
            // Show what this profile adds even when a value equals the
            // builtin, so an empty profile is obvious at a glance.
            let set_keys: Vec<&str> = crate::knobs::KEYS
                .iter()
                .copied()
                .filter(|k| st.model[key].get(k).is_some())
                .collect();
            if set_keys.is_empty() {
                println!("\n  {DIM}this profile sets no values{RESET}");
            }
        }
    }
    0
}

fn cmd_settings(args: &[String]) -> i32 {
    let mut st = settings::load();
    let mut origin = false;
    let mut scope: Option<SettingsScope> = None;
    let mut sets: Vec<(String, String)> = Vec::new();
    let mut unsets: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let val = |i: &mut usize| -> Option<String> {
            *i += 1;
            args.get(*i).cloned()
        };
        match a {
            "--show" | "-s" => {}
            "--origin" => origin = true,
            "--defaults" => {
                if matches!(scope, Some(SettingsScope::Model(_))) {
                    eprintln!("choose either --defaults or --model NAME, not both");
                    return 2;
                }
                scope = Some(SettingsScope::Defaults);
            }
            "--model" => {
                let Some(name) = val(&mut i) else {
                    eprintln!("--model needs a name");
                    return 2;
                };
                if matches!(scope, Some(SettingsScope::Defaults)) {
                    eprintln!("choose either --defaults or --model NAME, not both");
                    return 2;
                }
                scope = Some(SettingsScope::Model(name));
            }
            "--set" => {
                let Some(kv) = val(&mut i) else {
                    eprintln!("--set needs key=value");
                    return 2;
                };
                let Some((k, v)) = kv.split_once('=') else {
                    eprintln!("--set needs key=value, got '{kv}'");
                    return 2;
                };
                if k.trim().is_empty() {
                    eprintln!("--set needs a setting name before '='");
                    return 2;
                }
                sets.push((k.trim().to_string(), v.to_string()));
            }
            "--unset" => {
                let Some(k) = val(&mut i) else {
                    eprintln!("--unset needs a setting name");
                    return 2;
                };
                unsets.push(k.trim().to_string());
            }
            "--help" | "-h" => {
                help();
                return 0;
            }
            other => {
                eprintln!("settings: unknown option '{other}'");
                eprintln!("usage: localmodel settings [--show]");
                eprintln!(
                    "       localmodel settings (--defaults | --model NAME) --set key=value [--set ...] [--unset key]"
                );
                eprintln!("       localmodel settings (--defaults | --model NAME) --show --origin");
                return 2;
            }
        }
        i += 1;
    }

    let mutate = !sets.is_empty() || !unsets.is_empty();
    if mutate || origin {
        let Some(scope) = scope.clone() else {
            eprintln!("choose a scope: --model NAME or --defaults");
            return 2;
        };
        if mutate {
            return apply_settings(&mut st, &scope, &sets, &unsets);
        }
        return show_origins(&st, &scope);
    }
    if scope.is_some() {
        eprintln!("--model/--defaults only apply to --set, --unset and --show --origin");
        return 2;
    }

    print_settings(&st);
    0
}

fn print_settings(st: &settings::Settings) {
    let p = settings::path();
    println!("settings file: {}", p.display());
    match std::fs::read_to_string(&p) {
        Ok(text) => {
            println!();
            print!("{text}");
        }
        Err(_) => {
            println!("{DIM}(not created yet - built-in defaults are used){RESET}");
            let text = toml::to_string_pretty(st).unwrap_or_default();
            if !text.trim().is_empty() {
                println!();
                print!("{text}");
            }
        }
    }
    println!("\n{DIM}edit: localmodel hub (or settings --set)  ·  CLI flags override saved values{RESET}");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model with no file behind it: `build_args` only needs the name (for
    /// branch decisions) and the flags parsed from GGUF. A path that does not
    /// exist makes `jinja::caps_for_model` return None, which keeps argv
    /// deterministic on any machine.
    fn test_model(name: &str, layers: u32, moe: bool, nextn: bool) -> Model {
        Model {
            path: PathBuf::from(format!("/nonexistent/{name}.gguf")),
            name: format!("{name}.gguf"),
            size: 0,
            quant: "Q6_K".into(),
            layers,
            moe,
            experts_total: 0,
            experts_active: 0,
            nextn,
            meta: gguf::Meta::default(),
        }
    }

    fn test_cfg() -> Cfg {
        Cfg {
            ctx: 131072,
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
            kv_ram: false,
            split_mode: "layer".into(),
            tensor_split: None,
            log_verbosity: 4,
            vision: false,
            mmproj: None,
            alias: None,
            flash_attn: true,
            temp: 1.0,
            top_p: 0.95,
            host: "127.0.0.1".into(),
            ..Cfg::default()
        }
    }

    /// The argv contract. These strings are what the models actually run with
    /// today; the registry refactor must not change a single token.
    fn argv(m: &Model, draft: Option<&str>, c: &Cfg) -> String {
        let d = draft.map(PathBuf::from);
        // Mirror the real call site: the alias is the file stem, not the file.
        build_args(m, d.as_deref(), &models::stem(&m.name), 8080, c).join(" ")
    }

    #[test]
    fn argv_dense_with_builtin_mtp() {
        let m = test_model("Ornith-1.5-9B-Q6_K", 33, false, true);
        let mut c = test_cfg();
        c.ctx = 262144;
        c.draft_tokens = Some(2);
        c.extra = Some("--batch-size 1024 --ubatch-size 256".into());
        assert_eq!(
            argv(&m, None, &c),
            "-m /nonexistent/Ornith-1.5-9B-Q6_K.gguf --alias Ornith-1.5-9B-Q6_K \
             --ctx-size 262144 --flash-attn on --cache-type-k q8_0 \
             --cache-type-v q8_0 --temp 1 --top-p 0.95 --host 127.0.0.1 --port 8080 --jinja \
             --metrics --cache-reuse 256 --parallel 1 --log-verbosity 4 -ngl 99 --top-k 20 --min-p 0 \
             --spec-type draft-mtp --spec-draft-n-max 2 --batch-size 1024 --ubatch-size 256"
        );
    }

    #[test]
    fn argv_moe_with_builtin_mtp() {
        let m = test_model("Ornith-1.5-35B-A3B-Q6_K", 40, true, true);
        let c = test_cfg();
        assert_eq!(
            argv(&m, None, &c),
            "-m /nonexistent/Ornith-1.5-35B-A3B-Q6_K.gguf --alias Ornith-1.5-35B-A3B-Q6_K \
             --ctx-size 131072 --flash-attn on --cache-type-k q8_0 \
             --cache-type-v q8_0 --temp 1 --top-p 0.95 --host 127.0.0.1 --port 8080 --jinja \
             --metrics --cache-reuse 256 --batch-size 2048 --ubatch-size 512 --parallel 1 \
             --log-verbosity 4 -ngl 99 --n-cpu-moe 58 --spec-type draft-mtp --spec-draft-n-max 3"
        );
    }

    #[test]
    fn argv_nemotron_external_draft() {
        let m = test_model("NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB", 80, true, false);
        let mut c = test_cfg();
        c.ngl = Some(999);
        c.draft_tokens = Some(7);
        assert_eq!(
            argv(&m, Some("/nonexistent/nemo-LiLiCorr.gguf"), &c),
            "-m /nonexistent/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf \
             --alias NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB --ctx-size 131072 \
             --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --temp 1 --top-p 0.95 \
             --host 127.0.0.1 --port 8080 --jinja --metrics --cache-reuse 256 --batch-size 2048 \
             --ubatch-size 512 --parallel 1 --log-verbosity 4 -ngl 999 \
             --n-cpu-moe 58 --spec-type draft-dflash \
             --spec-draft-model /nonexistent/nemo-LiLiCorr.gguf --spec-draft-ngl 999 \
             --spec-draft-n-max 7"
        );
    }

    #[test]
    fn argv_dense_external_mtp_draft_and_vision() {
        let m = test_model("gemma-4-12b-it", 48, false, false);
        let mut c = test_cfg();
        c.ctx = 262144;
        c.draft_tokens = Some(4);
        c.draft_gpu = true;
        c.vision = true;
        c.mmproj = Some("/nonexistent/mmproj-F16.gguf".into());
        c.extra = Some("--batch-size 1024 --ubatch-size 256".into());
        assert_eq!(
            argv(&m, Some("/nonexistent/mtp-gemma-4-12b-it-Q8_0.gguf"), &c),
            "-m /nonexistent/gemma-4-12b-it.gguf --alias gemma-4-12b-it \
             --ctx-size 262144 --flash-attn on --cache-type-k q8_0 \
             --cache-type-v q8_0 --temp 1 --top-p 0.95 --host 127.0.0.1 --port 8080 --jinja \
             --mmproj /nonexistent/mmproj-F16.gguf --metrics --cache-reuse 256 --parallel 1 \
             --log-verbosity 4 -ngl 99 \
             --top-k 64 --min-p 0 --spec-type draft-mtp \
             --spec-draft-model /nonexistent/mtp-gemma-4-12b-it-Q8_0.gguf \
             --spec-draft-ngl 999 --spec-draft-n-max 4 --batch-size 1024 --ubatch-size 256"
        );
    }

    #[test]
    fn placement_knobs_reach_argv_without_extra() {
        // These used to be reachable only by hand-writing `extra`; they are
        // first-class settings now and must actually be emitted.
        let m = test_model("Ornith-1.5-35B-A3B-Q6_K", 40, true, false);
        let mut c = test_cfg();
        c.load_mode = Some("none".into());
        c.moe_cache_mib = Some(2048);
        c.cpu_moe = true;
        c.n_cpu_ffn = Some(4);
        c.override_tensor = Some("blk\\.1\\.ffn_up=CPU".into());
        c.lazy_mode = Some("off".into());
        c.numa = Some("isolate".into());
        c.no_op_offload = true;
        c.batch_size = Some(1024);
        c.ubatch_size = Some(256);
        c.parallel = Some(2);
        c.threads_batch = Some(8);
        c.prio = Some(1);
        c.cache_reuse = Some(0);
        c.draft_n_min = Some(1);
        c.draft_p_min = Some(0.5);
        c.top_k = Some(20);
        c.min_p = Some(0.05);
        c.repeat_penalty = Some(1.1);
        c.presence_penalty = Some(0.2);
        c.frequency_penalty = Some(0.3);
        c.seed = Some(42);
        c.api_key = Some("secret".into());
        c.timeout = Some(60);
        c.threads_http = Some(4);
        let a = argv(&m, None, &c);
        for needle in [
            "--load-mode none",
            "--moe-cache-mib 2048",
            "--cpu-moe",
            "--n-cpu-ffn 4",
            "--override-tensor blk\\.1\\.ffn_up=CPU",
            "--lazy-mode off",
            "--numa isolate",
            "--no-op-offload",
            "--batch-size 1024",
            "--ubatch-size 256",
            "--parallel 2",
            "--threads-batch 8",
            "--prio 1",
            "--cache-reuse 0",
            "--spec-draft-n-min 1",
            "--spec-draft-p-min 0.5",
            "--top-k 20",
            "--min-p 0.05",
            "--repeat-penalty 1.1",
            "--presence-penalty 0.2",
            "--frequency-penalty 0.3",
            "--seed 42",
            "--api-key secret",
            "--timeout 60",
            "--threads-http 4",
        ] {
            assert!(a.contains(needle), "missing `{needle}` in {a}");
        }
    }

    #[test]
    fn argv_dense_plain_no_draft() {
        let m = test_model("Ornith-1.5-9B-AD-Q8_0-Q6_K", 33, false, false);
        let mut c = test_cfg();
        c.ctx = 262144;
        c.extra = Some("--batch-size 1024 --ubatch-size 256".into());
        assert_eq!(
            argv(&m, None, &c),
            "-m /nonexistent/Ornith-1.5-9B-AD-Q8_0-Q6_K.gguf \
             --alias Ornith-1.5-9B-AD-Q8_0-Q6_K --ctx-size 262144 \
             --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --temp 1 --top-p 0.95 \
             --host 127.0.0.1 --port 8080 --jinja --metrics --cache-reuse 256 --parallel 1 \
             --log-verbosity 4 -ngl 99 \
             --top-k 20 --min-p 0 --batch-size 1024 --ubatch-size 256"
        );
    }

    #[test]
    fn profile_keys_match_the_argv_builder() {
        // Every knob the registry advertises must be one `build_args` reads,
        // and vice versa: a field nobody maps is a knob that does nothing.
        let m = test_model("probe", 1, false, false);
        for k in knobs::KNOBS {
            assert!(
                k.argv.is_empty() || knobs::KEYS.contains(&k.key),
                "{} has no argv mapping",
                k.key
            );
            let mut p = settings::Profile::default();
            let val: String = match k.kind {
                knobs::Kind::Bool => "off".into(),
                knobs::Kind::Num { min, .. } => min.to_string(),
                knobs::Kind::Float { .. } => "0.5".into(),
                knobs::Kind::Enum(v) => v[0].to_string(),
                knobs::Kind::Text => "x".into(),
                knobs::Kind::Path => "/tmp/x.gguf".into(),
                knobs::Kind::Flags => "--foo 1".into(),
            };
            p.set(k.key, &val).unwrap_or_else(|e| panic!("{}: {e}", k.key));
            let resolved = Cfg::resolve(&RunOpts::default(), &p);
            // Must not panic or drop an argument for any single knob.
            let a = build_args(&m, None, &m.name, 8080, &resolved);
            assert!(!a.is_empty());
        }
    }

    #[test]
    fn slash_target_routes_to_strata() {
        let a: Vec<String> = ["qwen36/ud-q4_k_xl", "--ctx", "131072"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let (f, s, rest) = strata_target(&a).expect("catalog target");
        assert_eq!(f, "qwen36");
        assert_eq!(s, "ud-q4_k_xl");
        assert_eq!(rest, vec!["--ctx".to_string(), "131072".to_string()]);

        let local: Vec<String> = ["Qwen3.8-27B-UD-Q3_K_XL.gguf"].iter().map(|s| s.to_string()).collect();
        assert!(strata_target(&local).is_none());
    }

    #[test]
    fn side_profiles_lists_only_linked_named_profiles() {
        use settings::Settings;
        let mut st = Settings::default();
        st.model.insert("agent-qwen38".into(), settings::Profile {
            applies_to: Some("Qwen3.8-27B-UD-Q3_K_XL".into()),
            ..Default::default()
        });
        st.model.insert("orphan".into(), settings::Profile {
            applies_to: None, ..Default::default()
        });
        st.model.insert("Qwen3.8-27B-UD-Q3_K_XL".into(), settings::Profile::default());
        // Build the list by hand: scanning ~/models made this test depend on
        // whatever the user happens to have downloaded.
        let ms = vec![test_model("Qwen3.8-27B-UD-Q3_K_XL", 64, false, false)];
        let profs = side_profiles(&st, &ms);
        assert_eq!(profs.len(), 1);
        assert_eq!(profs[0].0, "agent-qwen38");
    }

    #[test]
    fn warning_lines_stay_warnings() {
        // llama.cpp's auto-fit warning contains "failed" but is level W.
        let w = log_line_w("0.00.933.092 W common_fit_params: failed to fit params to free device memory", 68);
        assert!(w.contains(YELLOW), "{w}");
        assert!(!w.contains(RED), "{w}");
        let e = log_line_w("0.01.000.000 E srv load_model: failed to load model", 68);
        assert!(e.contains(RED), "{e}");
    }

    #[test]
    fn pack_settings_keys_match_both_spellings() {
        use settings::Settings;
        let mut st = Settings::default();
        st.model.insert(
            "qwen36/ud_q4_k_xl".into(),
            settings::Profile { extra: Some("--mtp off".into()), ..Default::default() },
        );
        // the hub writes the pack as qwen36/ud-q4_k_xl, a person as ..._...
        let key = settings::find_key(&st, "qwen36/ud-q4_k_xl").expect("same pack");
        assert_eq!(key, "qwen36/ud_q4_k_xl");
        assert_eq!(st.for_model(key).extra.as_deref(), Some("--mtp off"));
        assert!(settings::find_key(&st, "qwen36/ud-q6_k_xl").is_none());
    }

    #[test]
    fn strata_packs_parse_installed_listing() {
        let text = "qwen36-ud-q4_k_xl  present  qwen3.6-35b-a3b-ud-q4_k_xl\n\
                    qwen38-flash-next  absent  qwen3.8-flash-next\n\
                    garbage line without columns\n";
        let packs = strata_packs_from(text);
        assert_eq!(packs.len(), 1);
        assert_eq!(packs[0], ("qwen36".to_string(), "ud-q4_k_xl".to_string()));
    }

    /// Every line of a box must be exactly `cols` wide or narrower, and the
    /// top border must line up with the body and the bottom border. The old
    /// code was two cells too wide and one cell short on top.
    fn assert_box_width(card: &str, cols: usize) {
        for line in card.lines() {
            let w = vlen(line);
            // Strictly narrower than the terminal: a full-width line leaves
            // the terminal in wrap-pending state and shifts the next row.
            assert!(w < cols, "line is {w} cells, terminal is {cols}: {line:?}");
        }
        let mut lines = card.lines();
        let top = lines.next().expect("top border");
        let rest: Vec<&str> = lines.collect();
        let body = rest.iter().find(|l| l.starts_with('│')).expect("body row");
        assert_eq!(vlen(top), vlen(body), "top border vs body:\n{top}\n{body}");
    }

    #[test]
    fn boxed_card_fits_the_terminal() {
        let body = vec![
            format!("  {}", "x".repeat(60)),
            "  short".to_string(),
            format!("  {}", "y".repeat(10)),
        ];
        for cols in [40usize, 60, 80, 100, 120] {
            let card = boxed("title", &body, "footer", cols);
            assert_box_width(&card, cols);
            let top = card.lines().next().unwrap();
            let body_row = card.lines().find(|l| l.starts_with('│')).unwrap();
            assert_eq!(vlen(top), vlen(body_row));
        }
    }

    #[test]
    fn boxed_long_title_is_clipped_to_fit() {
        let title = "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-with-a-very-long-name";
        let card = boxed(title, &["  body".to_string()], "", 50);
        assert_box_width(&card, 50);
    }

    #[test]
    fn paint_erases_every_line_and_clips_to_the_terminal() {
        let out = paint_screen("aaa\nbbb\n", 10);
        assert_eq!(out.matches("\x1b[2K").count(), 2, "{out:?}");
        assert!(out.contains("aaa\r\n"), "{out:?}");
        // cols - 1 cells at most: the last column stays free so the terminal
        // never auto-wraps and shifts the following row.
        let wide = paint_screen(&"x".repeat(30), 10);
        assert!(!wide.contains(&"x".repeat(10)), "{wide:?}");
        assert!(wide.contains(&"x".repeat(9)), "{wide:?}");
    }

    #[test]
    fn live_card_shows_draft_acceptance_and_mmap_warning() {
        let m = test_model("nemotron", 88, true, false);
        let hw = fit::Hardware {
            ram_total_gb: 64.0,
            ram_avail_gb: 32.0,
            vram_total_gb: 24.0,
            disk_free_gb: 2000.0,
            gpus: vec![("GPU0".into(), 24.0)],
        };
        let cfg = test_cfg();
        let plan = mem::plan(&m, None, &cfg, &hw);
        let info = Info {
            alias: "nemotron".into(),
            port: 8080,
            pid: 0,
            ctx: 131072,
            ctk: "q8_0".into(),
            ctv: "q8_0".into(),
        };
        let ram = RamText {
            system: "8/64 GiB".into(),
            server: "1 GiB rss".into(),
            short: "1 GiB".into(),
        };
        let mut s = Stats::default();
        s.last_draft_accept = Some(0.62);
        s.last_draft_mean = Some(2.4);
        s.last_draft_per_pos = vec![0.8, 0.6];
        let live = telemetry::LoadInfo { mmap_cpu_experts: true, ..Default::default() };
        let screen = render_screen(
            &m,
            &info,
            &s,
            false,
            &[],
            &ram,
            Path::new("/tmp/nonexistent.log"),
            false,
            0,
            None,
            false,
            100,
            50,
            &cfg,
            &hw,
            &plan,
            Some(&live),
        );
        assert!(screen.contains("Draft accept"), "{screen}");
        assert!(screen.contains("62%"), "{screen}");
        assert!(screen.contains("from SSD"), "{screen}");
        assert_box_width(&screen, 100);

        // Before the first request there is no acceptance number yet, but the
        // row must still be visible when the model has a draft/MTP head.
        let empty = Stats::default();
        let idle = render_screen(
            &m,
            &info,
            &empty,
            true,
            &[],
            &ram,
            Path::new("/tmp/nonexistent.log"),
            false,
            0,
            None,
            false,
            100,
            50,
            &cfg,
            &hw,
            &plan,
            Some(&live),
        );
        assert!(idle.contains("waiting for the first request"), "{idle}");
    }
}
