//! `localmodel hub`: the one interactive screen.
//!
//! Before this file the launch settings lived in eight places (the struct,
//! `is_empty`, `overlay`, `Cfg`, `Cfg::resolve`, `build_args`, the dialoguer
//! editor and the dashboard table). `knobs.rs` fixed the data model and
//! `resolve.rs` added provenance; the hub is the screen that consumes both:
//!
//!   * left panel  — downloaded GGUF models, installed Strata packs and the
//!                   HuggingFace "add" entry;
//!   * right panel — only the knobs that can affect the selected model
//!                   (`Applies`), each with its effective value and the layer
//!                   it came from (`Origin`), plus a live VRAM estimate from
//!                   `fit::for_gguf`.
//!
//! It reuses the dashboard's primitives (`boxed`, colours, `screen_size`) and
//! the same crossterm setup, so there is no second rendering stack. Editing
//! writes into a `Settings` copy and `s` goes through `settings::save`, which
//! touches only the changed keys and keeps comments.
//!
//! With no TTY the command prints the same list and exits, so scripts keep a
//! non-interactive path (`settings --show/--set`, `run --dry-run`).

use std::collections::BTreeMap;
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;

use crossterm::cursor::{Hide, Show};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};

use crate::config;
use crate::fit;
use crate::gguf;
use crate::knobs::{self, Caps, Group, Kind, Profile};
use crate::models::{self, Model};
use crate::resolve::{Origin, Source};
use crate::settings;
use crate::{RunOpts, BLUE, BOLD, CYAN, DIM, GREEN, MAGENTA, RED, RESET, YELLOW};

pub fn cmd_hub(_args: &[String]) -> i32 {
    let tty = io::stdin().is_terminal() && io::stdout().is_terminal();
    let mut hub = Hub::new();
    if !tty {
        hub.print_list();
        return 0;
    }
    enter();
    let code = hub.run();
    leave();
    code
}

fn enter() {
    let mut out = io::stdout();
    let _ = terminal::enable_raw_mode();
    let _ = execute!(out, EnterAlternateScreen, Hide);
    let _ = out.flush();
}

fn leave() {
    let _ = terminal::disable_raw_mode();
    let _ = execute!(io::stdout(), LeaveAlternateScreen, Show);
    let _ = io::stdout().flush();
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq, Eq)]
enum Entry {
    Model(usize),
    Pack(String, String),
    Add,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Focus {
    Models,
    Settings,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Overlay {
    None,
    Help,
    Argv,
}

struct Edit {
    buf: String,
    /// The first typed character replaces the pre-filled value.
    fresh: bool,
}

struct Prompt {
    buf: String,
}

enum Row {
    Header(Group),
    Knob(&'static knobs::Knob),
}

struct DraftInfo {
    layers: Vec<gguf::KvLayer>,
    swa: u32,
    weights: u64,
    ctk: String,
    ctv: String,
}

struct EstimateView {
    line: String,
    detail: String,
    ctx_max: u32,
    verdict: fit::Verdict,
}

struct ArgvView {
    lines: Vec<String>,
    origins: Vec<Origin>,
    warnings: Vec<String>,
}

enum Launch {
    Run(String),
    Strata(String, String),
}

enum Step {
    Continue,
    Quit,
    Add(String),
    Launch(Launch),
}

struct Hub {
    ms: Vec<Model>,
    packs: Vec<(String, String)>,
    entries: Vec<Entry>,
    running: Vec<PathBuf>,

    /// Working copy that the screen edits, and the on-disk snapshot used to
    /// tell whether anything is unsaved.
    work: settings::Settings,
    disk: settings::Settings,

    hw: fit::Hardware,
    sel: usize,
    focus: Focus,
    row: usize,
    edit: Option<Edit>,
    prompt: Option<Prompt>,
    confirm: Option<usize>,
    overlay: Overlay,
    msg: Option<(String, bool)>,

    est: Option<EstimateView>,
    draft_cache: BTreeMap<PathBuf, (gguf::Meta, u64)>,
    argv: Option<ArgvView>,
}

impl Hub {
    fn new() -> Self {
        let dir = config::models_dir();
        let hw = fit::Hardware::probe(&dir);
        let work = settings::load();
        let disk = work.clone();
        let mut h = Self {
            ms: Vec::new(),
            packs: Vec::new(),
            entries: Vec::new(),
            running: Vec::new(),
            work,
            disk,
            hw,
            sel: 0,
            focus: Focus::Models,
            row: 0,
            edit: None,
            prompt: None,
            confirm: None,
            overlay: Overlay::None,
            msg: None,
            est: None,
            draft_cache: BTreeMap::new(),
            argv: None,
        };
        h.reload();
        h
    }

    fn reload(&mut self) {
        let dir = config::models_dir();
        self.ms = models::scan(&dir);
        self.packs = crate::strata_packs();
        self.running = {
            let servers = crate::server::list_servers(&dir.to_string_lossy());
            self.ms
                .iter()
                .filter(|m| {
                    let p = m.path.display().to_string();
                    servers.iter().any(|s| s.cmd.contains(&p))
                })
                .map(|m| m.path.clone())
                .collect()
        };
        self.entries = (0..self.ms.len()).map(Entry::Model).collect();
        for (f, s) in &self.packs {
            self.entries.push(Entry::Pack(f.clone(), s.clone()));
        }
        self.entries.push(Entry::Add);
        if self.sel >= self.entries.len() {
            self.sel = self.entries.len().saturating_sub(1);
        }
        self.row = first_knob_row(&self.rows());
        self.recompute();
    }

    fn print_list(&self) {
        let dir = config::models_dir();
        println!("models in {}:", dir.display());
        for m in &self.ms {
            let draft = if m.nextn {
                "  +mtp"
            } else if models::find_draft(&m.path).is_some() {
                "  +draft"
            } else {
                ""
            };
            let kind = if m.moe { "MoE" } else { "Dense" };
            println!(
                "  {:<46} {:<5} {:>7}  {}{}",
                m.name,
                kind,
                models::human(m.size),
                m.quant,
                draft
            );
        }
        if !self.packs.is_empty() {
            println!("\nstrata packs:");
            for (f, s) in &self.packs {
                println!("  {f}/{s}");
            }
        }
        println!("\nadd a model:   localmodel add <huggingface url>");
        println!("edit settings: localmodel settings --show --origin");
        println!("launch one:    localmodel run <model>");
    }

    // -- current selection ------------------------------------------------

    fn current(&self) -> Option<&Entry> {
        self.entries.get(self.sel)
    }

    fn current_model(&self) -> Option<&Model> {
        match self.current()? {
            Entry::Model(i) => self.ms.get(*i),
            _ => None,
        }
    }

    fn scope_key(&self) -> Option<String> {
        match self.current()? {
            Entry::Model(i) => self.ms.get(*i).map(|m| models::stem(&m.name)),
            Entry::Pack(f, s) => Some(format!("{f}/{s}")),
            Entry::Add => None,
        }
    }

    /// The key actually stored in `[model.*]`, preserving hand-written case.
    fn profile_key(&self) -> Option<String> {
        let scope = self.scope_key()?;
        Some(
            settings::find_key(&self.work, &scope)
                .map(str::to_string)
                .unwrap_or(scope),
        )
    }

    fn caps(&self) -> Caps {
        match self.current_model() {
            Some(m) => {
                let prof = self.profile();
                Caps {
                    moe: m.moe,
                    nextn: m.nextn,
                    draft: m.nextn || models::find_draft(&m.path).is_some(),
                    vision: models::find_mmproj(&m.path).is_some() || prof.mmproj.is_some(),
                }
            }
            None => Caps::default(),
        }
    }

    fn rows(&self) -> Vec<Row> {
        let caps = self.caps();
        let mut out = Vec::new();
        for g in Group::ORDER {
            let ks: Vec<&'static knobs::Knob> = knobs::KNOBS
                .iter()
                .filter(|k| k.group == g && k.applies.check(&caps))
                .collect();
            if ks.is_empty() {
                continue;
            }
            out.push(Row::Header(g));
            for k in ks {
                out.push(Row::Knob(k));
            }
        }
        out
    }

    fn current_knob(&self) -> Option<&'static knobs::Knob> {
        match self.rows().get(self.row) {
            Some(Row::Knob(k)) => Some(k),
            _ => None,
        }
    }

    fn profile(&self) -> Profile {
        self.profile_key()
            .and_then(|k| self.work.model.get(&k).cloned())
            .unwrap_or_default()
    }

    fn effective(&self) -> Profile {
        self.work.defaults.overlay(&self.profile())
    }

    fn resolved(&self) -> (crate::Cfg, Vec<Origin>) {
        let key = self.profile_key();
        let prof = key.as_ref().and_then(|k| self.work.model.get(k));
        crate::Cfg::explain(&RunOpts::default(), &self.work.defaults, prof)
    }

    fn dirty(&self) -> bool {
        self.work.defaults != self.disk.defaults || self.work.model != self.disk.model
    }

    // -- editing ----------------------------------------------------------

    fn set_key(&mut self, key: &str, value: &str) {
        let Some(scope) = self.scope_key() else { return };
        let canon = settings::find_key(&self.work, &scope)
            .map(str::to_string)
            .unwrap_or(scope);
        let mut p = self.work.model.get(&canon).cloned().unwrap_or_default();
        if let Err(e) = p.set(key, value) {
            self.msg = Some((e, true));
            return;
        }
        if p.is_empty() {
            self.work.model.remove(&canon);
        } else {
            self.work.model.insert(canon, p);
        }
        self.msg = None;
        self.recompute();
    }

    fn apply_text(&mut self, k: &'static knobs::Knob, raw: &str) {
        let t = raw.trim();
        if t.is_empty() || t == "-" {
            self.set_key(k.key, "-");
            return;
        }
        match k.kind {
            Kind::Num { min, max, .. } => match t.parse::<i64>() {
                Ok(v) if v >= min && v <= max => self.set_key(k.key, t),
                Ok(_) => self.msg = Some((format!("{} must be {min}..{max}", k.label), true)),
                Err(_) => self.msg = Some((format!("{}: expected a whole number", k.label), true)),
            },
            Kind::Float { min, max } => match t.parse::<f64>() {
                Ok(v) if v >= min && v <= max => self.set_key(k.key, t),
                Ok(_) => self.msg = Some((format!("{} must be {min}..{max}", k.label), true)),
                Err(_) => self.msg = Some((format!("{}: expected a number", k.label), true)),
            },
            Kind::Enum(vals) => {
                if vals.contains(&t) {
                    self.set_key(k.key, t);
                } else {
                    self.msg = Some((format!("{}: one of {}", k.label, vals.join(", ")), true));
                }
            }
            Kind::Path => {
                if !std::path::Path::new(t).exists() {
                    self.msg = Some((format!("{}: no such file: {t}", k.label), true));
                    return;
                }
                self.set_key(k.key, t);
            }
            _ => self.set_key(k.key, t),
        }
    }

    fn begin_edit(&mut self) {
        let Some(k) = self.current_knob() else { return };
        let cur = self.effective().get(k.key).unwrap_or_default();
        self.edit = Some(Edit { buf: cur, fresh: true });
    }

    fn activate(&mut self) {
        let Some(k) = self.current_knob() else { return };
        match k.kind {
            Kind::Bool => {
                let cur = self.effective().get(k.key).unwrap_or_else(|| "on".to_string());
                let next = if cur == "on" { "off" } else { "on" };
                self.set_key(k.key, next);
            }
            Kind::Enum(vals) => {
                let cur = self.effective().get(k.key).unwrap_or_default();
                let next = match vals.iter().position(|v| *v == cur) {
                    Some(i) => vals[(i + 1) % vals.len()],
                    None => vals[0],
                };
                self.set_key(k.key, next);
            }
            _ => self.begin_edit(),
        }
    }

    fn move_row(&mut self, dir: isize) {
        let rows = self.rows();
        if rows.is_empty() {
            return;
        }
        let mut i = self.row as isize;
        for _ in 0..rows.len() {
            i += dir;
            if i < 0 {
                i = 0;
                break;
            }
            if i as usize >= rows.len() {
                i = rows.len() as isize - 1;
                break;
            }
            if matches!(rows[i as usize], Row::Knob(_)) {
                break;
            }
        }
        self.row = i as usize;
    }

    // -- estimates --------------------------------------------------------

    fn draft_info(&mut self, m: &Model, cfg: &crate::Cfg) -> Option<DraftInfo> {
        if m.nextn {
            return Some(DraftInfo {
                layers: m.meta.kv.draft.clone(),
                swa: m.meta.kv.swa_window,
                weights: 0,
                ctk: cfg.ctk.clone(),
                ctv: cfg.ctv.clone(),
            });
        }
        let path = models::find_draft(&m.path)?;
        let (meta, bytes) = match self.draft_cache.get(&path).cloned() {
            Some(x) => x,
            None => {
                let meta = gguf::parse(&path);
                let bytes = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                self.draft_cache.insert(path.clone(), (meta.clone(), bytes));
                (meta, bytes)
            }
        };
        Some(DraftInfo {
            layers: meta.kv.main.clone(),
            swa: meta.kv.swa_window,
            weights: bytes,
            ctk: cfg.ctk.clone(),
            ctv: cfg.ctv.clone(),
        })
    }

    fn recompute(&mut self) {
        self.est = None;
        let Some(m) = self.current_model().cloned() else { return };
        if !m.meta.kv.has_kv() {
            return;
        }
        let (cfg, _) = self.resolved();
        let draft = self.draft_info(&m, &cfg);
        let dref = draft.as_ref().map(|d| fit::DraftKv {
            layers: &d.layers,
            swa_window: d.swa,
            weights_bytes: d.weights,
            ctk: &d.ctk,
            ctv: &d.ctv,
        });
        let e = fit::for_gguf(&m.meta, m.size, &self.hw, cfg.ctx, &cfg.ctk, &cfg.ctv, dref);
        let ctx_max = fit::suggest_ctx(&m.meta, m.size, &self.hw, &cfg.ctk, &cfg.ctv, dref);
        let spare = e.room_gb - e.need_vram_gb;
        let slack = if spare >= 0.0 {
            format!("{spare:.1} GB spare")
        } else {
            format!("{:.1} GB short", -spare)
        };
        self.est = Some(EstimateView {
            line: format!(
                "VRAM {:.1} / {:.1} GB  {}",
                e.need_vram_gb,
                e.room_gb,
                e.verdict.label()
            ),
            detail: format!(
                "{} tok · {:.1} weights + {:.1} KV + {:.1} overhead · {:.0} KB/token · {slack}",
                e.ctx,
                e.weights_gb,
                e.kv_main_gb + e.kv_draft_gb,
                e.overhead_gb,
                e.kv_gb_per_token * 1024.0 * 1024.0,
            ),
            ctx_max,
            verdict: e.verdict,
        });
    }

    // -- actions ----------------------------------------------------------

    fn save(&mut self) -> bool {
        if !self.dirty() {
            self.msg = Some(("no changes to save".into(), false));
            return true;
        }
        match settings::save(&mut self.work) {
            Ok(p) => {
                self.disk = self.work.clone();
                self.msg = Some((format!("saved {}", p.display()), false));
                true
            }
            Err(e) => {
                self.msg = Some((format!("save failed: {e}"), true));
                false
            }
        }
    }

    fn delete(&mut self) {
        let Some(i) = self.confirm.take() else { return };
        let Some(m) = self.ms.get(i) else { return };
        let name = m.name.clone();
        let size = m.size;
        let path = m.path.clone();
        match fs::remove_file(&path) {
            Ok(()) => {
                self.msg = Some((format!("deleted {name} (freed {})", models::human(size)), false));
                self.reload();
            }
            Err(e) => self.msg = Some((format!("could not delete {name}: {e}"), true)),
        }
    }

    fn build_argv(&mut self) {
        let (cfg, origins) = self.resolved();
        match self.current().cloned() {
            Some(Entry::Model(i)) => {
                let m = self.ms[i].clone();
                let alias = cfg
                    .alias
                    .clone()
                    .unwrap_or_else(|| models::stem(&m.name));
                let port = cfg.port.unwrap_or_else(|| crate::server::free_port(8080));
                let draft = models::find_draft(&m.path);
                let args = crate::build_args(&m, draft.as_deref(), &alias, port, &cfg);
                let bin = config::server_bin();
                let mut lines = Vec::new();
                lines.push(format!("{DIM}model{RESET}    {}", m.name));
                lines.push(format!(
                    "{DIM}profile{RESET}  [model.\"{}\"] over [defaults]",
                    self.profile_key().unwrap_or_else(|| models::stem(&m.name))
                ));
                lines.push(String::new());
                for l in wrap(&crate::shell_cmd(&bin, &args), 76) {
                    lines.push(l);
                }
                let active: Vec<&str> = origins.iter().map(|o| o.key).collect();
                let warnings = crate::validate::check(
                    &args,
                    cfg.extra.as_deref().unwrap_or(""),
                    &bin,
                    &active,
                );
                self.argv = Some(ArgvView { lines, origins, warnings });
            }
            Some(Entry::Pack(f, s)) => {
                self.argv = Some(ArgvView {
                    lines: vec![
                        format!("{DIM}strata{RESET}   {f}/{s}"),
                        String::new(),
                        "argv for a Strata pack is assembled by the engine;".into(),
                        format!("`localmodel run {f} {s} --dry-run` prints it."),
                    ],
                    origins,
                    warnings: Vec::new(),
                });
            }
            _ => self.msg = Some(("choose a model first".into(), true)),
        }
    }

    // -- input ------------------------------------------------------------

    fn handle(&mut self, k: KeyEvent) -> Step {
        if k.modifiers.contains(KeyModifiers::CONTROL)
            && matches!(k.code, KeyCode::Char('c') | KeyCode::Char('d'))
        {
            return Step::Quit;
        }
        // Enter is CR; a pty (or a test harness) may hand us LF instead, and
        // crossterm decodes 0x0a as Ctrl-J. Both mean "confirm" here, while a
        // plain `j` stays a movement key.
        let enter = matches!(k.code, KeyCode::Char('\n') | KeyCode::Char('\r'))
            || (k.modifiers.contains(KeyModifiers::CONTROL)
                && k.code == KeyCode::Char('j'));
        let k = if enter {
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)
        } else {
            k
        };

        if self.prompt.is_some() {
            return self.handle_prompt(k);
        }
        if self.edit.is_some() {
            match k.code {
                KeyCode::Enter => {
                    let buf = self.edit.take().map(|e| e.buf).unwrap_or_default();
                    if let Some(knob) = self.current_knob() {
                        self.apply_text(knob, &buf);
                    }
                }
                KeyCode::Esc => self.edit = None,
                KeyCode::Backspace => {
                    if let Some(e) = self.edit.as_mut() {
                        if e.fresh {
                            e.buf.clear();
                            e.fresh = false;
                        } else {
                            e.buf.pop();
                        }
                    }
                }
                KeyCode::Char(c) => {
                    if let Some(e) = self.edit.as_mut() {
                        if e.fresh {
                            e.buf.clear();
                            e.fresh = false;
                        }
                        e.buf.push(c);
                    }
                }
                _ => {}
            }
            return Step::Continue;
        }

        if let Some(i) = self.confirm {
            if matches!(k.code, KeyCode::Char('y') | KeyCode::Char('Y')) {
                self.confirm = Some(i);
                self.delete();
            } else {
                self.confirm = None;
                self.msg = Some(("delete cancelled".into(), false));
            }
            return Step::Continue;
        }

        if self.overlay != Overlay::None {
            self.overlay = Overlay::None;
            return Step::Continue;
        }

        match k.code {
            KeyCode::Char('?') | KeyCode::Char('h') => self.overlay = Overlay::Help,
            KeyCode::Char('q') => return Step::Quit,
            KeyCode::Esc => {
                if self.focus == Focus::Settings {
                    self.focus = Focus::Models;
                } else {
                    return Step::Quit;
                }
            }
            KeyCode::Tab | KeyCode::BackTab => {
                self.focus = if self.focus == Focus::Models {
                    Focus::Settings
                } else {
                    Focus::Models
                };
            }
            KeyCode::Char('s') => {
                self.save();
            }
            KeyCode::Char('d') => self.build_argv(),
            KeyCode::Char('L') => {
                let Some(target) = self.launch_target() else {
                    self.msg = Some(("choose a model first".into(), true));
                    return Step::Continue;
                };
                if self.dirty() && !self.save() {
                    return Step::Continue;
                }
                return Step::Launch(target);
            }
            KeyCode::Char('x') => match self.current().cloned() {
                Some(Entry::Model(i)) => self.confirm = Some(i),
                _ => self.msg = Some(("only downloaded models can be deleted".into(), true)),
            },
            KeyCode::Up | KeyCode::Char('k') => match self.focus {
                Focus::Models => self.move_sel(-1),
                Focus::Settings => self.move_row(-1),
            },
            KeyCode::Down | KeyCode::Char('j') => match self.focus {
                Focus::Models => self.move_sel(1),
                Focus::Settings => self.move_row(1),
            },
            KeyCode::Char('-') if self.focus == Focus::Settings => {
                if let Some(k) = self.current_knob() {
                    self.set_key(k.key, "-");
                }
            }
            KeyCode::Char(' ') if self.focus == Focus::Settings => self.activate(),
            KeyCode::Enter => match self.focus {
                Focus::Models => match self.current().cloned() {
                    Some(Entry::Add) => {
                        self.prompt = Some(Prompt { buf: String::new() });
                    }
                    _ => self.focus = Focus::Settings,
                },
                Focus::Settings => self.activate(),
            },
            _ => {}
        }
        Step::Continue
    }

    fn handle_prompt(&mut self, k: KeyEvent) -> Step {
        match k.code {
            KeyCode::Esc => self.prompt = None,
            KeyCode::Enter => {
                let url = self
                    .prompt
                    .take()
                    .map(|p| p.buf.trim().to_string())
                    .unwrap_or_default();
                if url.is_empty() {
                    self.msg = Some(("no url given".into(), true));
                } else {
                    return Step::Add(url);
                }
            }
            KeyCode::Backspace => {
                if let Some(p) = self.prompt.as_mut() {
                    p.buf.pop();
                }
            }
            KeyCode::Char(c) => {
                if let Some(p) = self.prompt.as_mut() {
                    p.buf.push(c);
                }
            }
            _ => {}
        }
        Step::Continue
    }

    fn move_sel(&mut self, dir: isize) {
        if self.entries.is_empty() {
            return;
        }
        let mut i = self.sel as isize + dir;
        if i < 0 {
            i = 0;
        }
        if i as usize >= self.entries.len() {
            i = self.entries.len() as isize - 1;
        }
        if i as usize != self.sel {
            self.sel = i as usize;
            self.row = first_knob_row(&self.rows());
            self.recompute();
        }
    }

    fn launch_target(&self) -> Option<Launch> {
        match self.current()? {
            Entry::Model(i) => self.ms.get(*i).map(|m| Launch::Run(m.name.clone())),
            Entry::Pack(f, s) => Some(Launch::Strata(f.clone(), s.clone())),
            Entry::Add => None,
        }
    }

    // -- main loop --------------------------------------------------------

    fn run(&mut self) -> i32 {
        loop {
            self.draw();
            let ev = match event::read() {
                Ok(e) => e,
                Err(_) => return 1,
            };
            let Event::Key(k) = ev else { continue };
            match self.handle(k) {
                Step::Continue => {}
                Step::Quit => return 0,
                Step::Launch(target) => {
                    leave();
                    let code = match target {
                        Launch::Run(name) => crate::cmd_run(&[name]),
                        Launch::Strata(f, s) => crate::cmd_strata(&[f, s]),
                    };
                    enter();
                    self.reload();
                    self.msg = Some((format!("exited with code {code}"), code != 0));
                }
                Step::Add(url) => {
                    leave();
                    let code = crate::cmd_add(&[url]);
                    enter();
                    self.reload();
                    self.msg = Some((format!("add exited with code {code}"), code != 0));
                }
            }
        }
    }

    // -- drawing ----------------------------------------------------------

    fn draw(&self) {
        // A pty without a size reports 0x0; keep the layout usable instead of
        // squeezing every panel down to its 10-cell minimum.
        let (cols, rows) = crate::screen_size();
        let cols = cols.max(40);
        let rows = rows.max(10);
        let screen = if self.overlay == Overlay::Help {
            self.render_help(cols)
        } else if self.overlay == Overlay::Argv {
            self.render_argv(cols)
        } else if self.prompt.is_some() {
            self.render_prompt(cols)
        } else {
            self.render_panels(cols, rows)
        };
        let mut out = io::stdout();
        let _ = write!(out, "\x1b[H{}\x1b[J", screen.replace('\n', "\r\n"));
        let _ = out.flush();
    }

    fn render_panels(&self, cols: usize, rows: usize) -> String {
        let h = rows.saturating_sub(1).max(6);
        let left_w = ((cols * 34) / 100)
            .max(18)
            .min(46)
            .min(cols.saturating_sub(20))
            .max(12);
        let right_w = cols.saturating_sub(left_w).max(12);

        let left_body = self.left_body(left_w.saturating_sub(2), h.saturating_sub(3));
        let left_footer = format!("{} · Enter · x", "↑↓");
        let left = panel(
            "models",
            &left_body,
            &left_footer,
            left_w,
            h,
            self.focus == Focus::Models,
            CYAN,
        );

        let right_body = self.right_body(right_w.saturating_sub(2), h.saturating_sub(3));
        let right_footer = if self.edit.is_some() {
            match self.current_knob() {
                Some(k) => {
                    let bounds = k.kind.bounds().unwrap_or_default();
                    if bounds.is_empty() {
                        format!("{} · Enter apply · Esc cancel", k.kind.hint())
                    } else {
                        format!("{} ({bounds}) · Enter apply · Esc cancel", k.kind.hint())
                    }
                }
                None => "Enter apply · Esc cancel".to_string(),
            }
        } else if self.focus == Focus::Settings {
            self.current_knob()
                .map(|k| k.help.to_string())
                .unwrap_or_else(|| "↑↓ knob · Enter edit · - reset · Tab panel".to_string())
        } else {
            "Enter opens this panel".to_string()
        };
        let right_title = match self.current() {
            Some(Entry::Model(i)) => self
                .ms
                .get(*i)
                .map(|m| models::stem(&m.name))
                .unwrap_or_else(|| "model".into()),
            Some(Entry::Pack(f, s)) => format!("{f}/{s}"),
            Some(Entry::Add) => "add from HuggingFace".into(),
            None => "model".into(),
        };
        let right = panel(
            &right_title,
            &right_body,
            &right_footer,
            right_w,
            h,
            self.focus == Focus::Settings,
            CYAN,
        );

        let mut out = String::new();
        for i in 0..h {
            out.push_str(&left[i]);
            out.push_str(&right[i]);
            out.push('\n');
        }
        out.push_str(&self.status_line(cols));
        out
    }

    fn left_body(&self, width: usize, height: usize) -> Vec<String> {
        let mut lines: Vec<(String, Option<usize>)> = Vec::new();
        lines.push((format!("{DIM}MODELS{RESET}"), None));
        let mut last_kind = 0u8;
        for (i, e) in self.entries.iter().enumerate() {
            let kind = match e {
                Entry::Model(_) => 1u8,
                Entry::Pack(..) => 2u8,
                Entry::Add => 3u8,
            };
            if kind != last_kind {
                if kind == 2 {
                    lines.push((String::new(), None));
                    lines.push((format!("{DIM}STRATA PACKS{RESET}"), None));
                } else if kind == 3 {
                    lines.push((String::new(), None));
                }
                last_kind = kind;
            }
            let line = match e {
                Entry::Model(mi) => {
                    let m = &self.ms[*mi];
                    let dot = if self.running.iter().any(|p| *p == m.path) {
                        format!("{GREEN}●{RESET}")
                    } else {
                        " ".to_string()
                    };
                    let badge = if m.nextn {
                        " +mtp"
                    } else if models::find_draft(&m.path).is_some() {
                        " +draft"
                    } else if m.moe {
                        " MoE"
                    } else {
                        ""
                    };
                    let text = crate::truncate(&m.name, width.saturating_sub(4));
                    format!(" {dot} {text}{DIM}{badge}{RESET}")
                }
                Entry::Pack(f, s) => format!("   {DIM}{f}/{s}{RESET}"),
                Entry::Add => format!(" {GREEN}+ add from HuggingFace{RESET}"),
            };
            lines.push((line, Some(i)));
        }

        let pos = lines
            .iter()
            .position(|(_, i)| *i == Some(self.sel))
            .unwrap_or(0);
        let start = window_start(lines.len(), pos, height);
        lines
            .iter()
            .skip(start)
            .take(height)
            .map(|(l, _)| l.clone())
            .collect()
    }

    fn right_body(&self, width: usize, height: usize) -> Vec<String> {
        let mut header: Vec<String> = Vec::new();
        match self.current().cloned() {
            Some(Entry::Model(i)) => {
                let m = &self.ms[i];
                let kind = if m.moe { "MoE" } else { "dense" };
                let mut traits = Vec::new();
                if m.nextn {
                    traits.push("MTP".to_string());
                } else if models::find_draft(&m.path).is_some() {
                    traits.push("external draft".to_string());
                }
                if m.meta.nextn_layers > 0 {
                    traits.push(format!("{} nextn", m.meta.nextn_layers));
                }
                if models::find_mmproj(&m.path).is_some() {
                    traits.push("vision".to_string());
                }
                let arch = if m.meta.arch.is_empty() {
                    String::new()
                } else {
                    format!(" · {}", m.meta.arch)
                };
                let extra = if traits.is_empty() {
                    String::new()
                } else {
                    format!(" · {}", traits.join(", "))
                };
                header.push(format!(
                    "{DIM}{} {}{} · {}L{} · {}{RESET}",
                    models::human(m.size),
                    m.quant,
                    arch,
                    m.layers,
                    extra,
                    kind,
                ));
                if let Some(e) = &self.est {
                    let color = e.verdict.color();
                    header.push(format!("{color}{}{RESET}", e.line));
                    header.push(format!("{DIM}{}{RESET}", e.detail));
                    if e.ctx_max > 0 {
                        header.push(format!(
                            "{DIM}ctx max for this card:{RESET} {}",
                            e.ctx_max
                        ));
                    } else {
                        header.push(format!("{RED}ctx max for this card: none fits{RESET}"));
                    }
                } else {
                    header.push(format!(
                        "{YELLOW}no GGUF KV metadata: cannot estimate VRAM{RESET}"
                    ));
                }
                header.push(String::new());
            }
            Some(Entry::Pack(f, s)) => {
                header.push(format!("{DIM}Strata hub pack {f}/{s}{RESET}"));
                header.push(format!("{DIM}launched by the strata engine{RESET}"));
                header.push(String::new());
                header.push(String::new());
            }
            Some(Entry::Add) => {
                header.push(format!("{DIM}paste a HuggingFace URL, fetch, then configure{RESET}"));
                header.push(String::new());
                header.push(String::new());
                header.push(String::new());
            }
            None => {
                header.push(String::new());
                header.push(String::new());
                header.push(String::new());
                header.push(String::new());
            }
        }

        let rows = self.rows();
        let avail = height.saturating_sub(header.len()).max(1);
        let start = window_start(rows.len(), self.row, avail);
        let (_, origins) = self.resolved();
        let mut body = header;
        for (off, r) in rows.iter().skip(start).take(avail).enumerate() {
            let idx = start + off;
            body.push(self.row_line(r, idx, &origins, width));
        }
        body
    }

    fn row_line(&self, r: &Row, idx: usize, origins: &[Origin], _width: usize) -> String {
        match r {
            Row::Header(g) => format!("{CYAN}{BOLD}{}{RESET}", g.title()),
            Row::Knob(k) => {
                let selected = idx == self.row && self.focus == Focus::Settings;
                let cursor = if selected {
                    format!("{CYAN}❯{RESET}")
                } else {
                    " ".to_string()
                };
                let (value, src, note) = match origins.iter().find(|o| o.key == k.key) {
                    Some(o) => (o.value.clone(), o.from, o.note),
                    None => ("auto".to_string(), Source::Builtin, ""),
                };
                let shown = if selected {
                    match &self.edit {
                        Some(e) => format!("{}▌", e.buf),
                        None => value,
                    }
                } else {
                    value
                };
                let label = format!("{:<18}", crate::truncate(k.label, 18));
                let shown = format!("{:<16}", crate::truncate(&shown, 16));
                let src_text = if note.is_empty() {
                    src.label().to_string()
                } else {
                    format!("{} ({note})", src.label())
                };
                format!(
                    " {cursor} {DIM}{label}{RESET}{shown}{}{src_text}{RESET}",
                    source_color(src)
                )
            }
        }
    }

    fn status_line(&self, cols: usize) -> String {
        if let Some(i) = self.confirm {
            if let Some(m) = self.ms.get(i) {
                return pad(
                    &format!(
                        "{RED}delete {} ({} free)?{RESET} {DIM}y / N{RESET}",
                        m.name,
                        models::human(m.size)
                    ),
                    cols,
                );
            }
        }
        if let Some((msg, err)) = &self.msg {
            let color = if *err { RED } else { GREEN };
            return pad(&format!("{color}{msg}{RESET}"), cols);
        }
        let dirty = if self.dirty() {
            format!(" {YELLOW}*unsaved{RESET}")
        } else {
            String::new()
        };
        let keys = "↑↓ move · Tab panel · Enter edit · d argv · L run · s save · x delete · ? help · q quit";
        pad(&format!("{DIM}{keys}{RESET}{dirty}"), cols)
    }

    fn render_help(&self, cols: usize) -> String {
        let body = vec![
            String::new(),
            crate::section("MODELS PANEL"),
            format!("  {GREEN}↑↓{RESET}      pick a model, pack or the add row"),
            format!("  {GREEN}Enter{RESET}   open its settings; on the add row, paste a URL"),
            format!("  {GREEN}x{RESET}       delete the selected .gguf (asks first)"),
            format!("  {GREEN}L{RESET}       save pending edits, then launch this model"),
            String::new(),
            crate::section("SETTINGS PANEL"),
            format!("  {GREEN}↑↓{RESET}      move between knobs (group headers are skipped)"),
            format!("  {GREEN}Enter{RESET}   bool toggles, enum cycles, the rest opens a line editor"),
            format!("  {GREEN}-{RESET}       remove the override, so the value falls back a layer"),
            format!("  {GREEN}Tab{RESET}     switch panel"),
            String::new(),
            crate::section("ANYWHERE"),
            format!("  {GREEN}s{RESET}       write changed keys to settings.toml"),
            format!("  {GREEN}d{RESET}       argv with the origin of every value"),
            format!("  {GREEN}?{RESET}       this help"),
            format!("  {GREEN}q{RESET}       quit (Esc also leaves the settings panel)"),
            String::new(),
            format!("  {DIM}values come from cli > model > defaults > builtin > derived{RESET}"),
            format!("  {DIM}only knobs that apply to this model are listed{RESET}"),
        ];
        crate::boxed(
            &format!("{CYAN}{BOLD}hub help{RESET}"),
            &body,
            &format!("{DIM}any key closes this help{RESET}"),
            cols,
        )
    }

    fn render_argv(&self, cols: usize) -> String {
        let mut body = vec![String::new()];
        if let Some(v) = &self.argv {
            for l in &v.lines {
                body.push(format!("  {l}"));
            }
            if !v.warnings.is_empty() {
                body.push(String::new());
                body.push(format!("  {YELLOW}warnings{RESET}"));
                for w in &v.warnings {
                    body.push(format!("  {YELLOW}- {w}{RESET}"));
                }
            }
            body.push(String::new());
            body.push(crate::section("PROVENANCE"));
            for o in &v.origins {
                let src = if o.note.is_empty() {
                    o.from.label().to_string()
                } else {
                    format!("{} ({})", o.from.label(), o.note)
                };
                body.push(format!(
                    "  {:<14} {:<26} {}{}{RESET}",
                    o.key,
                    crate::truncate(&o.value, 24),
                    source_color(o.from),
                    src
                ));
            }
            body.push(String::new());
        }
        crate::boxed(
            &format!("{CYAN}{BOLD}argv{RESET}"),
            &body,
            &format!("{DIM}any key closes · d reopens{RESET}"),
            cols,
        )
    }

    fn render_prompt(&self, cols: usize) -> String {
        let buf = self.prompt.as_ref().map(|p| p.buf.as_str()).unwrap_or("");
        let body = vec![
            String::new(),
            format!("  {DIM}https://huggingface.co/<repo>/blob/main/<file>.gguf{RESET}"),
            format!("  {DIM}https://huggingface.co/<repo>{RESET}"),
            String::new(),
            format!("  {CYAN}>{RESET} {buf}▌"),
            String::new(),
            format!("  {DIM}the download runs in the terminal; the hub reopens after it{RESET}"),
        ];
        crate::boxed(
            &format!("{CYAN}{BOLD}add from HuggingFace{RESET}"),
            &body,
            &format!("{DIM}Enter fetch · Esc cancel{RESET}"),
            cols,
        )
    }
}

// ---------------------------------------------------------------------------
// Small rendering helpers (the rest comes from main.rs)
// ---------------------------------------------------------------------------

fn first_knob_row(rows: &[Row]) -> usize {
    rows.iter()
        .position(|r| matches!(r, Row::Knob(_)))
        .unwrap_or(0)
}

fn source_color(s: Source) -> &'static str {
    match s {
        Source::Cli => CYAN,
        Source::Model => CYAN,
        Source::Defaults => BLUE,
        Source::Builtin => DIM,
        Source::Derived => MAGENTA,
    }
}

fn window_start(len: usize, cursor: usize, avail: usize) -> usize {
    if avail == 0 || len <= avail {
        0
    } else {
        cursor.saturating_sub(avail / 2).min(len - avail)
    }
}

fn pad(s: &str, width: usize) -> String {
    let mut out = crate::clip_visible(s, width);
    let n = crate::vlen(&out);
    if n < width {
        out.push_str(&" ".repeat(width - n));
    }
    out
}

/// A box of exactly `w` columns and `h` lines. `body` is already scrolled; the
/// last inner line is `footer`.
#[allow(clippy::too_many_arguments)]
fn panel(
    title: &str,
    body: &[String],
    footer: &str,
    w: usize,
    h: usize,
    active: bool,
    accent: &str,
) -> Vec<String> {
    let w = w.max(10);
    let h = h.max(4);
    let accent = if active { accent } else { DIM };
    let title_styled = format!(" {accent}{BOLD}{title}{RESET} ");
    let tvis = crate::vlen(&title_styled);
    let dashes = w.saturating_sub(3 + tvis);
    let inner_w = w - 2;
    let inner_h = h - 2;
    let body_h = inner_h.saturating_sub(1);

    let mut out = Vec::with_capacity(h);
    out.push(format!("┌─{}{}┐", title_styled, "─".repeat(dashes)));
    for i in 0..body_h {
        let line = body.get(i).map(String::as_str).unwrap_or("");
        out.push(format!("│{}│", pad(line, inner_w)));
    }
    out.push(format!("│{}│", pad(&format!("{DIM}{footer}{RESET}"), inner_w)));
    out.push(format!("└{}┘", "─".repeat(inner_w)));
    out
}

/// Greedy wrap for a command line; no ANSI in the input.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let width = width.max(20);
    let mut out = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        if !line.is_empty() && line.chars().count() + 1 + word.chars().count() > width {
            out.push(line);
            line = String::new();
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        out.push(line);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_keeps_each_line_within_width() {
        let s = "localmodel --very-long-flag value --another-flag value";
        for l in wrap(s, 20) {
            assert!(l.chars().count() <= 20, "{l:?} is too long");
        }
    }

    #[test]
    fn window_start_keeps_cursor_visible() {
        assert_eq!(window_start(10, 0, 5), 0);
        assert_eq!(window_start(10, 9, 5), 5);
        assert_eq!(window_start(10, 5, 5), 3);
        assert_eq!(window_start(3, 2, 5), 0);
    }
}
