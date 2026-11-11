//! `localmodel dev`: the playground — tweak every knob live, launch a
//! llama.cpp server, watch the log. Stop = reset to base (never saved).
//! Reuses the dashboard's crossterm rendering, no second stack.

use std::fs;
use std::io::{BufRead, BufReader, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;

use crossterm::cursor;
use crossterm::event::{self, Event, KeyCode, KeyEvent};
use crossterm::execute;
use crossterm::terminal;

use crate::config;
use crate::fit;
use crate::gguf;
use crate::knobs::{self, Caps, Group, Kind, Profile};
use crate::mem;
use crate::models::{self, Model};
use crate::server;
use crate::settings;
use crate::{screen_size, BOLD, CYAN, DIM, GREEN, RED, RESET, YELLOW, RunOpts};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Pick,
    Edit,
    Log,
}

#[derive(Clone, Copy, PartialEq)]
enum Overlay {
    None,
    Help,
}

struct Edit {
    buf: String,
    fresh: bool,
}

struct Server {
    child: Arc<Mutex<Child>>,
    log: Arc<Mutex<Vec<String>>>,
}

struct Est {
    line: String,
    placement: Vec<String>,
}

pub struct Dev {
    ms: Vec<Model>,
    sel: usize,
    o: RunOpts,
    work: Profile,
    defaults: Profile,
    base_work: Profile,
    cfg: crate::Cfg,
    bin: PathBuf,
    alias: String,
    port: u16,
    hw: fit::Hardware,
    started: bool,
    running: bool,
    exited: bool,
    server: Option<Server>,
    phase: Phase,
    row: usize,
    edit: Option<Edit>,
    logbuf: Vec<String>,
    overlay: Overlay,
    msg: Option<(String, bool)>,
    est: Option<Est>,
    dir: PathBuf,
    logpos: usize,
}
impl Dev {
    pub fn cmd_dev(args: &[String]) -> i32 {
        if !io_is_terminal() {
            println!("{YELLOW}localmodel dev{RESET} wants a terminal.{RESET}");
            println!("  Pick a model, tweak every knob live, launch, watch the log.");
            println!("  r (re)launch · l log · q stop+reset · ? help");
            println!();
            println!("  localmodel run <model> [opts]   launch a model");
            println!("  localmodel info <model>         print the card");
            println!("  localmodel params               llama.cpp flag cheat sheet");
            return 0;
        }
        let dir = config::models_dir();
        let bin = config::server_bin();
        if !bin.exists() {
            eprintln!("llama-server not found at {}", bin.display());
            return 1;
        }
        if models::scan(&dir).is_empty() {
            eprintln!("no models in {} — add one: localmodel add <url>", dir.display());
            return 1;
        }
        enter();
        let code = Dev::run(args, &dir, bin);
        leave();
        code
    }

    fn run(args: &[String], _dir: &Path, bin: PathBuf) -> i32 {
        let mut d = Self::new();
        d.bin = bin;
        d.start(args);
        d.loop_run();
        d.stop_all();
        0
    }

    fn new() -> Self {
        let dir = config::models_dir();
        let hw = fit::Hardware::probe(&dir);
        let saved = settings::load();
        Dev {
            ms: models::scan(&dir),
            sel: 0,
            o: RunOpts::default(),
            work: Profile::default(),
            defaults: saved.defaults.clone(),
            base_work: Profile::default(),
            cfg: crate::Cfg::default(),
            bin: PathBuf::new(),
            alias: String::new(),
            port: 0,
            hw,
            started: false,
            running: false,
            exited: false,
            server: None,
            est: None,
            dir: config::models_dir(),
            logpos: 0,
            phase: Phase::Pick,
            row: 0,
            edit: None,
            logbuf: Vec::new(),
            overlay: Overlay::None,
            msg: None,
        }
    }

    fn start(&mut self, args: &[String]) {
        let mut query: Vec<String> = Vec::new();
        for a in args {
            if a.starts_with('-') {
                continue;
            }
            query.push(a.clone());
        }
        if self.o.query.is_empty() {
            self.o.query = query.clone();
        }
        // With an argument go straight to that model; with none, stay on the
        // picker instead of silently editing the first entry in the list.
        if !query.is_empty() {
            let q = query.join(" ").to_lowercase();
            if let Some(i) = self.ms.iter().position(|m| m.name.to_lowercase().contains(&q)) {
                self.choose(i);
                return;
            }
        }
        self.phase = Phase::Pick;
    }

    /// Open one model in the playground, starting from its saved profile so
    /// the screen shows the settings that will actually be launched.
    fn choose(&mut self, i: usize) {
        if i >= self.ms.len() {
            return;
        }
        // Switching models must not leave the old server running: select()
        // drops the handle without killing the child.
        if let Some(s) = self.server.take() {
            let _ = self.kill(s);
        }
        self.sel = i;
        self.select();
        self.base_work = self.work.clone();
    }

    // -- model selection -------------------------------------------------

    fn caps(&self) -> knobs::Caps {
        let m = self.ms.get(self.sel);
        match m {
            Some(m) => Caps {
                moe: m.moe,
                nextn: m.nextn,
                draft: m.nextn || models::find_draft(&m.path).is_some(),
                vision: models::find_mmproj(&m.path).is_some() || self.work.mmproj.is_some(),
            },
            None => Caps::default(),
        }
    }

    /// All knobs that apply to the current model.
    fn knobs(&self) -> Vec<&'static knobs::Knob> {
        let c = self.caps();
        knobs::KNOBS.iter().filter(|k| k.applies.check(&c)).collect()
    }

    /// Pick the current model, rebuild the live config, and reset the
    /// running server. Returns an error string only if the model vanished.
    fn select(&mut self) {
        let Some(m) = self.ms.get(self.sel).cloned() else { return };
        self.alias = models::stem(&m.name);
        // Start the playground from the saved profile (not an empty one), so
        // "edit" means edit what is on disk. `[defaults]` still supplies the
        // fallbacks through Cfg::explain.
        let saved = settings::load();
        self.defaults = saved.defaults.clone();
        let scope = models::stem(&m.name);
        let key = settings::find_key(&saved, &scope)
            .map(str::to_string)
            .unwrap_or(scope);
        self.work = saved.model.get(&key).cloned().unwrap_or_default();
        let (cfg, _) = crate::Cfg::explain(&self.o, &self.defaults, Some(&self.work));
        let port = cfg.port.unwrap_or_else(|| server::free_port(8080));
        self.cfg = cfg;
        self.port = port;
        self.running = false;
        self.exited = false;
        self.server = None;
        self.started = true;
        self.phase = Phase::Edit;
        self.row = first_knob_row(&self.knobs());
    }

    // -- editing ---------------------------------------------------------

    fn set(&mut self, k: &'static knobs::Knob, raw: &str) -> Result<String, String> {
        let t = raw.trim();
        if t.is_empty() || t == "-" {
            self.work.set(k.key, "-")?;
            self.recompute();
            return Ok(String::new());
        }
        match k.kind {
            Kind::Num { min, max, .. } => {
                let digits: String = t.chars().filter(|c| c.is_ascii_digit()).collect();
                if digits.is_empty() {
                    return Err(format!("{}: expected a number", k.label));
                }
                let v: i64 = digits
                    .parse()
                    .map_err(|_| format!("{}: '{}' is not a number", k.label, raw))?;
                if v < min || v > max {
                    return Err(format!("{label} must be {min}..{max}", label = k.label));
                }
                self.work.set(k.key, &digits.to_string())?;
                self.recompute();
                Ok(digits)
            }
            Kind::Float { min, max } => match t.parse::<f64>() {
                Ok(v) if v >= min && v <= max => {
                    self.work.set(k.key, t)?;
                    self.recompute();
                    Ok(t.to_string())
                }
                Ok(_) => Err(format!("{label} must be {min}..{max}", label = k.label)),
                Err(_) => Err(format!("{}: expected a number", k.label)),
            },
            Kind::Enum(vals) => {
                if vals.contains(&t) {
                    self.work.set(k.key, t)?;
                    self.recompute();
                    Ok(t.to_string())
                } else {
                    Err(format!("{}: one of {}", k.label, vals.join(", ")))
                }
            }
            _ => {
                self.work.set(k.key, t)?;
                self.recompute();
                Ok(t.to_string())
            }
        }
    }

    fn activate(&mut self, k: &'static knobs::Knob) {
        let cur = match self.work.get(k.key) {
            Some(s) => s,
            None => knobs::Kind::hint(k.kind).to_string(),
        };
        let next = match k.kind {
            Kind::Bool => {
                let b = matches!(cur.to_lowercase().as_str(), "on" | "true" | "1" | "yes");
                if b { "off" } else { "on" }.to_string()
            }
            Kind::Enum(vals) => {
                let i = vals.iter().position(|v| *v == cur).unwrap_or(0);
                vals[(i + 1) % vals.len()].to_string()
            }
            _ => return,
        };
        let _ = self.set(k, &next);
    }

    fn begin_edit(&mut self) {
        let Some(k) = self.current_knob() else { return };
        // Pre-fill with the value the screen is showing: the model override
        // first, then [defaults]. Built-ins stay empty and are applied on
        // Enter if the user leaves the buffer alone.
        let cur = self
            .work
            .get(k.key)
            .or_else(|| self.defaults.get(k.key))
            .unwrap_or_default();
        self.edit = Some(Edit { buf: cur, fresh: true });
    }

    fn current_knob(&self) -> Option<&'static knobs::Knob> {
        self.knobs().get(self.row).map(|v| &**v)
    }

    fn move_row(&mut self, dir: isize) {
        let rows = self.knobs();
        let mut i = self.row as isize;
        for _ in 0..rows.len() {
            i += dir;
            if i < 0 {
                if dir > 0 {
                    break;
                }
                i = 0;
                continue;
            }
            if i as usize >= rows.len() {
                if dir < 0 {
                    break;
                }
                i = rows.len() as isize - 1;
                continue;
            }
            self.row = i as usize;
            return;
        }
        self.row = rows.iter().rposition(|k| k.key != rows[0].key).unwrap_or(0);
    }

    fn recompute(&mut self) {
        let (cfg, _) = crate::Cfg::explain(&self.o, &self.defaults, Some(&self.work));
        let port = cfg.port.unwrap_or_else(|| server::free_port(8080));
        // Editing a knob invalidates the running server. Kill it instead of
        // just dropping the handle, which used to leave an orphan process.
        if let Some(s) = self.server.take() {
            let _ = self.kill(s);
        }
        self.cfg = cfg;
        self.port = port;
        self.running = false;
        self.exited = false;
        self.logbuf.clear();
        self.phase = Phase::Edit;
        self.estimate();
    }

    fn estimate(&mut self) {
        let Some(m) = self.ms.get(self.sel) else { return };
        let gg: Option<(gguf::Meta, u64)> = if m.nextn {
            None
        } else {
            match models::find_draft(&m.path) {
                Some(p) => Some((gguf::parse(&p), fs::metadata(&p).map(|x| x.len()).unwrap_or(0))),
                None => None,
            }
        };
        let draft = gg.as_ref().map(|(meta, weights)| mem::Draft {
            meta,
            weights: *weights,
        });
        let cfg = &self.cfg;
        let e = fit::for_gguf(
            &m.meta,
            m.size,
            &self.hw,
            cfg.ctx,
            &cfg.ctk,
            &cfg.ctv,
            draft.as_ref().map(|d| fit::DraftKv {
                layers: &d.meta.kv.main,
                swa_window: d.meta.kv.swa_window,
                weights_bytes: d.weights,
                ctk: &cfg.draft_ctk,
                ctv: &cfg.draft_ctv,
            }),
            cfg.kv_ram,
        );
        let draft_plan = draft.as_ref().map(|d| mem::Draft { meta: d.meta, weights: d.weights });
        let plan = mem::plan(m, draft_plan, cfg, &self.hw);
        let kv = plan.kv_main_gb + plan.kv_draft_gb;
        let verdict = if e.room_gb - e.need_vram_gb >= 0.0 {
            "OK"
        } else {
            "SHORT"
        };
        let line = format!(
            "weights {:.1}G  KV {:.1}G  VRAM {:.1}/{:.1}G  {}  {}L/{}L",
            plan.total_weights_gb,
            kv,
            e.need_vram_gb,
            e.room_gb,
            verdict,
            plan.ngl,
            plan.block_count,
        );
        let placement: Vec<String> = plan
            .place_rows()
            .into_iter()
            .map(|(k, v)| format!("  {DIM}{k}{RESET}: {v}"))
            .collect();
        self.est = Some(Est { line, placement });
    }

//__NEXT__
    fn handle(&mut self, k: KeyEvent) -> Option<()> {
        if self.overlay == Overlay::Help {
            self.overlay = Overlay::None;
            return None;
        }
        if self.phase == Phase::Pick {
            return self.handle_pick(k);
        }
        if self.edit.is_some() {
            return self.handle_edit(k);
        }
        if self.phase == Phase::Log {
            return self.handle_log(k);
        }
        match k.code {
            KeyCode::Char('?') | KeyCode::Char('h') => { self.overlay = Overlay::Help; return None; }
            KeyCode::Char('q') => return Some(()),
            KeyCode::Esc => { self.stop(); return Some(()); }
            KeyCode::Up | KeyCode::Char('k') => self.move_row(-1),
            KeyCode::Down | KeyCode::Char('j') => self.move_row(1),
            KeyCode::Tab => self.phase = Phase::Log,
            KeyCode::Char('l') => { self.phase = if self.phase == Phase::Log { Phase::Edit } else { Phase::Log }; }
            KeyCode::Char('r') => { self.running = self.restart(); }
            KeyCode::Char('s') => { self.save_profile(); }
            KeyCode::Char('m') => { self.phase = Phase::Pick; }
            KeyCode::Char('-') => {
                if let Some(knob) = self.current_knob() {
                    let _ = self.set(knob, "-");
                }
            }
            KeyCode::Char(' ') | KeyCode::Enter => {
                if let Some(knob) = self.current_knob() {
                    match knob.kind {
                        Kind::Bool | Kind::Enum(_) => self.activate(knob),
                        _ => self.begin_edit(),
                    }
                }
            }
            _ => {}
        }
        None
    }

    fn handle_pick(&mut self, k: KeyEvent) -> Option<()> {
        match k.code {
            KeyCode::Char('?') | KeyCode::Char('h') => self.overlay = Overlay::Help,
            KeyCode::Char('q') => return Some(()),
            KeyCode::Up | KeyCode::Char('k') => {
                self.sel = self.sel.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.sel + 1 < self.ms.len() {
                    self.sel += 1;
                }
            }
            KeyCode::Enter | KeyCode::Char(' ') => self.choose(self.sel),
            _ => {}
        }
        None
    }

    fn handle_edit(&mut self, k: KeyEvent) -> Option<()> {
        match k.code {
            KeyCode::Enter => {
                if let Some(e) = self.edit.take() {
                    let new = self.set(self.current_knob()?, &e.buf);
                    match new {
                        Ok(v) => {
                            self.msg = Some((format!("{} = {}", self.current_knob().map(|k|k.label).unwrap_or("knob"), v), false));
                        }
                        Err(e) => self.msg = Some((e, true)),
                    }
                }
            }
            KeyCode::Esc => self.edit = None,
            KeyCode::Backspace => {
                if let Some(e) = self.edit.as_mut() {
                    if e.fresh { e.buf.clear(); e.fresh = false; } else { e.buf.pop(); }
                }
            }
            KeyCode::Char(c) => {
                if let Some(e) = self.edit.as_mut() {
                    if e.fresh { e.buf.clear(); e.fresh = false; }
                    e.buf.push(c);
                }
            }
            _ => {}
        }
        None
    }

    fn handle_log(&mut self, k: KeyEvent) -> Option<()> {
        match k.code {
            KeyCode::Up | KeyCode::Char('k') => self.logpos = self.logpos.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => self.logpos += 1,
            KeyCode::Esc | KeyCode::Char('q') => self.phase = Phase::Edit,
            _ => {}
        }
        None
    }

    fn loop_run(&mut self) {
        loop {
            self.refresh_log();
            self.check_exit();
            let (cols, rows) = screen_size();
            let html = self.paint(rows);
            let mut s = std::io::stdout();
            let _ = write!(s, "{}", crate::paint_screen(&html, cols));
            let _ = s.flush();
            match event::read() {
                Ok(Event::Key(k)) => {
                    if self.handle(k).is_some() {
                        break;
                    }
                }
                Ok(Event::Resize(_, _)) => {}
                Ok(Event::FocusGained) | Ok(Event::FocusLost) | Ok(Event::Mouse(_)) | Ok(Event::Paste(_)) => {}
                Err(_) => { self.stop(); break; }
            }
            if self.exited {
                break;
            }
        }
    }

    fn stop_all(&mut self) {
        self.stop();
    }

fn window_start(len: usize, cursor: usize, avail: usize) -> usize {
    if avail == 0 || len <= avail {
        0
    } else {
        cursor.saturating_sub(avail / 2).min(len - avail)
    }
}
    fn paint(&mut self, rows: usize) -> String {
        if self.overlay == Overlay::Help {
            return self.help(rows);
        }
        match self.phase {
            Phase::Pick => self.pick(rows),
            _ => self.edit(rows),
        }
    }

    fn pick(&self, _rows: usize) -> String {
        let w = screen_size().0.max(40);
        let _h = _rows.saturating_sub(2).max(4);
        let mut body = Vec::new();
        body.push(String::new());
        body.push(format!("  {CYAN}models in {}{RESET}", self.dir.display()));
        let running = self.running_servers();
        for (i, m) in self.ms.iter().enumerate() {
            let dot = if running.iter().any(|p| *p == m.path) { format!("{GREEN}●{RESET}") } else { " ".to_string() };
            let badge = if m.nextn { " +mtp" } else if models::find_draft(&m.path).is_some() { " +draft" } else if m.moe { " MoE" } else { "" };
            let name = crate::truncate(&m.name, w.saturating_sub(12));
            let sel = if i == self.sel { format!("{GREEN}●{RESET}") } else { " ".to_string() };
            body.push(format!(" {sel}{dot} {name}{DIM}{badge}{RESET}"));
        }
        let footer = format!("↑↓ pick · Enter {CYAN}begin{RESET} · q quit  ·  running: {}", running.len());
        crate::boxed(&format!("{CYAN}{BOLD}dev{RESET}  models"), &body, &footer, w)
    }

    fn edit(&self, rows: usize) -> String {
        let w = screen_size().0.max(40);
        let h = rows.saturating_sub(2).max(6);
        let mut body = Vec::new();
        let m = self.ms.get(self.sel);
        let hum = models::human(m.map(|x| x.size).unwrap_or(0));
        let quant = m.map(|x| x.quant.clone()).unwrap_or_default();
        let status = if self.running {
            format!(" {GREEN}running{RESET} {DIM}{alias}{RESET} {DIM}· r restart · q stop+reset · l log{RESET}", alias = self.alias)
        } else {
            "  playground · reset on stop".to_string()
        };
        body.push(format!("  {CYAN}{}{RESET} {DIM}{quant}{RESET} {DIM}:{port}{RESET}{status}", hum, port = self.port));
        body.push(String::new());
        let cur = self.current_knob().map(|k| k.group).unwrap_or(Group::Context);
        body.push(format!("  {DIM}{}{RESET}", Group::title(cur)));
        let ks = self.knobs();
        let start = Self::window_start(ks.len(), self.row, h.max(1));
        let end = (start + h).min(ks.len());
        for i in start..end {
            let k = ks[i];
            let cursor = if i == self.row { format!("{CYAN}❯{RESET}") } else { String::new() };
            let value = match self.work.get(k.key) {
                Some(s) => s,
                None => String::new(),
            };
            let label = format!("{:<18}", crate::truncate(k.label, 18));
            let shown = format!("{:<16}", crate::truncate(&value, 16));
            body.push(format!(" {cursor} {DIM}{label}{RESET}{shown}"));
        }
        if let Some(k) = self.current_knob() {
            // Same "what does this do" line the hub shows, so the playground
            // does not require memorising flag names.
            let help = crate::truncate(k.help, w.saturating_sub(6));
            body.push(format!("  {DIM}{help}{RESET}"));
        }
        if self.running || self.phase == Phase::Log {
            let start = self.logpos.saturating_sub(5);
            for l in self.logbuf.iter().skip(start).take(5).cloned() {
                body.push(format!("  {DIM}▎{l}{RESET}"));
            }
        } else if let Some(e) = &self.est {
            body.push(format!("  {GREEN}est{RESET}  {}", e.line));
            for p in &e.placement {
                body.push(p.clone());
            }
        }
        if let Some((msg, err)) = &self.msg {
            let color = if *err { RED } else { GREEN };
            body.push(format!("  {color}{msg}{RESET}"));
        }
        let footer = if self.running {
            "s save profile · r restart · m models · l log · q stop+reset"
        } else {
            "↑↓ knob · Enter edit · s save profile · m models · r launch · ? help"
        };
        let title = if self.running { format!("dev {GREEN}●{RESET} RUNNING") } else { "dev · ready".to_string() };
        let sub = if self.running { " · live".to_string() } else { "  playground · reset on stop".to_string() };
        crate::boxed(&format!("{CYAN}{BOLD}{title}{RESET}  {}:{}{sub}", self.alias, self.port), &body, footer, w)
    }

    fn help(&self, _rows: usize) -> String {
        let w = screen_size().0.max(40);
        let body = vec![
            String::new(),
            crate::section("EDIT"),
            format!("  ↑↓ move · Enter toggle/enum, else open a text editor"),
            format!("  Tab log · r launch/restart · m pick another model · - clear override"),
            format!("  q stop + reset to the saved profile"),
            format!("  {GREEN}s{RESET} save this experiment as [model.NAME] in settings.toml"),
            String::new(),
            crate::section("SAVE"),
            format!("  After a launch you like, press s to keep the knobs permanently."),
            format!("  Only touched keys are written; [defaults] still applies to the rest."),
            String::new(),
            crate::section("LOG VIEW"),
            format!("  ↑↓ scroll · Esc/q back to the screen"),
            String::new(),
            crate::section("ANYWHERE"),
            format!("  ? close help"),
            String::new(),
        ];
        crate::boxed(&format!("{CYAN}{BOLD}dev help{RESET}"), &body, "any key closes", w)
    }
    fn stop(&mut self) {
        if let Some(s) = self.server.take() {
            let _ = self.kill(s);
        }
        self.running = false;
        self.exited = false;
        self.logbuf.clear();
        self.work = self.base_work.clone();
        self.phase = Phase::Edit;
        self.logpos = 0;
        self.recompute();
    }

    fn kill(&self, s: Server) {
        if let Ok(mut c) = s.child.lock() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }

    fn refresh_log(&mut self) {
        if let Some(s) = &self.server {
            if let Ok(v) = s.log.lock() {
                self.logbuf = v.clone();
            }
        }
    }

    fn check_exit(&mut self) {
        if let Some(s) = &self.server {
            if let Ok(mut c) = s.child.lock() {
                if c.try_wait().ok().flatten().is_some() {
                    self.exited = true;
                    self.running = false;
                }
            }
        }
    }

    fn running_servers(&self) -> Vec<PathBuf> {
        let servers = server::list_servers(&self.dir.to_string_lossy());
        self.ms.iter()
            .filter(|m| servers.iter().any(|s| s.cmd.contains(&m.path.display().to_string())))
            .map(|m| m.path.clone())
            .collect()
    }

    fn launch(&mut self) -> bool {
        if self.server.is_some() {
            return false;
        }
        let Some(model) = self.ms.get(self.sel).cloned() else { return false; };
        let draft = models::find_draft(&model.path);
        let sv = crate::build_args(&model, draft.as_deref(), &self.alias, self.port, &self.cfg);
        let mut child = match Command::new(&self.bin).args(&sv).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::piped()).spawn() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("failed to start llama-server: {e}");
                return false;
            }
        };
        let stderr = child.stderr.take();
        let log = Arc::new(Mutex::new(Vec::new()));
        let log_thread = log.clone();
        thread::spawn(move || {
            let reader = BufReader::new(stderr.unwrap());
            for line in reader.lines().flatten() {
                if let Ok(mut b) = log_thread.lock() {
                    b.push(line);
                }
            }
        });
        self.server = Some(Server { child: Arc::new(Mutex::new(child)), log });
        self.running = true;
        self.exited = false;
        true
    }

    /// Restart with the *current* experiment, not the saved base: `r` while
    /// running is "apply these knobs", not "throw them away".
    fn restart(&mut self) -> bool {
        if let Some(s) = self.server.take() {
            let _ = self.kill(s);
        }
        self.running = false;
        self.exited = false;
        self.logbuf.clear();
        self.logpos = 0;
        self.launch()
    }

    /// Promote the current experiment to a permanent `[model.NAME]` profile.
    /// Only the keys the user touched are stored; everything else keeps
    /// falling back to `[defaults]` and the built-ins.
    fn save_profile(&mut self) -> bool {
        let Some(m) = self.ms.get(self.sel) else { return false };
        let scope = models::stem(&m.name);
        let mut st = settings::load();
        let key = settings::find_key(&st, &scope)
            .map(str::to_string)
            .unwrap_or(scope);
        let mut prof = self.work.clone();
        // `applies_to` is not a knob and the playground never edits it; keep
        // whatever the saved profile already had instead of dropping it.
        prof.applies_to = st.model.get(&key).and_then(|p| p.applies_to.clone());
        if prof.is_empty() {
            st.model.remove(&key);
        } else {
            st.model.insert(key.clone(), prof);
        }
        match settings::save(&mut st) {
            Ok(p) => {
                self.msg = Some((
                    format!("saved [model.\"{key}\"] to {}", p.display()),
                    false,
                ));
                true
            }
            Err(e) => {
                self.msg = Some((format!("save failed: {e}"), true));
                false
            }
        }
    }
}

fn io_is_terminal() -> bool {
    std::io::stdout().is_terminal()
}

fn enter() {
    let mut out = std::io::stdout();
    let _ = terminal::enable_raw_mode();
    let _ = execute!(out, cursor::Hide, terminal::EnterAlternateScreen);
    let _ = out.flush();
}

fn leave() {
    let _ = terminal::disable_raw_mode();
    let _ = execute!(std::io::stdout(), terminal::LeaveAlternateScreen, cursor::Show);
    let _ = std::io::stdout().flush();
}

fn first_knob_row(rows: &[&'static knobs::Knob]) -> usize {
    rows.first().map(|_| 0).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dev_without_a_model_stays_on_the_picker() {
        let mut d = Dev::new();
        d.start(&[]);
        assert!(matches!(d.phase, Phase::Pick), "dev must not auto-open the first model");
    }

    #[test]
    fn dev_with_a_query_opens_that_model() {
        let mut d = Dev::new();
        let Some(first) = d.ms.first().cloned() else { return };
        let q = models::stem(&first.name);
        d.start(&[q.clone()]);
        assert!(matches!(d.phase, Phase::Edit), "a query should open the editor");
        assert_eq!(d.ms[d.sel].name, first.name);
    }
}

