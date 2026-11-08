//! Where each effective launch setting came from.
//!
//! `Cfg::resolve` used to answer only "what is the value". That is not enough
//! once a value can arrive from four places: a CLI flag, `[model.NAME]`,
//! `[defaults]`, or a hardcoded fallback. `Cfg::explain` returns the same
//! `Cfg` plus one `Origin` per knob so `run --explain` and
//! `settings --show --origin` can say *why* a model is launched that way.
//!
//! Precedence is the same in both paths: cli > model > defaults > builtin.

use crate::knobs::Profile;
use crate::RunOpts;

/// The layer that supplied a value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Source {
    /// A flag on the command line.
    Cli,
    /// The model's own entry: `[model.NAME]`.
    Model,
    /// `[defaults]`.
    Defaults,
    /// A hardcoded fallback in this crate.
    Builtin,
    /// Computed from another value (e.g. `ctv` inherits `ctk`, the port is
    /// the first free one, or argv picks a per-model default).
    Derived,
}

impl Source {
    pub fn label(self) -> &'static str {
        match self {
            Source::Cli => "cli",
            Source::Model => "model",
            Source::Defaults => "defaults",
            Source::Builtin => "builtin",
            Source::Derived => "derived",
        }
    }
}

/// One resolved setting and where it came from.
#[derive(Clone, Debug)]
pub struct Origin {
    pub key: &'static str,
    pub value: String,
    pub from: Source,
    /// Short explanation for the table; empty when the source speaks for
    /// itself.
    pub note: &'static str,
}

impl Origin {
    fn new(key: &'static str, value: String, from: Source) -> Self {
        Self { key, value, from, note: "" }
    }

    fn with_note(key: &'static str, value: String, from: Source, note: &'static str) -> Self {
        Self { key, value, from, note }
    }
}

/// First layer that has a value wins; `builtin` is the last resort.
fn pick<T>(
    cli: Option<T>,
    model: Option<T>,
    defaults: Option<T>,
    builtin: Option<T>,
) -> (Option<T>, Source) {
    if let Some(v) = cli {
        (Some(v), Source::Cli)
    } else if let Some(v) = model {
        (Some(v), Source::Model)
    } else if let Some(v) = defaults {
        (Some(v), Source::Defaults)
    } else {
        (builtin, Source::Builtin)
    }
}

fn onoff(v: bool) -> String {
    if v { "on" } else { "off" }.to_string()
}

impl crate::Cfg {
    /// Resolve `o` over `defaults` and an optional model profile, recording
    /// the origin of every value that ends up in the `Cfg`.
    pub fn explain(
        o: &RunOpts,
        defaults: &Profile,
        model: Option<&Profile>,
    ) -> (Self, Vec<Origin>) {
        let mut origins: Vec<Origin> = Vec::new();

        let (ctx, ctx_src) = pick(o.ctx, model.and_then(|p| p.ctx), defaults.ctx, Some(131072));
        let ctx = ctx.expect("ctx has a builtin default");

        let (ctk, ctk_src) = pick(
            o.ctk.clone(),
            model.and_then(|p| p.ctk.clone()),
            defaults.ctk.clone(),
            Some("q8_0".to_string()),
        );
        let ctk = ctk.expect("ctk has a builtin default");

        let (ctv, mut ctv_src) = pick(
            o.ctv.clone(),
            model.and_then(|p| p.ctv.clone()),
            defaults.ctv.clone(),
            Some(ctk.clone()),
        );
        let ctv = ctv.expect("ctv has a builtin default");
        let mut ctv_note = "";
        if ctv_src == Source::Builtin {
            // Not set anywhere: it follows ctk, so "builtin" would hide the
            // fact that changing ctk changes ctv too.
            ctv_src = Source::Derived;
            ctv_note = "same as ctk";
        }

        let (ngl, ngl_src) = pick(o.ngl, model.and_then(|p| p.ngl), defaults.ngl, None);
        let (ncmoe, ncmoe_src) =
            pick(o.ncmoe, model.and_then(|p| p.ncmoe), defaults.ncmoe, None);
        let (port, port_src) = pick(o.port, model.and_then(|p| p.port), defaults.port, None);
        let (threads, threads_src) =
            pick(o.threads, model.and_then(|p| p.threads), defaults.threads, None);
        let (reasoning, reasoning_src) = pick(
            o.reasoning.clone(),
            model.and_then(|p| p.reasoning.clone()),
            defaults.reasoning.clone(),
            None,
        );
        let (effort, effort_src) = pick(
            o.effort.clone(),
            model.and_then(|p| p.effort.clone()),
            defaults.effort.clone(),
            None,
        );
        let (draft_tokens, draft_tokens_src) = pick(
            o.draft_tokens,
            model.and_then(|p| p.draft_tokens),
            defaults.draft_tokens,
            None,
        );
        let (draft_gpu, draft_gpu_src) = pick(
            o.draft_gpu,
            model.and_then(|p| p.draft_gpu),
            defaults.draft_gpu,
            Some(false),
        );

        // `--vision` historically switched vision on only for the literal
        // "on", so "off" could not override a saved `vision = true`. Treat
        // any explicit flag as a decision, like every other CLI knob.
        let cli_vision = o.vision.as_deref().map(|v| v != "off");
        let (vision, vision_src) = pick(
            cli_vision,
            model.and_then(|p| p.vision),
            defaults.vision,
            Some(false),
        );

        let (mmproj, mmproj_src) = pick(
            o.mmproj.clone(),
            model.and_then(|p| p.mmproj.clone()),
            defaults.mmproj.clone(),
            None,
        );
        let (extra, extra_src) = pick(
            o.extra.clone(),
            model.and_then(|p| p.extra.clone()),
            defaults.extra.clone(),
            None,
        );
        let (tools, tools_src) = pick(
            o.tools.clone(),
            model.and_then(|p| p.tools.clone()),
            defaults.tools.clone(),
            None,
        );
        let (alias, alias_src) = pick(
            o.alias.clone(),
            model.and_then(|p| p.alias.clone()),
            defaults.alias.clone(),
            None,
        );
        let (flash_attn, flash_attn_src) = pick(
            None,
            model.and_then(|p| p.flash_attn),
            defaults.flash_attn,
            Some(true),
        );
        let (temp, temp_src) = pick::<f64>(
            None,
            model.and_then(|p| p.temp),
            defaults.temp,
            Some(1.0),
        );
        let (top_p, top_p_src) = pick::<f64>(
            None,
            model.and_then(|p| p.top_p),
            defaults.top_p,
            Some(0.95),
        );
        let (host, host_src) = pick(
            None,
            model.and_then(|p| p.host.clone()),
            defaults.host.clone(),
            Some("127.0.0.1".to_string()),
        );

        // Declaration order of `knobs::KEYS`, so the table reads like the
        // settings file and like the hub will.
        origins.push(Origin::new("ctx", ctx.to_string(), ctx_src));
        origins.push(Origin::new("ctk", ctk.clone(), ctk_src));
        origins.push(Origin::with_note("ctv", ctv.clone(), ctv_src, ctv_note));
        if let Some(v) = ngl {
            origins.push(Origin::new("ngl", v.to_string(), ngl_src));
        }
        if let Some(v) = ncmoe {
            origins.push(Origin::new("ncmoe", v.to_string(), ncmoe_src));
        }
        if let Some(v) = port {
            origins.push(Origin::new("port", v.to_string(), port_src));
        }
        if let Some(v) = threads {
            origins.push(Origin::new("threads", v.to_string(), threads_src));
        }
        if let Some(ref v) = reasoning {
            origins.push(Origin::new("reasoning", v.clone(), reasoning_src));
        }
        if let Some(ref v) = effort {
            origins.push(Origin::new("effort", v.clone(), effort_src));
        }
        if let Some(v) = draft_tokens {
            origins.push(Origin::new("draft_tokens", v.to_string(), draft_tokens_src));
        }
        let draft_gpu = draft_gpu.expect("draft_gpu has a builtin default");
        origins.push(Origin::new("draft_gpu", onoff(draft_gpu), draft_gpu_src));
        let vision = vision.expect("vision has a builtin default");
        origins.push(Origin::new("vision", onoff(vision), vision_src));
        if let Some(ref v) = mmproj {
            origins.push(Origin::new("mmproj", v.clone(), mmproj_src));
        }
        if let Some(ref v) = extra {
            origins.push(Origin::new("extra", v.clone(), extra_src));
        }
        if let Some(ref v) = tools {
            origins.push(Origin::new("tools", v.clone(), tools_src));
        }
        if let Some(ref v) = alias {
            origins.push(Origin::new("alias", v.clone(), alias_src));
        }
        let flash_attn = flash_attn.expect("flash_attn has a builtin default");
        origins.push(Origin::new("flash_attn", onoff(flash_attn), flash_attn_src));
        let temp = temp.expect("temp has a builtin default");
        origins.push(Origin::new("temp", format!("{temp}"), temp_src));
        let top_p = top_p.expect("top_p has a builtin default");
        origins.push(Origin::new("top_p", format!("{top_p}"), top_p_src));
        let host = host.expect("host has a builtin default");
        origins.push(Origin::new("host", host.clone(), host_src));

        let cfg = crate::Cfg {
            ctx,
            port,
            ngl,
            ncmoe,
            ctk,
            ctv,
            threads,
            reasoning,
            effort,
            draft_tokens,
            extra,
            tools,
            draft_gpu,
            vision,
            mmproj,
            alias,
            flash_attn,
            temp,
            top_p,
            host,
        };
        (cfg, origins)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile_with_ctx(ctx: u32) -> Profile {
        Profile { ctx: Some(ctx), ..Default::default() }
    }

    #[test]
    fn cli_beats_model_beats_defaults() {
        let defaults = profile_with_ctx(4096);
        let model = profile_with_ctx(8192);
        let mut o = RunOpts::default();

        let (cfg, origins) = crate::Cfg::explain(&o, &defaults, Some(&model));
        assert_eq!(cfg.ctx, 8192);
        assert_eq!(origins[0].from, Source::Model);

        o.ctx = Some(16384);
        let (cfg, origins) = crate::Cfg::explain(&o, &defaults, Some(&model));
        assert_eq!(cfg.ctx, 16384);
        assert_eq!(origins[0].from, Source::Cli);

        let (cfg, origins) = crate::Cfg::explain(&RunOpts::default(), &defaults, None);
        assert_eq!(cfg.ctx, 4096);
        assert_eq!(origins[0].from, Source::Defaults);
    }

    #[test]
    fn ctv_inherits_ctk_and_says_so() {
        let p = Profile { ctk: Some("q4_0".into()), ..Default::default() };
        let (cfg, origins) = crate::Cfg::explain(&RunOpts::default(), &p, None);
        assert_eq!(cfg.ctk, "q4_0");
        assert_eq!(cfg.ctv, "q4_0");
        let ctv = origins.iter().find(|o| o.key == "ctv").unwrap();
        assert_eq!(ctv.from, Source::Derived);
        assert_eq!(ctv.note, "same as ctk");
    }

    #[test]
    fn explain_and_resolve_agree() {
        let defaults = Profile {
            ctx: Some(65536),
            ctk: Some("q8_0".into()),
            temp: Some(0.7),
            ..Default::default()
        };
        let model = Profile { ngl: Some(42), ..Default::default() };
        let mut o = RunOpts::default();
        o.port = Some(9000);
        o.vision = Some("off".into());
        let (explained, _) = crate::Cfg::explain(&o, &defaults, Some(&model));
        let merged = defaults.overlay(&model);
        let resolved = crate::Cfg::resolve(&o, &merged);
        assert_eq!(explained.ctx, resolved.ctx);
        assert_eq!(explained.ctk, resolved.ctk);
        assert_eq!(explained.ctv, resolved.ctv);
        assert_eq!(explained.ngl, resolved.ngl);
        assert_eq!(explained.temp, resolved.temp);
        assert_eq!(explained.vision, resolved.vision);
        assert_eq!(explained.host, resolved.host);
    }
}
