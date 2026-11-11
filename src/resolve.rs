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

        // The draft/MTP context has its own cache types. Empty means "keep
        // llama.cpp's f16 default"; a set value is emitted into argv as
        // `-ctkd`/`-ctvd`. They resolve like ctk (cli > model > defaults),
        // so the provenance table says which layer picked them.
        let (draft_ctk, draft_ctk_src) = pick(
            o.draft_ctk.clone(),
            model.and_then(|p| p.draft_ctk.clone()),
            defaults.draft_ctk.clone(),
            None,
        );
        let (draft_ctv, draft_ctv_src) = pick(
            o.draft_ctv.clone(),
            model.and_then(|p| p.draft_ctv.clone()),
            defaults.draft_ctv.clone(),
            None,
        );
        // Empty string means "keep llama.cpp's f16 default" — not worth an
        // origin entry, so it is only surfaced when the user actually set it.
        let draft_ctk = draft_ctk.unwrap_or_default();
        let draft_ctv = draft_ctv.unwrap_or_default();

        let (ngl, ngl_src) = pick(o.ngl, model.and_then(|p| p.ngl), defaults.ngl, None);
        let (ncmoe, ncmoe_src) =
            pick(o.ncmoe, model.and_then(|p| p.ncmoe), defaults.ncmoe, None);
        // Memory placement knobs llama.cpp exposes but localmodel used to
        // hide behind `extra`. They resolve like every other knob so the hub,
        // `settings --show` and `run --explain` all agree.
        let (load_mode, load_mode_src) = pick(
            o.load_mode.clone(),
            model.and_then(|p| p.load_mode.clone()),
            defaults.load_mode.clone(),
            None,
        );
        let (moe_cache_mib, moe_cache_src) = pick(
            o.moe_cache_mib,
            model.and_then(|p| p.moe_cache_mib),
            defaults.moe_cache_mib,
            None,
        );
        let (cpu_moe, cpu_moe_src) = pick(
            o.cpu_moe,
            model.and_then(|p| p.cpu_moe),
            defaults.cpu_moe,
            Some(false),
        );
        let (n_cpu_ffn, n_cpu_ffn_src) = pick(
            o.n_cpu_ffn,
            model.and_then(|p| p.n_cpu_ffn),
            defaults.n_cpu_ffn,
            None,
        );
        let (override_tensor, override_src) = pick(
            o.override_tensor.clone(),
            model.and_then(|p| p.override_tensor.clone()),
            defaults.override_tensor.clone(),
            None,
        );
        let (lazy_mode, lazy_mode_src) = pick(
            o.lazy_mode.clone(),
            model.and_then(|p| p.lazy_mode.clone()),
            defaults.lazy_mode.clone(),
            None,
        );
        let (numa, numa_src) = pick(
            o.numa.clone(),
            model.and_then(|p| p.numa.clone()),
            defaults.numa.clone(),
            None,
        );
        let (no_op_offload, no_op_src) = pick(
            o.no_op_offload,
            model.and_then(|p| p.no_op_offload),
            defaults.no_op_offload,
            Some(false),
        );
        let (batch_size, batch_src) = pick(
            o.batch_size,
            model.and_then(|p| p.batch_size),
            defaults.batch_size,
            Some(2048),
        );
        let (ubatch_size, ubatch_src) = pick(
            o.ubatch_size,
            model.and_then(|p| p.ubatch_size),
            defaults.ubatch_size,
            Some(512),
        );
        let (parallel, parallel_src) = pick(
            o.parallel,
            model.and_then(|p| p.parallel),
            defaults.parallel,
            Some(1),
        );
        let (threads_batch, threads_batch_src) = pick(
            o.threads_batch,
            model.and_then(|p| p.threads_batch),
            defaults.threads_batch,
            None,
        );
        let (prio, prio_src) =
            pick(o.prio, model.and_then(|p| p.prio), defaults.prio, None);
        let (cache_reuse, cache_reuse_src) = pick(
            o.cache_reuse,
            model.and_then(|p| p.cache_reuse),
            defaults.cache_reuse,
            Some(256),
        );
        let (draft_n_min, draft_n_min_src) = pick(
            o.draft_n_min,
            model.and_then(|p| p.draft_n_min),
            defaults.draft_n_min,
            None,
        );
        let (draft_p_min, draft_p_min_src) = pick(
            o.draft_p_min,
            model.and_then(|p| p.draft_p_min),
            defaults.draft_p_min,
            None,
        );
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
        let (kv_ram, kv_ram_src) =
            pick(o.kv_ram, model.and_then(|p| p.kv_ram), defaults.kv_ram, Some(false));
        let (split_mode, split_mode_src) = pick(
            o.split_mode.clone(),
            model.and_then(|p| p.split_mode.clone()),
            defaults.split_mode.clone(),
            Some("layer".to_string()),
        );
        let split_mode = split_mode.expect("split_mode has a builtin default");
        let (tensor_split, tensor_split_src) = pick(
            o.tensor_split.clone(),
            model.and_then(|p| p.tensor_split.clone()),
            defaults.tensor_split.clone(),
            None,
        );
        let (log_verbosity, log_verbosity_src) = pick(
            o.log_verbosity,
            model.and_then(|p| p.log_verbosity),
            defaults.log_verbosity,
            Some(4),
        );
        let log_verbosity = log_verbosity.expect("log_verbosity has a builtin default");

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
        let (top_k, top_k_src) =
            pick(o.top_k, model.and_then(|p| p.top_k), defaults.top_k, None);
        let (min_p, min_p_src) = pick::<f64>(
            o.min_p,
            model.and_then(|p| p.min_p),
            defaults.min_p,
            None,
        );
        let (repeat_penalty, repeat_penalty_src) = pick(
            o.repeat_penalty,
            model.and_then(|p| p.repeat_penalty),
            defaults.repeat_penalty,
            None,
        );
        let (presence_penalty, presence_penalty_src) = pick(
            o.presence_penalty,
            model.and_then(|p| p.presence_penalty),
            defaults.presence_penalty,
            None,
        );
        let (frequency_penalty, frequency_penalty_src) = pick(
            o.frequency_penalty,
            model.and_then(|p| p.frequency_penalty),
            defaults.frequency_penalty,
            None,
        );
        let (seed, seed_src) =
            pick(o.seed, model.and_then(|p| p.seed), defaults.seed, None);
        let (api_key, api_key_src) = pick(
            o.api_key.clone(),
            model.and_then(|p| p.api_key.clone()),
            defaults.api_key.clone(),
            None,
        );
        let (timeout, timeout_src) = pick(
            o.timeout,
            model.and_then(|p| p.timeout),
            defaults.timeout,
            None,
        );
        let (threads_http, threads_http_src) = pick(
            o.threads_http,
            model.and_then(|p| p.threads_http),
            defaults.threads_http,
            None,
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
        if !draft_ctk.is_empty() {
            origins.push(Origin::new("draft_ctk", draft_ctk.clone(), draft_ctk_src));
        }
        if !draft_ctv.is_empty() {
            origins.push(Origin::new("draft_ctv", draft_ctv.clone(), draft_ctv_src));
        }
        if let Some(v) = ngl {
            origins.push(Origin::new("ngl", v.to_string(), ngl_src));
        }
        if let Some(v) = ncmoe {
            origins.push(Origin::new("ncmoe", v.to_string(), ncmoe_src));
        }
        if let Some(ref v) = load_mode {
            origins.push(Origin::new("load_mode", v.clone(), load_mode_src));
        }
        if let Some(v) = moe_cache_mib {
            origins.push(Origin::new("moe_cache_mib", v.to_string(), moe_cache_src));
        }
        let cpu_moe = cpu_moe.expect("cpu_moe has a builtin default");
        origins.push(Origin::new("cpu_moe", onoff(cpu_moe), cpu_moe_src));
        if let Some(v) = n_cpu_ffn {
            origins.push(Origin::new("n_cpu_ffn", v.to_string(), n_cpu_ffn_src));
        }
        if let Some(ref v) = override_tensor {
            origins.push(Origin::new("override_tensor", v.clone(), override_src));
        }
        if let Some(ref v) = lazy_mode {
            origins.push(Origin::new("lazy_mode", v.clone(), lazy_mode_src));
        }
        if let Some(ref v) = numa {
            origins.push(Origin::new("numa", v.clone(), numa_src));
        }
        let no_op_offload = no_op_offload.expect("no_op_offload has a builtin default");
        origins.push(Origin::new("no_op_offload", onoff(no_op_offload), no_op_src));
        let batch_size = batch_size.expect("batch_size has a builtin default");
        origins.push(Origin::new("batch_size", batch_size.to_string(), batch_src));
        let ubatch_size = ubatch_size.expect("ubatch_size has a builtin default");
        origins.push(Origin::new("ubatch_size", ubatch_size.to_string(), ubatch_src));
        let parallel = parallel.expect("parallel has a builtin default");
        origins.push(Origin::new("parallel", parallel.to_string(), parallel_src));
        if let Some(v) = threads_batch {
            origins.push(Origin::new("threads_batch", v.to_string(), threads_batch_src));
        }
        if let Some(v) = prio {
            origins.push(Origin::new("prio", v.to_string(), prio_src));
        }
        let cache_reuse = cache_reuse.expect("cache_reuse has a builtin default");
        origins.push(Origin::new("cache_reuse", cache_reuse.to_string(), cache_reuse_src));
        if let Some(v) = draft_n_min {
            origins.push(Origin::new("draft_n_min", v.to_string(), draft_n_min_src));
        }
        if let Some(v) = draft_p_min {
            origins.push(Origin::new("draft_p_min", format!("{v}"), draft_p_min_src));
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
        let kv_ram = kv_ram.expect("kv_ram has a builtin default");
        origins.push(Origin::new("kv_ram", onoff(kv_ram), kv_ram_src));
        origins.push(Origin::new("split_mode", split_mode.clone(), split_mode_src));
        if let Some(ref v) = tensor_split {
            origins.push(Origin::new("tensor_split", v.clone(), tensor_split_src));
        }
        origins.push(Origin::new(
            "log_verbosity",
            log_verbosity.to_string(),
            log_verbosity_src,
        ));
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
        if let Some(v) = top_k {
            origins.push(Origin::new("top_k", v.to_string(), top_k_src));
        }
        if let Some(v) = min_p {
            origins.push(Origin::new("min_p", format!("{v}"), min_p_src));
        }
        if let Some(v) = repeat_penalty {
            origins.push(Origin::new("repeat_penalty", format!("{v}"), repeat_penalty_src));
        }
        if let Some(v) = presence_penalty {
            origins.push(Origin::new("presence_penalty", format!("{v}"), presence_penalty_src));
        }
        if let Some(v) = frequency_penalty {
            origins.push(Origin::new("frequency_penalty", format!("{v}"), frequency_penalty_src));
        }
        if let Some(v) = seed {
            origins.push(Origin::new("seed", v.to_string(), seed_src));
        }
        if let Some(ref v) = api_key {
            origins.push(Origin::new("api_key", v.clone(), api_key_src));
        }
        if let Some(v) = timeout {
            origins.push(Origin::new("timeout", v.to_string(), timeout_src));
        }
        if let Some(v) = threads_http {
            origins.push(Origin::new("threads_http", v.to_string(), threads_http_src));
        }
        let host = host.expect("host has a builtin default");
        origins.push(Origin::new("host", host.clone(), host_src));

        let cfg = crate::Cfg {
            ctx,
            port,
            ngl,
            ncmoe,
            load_mode,
            moe_cache_mib,
            cpu_moe,
            n_cpu_ffn,
            override_tensor,
            lazy_mode,
            numa,
            no_op_offload,
            batch_size: Some(batch_size),
            ubatch_size: Some(ubatch_size),
            parallel: Some(parallel),
            threads_batch,
            prio,
            cache_reuse: Some(cache_reuse),
            draft_n_min,
            draft_p_min,
            ctk,
            ctv,
            draft_ctk,
            draft_ctv,
            threads,
            reasoning,
            effort,
            draft_tokens,
            extra,
            tools,
            draft_gpu,
            kv_ram,
            split_mode,
            tensor_split,
            log_verbosity,
            vision,
            mmproj,
            alias,
            flash_attn,
            temp,
            top_p,
            top_k,
            min_p,
            repeat_penalty,
            presence_penalty,
            frequency_penalty,
            seed,
            host,
            api_key,
            timeout,
            threads_http,
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
        assert_eq!(explained.draft_ctk, resolved.draft_ctk);
        assert_eq!(explained.draft_ctv, resolved.draft_ctv);
    }
}
