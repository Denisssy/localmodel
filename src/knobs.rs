//! Registry of launch settings: the single source of truth.
//!
//! Every knob is declared once in the `profile_knobs!` table below. The macro
//! emits the `Profile` struct (with serde + `deny_unknown_fields`),
//! `is_empty`, `overlay`, keyed `get`/`set` accessors and the `KNOBS`
//! metadata that the hub renders.
//!
//! Before this table the same field was spelled out in eight places: the
//! struct, `is_empty`, `overlay`, `Cfg`, `Cfg::resolve`, `build_args`, the
//! dialoguer editor and the dashboard table. Adding a knob meant eight edits,
//! and forgetting one silently dropped the value (a field missing from
//! `overlay` never applies; a field missing from `is_empty` never saves).
//!
//! Adding a setting now is one row. Keep the row order: it is also the order
//! of keys in `settings.toml`, so saved profiles stay diff-free.

use serde::{Deserialize, Serialize};

/// Cache types llama.cpp accepts for `--cache-type-k/v`.
pub const KV_TYPES: [&str; 9] = ["f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl", "q6_k"];
const KV_ENUM: &[&str] = &KV_TYPES;

/// Reasoning-effort values the templates understand (`""` means unset and is
/// not offered as a choice; `-` in the hub clears the override).
const EFFORT_ENUM: &[&str] = &["high", "xhigh", "medium", "low", "off"];

/// Where a knob belongs on the hub screen.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Group {
    Context,
    Memory,
    Speed,
    Sampling,
    Server,
    Advanced,
}

impl Group {
    pub const ORDER: [Group; 6] = [
        Group::Context,
        Group::Memory,
        Group::Speed,
        Group::Sampling,
        Group::Server,
        Group::Advanced,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Group::Context => "CONTEXT",
            Group::Memory => "MEMORY / OFFLOAD",
            Group::Speed => "SPEED",
            Group::Sampling => "SAMPLING",
            Group::Server => "SERVER",
            Group::Advanced => "ADVANCED",
        }
    }
}

/// How a value is edited and validated.
#[derive(Clone, Copy)]
pub enum Kind {
    Bool,
    Num { min: i64, max: i64, unit: &'static str },
    Float { min: f64, max: f64 },
    Enum(&'static [&'static str]),
    Text,
    Path,
    Flags,
}

impl Kind {
    /// One-line hint shown while editing.
    pub fn hint(self) -> &'static str {
        match self {
            Kind::Bool => "on / off",
            Kind::Num { .. } => "number",
            Kind::Float { .. } => "number, e.g. 0.95",
            Kind::Enum(_) => "Tab cycles",
            Kind::Text => "text, '-' clears",
            Kind::Path => "path, '-' clears",
            Kind::Flags => "raw llama.cpp flags",
        }
    }

    /// Numeric bounds as text, for the hub's edit line ("256..8388608").
    pub fn bounds(&self) -> Option<String> {
        match *self {
            Kind::Num { min, max, unit } if unit.is_empty() => Some(format!("{min}..{max}")),
            Kind::Num { min, max, unit } => Some(format!("{min}..{max} {unit}")),
            Kind::Float { min, max } => Some(format!("{min}..{max}")),
            _ => None,
        }
    }
}

/// Model traits that decide whether a knob is worth showing at all.
#[derive(Clone, Copy)]
pub enum Applies {
    All,
    /// MoE only: `--n-cpu-moe` does nothing for a dense model.
    Moe,
    /// Any speculative decoding: built-in MTP or an external draft file.
    Draft,
    /// Vision input (mmproj).
    Vision,
}

/// What we know about a model before launch.
#[derive(Clone, Copy, Default)]
pub struct Caps {
    pub moe: bool,
    pub nextn: bool,
    pub draft: bool,
    /// A multimodal projector is configured.
    pub vision: bool,
}

impl Applies {
    pub fn check(self, c: &Caps) -> bool {
        match self {
            Applies::All => true,
            Applies::Moe => c.moe,
            Applies::Draft => c.nextn || c.draft,
            Applies::Vision => c.vision,
        }
    }
}

/// Metadata for one knob: what the hub renders and `settings --show --origin`
/// explains. Kept next to the value so the two cannot drift.
#[derive(Clone, Copy)]
pub struct Knob {
    pub key: &'static str,
    pub label: &'static str,
    pub group: Group,
    pub kind: Kind,
    pub applies: Applies,
    /// Primary llama-server flag this knob maps to ("" = localmodel-only).
    pub argv: &'static str,
    /// One line under the cursor.
    pub help: &'static str,
}

/// Typed access used by the generated `get`/`set`, so `--set`/the hub never
/// need to know the field list.
pub trait KnobValue: Clone {
    fn to_text(&self) -> String;
    fn from_text(s: &str) -> Result<Self, String>
    where
        Self: Sized;
}

impl KnobValue for u32 {
    fn to_text(&self) -> String {
        self.to_string()
    }
    fn from_text(s: &str) -> Result<Self, String> {
        s.parse().map_err(|_| format!("expected a whole number, got '{s}'"))
    }
}

impl KnobValue for u16 {
    fn to_text(&self) -> String {
        self.to_string()
    }
    fn from_text(s: &str) -> Result<Self, String> {
        s.parse().map_err(|_| format!("expected a whole number, got '{s}'"))
    }
}

impl KnobValue for i32 {
    fn to_text(&self) -> String {
        self.to_string()
    }
    fn from_text(s: &str) -> Result<Self, String> {
        s.parse().map_err(|_| format!("expected a whole number, got '{s}'"))
    }
}

impl KnobValue for f64 {
    fn to_text(&self) -> String {
        // Match the argv spelling: 1.0 prints as "1", 0.95 as "0.95".
        format!("{self}")
    }
    fn from_text(s: &str) -> Result<Self, String> {
        s.parse().map_err(|_| format!("expected a number, got '{s}'"))
    }
}

impl KnobValue for bool {
    fn to_text(&self) -> String {
        if *self { "on" } else { "off" }.to_string()
    }
    fn from_text(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "on" | "true" | "yes" | "1" => Ok(true),
            "off" | "false" | "no" | "0" => Ok(false),
            _ => Err(format!("expected on/off, got '{s}'")),
        }
    }
}

impl KnobValue for String {
    fn to_text(&self) -> String {
        self.clone()
    }
    fn from_text(s: &str) -> Result<Self, String> {
        Ok(s.to_string())
    }
}

macro_rules! profile_knobs {
    ($(
        $key:ident : $ty:ty
            , $label:literal
            , $group:expr
            , $kind:expr
            , $applies:expr
            , $argv:literal
            , $help:literal
        ;
    )*) => {
        /// One scope of launch settings: `[defaults]` or `[model.NAME]`.
        /// Every field is optional so a profile overrides just what it needs.
        #[derive(Default, Clone, PartialEq, Serialize, Deserialize)]
        #[serde(default, deny_unknown_fields)]
        pub struct Profile {
            /// model file stem this profile applies to; set on named side
            /// profiles (e.g. an "agent" variant) so the picker lists them
            pub applies_to: Option<String>,
            $( pub $key: Option<$ty>, )*
        }

        impl Profile {
            pub fn is_empty(&self) -> bool {
                self.applies_to.is_none() $( && self.$key.is_none() )*
            }

            /// `self` supplies fallbacks, `over` wins where set.
            pub fn overlay(&self, over: &Profile) -> Profile {
                Profile {
                    applies_to: over.applies_to.clone().or_else(|| self.applies_to.clone()),
                    $( $key: over.$key.clone().or_else(|| self.$key.clone()), )*
                }
            }

            /// Effective value of one knob as text; `None` means "not set here".
            pub fn get(&self, key: &str) -> Option<String> {
                match key {
                    "applies_to" => self.applies_to.clone(),
                    $( stringify!($key) => self.$key.as_ref().map(KnobValue::to_text), )*
                    _ => None,
                }
            }

            /// Set one knob from text. `""` and `"-"` clear it, so a value can
            /// always be removed without editing TOML by hand.
            pub fn set(&mut self, key: &str, raw: &str) -> Result<(), String> {
                let raw = raw.trim();
                match key {
                    "applies_to" => {
                        self.applies_to =
                            if raw.is_empty() || raw == "-" { None } else { Some(raw.to_string()) };
                        Ok(())
                    }
                    $(
                        stringify!($key) => {
                            if raw.is_empty() || raw == "-" {
                                self.$key = None;
                                return Ok(());
                            }
                            match <$ty as KnobValue>::from_text(raw) {
                                Ok(v) => {
                                    self.$key = Some(v);
                                    Ok(())
                                }
                                Err(e) => Err(e),
                            }
                        }
                    )*
                    _ => Err(format!("unknown setting '{key}'")),
                }
            }
        }

        /// Knob keys in declaration order, i.e. the order they appear in
        /// `settings.toml`.
        pub const KEYS: &[&str] = &[ $( stringify!($key) ),* ];

        /// Metadata for the hub and `settings --show --origin`.
        pub const KNOBS: &[Knob] = &[ $( Knob {
            key: stringify!($key),
            label: $label,
            group: $group,
            kind: $kind,
            applies: $applies,
            argv: $argv,
            help: $help,
        } ),* ];
    };
}

profile_knobs! {
    ctx : u32
        , "context size"
        , Group::Context
        , Kind::Num { min: 256, max: 8_388_608, unit: "tok" }
        , Applies::All
        , "--ctx-size"
        , "Context window in tokens. KV memory grows linearly with it.";
    ctk : String
        , "KV cache K"
        , Group::Context
        , Kind::Enum(KV_ENUM)
        , Applies::All
        , "--cache-type-k"
        , "Precision of the K half of the cache; q8_0 is the usual sweet spot.";
    ctv : String
        , "KV cache V"
        , Group::Context
        , Kind::Enum(KV_ENUM)
        , Applies::All
        , "--cache-type-v"
        , "Precision of the V half; unset means 'same as K'.";
    draft_ctk : String
        , "KV draft/MTP K"
        , Group::Context
        , Kind::Enum(KV_ENUM)
        , Applies::Draft
        , "--cache-type-k-draft"
        , "Precision of the draft/MTP context K; empty keeps llama.cpp's f16.";
    draft_ctv : String
        , "KV draft/MTP V"
        , Group::Context
        , Kind::Enum(KV_ENUM)
        , Applies::Draft
        , "--cache-type-v-draft"
        , "Precision of the draft/MTP context V; empty keeps llama.cpp's f16.";
    ngl : u16
        , "GPU layers"
        , Group::Memory
        , Kind::Num { min: 0, max: 999, unit: "" }
        , Applies::All
        , "-ngl"
        , "Whole transformer blocks offloaded to the GPU; 99/999 mean all of them. This does not decide where MoE experts live - use --n-cpu-moe for that.";
    ncmoe : u16
        , "experts on CPU"
        , Group::Memory
        , Kind::Num { min: 0, max: 999, unit: "" }
        , Applies::Moe
        , "--n-cpu-moe"
        , "MoE only: keep the routed expert tensors of the first N blocks in host RAM, even when those blocks are offloaded. With mmap they are read from disk on demand; --load-mode none keeps them in RAM.";
    load_mode : String
        , "load mode"
        , Group::Memory
        , Kind::Enum(&["auto", "mmap", "none", "mlock", "mmap+mlock", "dio"])
        , Applies::All
        , "--load-mode"
        , "How weights are read: mmap pages them from disk on demand; none loads them into anonymous RAM (no SSD paging); mlock pins them; dio uses direct I/O.";
    moe_cache_mib : u32
        , "MoE cache MiB"
        , Group::Memory
        , Kind::Num { min: 0, max: 65536, unit: "MiB" }
        , Applies::Moe
        , "--moe-cache-mib"
        , "GPU cache for experts that stay on the CPU; hot experts stop being re-read from SSD. 0 disables it, 2048 is a useful start.";
    cpu_moe : bool
        , "all experts CPU"
        , Group::Memory
        , Kind::Bool
        , Applies::Moe
        , "--cpu-moe"
        , "Keep every routed expert in host RAM. The rest of each block can still be on the GPU.";
    n_cpu_ffn : u16
        , "dense FFN on CPU"
        , Group::Memory
        , Kind::Num { min: 0, max: 999, unit: "" }
        , Applies::All
        , "--n-cpu-ffn"
        , "Keep the dense feed-forward weights of the first N blocks in host RAM (dense models; MoE experts use n-cpu-moe).";
    override_tensor : String
        , "tensor overrides"
        , Group::Memory
        , Kind::Text
        , Applies::All
        , "--override-tensor"
        , "Fine placement, e.g. blk\\.1[0-9]\\.ffn_up=CPU or \\.ffn_exps\\.=CUDA0. Comma-separate several rules.";
    lazy_mode : String
        , "lazy tensors"
        , Group::Memory
        , Kind::Enum(&["on", "auto", "off"])
        , Applies::All
        , "--lazy-mode"
        , "Read large per-layer tensors from disk on demand (needs mmap); off keeps them resident.";
    numa : String
        , "NUMA"
        , Group::Memory
        , Kind::Enum(&["distribute", "isolate", "numactl"])
        , Applies::All
        , "--numa"
        , "NUMA strategy on multi-socket hosts; leave unset on a single socket.";
    no_op_offload : bool
        , "output on CPU"
        , Group::Memory
        , Kind::Bool
        , Applies::All
        , "--no-op-offload"
        , "Keep the output layer in host RAM instead of VRAM; frees a little card memory.";
    kv_ram : bool
        , "KV in RAM"
        , Group::Memory
        , Kind::Bool
        , Applies::All
        , "--no-kv-offload"
        , "Keep the KV cache in host RAM instead of VRAM. Frees the card, costs decode speed.";
    split_mode : String
        , "multi-GPU split"
        , Group::Memory
        , Kind::Enum(&["layer", "row", "tensor", "none"])
        , Applies::All
        , "--split-mode"
        , "How two cards share the model: layer is the safe default, row suits MoE.";
    tensor_split : String
        , "tensor split"
        , Group::Memory
        , Kind::Text
        , Applies::All
        , "--tensor-split"
        , "Per-card share, e.g. 0.57,0.43. Empty means an even split.";
    port : u16
        , "port"
        , Group::Server
        , Kind::Num { min: 1, max: 65535, unit: "" }
        , Applies::All
        , "--port"
        , "HTTP port; unset picks the first free one from 8080.";
    threads : u32
        , "threads"
        , Group::Speed
        , Kind::Num { min: 1, max: 256, unit: "" }
        , Applies::All
        , "--threads"
        , "CPU threads; physical cores usually beat SMT siblings.";
    threads_batch : u32
        , "batch threads"
        , Group::Speed
        , Kind::Num { min: 1, max: 256, unit: "" }
        , Applies::All
        , "--threads-batch"
        , "Threads used for prompt processing; empty reuses --threads.";
    batch_size : u32
        , "batch size"
        , Group::Speed
        , Kind::Num { min: 1, max: 65536, unit: "" }
        , Applies::All
        , "--batch-size"
        , "Logical batch for prompt processing; larger batches use more compute buffers.";
    ubatch_size : u32
        , "micro-batch"
        , Group::Speed
        , Kind::Num { min: 1, max: 65536, unit: "" }
        , Applies::All
        , "--ubatch-size"
        , "Physical micro-batch; must divide batch-size. 512 is a safe default.";
    parallel : u32
        , "parallel slots"
        , Group::Speed
        , Kind::Num { min: 1, max: 64, unit: "" }
        , Applies::All
        , "--parallel"
        , "Concurrent request slots; each slot gets its own KV cache, so memory grows with it.";
    prio : i32
        , "thread priority"
        , Group::Speed
        , Kind::Num { min: -1, max: 3, unit: "" }
        , Applies::All
        , "--prio"
        , "Process priority: low -1, normal 0, medium 1, high 2, realtime 3.";
    cache_reuse : u32
        , "cache reuse"
        , Group::Speed
        , Kind::Num { min: 0, max: 65536, unit: "tok" }
        , Applies::All
        , "--cache-reuse"
        , "Reuse a matching KV prefix in chunks of this size; 0 disables reuse. 256 skips re-prefill of repeated prompts.";
    draft_n_min : u16
        , "draft min tokens"
        , Group::Speed
        , Kind::Num { min: 0, max: 16, unit: "" }
        , Applies::Draft
        , "--spec-draft-n-min"
        , "Minimum tokens proposed per speculative step; unset uses the engine default.";
    draft_p_min : f64
        , "draft min prob"
        , Group::Speed
        , Kind::Float { min: 0.0, max: 1.0 }
        , Applies::Draft
        , "--spec-draft-p-min"
        , "Stop drafting when the draft probability drops below this; unset uses the engine default.";
    reasoning : String
        , "reasoning"
        , Group::Sampling
        , Kind::Enum(&["on", "off"])
        , Applies::All
        , "--reasoning"
        , "Show or hide the model's thinking in the response.";
    effort : String
        , "reasoning effort"
        , Group::Sampling
        , Kind::Enum(EFFORT_ENUM)
        , Applies::All
        , "--chat-template-kwargs"
        , "Template-level thinking budget (high/xhigh/medium/low).";
    draft_tokens : u16
        , "draft tokens"
        , Group::Speed
        , Kind::Num { min: 1, max: 16, unit: "" }
        , Applies::Draft
        , "--spec-draft-n-max"
        , "Tokens proposed per speculative step; 2-3 pays off for MTP.";
    draft_gpu : bool
        , "draft on GPU"
        , Group::Speed
        , Kind::Bool
        , Applies::Draft
        , "--spec-draft-ngl"
        , "Keep the external draft head on the GPU (needs a small ctx).";
    vision : bool
        , "vision input"
        , Group::Advanced
        , Kind::Bool
        , Applies::Vision
        , "--mmproj"
        , "Enable image input for multimodal models.";
    mmproj : String
        , "mmproj path"
        , Group::Advanced
        , Kind::Path
        , Applies::Vision
        , "--mmproj"
        , "Multimodal projector file; empty means --mmproj-auto.";
    extra : String
        , "extra flags"
        , Group::Advanced
        , Kind::Flags
        , Applies::All
        , ""
        , "Raw llama-server flags appended verbatim. They win over the knobs.";
    tools : String
        , "tools"
        , Group::Advanced
        , Kind::Text
        , Applies::All
        , "--tools"
        , "llama.cpp built-in agent tools (file/shell access).";
    alias : String
        , "served alias"
        , Group::Server
        , Kind::Text
        , Applies::All
        , "--alias"
        , "Model name reported by the /v1 API.";
    flash_attn : bool
        , "flash attention"
        , Group::Speed
        , Kind::Bool
        , Applies::All
        , "--flash-attn"
        , "Faster attention with less KV traffic; on by default.";
    temp : f64
        , "temperature"
        , Group::Sampling
        , Kind::Float { min: 0.0, max: 5.0 }
        , Applies::All
        , "--temp"
        , "Sampling temperature; 1.0 means the model's own scale.";
    top_p : f64
        , "top-p"
        , Group::Sampling
        , Kind::Float { min: 0.0, max: 1.0 }
        , Applies::All
        , "--top-p"
        , "Nucleus sampling cutoff; 0.95 is the common default.";
    top_k : u32
        , "top-k"
        , Group::Sampling
        , Kind::Num { min: 0, max: 100000, unit: "" }
        , Applies::All
        , "--top-k"
        , "Keep only the K most likely tokens; 0 disables the cutoff. 20 suits thinking models, 64 is a general default.";
    min_p : f64
        , "min-p"
        , Group::Sampling
        , Kind::Float { min: 0.0, max: 1.0 }
        , Applies::All
        , "--min-p"
        , "Drop tokens below this fraction of the top token's probability; 0 disables it.";
    repeat_penalty : f64
        , "repeat penalty"
        , Group::Sampling
        , Kind::Float { min: 0.0, max: 10.0 }
        , Applies::All
        , "--repeat-penalty"
        , "Penalty for tokens already in the context; 1.0 means no penalty.";
    presence_penalty : f64
        , "presence penalty"
        , Group::Sampling
        , Kind::Float { min: -2.0, max: 2.0 }
        , Applies::All
        , "--presence-penalty"
        , "Flat penalty for any token that already appeared; 0 disables it.";
    frequency_penalty : f64
        , "frequency penalty"
        , Group::Sampling
        , Kind::Float { min: -2.0, max: 2.0 }
        , Applies::All
        , "--frequency-penalty"
        , "Penalty proportional to how often a token appeared; 0 disables it.";
    seed : i32
        , "seed"
        , Group::Sampling
        , Kind::Num { min: -1, max: 2_147_483_647, unit: "" }
        , Applies::All
        , "--seed"
        , "Sampling seed; -1 draws a random one at startup.";
    host : String
        , "host"
        , Group::Server
        , Kind::Text
        , Applies::All
        , "--host"
        , "Bind address; 127.0.0.1 keeps the server local.";
    api_key : String
        , "API key"
        , Group::Server
        , Kind::Text
        , Applies::All
        , "--api-key"
        , "Require this key in the Authorization header; empty leaves the server open.";
    timeout : u32
        , "request timeout"
        , Group::Server
        , Kind::Num { min: 0, max: 86400, unit: "s" }
        , Applies::All
        , "--timeout"
        , "Stop a request after this many seconds; 0 means no limit.";
    threads_http : u32
        , "HTTP threads"
        , Group::Server
        , Kind::Num { min: 1, max: 256, unit: "" }
        , Applies::All
        , "--threads-http"
        , "Worker threads for HTTP handling; unset uses the engine default.";
    log_verbosity : u32
        , "log verbosity"
        , Group::Advanced
        , Kind::Num { min: 0, max: 4, unit: "" }
        , Applies::All
        , "--log-verbosity"
        , "4 prints the memory breakdown the placement card reads; 3 is quieter.";
}

/// Look up a knob by its key.
pub fn knob(key: &str) -> Option<&'static Knob> {
    KNOBS.iter().find(|k| k.key == key)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A value that fits every `Kind`, distinct for n=0 and n=1. The hub,
    /// `--set` and the provenance view all rely on values round-tripping.
    fn sample(k: &Knob, n: usize) -> String {
        match k.kind {
            Kind::Bool => if n == 0 { "on" } else { "off" }.into(),
            Kind::Num { min, .. } => (min + n as i64).to_string(),
            Kind::Float { min, .. } => format!("{}", min + n as f64 * 0.5),
            Kind::Enum(v) => v[n.min(v.len() - 1)].to_string(),
            Kind::Text => format!("text{n}"),
            Kind::Path => format!("/tmp/p{n}.gguf"),
            Kind::Flags => format!("--flag{n} 1"),
        }
    }

    #[test]
    fn table_and_struct_agree() {
        assert_eq!(KEYS.len(), KNOBS.len());
        for (k, key) in KNOBS.iter().zip(KEYS) {
            assert_eq!(k.key, *key);
        }
        let mut seen = std::collections::HashSet::new();
        for k in KNOBS {
            assert!(seen.insert(k.key), "duplicate knob {}", k.key);
            assert_eq!(knob(k.key).map(|x| x.key), Some(k.key));
            assert!(!k.help.is_empty(), "{} has no help", k.key);
        }
    }

    #[test]
    fn every_knob_round_trips_through_set_and_get() {
        for k in KNOBS {
            let mut p = Profile::default();
            let v = sample(k, 0);
            p.set(k.key, &v).unwrap_or_else(|e| panic!("{}: {e}", k.key));
            assert!(p.get(k.key).is_some(), "{} did not stick", k.key);
            // Two different values must stringify differently, otherwise the
            // provenance view cannot tell them apart.
            let mut q = Profile::default();
            q.set(k.key, &sample(k, 1)).unwrap_or_else(|e| panic!("{}: {e}", k.key));
            assert_ne!(p.get(k.key), q.get(k.key), "{} samples collide", k.key);
            // "-" clears.
            p.set(k.key, "-").unwrap();
            assert_eq!(p.get(k.key), None, "{} did not clear", k.key);
        }
    }

    #[test]
    fn applies_filter_matches_model_caps() {
        let visible = |c: Caps| -> Vec<&'static str> {
            KNOBS
                .iter()
                .filter(|k| k.applies.check(&c))
                .map(|k| k.key)
                .collect()
        };
        let dense = Caps::default();
        let shown = visible(dense);
        assert!(!shown.contains(&"ncmoe"), "ncmoe leaked into a dense model");
        assert!(!shown.contains(&"draft_tokens"), "draft leaked without a draft");
        assert!(!shown.contains(&"mmproj"), "vision leaked without an mmproj");
        assert!(shown.contains(&"ctx") && shown.contains(&"ctk"));

        let moe = Caps { moe: true, ..Caps::default() };
        assert!(visible(moe).contains(&"ncmoe"));

        let mtp = Caps { nextn: true, draft: true, ..Caps::default() };
        assert!(visible(mtp).contains(&"draft_tokens"));
        assert!(visible(mtp).contains(&"draft_gpu"));

        let vision = Caps { vision: true, ..Caps::default() };
        assert!(visible(vision).contains(&"vision"));
        assert!(visible(vision).contains(&"mmproj"));
    }

    #[test]
    fn overlay_covers_every_knob() {
        let mut base = Profile::default();
        let mut over = Profile::default();
        for k in KNOBS {
            base.set(k.key, &sample(k, 1)).unwrap();
            over.set(k.key, &sample(k, 0)).unwrap();
        }
        base.applies_to = Some("base".into());
        over.applies_to = Some("over".into());

        // The override wins where set...
        let merged = base.overlay(&over);
        for k in KNOBS {
            assert_eq!(merged.get(k.key), over.get(k.key), "{} lost the override", k.key);
        }
        assert_eq!(merged.applies_to.as_deref(), Some("over"));

        // ...and the base survives where the override is empty.
        let empty = Profile::default();
        let kept = base.overlay(&empty);
        for k in KNOBS {
            assert_eq!(kept.get(k.key), base.get(k.key), "{} lost the fallback", k.key);
        }
        assert_eq!(kept.applies_to.as_deref(), Some("base"));
    }

    #[test]
    fn toml_keys_keep_declaration_order() {
        // `settings.toml` is a hand-edited file: a refactor must not reshuffle
        // it. Serialization order == registry order == the original struct.
        let mut p = Profile::default();
        for k in KNOBS {
            p.set(k.key, &sample(k, 0)).unwrap();
        }
        let text = toml::to_string(&p).expect("serializes");
        let order: Vec<&str> = text
            .lines()
            .filter_map(|l| l.split('=').next())
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect();
        assert_eq!(order, KEYS.to_vec());
    }

    #[test]
    fn unknown_keys_are_rejected() {
        let mut p = Profile::default();
        assert!(p.set("nope", "1").is_err());
        assert!(p.set("ctx", "many").is_err());
        assert!(p.set("flash_attn", "maybe").is_err());
        assert!(p.set("ctx", "131072").is_ok());
        assert_eq!(p.ctx, Some(131072));
    }

    #[test]
    fn applies_filters_by_model_traits() {
        let dense = Caps { moe: false, nextn: false, draft: false, vision: false };
        let moe_mtp = Caps { moe: true, nextn: true, draft: false, vision: false };
        assert!(Applies::All.check(&dense));
        assert!(!Applies::Moe.check(&dense));
        assert!(!Applies::Draft.check(&dense));
        assert!(Applies::Moe.check(&moe_mtp));
        assert!(Applies::Draft.check(&moe_mtp));
    }
}
