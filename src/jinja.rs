//! Reads the chat template embedded in a GGUF and derives which
//! reasoning-effort knobs the model actually understands. Templates differ:
//! Qwen3.8 takes `reasoning_effort` ('xhigh'|'medium'|'low'), Nemotron takes a
//! boolean `medium_effort`. Passing the wrong variable is silently ignored -
//! or, worse, raises inside the template - so we look before we pass.

use serde_json::{json, Value};
use std::path::Path;

#[derive(Default, Debug, Clone)]
pub struct TemplateCaps {
    /// `reasoning_effort` values the template renders distinctly
    /// (xhigh/medium/low/...), collected from quoted literals on lines that
    /// mention `reasoning_effort`.
    pub effort_values: Vec<String>,
    pub has_medium_effort: bool,
    pub has_enable_thinking: bool,
    /// The template reads `reasoning_effort` (Qwen style).
    pub has_reasoning_effort: bool,
    /// The template reads `reasoning_strength` (Muse style). llama.cpp
    /// mirrors REQUEST-level reasoning_effort into it, but server-default
    /// `--chat-template-kwargs` are applied verbatim - the default must use
    /// this exact name or the model never sees it.
    pub has_reasoning_strength: bool,
}

pub fn caps_from_template(t: &str) -> TemplateCaps {
    const LEVELS: [&str; 7] = ["minimal", "low", "medium", "high", "xhigh", "max", "none"];
    let mut values: Vec<String> = Vec::new();
    for line in t.lines() {
        if !line.contains("reasoning_effort") {
            continue;
        }
        let mut rest = line;
        while let Some(i) = rest.find('\'') {
            let tail = &rest[i + 1..];
            let Some(j) = tail.find('\'') else { break };
            let w = &tail[..j];
            if LEVELS.contains(&w) && !values.iter().any(|v| v == w) {
                values.push(w.to_string());
            }
            rest = &tail[j + 1..];
        }
    }
    TemplateCaps {
        effort_values: values,
        has_medium_effort: t.contains("medium_effort"),
        has_enable_thinking: t.contains("enable_thinking"),
        has_reasoning_effort: t.contains("reasoning_effort"),
        has_reasoning_strength: t.contains("reasoning_strength"),
    }
}

/// `--chat-template-kwargs` for the requested effort level, or None when the
/// template has no matching knob (server then uses the template default).
pub fn effort_kwargs(caps: &TemplateCaps, effort: Option<&str>) -> Option<Value> {
    let e = effort?.trim().to_lowercase();
    if e.is_empty() || e == "default" || e == "auto" {
        return None;
    }
    if caps.has_medium_effort {
        // Nemotron-style: full thinking by default, 'medium_effort' selects
        // the efficient mode. There is no xhigh knob here.
        return Some(match e.as_str() {
            "medium" | "low" => json!({ "medium_effort": true }),
            "off" | "none" => json!({ "enable_thinking": false }),
            _ => json!({ "enable_thinking": true }),
        });
    }
    if caps.has_reasoning_effort {
        // Qwen-style. 'off' binds to enable_thinking when present.
        if e == "off" || e == "none" {
            return caps.has_enable_thinking.then(|| json!({ "enable_thinking": false }));
        }
        if caps.effort_values.is_empty() {
            return None;
        }
        let chosen = if caps.effort_values.iter().any(|v| *v == e) {
            e
        } else if e == "high" && caps.effort_values.iter().any(|v| v == "xhigh") {
            // qwen has no separate 'high': the template itself repprofiles
            // high to xhigh, so either spelling lands on the same render
            "xhigh".to_string()
        } else {
            return None; // unsupported level: keep the template default
        };
        return Some(json!({ "reasoning_effort": chosen }));
    }
    if caps.has_reasoning_strength {
        // Muse/Glimmer-style: strength goes into the system prompt; there is
        // no way to switch thinking off. Any canonical level is accepted
        // (the template compares the string and defaults to 'high').
        if e == "off" || e == "none" {
            return None;
        }
        return Some(json!({ "reasoning_strength": e }));
    }
    // enable_thinking-only templates (Ornith): only on/off.
    if e == "off" || e == "none" {
        return caps.has_enable_thinking.then(|| json!({ "enable_thinking": false }));
    }
    None
}

/// Minimal GGUF header reader lives in `gguf.rs`; this turns the template
/// text into capability flags.
pub fn caps_for_model(path: &Path) -> Option<TemplateCaps> {
    crate::gguf::chat_template(path).map(|t| caps_from_template(&t))
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN: &str = "{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}\n\
        {%- if resolved_reasoning_effort == 'high' %}\n    {%- set resolved_reasoning_effort = 'xhigh' %}\n\
        {%- endif %}\n{%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}\n\
        {{- raise_exception('Unexpected reasoning effort') }}\n{%- endif %}\n\
        {%- if enable_thinking is defined and enable_thinking is false %}{%- endif %}";

    const NEMOTRON: &str = "{%- set enable_thinking = enable_thinking if enable_thinking is defined else True %}\n\
        {%- set medium_effort = medium_effort if medium_effort is defined else False %}";

    #[test]
    fn qwen_caps_and_kwargs() {
        let caps = caps_from_template(QWEN);
        assert!(caps.effort_values.contains(&"xhigh".to_string()));
        assert!(caps.effort_values.contains(&"medium".to_string()));
        assert!(caps.effort_values.contains(&"low".to_string()));
        assert!(!caps.has_medium_effort);
        assert_eq!(
            effort_kwargs(&caps, Some("medium")),
            Some(json!({ "reasoning_effort": "medium" }))
        );
        // generic 'high' is one of the template's literals and it maps to
        // xhigh inside the template itself, so it passes through unchanged
        assert_eq!(
            effort_kwargs(&caps, Some("high")),
            Some(json!({ "reasoning_effort": "high" }))
        );
        assert_eq!(
            effort_kwargs(&caps, Some("off")),
            Some(json!({ "enable_thinking": false }))
        );
        assert_eq!(effort_kwargs(&caps, Some("max")), None);
        assert_eq!(effort_kwargs(&caps, None), None);
    }

    #[test]
    fn nemotron_caps_and_kwargs() {
        let caps = caps_from_template(NEMOTRON);
        assert!(caps.has_medium_effort);
        assert!(caps.effort_values.is_empty());
        // 'high' means full thinking: no downgrade kwarg, just thinking on
        assert_eq!(
            effort_kwargs(&caps, Some("high")),
            Some(json!({ "enable_thinking": true }))
        );
        assert_eq!(
            effort_kwargs(&caps, Some("medium")),
            Some(json!({ "medium_effort": true }))
        );
        assert_eq!(
            effort_kwargs(&caps, Some("off")),
            Some(json!({ "enable_thinking": false }))
        );
    }

    #[test]
    fn enable_thinking_only_template_binds_off() {
        // Ornith-style: no reasoning_effort literals, only the bool switch.
        const ORNITH: &str = "{%- if enable_thinking is defined and enable_thinking is false %}<|im_start|>user\n{%- endif %}";
        let caps = caps_from_template(ORNITH);
        assert!(caps.effort_values.is_empty());
        assert!(caps.has_enable_thinking);
        assert!(!caps.has_medium_effort);
        // the regression: 'off' must still reach enable_thinking
        assert_eq!(
            effort_kwargs(&caps, Some("off")),
            Some(json!({ "enable_thinking": false }))
        );
        assert_eq!(effort_kwargs(&caps, Some("medium")), None);
        assert_eq!(effort_kwargs(&caps, None), None);
    }

    #[test]
    fn glimmer_strength_knob_uses_its_own_name() {
        // Muse template reads `reasoning_strength`, default 'high'; no
        // reasoning_effort anywhere. Server-default kwargs are verbatim, so
        // the kwarg must be named reasoning_strength or the model never
        // sees it (request-level reasoning_effort is mirrored by llama.cpp).
        const GLIMMER: &str = "{%- set rs = reasoning_strength if reasoning_strength is defined and reasoning_strength else 'high' -%}\n{{ rs }}";
        let caps = caps_from_template(GLIMMER);
        assert!(!caps.has_reasoning_effort);
        assert!(caps.has_reasoning_strength);
        assert!(!caps.has_enable_thinking);
        assert_eq!(
            effort_kwargs(&caps, Some("medium")),
            Some(json!({ "reasoning_strength": "medium" }))
        );
        assert_eq!(
            effort_kwargs(&caps, Some("xhigh")),
            Some(json!({ "reasoning_strength": "xhigh" }))
        );
        // no way to switch thinking off on this template
        assert_eq!(effort_kwargs(&caps, Some("off")), None);
        assert_eq!(effort_kwargs(&caps, None), None);
    }

    #[test]
    fn real_gguf_templates_on_this_machine() {
        // Guarded: only runs where the user's models exist. Keeps the caps
        // mapping honest against the actual Qwen3.8 / Nemotron templates.
        let home = std::env::var("HOME").unwrap_or_default();
        let q = std::path::Path::new(&home).join("models/Qwen3.8-27B-UD-Q3_K_XL.gguf");
        if q.exists() {
            let caps = caps_for_model(&q).expect("qwen3.8 template");
            assert!(caps.effort_values.contains(&"xhigh".to_string()));
            assert!(caps.effort_values.contains(&"medium".to_string()));
            assert!(!caps.has_medium_effort);
            assert_eq!(
                effort_kwargs(&caps, Some("medium")),
                Some(json!({ "reasoning_effort": "medium" }))
            );
        }
        let n = std::path::Path::new(&home).join("models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf");
        if n.exists() {
            let caps = caps_for_model(&n).expect("nemotron template");
            assert!(caps.has_medium_effort);
            assert_eq!(
                effort_kwargs(&caps, Some("high")),
                Some(json!({ "enable_thinking": true }))
            );
        }
    }
}
