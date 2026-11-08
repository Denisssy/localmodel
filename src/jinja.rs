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
    if caps.effort_values.is_empty() {
        return None;
    }
    if e == "off" || e == "none" {
        return caps.has_enable_thinking.then(|| json!({ "enable_thinking": false }));
    }
    let chosen = if caps.effort_values.iter().any(|v| *v == e) {
        e
    } else if e == "high" && caps.effort_values.iter().any(|v| v == "xhigh") {
        "xhigh".to_string()
    } else {
        return None; // unsupported level: keep the template default
    };
    Some(json!({ "reasoning_effort": chosen }))
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
