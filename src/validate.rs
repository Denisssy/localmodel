//! Checks raw `extra` flags against the installed `llama-server --help`.
//!
//! `extra` is deliberately open-ended: llama.cpp gains flags faster than we
//! add knobs. The cost used to be invisibility - a typo like `-kvod` on a
//! build without the patch, or `-ctkd q4_0` shadowing the typed `ctk` knob,
//! was only discovered at launch. Here the binary's own help is the schema.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::UNIX_EPOCH;

/// Flags that follow `flag` in a command line, or a raw flag string.
pub fn flags_in(args: &[String]) -> Vec<String> {
    args.iter()
        .filter(|t| t.starts_with('-') && t.chars().any(|c| c.is_ascii_alphabetic()))
        .map(|t| t.split('=').next().unwrap_or(t).trim_end_matches(',').to_string())
        .collect()
}

/// `extra` is a single whitespace-separated string, not argv. Values that look
/// like negative numbers (`-1`) are not flags; a value like `--foo=bar` is
/// reduced to `--foo`.
pub fn extra_flags(extra: &str) -> Vec<String> {
    let args: Vec<String> = extra.split_whitespace().map(String::from).collect();
    flags_in(&args)
}

/// Split a help line into its flag column. Descriptions start at the first
/// wide gap followed by a lowercase word or a parenthesis; before that, every
/// token starting with `-` is an alias of the same option.
fn line_flags(line: &str) -> Vec<String> {
    if !line.trim_start().starts_with('-') {
        return Vec::new();
    }
    // Find the end of the flag column.
    let bytes = line.as_bytes();
    let mut cut = line.len();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b' ' {
            let start = i;
            while i < bytes.len() && bytes[i] == b' ' {
                i += 1;
            }
            if i - start >= 2 {
                let rest = &line[i..];
                let next = rest.chars().next();
                let looks_like_description = matches!(
                    next,
                    Some(c) if c.is_ascii_lowercase() || c == '(' || c == '\''
                );
                if looks_like_description {
                    cut = start;
                    break;
                }
            }
        } else {
            i += 1;
        }
    }
    line[..cut]
        .split(|c: char| c.is_whitespace() || c == ',')
        .map(|t| t.trim())
        .filter(|t| t.starts_with('-') && t.chars().any(|c| c.is_ascii_alphabetic()))
        .map(|t| t.split('=').next().unwrap_or(t).to_string())
        .collect()
}

/// Alias groups from `llama-server --help`; `-c, --ctx-size N` is one group.
fn help_groups(help: &str) -> Vec<Vec<String>> {
    help.lines().map(line_flags).filter(|g| !g.is_empty()).collect()
}

/// Which knob a flag shadows. The help line supplies the aliases, the knob
/// registry supplies the canonical flag (`--ctx-size` -> `ctx`, `-c` too).
fn knob_index(groups: &[Vec<String>]) -> HashMap<String, &'static str> {
    let mut index = HashMap::new();
    for knob in crate::knobs::KNOBS {
        if knob.argv.is_empty() {
            continue;
        }
        for g in groups {
            if g.iter().any(|f| f == knob.argv) {
                for f in g {
                    index.insert(f.clone(), knob.key);
                }
            }
        }
    }
    index
}

fn check_text(argv: &[String], extra: &str, help: &str, active: &[&str]) -> Vec<String> {
    let groups = help_groups(help);
    let known: HashSet<&str> = groups.iter().flatten().map(String::as_str).collect();
    if known.is_empty() {
        return Vec::new();
    }
    let shadow = knob_index(&groups);
    let extra_flags = extra_flags(extra);
    let mut seen = HashSet::new();
    let mut out = Vec::new();

    for flag in flags_in(argv).into_iter().chain(extra_flags.iter().cloned()) {
        if !known.contains(flag.as_str()) {
            let msg = format!(
                "unknown flag `{flag}`: not in `llama-server --help` (typo, or this build lacks it)"
            );
            if seen.insert(msg.clone()) {
                out.push(msg);
            }
        }
    }

    for flag in &extra_flags {
        if let Some(knob) = shadow.get(flag) {
            let msg = if active.contains(knob) {
                format!("extra flag `{flag}` overrides the `{knob}` setting")
            } else {
                format!("extra flag `{flag}` duplicates the `{knob}` setting; use the knob instead")
            };
            if seen.insert(msg.clone()) {
                out.push(msg);
            }
        }
    }
    out
}

/// `~/.cache/localmodel/help-<mtime>.txt`, so a 2 MB help text is extracted
/// once per llama-server build, not on every launch.
fn help_cache_path(bin: &Path) -> Option<std::path::PathBuf> {
    let mtime = fs::metadata(bin)
        .ok()?
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_secs();
    Some(
        crate::config::home()
            .join(".cache")
            .join("localmodel")
            .join(format!("help-{mtime}.txt")),
    )
}

fn load_help(bin: &Path) -> Option<String> {
    let cache = help_cache_path(bin)?;
    if let Ok(text) = fs::read_to_string(&cache) {
        if !text.trim().is_empty() {
            return Some(text);
        }
    }
    let out = Command::new(bin).arg("--help").output().ok()?;
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    if text.trim().is_empty() {
        return None;
    }
    if let Some(dir) = cache.parent() {
        let _ = fs::create_dir_all(dir);
        // Keep only the cache for the current build; old ones are dead weight.
        if let Ok(entries) = fs::read_dir(dir) {
            for e in entries.flatten() {
                let name = e.file_name();
                if name.to_string_lossy().starts_with("help-") && e.path() != cache {
                    let _ = fs::remove_file(e.path());
                }
            }
        }
        let _ = fs::write(&cache, &text);
    }
    Some(text)
}

/// Warnings for `argv` plus raw `extra` flags. Empty when the binary or its
/// help is unavailable: absence of the checker must not block a launch.
pub fn check(argv: &[String], extra: &str, bin: &Path, active: &[&str]) -> Vec<String> {
    match load_help(bin) {
        Some(help) => check_text(argv, extra, &help, active),
        None => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HELP: &str = "\
----- common params -----

-h,    --help, --usage                  print usage and exit
-c,    --ctx-size N                     size of the prompt context (default: 0)
-b,    --batch-size N                   logical maximum batch size (default: 2048)
-ngl,  --gpu-layers N                   number of layers to store in VRAM
-t,    --threads N                      number of CPU threads to use
-fa,   --flash-attn [on|off|auto]       set Flash Attention use
--cache-type-k TYPE                     KV cache data type for K (default: f16)
--mmproj FILE                           path to a multimodal projector file
--jinja                                 enable Jinja2 chat template
--cache-reuse N                         min chunk size to reuse from the cache
--spec-type TYPE                        speculative decoding type
";

    #[test]
    fn aliases_land_in_one_group() {
        let groups = help_groups(HELP);
        let ctx = groups
            .iter()
            .find(|g| g.iter().any(|f| f == "--ctx-size"))
            .unwrap();
        assert_eq!(ctx, &["-c".to_string(), "--ctx-size".to_string()]);
        let help = groups.iter().find(|g| g.iter().any(|f| f == "--help")).unwrap();
        assert!(help.contains(&"--usage".to_string()));
    }

    #[test]
    fn extras_are_parsed_without_values() {
        let f = extra_flags("--batch-size 1024 --ubatch-size=256 -nkvo -1 --temp 0.6");
        assert_eq!(f, vec!["--batch-size", "--ubatch-size", "-nkvo", "--temp"]);
    }

    #[test]
    fn unknown_and_shadowed_flags_are_reported() {
        let argv = vec!["-m".to_string(), "x.gguf".to_string(), "--ctx-size".to_string(), "4096".to_string()];
        let extra = "--ctx-size 8192 --bogus 1 -nkvo";
        let out = check_text(&argv, extra, HELP, &["ctx"]);
        assert!(out.iter().any(|m| m.contains("unknown flag `--bogus`")), "{out:?}");
        assert!(out.iter().any(|m| m.contains("unknown flag `-nkvo`")), "{out:?}");
        assert!(out.iter().any(|m| m.contains("overrides the `ctx` setting")), "{out:?}");
        // A knob-generated flag is not an unknown one, and its own value does
        // not trigger the shadow rule: only `extra` can shadow.
        assert!(!out.iter().any(|m| m.contains("--ctx-size") && m.contains("unknown")));
    }

    #[test]
    fn unset_knob_is_a_duplicate_not_an_override() {
        let out = check_text(&[], "-t 6", HELP, &[]);
        assert!(
            out.iter().any(|m| m.contains("duplicates the `threads` setting")),
            "{out:?}"
        );
        assert!(!out.iter().any(|m| m.contains("overrides")), "{out:?}");
    }

    #[test]
    fn short_alias_shadows_the_long_knob_name() {
        let out = check_text(&[], "-c 4096", HELP, &["ctx"]);
        assert!(out.iter().any(|m| m.contains("overrides the `ctx` setting")), "{out:?}");
    }
}
