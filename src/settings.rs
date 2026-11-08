use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

/// Launch settings for one scope: `defaults` or a single model.
/// The struct itself is generated from the knob registry (`knobs.rs`), so its
/// fields, serde form, `is_empty` and `overlay` stay in sync by construction.
pub use crate::knobs::Profile;

#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    #[serde(skip_serializing_if = "Profile::is_empty")]
    pub defaults: Profile,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub model: BTreeMap<String, Profile>,
}

impl Settings {
    /// Effective profile for a model: defaults overlaid with its own entry.
    pub fn for_model(&self, model: &str) -> Profile {
        match self.model.get(model) {
            Some(p) => self.defaults.overlay(p),
            None => self.defaults.clone(),
        }
    }
}

/// Case, `-` and `_` do not tell settings keys apart: the same Strata pack is
/// written `qwen36/ud_q4_k_xl` by hand and `qwen36/ud-q4_k_xl` by the hub.
pub fn norm_key(k: &str) -> String {
    k.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// The saved entry that applies to `wanted`: an exact key first, then one that
/// differs only in case or in `-` vs `_`.
pub fn find_key<'a>(s: &'a Settings, wanted: &str) -> Option<&'a str> {
    if s.model.contains_key(wanted) {
        return s.model.get_key_value(wanted).map(|(k, _)| k.as_str());
    }
    let n = norm_key(wanted);
    s.model
        .keys()
        .find(|k| norm_key(k) == n)
        .map(|k| k.as_str())
}

pub fn path() -> PathBuf {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| crate::config::home().join(".config"));
    base.join("localmodel").join("settings.toml")
}

/// Parse settings table by table. A typo in one `[model.X]` costs only that
/// profile, with a warning naming it, instead of silently dropping every
/// profile the user has.
///
/// Returns the parsed settings plus one warning per ignored table. A TOML
/// *syntax* error still fails the whole file: there is nothing reliable to
/// parse in that case.
fn parse_settings(text: &str) -> Result<(Settings, Vec<String>), toml::de::Error> {
    let value: toml::Value = toml::from_str(text)?;
    let mut st = Settings::default();
    let mut warnings = Vec::new();

    if let Some(v) = value.get("defaults") {
        match v.clone().try_into::<Profile>() {
            Ok(p) => st.defaults = p,
            Err(e) => warnings.push(format!("[defaults] ignored: {e}")),
        }
    }
    if let Some(v) = value.get("model") {
        match v.as_table() {
            Some(t) => {
                for (name, tv) in t {
                    match tv.clone().try_into::<Profile>() {
                        Ok(p) => {
                            st.model.insert(name.clone(), p);
                        }
                        Err(e) => warnings.push(format!("[model.\"{name}\"] ignored: {e}")),
                    }
                }
            }
            None => warnings.push("[model] must be a table of profiles; ignored".into()),
        }
    }
    if let Some(t) = value.as_table() {
        for k in t.keys() {
            if k != "defaults" && k != "model" {
                warnings.push(format!("unknown top-level table [{k}] ignored"));
            }
        }
    }
    Ok((st, warnings))
}

pub fn load() -> Settings {
    let p = path();
    let Ok(text) = fs::read_to_string(&p) else {
        return Settings::default();
    };
    match parse_settings(&text) {
        Ok((s, warnings)) => {
            for w in warnings {
                eprintln!("warning: {}: {w}", p.display());
            }
            s
        }
        Err(e) => {
            eprintln!("warning: {} is invalid ({e}); using defaults", p.display());
            Settings::default()
        }
    }
}

/// Keys that differ between two profiles, in registry order.
fn changed_keys(old: &Profile, new: &Profile) -> Vec<&'static str> {
    let mut keys = Vec::new();
    if old.applies_to != new.applies_to {
        keys.push("applies_to");
    }
    for k in crate::knobs::KEYS {
        if old.get(k) != new.get(k) {
            keys.push(*k);
        }
    }
    keys
}

fn invalid(msg: impl std::fmt::Display) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg.to_string())
}

/// Replace one key, keeping the indentation and any trailing comment that was
/// attached to the old value. Comments before a key live on the key itself and
/// survive because the key entry is not recreated.
fn set_value(table: &mut toml_edit::Table, key: &str, item: toml_edit::Item) {
    match item {
        toml_edit::Item::Value(mut newv) => {
            if let Some(old) = table.get_mut(key) {
                if let Some(ov) = old.as_value() {
                    *newv.decor_mut() = ov.decor().clone();
                }
                *old = toml_edit::Item::Value(newv);
            } else {
                table.insert(key, toml_edit::Item::Value(newv));
            }
        }
        other => {
            table.insert(key, other);
        }
    }
}

/// Write only the keys that changed, so comments, ordering and untouched
/// values in a hand-written settings.toml survive a save from the TUI.
pub fn save(s: &mut Settings) -> std::io::Result<PathBuf> {
    let p = path();
    save_to(&p, s)?;
    Ok(p)
}

fn save_to(p: &std::path::Path, s: &mut Settings) -> std::io::Result<()> {
    s.model.retain(|_, p| !p.is_empty());
    if let Some(dir) = p.parent() {
        fs::create_dir_all(dir)?;
    }
    let text = fs::read_to_string(p).unwrap_or_default();

    // Never overwrite a file we cannot parse: the in-memory copy is defaults,
    // and writing it would erase the user's profiles. Per-table problems are
    // fine: those tables are left untouched by the diff below.
    let old: Settings = if text.trim().is_empty() {
        Settings::default()
    } else {
        parse_settings(&text)
            .map_err(|e| invalid(format!("refusing to overwrite {}: {e}", p.display())))?
            .0
    };

    let header = "# localmodel settings - edit with `localmodel settings`\n\
                  # [defaults] applies to every model; [model.NAME] overrides per model.\n\
                  # Strata hub packs use family/size: [model.\"qwen36/ud_q4_k_xl\"].\n\
                  # Model names containing dots must be quoted: [model.\"name.with.dots\"]\n\n";
    let mut doc = if text.trim().is_empty() {
        header.parse::<toml_edit::DocumentMut>().map_err(invalid)?
    } else {
        text.parse::<toml_edit::DocumentMut>().map_err(invalid)?
    };

    // --- [defaults] -----------------------------------------------------
    let mut dirty = false;
    if s.defaults != old.defaults {
        dirty = true;
        let root = doc.as_table_mut();
        if s.defaults.is_empty() {
            root.remove("defaults");
        } else {
            let item = root
                .entry("defaults")
                .or_insert(toml_edit::Item::Table(toml_edit::Table::new()));
            let table = item
                .as_table_mut()
                .ok_or_else(|| invalid("[defaults] is not a table"))?;
            let serialized = toml::to_string(&s.defaults).map_err(invalid)?;
            let tmp = serialized
                .parse::<toml_edit::DocumentMut>()
                .map_err(invalid)?;
            for key in changed_keys(&old.defaults, &s.defaults) {
                match s.defaults.get(key) {
                    Some(_) => {
                        let it = tmp
                            .as_table()
                            .get(key)
                            .cloned()
                            .ok_or_else(|| invalid(format!("missing serialized key '{key}'")))?;
                        set_value(table, key, it);
                    }
                    None => {
                        table.remove(key);
                    }
                }
            }
        }
    }

    // --- [model.*] ------------------------------------------------------
    let model_changed = s.model != old.model;
    if model_changed {
        dirty = true;
        let root = doc.as_table_mut();
        if s.model.is_empty() {
            root.remove("model");
        } else {
            let item = root
                .entry("model")
                .or_insert(toml_edit::Item::Table(toml_edit::Table::new()));
            let model_table = item
                .as_table_mut()
                .ok_or_else(|| invalid("[model] is not a table"))?;

            // Profiles deleted in memory disappear from the file; their
            // comments go with them, which is what deleting a profile means.
            // A table that failed to parse is not in `old`, so it is left
            // alone for the user to fix rather than deleted silently.
            let existing: Vec<String> = model_table.iter().map(|(k, _)| k.to_string()).collect();
            for k in existing {
                if old.model.contains_key(&k) && !s.model.contains_key(&k) {
                    model_table.remove(&k);
                }
            }

            for (name, prof) in s.model.iter() {
                let oldp = old.model.get(name).cloned().unwrap_or_default();
                if old.model.get(name) == Some(prof) {
                    continue;
                }
                let item = model_table
                    .entry(name)
                    .or_insert(toml_edit::Item::Table(toml_edit::Table::new()));
                let table = item
                    .as_table_mut()
                    .ok_or_else(|| invalid(format!("[model.{name}] is not a table")))?;
                let serialized = toml::to_string(prof).map_err(invalid)?;
                let tmp = serialized
                    .parse::<toml_edit::DocumentMut>()
                    .map_err(invalid)?;
                for key in changed_keys(&oldp, prof) {
                    match prof.get(key) {
                        Some(_) => {
                            let it = tmp
                                .as_table()
                                .get(key)
                                .cloned()
                                .ok_or_else(|| {
                                    invalid(format!("missing serialized key '{key}'"))
                                })?;
                            set_value(table, key, it);
                        }
                        None => {
                            table.remove(key);
                        }
                    }
                }
            }
            if model_table.is_empty() {
                root.remove("model");
            }
        }
    }

    if !dirty {
        return Ok(());
    }
    fs::write(p, doc.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(tag: &str) -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "localmodel-settings-{}-{}-{tag}",
            std::process::id(),
            n
        ));
        fs::create_dir_all(&dir).unwrap();
        dir.join("settings.toml")
    }

    #[test]
    fn save_preserves_comments_and_quotes_dotted_names() {
        let p = temp_path("comments");
        fs::write(
            &p,
            "# hand-written header\n\n[defaults]\n# ctx note\nctx = 131072  # inline\n\n[model.\"Old.Name\"]\nctx = 4096\n",
        )
        .unwrap();

        let mut st = Settings::default();
        st.defaults.ctx = Some(262144);
        let mut prof = Profile::default();
        prof.ctx = Some(8192);
        prof.ctk = Some("q4_0".into());
        st.model.insert("New.Name-X".into(), prof);

        save_to(&p, &mut st).unwrap();
        let text = fs::read_to_string(&p).unwrap();
        assert!(text.contains("# hand-written header"), "{text}");
        assert!(text.contains("# ctx note"), "{text}");
        assert!(text.contains("# inline"), "{text}");
        assert!(text.contains("[model.\"New.Name-X\"]"), "{text}");
        assert!(!text.contains("Old.Name"), "{text}");

        let back: Settings = toml::from_str(&text).unwrap();
        assert_eq!(back.defaults.ctx, Some(262144));
        assert_eq!(back.model["New.Name-X"].ctk.as_deref(), Some("q4_0"));
        fs::remove_dir_all(p.parent().unwrap()).ok();
    }

    #[test]
    fn one_bad_profile_does_not_hide_the_rest() {
        let text = "[defaults]\nctx = 4096\n\n[model.\"Good\"]\nctx = 8192\n\n[model.\"Bad\"]\nctk = \"q4_0\"\nnope = 1\n";
        let (st, warnings) = parse_settings(text).unwrap();
        assert_eq!(st.defaults.ctx, Some(4096));
        assert_eq!(st.model["Good"].ctx, Some(8192));
        assert!(!st.model.contains_key("Bad"));
        assert!(warnings.iter().any(|w| w.contains("Bad")), "{warnings:?}");
    }

    #[test]
    fn save_keeps_a_profile_that_failed_to_parse() {
        let p = temp_path("keep-bad");
        let original = "[model.\"Bad\"]\nctk = \"q4_0\"\nnope = 1\n";
        fs::write(&p, original).unwrap();
        let mut st = Settings::default();
        st.defaults.ctx = Some(4096);
        save_to(&p, &mut st).unwrap();
        let text = fs::read_to_string(&p).unwrap();
        // The broken table is untouched, so the typo can be fixed by hand.
        assert!(text.contains("nope = 1"), "{text}");
        assert!(text.contains("[defaults]"), "{text}");
        fs::remove_dir_all(p.parent().unwrap()).ok();
    }

    #[test]
    fn save_refuses_to_overwrite_an_invalid_file() {
        let p = temp_path("invalid");
        let original = "[defaults]\nctx = 12x\n";
        fs::write(&p, original).unwrap();
        let mut st = Settings::default();
        let err = save_to(&p, &mut st).unwrap_err();
        assert!(err.to_string().contains("refusing to overwrite"), "{err}");
        assert_eq!(fs::read_to_string(&p).unwrap(), original);
        fs::remove_dir_all(p.parent().unwrap()).ok();
    }
}
