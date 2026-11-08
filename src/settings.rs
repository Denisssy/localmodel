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

pub fn load() -> Settings {
    let p = path();
    let Ok(text) = fs::read_to_string(&p) else {
        return Settings::default();
    };
    match toml::from_str(&text) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("warning: {} is invalid ({e}); using defaults", p.display());
            Settings::default()
        }
    }
}

pub fn save(s: &mut Settings) -> std::io::Result<PathBuf> {
    s.model.retain(|_, p| !p.is_empty());
    let p = path();
    if let Some(dir) = p.parent() {
        fs::create_dir_all(dir)?;
    }
    let text = toml::to_string_pretty(s)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let header = "# localmodel settings - edit with `localmodel settings`\n\
                  # [defaults] applies to every model; [model.NAME] overrides per model.\n\
                  # Strata hub packs use family/size: [model.\"qwen36/ud_q4_k_xl\"].\n\
                  # Model names containing dots must be quoted: [model.\"name.with.dots\"]\n\n";
    fs::write(&p, format!("{header}{text}"))?;
    Ok(p)
}
