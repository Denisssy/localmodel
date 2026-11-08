use std::env;
use std::path::PathBuf;

pub fn home() -> PathBuf {
    env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."))
}

pub fn models_dir() -> PathBuf {
    if let Some(d) = env::var_os("LOCALMODEL_DIR") {
        return PathBuf::from(d);
    }
    home().join("models")
}

pub fn server_bin() -> PathBuf {
    if let Some(b) = env::var_os("LLAMA_SERVER") {
        return PathBuf::from(b);
    }
    home().join("llama.cpp/build/bin/llama-server")
}
