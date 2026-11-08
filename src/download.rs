use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn filename_from_url(url: &str) -> String {
    let no_q = url.split(['?', '#']).next().unwrap_or(url);
    let trimmed = no_q.trim_end_matches('/');
    let seg = trimmed.rsplit('/').next().unwrap_or("");
    if seg.to_lowercase().ends_with(".gguf") {
        return seg.to_string();
    }
    if let Some(p) = no_q.split('/').find(|s| s.to_lowercase().ends_with(".gguf")) {
        return p.rsplit('/').next().unwrap_or(p).to_string();
    }
    if !seg.is_empty() {
        return format!("{seg}.gguf");
    }
    "model.gguf".to_string()
}

pub fn add(url: &str, dir: &Path, name: Option<&str>) -> io::Result<PathBuf> {
    fs::create_dir_all(dir)?;
    let fname = name.map(|s| s.to_string()).unwrap_or_else(|| filename_from_url(url));
    let dest = dir.join(&fname);
    let part = dir.join(format!("{fname}.part"));

    if dest.exists() {
        println!("already present: {}", dest.display());
        return Ok(dest);
    }

    println!("downloading  {url}");
    println!("          -> {}", dest.display());
    let status = Command::new("curl")
        .args(["-L", "--fail", "--retry", "3", "--retry-delay", "2", "-C", "-", "--progress-bar"])
        .arg("-o")
        .arg(&part)
        .arg(url)
        .status()?;
    if !status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("curl exited with {status} (partial kept at {})", part.display()),
        ));
    }
    fs::rename(&part, &dest)?;
    println!("done: {fname}");
    Ok(dest)
}
