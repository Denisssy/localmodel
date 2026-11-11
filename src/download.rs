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

/// Rewrite a HuggingFace `/blob/<rev>/` URL to `/resolve/<rev>/`. The blob
/// page is HTML (a few hundred KB, downloads in under a second); `resolve`
/// returns the raw file. Without this `add` saves the page instead of the
/// model, and the magic check below would catch it too late.
pub fn hf_blob_to_resolve(url: &str) -> String {
    let blob = "/blob/";
    let no_q = url.split(['?', '#']).next().unwrap_or(url);
    if let Some(pos) = no_q.find(blob) {
        let mut out = String::with_capacity(no_q.len() + 2);
        out.push_str(&no_q[..pos]);
        out.push_str("/resolve/");
        out.push_str(&no_q[pos + blob.len()..]);
        return out;
    }
    url.to_string()
}

/// True when the file starts with the GGUF magic bytes (`GGUF`).
fn is_gguf(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && &bytes[0..4] == b"GGUF"
}

pub fn add(url: &str, dir: &Path, name: Option<&str>) -> io::Result<PathBuf> {
    fs::create_dir_all(dir)?;
    let fname = name.map(|s| s.to_string()).unwrap_or_else(|| filename_from_url(&hf_blob_to_resolve(url)));
    let dest = dir.join(&fname);
    let part = dir.join(format!("{fname}.part"));

    if dest.exists() {
        println!("already present: {}", dest.display());
        return Ok(dest);
    }

    let source = hf_blob_to_resolve(url);

    println!("downloading  {source}");
    println!("          -> {}", dest.display());
    let status = Command::new("curl")
        .args(["-L", "--fail", "--retry", "3", "--retry-delay", "2", "-C", "-", "--progress-bar"])
        .arg("-o")
        .arg(&part)
        .arg(&source)
        .status()?;
    if !status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("curl exited with {status} (partial kept at {})", part.display()),
        ));
    }

    // Reject a page / redirect target: if the bytes are not a GGUF file,
    // the URL was wrong (blob page, a directory listing, ...) and we must
    // not silently rename a few hundred KB of HTML into the model name.
    let head = fs::read(&part).unwrap_or_default();
    if !is_gguf(&head) {
        let kind = if head.len() > 4000 { "HTML page" } else { "unexpected content" };
        let _ = fs::remove_file(&part);
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("not a GGUF file ({kind}) for {fname}; the URL did not point to the model"),
        ));
    }

    fs::rename(&part, &dest)?;
    println!("done: {fname}");
    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_blob_url_is_rewritten_to_resolve() {
        let in_url = "https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF/blob/main/Ornith-1.5-9B-Q6_K.gguf";
        let want = "https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF/resolve/main/Ornith-1.5-9B-Q6_K.gguf";
        assert_eq!(hf_blob_to_resolve(in_url), want);
    }

    #[test]
    fn non_hf_urls_are_left_untouched() {
        let url = "https://example.com/model.gguf";
        assert_eq!(hf_blob_to_resolve(url), url);
    }

    #[test]
    fn gguf_magic_is_detected() {
        let gguf = [0x47, 0x47, 0x55, 0x46, 0, 1, 2, 3];
        let not_gguf = [0x54, 0x52, 0x55, 0x45, 0, 1, 2, 3];
        assert!(is_gguf(&gguf));
        assert!(!is_gguf(&not_gguf));
    }
}
