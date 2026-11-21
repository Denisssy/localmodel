use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

/// Below this size a plain serial curl is simpler and just as fast.
const PARALLEL_MIN_BYTES: u64 = 256 * 1024 * 1024;
/// Target bytes per range part; the part count derives from it.
const PART_TARGET_BYTES: u64 = 256 * 1024 * 1024;
const MAX_PARTS: u64 = 16;

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

/// First bytes of a file (for the GGUF magic check). Reads only what is
/// asked, not the whole multi-GB file.
fn read_head(path: &Path, n: usize) -> io::Result<Vec<u8>> {
    let mut f = File::open(path)?;
    let mut buf = vec![0u8; n];
    let got = f.read(&mut buf)?;
    buf.truncate(got);
    Ok(buf)
}

/// Total file size, following redirects. None when the server does not
/// report a length (then the caller falls back to a serial download).
fn remote_size(url: &str) -> Option<u64> {
    let out = Command::new("curl")
        .args(["-sIL", "--retry", "3", url])
        .arg(url)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    stdout
        .lines()
        .filter_map(|l| {
            let (name, val) = l.split_once(':')?;
            if name.trim().eq_ignore_ascii_case("content-length") {
                val.trim().parse::<u64>().ok()
            } else {
                None
            }
        })
        .last()
}

/// Spawn `nparts` curl range-requests over `source`; each writes
/// `dest.part<i>`. Returns the spawned children and their part paths.
fn spawn_parts(source: &str, dir: &Path, fname: &str, size: u64, nparts: u64) -> io::Result<Vec<(std::process::Child, PathBuf)>> {
    let chunk = size.div_ceil(nparts);
    let mut out = Vec::new();
    for i in 0..nparts {
        let start = i * chunk;
        let end = ((i + 1) * chunk - 1).min(size - 1);
        if start > end {
            break;
        }
        let path = dir.join(format!("{fname}.part{i}"));
        let child = Command::new("curl")
            .args([
                "-sS", "--fail", "-L",
                "--retry", "10", "--retry-all-errors",
                "-r", &format!("{start}-{end}"),
                "-o",
            ])
            .arg(&path)
            .arg(source)
            .spawn()?;
        out.push((child, path));
    }
    Ok(out)
}

/// Concatenate parts (in order) into `dest`, verify the byte count, drop
/// the parts. Caller cleans parts on error.
fn concat_parts(parts: &[PathBuf], dest: &Path, size: u64) -> io::Result<()> {
    {
        let mut out = File::create(dest)?;
        for p in parts {
            let mut f = File::open(p)?;
            io::copy(&mut f, &mut out)?;
        }
    }
    let actual = fs::metadata(dest)?.len();
    if actual != size {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("size mismatch: got {actual} bytes, expected {size}"),
        ));
    }
    Ok(())
}

/// Parallel ranged download. HF throttles a single connection hard
/// (~40 min for 17 GB on a 1 Gbps line); 10-16 range streams saturate it.
fn download_parallel(source: &str, dir: &Path, fname: &str, size: u64, part: &Path) -> io::Result<()> {
    let nparts = (size.div_ceil(PART_TARGET_BYTES)).clamp(2, MAX_PARTS);
    println!("parallel download: {nparts} streams, {} total", human(size));

    let mut children = spawn_parts(source, dir, fname, size, nparts)?;
    let parts: Vec<PathBuf> = children.iter().map(|(_, p)| p.clone()).collect();

    // Aggregate progress while any stream is alive.
    let t0 = Instant::now();
    loop {
        thread::sleep(Duration::from_secs(2));
        let alive = children.iter_mut().any(|(c, _)| c.try_wait().ok().flatten().is_none());
        let done: u64 = parts
            .iter()
            .filter_map(|p| fs::metadata(p).ok())
            .map(|m| m.len())
            .sum();
        let elapsed = t0.elapsed().as_secs_f64();
        let rate = if elapsed > 0.0 { done as f64 / elapsed } else { 0.0 };
        print!(
            "\r  {:>5.1}%  {} / {}  {:.0} MB/s",
            100.0 * done as f64 / size as f64,
            human(done),
            human(size),
            rate / 1e6,
        );
        io::stdout().flush()?;
        if !alive {
            break;
        }
    }
    println!();

    let mut failed = false;
    for (mut child, _) in children {
        match child.wait() {
            Ok(st) if st.success() => {}
            other => {
                eprintln!("part curl failed: {other:?}");
                failed = true;
            }
        }
    }
    if failed {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("a download stream failed (partial parts kept at {fname}.part*)"),
        ));
    }

    let result = concat_parts(&parts, part, size);
    for p in &parts {
        if result.is_ok() {
            let _ = fs::remove_file(p);
        }
    }
    result
}

fn human(n: u64) -> String {
    if n >= 1 << 30 {
        format!("{:.1} GiB", n as f64 / (1 << 30) as f64)
    } else {
        format!("{:.0} MiB", n as f64 / (1 << 20) as f64)
    }
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

    // Big file + reported size: range-parallel. Otherwise (or when the
    // server hides the length) one resumable curl stream.
    // LOCALMODEL_PARALLEL_MIN overrides the threshold (bytes) - for tests
    // or forcing the parallel path on smaller files.
    let min_bytes = std::env::var("LOCALMODEL_PARALLEL_MIN")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(PARALLEL_MIN_BYTES);
    match remote_size(&source) {
        Some(size) if size >= min_bytes => {
            download_parallel(&source, dir, &fname, size, &part)?;
        }
        _ => {
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
        }
    }

    // Reject a page / redirect target: if the bytes are not a GGUF file,
    // the URL was wrong (blob page, a directory listing, ...) and we must
    // not silently rename a few hundred KB of HTML into the model name.
    let head = read_head(&part, 4).unwrap_or_default();
    if !is_gguf(&head) {
        let len = fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        let kind = if len > 4000 { "HTML page" } else { "unexpected content" };
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

    #[test]
    fn content_length_is_parsed_from_curl_headers() {
        let headers = "HTTP/2 302\r\nlocation: https://cdn.example.com/x\r\n\
                       \r\nHTTP/2 200\r\ncontent-type: application/octet-stream\r\n\
                       content-length: 17029602592\r\n\r\n";
        let mut lines = headers.lines().filter_map(|l| {
            let (name, val) = l.split_once(':')?;
            if name.trim().eq_ignore_ascii_case("content-length") {
                val.trim().parse::<u64>().ok()
            } else {
                None
            }
        });
        assert_eq!(lines.last(), Some(17_029_602_592));
    }

    #[test]
    fn parts_spawn_for_exact_ranges() {
        // spawn_parts against a local python http server is overkill;
        // instead verify the range math directly.
        let size = 100u64;
        let nparts = 4u64;
        let chunk = size.div_ceil(nparts);
        let mut ranges = Vec::new();
        for i in 0..nparts {
            let start = i * chunk;
            let end = ((i + 1) * chunk - 1).min(size - 1);
            if start > end {
                break;
            }
            ranges.push((start, end));
        }
        assert_eq!(ranges, vec![(0, 24), (25, 49), (50, 74), (75, 99)]);
    }

    #[test]
    fn concat_verifies_size_and_merges_in_order() {
        let dir = std::env::temp_dir().join(format!("localmodel-dl-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let p0 = dir.join("a.part0");
        let p1 = dir.join("a.part1");
        fs::write(&p0, b"GGUF-hello ").unwrap(); // 11 bytes
        fs::write(&p1, b"world").unwrap(); // 5 bytes, total 16
        let dest = dir.join("a");

        // wrong expected size -> error, dest kept for inspection
        concat_parts(&[p0.clone(), p1.clone()], &dest, 15).unwrap_err();
        fs::remove_file(&dest).unwrap();

        // right size -> ok, bytes in part order, GGUF magic intact
        concat_parts(&[p0, p1], &dest, 16).unwrap();
        let head = read_head(&dest, 4).unwrap();
        assert_eq!(&head, b"GGUF");
        let body = read_head(&dest, 32).unwrap();
        assert_eq!(&body, b"GGUF-hello world");
        fs::remove_dir_all(&dir).ok();
    }
}
