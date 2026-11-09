use std::fs;
use std::io::Read;
use std::path::Path;

/// Read only a bounded prefix of the file: the GGUF header, KV metadata and
/// tensor table are all stored before the actual tensor data, so a few hundred
/// megabytes are enough to count layers without loading a 20GB model.
const MAX_HEADER: u64 = 256 * 1024 * 1024;

struct Cursor<'a> {
    data: &'a [u8],
    off: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, off: 0 }
    }

    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        if self.off + n > self.data.len() {
            return None;
        }
        let s = &self.data[self.off..self.off + n];
        self.off += n;
        Some(s)
    }

    fn u32(&mut self) -> Option<u32> {
        let b = self.take(4)?;
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Option<u64> {
        let b = self.take(8)?;
        Some(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }

    fn skip(&mut self, n: usize) -> Option<()> {
        self.take(n).map(|_| ())
    }

    /// gguf_str: uint64 length followed by raw bytes.
    fn string(&mut self) -> Option<&'a [u8]> {
        let n = self.u64()? as usize;
        self.take(n)
    }
}

/// One transformer block that owns a KV cache.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KvLayer {
    /// KV heads in this block. Zero means the block has no KV cache at all
    /// (Mamba/SSM blocks in hybrid models).
    pub heads: u32,
    /// Elements per head in K.
    pub k: u32,
    /// Elements per head in V.
    pub v: u32,
    /// Sliding-window block: it caches at most `KvGeometry::swa_window` tokens.
    pub swa: bool,
}

/// Which blocks allocate a KV cache, and how wide each one is.
///
/// Hybrid models (qwen35, nemotron_h_moe) only keep KV for their full
/// attention blocks; the SSM blocks carry a fixed recurrent state instead.
/// Gemma mixes sliding-window and full-attention blocks with different head
/// dimensions, both of which change the cache size by an order of magnitude.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KvGeometry {
    /// Full-context attention blocks of the main model.
    pub main: Vec<KvLayer>,
    /// Built-in MTP/nextn blocks (the draft context when `--spec-type draft-mtp`).
    pub draft: Vec<KvLayer>,
    /// Sliding-window length for blocks with `swa = true`; 0 = no window.
    pub swa_window: u32,
}

impl KvGeometry {
    /// True when at least one layer has enough metadata to size the cache.
    pub fn has_kv(&self) -> bool {
        self.main.iter().chain(self.draft.iter()).any(|l| l.heads > 0 && l.k > 0 && l.v > 0)
    }
}

#[derive(Clone, Default)]
pub struct Meta {
    pub layers: u32,
    /// Dominant ggml tensor type, e.g. "IQ2_XXS", "Q4_K".
    pub quant: String,
    /// The file contains a built-in multi-token-prediction head
    /// (`blk.N.nextn.*`), so `--spec-type draft-mtp` needs no separate
    /// `--spec-draft-model` file.
    pub nextn: bool,
    /// `general.architecture`, e.g. "qwen35", "gemma4", "nemotron_h_moe".
    pub arch: String,
    /// Transformer blocks declared by the file (includes the MTP block).
    pub block_count: u32,
    /// Training context length (`*.context_length`).
    pub ctx_train: u32,
    /// Number of built-in MTP/nextn blocks.
    pub nextn_layers: u32,
    /// Per-block KV shape, used by `fit::for_gguf`.
    pub kv: KvGeometry,
}

#[derive(Default)]
struct RawMeta {
    arch: String,
    block_count: Option<u32>,
    ctx_train: Option<u32>,
    n_embd: Option<u32>,
    head_count: Option<u32>,
    kv_heads: Option<u32>,
    kv_heads_per_layer: Option<Vec<u32>>,
    key_len: Option<u32>,
    value_len: Option<u32>,
    key_len_swa: Option<u32>,
    value_len_swa: Option<u32>,
    full_attention_interval: Option<u32>,
    nextn_layers: Option<u32>,
    swa_window: Option<u32>,
    swa_pattern: Option<Vec<bool>>,
}

/// Parse layer count (from blk.N tensor names) and the dominant tensor
/// quantization type in one pass over the file header.
pub fn parse(path: &Path) -> Meta {
    let mut meta = Meta {
        layers: 0,
        quant: String::new(),
        nextn: false,
        arch: String::new(),
        block_count: 0,
        ctx_train: 0,
        nextn_layers: 0,
        kv: KvGeometry::default(),
    };
    let data = match read_prefix(path) {
        Some(d) => d,
        None => return meta,
    };
    let mut c = Cursor::new(&data);
    if c.take(4) != Some(b"GGUF") {
        return meta;
    }
    if c.u32().is_none() {
        return meta;
    }
    let tensor_count = match c.u64() {
        Some(v) => v,
        None => return meta,
    };
    let kv_count = match c.u64() {
        Some(v) => v,
        None => return meta,
    };
    let mut raw = RawMeta::default();
    for _ in 0..kv_count {
        let key = match c.string() {
            Some(k) => k,
            None => return meta,
        };
        let vt = match c.u32() {
            Some(v) => v,
            None => return meta,
        };
        if !capture_kv(&mut c, key, vt, &mut raw) && skip_value(&mut c, vt).is_none() {
            return meta;
        }
    }
    let mut max_layer = 0u32;
    // Dominant tensor type weighted by element count (the bulk of the weights).
    let mut counts: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
    for _ in 0..tensor_count {
        let name = match c.string() {
            Some(n) => n,
            None => break,
        };
        let ndim = match c.u32() {
            Some(v) => v,
            None => break,
        };
        let mut numel: u64 = 1;
        let mut ok = true;
        for _ in 0..ndim {
            match c.u64() {
                Some(d) => numel = numel.saturating_mul(d),
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            break;
        }
        let t = match c.u32() {
            Some(v) => v,
            None => break,
        };
        if c.u64().is_none() {
            break; // tensor data offset
        }
        let e = counts.entry(t).or_insert(0);
        *e = e.saturating_add(numel);
        // Unsloth UD quants ship the MTP head inside the main file.
        if name.windows(7).any(|w| w == b".nextn.") {
            meta.nextn = true;
        }
        if let Some(l) = parse_blk_index(name) {
            if l > max_layer {
                max_layer = l;
            }
        }
    }
    meta.layers = max_layer.saturating_add(1);
    if let Some((&t, _)) = counts.iter().max_by_key(|(_, &v)| v) {
        meta.quant = ggml_type_name(t).to_string();
    }
    meta.arch = raw.arch.clone();
    meta.block_count = raw.block_count.filter(|&v| v > 0).unwrap_or(meta.layers);
    meta.ctx_train = raw.ctx_train.unwrap_or(0);
    meta.nextn_layers = raw.nextn_layers.unwrap_or(if meta.nextn { 1 } else { 0 });
    meta.kv = build_kv(&raw, meta.block_count, meta.nextn_layers);
    meta
}

/// Consume one metadata value when it is interesting for the KV calculator.
/// Returns false (without touching the cursor) when the caller should skip it.
fn capture_kv(c: &mut Cursor, key: &[u8], vt: u32, raw: &mut RawMeta) -> bool {
    let Ok(name) = std::str::from_utf8(key) else {
        return false;
    };
    if name == "general.architecture" {
        if vt != 8 {
            return false;
        }
        raw.arch = c
            .string()
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .unwrap_or_default();
        return true;
    }
    // Metadata keys are "<arch>.<field>"; match the field so the parser does
    // not need a table of every architecture name.
    let field = match name.split_once('.') {
        Some((_, rest)) => rest,
        None => return false,
    };
    let is_u32 = matches!(vt, 4 | 5);
    match field {
        "block_count" if is_u32 => raw.block_count = read_u32(c, vt),
        "context_length" if is_u32 => raw.ctx_train = read_u32(c, vt),
        "embedding_length" if is_u32 => raw.n_embd = read_u32(c, vt),
        "attention.head_count" if is_u32 => raw.head_count = read_u32(c, vt),
        "attention.head_count_kv" if is_u32 => raw.kv_heads = read_u32(c, vt),
        "attention.head_count_kv" if vt == 9 => raw.kv_heads_per_layer = read_u32_array(c),
        "attention.key_length" if is_u32 => raw.key_len = read_u32(c, vt),
        "attention.value_length" if is_u32 => raw.value_len = read_u32(c, vt),
        "attention.key_length_swa" if is_u32 => raw.key_len_swa = read_u32(c, vt),
        "attention.value_length_swa" if is_u32 => raw.value_len_swa = read_u32(c, vt),
        "full_attention_interval" if is_u32 => raw.full_attention_interval = read_u32(c, vt),
        "nextn_predict_layers" if is_u32 => raw.nextn_layers = read_u32(c, vt),
        "attention.sliding_window" if is_u32 => raw.swa_window = read_u32(c, vt),
        "attention.sliding_window_pattern" if vt == 9 => raw.swa_pattern = read_bool_array(c),
        _ => return false,
    }
    true
}

/// uint32/int32 scalar. The caller checks the GGUF type first.
fn read_u32(c: &mut Cursor, vt: u32) -> Option<u32> {
    match vt {
        4 | 5 => c.u32(),
        _ => None,
    }
}

fn read_u32_array(c: &mut Cursor) -> Option<Vec<u32>> {
    let elem = c.u32()?;
    let n = c.u64()?;
    // Every element must be consumed even if the type is unexpected, otherwise
    // the cursor would desynchronise and the tensor table would be garbage.
    let mut out = Vec::with_capacity((n.min(4096)) as usize);
    if !matches!(elem, 4 | 5) {
        for _ in 0..n {
            skip_value(c, elem)?;
        }
        return None;
    }
    for _ in 0..n {
        out.push(c.u32()?);
    }
    Some(out)
}

fn read_bool_array(c: &mut Cursor) -> Option<Vec<bool>> {
    let elem = c.u32()?;
    let n = c.u64()?;
    let mut out = Vec::with_capacity((n.min(4096)) as usize);
    if !matches!(elem, 0 | 7) {
        for _ in 0..n {
            skip_value(c, elem)?;
        }
        return None;
    }
    for _ in 0..n {
        out.push(c.take(1)?[0] != 0);
    }
    Some(out)
}

/// Turn raw metadata into per-block KV shapes.
///
/// Three layouts exist in the wild:
///   * scalar `head_count_kv` + `full_attention_interval` (qwen35: every 4th
///     block is attention, the rest are SSM);
///   * per-block `head_count_kv` where 0 marks an SSM block (nemotron_h_moe);
///   * per-block `head_count_kv` plus a sliding-window pattern (gemma4).
fn build_kv(raw: &RawMeta, block_count: u32, nextn_layers: u32) -> KvGeometry {
    let nextn_layers = nextn_layers.min(block_count);
    let base_blocks = block_count.saturating_sub(nextn_layers);
    let interval = raw.full_attention_interval.unwrap_or(1).max(1);
    let head_count = raw.head_count.unwrap_or(0);
    let default_dim = (head_count > 0)
        .then(|| raw.n_embd.unwrap_or(0).checked_div(head_count).unwrap_or(0))
        .unwrap_or(0);

    let heads_at = |i: u32| -> u32 {
        if let Some(arr) = &raw.kv_heads_per_layer {
            arr.get(i as usize).copied().unwrap_or(0)
        } else if let Some(h) = raw.kv_heads {
            if interval > 1 && (i + 1) % interval != 0 {
                0
            } else {
                h
            }
        } else if head_count > 0 {
            if interval > 1 && (i + 1) % interval != 0 {
                0
            } else {
                head_count
            }
        } else {
            0
        }
    };
    let swa_at = |i: u32| -> bool {
        if let Some(pattern) = &raw.swa_pattern {
            pattern.get(i as usize).copied().unwrap_or(false)
        } else {
            raw.swa_window.unwrap_or(0) > 0
        }
    };
    let layer_at = |i: u32| -> Option<KvLayer> {
        let is_draft = i >= base_blocks;
        // MTP blocks are always full attention; the interval only describes
        // the main model's SSM/attention alternation.
        let heads = if is_draft && raw.kv_heads_per_layer.is_none() {
            raw.kv_heads
                .or((head_count > 0).then_some(head_count))
                .unwrap_or(0)
        } else {
            heads_at(i)
        };
        if heads == 0 {
            return None;
        }
        let swa = !is_draft && swa_at(i);
        let (k, v) = if swa {
            (
                raw.key_len_swa.or(raw.key_len).unwrap_or(default_dim),
                raw.value_len_swa.or(raw.value_len).unwrap_or(default_dim),
            )
        } else {
            (
                raw.key_len.unwrap_or(default_dim),
                raw.value_len.unwrap_or(default_dim),
            )
        };
        Some(KvLayer { heads, k, v, swa })
    };

    let mut main = Vec::new();
    for i in 0..base_blocks {
        if let Some(l) = layer_at(i) {
            main.push(l);
        }
    }
    let mut draft = Vec::new();
    for j in 0..nextn_layers {
        if let Some(l) = layer_at(base_blocks + j) {
            draft.push(l);
        }
    }
    KvGeometry { main, draft, swa_window: raw.swa_window.unwrap_or(0) }
}

/// The model's embedded `tokenizer.chat_template` Jinja source, if present.
pub fn chat_template(path: &Path) -> Option<String> {
    let data = read_prefix(path)?;
    let mut c = Cursor::new(&data);
    if c.take(4) != Some(b"GGUF") || c.u32().is_none() {
        return None;
    }
    let tensor_count = c.u64()?;
    let kv_count = c.u64()?;
    let mut found = None;
    for _ in 0..kv_count {
        let key = c.string()?;
        let vt = c.u32()?;
        if vt == 8 {
            let v = c.string()?;
            if key == b"tokenizer.chat_template" {
                found = String::from_utf8(v.to_vec()).ok();
            }
        } else if skip_value(&mut c, vt).is_none() {
            return None;
        }
    }
    let _ = tensor_count;
    found
}

fn ggml_type_name(t: u32) -> &'static str {
    match t {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        _ => "unknown",
    }
}

fn skip_value(c: &mut Cursor, vt: u32) -> Option<()> {
    match vt {
        0 | 1 | 7 => c.skip(1),            // uint8, int8, bool
        2 | 3 => c.skip(2),                // uint16, int16
        4 | 5 | 6 => c.skip(4),            // uint32, int32, float32
        8 => c.string().map(|_| ()),       // string
        9 => {
            // array: element type + count + elements
            let elem = c.u32()?;
            let n = c.u64()?;
            for _ in 0..n {
                if elem == 8 {
                    c.string()?;
                } else {
                    c.skip(scalar_size(elem)?)?;
                }
            }
            Some(())
        }
        10 | 11 | 12 => c.skip(8),         // uint64, int64, float64
        _ => None,
    }
}

fn scalar_size(vt: u32) -> Option<usize> {
    match vt {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4 | 5 | 6 => Some(4),
        10 | 11 | 12 => Some(8),
        _ => None,
    }
}

fn parse_blk_index(name: &[u8]) -> Option<u32> {
    let s = String::from_utf8_lossy(name);
    let rest = s.strip_prefix("blk.")?;
    rest.split('.').next()?.parse().ok()
}

fn read_prefix(path: &Path) -> Option<Vec<u8>> {
    let mut f = fs::File::open(path).ok()?;
    let mut data = Vec::with_capacity(MAX_HEADER as usize);
    let mut buf = [0u8; 65536];
    let mut total = 0u64;
    while total < MAX_HEADER {
        let n = f.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        data.extend_from_slice(&buf[..n]);
        total += n as u64;
    }
    Some(data)
}

/// Guess quantization type from the model file name (e.g. Q4_K_M, Q8_0, IQ2_XXS).
pub fn quant_from_name(name: &str) -> String {
    let lower = name.to_lowercase();
    // Prefer more specific/longer tokens first.
    let tokens = [
        "iq2_sxxs", "iq2_xxs", "iq3_sxxs", "iq3_xxs", "iq4_sxxs", "iq4_xxs",
        "iq5_sxxs", "iq5_xxs",
        "q8_0_xs", "q6_k_m_xs", "q5_k_m_xs", "q4_k_m_xs",
        "q6_k_xl", "q5_k_xl", "q4_k_xl", "q3_k_xl", "q2_k_xl",
        "iq3_s", "iq2_s", "iq4_s", "iq5_s",
        "q8_0", "q8_k_m",
        "q6_k_m", "q6_0",
        "q5_k_m", "q5_1", "q5_0",
        "q4_k_m", "q4_k_s", "q4_1", "q4_0",
        "q3_k_m", "q3_k_s",
        "q2_k_m", "q2_k_s",
        "q2_xxs", "q3_xxs", "q4_xxs", "q5_xxs", "q6_xxs",
        "f16", "bf16", "f32",
    ];
    for t in tokens {
        if lower.contains(t) {
            return t.to_uppercase();
        }
    }
    "unknown".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_str(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    fn put_u32(out: &mut Vec<u8>, key: &str, v: u32) {
        put_str(out, key);
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&v.to_le_bytes());
    }

    fn put_str_val(out: &mut Vec<u8>, key: &str, v: &str) {
        put_str(out, key);
        out.extend_from_slice(&8u32.to_le_bytes());
        put_str(out, v);
    }

    fn put_u32_array(out: &mut Vec<u8>, key: &str, vals: &[u32]) {
        put_str(out, key);
        out.extend_from_slice(&9u32.to_le_bytes());
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for v in vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }

    fn put_bool_array(out: &mut Vec<u8>, key: &str, vals: &[bool]) {
        put_str(out, key);
        out.extend_from_slice(&9u32.to_le_bytes());
        out.extend_from_slice(&7u32.to_le_bytes());
        out.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for v in vals {
            out.push(u8::from(*v));
        }
    }

    fn write_gguf(name: &str, body: impl FnOnce(&mut Vec<u8>), count: usize) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "localmodel-{name}-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes()); // tensors
        out.extend_from_slice(&(count as u64).to_le_bytes());
        body(&mut out);
        std::fs::write(&path, out).unwrap();
        path
    }

    #[test]
    fn parses_qwen35_attention_interval_and_nextn() {
        let path = write_gguf(
            "qwen35",
            |b| {
                put_str_val(b, "general.architecture", "qwen35");
                put_u32(b, "qwen35.block_count", 33);
                put_u32(b, "qwen35.context_length", 262144);
                put_u32(b, "qwen35.attention.head_count", 16);
                put_u32(b, "qwen35.attention.head_count_kv", 4);
                put_u32(b, "qwen35.attention.key_length", 256);
                put_u32(b, "qwen35.attention.value_length", 256);
                put_u32(b, "qwen35.full_attention_interval", 4);
                put_u32(b, "qwen35.nextn_predict_layers", 1);
            },
            9,
        );
        let m = parse(&path);
        std::fs::remove_file(&path).ok();
        assert_eq!(m.arch, "qwen35");
        assert_eq!(m.block_count, 33);
        assert_eq!(m.ctx_train, 262144);
        assert_eq!(m.nextn_layers, 1);
        // Every 4th of the 32 non-MTP blocks, plus the MTP block.
        assert_eq!(m.kv.main.len(), 8);
        assert_eq!(m.kv.draft.len(), 1);
        assert_eq!(m.kv.main[0], KvLayer { heads: 4, k: 256, v: 256, swa: false });
    }

    #[test]
    fn parses_per_layer_heads_and_sliding_window() {
        let path = write_gguf(
            "gemma4",
            |b| {
                put_str_val(b, "general.architecture", "gemma4");
                put_u32(b, "gemma4.block_count", 6);
                put_u32(b, "gemma4.context_length", 262144);
                put_u32(b, "gemma4.attention.head_count", 16);
                put_u32_array(b, "gemma4.attention.head_count_kv", &[8, 8, 8, 8, 8, 1]);
                put_u32(b, "gemma4.attention.key_length", 512);
                put_u32(b, "gemma4.attention.value_length", 512);
                put_u32(b, "gemma4.attention.key_length_swa", 256);
                put_u32(b, "gemma4.attention.value_length_swa", 256);
                put_u32(b, "gemma4.attention.sliding_window", 1024);
                put_bool_array(
                    b,
                    "gemma4.attention.sliding_window_pattern",
                    &[true, true, true, true, true, false],
                );
            },
            11,
        );
        let m = parse(&path);
        std::fs::remove_file(&path).ok();
        assert_eq!(m.kv.main.len(), 6);
        assert_eq!(m.kv.swa_window, 1024);
        for l in &m.kv.main[..5] {
            assert_eq!(*l, KvLayer { heads: 8, k: 256, v: 256, swa: true });
        }
        assert_eq!(m.kv.main[5], KvLayer { heads: 1, k: 512, v: 512, swa: false });
    }

    #[test]
    fn zero_heads_mark_ssm_blocks() {
        let path = write_gguf(
            "nemotron",
            |b| {
                put_str_val(b, "general.architecture", "nemotron_h_moe");
                put_u32(b, "nemotron_h_moe.block_count", 8);
                put_u32(b, "nemotron_h_moe.attention.head_count", 32);
                put_u32_array(
                    b,
                    "nemotron_h_moe.attention.head_count_kv",
                    &[0, 0, 0, 0, 0, 0, 0, 2],
                );
                put_u32(b, "nemotron_h_moe.attention.key_length", 128);
                put_u32(b, "nemotron_h_moe.attention.value_length", 128);
            },
            6,
        );
        let m = parse(&path);
        std::fs::remove_file(&path).ok();
        // Only the last block owns a KV cache.
        assert_eq!(m.kv.main.len(), 1);
        assert_eq!(m.kv.main[0], KvLayer { heads: 2, k: 128, v: 128, swa: false });
    }
}
