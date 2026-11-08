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

pub struct Meta {
    pub layers: u32,
    /// Dominant ggml tensor type, e.g. "IQ2_XXS", "Q4_K".
    pub quant: String,
    /// The file contains a built-in multi-token-prediction head
    /// (`blk.N.nextn.*`), so `--spec-type draft-mtp` needs no separate
    /// `--spec-draft-model` file.
    pub nextn: bool,
}

/// Parse layer count (from blk.N tensor names) and the dominant tensor
/// quantization type in one pass over the file header.
pub fn parse(path: &Path) -> Meta {
    let mut meta = Meta { layers: 0, quant: String::new(), nextn: false };
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
    for _ in 0..kv_count {
        if c.string().is_none() {
            return meta;
        }
        let vt = match c.u32() {
            Some(v) => v,
            None => return meta,
        };
        if skip_value(&mut c, vt).is_none() {
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
    meta
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
