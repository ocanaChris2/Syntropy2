//! Native byte-level BPE tokenizer for the neural model's `tokenizer.json` (no Python).
//!
//! `tok.py` feeds the tokenizer `bytes.decode("latin-1")`: every byte becomes one char
//! U+0000..=U+00FF. The ByteLevel pre-tokenizer then UTF-8-encodes the text and spells each UTF-8
//! byte with GPT-2's printable byte alphabet. Decoding inverts both steps, so it is exact by
//! construction: token strings → UTF-8 bytes → chars → original bytes.
use crate::json;

/// GPT-2 `bytes_to_unicode`: printable bytes map to themselves; the remaining 68 bytes map, in
/// increasing order, to U+0100 upwards.
pub fn byte_to_char() -> [char; 256] {
    let mut m = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        m[b as usize] = if printable { char::from_u32(b).unwrap() } else { let c = char::from_u32(256 + n).unwrap(); n += 1; c };
    }
    m
}

pub struct Tokenizer {
    /// id → the token's UTF-8 bytes (before the latin-1 step)
    id_bytes: Vec<Vec<u8>>,
    /// id of each single UTF-8 byte (the byte-level base alphabet)
    byte_id: [u32; 256],
    /// (left id, right id) → (merge rank, merged id)
    merges: std::collections::HashMap<(u32, u32), (u32, u32)>,
}

/// Pre-tokenizer character classes over the latin-1 domain, as the reference tokenizer's regex
/// (Unicode \p{L}, \p{N}, \s) sees them. Measured against the reference, not guessed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Cl { L, N, S, O }

const fn build_classes() -> [Cl; 256] {
    let mut t = [Cl::O; 256];
    let mut b = 0;
    while b < 256 {
        let c = b as u8;
        t[b] = if c.is_ascii_alphabetic() || c == 0xAA || c == 0xB5 || c == 0xBA
            || (c >= 0xC0 && c <= 0xD6) || (c >= 0xD8 && c <= 0xF6) || c >= 0xF8 { Cl::L }
        else if c.is_ascii_digit() || c == 0xB2 || c == 0xB3 || c == 0xB9 || (c >= 0xBC && c <= 0xBE) { Cl::N }
        else if (c >= 0x09 && c <= 0x0D) || c == 0x20 || c == 0x85 || c == 0xA0 { Cl::S }
        else { Cl::O };
        b += 1;
    }
    t
}
static CLASS: [Cl; 256] = build_classes();

/// GPT-2 pre-tokenization (`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`)
/// over bytes viewed as latin-1 chars. Calls `f` with each pre-token's byte range.
pub fn pretokenize(t: &[u8], mut f: impl FnMut(usize, usize)) {
    let n = t.len();
    let mut i = 0;
    while i < n {
        let c = t[i];
        if c == b'\'' && i + 1 < n {
            let r = &t[i + 1..];
            let l = if r.starts_with(b"re") || r.starts_with(b"ve") || r.starts_with(b"ll") { 3 }
                    else if matches!(r[0], b's' | b't' | b'm' | b'd') { 2 } else { 0 };
            if l > 0 { f(i, i + l); i += l; continue; }
        }
        let j = if c == b' ' && i + 1 < n && CLASS[t[i + 1] as usize] != Cl::S { i + 1 } else { i };
        let k = CLASS[t[j] as usize];
        if k != Cl::S {
            let mut e = j + 1;
            while e < n && CLASS[t[e] as usize] == k { e += 1; }
            f(i, e);
            i = e;
            continue;
        }
        let mut e = i + 1;
        while e < n && CLASS[t[e] as usize] == Cl::S { e += 1; }
        // \s+(?!\S): a whitespace run before a non-space keeps its last char for the next token
        let end = if e < n && e - i >= 2 { e - 1 } else { e };
        f(i, end);
        i = end;
    }
}

impl Tokenizer {
    pub fn load(path: &std::path::Path) -> Result<Tokenizer, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        Self::from_json(&text)
    }

    pub fn from_json(text: &str) -> Result<Tokenizer, String> {
        let v = json::parse(text)?;
        let model = v.get("model").ok_or("tokenizer.json: no model")?;
        if model.get("type").and_then(|t| t.as_str()) != Some("BPE") { return Err("tokenizer.json: only BPE models are supported".into()); }
        let vocab = model.get("vocab").and_then(|x| x.as_obj()).ok_or("tokenizer.json: no vocab")?;
        let b2c = byte_to_char();
        let mut c2b = std::collections::HashMap::new();
        for (b, &c) in b2c.iter().enumerate() { c2b.insert(c, b as u8); }
        let mut id_bytes: Vec<Vec<u8>> = vec![vec![]; vocab.len()];
        for (tok, idv) in vocab {
            let id = idv.as_u64().ok_or("tokenizer.json: bad id")? as usize;
            if id >= id_bytes.len() { id_bytes.resize(id + 1, vec![]); }
            let mut bytes = Vec::with_capacity(tok.len());
            for ch in tok.chars() {
                bytes.push(*c2b.get(&ch).ok_or_else(|| format!("tokenizer.json: token {tok:?} is not byte-level"))?);
            }
            id_bytes[id] = bytes;
        }
        if v.get("added_tokens").and_then(|a| a.as_arr()).is_some_and(|a| !a.is_empty()) {
            return Err("tokenizer.json: added tokens are not supported".into());
        }
        let mut by_bytes: std::collections::HashMap<&[u8], u32> = std::collections::HashMap::new();
        for (id, b) in id_bytes.iter().enumerate() { by_bytes.insert(b.as_slice(), id as u32); }
        let mut byte_id = [u32::MAX; 256];
        for b in 0..256usize {
            byte_id[b] = *by_bytes.get(&[b as u8][..]).ok_or("tokenizer.json: byte-level base alphabet incomplete")?;
        }
        let mut merges = std::collections::HashMap::new();
        if let Some(ms) = model.get("merges").and_then(|m| m.as_arr()) {
            for (rank, m) in ms.iter().enumerate() {
                let (a, b) = match m {
                    json::Value::Str(s) => { let mut it = s.splitn(2, ' '); (it.next().unwrap_or("").to_string(), it.next().unwrap_or("").to_string()) }
                    json::Value::Arr(p) if p.len() == 2 => (p[0].as_str().unwrap_or("").to_string(), p[1].as_str().unwrap_or("").to_string()),
                    _ => return Err("tokenizer.json: bad merge entry".into()),
                };
                let enc = |s: &str| -> Option<Vec<u8>> { s.chars().map(|ch| c2b.get(&ch).copied()).collect() };
                let (Some(ab), Some(bb)) = (enc(&a), enc(&b)) else { return Err("tokenizer.json: merge not byte-level".into()) };
                let (Some(&ia), Some(&ib)) = (by_bytes.get(ab.as_slice()), by_bytes.get(bb.as_slice())) else { continue };
                let mut cat = ab.clone();
                cat.extend_from_slice(&bb);
                if let Some(&im) = by_bytes.get(cat.as_slice()) { merges.entry((ia, ib)).or_insert((rank as u32, im)); }
            }
        }
        Ok(Tokenizer { id_bytes, byte_id, merges })
    }

    /// BPE-encode one pre-token given as UTF-8 bytes (heap-ordered merges: lowest rank first,
    /// leftmost on ties — the reference tokenizer's order).
    fn bpe(&self, word: &[u8], out: &mut Vec<u32>) {
        let n = word.len();
        if n == 1 { out.push(self.byte_id[word[0] as usize]); return; }
        let mut sym: Vec<u32> = word.iter().map(|&b| self.byte_id[b as usize]).collect();
        let mut next: Vec<isize> = (1..=n as isize).collect();
        next[n - 1] = -1;
        let mut prev: Vec<isize> = (-1..n as isize - 1).collect();
        let mut alive = vec![true; n];
        let mut heap = std::collections::BinaryHeap::new();
        for i in 0..n - 1 {
            if let Some(&(r, m)) = self.merges.get(&(sym[i], sym[i + 1])) { heap.push(std::cmp::Reverse((r, i, m))); }
        }
        while let Some(std::cmp::Reverse((r, i, m))) = heap.pop() {
            if !alive[i] || next[i] < 0 { continue; }
            let j = next[i] as usize;
            if self.merges.get(&(sym[i], sym[j])) != Some(&(r, m)) { continue; }
            sym[i] = m;
            alive[j] = false;
            next[i] = next[j];
            if next[j] >= 0 { prev[next[j] as usize] = i as isize; }
            if prev[i] >= 0 {
                let p = prev[i] as usize;
                if let Some(&(r2, m2)) = self.merges.get(&(sym[p], sym[i])) { heap.push(std::cmp::Reverse((r2, p, m2))); }
            }
            if next[i] >= 0 {
                let q = next[i] as usize;
                if let Some(&(r2, m2)) = self.merges.get(&(sym[i], sym[q])) { heap.push(std::cmp::Reverse((r2, i, m2))); }
            }
        }
        let mut k = 0isize;
        while k >= 0 {
            out.push(sym[k as usize]);
            k = next[k as usize];
        }
    }

    /// Original file bytes → token ids (the same ids the reference tokenizer produces).
    pub fn encode(&self, data: &[u8]) -> Vec<usize> {
        let mut out: Vec<u32> = Vec::with_capacity(data.len() / 3 + 16);
        let mut cache: std::collections::HashMap<Vec<u8>, Vec<u32>> = std::collections::HashMap::new();
        let mut utf8 = Vec::with_capacity(64);
        pretokenize(data, |a, b| {
            utf8.clear();
            for &c in &data[a..b] {
                if c < 0x80 { utf8.push(c); } else { utf8.push(0xC0 | (c >> 6)); utf8.push(0x80 | (c & 0x3F)); }
            }
            if let Some(ids) = cache.get(&utf8) { out.extend_from_slice(ids); return; }
            let start = out.len();
            self.bpe(&utf8, &mut out);
            if utf8.len() <= 64 {
                if cache.len() >= 1 << 16 { cache.clear(); }
                cache.insert(utf8.clone(), out[start..].to_vec());
            }
        });
        out.into_iter().map(|x| x as usize).collect()
    }

    /// Token ids → original file bytes.
    pub fn decode(&self, ids: &[usize]) -> Result<Vec<u8>, String> {
        let mut utf8 = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            utf8.extend_from_slice(self.id_bytes.get(id).ok_or_else(|| format!("token id {id} outside the vocabulary"))?);
        }
        let s = std::str::from_utf8(&utf8).map_err(|_| "decoded tokens are not valid UTF-8 (wrong tokenizer or corrupt file)".to_string())?;
        let mut out = Vec::with_capacity(s.len());
        for ch in s.chars() {
            let c = ch as u32;
            if c > 0xff { return Err("decoded text leaves the latin-1 byte range (wrong tokenizer?)".into()); }
            out.push(c as u8);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_alphabet_is_a_bijection() {
        let m = byte_to_char();
        let set: std::collections::BTreeSet<char> = m.iter().copied().collect();
        assert_eq!(set.len(), 256);
        assert_eq!(m[b'A' as usize], 'A');
        assert_eq!(m[b' ' as usize], '\u{120}'); // Ġ
        assert_eq!(m[b'\n' as usize], '\u{10a}'); // Ċ
    }

    fn tiny() -> Tokenizer {
        // full byte-level base alphabet (byte b -> id b) plus merges "ab" -> 256, "Ġa" -> 257
        let m = byte_to_char();
        let mut vocab: Vec<String> = (0..256).map(|b| format!("\"{}\":{}", json_escape(m[b]), b)).collect();
        vocab.push(format!("\"ab\":256"));
        vocab.push(format!("\"{}a\":257", json_escape(m[b' ' as usize])));
        let j = format!(r#"{{"model":{{"type":"BPE","vocab":{{{}}},"merges":[["a","b"],["{}","a"]]}},"added_tokens":[]}}"#,
            vocab.join(","), json_escape(m[b' ' as usize]));
        Tokenizer::from_json(&j).unwrap()
    }

    fn json_escape(c: char) -> String {
        match c { '"' => "\\\"".into(), '\\' => "\\\\".into(), c => c.to_string() }
    }

    #[test]
    fn encode_decode_roundtrip() {
        let t = tiny();
        assert_eq!(t.encode(b"aab"), vec![b'a' as usize, 256]);
        assert_eq!(t.encode(b"x ab"), vec![b'x' as usize, b' ' as usize, 256]); // rank 0 (a+b) beats rank 1 (Ġ+a)
        assert_eq!(t.encode(b"x ac"), vec![b'x' as usize, 257, b'c' as usize]);
        let data = vec![b'a', 0xE9, b' ', b' ', b'a', b'b', b'\n', 0x00, 0xFF];
        assert_eq!(t.decode(&t.encode(&data)).unwrap(), data);
        assert!(t.decode(&[0xC3]).is_err()); // dangling UTF-8 lead byte
    }

    #[test]
    fn pretokenize_matches_reference_rules() {
        let split = |s: &[u8]| { let mut v = vec![]; pretokenize(s, |a, b| v.push(s[a..b].to_vec())); v };
        let as_str = |v: Vec<Vec<u8>>| v.into_iter().map(|x| String::from_utf8_lossy(&x).into_owned()).collect::<Vec<_>>();
        assert_eq!(as_str(split(b"x  y")), ["x", " ", " y"]);
        assert_eq!(as_str(split(b"x\t\ty")), ["x", "\t", "\t", "y"]);
        assert_eq!(as_str(split(b"it's 42!? ok  ")), ["it", "'s", " 42", "!?", " ok", "  "]);
    }
}
