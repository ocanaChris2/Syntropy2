//! `.nsc` compressed-file container.
//!
//! v2 (`NSC2`) records which model version produced the file, so `decompress` can select the exact
//! model from the registry — a file stays decodable after the default model is upgraded, and a
//! mismatched/tampered registry is caught by the fingerprint. v1 (`NSCZ`) had no model binding and
//! is still readable (the caller supplies the model).
//!
//! v2 layout: magic "NSC2" | fmt u16 (=2) | id_len u16 | id bytes | fingerprint u64 |
//!            vocab u32 | seg u32 | n_tokens u32 | first_token u32 | n_bits u32 | payload
//! v1 layout: magic "NSCZ" | vocab u32 | seg u32 | n_tokens u32 | first_token u32 | n_bits u32 | payload
//!
//! Stored layout: magic "NSCR" | fmt u16 (=1) | orig_len u64 | raw bytes verbatim.
//! Used as a universal fallback: already-compressed / binary input (mp4, images, archives) that the
//! neural model would only expand — and might exhaust memory to tokenize — is stored as-is so
//! `compress`/`decompress` always round-trips byte-exact and never fails.
use crate::codec::{ru16, ru32, ru64};
use std::fs;
use std::io::{Read, Write};
use std::path::Path;

pub const FORMAT_VERSION: u16 = 2;
pub const STORED_MAGIC: &[u8; 4] = b"NSCR";
const STORED_HEADER_LEN: u64 = 4 + 2 + 8; // magic + fmt + orig_len

/// Which container a file uses (by magic).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Nsc2,
    Nscz,
    Stored,
    Nsc3,
}

pub struct Header {
    pub format: Format,
    pub model_id: Option<String>, // None for legacy NSCZ and stored files
    pub fingerprint: u64,         // model fingerprint; 0 for legacy/stored
    pub vocab: usize,
    pub seg: usize,
    pub n: usize,                 // neural: token count; stored: original byte length
    pub first: usize,
    pub n_bits: usize,
    pub stored: bool,             // true => NSCR (raw fallback), no model/tokenizer needed
}

pub struct Container {
    pub header: Header,
    pub payload: Vec<u8>,
}

/// Serialize a v2 container to `path`.
pub fn write(path: &str, model_id: &str, fingerprint: u64, vocab: usize, seg: usize,
             n: usize, first: usize, n_bits: usize, payload: &[u8]) -> Result<(), String> {
    let id = model_id.as_bytes();
    if id.len() > u16::MAX as usize { return Err("model id too long".into()); }
    let mut out = Vec::with_capacity(32 + id.len() + payload.len());
    out.extend_from_slice(b"NSC2");
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&(id.len() as u16).to_le_bytes());
    out.extend_from_slice(id);
    out.extend_from_slice(&fingerprint.to_le_bytes());
    for x in [vocab as u32, seg as u32, n as u32, first as u32, n_bits as u32] {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out.extend_from_slice(payload);
    fs::write(path, &out).map_err(|e| format!("write {path}: {e}"))
}

/// Parse a header from a byte prefix; returns (header, offset-of-payload). Cheap: reads no payload.
pub fn parse_header(b: &[u8]) -> Result<(Header, usize), String> {
    if b.len() < 4 { return Err("file too short".into()); }
    let mut o = 4usize;
    match &b[0..4] {
        b"NSC2" => {
            if b.len() < 8 { return Err("truncated NSC2 header".into()); }
            let _fmt = ru16(b, &mut o);
            let id_len = ru16(b, &mut o) as usize;
            if b.len() < o + id_len + 8 + 20 { return Err("truncated NSC2 header".into()); }
            let model_id = String::from_utf8(b[o..o + id_len].to_vec())
                .map_err(|_| "bad model id utf8".to_string())?;
            o += id_len;
            let fingerprint = ru64(b, &mut o);
            let vocab = ru32(b, &mut o) as usize;
            let seg = ru32(b, &mut o) as usize;
            let n = ru32(b, &mut o) as usize;
            let first = ru32(b, &mut o) as usize;
            let n_bits = ru32(b, &mut o) as usize;
            Ok((Header { format: Format::Nsc2, model_id: Some(model_id), fingerprint, vocab, seg, n, first, n_bits, stored: false }, o))
        }
        b"NSCR" => {
            if b.len() < STORED_HEADER_LEN as usize { return Err("truncated NSCR header".into()); }
            let _fmt = ru16(b, &mut o);
            let orig = ru64(b, &mut o) as usize;
            Ok((Header { format: Format::Stored, model_id: None, fingerprint: 0, vocab: 0, seg: 0, n: orig, first: 0, n_bits: 0, stored: true }, o))
        }
        b"NSCZ" => {
            if b.len() < 24 { return Err("truncated NSCZ header".into()); }
            let vocab = ru32(b, &mut o) as usize;
            let seg = ru32(b, &mut o) as usize;
            let n = ru32(b, &mut o) as usize;
            let first = ru32(b, &mut o) as usize;
            let n_bits = ru32(b, &mut o) as usize;
            Ok((Header { format: Format::Nscz, model_id: None, fingerprint: 0, vocab, seg, n, first, n_bits, stored: false }, o))
        }
        b"NSC3" => Ok((Header { format: Format::Nsc3, model_id: None, fingerprint: 0, vocab: 0, seg: 0, n: 0, first: 0, n_bits: 0, stored: false }, 4)),
        _ => Err("not an .nsc file (bad magic)".into()),
    }
}

/// Read a full container (header + payload) from `path`.
pub fn read(path: &str) -> Result<Container, String> {
    let b = fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
    let (header, off) = parse_header(&b)?;
    Ok(Container { header, payload: b[off..].to_vec() })
}

/// Read just the header from `path` (for `nsc info`, and to route decompress) without loading the
/// payload. Reads only a small prefix, so it stays cheap even on multi-hundred-MB stored files.
pub fn read_header(path: &str) -> Result<Header, String> {
    let mut f = fs::File::open(path).map_err(|e| format!("read {path}: {e}"))?;
    let mut buf = [0u8; 8192];
    let mut got = 0usize;
    loop {
        match f.read(&mut buf[got..]) {
            Ok(0) => break,
            Ok(k) => { got += k; if got == buf.len() { break; } }
            Err(e) => return Err(format!("read {path}: {e}")),
        }
    }
    Ok(parse_header(&buf[..got])?.0)
}

/// Write a stored (raw, no-neural) container: a small header followed by `src` copied verbatim.
/// Streams the copy, so arbitrarily large files never load fully into memory. Returns bytes written.
pub fn write_stored(dst: &str, src: &Path) -> Result<u64, String> {
    let orig_len = fs::metadata(src).map_err(|e| format!("stat {}: {e}", src.display()))?.len();
    let mut r = fs::File::open(src).map_err(|e| format!("open {}: {e}", src.display()))?;
    let mut w = std::io::BufWriter::new(fs::File::create(dst).map_err(|e| format!("create {dst}: {e}"))?);
    w.write_all(STORED_MAGIC)
        .and_then(|_| w.write_all(&1u16.to_le_bytes()))
        .and_then(|_| w.write_all(&orig_len.to_le_bytes()))
        .map_err(|e| format!("write {dst}: {e}"))?;
    let copied = std::io::copy(&mut r, &mut w).map_err(|e| format!("copy -> {dst}: {e}"))?;
    w.flush().map_err(|e| format!("flush {dst}: {e}"))?;
    if copied != orig_len { return Err(format!("stored copy short: {copied} != {orig_len} bytes")); }
    Ok(STORED_HEADER_LEN + copied)
}

/// Extract a stored (NSCR) container to `dst`, streaming and verifying the declared length. Returns
/// the number of payload bytes written.
pub fn extract_stored(src: &str, dst: &str) -> Result<u64, String> {
    let mut r = fs::File::open(src).map_err(|e| format!("open {src}: {e}"))?;
    let mut hdr = [0u8; STORED_HEADER_LEN as usize];
    r.read_exact(&mut hdr).map_err(|e| format!("read {src}: {e}"))?;
    if &hdr[0..4] != STORED_MAGIC { return Err("not a stored .nsc file".into()); }
    let orig_len = u64::from_le_bytes(hdr[6..14].try_into().unwrap());
    let partial = format!("{dst}.partial");
    let res = (|| {
        let mut w = std::io::BufWriter::new(fs::File::create(&partial).map_err(|e| format!("create {partial}: {e}"))?);
        let copied = std::io::copy(&mut r, &mut w).map_err(|e| format!("copy -> {partial}: {e}"))?;
        w.flush().map_err(|e| format!("flush {partial}: {e}"))?;
        if copied != orig_len { return Err(format!("stored container corrupt: {copied} bytes != declared {orig_len}")); }
        Ok(copied)
    })();
    match res {
        Ok(n) => { fs::rename(&partial, dst).map_err(|e| format!("rename {partial} -> {dst}: {e}"))?; Ok(n) }
        Err(e) => { let _ = fs::remove_file(&partial); Err(e) }
    }
}
