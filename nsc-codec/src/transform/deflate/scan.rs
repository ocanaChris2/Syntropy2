//! Find recompressible deflate streams in a buffer: gzip members, zip entries (method 8), zlib
//! bodies of PDF streams, and the IDAT zlib stream of PNG images (whose data is split across
//! chunks). A stream is accepted only if its reconstruction record is small and the rebuild from
//! (decompressed data, record) reproduces the original bytes exactly, so recompression is safe by
//! construction.
use super::{front, inflate, side, writer};
use crate::crc::crc32;

/// A deflate stream that rebuilds exactly from (`inner`, `side`).
pub struct Deflate {
    pub start: usize,
    pub len: usize,
    pub inner: Vec<u8>,
    pub side: Vec<u8>,
}

/// The IHDR facts needed to undo row filters.
#[derive(Clone, Copy, Debug)]
pub struct PngInfo {
    pub width: u32,
    pub height: u32,
    pub depth: u8,
    pub color: u8,
    pub interlace: u8,
}

/// A PNG's run of IDAT chunks. The concatenated chunk data is a zlib stream: `zhead` (2 bytes),
/// the deflate `body` (offsets relative to the zlib stream), then `tail` (adler-32 and anything
/// after it).
pub struct Png {
    pub start: usize,
    pub len: usize,
    pub sizes: Vec<u32>,
    pub zhead: Vec<u8>,
    pub body: Deflate,
    pub tail: Vec<u8>,
    pub info: PngInfo,
}

/// A JPEG that re-encodes exactly from its coefficients.
pub struct Jpeg {
    pub start: usize,
    pub split: crate::transform::jpeg::Split,
}

pub enum Found {
    Deflate(Deflate),
    Png(Png),
    Jpeg(Jpeg),
    /// A stream taken apart by a format recoder (`crate::recode`).
    Recode(crate::recode::Recoded),
}

impl Found {
    pub fn start(&self) -> usize {
        match self { Found::Deflate(d) => d.start, Found::Png(p) => p.start, Found::Jpeg(j) => j.start, Found::Recode(r) => r.start }
    }
    pub fn len(&self) -> usize {
        match self { Found::Deflate(d) => d.len, Found::Png(p) => p.len, Found::Jpeg(j) => j.split.len, Found::Recode(r) => r.len }
    }
    pub fn inner_len(&self) -> usize {
        match self {
            Found::Deflate(d) => d.inner.len(),
            Found::Png(p) => p.body.inner.len(),
            Found::Jpeg(j) => j.split.coefs.len(),
            Found::Recode(r) => r.inner_len(),
        }
    }
}

/// Don't bother with streams that decompress to less than this.
const MIN_INNER: usize = 256;

fn gzip_body(b: &[u8], i: usize) -> Option<usize> {
    if b.len() < i + 18 || b[i] != 0x1f || b[i + 1] != 0x8b || b[i + 2] != 8 { return None; }
    let flg = b[i + 3];
    if flg & 0xe0 != 0 { return None; }
    let mut o = i + 10;
    if flg & 4 != 0 { o += 2 + u16::from_le_bytes([*b.get(o)?, *b.get(o + 1)?]) as usize; }
    if flg & 8 != 0 { while *b.get(o)? != 0 { o += 1; } o += 1; }
    if flg & 16 != 0 { while *b.get(o)? != 0 { o += 1; } o += 1; }
    if flg & 2 != 0 { o += 2; }
    (o < b.len()).then_some(o)
}

fn zip_body(b: &[u8], i: usize) -> Option<usize> {
    if b.len() < i + 30 || &b[i..i + 4] != b"PK\x03\x04" { return None; }
    let flags = u16::from_le_bytes([b[i + 6], b[i + 7]]);
    let method = u16::from_le_bytes([b[i + 8], b[i + 9]]);
    if method != 8 || flags & 1 != 0 { return None; }
    let o = i + 30 + u16::from_le_bytes([b[i + 26], b[i + 27]]) as usize + u16::from_le_bytes([b[i + 28], b[i + 29]]) as usize;
    (o < b.len()).then_some(o)
}

/// A valid zlib header (deflate, no preset dictionary) at `o`: its window bits.
pub fn zlib_header(b: &[u8], o: usize) -> Option<u8> {
    let (cmf, flg) = (*b.get(o)?, *b.get(o + 1)?);
    if cmf & 0x0f != 8 || cmf >> 4 > 7 || flg & 0x20 != 0 || ((cmf as u16) << 8 | flg as u16) % 31 != 0 { return None; }
    Some((cmf >> 4) + 8)
}

/// zlib header right after a PDF `stream` keyword: returns (deflate start, window bits).
fn pdf_zlib_body(b: &[u8], i: usize) -> Option<(usize, u8)> {
    if !b[i..].starts_with(b"stream") { return None; }
    let mut o = i + 6;
    if b.get(o) == Some(&b'\r') { o += 1; }
    if b.get(o) != Some(&b'\n') { return None; }
    o += 1;
    Some((o + 2, zlib_header(b, o)?))
}

/// Diagnostics sink for `nsc deflate-scan` (None in normal use).
type Log<'a> = Option<&'a mut Vec<String>>;

fn note(log: &mut Log, msg: impl FnOnce() -> String) {
    if let Some(l) = log.as_mut() { l.push(msg()); }
}

/// A stream's record is worth it up to this share of the stream: an eighth as zlib's parse
/// with its exceptions listed; a third, as it is estimated to code, noted for any encoder (a
/// stream that then comes to more than its bytes is left as them: `nsc3::block::losers`).
const ZLIB_SHARE: usize = 8;
const ANY_SHARE: usize = 2;
/// A record of zlib's parse above this share of the stream (one in so many) says the encoder was
/// another, or zlib not quite: the record for any encoder is tried as well.
const ZLIB_SURE: usize = 50;

/// The record to rebuild the deflate stream of `parse` from `inner`, its content: zlib's parse
/// with what differs from it, or — where that is too much, and `any` allows — the record that
/// assumes no encoder (`front`). Err: neither is small enough (the sizes they came to).
pub fn record(inner: &[u8], parse: &super::Parse, wbits: u8, any: bool) -> Result<Vec<u8>, (usize, Option<usize>)> {
    let rec = side::encode(inner, parse, wbits);
    let fits = rec.len() * ZLIB_SHARE <= parse.consumed;
    if !any || rec.len() * ZLIB_SURE <= parse.consumed { return if fits { Ok(rec) } else { Err((rec.len(), None)) }; }
    // the one that is estimated to code smaller
    match front::encode(inner, parse) {
        Some(r) if front::cost(&r) * ANY_SHARE <= parse.consumed && (!fits || front::cost(&r) < front::cost(&rec)) => Ok(r),
        _ if fits => Ok(rec),
        r => Err((rec.len(), r.map(|r| front::cost(&r)))),
    }
}

/// Try one candidate deflate body at `s`. With `verify`, the rebuild from the record is checked
/// here; otherwise the caller must check it (the block encoder's decode-back does). `any`:
/// streams of other encoders than zlib too (for decoders of level 2).
fn attempt(b: &[u8], s: usize, wbits: u8, budget: usize, verify: bool, any: bool, log: &mut Log) -> Option<Deflate> {
    let (inner, parse) = match inflate::inflate(&b[s..], budget) {
        Ok(r) => r,
        Err(e) => { note(log, || format!("@{s}: not inflatable ({e})")); return None; }
    };
    if inner.len() < MIN_INNER || parse.consumed < 16 {
        note(log, || format!("@{s}: too small ({} -> {} bytes)", parse.consumed, inner.len()));
        return None;
    }
    let rec = match record(&inner, &parse, wbits, any) {
        Ok(r) => r,
        Err((zlib, other)) => {
            note(log, || format!("@{s}: record too large ({zlib} bytes for a {}-byte stream, {} blocks{})", parse.consumed, parse.blocks.len(),
                other.map_or(String::new(), |c| format!("; noted for any encoder about {c} bytes{}", front::census(&inner, &parse)))));
            return None;
        }
    };
    if verify {
        let back = side::decode(&inner, &rec).ok()?;
        let bytes = writer::write(&back, &inner).ok()?;
        if bytes != b[s..s + parse.consumed] {
            note(log, || format!("@{s}: rebuild mismatch"));
            return None;
        }
    }
    note(log, || if rec[0] == front::VERSION {
        format!("@{s}: OK {} -> {} bytes, not zlib's parse: record {} bytes, about {} coded ({} blocks, habits {}){}", parse.consumed, inner.len(), rec.len(), front::cost(&rec), parse.blocks.len(), rec[1],
            if std::env::var_os("NSC_DP").is_some() { format!("\n      {}", front::dp_census(&inner, &parse)) } else { String::new() })
    } else {
        format!("@{s}: OK {} -> {} bytes, record {} bytes ({} blocks, params {:02x?})", parse.consumed, inner.len(), rec.len(), parse.blocks.len(), &rec[1..3])
    });
    Some(Deflate { start: s, len: parse.consumed, inner, side: rec })
}

fn be32(b: &[u8], o: usize) -> Option<u32> { Some(u32::from_be_bytes(b.get(o..o + 4)?.try_into().ok()?)) }

/// A PNG at `i`: its first run of IDAT chunks, if every chunk's CRC is right and the zlib stream
/// they carry recompresses.
fn png(b: &[u8], i: usize, budget: usize, verify: bool, any: bool, refused: &mut Option<usize>, log: &mut Log) -> Option<Png> {
    if !b[i..].starts_with(b"\x89PNG\r\n\x1a\n") { return None; }
    let mut o = i + 8;
    let mut info = None;
    let (mut start, mut sizes, mut z) = (0usize, Vec::new(), Vec::new());
    loop {
        let len = be32(b, o)? as usize;
        let ty = b.get(o + 4..o + 8)?;
        let end = o.checked_add(12 + len)?;
        if end > b.len() { return None; }
        if o == i + 8 {
            if ty != b"IHDR" || len != 13 { return None; }
            let d = &b[o + 8..o + 21];
            info = Some(PngInfo { width: be32(d, 0)?, height: be32(d, 4)?, depth: d[8], color: d[9], interlace: d[12] });
        }
        if ty == b"IDAT" {
            if sizes.is_empty() { start = o; }
            if crc32(&b[o + 4..o + 8 + len]) != be32(b, o + 8 + len)? { note(log, || format!("@{o}: png IDAT CRC mismatch")); return None; }
            sizes.push(len as u32);
            z.extend_from_slice(&b[o + 8..o + 8 + len]);
        } else if !sizes.is_empty() || ty == b"IEND" {
            break;
        }
        o = end;
    }
    if sizes.is_empty() { return None; }
    let wbits = zlib_header(&z, 0)?;
    note(log, || format!("@{i}: png {:?}, {} IDAT chunks", info, sizes.len()));
    let body = attempt(&z, 2, wbits, budget, verify, any, log);
    // (the same stream stands in the buffer as a zlib stream of its own: not to be tried twice)
    if body.is_none() && sizes.len() == 1 { *refused = Some(start + 8); }
    let body = body?;
    let len = sizes.iter().map(|&s| 12 + s as usize).sum();
    let tail = z[2 + body.len..].to_vec();
    Some(Png { start, len, sizes, zhead: z[..2].to_vec(), body, tail, info: info? })
}

/// All recompressible streams in `b`, in order and non-overlapping, decompressing to at most
/// `budget` bytes in total. `verify`: prove each rebuild here (see [`attempt`]).
/// `recode`: also try the format recoders (MP3, …).
pub fn find(b: &[u8], budget: usize, verify: bool, recode: bool) -> Vec<Found> { find_logged(b, budget, verify, recode, &[], None) }

/// [`find`] in a planned block, whose container index may point out sample runs (`hints`).
pub fn find_hinted(b: &[u8], budget: usize, verify: bool, recode: bool, hints: &[crate::nsc3::plan::Hint]) -> Vec<Found> {
    find_logged(b, budget, verify, recode, hints, None)
}

/// [`find`] with a line of diagnostics per candidate.
pub fn find_logged(b: &[u8], mut budget: usize, verify: bool, recode: bool, hints: &[crate::nsc3::plan::Hint], mut log: Log) -> Vec<Found> {
    let mut out: Vec<Found> = vec![];
    let mut i = 0usize;
    // the sample runs the container's index points out are taken apart together, where the
    // first of them starts
    let mut runs = if recode { crate::recode::from_hints(b, hints, budget, &mut log) } else { None };
    // a stream found by a signature that stands before it (a TIFF's strips, a PNG's image data):
    // what lies between the two is scanned like anything else, and the stream taken when the
    // scan gets to it
    let mut held: Option<Found> = None;
    let mut refused: Option<usize> = None;
    while i + 4 < b.len() && (budget >= MIN_INNER || held.is_some()) {
        if runs.as_ref().is_some_and(|r| r.start == i) {
            let r = runs.take().unwrap();
            budget = budget.saturating_sub(r.inner_len());
            i = r.start + r.len;
            out.push(Found::Recode(r));
            continue;
        }
        if held.as_ref().is_some_and(|f| f.start() <= i) {
            let f = held.take().unwrap();
            i = f.start() + f.len();
            out.push(f);
            continue;
        }
        if budget < MIN_INNER {
            // nothing more is taken apart: on to the stream that is held
            i = held.as_ref().map_or(b.len(), |f| f.start());
            continue;
        }
        // nothing found before them may run into them
        let b = match &runs { Some(r) => &b[..r.start], None => b };
        let b = match &held { Some(f) => &b[..f.start().min(b.len())], None => b };
        if i + 4 >= b.len() {
            i += 1;
            continue;
        }
        let found = match b[i] {
            0x1f => gzip_body(b, i).and_then(|s| attempt(b, s, 15, budget, verify, recode, &mut log)).map(Found::Deflate),
            // an archive as a whole (its headers' fields derived), or an entry by itself
            b'P' => match recode.then(|| crate::recode::zip_at(b, i, budget, &mut log)).flatten() {
                Some(r) => Some(Found::Recode(r)),
                None => zip_body(b, i).and_then(|s| attempt(b, s, 15, budget, verify, recode, &mut log)).map(Found::Deflate),
            },
            b's' => pdf_zlib_body(b, i).and_then(|(s, w)| attempt(b, s, w, budget, verify, recode, &mut log)).map(Found::Deflate),
            0x89 => png(b, i, budget, verify, recode, &mut refused, &mut log).map(Found::Png),
            0xff if b[i + 1] == 0xd8 && b[i + 2] == 0xff => match crate::transform::jpeg::analyze(&b[i..]) {
                Ok(split) if split.coefs.len() <= budget => {
                    note(&mut log, || format!("@{i}: jpeg OK {} bytes -> {} coefficient bytes", split.len, split.coefs.len()));
                    Some(Found::Jpeg(Jpeg { start: i, split }))
                }
                Ok(_) => None,
                // not a JPEG of coefficients: one of samples (a camera's raw data), perhaps
                Err(e) => match recode.then(|| crate::recode::ljpeg_at(b, i, budget, &mut log)).flatten() {
                    Some(r) => Some(Found::Recode(r)),
                    None => { note(&mut log, || format!("@{i}: {e}")); None }
                },
            },
            // a zlib stream on its own (in a database row, an object store, a game's archive)
            0x78 if recode && refused != Some(i) => zlib_header(b, i).and_then(|w| attempt(b, i + 2, w, budget, verify, recode, &mut log)).map(Found::Deflate),
            x if recode && crate::recode::may_start(x) => crate::recode::find_at(b, i, budget, &mut log).map(Found::Recode),
            _ => None,
        };
        match found {
            // (a writer for decoders of level 1 does as nsc 0.3 did: on from the stream's end)
            Some(f) if recode && f.start() > i && held.is_none() => {
                budget -= f.inner_len();
                held = Some(f);
                i += 1;
            }
            Some(f) => {
                budget -= f.inner_len();
                i = f.start() + f.len();
                out.push(f);
            }
            None => i += 1,
        }
    }
    out.extend(held);
    out
}

// ------------------------------------------------------------------ block planning support

/// The stream (if any) that crosses a planned block cut.
#[derive(Debug, PartialEq)]
pub enum Cross {
    None,
    /// Ends at `end` (offset in the surveyed buffer) and decompresses to about `inner` bytes.
    Ends { end: usize, inner: usize },
    /// Starts at candidate offset `start` and runs past the end of the surveyed buffer.
    Open { start: usize },
}

pub struct Survey {
    /// Decompressed bytes of the streams that end before the cut (an estimate for memory planning).
    pub inner: usize,
    pub cross: Cross,
}

/// The IDAT run of a PNG at `i` by chunk structure only: (run end, estimated decompressed size).
fn png_extent(b: &[u8], i: usize) -> Result<(usize, usize), inflate::ExtentError> {
    use inflate::ExtentError::*;
    if !b[i..].starts_with(b"\x89PNG\r\n\x1a\n") { return Err(Invalid); }
    let mut o = i + 8;
    let (mut est, mut run_end) = (0usize, None);
    loop {
        let (Some(len), Some(ty)) = (be32(b, o), b.get(o + 4..o + 8)) else { return Err(Truncated) };
        let end = o + 12 + len as usize;
        if o == i + 8 {
            if ty != b"IHDR" || len != 13 { return Err(Invalid); }
            let (w, h) = (be32(b, o + 8).ok_or(Truncated)? as usize, be32(b, o + 12).ok_or(Truncated)? as usize);
            let bits = *b.get(o + 16).ok_or(Truncated)? as usize * 4;
            est = h.saturating_mul(1 + (w.saturating_mul(bits)).div_ceil(8));
        }
        if ty == b"IDAT" {
            if end > b.len() { return Err(Truncated); }
            run_end = Some(end);
        } else if run_end.is_some() || ty == b"IEND" {
            return run_end.map(|e| (e, est)).ok_or(Invalid);
        }
        o = end;
        if o > b.len() { return Err(Truncated); }
    }
}

/// Extent of the candidate stream whose signature is at `i`: (end, decompressed size).
/// `recode`: format recoders' streams count too.
pub fn stream_extent(b: &[u8], i: usize, max_inner: usize, recode: bool) -> Result<(usize, usize), inflate::ExtentError> {
    use inflate::ExtentError::*;
    let body = match b[i] {
        0x1f => gzip_body(b, i),
        b'P' => zip_body(b, i),
        b's' => pdf_zlib_body(b, i).map(|(s, _)| s),
        0x89 => return png_extent(b, i),
        0xff if b.get(i + 1) == Some(&0xd8) && b.get(i + 2) == Some(&0xff) => {
            // coefficient bytes run about 2-4x the JPEG size
            return match crate::transform::jpeg::extent(&b[i..]) {
                Ok(n) => Ok((i + n, 4 * n)),
                Err(true) => Err(Truncated),
                Err(false) if recode => crate::recode::ljpeg::extent(b, i),
                Err(false) => Err(Invalid),
            };
        }
        0x78 if recode => zlib_header(b, i).map(|_| i + 2),
        x if recode && crate::recode::may_start(x) => return crate::recode::extent(b, i),
        _ => None,
    };
    let s = body.ok_or(Invalid)?;
    let (n, out) = inflate::extent(&b[s..], max_inner)?;
    if out < MIN_INNER { return Err(Invalid); }
    Ok((s + n, out))
}

/// Walk the stream candidates before `cut` the way [`find`] does, by structure only: what the
/// streams that end before the cut decompress to, and the stream that crosses the cut.
pub fn survey(b: &[u8], cut: usize, max_inner: usize, recode: bool) -> Survey {
    let mut i = 0usize;
    let mut inner = 0usize;
    while i < cut && i + 4 < b.len() {
        if matches!(b[i], 0x1f | b'P' | b's' | 0x89 | 0xff) || (recode && (b[i] == 0x78 || crate::recode::may_start(b[i]))) {
            match stream_extent(b, i, max_inner, recode) {
                Ok((end, out)) if end > cut => return Survey { inner, cross: Cross::Ends { end, inner: out } },
                Ok((end, out)) => { inner += out; i = end; continue; }
                Err(inflate::ExtentError::Truncated) => return Survey { inner, cross: Cross::Open { start: i } },
                Err(inflate::ExtentError::Invalid) => {}
            }
        }
        i += 1;
    }
    Survey { inner, cross: Cross::None }
}
