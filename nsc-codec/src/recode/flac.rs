//! FLAC: frames back to samples.
//!
//! A FLAC frame says, for each channel of a block of samples, how to predict them (a fixed
//! difference or a linear filter with its coefficients) and then holds what the prediction left,
//! in Rice codes whose parameter is chosen per partition of the block. The samples can be coded
//! shorter than that by a model that adapts to them sample by sample (`cm::audio`), but the file
//! must come back as it was: so [`split`] decodes the frames to samples and keeps, of each
//! frame, everything that is not sample values — its header, each channel's predictor and Rice
//! parameters. From the samples and that record, [`rebuild`] computes the residuals as the
//! encoder did and writes the frame again, bit for bit, with its two checksums.
//!
//! A frame that does not decode, or does not come back exactly (padding bits that are not zero,
//! a checksum that was wrong), ends the run of frames taken apart; what follows is bytes.
use crate::bits::{BitReader, BitWriter};
use crate::transform::deflate::inflate::ExtentError;

/// The longest run of zero bits a Rice code may start with (a file is not written with more).
const MAX_UNARY: u32 = 1 << 16;
/// Fewest frames worth a container.
const MIN_FRAMES: usize = 2;
/// Bytes a frame is searched for its end in (64 KB of eight channels of 24 bits and more).
const MAX_FRAME: usize = 4 << 20;

const fn crc8_table() -> [u8; 256] {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u8;
        let mut k = 0;
        while k < 8 {
            c = if c & 0x80 != 0 { (c << 1) ^ 0x07 } else { c << 1 };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
}

const fn crc16_table() -> [u16; 256] {
    let mut t = [0u16; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = (i as u16) << 8;
        let mut k = 0;
        while k < 8 {
            c = if c & 0x8000 != 0 { (c << 1) ^ 0x8005 } else { c << 1 };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
}

static CRC8: [u8; 256] = crc8_table();
static CRC16: [u16; 256] = crc16_table();

fn crc8(b: &[u8]) -> u8 { b.iter().fold(0, |c, &x| CRC8[(c ^ x) as usize]) }
fn crc16(b: &[u8]) -> u16 { b.iter().fold(0, |c, &x| (c << 8) ^ CRC16[((c >> 8) as u8 ^ x) as usize]) }

/// What a frame header says of its frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Header {
    /// Bytes of the header, its checksum included.
    pub len: usize,
    pub block: usize,
    pub channels: usize,
    /// How a stereo pair is coded: 0 as it is, 8 left and side, 9 side and right, 10 mid and side.
    pub pair: u8,
    pub bits: u32,
}

/// The frame header at `b[0]`, its checksum right (None also for the sample sizes a header only
/// refers to, and for 32-bit samples).
pub fn header(b: &[u8]) -> Option<Header> {
    if b.len() < 6 || b[0] != 0xFF || b[1] & 0xFE != 0xF8 || b[3] & 1 != 0 { return None; }
    let (bs, sr) = (b[2] >> 4, b[2] & 15);
    let (ch, ss) = (b[3] >> 4, b[3] >> 1 & 7);
    let (channels, pair) = match ch { 0..=7 => (ch as usize + 1, 0), 8..=10 => (2, ch), _ => return None };
    let bits = match ss { 1 => 8, 2 => 12, 4 => 16, 5 => 20, 6 => 24, _ => return None };
    if bs == 0 || sr == 15 { return None; }
    // the frame's or first sample's number, coded as UTF-8 codes its characters
    let mut o = 4 + match b[4] { 0..=0x7F => 1, 0xC0..=0xDF => 2, 0xE0..=0xEF => 3, 0xF0..=0xF7 => 4, 0xF8..=0xFB => 5, 0xFC..=0xFD => 6, 0xFE => 7, _ => return None };
    if b.len() < o || b[5..o].iter().any(|&x| x & 0xC0 != 0x80) { return None; }
    let block = match bs {
        1 => 192,
        2..=5 => 576 << (bs - 2),
        6 => { o += 1; *b.get(o - 1)? as usize + 1 }
        7 => { o += 2; ((*b.get(o - 2)? as usize) << 8 | *b.get(o - 1)? as usize) + 1 }
        _ => 256 << (bs - 8),
    };
    o += match sr { 12 => 1, 13 | 14 => 2, _ => 0 };
    (b.len() > o && crc8(&b[..o]) == b[o]).then_some(Header { len: o + 1, block, channels, pair, bits })
}

#[inline]
fn signed(v: u32, n: u32) -> i32 { if n == 0 { 0 } else { ((v << (32 - n)) as i32) >> (32 - n) } }

/// A count of zero bits ended by a one.
#[inline]
fn unary(r: &mut BitReader) -> Option<u32> {
    let mut q = 0u32;
    loop {
        let v = r.peek(32);
        if v != 0 {
            let z = v.leading_zeros();
            r.skip(z + 1);
            return Some(q + z);
        }
        r.skip(32);
        q += 32;
        if q > MAX_UNARY { return None; }
    }
}

/// The prediction of sample `i` of `s` by a fixed predictor of `order`, or by `coefs` and `shift`.
#[inline]
fn fixed(s: &[i32], i: usize, order: usize) -> i64 {
    let at = |k: usize| s[i - k] as i64;
    match order {
        0 => 0,
        1 => at(1),
        2 => 2 * at(1) - at(2),
        3 => 3 * at(1) - 3 * at(2) + at(3),
        _ => 4 * at(1) - 6 * at(2) + 4 * at(3) - at(4),
    }
}

#[inline]
fn linear(s: &[i32], i: usize, coefs: &[i32], shift: u32) -> i64 {
    let mut p = 0i64;
    for (k, &c) in coefs.iter().enumerate() { p += c as i64 * s[i - 1 - k] as i64; }
    p >> shift
}

/// Decode the subframe at `r` (`n` samples of `bits` bits) into `s`, appending what [`subframe`]
/// needs beside the samples to write it again to `rec`.
fn read_subframe(r: &mut BitReader, n: usize, bits: u32, s: &mut Vec<i32>, rec: &mut Vec<u8>) -> Option<()> {
    let head = r.get(8);
    if head & 0x80 != 0 { return None; }
    rec.push(head as u8);
    let wasted = if head & 1 != 0 { unary(r)? + 1 } else { 0 };
    if wasted >= bits || bits > 32 { return None; }
    if wasted > 0 { rec.push(wasted as u8); }
    let bits = bits - wasted;
    let wide = |r: &mut BitReader| -> i32 { signed(r.get(bits), bits) };
    s.clear();
    let ty = head >> 1;
    let order = match ty {
        0 => {
            s.resize(n, wide(r));
            0
        }
        1 => {
            for _ in 0..n { s.push(wide(r)); }
            0
        }
        8..=12 => (ty - 8) as usize,
        32..=63 => (ty - 31) as usize,
        _ => return None,
    };
    if ty >= 8 {
        if order > n { return None; }
        for _ in 0..order { s.push(wide(r)); }
        let mut coefs = [0i32; 32];
        let mut shift = 0;
        if ty >= 32 {
            let precision = r.get(4) + 1;
            let sh = signed(r.get(5), 5);
            if precision == 16 || sh < 0 { return None; }
            rec.extend_from_slice(&[precision as u8, sh as u8]);
            for c in coefs.iter_mut().take(order) {
                *c = signed(r.get(precision), precision);
                rec.extend_from_slice(&(*c as i16).to_be_bytes());
            }
            shift = sh as u32;
        }
        // what the prediction left: Rice codes, a parameter to a partition
        let (method, porder) = (r.get(2), r.get(4));
        if method > 1 { return None; }
        rec.push((method << 4 | porder) as u8);
        let (pbits, escape) = if method == 0 { (4, 15) } else { (5, 31) };
        let per = n >> porder;
        if per << porder != n || per < order.max(1) && porder > 0 { return None; }
        for p in 0..1usize << porder {
            let count = if p == 0 { per.checked_sub(order)? } else { per };
            let k = r.get(pbits);
            let raw = if k == escape { Some(r.get(5)) } else { None };
            rec.push(match raw { Some(w) => 0x80 | w as u8, None => k as u8 });
            for _ in 0..count {
                let res = match raw {
                    Some(w) => signed(r.get(w), w),
                    None => {
                        let u = (unary(r)? as u64) << k | r.get(k) as u64;
                        if u > u32::MAX as u64 { return None; }
                        ((u >> 1) as i32) ^ -((u & 1) as i32)
                    }
                };
                let i = s.len();
                let p = if ty >= 32 { linear(s, i, &coefs[..order], shift) } else { fixed(s, i, order) };
                let v = p + res as i64;
                if v != v as i32 as i64 { return None; }
                s.push(v as i32);
            }
        }
    }
    if wasted > 0 {
        for v in s.iter_mut() {
            let w = (*v as i64) << wasted;
            if w != w as i32 as i64 { return None; }
            *v = w as i32;
        }
    }
    Some(())
}

/// Write a subframe: the samples `s` (of `bits` bits) as the record at `rec[*p]` says. The frame
/// may not grow past `limit` bits: no encoder writes a frame much longer than its samples are
/// (and Rice codes with a wrong parameter would be thousands of times that).
fn subframe(w: &mut BitWriter, s: &[i32], bits: u32, rec: &[u8], p: &mut usize, res: &mut Vec<i32>, limit: usize) -> Result<(), String> {
    let mut byte = || -> Result<u8, String> {
        let x = *rec.get(*p).ok_or_else(bad)?;
        *p += 1;
        Ok(x)
    };
    let head = byte()?;
    if head & 0x80 != 0 { return Err(bad()); }
    let wasted = if head & 1 != 0 { byte()? as u32 } else { 0 };
    if head & 1 != 0 && wasted == 0 || wasted >= bits || bits > 32 { return Err(bad()); }
    w.put(head as u32, 8);
    if wasted > 0 {
        // the zero bits and the one that ends them
        for _ in 0..(wasted - 1) / 32 { w.put(0, 32); }
        w.put(1, (wasted - 1) % 32 + 1);
    }
    let bits = bits - wasted;
    let n = s.len();
    let at = |i: usize| -> Result<i32, String> { if s[i] & ((1i64 << wasted) - 1) as i32 != 0 { Err(bad()) } else { Ok(s[i] >> wasted) } };
    let wide = |w: &mut BitWriter, v: i32| -> Result<(), String> {
        // the value must be one of `bits` bits
        if bits < 32 && (v >= 1 << (bits - 1) || v < -(1 << (bits - 1))) { return Err(bad()); }
        w.put(v as u32 & (u32::MAX >> (32 - bits)), bits);
        Ok(())
    };
    let ty = (head >> 1) as usize;
    let order = match ty {
        0 => {
            let v = at(0)?;
            for i in 1..n { if at(i)? != v { return Err(bad()); } }
            return wide(w, v);
        }
        1 => {
            for i in 0..n { wide(w, at(i)?)?; }
            return Ok(());
        }
        8..=12 => ty - 8,
        32..=63 => ty - 31,
        _ => return Err(bad()),
    };
    if order > n { return Err(bad()); }
    // the samples as the predictor saw them
    res.clear();
    for i in 0..n { res.push(at(i)?); }
    for &v in &res[..order] { wide(w, v)?; }
    let mut coefs = [0i32; 32];
    let mut shift = 0;
    if ty >= 32 {
        let (precision, sh) = (byte()? as u32, byte()? as u32);
        if !(1..=15).contains(&precision) || sh > 15 { return Err(bad()); }
        w.put(precision - 1, 4);
        w.put(sh, 5);
        for c in coefs.iter_mut().take(order) {
            *c = i16::from_be_bytes([byte()?, byte()?]) as i32;
            if precision < 16 && (*c >= 1 << (precision - 1) || *c < -(1 << (precision - 1))) { return Err(bad()); }
            w.put(*c as u32 & ((1 << precision) - 1), precision);
        }
        shift = sh;
    }
    let mp = byte()?;
    let (method, porder) = ((mp >> 4) as u32, (mp & 15) as u32);
    if method > 1 { return Err(bad()); }
    w.put(method, 2);
    w.put(porder, 4);
    let (pbits, escape) = if method == 0 { (4, 15) } else { (5, 31) };
    let per = n >> porder;
    if per << porder != n { return Err(bad()); }
    let mut i = order;
    for part in 0..1usize << porder {
        let count = if part == 0 { per.checked_sub(order).ok_or_else(bad)? } else { per };
        let k = byte()?;
        if k & 0x80 != 0 {
            let width = (k & 0x7F) as u32;
            if width > 31 { return Err(bad()); }
            w.put(escape, pbits);
            w.put(width, 5);
            for _ in 0..count {
                let r = res[i] as i64 - if ty >= 32 { linear(res, i, &coefs[..order], shift) } else { fixed(res, i, order) };
                if width == 0 && r != 0 || width > 0 && (r >= 1 << (width - 1) || r < -(1 << (width - 1))) { return Err(bad()); }
                w.put(r as u32 & (u32::MAX >> (32 - width.max(1))), width);
                i += 1;
            }
            continue;
        }
        let k = k as u32;
        if k >= escape { return Err(bad()); }
        w.put(k, pbits);
        for _ in 0..count {
            let r = res[i] as i64 - if ty >= 32 { linear(res, i, &coefs[..order], shift) } else { fixed(res, i, order) };
            if r != r as i32 as i64 { return Err(bad()); }
            let u = ((r as i32) << 1 ^ (r as i32) >> 31) as u32;
            let q = u >> k;
            if q > MAX_UNARY { return Err(bad()); }
            if w.len() + q as usize > limit { return Err(bad()); }
            for _ in 0..q / 32 { w.put(0, 32); }
            w.put(1, q % 32 + 1);
            w.put(u & ((1u32 << k) - 1), k);
            i += 1;
        }
    }
    Ok(())
}

fn bad() -> String { "flac: bad frame record (corrupt file)".to_string() }

/// Decode the frame at `b[0]` (header `h`): its channels' samples into `chan` as they are heard
/// (left, right, …), its record appended to `rec`. Returns the frame's length.
fn read_frame(b: &[u8], h: &Header, chan: &mut [Vec<i32>], rec: &mut Vec<u8>) -> Option<usize> {
    rec.push((h.len - 1) as u8);
    rec.extend_from_slice(&b[..h.len - 1]);
    let mut r = BitReader::new(b, h.len * 8);
    for (c, s) in chan.iter_mut().enumerate().take(h.channels) {
        // the difference of a pair has one bit more
        let side = matches!((h.pair, c), (8, 1) | (9, 0) | (10, 1));
        read_subframe(&mut r, h.block, h.bits + side as u32, s, rec)?;
        if r.pos() > b.len() * 8 { return None; }
    }
    // zero bits to the end of the byte, then the checksum of all before
    let pad = (8 - r.pos() % 8) % 8;
    if r.get(pad as u32) != 0 { return None; }
    let end = r.pos() / 8;
    if end + 2 > b.len() || crc16(&b[..end]).to_be_bytes() != b[end..end + 2] { return None; }
    if h.pair != 0 {
        let (a, rest) = chan.split_at_mut(1);
        for (x, y) in a[0].iter_mut().zip(rest[0].iter_mut()) {
            let (l, r) = match h.pair {
                8 => (*x as i64, *x as i64 - *y as i64),
                9 => (*x as i64 + *y as i64, *y as i64),
                _ => {
                    let mid = (*x as i64) << 1 | (*y as i64 & 1);
                    ((mid + *y as i64) >> 1, (mid - *y as i64) >> 1)
                }
            };
            if l != l as i32 as i64 || r != r as i32 as i64 { return None; }
            (*x, *y) = (l as i32, r as i32);
        }
    }
    Some(end + 2)
}

/// Write the frame whose record is at `rec[*p]` from its channels' samples (as they are heard;
/// each of the header's block length).
fn write_frame(rec: &[u8], p: &mut usize, chan: &mut [Vec<i32>], scratch: &mut Vec<i32>, out: &mut Vec<u8>) -> Result<(), String> {
    let hl = *rec.get(*p).ok_or_else(bad)? as usize;
    let mut head = rec.get(*p + 1..*p + 1 + hl).ok_or_else(bad)?.to_vec();
    *p += 1 + hl;
    head.push(crc8(&head));
    let h = header(&head).filter(|h| h.len == head.len()).ok_or_else(bad)?;
    if chan.len() < h.channels || chan[..h.channels].iter().any(|c| c.len() != h.block) { return Err(bad()); }
    if h.pair != 0 {
        let (a, rest) = chan.split_at_mut(1);
        for (x, y) in a[0].iter_mut().zip(rest[0].iter_mut()) {
            let (l, r) = (*x as i64, *y as i64);
            let (u, v) = match h.pair { 8 => (l, l - r), 9 => (l - r, r), _ => ((l + r) >> 1, l - r) };
            if u != u as i32 as i64 || v != v as i32 as i64 { return Err(bad()); }
            (*x, *y) = (u as i32, v as i32);
        }
    }
    let mut w = BitWriter::new();
    w.extend(&head);
    let limit = 8 * (32 + h.channels * (5 * h.block + 64));
    for (c, s) in chan.iter().enumerate().take(h.channels) {
        let side = matches!((h.pair, c), (8, 1) | (9, 0) | (10, 1));
        subframe(&mut w, s, h.bits + side as u32, rec, p, scratch, limit)?;
    }
    w.align();
    let mut frame = w.into_bytes();
    frame.extend_from_slice(&crc16(&frame).to_be_bytes());
    out.extend_from_slice(&frame);
    Ok(())
}

/// A run of frames taken apart: its length, the frames' records, the samples (interleaved,
/// little-endian, of `width` bytes each), and how many frames.
pub struct Split {
    pub len: usize,
    pub records: Vec<u8>,
    pub pcm: Vec<u8>,
    pub channels: u8,
    pub width: u8,
    pub frames: usize,
}

/// Take apart the run of frames at the start of `b` (frames of the same channels and sample
/// size, each of which comes back exactly); None unless there are a few.
pub fn split(b: &[u8]) -> Option<Split> {
    let first = header(b)?;
    let width = first.bits.div_ceil(8) as usize;
    let mut sp = Split { len: 0, records: vec![], pcm: vec![], channels: first.channels as u8, width: width as u8, frames: 0 };
    let mut chan: Vec<Vec<i32>> = vec![vec![]; first.channels];
    let (mut back, mut scratch) = (vec![], vec![]);
    while let Some(h) = header(&b[sp.len..]).filter(|h| (h.channels, h.bits) == (first.channels, first.bits)) {
        let frame = &b[sp.len..];
        let mark = sp.records.len();
        let Some(len) = read_frame(frame, &h, &mut chan, &mut sp.records) else {
            sp.records.truncate(mark);
            break;
        };
        // the samples must fit their bytes, and the frame must come back from them
        let lim = 1i64 << (8 * width - 1);
        let at = sp.pcm.len();
        for i in 0..h.block {
            for c in &chan { sp.pcm.extend_from_slice(&c[i].to_le_bytes()[..width]); }
        }
        back.clear();
        let mut p = mark;
        let fits = chan.iter().all(|c| c.iter().all(|&v| (v as i64) < lim && (v as i64) >= -lim));
        if !fits || write_frame(&sp.records, &mut p, &mut chan, &mut scratch, &mut back).is_err() || back != frame[..len] || p != sp.records.len() {
            sp.records.truncate(mark);
            sp.pcm.truncate(at);
            break;
        }
        sp.len += len;
        sp.frames += 1;
    }
    (sp.frames >= MIN_FRAMES).then_some(sp)
}

/// Append the frames [`split`] took apart, from their records and samples.
pub fn rebuild(records: &[u8], pcm: &[u8], channels: usize, width: usize, out: &mut Vec<u8>) -> Result<(), String> {
    if !(1..=8).contains(&channels) || !(1..=3).contains(&width) { return Err(bad()); }
    let mut chan: Vec<Vec<i32>> = vec![vec![]; channels];
    let mut scratch = vec![];
    let (mut p, mut at) = (0usize, 0usize);
    while p < records.len() {
        // the block's length is in the header: read it before the samples
        let hl = records[p] as usize;
        let mut head = records.get(p + 1..p + 1 + hl).ok_or_else(bad)?.to_vec();
        head.push(crc8(&head));
        let h = header(&head).filter(|h| h.channels == channels && h.bits.div_ceil(8) as usize == width).ok_or_else(bad)?;
        let bytes = h.block * channels * width;
        let samples = pcm.get(at..at + bytes).ok_or_else(bad)?;
        at += bytes;
        for (c, s) in chan.iter_mut().enumerate() {
            s.clear();
            for f in samples.chunks_exact(channels * width) {
                let x = &f[c * width..(c + 1) * width];
                let mut v = [if x[width - 1] & 0x80 != 0 { 0xFF } else { 0 }; 4];
                v[..width].copy_from_slice(x);
                s.push(i32::from_le_bytes(v));
            }
        }
        write_frame(records, &mut p, &mut chan, &mut scratch, out)?;
    }
    if at != pcm.len() { return Err(bad()); }
    Ok(())
}

/// The extent of the run of frames at `b[i]`, by the frames' headers only: (end, work bytes).
/// The run goes on while a frame's header is followed, where the frame can end, by another.
pub fn extent(b: &[u8], i: usize) -> Result<(usize, usize), ExtentError> {
    let first = header(&b[i..]).ok_or(if b.len() - i < 16 && b[i..].starts_with(&[0xFF]) { ExtentError::Truncated } else { ExtentError::Invalid })?;
    let (mut o, mut frames, mut work) = (i, 0usize, 0usize);
    loop {
        let Some(h) = header(&b[o..]).filter(|h| (h.channels, h.bits) == (first.channels, first.bits)) else { break };
        // the next header, a frame's length further at most
        let from = o + h.len + 2;
        let to = (o + MAX_FRAME).min(b.len());
        let next = (from..to).find(|&q| b[q] == 0xFF && header(&b[q..]).is_some_and(|n| (n.channels, n.bits) == (first.channels, first.bits)));
        work += h.block * h.channels * first.bits.div_ceil(8) as usize;
        match next {
            Some(q) => {
                o = q;
                frames += 1;
            }
            // the last frame: it ends where the buffer does, or goes on past it
            None if to == b.len() && b.len() - o < MAX_FRAME => return if frames >= MIN_FRAMES { Err(ExtentError::Truncated) } else { Err(ExtentError::Invalid) },
            None => break,
        }
    }
    if frames < MIN_FRAMES { return Err(ExtentError::Invalid); }
    Ok((o, work))
}

#[cfg(test)]
pub mod tests {
    use super::*;

    fn rng(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }

    /// A frame of `chan` (as heard) written the way an encoder would: each channel by a predictor
    /// of its own (`kinds`: 0 constant if it can be, 1 verbatim, 8–12 fixed, 32.. linear), the
    /// pair coded as `pair` says.
    pub fn frame(number: u32, chan: &[Vec<i32>], bits: u32, pair: u8, kinds: &[u8], seed: u64) -> Vec<u8> {
        let mut s = seed;
        let n = chan[0].len();
        let mut rec = vec![];
        // header: fixed block size, the size in 16 bits, 44.1 kHz
        let ss = match bits { 8 => 1, 12 => 2, 16 => 4, 20 => 5, _ => 6 };
        let mut head = vec![0xFF, 0xF8, 0x79, (if pair != 0 { pair } else { chan.len() as u8 - 1 }) << 4 | ss << 1];
        if number < 0x80 { head.push(number as u8); } else { head.extend_from_slice(&[0xC0 | (number >> 6) as u8, 0x80 | (number & 63) as u8]); }
        head.extend_from_slice(&((n - 1) as u16).to_be_bytes());
        rec.push(head.len() as u8);
        rec.extend_from_slice(&head);
        let mut coded: Vec<Vec<i32>> = chan.to_vec();
        if pair != 0 {
            for i in 0..n {
                let (l, r) = (chan[0][i], chan[1][i]);
                (coded[0][i], coded[1][i]) = match pair { 8 => (l, l - r), 9 => (l - r, r), _ => ((l + r) >> 1, l - r) };
            }
        }
        for (c, x) in coded.iter().enumerate() {
            let kind = kinds[c % kinds.len()];
            let wasted = x.iter().fold(0u32, |a, &v| a | v as u32).trailing_zeros().min(8);
            let wasted = if kind == 1 || x.iter().all(|&v| v == 0) { 0 } else { wasted };
            let kind = if kind == 0 && x.iter().any(|&v| v != x[0]) { 9 } else { kind };
            rec.push(kind << 1 | (wasted > 0) as u8);
            if wasted > 0 { rec.push(wasted as u8); }
            if kind >= 32 {
                let order = (kind - 31) as usize;
                // about twice the last sample less the one before, and a little of the others
                rec.extend_from_slice(&[13, 10]);
                for k in 0..order {
                    let c = match (order, k) { (1, _) => 1000, (_, 0) => 2040, (_, 1) => -1020, _ => (rng(&mut s) % 31) as i16 - 15 };
                    rec.extend_from_slice(&c.to_be_bytes());
                }
            }
            if kind >= 8 {
                // partitions of an eighth of the block, a parameter each (large where the
                // predictor leaves the samples as they are); now and then raw
                let porder = if n % 8 == 0 && n / 8 > 32 { 3 } else { 0 };
                let k = if kind == 8 { bits as u8 - 3 } else if bits > 16 { 12 } else { 6 };
                rec.push((if k >= 13 { 1 } else { (rng(&mut s) % 2) as u8 }) << 4 | porder);
                for _ in 0..1 << porder { rec.push(if rng(&mut s) % 9 == 0 { 0x80 | 31 } else { k + (rng(&mut s) % 3) as u8 }); }
            }
        }
        let mut out = vec![];
        let mut work = chan.to_vec();
        write_frame(&rec, &mut 0, &mut work, &mut vec![], &mut out).unwrap();
        out
    }

    /// Samples of a tone under noise, `n` to each of `channels`.
    pub fn tone(n: usize, channels: usize, bits: u32, seed: u64) -> Vec<Vec<i32>> {
        let mut s = seed;
        (0..channels).map(|c| (0..n).map(|i| {
            let amp = (1i64 << (bits - 3)) as f64;
            (amp * ((i as f64) * 0.02 * (c as f64 + 1.3)).sin()) as i32 + (rng(&mut s) % 9) as i32 - 4
        }).collect()).collect()
    }

    /// A stream of `frames` frames of 1152 samples (the last one shorter), each channel and
    /// frame by another predictor.
    pub fn stream(frames: usize, channels: usize, bits: u32) -> Vec<u8> {
        let mut out = vec![];
        for f in 0..frames {
            let n = if f + 1 == frames { 700 } else { 1152 };
            let mut chan = tone(n, channels, bits, 11 + f as u64);
            if f % 5 == 4 { for v in chan[0].iter_mut() { *v = 77; } }
            if f % 6 == 3 { for c in chan.iter_mut() { for v in c.iter_mut() { *v &= !7; } } }
            let pair = if channels == 2 { [0, 8, 9, 10][f % 4] } else { 0 };
            let kinds = [[8u8, 34, 10, 1], [44, 12, 0, 39], [11, 33, 9, 63]][f % 3];
            out.extend(frame(f as u32, &chan, bits, pair, &kinds[f % 2..], 5 + f as u64));
        }
        out
    }

    #[test]
    fn frames_come_back() {
        assert_eq!((crc8(b"123456789"), crc16(b"123456789")), (0xF4, 0xFEE8));
        for (channels, bits) in [(2usize, 16u32), (1, 8), (2, 24), (6, 16), (1, 20), (2, 12)] {
            let mut file = stream(40, channels, bits);
            let frames = file.len();
            file.extend_from_slice(b"TAG and what follows the frames");
            let sp = split(&file).unwrap_or_else(|| panic!("{channels} channels of {bits} bits: not split"));
            assert_eq!((sp.len, sp.frames, sp.channels as usize, sp.width as u32), (frames, 40, channels, bits.div_ceil(8)));
            assert_eq!(sp.pcm.len(), (39 * 1152 + 700) * channels * sp.width as usize);
            let mut out = vec![];
            rebuild(&sp.records, &sp.pcm, channels, sp.width as usize, &mut out).unwrap();
            assert!(out == file[..frames], "{channels} channels of {bits} bits");
            // by headers alone the run goes on to the last header found: here the file ends
            // before another could come; with enough after it, the run is seen to end
            assert_eq!(extent(&file, 0), Err(ExtentError::Truncated));
            file.extend_from_slice(&vec![0u8; MAX_FRAME]);
            let end = extent(&file, 0).expect("the run's extent").0;
            assert!(end < frames && end + 8 * 1152 * channels > frames && header(&file[end..]).is_some(), "extent {end} of {frames}");
        }
        // a frame damaged in its samples ends the run before it; a run cut short, at the cut
        let file = stream(12, 2, 16);
        let third = {
            let mut o = 0;
            for _ in 0..3 { o += read_frame(&file[o..], &header(&file[o..]).unwrap(), &mut vec![vec![]; 2], &mut vec![]).unwrap(); }
            o
        };
        let mut bad = file.clone();
        bad[third + 40] ^= 4;
        assert_eq!(split(&bad).map(|s| (s.len, s.frames)), Some((third, 3)));
        assert_eq!(split(&file[..third + 100]).map(|s| (s.len, s.frames)), Some((third, 3)));
        assert!(split(&file[..third / 3]).is_none() && split(b"\xff\xf8 not a frame at all").is_none());
    }

    /// Damaged records and samples: an error or other bytes, never a panic or an output without
    /// bound.
    #[test]
    fn damaged_records_do_not_panic() {
        let file = stream(10, 2, 16);
        let sp = split(&file).unwrap();
        let mut s = 3u64;
        for _ in 0..3000 {
            let (mut rec, mut pcm) = (sp.records.clone(), sp.pcm.clone());
            for _ in 0..1 + rng(&mut s) % 3 {
                let which = if rng(&mut s) % 4 == 0 { &mut pcm } else { &mut rec };
                if which.is_empty() { continue; }
                let i = (rng(&mut s) % which.len() as u64) as usize;
                match rng(&mut s) % 3 {
                    0 => which[i] = rng(&mut s) as u8,
                    1 => which[i] ^= 1 << (rng(&mut s) % 8),
                    _ => which.truncate(i),
                }
            }
            let mut out = vec![];
            let _ = rebuild(&rec, &pcm, 2, 2, &mut out);
            // a frame is held to a few bytes a sample, whatever its record says
            assert!(out.len() <= 3 * sp.pcm.len(), "{} bytes from damaged records of {} bytes of samples", out.len(), sp.pcm.len());
        }
    }
}
