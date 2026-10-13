//! Lossless JPEG: samples out of their Huffman codes.
//!
//! The lossless mode of JPEG (ITU-T T.81 process 14, "LJ92") is what Canon's CR2 and most
//! compressed DNG files hold their sensor data in: each sample is predicted from its
//! neighbours and the difference written as a Huffman-coded size and that many bits. [`split`]
//! decodes a stream to its samples — for the image model, which predicts them far better than
//! one fixed predictor and a static code — and [`rebuild`] writes the codes again from the
//! stream's own header, which has the tables. A sample has one code, so the stream comes back
//! exactly; only the last byte's padding is noted.
use crate::transform::deflate::inflate::ExtentError;

/// A stream taken apart.
pub struct Split {
    pub len: usize,
    /// The stream up to its entropy-coded data, then one byte: the padding of its last byte.
    pub header: Vec<u8>,
    /// The samples, row by row, the components of a pixel together: one byte each up to 8 bits
    /// of precision, two (low first) above.
    pub samples: Vec<u8>,
    /// Samples a row (pixels times components), bytes a sample, components.
    pub row: usize,
    pub wide: bool,
    pub components: usize,
}

/// What a header says of its scan.
struct Frame {
    precision: u32,
    width: usize,
    height: usize,
    /// Per component: its Huffman table (code lengths' counts, then values).
    tables: Vec<([u8; 17], Vec<u8>)>,
    predictor: u8,
    shift: u32,
    /// Where the entropy-coded data starts.
    data: usize,
}

const MAX_SAMPLES: usize = 1 << 28;

fn be16(b: &[u8], o: usize) -> Option<usize> { Some(u16::from_be_bytes([*b.get(o)?, *b.get(o + 1)?]) as usize) }

/// Read the header of the lossless stream at the start of `b`. Err(true): `b` ends inside it.
fn frame(b: &[u8]) -> Result<Frame, bool> {
    if b.len() < 4 { return Err(b.len() < 2 || b[..2] == [0xFF, 0xD8]); }
    if b[..3] != [0xFF, 0xD8, 0xFF] { return Err(false); }
    let mut dht: [Option<([u8; 17], Vec<u8>)>; 4] = [None, None, None, None];
    let mut sof: Option<(u32, usize, usize, Vec<u8>)> = None;
    let mut o = 2usize;
    loop {
        if b.get(o).ok_or(true)? != &0xFF { return Err(false); }
        let m = *b.get(o + 1).ok_or(true)?;
        let len = be16(b, o + 2).ok_or(true)?;
        if len < 2 { return Err(false); }
        let seg = b.get(o + 4..o + 2 + len).ok_or(true)?;
        match m {
            0xC4 => {
                let mut p = 0usize;
                while p + 17 <= seg.len() {
                    let (class, id) = (seg[p] >> 4, (seg[p] & 15) as usize);
                    let mut counts = [0u8; 17];
                    counts[1..].copy_from_slice(&seg[p + 1..p + 17]);
                    let n: usize = counts.iter().map(|&c| c as usize).sum();
                    let vals = seg.get(p + 17..p + 17 + n).ok_or(false)?;
                    if class != 0 || id > 3 || n == 0 || n > 17 || vals.iter().any(|&v| v > 16) { return Err(false); }
                    dht[id] = Some((counts, vals.to_vec()));
                    p += 17 + n;
                }
                if p != seg.len() { return Err(false); }
            }
            0xC3 => {
                if seg.len() < 6 || sof.is_some() { return Err(false); }
                let n = seg[5] as usize;
                if !(1..=4).contains(&n) || seg.len() != 6 + 3 * n || (0..n).any(|k| seg[7 + 3 * k] != 0x11) { return Err(false); }
                sof = Some((seg[0] as u32, be16(seg, 3).ok_or(false)?, be16(seg, 1).ok_or(false)?, (0..n).map(|k| seg[6 + 3 * k]).collect()));
            }
            0xDA => {
                let (precision, width, height, ids) = sof.ok_or(false)?;
                let n = *seg.first().ok_or(false)? as usize;
                if n != ids.len() || seg.len() != 4 + 2 * n { return Err(false); }
                let mut tables = vec![];
                for k in 0..n {
                    if seg[1 + 2 * k] != ids[k] { return Err(false); }
                    tables.push(dht[(seg[2 + 2 * k] >> 4) as usize & 3].clone().ok_or(false)?);
                }
                let (predictor, shift) = (seg[1 + 2 * n], (seg[3 + 2 * n] & 15) as u32);
                if !(1..=7).contains(&predictor) || !(2..=16).contains(&precision) || shift >= precision || width == 0 || height == 0
                    || width * height * n > MAX_SAMPLES { return Err(false); }
                return Ok(Frame { precision, width, height, tables, predictor, shift, data: o + 2 + len });
            }
            // restart intervals, other processes, and what is no header segment: not taken
            0xDD | 0xC0..=0xC2 | 0xC5..=0xCF | 0xD0..=0xD9 | 0x00 | 0xFF => return Err(false),
            _ => {}
        }
        o += 2 + len;
    }
}

/// Canonical codes of a table: per value (a size 0..=16), its code and length.
fn codes(t: &([u8; 17], Vec<u8>)) -> [(u32, u32); 17] {
    let mut out = [(0u32, 0u32); 17];
    let (mut code, mut k) = (0u32, 0usize);
    for len in 1..=16u32 {
        for _ in 0..t.0[len as usize] {
            if let Some(&v) = t.1.get(k) { out[v as usize] = (code, len); }
            code += 1;
            k += 1;
        }
        code <<= 1;
    }
    out
}

/// The prediction of the sample at column `x` (in pixels) of row `y`, component `c`.
#[inline]
fn predict(s: &[u16], f: &Frame, n: usize, x: usize, y: usize, c: usize) -> u16 {
    let at = (y * f.width + x) * n + c;
    if y == 0 {
        return if x == 0 { 1u16 << (f.precision - f.shift - 1) } else { s[at - n] };
    }
    let up = s[at - f.width * n];
    if x == 0 { return up; }
    let (a, b, c) = (s[at - n] as i32, up as i32, s[at - f.width * n - n] as i32);
    (match f.predictor {
        1 => a,
        2 => b,
        3 => c,
        4 => a + b - c,
        5 => a + ((b - c) >> 1),
        6 => b + ((a - c) >> 1),
        _ => (a + b) >> 1,
    }) as u16
}

/// Take the lossless JPEG at the start of `b` apart; None unless it is one that decodes to its end.
pub fn split(b: &[u8]) -> Option<Split> {
    let f = frame(b).ok()?;
    let n = f.tables.len();
    // per table: for the next 16 bits, the size coded and its code's length
    let look: Vec<Vec<(u8, u8)>> = f.tables.iter().map(|t| {
        let mut l = vec![(0u8, 0u8); 1 << 16];
        for (v, &(code, len)) in codes(t).iter().enumerate() {
            if len == 0 { continue; }
            let from = (code << (16 - len)) as usize;
            for e in &mut l[from..from + (1 << (16 - len))] { *e = (v as u8, len as u8); }
        }
        l
    }).collect();
    let total = f.width * f.height * n;
    let mut s = vec![0u16; total];
    // the bits: a stuffed zero after each FF byte is skipped as the bytes are taken
    let (mut acc, mut bits, mut p) = (0u64, 0u32, f.data);
    let mut marker = false;
    for y in 0..f.height {
        for x in 0..f.width {
            for c in 0..n {
                while bits <= 48 {
                    let mut byte = 0u8;
                    if !marker {
                        match (b.get(p), b.get(p + 1)) {
                            (Some(0xFF), Some(0)) => { byte = 0xFF; p += 2; }
                            (Some(0xFF), _) => marker = true,
                            (Some(&v), _) => { byte = v; p += 1; }
                            (None, _) => marker = true,
                        }
                    }
                    acc = acc << 8 | byte as u64;
                    bits += 8;
                }
                let (size, len) = look[c][(acc >> (bits - 16)) as usize & 0xFFFF];
                if len == 0 { return None; }
                bits -= len as u32;
                let diff = if size == 16 { 32768i32 } else if size == 0 { 0 } else {
                    bits -= size as u32;
                    let v = (acc >> bits) as i32 & ((1 << size) - 1);
                    if v < 1 << (size - 1) { v - (1 << size) + 1 } else { v }
                };
                let at = (y * f.width + x) * n + c;
                s[at] = predict(&s, &f, n, x, y, c).wrapping_add(diff as u16);
            }
        }
    }
    let _ = p;
    let wide = f.precision > 8;
    let mut samples = Vec::with_capacity(total * (1 + wide as usize));
    for &v in &s { if wide { samples.extend_from_slice(&v.to_le_bytes()); } else { samples.push(v as u8); } }
    let mut header = b[..f.data].to_vec();
    header.push(0);
    // what is written again must be the stream, and says where it ends; the last data byte's
    // padding (ones as a rule, or zeros) is taken from the stream
    for pad in [0xFFu8, 0] {
        *header.last_mut()? = pad;
        let mut back = Vec::with_capacity(b.len().min(total * 3));
        rebuild(&header, &samples, &mut back).ok()?;
        let len = back.len();
        if len > b.len() || len < f.data + 3 || b[len - 2..len] != [0xFF, 0xD9] { continue; }
        *header.last_mut()? = if b[len - 3] == 0 && b[len - 4] == 0xFF { 0xFF } else { b[len - 3] };
        back.clear();
        rebuild(&header, &samples, &mut back).ok()?;
        if back == b[..len.min(b.len())] && back.len() == len { return Some(Split { len, header, samples, row: f.width * n, wide, components: n }); }
    }
    None
}

/// Append the stream [`split`] took apart.
pub fn rebuild(header: &[u8], samples: &[u8], out: &mut Vec<u8>) -> Result<(), String> {
    let bad = || "lossless jpeg: bad record (corrupt file)".to_string();
    let (&last, head) = header.split_last().ok_or_else(bad)?;
    let f = frame(head).map_err(|_| bad())?;
    if f.data != head.len() { return Err(bad()); }
    let n = f.tables.len();
    let wide = f.precision > 8;
    let total = f.width * f.height * n;
    if samples.len() != total * (1 + wide as usize) { return Err(bad()); }
    let s: Vec<u16> = if wide { samples.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect() } else { samples.iter().map(|&v| v as u16).collect() };
    let tables: Vec<[(u32, u32); 17]> = f.tables.iter().map(codes).collect();
    out.extend_from_slice(head);
    let start = out.len();
    let (mut acc, mut bits) = (0u64, 0u32);
    let mut put = |out: &mut Vec<u8>, acc: &mut u64, bits: &mut u32, v: u32, len: u32| {
        *acc = *acc << len | v as u64;
        *bits += len;
        while *bits >= 8 {
            *bits -= 8;
            let byte = (*acc >> *bits) as u8;
            out.push(byte);
            if byte == 0xFF { out.push(0); }
        }
    };
    for y in 0..f.height {
        for x in 0..f.width {
            for c in 0..n {
                let at = (y * f.width + x) * n + c;
                let d = s[at].wrapping_sub(predict(&s, &f, n, x, y, c)) as i16;
                let (size, low) = if d == i16::MIN { (16u32, 0u32) } else {
                    let m = d.unsigned_abs() as u32;
                    let size = 32 - m.leading_zeros();
                    (size, if d < 0 { (d as i32 + (1 << size) - 1) as u32 } else { d as u32 })
                };
                let (code, len) = tables[c][size as usize];
                if len == 0 { return Err(bad()); }
                put(out, &mut acc, &mut bits, code, len);
                if size > 0 && size < 16 { put(out, &mut acc, &mut bits, low, size); }
            }
        }
    }
    if bits > 0 {
        // the last byte as the stream had it (its padding)
        let byte = ((acc << (8 - bits)) as u8) | (last & (0xFF >> bits));
        out.push(byte);
        if byte == 0xFF { out.push(0); }
    }
    if out.len() == start { return Err(bad()); }
    out.extend_from_slice(&[0xFF, 0xD9]);
    Ok(())
}

/// The extent of the lossless JPEG at `b[i]`: (end, bytes of samples), by its header and the
/// marker that ends its data.
pub fn extent(b: &[u8], i: usize) -> Result<(usize, usize), ExtentError> {
    let f = frame(&b[i..]).map_err(|cut| if cut { ExtentError::Truncated } else { ExtentError::Invalid })?;
    let mut p = i + f.data;
    loop {
        match b[p..].iter().position(|&x| x == 0xFF) {
            None => return Err(ExtentError::Truncated),
            Some(k) => match b.get(p + k + 1) {
                None => return Err(ExtentError::Truncated),
                Some(0) => p += k + 2,
                Some(0xD9) => return Ok((p + k + 2, f.width * f.height * f.tables.len() * 2)),
                Some(_) => return Err(ExtentError::Invalid),
            },
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// A lossless JPEG of `w` by `h` pixels of `n` components and `precision` bits, predictor
    /// `predictor`: smooth samples with noise.
    pub fn stream(w: usize, h: usize, n: usize, precision: u32, predictor: u8, seed: u64) -> (Vec<u8>, Vec<u16>) {
        let mut s = seed;
        let mut rng = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let max = (1u32 << precision) - 1;
        let px: Vec<u16> = (0..w * h * n).map(|i| {
            let (x, y) = (i / n % w, i / n / w);
            let v = (x * 7 + y * 5 + (i % n) * 40) as u32 % (max / 2 + 1) + max / 4 + (rng() % 9) as u32;
            if rng() % 97 == 0 { (rng() as u32 & max) as u16 } else { v.min(max) as u16 }
        }).collect();
        // one table for all: sizes 0..=16 with lengths that make a full code
        let mut head = vec![0xFF, 0xD8, 0xFF, 0xC4, 0, 36, 0];
        head.extend_from_slice(&[0, 0, 2, 5, 6, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]);
        head.extend_from_slice(&[2, 3, 1, 4, 5, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        head.extend_from_slice(&[0xFF, 0xC3, 0, (8 + 3 * n) as u8, precision as u8, (h >> 8) as u8, h as u8, (w >> 8) as u8, w as u8, n as u8]);
        for k in 0..n { head.extend_from_slice(&[k as u8 + 1, 0x11, 0]); }
        head.extend_from_slice(&[0xFF, 0xDA, 0, (6 + 2 * n) as u8, n as u8]);
        for k in 0..n { head.extend_from_slice(&[k as u8 + 1, 0]); }
        head.extend_from_slice(&[predictor, 0, 0]);
        head.push(0xFF);
        let bytes: Vec<u8> = if precision > 8 { px.iter().flat_map(|v| v.to_le_bytes()).collect() } else { px.iter().map(|&v| v as u8).collect() };
        let mut out = vec![];
        rebuild(&head, &bytes, &mut out).unwrap();
        (out, px)
    }

    #[test]
    fn samples_come_back_and_so_does_the_stream() {
        for (w, h, n, precision, predictor) in [(64usize, 40usize, 1usize, 12u32, 1u8), (33, 21, 2, 14, 4), (50, 30, 3, 8, 7), (17, 9, 4, 16, 6), (1, 1, 1, 2, 1), (40, 25, 2, 12, 5), (20, 20, 1, 10, 2), (20, 20, 1, 10, 3)] {
            let what = format!("{w}x{h}, {n} components of {precision} bits, predictor {predictor}");
            let (mut f, px) = stream(w, h, n, precision, predictor, 5);
            let end = f.len();
            f.extend_from_slice(b"after");
            let sp = split(&f).unwrap_or_else(|| panic!("{what}: not split"));
            assert_eq!((sp.len, sp.row, sp.wide, sp.components), (end, w * n, precision > 8, n), "{what}");
            let got: Vec<u16> = if sp.wide { sp.samples.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect() } else { sp.samples.iter().map(|&v| v as u16).collect() };
            assert!(got == px, "{what}: samples");
            assert_eq!(extent(&f, 0).map(|e| e.0), Ok(end), "{what}");
            assert_eq!(extent(&f[..end - 1], 0), Err(ExtentError::Truncated), "{what}");
        }
        assert!(split(b"\xFF\xD8\xFF\xE0\0\x10JFIF").is_none());
    }

    /// Damaged streams: what is split comes back. Damaged records: an error or other bytes.
    #[test]
    fn damaged_streams_and_records_do_not_panic() {
        let (f, _) = stream(60, 45, 2, 12, 4, 9);
        let mut s = 3u64;
        let mut rng = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let sp = split(&f).unwrap();
        for _ in 0..1500 {
            let mut bad = f.clone();
            for _ in 0..1 + rng() % 3 {
                let i = (rng() % bad.len() as u64) as usize;
                if rng() % 6 == 0 { bad.truncate(i.max(4)); } else { bad[i] ^= 1 << (rng() % 8); }
            }
            let _ = extent(&bad, 0);
            if let Some(sp) = split(&bad) {
                let mut out = vec![];
                rebuild(&sp.header, &sp.samples, &mut out).unwrap();
                assert!(out == bad[..sp.len]);
            }
            let (mut head, mut samples) = (sp.header.clone(), sp.samples.clone());
            let which = if rng() % 2 == 0 { &mut head } else { &mut samples };
            let i = (rng() % which.len() as u64) as usize;
            if rng() % 4 == 0 { which.truncate(i); } else { which[i] ^= 1 << (rng() % 8); }
            let mut out = vec![];
            let _ = rebuild(&head, &samples, &mut out);
            assert!(out.len() < 40 * f.len() + (1 << 16));
        }
    }
}
