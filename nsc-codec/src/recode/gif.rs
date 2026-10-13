//! GIF: pictures out of their LZW codes.
//!
//! A GIF is a screen descriptor, colour tables, and blocks: extensions (timing, comments, loop
//! counts) and images, each a rectangle of colour indices compressed with LZW and cut into
//! sub-blocks of at most 255 bytes. [`split`] decodes every image to its indices — one byte a
//! pixel, rows in their natural order also where the file interlaces them — as a stream of
//! frames for the model of indexed pictures (`cm::indexed`), and keeps the rest as a record:
//! the bytes between the images as they are, and of each image what its encoder did beyond
//! plain LZW (`transform::lzw`), how it cut the sub-blocks, and anything after the end code.
//! The colour tables go with the frames (the model predicts by colour); the record says where
//! they stood. [`rebuild`] writes the file again from both.
//!
//! An image that does not decode to exactly its rectangle stays in the record as bytes.
use crate::cm::indexed;
use crate::transform::deflate::inflate::ExtentError;
use crate::transform::lzw;

/// Largest image taken apart (pixels).
const MAX_PIXELS: usize = 1 << 28;

/// A GIF taken apart.
pub struct Split {
    pub len: usize,
    pub records: Vec<u8>,
    /// The decoded images as a stream of frames (see `cm::indexed`).
    pub frames: Vec<u8>,
    /// Images in the file, and how many of them were decoded.
    pub found: usize,
    pub decoded: usize,
}

fn put_var(o: &mut Vec<u8>, v: usize) { crate::nsc3::format::put_var(o, v as u64) }
fn get_var(b: &[u8], p: &mut usize) -> Option<usize> { super::h264::get_var(b, p) }

/// The rows of an interlaced image in the order the file holds them: every eighth from the
/// first, every eighth from the fifth, every fourth from the third, every second from the second.
fn interlaced(height: usize) -> impl Iterator<Item = usize> {
    [(0usize, 8usize), (4, 8), (2, 4), (1, 2)].into_iter().flat_map(move |(from, step)| (from..height).step_by(step))
}

/// Where the blocks after the header start: past the screen descriptor and the global colours.
fn header(b: &[u8]) -> Option<usize> {
    if b.len() < 13 || !(b.starts_with(b"GIF87a") || b.starts_with(b"GIF89a")) { return None; }
    let o = 13 + if b[10] & 0x80 != 0 { 3 << ((b[10] & 7) + 1) } else { 0 };
    (o <= b.len()).then_some(o)
}

/// The sub-blocks at `b[o]`: (their content, their sizes, the offset after their terminator).
fn sub_blocks(b: &[u8], mut o: usize) -> Option<(Vec<u8>, Vec<u8>, usize)> {
    let (mut data, mut sizes) = (vec![], vec![]);
    loop {
        let n = *b.get(o)? as usize;
        o += 1;
        if n == 0 { return Some((data, sizes, o)); }
        data.extend_from_slice(b.get(o..o + n)?);
        sizes.push(n as u8);
        o += n;
    }
}

/// The offset after the sub-blocks at `b[o]`.
fn skip_blocks(b: &[u8], mut o: usize) -> Option<usize> {
    loop {
        let n = *b.get(o)? as usize;
        o += 1 + n;
        if n == 0 { return (o <= b.len()).then_some(o); }
    }
}

/// The sizes an encoder cuts `n` bytes into, as a rule: 255 each, the rest last.
fn usual_sizes(n: usize) -> impl Iterator<Item = u8> { (0..n.div_ceil(255)).map(move |k| (n - 255 * k).min(255) as u8) }

/// Take the GIF at the start of `b` apart: None unless it is one with an image that decodes.
pub fn split(b: &[u8]) -> Option<Split> {
    let mut o = header(b)?;
    let screen = (u16::from_le_bytes([b[6], b[7]]), u16::from_le_bytes([b[8], b[9]]));
    let mut sp = Split { len: 0, records: vec![], frames: indexed::header(screen.0, screen.1, &b[13..o]), found: 0, decoded: 0 };
    // what the graphic control extension before an image says of it: frame flags, transparent index
    let mut control = (0u8, 0u8);
    let flush = |records: &mut Vec<u8>, from: usize, to: usize| {
        if to > from {
            records.push(0);
            put_var(records, to - from);
            records.extend_from_slice(&b[from..to]);
        }
    };
    // the header as it is; its colours are with the frames
    flush(&mut sp.records, 0, 13);
    if o > 13 { sp.records.push(2); }
    // bytes kept as they are, from `raw` on
    let mut raw = o;
    loop {
        match b.get(o) {
            Some(0x3B) => {
                o += 1;
                break;
            }
            Some(0x21) => match skip_blocks(b, o + 2) {
                Some(end) => {
                    if let [0xF9, 4, packed, _, _, index, ..] = b[o + 1..end] {
                        control = ((packed & 1) * indexed::TRANSPARENT | if packed >> 2 & 7 == 2 { indexed::CLEAR } else { 0 }, index);
                    }
                    o = end;
                }
                None => break,
            },
            Some(0x2C) => {
                let Some(d) = b.get(o..o + 10) else { break };
                let (w, h) = (u16::from_le_bytes([d[5], d[6]]) as usize, u16::from_le_bytes([d[7], d[8]]) as usize);
                let at = o + 10 + if d[9] & 0x80 != 0 { 3 << ((d[9] & 7) + 1) } else { 0 };
                let Some(&min) = b.get(at) else { break };
                let Some((data, sizes, end)) = sub_blocks(b, at + 1) else { break };
                sp.found += 1;
                let pixels = w * h;
                let decoded = if (2..=8).contains(&min) && pixels > 0 && pixels <= MAX_PIXELS { lzw::decode(&data, &lzw::Params::gif(min as u32), pixels) } else { None };
                if let Some(d2) = decoded.filter(|x| x.bytes.len() == pixels) {
                    // the descriptor as it is, then the image (its colours are with its frame)
                    flush(&mut sp.records, raw, o + 10);
                    let interlace = d[9] & 0x40 != 0;
                    let r = &mut sp.records;
                    r.extend_from_slice(&[1, min, interlace as u8]);
                    put_var(r, d2.record.len());
                    r.extend_from_slice(&d2.record);
                    if sizes.iter().copied().eq(usual_sizes(data.len())) {
                        put_var(r, 0);
                    } else {
                        put_var(r, sizes.len() + 1);
                        r.extend_from_slice(&sizes);
                    }
                    put_var(r, data.len() - d2.consumed);
                    r.extend_from_slice(&data[d2.consumed..]);
                    let at16 = |k: usize| u16::from_le_bytes([d[k], d[k + 1]]);
                    let local = if at > o + 10 { indexed::LOCAL } else { 0 };
                    sp.frames.extend_from_slice(&indexed::frame(at16(1), at16(3), w as u16, h as u16, control.0 | local, control.1, &b[o + 10..at]));
                    if interlace {
                        let at = sp.frames.len();
                        sp.frames.resize(at + pixels, 0);
                        for (k, y) in interlaced(h).enumerate() { sp.frames[at + y * w..at + (y + 1) * w].copy_from_slice(&d2.bytes[k * w..(k + 1) * w]); }
                    } else {
                        sp.frames.extend_from_slice(&d2.bytes);
                    }
                    sp.decoded += 1;
                    raw = end;
                }
                // a control extension is for the image after it only
                control = (0, 0);
                o = end;
            }
            _ => break,
        }
    }
    if sp.decoded == 0 { return None; }
    flush(&mut sp.records, raw, o);
    sp.len = o;
    Some(sp)
}

/// Append the GIF [`split`] took apart, from its record and its stream of frames.
pub fn rebuild(records: &[u8], frames: &[u8], out: &mut Vec<u8>) -> Result<(), String> {
    let bad = || "gif: bad record (corrupt file)".to_string();
    let mut p = 0usize;
    let (_, global, mut reader) = indexed::Reader::new(frames).ok_or_else(bad)?;
    let mut rows = vec![];
    while p < records.len() {
        let tag = records[p];
        p += 1;
        match tag {
            0 => {
                let n = get_var(records, &mut p).ok_or_else(bad)?;
                out.extend_from_slice(records.get(p..p.checked_add(n).ok_or_else(bad)?).ok_or_else(bad)?);
                p += n;
                continue;
            }
            1 => {}
            2 => {
                out.extend_from_slice(global);
                continue;
            }
            _ => return Err(bad()),
        }
        let (&min, &interlace) = (records.get(p).ok_or_else(bad)?, records.get(p + 1).ok_or_else(bad)?);
        p += 2;
        // the next frame: its colours stand before its codes
        let frame = reader.next().ok_or_else(bad)?;
        out.extend_from_slice(frame.palette);
        let (w, h, pixels) = (frame.width, frame.height, frame.pixels);
        let n = get_var(records, &mut p).ok_or_else(bad)?;
        let record = records.get(p..p.checked_add(n).ok_or_else(bad)?).ok_or_else(bad)?;
        p += n;
        let mut data = vec![];
        if interlace != 0 {
            rows.clear();
            for y in interlaced(h) { rows.extend_from_slice(&pixels[y * w..(y + 1) * w]); }
            lzw::encode(&rows, record, &lzw::Params::gif(min as u32), &mut data)?;
        } else {
            lzw::encode(pixels, record, &lzw::Params::gif(min as u32), &mut data)?;
        }
        let blocks = get_var(records, &mut p).ok_or_else(bad)?;
        let sizes = if blocks == 0 { None } else {
            let s = records.get(p..p.checked_add(blocks - 1).ok_or_else(bad)?).ok_or_else(bad)?;
            p += blocks - 1;
            Some(s)
        };
        let n = get_var(records, &mut p).ok_or_else(bad)?;
        data.extend_from_slice(records.get(p..p.checked_add(n).ok_or_else(bad)?).ok_or_else(bad)?);
        p += n;
        out.push(min);
        let mut o = 0usize;
        let mut block = |n: u8, out: &mut Vec<u8>| -> Result<(), String> {
            let piece = data.get(o..o + n as usize).filter(|_| n > 0).ok_or_else(bad)?;
            out.push(n);
            out.extend_from_slice(piece);
            o += n as usize;
            Ok(())
        };
        match sizes {
            Some(s) => for &n in s { block(n, out)?; },
            None => for n in usual_sizes(data.len()) { block(n, out)?; },
        }
        if o != data.len() { return Err(bad()); }
        out.push(0);
    }
    if !reader.done() { return Err(bad()); }
    Ok(())
}

/// The extent of the GIF at `b[i]`, by its blocks: (end, pixels).
pub fn extent(b: &[u8], i: usize) -> Result<(usize, usize), ExtentError> {
    let b = &b[i..];
    let Some(mut o) = header(b) else {
        return Err(if b.len() < 13 && (b"GIF87a".starts_with(&b[..b.len().min(6)]) || b"GIF89a".starts_with(&b[..b.len().min(6)])) { ExtentError::Truncated } else { ExtentError::Invalid });
    };
    let mut work = 0usize;
    loop {
        match b.get(o) {
            None => return Err(ExtentError::Truncated),
            Some(0x3B) => return if work > 0 { Ok((i + o + 1, work)) } else { Err(ExtentError::Invalid) },
            Some(0x21) => o = skip_blocks(b, o + 2).ok_or(ExtentError::Truncated)?,
            Some(0x2C) => {
                let d = b.get(o..o + 10).ok_or(ExtentError::Truncated)?;
                work += u16::from_le_bytes([d[5], d[6]]) as usize * u16::from_le_bytes([d[7], d[8]]) as usize;
                let at = o + 10 + if d[9] & 0x80 != 0 { 3 << ((d[9] & 7) + 1) } else { 0 };
                o = skip_blocks(b, at + 1).ok_or(ExtentError::Truncated)?;
            }
            // not a block: the file ends here
            Some(_) => return if work > 0 { Ok((i + o, work)) } else { Err(ExtentError::Invalid) },
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::transform::lzw::tests::{compress, picture, Style, USUAL};

    fn rng(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }

    /// An image block: its descriptor (with local colours if `local`), and `pixels` compressed
    /// by an encoder of `style` into sub-blocks of `block` bytes.
    pub fn image(w: usize, h: usize, pixels: &[u8], bits: u32, interlace: bool, local: bool, style: Style, block: usize, tail: &[u8]) -> Vec<u8> {
        let mut out = vec![0x2C, 3, 0, 5, 0];
        out.extend_from_slice(&(w as u16).to_le_bytes());
        out.extend_from_slice(&(h as u16).to_le_bytes());
        out.push((interlace as u8) << 6 | if local { 0x80 | (bits as u8 - 1) } else { 0 });
        if local { out.extend((0..3usize << bits).map(|k| (k * 7) as u8)); }
        out.push(bits as u8);
        let rows: Vec<u8> = if interlace { interlaced(h).flat_map(|y| pixels[y * w..(y + 1) * w].to_vec()).collect() } else { pixels.to_vec() };
        let mut data = compress(&rows, &lzw::Params::gif(bits), style);
        data.extend_from_slice(tail);
        for piece in data.chunks(block) {
            out.push(piece.len() as u8);
            out.extend_from_slice(piece);
        }
        out.push(0);
        out
    }

    /// An animation of `frames` frames of `w` by `h` pixels of `bits` bits, with what real files
    /// have between them: timing, a loop count, a comment.
    pub fn animation(frames: usize, w: usize, h: usize, bits: u32) -> (Vec<u8>, Vec<Vec<u8>>) {
        let mut out = b"GIF89a".to_vec();
        out.extend_from_slice(&(w as u16).to_le_bytes());
        out.extend_from_slice(&(h as u16).to_le_bytes());
        out.extend_from_slice(&[0x80 | (bits as u8 - 1), 0, 0]);
        out.extend((0..3usize << bits).map(|k| (k * 5) as u8));
        out.extend_from_slice(b"\x21\xFF\x0BNETSCAPE2.0\x03\x01\x00\x00\x00");
        let mut all = vec![];
        let mut pixels = picture(w * h, bits, 40);
        let mut s = 9u64;
        for f in 0..frames {
            // the frame before, with a part of it changed
            for _ in 0..w * h / 40 {
                let at = (rng(&mut s) % (w * h) as u64) as usize;
                pixels[at] = (rng(&mut s) % (1 << bits)) as u8;
            }
            out.extend_from_slice(&[0x21, 0xF9, 4, 4, 10, 0, 0, 0]);
            if f == 1 { out.extend_from_slice(b"\x21\xFE\x08a remark\x00"); }
            let style = [USUAL, Style { clear_at: 4095, ..USUAL }, Style { clear_at: 0, ..USUAL }, Style { lazy: 60, ..USUAL }][f % 4];
            let tail: &[u8] = if f % 5 == 3 { b"\0\0" } else { b"" };
            out.extend(image(w, h, &pixels, bits, f % 3 == 1, f % 4 == 2, style, if f % 4 == 3 { 254 } else { 255 }, tail));
            all.push(pixels.clone());
        }
        out.push(0x3B);
        (out, all)
    }

    #[test]
    fn images_come_back() {
        for (frames, w, h, bits) in [(9usize, 61usize, 47usize, 8u32), (3, 200, 150, 4), (5, 16, 16, 2), (2, 300, 1, 8), (1, 1, 1, 2)] {
            let (mut file, pixels) = animation(frames, w, h, bits);
            let end = file.len();
            file.extend_from_slice(b"what follows the file");
            let sp = split(&file).unwrap_or_else(|| panic!("{frames} frames of {w}x{h}: not split"));
            assert_eq!((sp.len, sp.found, sp.decoded), (end, frames, frames), "{w}x{h}");
            let (screen, global, mut reader) = indexed::Reader::new(&sp.frames).unwrap();
            assert_eq!((screen, global.len()), ((w, h), 3 << bits));
            for (k, px) in pixels.iter().enumerate() {
                let f = reader.next().unwrap();
                assert_eq!((f.left, f.top, f.width, f.height), (3, 5, w, h));
                assert_eq!((f.flags & indexed::LOCAL != 0, f.palette.len()), (k % 4 == 2, if k % 4 == 2 { 3 << bits } else { 0 }));
                assert!(f.pixels == &px[..], "{w}x{h}: frame {k} is not its pixels in their natural order");
            }
            assert!(reader.done());
            let mut out = vec![];
            rebuild(&sp.records, &sp.frames, &mut out).unwrap();
            assert!(out == file[..end], "{w}x{h}");
            assert_eq!(extent(&file, 0), Ok((end, frames * w * h)));
            // the record is the bytes that are neither pixels nor colours (the header, the
            // extensions, the descriptors), and little more
            if frames == 9 { assert!(sp.records.len() < 600, "{w}x{h}: record of {} bytes", sp.records.len()); }
        }
        // a file cut short keeps the images before the cut; an image that does not decode stays
        let (file, _) = animation(6, 80, 60, 8);
        let third = {
            let mut o = header(&file).unwrap();
            let mut seen = 0;
            loop {
                match file[o] {
                    0x21 => o = skip_blocks(&file, o + 2).unwrap(),
                    _ if seen == 3 => break o,
                    _ => {
                        let at = o + 10 + if file[o + 9] & 0x80 != 0 { 3 << ((file[o + 9] & 7) + 1) } else { 0 };
                        o = skip_blocks(&file, at + 1).unwrap();
                        seen += 1;
                    }
                }
            }
        };
        let cut = &file[..third + 40];
        let sp = split(cut).unwrap();
        assert_eq!((sp.decoded, extent(cut, 0)), (3, Err(ExtentError::Truncated)));
        let mut out = vec![];
        rebuild(&sp.records, &sp.frames, &mut out).unwrap();
        assert!(out == cut[..sp.len] && sp.len <= third);
        let mut bad = file.clone();
        bad[third + 30] ^= 0x55;
        if let Some(sp) = split(&bad) {
            let mut out = vec![];
            rebuild(&sp.records, &sp.frames, &mut out).unwrap();
            assert!(out == bad[..sp.len]);
        }
        assert!(split(b"GIF89a but not one").is_none() && split(b"GIF").is_none());
    }

    /// Damaged files: what is split comes back. Damaged records: an error or other bytes, never
    /// a panic or an output without bound.
    #[test]
    fn damaged_files_and_records_do_not_panic() {
        let (file, _) = animation(7, 50, 40, 8);
        let mut s = 4u64;
        for _ in 0..1500 {
            let mut bad = file.clone();
            for _ in 0..1 + rng(&mut s) % 4 {
                let i = (rng(&mut s) % bad.len() as u64) as usize;
                if rng(&mut s) % 6 == 0 { bad.truncate(i.max(14)); } else { bad[i] ^= 1 << (rng(&mut s) % 8); }
            }
            let _ = extent(&bad, 0);
            if let Some(sp) = split(&bad) {
                let mut out = vec![];
                rebuild(&sp.records, &sp.frames, &mut out).unwrap();
                assert!(out == bad[..sp.len], "a damaged file that splits must come back");
            }
        }
        let sp = split(&file).unwrap();
        for _ in 0..1500 {
            let (mut rec, mut frames) = (sp.records.clone(), sp.frames.clone());
            let which = if rng(&mut s) % 3 == 0 { &mut frames } else { &mut rec };
            let i = (rng(&mut s) % which.len() as u64) as usize;
            if rng(&mut s) % 4 == 0 { which.truncate(i); } else { which[i] ^= 1 << (rng(&mut s) % 8); }
            let mut out = vec![];
            let _ = rebuild(&rec, &frames, &mut out);
            assert!(out.len() < 8 * file.len() + (1 << 16), "{} bytes from a damaged record", out.len());
        }
    }
}
