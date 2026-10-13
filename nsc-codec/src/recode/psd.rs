//! Photoshop documents: pixels out of their run-length rows.
//!
//! A PSD keeps the channels of every layer, and once more of the whole picture, one plane
//! after the other, each row packed by itself with PackBits behind a table of the rows' packed
//! sizes. [`split`] unpacks the rows and puts the planes of a layer (or of the picture) back
//! together as pixels — red, green and blue of a pixel side by side — for the image model,
//! which is where the gain is: packed, one channel's rows say nothing of the next channel's.
//! What it keeps beside the pixels is where each plane's data stood and, where a row was not
//! packed the way Photoshop packs (`transform::packbits`), how it was; the tables of sizes
//! follow from the packing. [`rebuild`] writes the file from its first packed plane to its
//! last again.
use crate::detect::{image_params, imgfmt};
use crate::transform::deflate::inflate::ExtentError;
use crate::transform::packbits::{self, Style};

const VERSION: u8 = 1;
/// Bounds: layers of a file, channels of a layer, rows and row bytes of a plane.
const MAX_LAYERS: usize = 8192;
const MAX_CHANNELS: usize = 56;
const MAX_SIDE: usize = 300_000;

fn put_var(o: &mut Vec<u8>, v: usize) { crate::nsc3::format::put_var(o, v as u64) }
fn get_var(b: &[u8], p: &mut usize) -> Option<usize> { super::h264::get_var(b, p) }
fn be16(b: &[u8], o: usize) -> Option<usize> { Some(u16::from_be_bytes([*b.get(o)?, *b.get(o + 1)?]) as usize) }
fn be32(b: &[u8], o: usize) -> Option<usize> { Some(u32::from_be_bytes(b.get(o..o + 4)?.try_into().ok()?) as usize) }

/// Planes packed row by row behind one table of row sizes: where the table is, the planes'
/// rows and bytes a row, how many planes, and which pixels they are planes of.
#[derive(Clone, Copy)]
struct Block {
    at: usize,
    rows: usize,
    width: usize,
    planes: usize,
    group: usize,
    /// Where the planes' left edge is in the document (bytes).
    left: usize,
    /// The channel (0 red, 1 green, 2 blue, -1 transparency; `MERGED`: the whole picture's),
    /// and where the layer's corner is in the document (pixels).
    id: i16,
    corner: (i64, i64),
    /// The layer's opacity, and whether it is shown.
    opacity: u8,
    shown: bool,
}

const MERGED: i16 = i16::MAX;

/// A layer as the merged picture is made of it: its pixels (`width` a row, `samples` each),
/// its corner in the document, which samples are red, green and blue and which (if any) its
/// transparency, and its opacity.
struct Under<'a> {
    pixels: &'a [u8],
    width: usize,
    samples: usize,
    corner: (i64, i64),
    rgb: [usize; 3],
    alpha: Option<usize>,
    opacity: u8,
}

/// Take what the layers make together off the picture (`back`: put it on again). A document's
/// merged picture is its layers laid over each other, the lowest first, each by its
/// transparency — where the layers are plain ones, exactly, and the difference is nothing.
/// `picture` is `pw` pixels wide of `pc` samples, the first three its colours.
fn overlay(picture: &mut [u8], pw: usize, pc: usize, layers: &[Under], back: bool) {
    let ph = picture.len() / (pw * pc);
    let mut made = vec![255u8; pw * ph * 3];
    for u in layers {
        let lh = u.pixels.len() / (u.width * u.samples);
        let (top, left) = u.corner;
        for y in top.max(0)..(top + lh as i64).min(ph as i64) {
            for x in left.max(0)..(left + u.width as i64).min(pw as i64) {
                let l = ((y - top) as usize * u.width + (x - left) as usize) * u.samples;
                let mut a = u.alpha.map_or(255, |k| u.pixels[l + k] as u32);
                if u.opacity < 255 { a = (a * u.opacity as u32 + 127) / 255; }
                let m = (y as usize * pw + x as usize) * 3;
                for c in 0..3 {
                    made[m + c] = ((u.pixels[l + u.rgb[c]] as u32 * a + made[m + c] as u32 * (255 - a) + 127) / 255) as u8;
                }
            }
        }
    }
    for (p, m) in picture.chunks_exact_mut(pc).zip(made.chunks_exact(3)) {
        for c in 0..3 { p[c] = if back { p[c].wrapping_add(m[c]) } else { p[c].wrapping_sub(m[c]) }; }
    }
}


/// What the document's sections say: its packed planes in file order, where the document
/// ends, and the bytes of a sample. Err(true): the buffer ends inside it.
fn blocks(b: &[u8]) -> Result<(Vec<Block>, usize, usize), bool> {
    if b.len() < 26 { return Err(b"8BPS".starts_with(&b[..b.len().min(4)])); }
    if &b[..4] != b"8BPS" || b[4..6] != [0, 1] { return Err(false); }
    let (channels, height, width, depth) = (be16(b, 12).unwrap(), be32(b, 14).unwrap(), be32(b, 18).unwrap(), be16(b, 22).unwrap());
    if !(1..=MAX_CHANNELS).contains(&channels) || height == 0 || width == 0 || height > MAX_SIDE || width > MAX_SIDE || !matches!(depth, 1 | 8 | 16 | 32) { return Err(false); }
    let row = |w: usize| (w * depth).div_ceil(8);
    let mut o = 26usize;
    for _ in 0..2 { o = o + 4 + be32(b, o).ok_or(true)?; }
    let lm = be32(b, o).ok_or(true)?;
    let (layers_at, merged) = (o + 4, o + 4 + lm);
    let mut out = vec![];
    let mut group = 0usize;
    if lm >= 6 {
        // the layers' records, then every channel's data in the same order
        let count = (be16(b, layers_at + 4).ok_or(true)? as u16 as i16).unsigned_abs() as usize;
        if count > MAX_LAYERS { return Err(false); }
        let mut q = layers_at + 6;
        let mut layers = Vec::with_capacity(count);
        for _ in 0..count {
            let rect: Vec<i64> = (0..4).map(|k| be32(b, q + 4 * k).map(|v| v as u32 as i32 as i64)).collect::<Option<_>>().ok_or(true)?;
            let n = be16(b, q + 16).ok_or(true)?;
            if n > MAX_CHANNELS { return Err(false); }
            let chans: Vec<(i16, usize)> = (0..n).map(|k| Some((be16(b, q + 18 + 6 * k)? as u16 as i16, be32(b, q + 20 + 6 * k)?))).collect::<Option<_>>().ok_or(true)?;
            let look = b.get(q + 18 + 6 * n + 8..q + 18 + 6 * n + 11).ok_or(true)?;
            let (opacity, shown) = (look[0], look[2] & 2 == 0);
            q += 18 + 6 * n + 12;
            q += 4 + be32(b, q).ok_or(true)?;
            let (h, w) = ((rect[2] - rect[0]).max(0) as usize, (rect[3] - rect[1]).max(0) as usize);
            if h > MAX_SIDE || w > MAX_SIDE { return Err(false); }
            layers.push((h, row(w), chans, row(rect[1].rem_euclid(1 << 16) as usize), (rect[0], rect[1]), opacity, shown));
        }
        for (h, w, chans, left, corner, opacity, shown) in layers {
            for (id, len) in chans {
                // a channel of the layer's own size (masks have theirs), packed
                if id >= -1 && h > 0 && w > 0 && len > 2 + 2 * h && be16(b, q) == Some(1) {
                    out.push(Block { at: q + 2, rows: h, width: w, planes: 1, group, left, id, corner, opacity, shown });
                }
                q = q.checked_add(len).ok_or(false)?;
            }
            group += 1;
        }
        if q > merged { return Err(false); }
    }
    // the picture as a whole: all its channels behind one table
    let end = match be16(b, merged).ok_or(true)? {
        0 => merged + 2 + channels * height * row(width),
        1 => {
            let rows = channels * height;
            let table = b.get(merged + 2..merged + 2 + 2 * rows).ok_or(true)?;
            out.push(Block { at: merged + 2, rows: height, width: row(width), planes: channels, group, left: 0, id: MERGED, corner: (0, 0), opacity: 255, shown: true });
            merged + 2 + 2 * rows + table.chunks_exact(2).map(|c| u16::from_be_bytes([c[0], c[1]]) as usize).sum::<usize>()
        }
        _ => return Err(false),
    };
    if end > b.len() { return Err(true); }
    Ok((out, end, depth.div_ceil(8).min(2)))
}

/// The packed sizes of the rows of `width` bytes in `packed`; None if a packet runs over a row's end.
fn row_sizes(packed: &[u8], width: usize, rows: usize) -> Option<Vec<usize>> {
    let mut out = Vec::with_capacity(rows);
    let (mut p, mut from, mut got) = (0usize, 0usize, 0usize);
    while out.len() < rows {
        let c = *packed.get(p)? as usize;
        let (n, takes) = match c {
            0..=127 => (c + 1, c + 2),
            128 => (0, 1),
            _ => (257 - c, 2),
        };
        p += takes;
        got += n;
        if got > width { return None; }
        if got == width {
            out.push(p - from);
            (from, got) = (p, 0);
        }
    }
    (p == packed.len()).then_some(out)
}

/// A document taken apart.
pub struct Split {
    /// The span taken apart: from its first packed plane to its last.
    pub start: usize,
    pub len: usize,
    pub records: Vec<u8>,
    pub gaps: Vec<u8>,
    /// The layers' (and the picture's) pixels with the image model's parameters.
    pub images: Vec<(Vec<u8>, Vec<u8>)>,
    pub planes: usize,
}

/// Take the Photoshop document at the start of `b` apart, for at most `budget` bytes of pixels.
pub fn split(b: &[u8], budget: usize) -> Option<Split> {
    let (all, _, sample) = blocks(b).ok()?;
    // per group of planes: its blocks unpacked — (block, the planes' bytes, the packing's record)
    let mut groups: Vec<Vec<(Block, Vec<u8>, Vec<u8>, usize)>> = vec![];
    let mut left = budget;
    let mut k = 0usize;
    while k < all.len() {
        let n = all[k..].iter().take_while(|x| x.group == all[k].group).count();
        let these = &all[k..k + n];
        k += n;
        let (rows, width) = (these[0].rows, these[0].width);
        let planes: usize = these.iter().map(|x| x.planes).sum();
        let total = rows.checked_mul(width)?.checked_mul(planes)?;
        if total == 0 || total > left || these.iter().any(|x| (x.rows, x.width) != (rows, width)) { continue; }
        let mut got = Vec::with_capacity(n);
        for x in these {
            let count = x.rows * x.planes;
            let Some(table) = b.get(x.at..x.at + 2 * count) else { break };
            let sizes: Vec<usize> = table.chunks_exact(2).map(|c| u16::from_be_bytes([c[0], c[1]]) as usize).collect();
            let Some(data) = b.get(x.at + 2 * count..x.at + 2 * count + sizes.iter().sum::<usize>()) else { break };
            // the rows must be packed each by itself, as the table says
            if row_sizes(data, x.width, count).as_deref() != Some(&sizes[..]) { break; }
            // Photoshop packs up to the edges of its tiles: which size of tile fits the first rows?
            let some: usize = sizes.iter().take(24).sum();
            let rows = count.min(24);
            let lead = [0usize, 1024, 512, 256, 2048, 4096].iter().map(|&t| if t == 0 { 0 } else { (t - x.left % t) % t })
                .min_by_key(|&l| packbits::decode(&data[..some], x.width, rows * x.width, Style::Adobe(l)).map_or(usize::MAX, |d| d.record.len())).unwrap_or(0);
            let Some(d) = packbits::decode(data, x.width, count * x.width, Style::Adobe(lead)).filter(|d| d.consumed == data.len()) else { break };
            got.push((*x, d.bytes, d.record, lead));
        }
        if got.len() == n {
            left -= total;
            groups.push(got);
        }
    }
    if groups.is_empty() { return None; }
    let mut records = vec![VERSION];
    put_var(&mut records, groups.len());
    let mut images: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(groups.len());
    let mut sides = Vec::with_capacity(groups.len());
    for g in &groups {
        let (rows, width) = (g[0].0.rows, g[0].0.width);
        let planes: Vec<&[u8]> = g.iter().flat_map(|x| x.1.chunks_exact(rows * width)).collect();
        // more planes than a pixel has samples for the model: each a picture of its own, in a row
        let n = planes.len();
        let side = if n <= 4 && width % sample == 0 { n } else { 1 };
        put_var(&mut records, width);
        put_var(&mut records, rows);
        records.extend_from_slice(&[n as u8, sample as u8, side as u8]);
        let mut px = Vec::with_capacity(rows * width * n);
        if side == 1 {
            for p in &planes { px.extend_from_slice(p); }
        } else {
            for at in (0..rows * width).step_by(sample) { for p in &planes { px.extend_from_slice(&p[at..at + sample]); } }
        }
        let w = (width / sample) as u32;
        let params = match (sample, side) {
            (1, 3) => image_params(imgfmt::RGB24, w, (width * 3) as u32, 3, false),
            (1, 4) => image_params(imgfmt::RGBA32, w, (width * 4) as u32, 4, false),
            (1, s) => image_params(imgfmt::GRAY8, w, (width * s) as u32, s as u8, false),
            (_, 1) => image_params(imgfmt::GRAY16BE, w, width as u32, 1, false),
            (_, s) => image_params(imgfmt::RGB48BE, w, (width * s) as u32, s as u8, false),
        };
        images.push((params, px));
        sides.push((width, n, side, sample));
    }
    // the merged picture as its difference to what the layers make together, where that is
    // most of it (by a sample of its bytes)
    let mut under: Vec<(usize, [usize; 3], u8)> = vec![];
    let merged = groups.iter().position(|g| g[0].0.id == MERGED).filter(|&m| sides[m].2 >= 3 && sides[m].3 == 1);
    if let Some(m) = merged {
        for (gi, g) in groups.iter().enumerate() {
            let (_, lc, side, sample) = sides[gi];
            let find = |id: i16| g.iter().position(|x| x.0.id == id);
            let (Some(r), Some(gr), Some(bl)) = (find(0), find(1), find(2)) else { continue };
            if gi != m && side == lc && sample == 1 && g[0].0.shown { under.push((gi, [r, gr, bl], find(-1).map_or(255, |k| k as u8))); }
        }
        if !under.is_empty() {
            let (pw, pc) = (sides[m].0, sides[m].1);
            let mut picture = std::mem::take(&mut images[m].1);
            let before = picture.iter().step_by(61).filter(|&&x| x == 0).count();
            let layers: Vec<Under> = under.iter().map(|&(gi, rgb, alpha)| Under { pixels: &images[gi].1, width: sides[gi].0, samples: sides[gi].1, corner: groups[gi][0].0.corner,
                rgb, alpha: (alpha != 255).then_some(alpha as usize), opacity: groups[gi][0].0.opacity }).collect();
            overlay(&mut picture, pw, pc, &layers, false);
            let after = picture.iter().step_by(61).filter(|&&x| x == 0).count();
            if after * 2 < picture.len() / 61 || after <= before {
                overlay(&mut picture, pw, pc, &layers, true);
                under.clear();
            }
            drop(layers);
            images[m].1 = picture;
        }
    }
    records.push(!under.is_empty() as u8);
    if !under.is_empty() {
        put_var(&mut records, merged.unwrap_or(0));
        put_var(&mut records, under.len());
        for &(gi, rgb, alpha) in &under {
            put_var(&mut records, gi);
            let (top, left) = groups[gi][0].0.corner;
            for v in [top, left] { put_var(&mut records, ((v << 1) ^ (v >> 63)) as usize); }
            records.extend(rgb.iter().map(|&c| c as u8));
            records.extend_from_slice(&[alpha, groups[gi][0].0.opacity]);
        }
    }
    // the blocks in file order (the groups are), with what lies between them
    let count: usize = groups.iter().map(|g| g.len()).sum();
    put_var(&mut records, count);
    let start = groups[0][0].0.at;
    let (mut at, mut gaps) = (start, vec![]);
    for (gi, g) in groups.iter().enumerate() {
        let mut plane = 0usize;
        for (x, _, rec, lead) in g {
            put_var(&mut records, x.at - at);
            gaps.extend_from_slice(&b[at..x.at]);
            put_var(&mut records, gi);
            records.extend_from_slice(&[plane as u8, x.planes as u8]);
            put_var(&mut records, *lead);
            put_var(&mut records, rec.len());
            records.extend_from_slice(rec);
            plane += x.planes;
            let rows = x.rows * x.planes;
            at = x.at + 2 * rows + b[x.at..x.at + 2 * rows].chunks_exact(2).map(|c| u16::from_be_bytes([c[0], c[1]]) as usize).sum::<usize>();
        }
    }
    Some(Split { start, len: at - start, records, gaps, images, planes: groups.iter().flatten().map(|x| x.0.planes).sum() })
}

/// Append the span [`split`] took apart.
pub fn rebuild(records: &[u8], gaps: &[u8], images: &[Vec<u8>], out: &mut Vec<u8>) -> Result<(), String> {
    let bad = || "psd: bad record (corrupt file)".to_string();
    let mut p = 1usize;
    if records.first() != Some(&VERSION) || get_var(records, &mut p) != Some(images.len()) { return Err(bad()); }
    // per group: bytes a row, rows, planes, bytes a sample, planes side by side
    let mut groups = Vec::with_capacity(images.len());
    for im in images {
        let (width, rows) = (get_var(records, &mut p).ok_or_else(bad)?, get_var(records, &mut p).ok_or_else(bad)?);
        let &[n, sample, side] = records.get(p..p + 3).ok_or_else(bad)? else { return Err(bad()) };
        p += 3;
        let (n, sample, side) = (n as usize, sample as usize, side as usize);
        if width == 0 || rows == 0 || n == 0 || !matches!(sample, 1 | 2) || (side != 1 && (side != n || width % sample != 0))
            || rows.checked_mul(width).and_then(|x| x.checked_mul(n)) != Some(im.len()) { return Err(bad()); }
        groups.push((width, rows, n, sample, side));
    }
    // the merged picture, where it is kept as its difference to a layer
    let mut restored: Option<(usize, Vec<u8>)> = None;
    let flag = *records.get(p).ok_or_else(bad)?;
    p += 1;
    if flag == 1 {
        let (m, n) = (get_var(records, &mut p).ok_or_else(bad)?, get_var(records, &mut p).ok_or_else(bad)?);
        let &(pw, _, pc, ps, pside) = groups.get(m).ok_or_else(bad)?;
        if ps != 1 || pside != pc || pc < 3 || n > groups.len() { return Err(bad()); }
        let mut layers = Vec::with_capacity(n);
        for _ in 0..n {
            let gi = get_var(records, &mut p).ok_or_else(bad)?;
            let mut corner = [0i64; 2];
            for c in &mut corner {
                let v = get_var(records, &mut p).ok_or_else(bad)? as u64;
                *c = (v >> 1) as i64 ^ -((v & 1) as i64);
            }
            let &[r, g, b, alpha, opacity] = records.get(p..p + 5).ok_or_else(bad)? else { return Err(bad()) };
            p += 5;
            let &(lw, _, lc, ls, lside) = groups.get(gi).ok_or_else(bad)?;
            if gi == m || ls != 1 || lside != lc || [r, g, b].iter().any(|&c| c as usize >= lc) || (alpha != 255 && alpha as usize >= lc) || corner.iter().any(|c| c.unsigned_abs() > 1 << 30) { return Err(bad()); }
            layers.push(Under { pixels: &images[gi], width: lw, samples: lc, corner: (corner[0], corner[1]), rgb: [r as usize, g as usize, b as usize], alpha: (alpha != 255).then_some(alpha as usize), opacity });
        }
        let mut picture = images[m].clone();
        overlay(&mut picture, pw, pc, &layers, true);
        restored = Some((m, picture));
    } else if flag != 0 {
        return Err(bad());
    }
    let count = get_var(records, &mut p).ok_or_else(bad)?;
    let mut gp = 0usize;
    let (mut plane_bytes, mut packed) = (vec![], vec![]);
    for _ in 0..count {
        let gap = get_var(records, &mut p).ok_or_else(bad)?;
        out.extend_from_slice(gaps.get(gp..gp.checked_add(gap).ok_or_else(bad)?).ok_or_else(bad)?);
        gp += gap;
        let gi = get_var(records, &mut p).ok_or_else(bad)?;
        let &(width, rows, n, sample, side) = groups.get(gi).ok_or_else(bad)?;
        let &[first, planes] = records.get(p..p + 2).ok_or_else(bad)? else { return Err(bad()) };
        p += 2;
        let (first, planes) = (first as usize, planes as usize);
        if planes == 0 || first + planes > n { return Err(bad()); }
        let lead = get_var(records, &mut p).ok_or_else(bad)?;
        let len = get_var(records, &mut p).ok_or_else(bad)?;
        let rec = records.get(p..p.checked_add(len).ok_or_else(bad)?).ok_or_else(bad)?;
        p += len;
        // the planes out of the pixels
        let im: &[u8] = match &restored {
            Some((m, picture)) if *m == gi => picture,
            _ => &images[gi],
        };
        plane_bytes.clear();
        for k in first..first + planes {
            if side == 1 {
                plane_bytes.extend_from_slice(&im[k * rows * width..(k + 1) * rows * width]);
            } else {
                for px in im.chunks_exact(n * sample) { plane_bytes.extend_from_slice(&px[k * sample..(k + 1) * sample]); }
            }
        }
        packed.clear();
        packbits::encode(&plane_bytes, rec, width, Style::Adobe(lead), &mut packed)?;
        let sizes = row_sizes(&packed, width, rows * planes).ok_or_else(bad)?;
        for s in sizes {
            if s > u16::MAX as usize { return Err(bad()); }
            out.extend_from_slice(&(s as u16).to_be_bytes());
        }
        out.extend_from_slice(&packed);
    }
    if p != records.len() || gp != gaps.len() { return Err(bad()); }
    Ok(())
}

/// The extent of the document at `b[i]`: (end, bytes of pixels).
pub fn extent(b: &[u8], i: usize) -> Result<(usize, usize), ExtentError> {
    match blocks(&b[i..]) {
        Ok((all, end, _)) if !all.is_empty() => Ok((i + end, all.iter().map(|x| x.rows * x.width * x.planes).sum())),
        Ok(_) | Err(false) => Err(ExtentError::Invalid),
        Err(true) => Err(ExtentError::Truncated),
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    fn rng(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }

    /// Planes of `rows` rows of `width` bytes packed as Photoshop packs (or, `other`, not quite).
    fn packed(planes: &[Vec<u8>], width: usize, other: bool, lead: usize) -> Vec<u8> {
        let (mut table, mut data) = (vec![], vec![]);
        for (k, r) in planes.iter().flat_map(|p| p.chunks(width)).enumerate() {
            let at = data.len();
            if other && k % 5 == 2 { data.extend(packbits::tests::plain(r, 0, 3)); } else { packbits::pack(r, Style::Adobe(lead), &mut data); }
            table.extend_from_slice(&((data.len() - at) as u16).to_be_bytes());
        }
        table.extend(data);
        table
    }

    /// A document of `w` by `h` pixels, three channels of `depth` bits, with `layers` layers.
    pub fn document(w: usize, h: usize, depth: usize, layers: usize, other: bool, seed: u64) -> (Vec<u8>, Vec<Vec<Vec<u8>>>) {
        let mut s = seed;
        let width = w * depth / 8;
        let mut plane = |rows: usize, width: usize| -> Vec<u8> {
            let mut v = Vec::with_capacity(rows * width);
            let mut x = 0u8;
            while v.len() < rows * width {
                match rng(&mut s) % 5 {
                    0 => v.extend(std::iter::repeat_n(x, 2 + (rng(&mut s) % 300) as usize)),
                    _ => { x = x.wrapping_add((rng(&mut s) % 5) as u8); v.push(x); }
                }
            }
            v.truncate(rows * width);
            v
        };
        let mut out = b"8BPS\0\x01\0\0\0\0\0\0".to_vec();
        out.extend_from_slice(&3u16.to_be_bytes());
        out.extend_from_slice(&(h as u32).to_be_bytes());
        out.extend_from_slice(&(w as u32).to_be_bytes());
        out.extend_from_slice(&(depth as u16).to_be_bytes());
        out.extend_from_slice(&[0, 3, 0, 0, 0, 0, 0, 0, 0, 6, 1, 2, 3, 4, 5, 6]);
        let mut all = vec![];
        let (mut recs, mut data) = (vec![], vec![]);
        for l in 0..layers {
            let (lh, lw) = (h - l, w - 2 * l);
            // a layer that starts off the edge of a tile
            let left = 45 * l as u32;
            for v in [0u32, left, lh as u32, left + lw as u32] { recs.extend_from_slice(&v.to_be_bytes()); }
            let ids: &[i16] = if l % 2 == 0 { &[-1, 0, 1, 2] } else { &[0, 1, 2] };
            recs.extend_from_slice(&(ids.len() as u16).to_be_bytes());
            let mut planes = vec![];
            for &id in ids {
                let pl = plane(lh, lw * depth / 8);
                let d = packed(std::slice::from_ref(&pl), lw * depth / 8, other, (1024 - left as usize * depth / 8 % 1024) % 1024);
                recs.extend_from_slice(&id.to_be_bytes());
                recs.extend_from_slice(&(d.len() as u32 + 2).to_be_bytes());
                data.extend_from_slice(&[0, 1]);
                data.extend(d);
                planes.push(pl);
            }
            recs.extend_from_slice(b"8BIMnorm\xFF\0\0\0\0\0\0\x04abcd");
            all.push(planes);
        }
        let info_len = 2 + recs.len() + data.len();
        out.extend_from_slice(&((info_len + 4 + 8) as u32).to_be_bytes());
        out.extend_from_slice(&(info_len as u32).to_be_bytes());
        out.extend_from_slice(&(layers as u16).to_be_bytes());
        out.extend(recs);
        out.extend(data);
        out.extend_from_slice(b"\0\0\0\x04mask");
        let merged: Vec<Vec<u8>> = (0..3).map(|_| plane(h, width)).collect();
        out.extend_from_slice(&[0, 1]);
        out.extend(packed(&merged, width, other, 0));
        all.push(merged);
        (out, all)
    }

    fn back(sp: &Split) -> Vec<u8> {
        let images: Vec<Vec<u8>> = sp.images.iter().map(|im| im.1.clone()).collect();
        let mut out = vec![];
        rebuild(&sp.records, &sp.gaps, &images, &mut out).unwrap();
        out
    }

    #[test]
    fn documents_come_back() {
        for (w, h, depth, layers, other) in [(90usize, 40usize, 8usize, 2usize, false), (300, 25, 8, 0, false), (64, 30, 16, 3, false), (200, 33, 8, 3, true), (40, 20, 1, 1, false), (2100, 12, 8, 2, false)] {
            let what = format!("{w}x{h}, {depth} bits, {layers} layers{}", if other { ", another packer" } else { "" });
            let (mut f, planes) = document(w, h, depth, layers, other, 5);
            let end = f.len();
            f.extend_from_slice(b"after");
            let sp = split(&f, 1 << 30).unwrap_or_else(|| panic!("{what}: not split"));
            assert_eq!(sp.images.len(), layers + 1, "{what}");
            assert!(back(&sp) == f[sp.start..sp.start + sp.len] && sp.start + sp.len == end, "{what}: rebuild");
            assert_eq!(extent(&f, 0).map(|e| e.0), Ok(end), "{what}");
            assert_eq!(extent(&f[..end - 3], 0), Err(ExtentError::Truncated), "{what}");
            // the pixels: a layer's planes side by side
            let (im, pl) = (&sp.images[layers].1, &planes[layers]);
            let sample = depth.div_ceil(8);
            assert!((0..im.len() / 3 / sample).all(|k| (0..3).all(|c| im[(3 * k + c) * sample..(3 * k + c + 1) * sample] == pl[c][k * sample..(k + 1) * sample])), "{what}: pixels");
            if !other { assert!(sp.records.len() < 40 + 12 * sp.planes, "{what}: record of {} bytes", sp.records.len()); }
        }
        assert!(split(b"8BPS\0\x01 but not one, not at all", 1 << 30).is_none());
        // a picture less what its layers make, and plus it again; a layer may lie partly outside
        let picture: Vec<u8> = (0..50 * 30 * 3).map(|i| (i * 7 % 251) as u8).collect();
        let layer: Vec<u8> = (0..40 * 40 * 4).map(|i| (i * 13 % 241) as u8).collect();
        for corner in [(0i64, 0i64), (-7, 20), (5, -12), (100, 100)] {
            let under = [Under { pixels: &layer, width: 40, samples: 4, corner, rgb: [1, 2, 3], alpha: Some(0), opacity: 200 }, Under { pixels: &layer, width: 40, samples: 4, corner: (3, 3), rgb: [1, 2, 3], alpha: None, opacity: 255 }];
            let mut p = picture.clone();
            overlay(&mut p, 50, 3, &under, false);
            assert!(p != picture);
            overlay(&mut p, 50, 3, &under, true);
            assert!(p == picture);
        }
        // damaged files and records
        let (f, _) = document(80, 30, 8, 2, true, 9);
        let sp = split(&f, 1 << 30).unwrap();
        let images: Vec<Vec<u8>> = sp.images.iter().map(|im| im.1.clone()).collect();
        let mut s = 3u64;
        for _ in 0..1500 {
            let mut bad = f.clone();
            for _ in 0..1 + rng(&mut s) % 3 {
                let i = (rng(&mut s) % bad.len() as u64) as usize;
                if rng(&mut s) % 6 == 0 { bad.truncate(i.max(8)); } else { bad[i] ^= 1 << (rng(&mut s) % 8); }
            }
            let _ = extent(&bad, 0);
            if let Some(sp) = split(&bad, 1 << 24) { assert!(back(&sp) == bad[sp.start..sp.start + sp.len]); }
            let (mut rec, mut gaps, mut images) = (sp.records.clone(), sp.gaps.clone(), images.clone());
            let which = match rng(&mut s) % 4 { 0 => &mut gaps, 1 => &mut images[(rng(&mut s) % 3) as usize], _ => &mut rec };
            if which.is_empty() { continue; }
            let i = (rng(&mut s) % which.len() as u64) as usize;
            if rng(&mut s) % 4 == 0 { which.truncate(i); } else { which[i] ^= 1 << (rng(&mut s) % 8); }
            let mut out = vec![];
            let _ = rebuild(&rec, &gaps, &images, &mut out);
            assert!(out.len() < 8 * f.len() + (1 << 16));
        }
    }
}
