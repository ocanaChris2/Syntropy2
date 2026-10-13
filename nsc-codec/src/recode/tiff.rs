//! TIFF: pictures out of their compressed strips.
//!
//! A TIFF is directories of tags — one per picture: its size, the layout of its samples, how it
//! is compressed — that point at strips or tiles of pixel data, each compressed by itself: with
//! LZW, PackBits, zlib, or not at all. [`split`] reads the directories and decodes the strips of
//! every picture it understands to pixels, with the horizontal differences undone that many
//! writers take before compressing, so each picture is one piece for the image model. Beside the
//! pixels it keeps a record: where each strip stood, and what its encoder did beyond the usual
//! (`transform::lzw`, `transform::packbits`, the deflate record); and the bytes between the
//! strips. [`rebuild`] writes the file from its first strip to its last again from these.
//!
//! The directories stay where they are, as bytes: nothing here needs them to decode. A picture
//! one of whose strips does not come back exactly stays as bytes too.
use crate::detect::{image_params, imgfmt};
use crate::transform::deflate::inflate::ExtentError;
use crate::transform::deflate::{inflate, side, writer};
use crate::transform::{lzw, packbits};

const VERSION: u8 = 1;
/// How a picture's strips are compressed.
const NONE: u8 = 0;
const LZW: u8 = 1;
const PACKBITS: u8 = 2;
const ZLIB: u8 = 3;
/// A picture's flags: its samples are differences to the one before in the row; they are of
/// two bytes; and those the high byte first.
const DIFF: u8 = 1;
const WIDE: u8 = 2;
const BIG: u8 = 4;
/// Samples of 12 bits, two packed into three bytes (a camera's raw data): the pixels are
/// given as samples of two bytes, low byte first.
const PACK12: u8 = 8;

/// Bounds of what is read: directories of a file, entries of a directory, strips of a picture.
const MAX_DIRS: usize = 256;
const MAX_ENTRIES: usize = 4096;
const MAX_CHUNKS: usize = 1 << 20;
/// Pictures of fewer bytes are not worth a record.
const MIN_PIXELS: usize = 1024;

fn put_var(o: &mut Vec<u8>, v: usize) { crate::nsc3::format::put_var(o, v as u64) }
fn get_var(b: &[u8], p: &mut usize) -> Option<usize> { super::h264::get_var(b, p) }

/// Adler-32 of `d`, as zlib ends a stream with.
fn adler32(d: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for chunk in d.chunks(5552) {
        for &x in chunk {
            a += x as u32;
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    b << 16 | a
}

/// A picture as its directory describes it.
struct Picture {
    method: u8,
    flags: u8,
    /// Samples between a sample and the one it is a difference to; bytes of a row.
    dist: usize,
    row: usize,
    /// The strips or tiles in the picture's order: (offset, bytes, bytes they decode to).
    chunks: Vec<(usize, usize, usize)>,
    /// The image model's parameters for the pixels.
    params: Vec<u8>,
}

impl Picture {
    fn pixels(&self) -> usize { self.chunks.iter().map(|c| c.2).sum() }
}

/// The directories of a file.
struct Dirs {
    pictures: Vec<Picture>,
    /// Where the last byte the directories point at is.
    end: usize,
    /// Something they point at lies past the end of the buffer.
    cut: bool,
}

struct File<'a> {
    b: &'a [u8],
    le: bool,
}

impl File<'_> {
    fn u16(&self, o: usize) -> Option<usize> {
        let s = self.b.get(o..o.checked_add(2)?)?;
        Some(if self.le { u16::from_le_bytes([s[0], s[1]]) } else { u16::from_be_bytes([s[0], s[1]]) } as usize)
    }

    fn u32(&self, o: usize) -> Option<usize> {
        let s: [u8; 4] = self.b.get(o..o.checked_add(4)?)?.try_into().ok()?;
        Some(if self.le { u32::from_le_bytes(s) } else { u32::from_be_bytes(s) } as usize)
    }
}

/// Bytes of a value of tag type `ty`.
fn type_size(ty: usize) -> Option<usize> {
    Some(match ty {
        1 | 2 | 6 | 7 => 1,
        3 | 8 => 2,
        4 | 9 | 11 | 13 => 4,
        5 | 10 | 12 => 8,
        _ => return None,
    })
}

/// The tags of a directory that say how its picture is stored.
#[derive(Default)]
struct Tags {
    width: usize,
    height: usize,
    bits: Vec<usize>,
    compression: usize,
    photometric: usize,
    samples: usize,
    rows_per_strip: usize,
    planar: usize,
    predictor: usize,
    strips: (Vec<usize>, Vec<usize>),
    tile: (usize, usize),
    tiles: (Vec<usize>, Vec<usize>),
    /// A value of these lies past the end of the buffer.
    cut: bool,
}

impl Tags {
    /// The picture, if it is stored in a way taken apart here; and where its data ends.
    fn picture(&self, le: bool) -> (Option<Picture>, usize) {
        let tiled = !self.tiles.0.is_empty();
        let (offsets, counts) = if tiled { &self.tiles } else { &self.strips };
        let end = offsets.iter().zip(counts).map(|(&o, &n)| o.saturating_add(n)).max().unwrap_or(0);
        let method = match self.compression {
            1 => NONE,
            5 => LZW,
            8 | 32946 => ZLIB,
            32773 => PACKBITS,
            _ => return (None, end),
        };
        let bits = self.bits.first().copied().unwrap_or(1);
        let (w, h, spp) = (self.width, self.height, self.samples.max(1));
        // subsampled colour has strips of another size
        if w == 0 || h == 0 || w > 1 << 24 || h > 1 << 24 || spp > 64 || !(1..=64).contains(&bits) || self.bits.iter().any(|&x| x != bits)
            || self.photometric == 6 || offsets.len() != counts.len() || offsets.is_empty() || offsets.len() > MAX_CHUNKS {
            return (None, end);
        }
        // the samples of a pixel together, or each plane by itself
        let (planes, per) = if self.planar == 2 { (spp, 1) } else { (1, spp) };
        // strips of rows of the picture's width, or tiles of their own
        let (cw, ch, across, down) = if tiled {
            let (tw, th) = self.tile;
            if tw == 0 || th == 0 || tw > 1 << 24 || th > 1 << 24 { return (None, end); }
            (tw, th, w.div_ceil(tw), h.div_ceil(th))
        } else {
            let rows = self.rows_per_strip.clamp(1, h);
            (w, rows, 1, h.div_ceil(rows))
        };
        let row = (cw as u64 * per as u64 * bits as u64).div_ceil(8);
        if row > 1 << 24 || across as u64 * down as u64 * planes as u64 != offsets.len() as u64 { return (None, end); }
        let row = row as usize;
        let mut chunks = Vec::with_capacity(offsets.len());
        for (k, (&o, &n)) in offsets.iter().zip(counts).enumerate() {
            // a tile is whole also where the picture ends inside it; the last strip is not
            let rows = if tiled { ch } else { ch.min(h - (k % down) * ch) };
            let Some(want) = rows.checked_mul(row).filter(|&x| x <= u32::MAX as usize) else { return (None, end) };
            if n == 0 { return (None, end); }
            chunks.push((o, n, want));
        }
        let wide = bits == 16;
        let params = match (bits, per) {
            (8, 1 | 2) => image_params(imgfmt::GRAY8, cw as u32, row as u32, per as u8, false),
            (8, 3) => image_params(imgfmt::RGB24, cw as u32, row as u32, 3, false),
            (8, 4) => image_params(imgfmt::RGBA32, cw as u32, row as u32, 4, false),
            (16, 1..=4) => image_params(if le { imgfmt::GRAY16LE } else if per == 1 { imgfmt::GRAY16BE } else { imgfmt::RGB48BE }, cw as u32, row as u32, per as u8, false),
            (12, 1..=4) if cw * per % 2 == 0 => image_params(imgfmt::GRAY16LE, cw as u32, (2 * cw * per) as u32, per as u8, false),
            // samples of other sizes: bytes for the byte model (no parameters)
            _ => vec![],
        };
        let diff = self.predictor == 2 && (bits == 8 || wide);
        let flags = diff as u8 * DIFF | wide as u8 * WIDE | (wide && !le) as u8 * BIG | (bits == 12 && !params.is_empty()) as u8 * PACK12;
        (Some(Picture { method, flags, dist: per, row, chunks, params }), end)
    }
}

/// The directories of the TIFF at the start of `b`; None if it is none.
fn directories(b: &[u8]) -> Option<Dirs> {
    let le = match b.get(..4)? {
        b"II*\0" => true,
        b"MM\0*" => false,
        _ => return None,
    };
    let f = File { b, le };
    let mut dirs = Dirs { pictures: vec![], end: 8, cut: false };
    // directories still to read: those the file chains, and those tags point at
    let mut todo = vec![f.u32(4)?];
    let (mut k, mut read) = (0usize, 0usize);
    while k < todo.len() && k < MAX_DIRS && read <= 4 * MAX_CHUNKS {
        let at = todo[k];
        k += 1;
        if at < 8 || todo[..k - 1].contains(&at) {
            if k == 1 { return None; }
            continue;
        }
        let Some(n) = f.u16(at) else {
            (dirs.cut, dirs.end) = (true, dirs.end.max(at.saturating_add(2)));
            continue;
        };
        let after = at + 2 + 12 * n;
        dirs.end = dirs.end.max(after.saturating_add(4));
        if n == 0 || n > MAX_ENTRIES {
            if k == 1 { return None; }
            continue;
        }
        if after + 4 > b.len() {
            dirs.cut = true;
            continue;
        }
        let mut t = Tags { compression: 1, samples: 1, rows_per_strip: usize::MAX, planar: 1, predictor: 1, ..Tags::default() };
        let mut valid = true;
        for e in (at + 2..after).step_by(12) {
            let (tag, ty, count) = (f.u16(e)?, f.u16(e + 2)?, f.u32(e + 4)?);
            let Some(size) = type_size(ty) else {
                valid = false;
                break;
            };
            if !matches!(tag, 256..=259 | 262 | 273 | 277..=279 | 284 | 317 | 322..=325 | 330) { continue; }
            // numbers of one, two or four bytes, in the entry itself when they fit
            if !matches!(ty, 1 | 3 | 4 | 13) || count == 0 || count > MAX_CHUNKS { continue; }
            let from = if size * count <= 4 { e + 8 } else { f.u32(e + 8)? };
            read += count;
            dirs.end = dirs.end.max(from.saturating_add(size * count));
            if from.saturating_add(size * count) > b.len() {
                (t.cut, dirs.cut) = (true, true);
                continue;
            }
            let get = |i: usize| match size {
                1 => b[from + i] as usize,
                2 => f.u16(from + 2 * i).unwrap_or(0),
                _ => f.u32(from + 4 * i).unwrap_or(0),
            };
            let all = || (0..count).map(get).collect::<Vec<usize>>();
            match tag {
                256 => t.width = get(0),
                257 => t.height = get(0),
                258 => t.bits = all(),
                259 => t.compression = get(0),
                262 => t.photometric = get(0),
                273 => t.strips.0 = all(),
                277 => t.samples = get(0),
                278 => t.rows_per_strip = get(0),
                279 => t.strips.1 = all(),
                284 => t.planar = get(0),
                317 => t.predictor = get(0),
                322 => t.tile.0 = get(0),
                323 => t.tile.1 = get(0),
                324 => t.tiles.0 = all(),
                325 => t.tiles.1 = all(),
                _ => if count <= 16 { todo.extend(all()) },
            }
        }
        if !valid {
            if k == 1 { return None; }
            continue;
        }
        todo.push(f.u32(after)?);
        if t.cut { continue; }
        let (picture, end) = t.picture(le);
        dirs.end = dirs.end.max(end);
        match picture {
            Some(_) if end > b.len() => dirs.cut = true,
            Some(p) => dirs.pictures.push(p),
            None => {}
        }
    }
    Some(dirs)
}

/// The samples of rows of `p.row` bytes from their differences to the sample `p.dist` before
/// (`back`: the differences from the samples), in place.
fn differences(buf: &mut [u8], p: &Picture, back: bool) {
    let d = p.dist;
    for r in buf.chunks_mut(p.row) {
        if p.flags & WIDE == 0 {
            if back {
                for x in (d..r.len()).rev() { r[x] = r[x].wrapping_sub(r[x - d]); }
            } else {
                for x in d..r.len() { r[x] = r[x].wrapping_add(r[x - d]); }
            }
            continue;
        }
        let big = p.flags & BIG != 0;
        let get = |r: &[u8], k: usize| if big { u16::from_be_bytes([r[2 * k], r[2 * k + 1]]) } else { u16::from_le_bytes([r[2 * k], r[2 * k + 1]]) };
        let put = |r: &mut [u8], k: usize, v: u16| r[2 * k..2 * k + 2].copy_from_slice(&if big { v.to_be_bytes() } else { v.to_le_bytes() });
        let n = r.len() / 2;
        if back {
            for k in (d..n).rev() {
                let v = get(r, k).wrapping_sub(get(r, k - d));
                put(r, k, v);
            }
        } else {
            for k in d..n {
                let v = get(r, k).wrapping_add(get(r, k - d));
                put(r, k, v);
            }
        }
    }
}

/// A zlib stream that holds `want` bytes: the bytes, and what [`rebuild`] needs to write it again.
fn unzlib(data: &[u8], want: usize) -> Option<(Vec<u8>, Vec<u8>)> {
    let wbits = crate::transform::deflate::scan::zlib_header(data, 0)?;
    let (inner, parse) = inflate::inflate(&data[2..], want).ok()?;
    if inner.len() != want { return None; }
    let rec = crate::transform::deflate::scan::record(&inner, &parse, wbits, true).ok()?;
    let back = side::decode(&inner, &rec).ok()?;
    if writer::write(&back, &inner).ok()? != data[2..2 + parse.consumed] { return None; }
    let mut pay = data[..2].to_vec();
    put_var(&mut pay, rec.len());
    pay.extend_from_slice(&rec);
    // after the stream: its checksum as a rule, which the bytes give
    let tail = &data[2 + parse.consumed..];
    if tail == adler32(&inner).to_be_bytes() {
        put_var(&mut pay, 0);
    } else {
        put_var(&mut pay, tail.len() + 1);
        pay.extend_from_slice(tail);
    }
    Some((inner, pay))
}

/// The pixels of `p` and each strip's part of the record, if every strip decodes and can be
/// written again.
fn decode(b: &[u8], p: &Picture) -> Option<(Vec<u8>, Vec<Vec<u8>>)> {
    let mut pixels = Vec::with_capacity(p.pixels());
    let mut parts = Vec::with_capacity(p.chunks.len());
    for &(o, n, want) in &p.chunks {
        let data = b.get(o..o.checked_add(n)?)?;
        let at = pixels.len();
        let mut pay = vec![];
        let tail = |pay: &mut Vec<u8>, record: &[u8], consumed: usize| {
            put_var(pay, record.len());
            pay.extend_from_slice(record);
            put_var(pay, n - consumed);
            pay.extend_from_slice(&data[consumed..]);
        };
        match p.method {
            NONE => {
                if n != want { return None; }
                pixels.extend_from_slice(data);
            }
            LZW => {
                let d = lzw::decode(data, &lzw::Params::TIFF, want).filter(|d| d.bytes.len() == want)?;
                tail(&mut pay, &d.record, d.consumed);
                pixels.extend_from_slice(&d.bytes);
            }
            PACKBITS => {
                let d = packbits::decode(data, p.row, want, packbits::Style::Tiff)?;
                tail(&mut pay, &d.record, d.consumed);
                pixels.extend_from_slice(&d.bytes);
            }
            _ => {
                let (inner, z) = unzlib(data, want)?;
                pay = z;
                pixels.extend_from_slice(&inner);
            }
        }
        if p.flags & DIFF != 0 { differences(&mut pixels[at..], p, false); }
        if p.flags & PACK12 != 0 {
            let packed = pixels.split_off(at);
            for t in packed.chunks_exact(3) {
                pixels.extend_from_slice(&[t[0] << 4 | t[1] >> 4, t[0] >> 4, t[2], t[1] & 15]);
            }
        }
        parts.push(pay);
    }
    Some((pixels, parts))
}

/// A TIFF taken apart.
pub struct Split {
    /// The span of the file taken apart: from its first strip to its last.
    pub start: usize,
    pub len: usize,
    pub records: Vec<u8>,
    /// The bytes between the strips.
    pub gaps: Vec<u8>,
    /// The pictures: the image model's parameters, the pixels.
    pub images: Vec<(Vec<u8>, Vec<u8>)>,
    /// Strips and tiles decoded.
    pub chunks: usize,
    /// The span is one picture's pixels as they are: not compressed, strip after strip.
    pub plain: bool,
}

/// Take the TIFF at the start of `b` apart, for at most `budget` bytes of pixels: None unless
/// it is one with a picture whose strips decode.
pub fn split(b: &[u8], budget: usize) -> Option<Split> {
    let dirs = directories(b)?;
    // the pictures taken: where their strips lie, their pixels, their strips' records
    let mut taken: Vec<(Picture, Vec<u8>, Vec<Vec<u8>>)> = vec![];
    let mut left = budget;
    for p in dirs.pictures {
        let total = p.pixels();
        if total < MIN_PIXELS || total > left { continue; }
        // strips in file order, apart from each other and from the pictures before
        let (lo, hi) = (p.chunks[0].0, p.chunks.last().map_or(0, |c| c.0 + c.1));
        if !p.chunks.windows(2).all(|w| w[0].0 + w[0].1 <= w[1].0) { continue; }
        if taken.iter().any(|(q, ..)| lo < q.chunks.last().map_or(0, |c| c.0 + c.1) && q.chunks[0].0 < hi) { continue; }
        let Some((pixels, parts)) = decode(b, &p) else { continue };
        left -= total;
        taken.push((p, pixels, parts));
    }
    if taken.is_empty() { return None; }
    let mut records = vec![VERSION];
    put_var(&mut records, taken.len());
    for (p, ..) in &taken {
        records.extend_from_slice(&[p.method, p.flags]);
        put_var(&mut records, p.dist);
        put_var(&mut records, p.row);
    }
    // the strips in file order, with what lies between them
    let mut order: Vec<usize> = (0..taken.len()).collect();
    order.sort_by_key(|&i| taken[i].0.chunks[0].0);
    let chunks: usize = taken.iter().map(|t| t.0.chunks.len()).sum();
    put_var(&mut records, chunks);
    let start = taken[order[0]].0.chunks[0].0;
    let (mut at, mut gaps) = (start, vec![]);
    for &i in &order {
        let (p, _, parts) = &taken[i];
        for (&(o, n, want), part) in p.chunks.iter().zip(parts) {
            put_var(&mut records, o - at);
            gaps.extend_from_slice(&b[at..o]);
            put_var(&mut records, i);
            put_var(&mut records, if p.flags & PACK12 != 0 { want / 3 * 4 } else { want });
            records.extend_from_slice(part);
            at = o + n;
        }
    }
    let plain = taken.len() == 1 && taken[0].0.method == NONE && taken[0].0.flags & (DIFF | PACK12) == 0 && gaps.is_empty();
    let images = taken.into_iter().map(|(p, pixels, _)| (p.params, pixels)).collect();
    Some(Split { start, len: at - start, records, gaps, images, chunks, plain })
}

/// Append the span [`split`] took apart, from its record, the bytes between its strips and its
/// pictures' pixels.
pub fn rebuild(records: &[u8], gaps: &[u8], images: &[Vec<u8>], out: &mut Vec<u8>) -> Result<(), String> {
    let bad = || "tiff: bad record (corrupt file)".to_string();
    let mut p = 1usize;
    if records.first() != Some(&VERSION) || get_var(records, &mut p) != Some(images.len()) { return Err(bad()); }
    let mut pictures = Vec::with_capacity(images.len());
    for _ in images {
        let (&method, &flags) = (records.get(p).ok_or_else(bad)?, records.get(p + 1).ok_or_else(bad)?);
        p += 2;
        let (dist, row) = (get_var(records, &mut p).ok_or_else(bad)?, get_var(records, &mut p).ok_or_else(bad)?);
        if method > ZLIB || flags > 15 || dist == 0 || row == 0 || dist > row { return Err(bad()); }
        pictures.push(Picture { method, flags, dist, row, chunks: vec![], params: vec![] });
    }
    let chunks = get_var(records, &mut p).ok_or_else(bad)?;
    // how far the gaps and each picture's pixels are used
    let mut gp = 0usize;
    let mut at = vec![0usize; images.len()];
    let (mut scratch, mut packed) = (vec![], vec![]);
    // a counted run of bytes of the record
    fn counted<'a>(records: &'a [u8], p: &mut usize) -> Option<&'a [u8]> {
        let n = get_var(records, p)?;
        let s = records.get(*p..p.checked_add(n)?)?;
        *p += n;
        Some(s)
    }
    let take = |p: &mut usize| counted(records, p).ok_or_else(bad);
    for _ in 0..chunks {
        let gap = get_var(records, &mut p).ok_or_else(bad)?;
        out.extend_from_slice(gaps.get(gp..gp.checked_add(gap).ok_or_else(bad)?).ok_or_else(bad)?);
        gp += gap;
        let i = get_var(records, &mut p).ok_or_else(bad)?;
        let pic = pictures.get(i).ok_or_else(bad)?;
        let n = get_var(records, &mut p).ok_or_else(bad)?;
        let mut px = images[i].get(at[i]..at[i].checked_add(n).ok_or_else(bad)?).ok_or_else(bad)?;
        at[i] += n;
        if pic.flags & DIFF != 0 {
            // the differences the writer took before compressing
            scratch.clear();
            scratch.extend_from_slice(px);
            differences(&mut scratch, pic, true);
            px = &scratch;
        }
        if pic.flags & PACK12 != 0 {
            // two samples back into three bytes
            if n % 4 != 0 || pic.flags & DIFF != 0 { return Err(bad()); }
            packed.clear();
            for q in px.chunks_exact(4) {
                if q[1] > 15 || q[3] > 15 { return Err(bad()); }
                packed.extend_from_slice(&[q[1] << 4 | q[0] >> 4, q[0] << 4 | q[3], q[2]]);
            }
            px = &packed;
        }
        match pic.method {
            NONE => out.extend_from_slice(px),
            LZW => {
                lzw::encode(px, take(&mut p)?, &lzw::Params::TIFF, out)?;
                out.extend_from_slice(take(&mut p)?);
            }
            PACKBITS => {
                packbits::encode(px, take(&mut p)?, pic.row, packbits::Style::Tiff, out)?;
                out.extend_from_slice(take(&mut p)?);
            }
            _ => {
                out.extend_from_slice(records.get(p..p + 2).ok_or_else(bad)?);
                p += 2;
                let parse = side::decode(px, take(&mut p)?)?;
                out.extend_from_slice(&writer::write(&parse, px)?);
                match get_var(records, &mut p).ok_or_else(bad)? {
                    0 => out.extend_from_slice(&adler32(px).to_be_bytes()),
                    n => {
                        out.extend_from_slice(records.get(p..p.checked_add(n - 1).ok_or_else(bad)?).ok_or_else(bad)?);
                        p += n - 1;
                    }
                }
            }
        }
    }
    if p != records.len() || gp != gaps.len() || at.iter().zip(images).any(|(&a, im)| a != im.len()) { return Err(bad()); }
    Ok(())
}

/// The extent of the TIFF at `b[i]`, by its directories: (end, bytes of pixels). `Truncated`
/// when they point past the end of `b`.
pub fn extent(b: &[u8], i: usize) -> Result<(usize, usize), ExtentError> {
    let b = &b[i..];
    if b.len() < 8 {
        return Err(if b"II*\0".starts_with(&b[..b.len().min(4)]) || b"MM\0*".starts_with(&b[..b.len().min(4)]) { ExtentError::Truncated } else { ExtentError::Invalid });
    }
    let dirs = directories(b).ok_or(ExtentError::Invalid)?;
    if dirs.cut || dirs.end > b.len() { return Err(ExtentError::Truncated); }
    let work: usize = dirs.pictures.iter().map(|p| p.pixels()).filter(|&n| n >= MIN_PIXELS).sum();
    if work == 0 { return Err(ExtentError::Invalid); }
    Ok((i + dirs.end, work))
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::transform::lzw::tests::{compress, Style, USUAL};

    fn rng(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }

    /// Pixels of a picture: smooth, with flat stretches (so every compression has something to
    /// do), `row` bytes a row.
    pub fn pixels(rows: usize, row: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        let mut out = Vec::with_capacity(rows * row);
        for y in 0..rows {
            let mut v = (y * 3) as u8;
            let mut x = 0;
            while x < row {
                let run = if rng(&mut s) % 5 == 0 { 2 + (rng(&mut s) % 40) as usize } else { 1 };
                for _ in 0..run.min(row - x) { out.push(v); }
                x += run.min(row - x);
                v = v.wrapping_add((rng(&mut s) % 7) as u8).wrapping_sub(3);
            }
        }
        out
    }

    /// How a test picture is stored.
    #[derive(Clone, Copy)]
    pub struct Layout {
        pub width: usize,
        pub height: usize,
        pub samples: usize,
        pub bits: usize,
        pub compression: usize,
        pub predictor: usize,
        pub rows_per_strip: usize,
        /// Tiles of this size instead of strips (0: strips).
        pub tile: usize,
        pub planar: bool,
    }

    pub const RGB: Layout = Layout { width: 64, height: 40, samples: 3, bits: 8, compression: 5, predictor: 1, rows_per_strip: 7, tile: 0, planar: false };

    /// One strip's (or tile's) pixels compressed as `l` says.
    fn pack(px: &[u8], l: &Layout, row: usize, k: usize, le: bool) -> Vec<u8> {
        let mut px = px.to_vec();
        if l.predictor == 2 {
            let wide = l.bits == 16;
            let p = Picture { method: 0, flags: DIFF | wide as u8 * WIDE | (wide && !le) as u8 * BIG, dist: if l.planar { 1 } else { l.samples }, row, chunks: vec![], params: vec![] };
            differences(&mut px, &p, true);
        }
        match l.compression {
            1 => px,
            5 => {
                // encoders differ: one that clears early, one that ends a strip with two bytes more
                let mut out = compress(&px, &lzw::Params::TIFF, if k % 3 == 1 { Style { clear_at: 4094, ..USUAL } } else { USUAL });
                if k % 4 == 3 { out.extend_from_slice(&[0, 0]); }
                out
            }
            32773 => {
                let mut out = vec![];
                if k % 3 == 2 {
                    out = packbits::tests::plain(&px, row, 3);
                } else {
                    for r in px.chunks(row) { packbits::usual(r, &mut out); }
                }
                out
            }
            _ => {
                let mut out = vec![0x78, 0x9C];
                out.extend(crate::transform::deflate::deflate_like_zlib(&px, 6));
                out.extend_from_slice(&if k % 5 == 4 { [1, 2, 3, 4] } else { adler32(&px).to_be_bytes() });
                out
            }
        }
    }

    /// A TIFF of the pictures `layouts`, in either byte order, its directories before the
    /// pixel data (`first`) or after it; with the pixels of each picture.
    pub fn file(layouts: &[Layout], le: bool, first: bool, seed: u64) -> (Vec<u8>, Vec<Vec<u8>>) {
        let w16 = |v: usize| if le { (v as u16).to_le_bytes() } else { (v as u16).to_be_bytes() };
        let w32 = |v: usize| if le { (v as u32).to_le_bytes() } else { (v as u32).to_be_bytes() };
        // each picture's strips, and its pixels in the order of the strips
        let mut data: Vec<(Vec<Vec<u8>>, Vec<u8>)> = vec![];
        for (n, l) in layouts.iter().enumerate() {
            let (planes, per) = if l.planar { (l.samples, 1) } else { (1, l.samples) };
            let (cw, ch) = if l.tile > 0 { (l.tile, l.tile) } else { (l.width, l.rows_per_strip.min(l.height)) };
            let row = (cw * per * l.bits).div_ceil(8);
            let (across, down) = if l.tile > 0 { (l.width.div_ceil(cw), l.height.div_ceil(ch)) } else { (1, l.height.div_ceil(ch)) };
            let (mut strips, mut all) = (vec![], vec![]);
            for k in 0..across * down * planes {
                let rows = if l.tile > 0 { ch } else { ch.min(l.height - (k % down) * ch) };
                let px = pixels(rows, row, seed + 31 * n as u64 + k as u64);
                strips.push(pack(&px, l, row, k, le));
                all.extend_from_slice(&px);
            }
            data.push((strips, all));
        }
        // the directories: their size is known before the offsets in them are
        let entries = |l: &Layout| 10 + (l.predictor == 2) as usize + if l.tile > 0 { 1 } else { 0 };
        let dir_len = |l: &Layout, strips: usize| 2 + 12 * entries(l) + 4 + 2 * l.samples + if strips > 1 { 8 * strips } else { 0 };
        let dirs_len: usize = layouts.iter().zip(&data).map(|(l, d)| dir_len(l, d.0.len())).sum();
        let data_len: usize = data.iter().map(|d| d.0.iter().map(|s| s.len() + 3).sum::<usize>()).sum();
        let (mut dir_at, mut data_at) = if first { (8, 8 + dirs_len) } else { (8 + data_len, 8) };
        let mut out = if le { b"II*\0".to_vec() } else { b"MM\0*".to_vec() };
        out.extend_from_slice(&w32(dir_at));
        out.resize(8 + dirs_len + data_len, 0);
        for (n, (l, (strips, _))) in layouts.iter().zip(&data).enumerate() {
            // the strips, three bytes of something else between them
            let mut places = vec![];
            for s in strips {
                places.push((data_at, s.len()));
                out[data_at..data_at + s.len()].copy_from_slice(s);
                out[data_at + s.len()..data_at + s.len() + 3].copy_from_slice(b"gap");
                data_at += s.len() + 3;
            }
            let mut d = vec![];
            let after = dir_at + 2 + 12 * entries(l) + 4;
            let mut extra = vec![];
            let entry = |d: &mut Vec<u8>, extra: &mut Vec<u8>, tag: usize, ty: usize, vals: &[usize]| {
                d.extend_from_slice(&w16(tag));
                d.extend_from_slice(&w16(ty));
                d.extend_from_slice(&w32(vals.len()));
                let size = if ty == 3 { 2 } else { 4 };
                let mut v = vec![];
                for &x in vals { if ty == 3 { v.extend_from_slice(&w16(x)) } else { v.extend_from_slice(&w32(x)) } }
                if size * vals.len() <= 4 {
                    v.resize(4, 0);
                    d.extend_from_slice(&v);
                } else {
                    d.extend_from_slice(&w32(after + extra.len()));
                    extra.extend_from_slice(&v);
                }
            };
            d.extend_from_slice(&w16(entries(l)));
            entry(&mut d, &mut extra, 256, 3, &[l.width]);
            entry(&mut d, &mut extra, 257, 4, &[l.height]);
            entry(&mut d, &mut extra, 258, 3, &vec![l.bits; l.samples]);
            entry(&mut d, &mut extra, 259, 3, &[l.compression]);
            entry(&mut d, &mut extra, 262, 3, &[if l.samples >= 3 { 2 } else { 1 }]);
            let offsets: Vec<usize> = places.iter().map(|p| p.0).collect();
            let counts: Vec<usize> = places.iter().map(|p| p.1).collect();
            if l.tile == 0 { entry(&mut d, &mut extra, 273, 4, &offsets); }
            entry(&mut d, &mut extra, 277, 3, &[l.samples]);
            if l.tile == 0 {
                entry(&mut d, &mut extra, 278, 3, &[l.rows_per_strip]);
                entry(&mut d, &mut extra, 279, 4, &counts);
            }
            entry(&mut d, &mut extra, 284, 3, &[if l.planar { 2 } else { 1 }]);
            if l.predictor == 2 { entry(&mut d, &mut extra, 317, 3, &[2]); }
            if l.tile > 0 {
                entry(&mut d, &mut extra, 322, 3, &[l.tile]);
                entry(&mut d, &mut extra, 323, 3, &[l.tile]);
                entry(&mut d, &mut extra, 324, 4, &offsets);
                entry(&mut d, &mut extra, 325, 4, &counts);
            }
            let len = dir_len(l, strips.len());
            // a sample's bits fit the entry when there are two of them at most
            let pad = len - (d.len() + 4 + extra.len());
            d.extend_from_slice(&w32(if n + 1 < layouts.len() { dir_at + len } else { 0 }));
            d.extend_from_slice(&extra);
            d.resize(d.len() + pad, 0);
            out[dir_at..dir_at + len].copy_from_slice(&d);
            dir_at += len;
        }
        (out, data.into_iter().map(|d| d.1).collect())
    }

    fn back(sp: &Split) -> Vec<u8> {
        let images: Vec<Vec<u8>> = sp.images.iter().map(|im| im.1.clone()).collect();
        let mut out = vec![];
        rebuild(&sp.records, &sp.gaps, &images, &mut out).unwrap();
        out
    }

    #[test]
    fn pictures_come_back() {
        let gray16 = Layout { samples: 1, bits: 16, width: 50, ..RGB };
        let cases: Vec<(&str, Vec<Layout>)> = vec![
            ("lzw", vec![RGB]),
            ("lzw of differences", vec![Layout { predictor: 2, ..RGB }]),
            ("packbits", vec![Layout { compression: 32773, ..RGB }]),
            ("zlib", vec![Layout { compression: 8, rows_per_strip: 16, ..RGB }]),
            ("zlib of differences, 16 bits", vec![Layout { compression: 8, predictor: 2, ..gray16 }]),
            ("lzw of differences, 16 bits, three samples", vec![Layout { samples: 3, predictor: 2, ..gray16 }]),
            ("not compressed", vec![Layout { compression: 1, ..RGB }]),
            ("not compressed, one strip", vec![Layout { compression: 1, rows_per_strip: 1 << 15, ..RGB }]),
            ("tiles", vec![Layout { tile: 32, width: 70, height: 50, ..RGB }]),
            ("planes apart", vec![Layout { planar: true, predictor: 2, ..RGB }]),
            ("one strip", vec![Layout { rows_per_strip: 1 << 15, ..RGB }]),
            ("two pictures", vec![RGB, Layout { compression: 32773, samples: 1, width: 300, height: 9, ..RGB }]),
            ("four bits a sample", vec![Layout { samples: 1, bits: 4, width: 333, ..RGB }]),
            ("twelve bits a sample", vec![Layout { samples: 1, bits: 12, width: 100, compression: 1, ..RGB }]),
            ("twelve bits a sample, lzw", vec![Layout { samples: 1, bits: 12, width: 100, ..RGB }]),
        ];
        for (what, layouts) in &cases {
            for (le, first) in [(true, false), (false, true), (true, true), (false, false)] {
                let what = format!("{what}, {} byte order, directories {}", if le { "little" } else { "big" }, if first { "first" } else { "last" });
                let (mut f, px) = file(layouts, le, first, 7);
                let end = f.len();
                f.extend_from_slice(b"what follows the file");
                let sp = split(&f, 1 << 30).unwrap_or_else(|| panic!("{what}: not split"));
                assert_eq!(sp.images.len(), layouts.len(), "{what}: pictures");
                assert_eq!(sp.plain, layouts[0].compression == 1 && sp.chunks == 1, "{what}: plain");
                for (im, px) in sp.images.iter().zip(&px) { assert!(im.1 == *px || layouts[0].bits == 12 && im.1.len() == px.len() / 3 * 4, "{what}: pixels"); }
                assert!(back(&sp) == f[sp.start..sp.start + sp.len], "{what}: rebuild");
                assert!(sp.start >= 8 && sp.start + sp.len <= end, "{what}: span");
                assert_eq!(extent(&f, 0).map(|e| e.1), Ok(px.iter().map(|p| p.len()).sum()), "{what}: extent");
                if layouts[0].bits == 12 { assert_eq!(sp.images[0].0, image_params(imgfmt::GRAY16LE, 100, 200, 1, false), "{what}"); }
                assert!(extent(&f, 0).unwrap().0 <= end, "{what}: extent");
                // strips are their pixels and little more
                let strips: usize = sp.chunks;
                assert!(sp.gaps.len() == 3 * (strips - 1) || layouts.len() > 1, "{what}: {} bytes between {strips} strips", sp.gaps.len());
                if matches!(layouts[0].compression, 1 | 5) { assert!(sp.records.len() < 40 + 24 * strips, "{what}: record of {} bytes for {strips} strips", sp.records.len()); }
                // a file cut short is none, or is taken as far as it goes
                assert_eq!(extent(&f[..end - 9], 0), Err(ExtentError::Truncated), "{what}: cut short");
                if let Some(sp) = split(&f[..end - 9], 1 << 30) { assert!(back(&sp) == f[sp.start..sp.start + sp.len], "{what}: cut short"); }
            }
        }
        // the image model is told what the pixels are
        let (f, _) = file(&[RGB], true, false, 1);
        assert_eq!(split(&f, 1 << 30).unwrap().images[0].0, image_params(imgfmt::RGB24, 64, 192, 3, false));
        let (f, _) = file(&[Layout { tile: 32, samples: 1, bits: 16, compression: 8, width: 70, height: 50, ..RGB }], false, true, 1);
        assert_eq!(split(&f, 1 << 30).unwrap().images[0].0, image_params(imgfmt::GRAY16BE, 32, 64, 1, false));
        // what is not a TIFF, or holds too little, or more than may be taken
        assert!(split(b"II*\0 but not one", 1 << 30).is_none() && split(b"MM", 1 << 30).is_none());
        assert!(split(&f, 1000).is_none());
        assert!(split(&file(&[Layout { width: 9, height: 9, ..RGB }], true, true, 1).0, 1 << 30).is_none());
    }

    /// Damaged files: what is split comes back. Damaged records: an error or other bytes, never
    /// a panic or an output without bound.
    #[test]
    fn damaged_files_and_records_do_not_panic() {
        let layouts = [RGB, Layout { compression: 32773, predictor: 1, ..RGB }, Layout { compression: 8, predictor: 2, samples: 1, bits: 16, ..RGB }];
        let mut s = 4u64;
        for (le, first) in [(true, true), (false, false)] {
            let (f, _) = file(&layouts, le, first, 11);
            for _ in 0..1200 {
                let mut bad = f.clone();
                for _ in 0..1 + rng(&mut s) % 4 {
                    let i = (rng(&mut s) % bad.len() as u64) as usize;
                    if rng(&mut s) % 6 == 0 { bad.truncate(i.max(9)); } else { bad[i] ^= 1 << (rng(&mut s) % 8); }
                }
                let _ = extent(&bad, 0);
                if let Some(sp) = split(&bad, 1 << 24) { assert!(back(&sp) == bad[sp.start..sp.start + sp.len], "a damaged file that splits must come back"); }
            }
            let sp = split(&f, 1 << 30).unwrap();
            let images: Vec<Vec<u8>> = sp.images.iter().map(|im| im.1.clone()).collect();
            for _ in 0..1200 {
                let (mut rec, mut gaps, mut images) = (sp.records.clone(), sp.gaps.clone(), images.clone());
                let which = match rng(&mut s) % 4 { 0 => &mut gaps, 1 => &mut images[(rng(&mut s) % 3) as usize], _ => &mut rec };
                let i = (rng(&mut s) % which.len() as u64) as usize;
                if rng(&mut s) % 4 == 0 { which.truncate(i); } else { which[i] ^= 1 << (rng(&mut s) % 8); }
                let mut out = vec![];
                let _ = rebuild(&rec, &gaps, &images, &mut out);
                assert!(out.len() < 8 * f.len() + (1 << 16), "{} bytes from a damaged record", out.len());
            }
        }
    }
}
