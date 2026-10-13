//! PNG scanline filters (PNG spec §9): undo them to give the image model plain pixels, and
//! re-apply them with each row's recorded filter type to rebuild the exact scanlines. Filtering
//! is a deterministic function of (pixels, types), so the rebuild is always exact.

/// Scanline layout of a non-interlaced PNG image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geom {
    pub rows: usize,
    /// Bytes per row, without the filter-type byte.
    pub rowbytes: usize,
    /// Filter distance: bytes per complete pixel (at least 1).
    pub bpp: usize,
}

impl Geom {
    /// Layout for an IHDR (width, height, bit depth, colour type, interlace). None for interlaced
    /// or invalid images.
    pub fn new(width: u32, height: u32, depth: u8, color: u8, interlace: u8) -> Option<Geom> {
        if interlace != 0 || width == 0 || height == 0 { return None; }
        let ch: u64 = match (color, depth) {
            (0, 1 | 2 | 4 | 8 | 16) => 1,
            (2, 8 | 16) => 3,
            (3, 1 | 2 | 4 | 8) => 1,
            (4, 8 | 16) => 2,
            (6, 8 | 16) => 4,
            _ => return None,
        };
        let bits = ch * depth as u64;
        let rowbytes = (width as u64 * bits).div_ceil(8);
        if rowbytes > 1 << 28 || height as u64 * (rowbytes + 1) > 1 << 32 { return None; }
        Some(Geom { rows: height as usize, rowbytes: rowbytes as usize, bpp: (bits as usize / 8).max(1) })
    }

    pub fn filtered_len(&self) -> usize { self.rows * (self.rowbytes + 1) }
}

#[inline]
fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let p = a as i16 + b as i16 - c as i16;
    let (pa, pb, pc) = ((p - a as i16).abs(), (p - b as i16).abs(), (p - c as i16).abs());
    if pa <= pb && pa <= pc { a } else if pb <= pc { b } else { c }
}

/// Predictor of byte `x` of a row: `cur` is the row being (re)built, `prev` the row above.
#[inline]
fn predict(t: u8, cur: &[u8], prev: &[u8], x: usize, bpp: usize) -> u8 {
    let a = if x >= bpp { cur[x - bpp] } else { 0 };
    let b = prev[x];
    let c = if x >= bpp { prev[x - bpp] } else { 0 };
    match t {
        1 => a,
        2 => b,
        3 => ((a as u16 + b as u16) / 2) as u8,
        4 => paeth(a, b, c),
        _ => 0,
    }
}

/// Split filtered scanlines into (filter type per row, unfiltered pixel rows). None if the length
/// does not match the layout or a row has an unknown filter type.
pub fn unfilter(data: &[u8], g: &Geom) -> Option<(Vec<u8>, Vec<u8>)> {
    if data.len() != g.filtered_len() { return None; }
    let w = g.rowbytes;
    let mut types = Vec::with_capacity(g.rows);
    let mut pix = vec![0u8; g.rows * w];
    let zero = vec![0u8; w];
    for r in 0..g.rows {
        let src = &data[r * (w + 1)..(r + 1) * (w + 1)];
        let t = src[0];
        if t > 4 { return None; }
        types.push(t);
        let (done, rest) = pix.split_at_mut(r * w);
        let prev = if r == 0 { &zero[..] } else { &done[(r - 1) * w..] };
        let cur = &mut rest[..w];
        for x in 0..w {
            cur[x] = src[1 + x].wrapping_add(predict(t, cur, prev, x, g.bpp));
        }
    }
    Some((types, pix))
}

/// Re-apply the filters: append the scanlines (type byte + filtered row) to `out`.
pub fn refilter(types: &[u8], pix: &[u8], g: &Geom, out: &mut Vec<u8>) -> Result<(), String> {
    let w = g.rowbytes;
    if types.len() != g.rows || pix.len() != g.rows * w { return Err("png rows: length mismatch (corrupt file)".into()); }
    out.reserve(g.filtered_len());
    let zero = vec![0u8; w];
    for r in 0..g.rows {
        let t = types[r];
        if t > 4 { return Err("png rows: bad filter type (corrupt file)".into()); }
        let cur = &pix[r * w..(r + 1) * w];
        let prev = if r == 0 { &zero[..] } else { &pix[(r - 1) * w..r * w] };
        out.push(t);
        for x in 0..w {
            out.push(cur[x].wrapping_sub(predict(t, cur, prev, x, g.bpp)));
        }
    }
    Ok(())
}

/// IMAGE segment params for the unfiltered pixels (see `detect::image_params`).
pub fn image_params(width: u32, depth: u8, color: u8, g: &Geom) -> Vec<u8> {
    use crate::detect::{image_params as ip, imgfmt::*};
    let stride = g.rowbytes as u32;
    match (color, depth) {
        (2, 8) => ip(RGB24, width, stride, 3, false),
        (6, 8) => ip(RGBA32, width, stride, 4, false),
        (4, 8) => ip(GRAY8, width, stride, 2, false),
        (0, 8) | (3, 8) => ip(GRAY8, width, stride, 1, false),
        (0, 16) => ip(GRAY16BE, width, stride, 1, false),
        (4, 16) => ip(GRAY16BE, width, stride, 2, false),
        (2, 16) => ip(RGB48BE, width, stride, 3, false),
        (6, 16) => ip(RGB48BE, width, stride, 4, false),
        // packed sub-byte samples: model the bytes as one gray row
        _ => ip(GRAY8, stride, stride, 1, false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_invert_exactly() {
        let mut s = 7u64;
        for (w, h, d, c) in [(17u32, 9u32, 8u8, 2u8), (5, 4, 16, 6), (33, 3, 1, 0), (8, 8, 8, 4), (1, 1, 8, 0), (13, 7, 4, 3)] {
            let g = Geom::new(w, h, d, c, 0).unwrap();
            let pix: Vec<u8> = (0..g.rows * g.rowbytes).map(|i| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (i as u8).wrapping_mul(3) ^ (s as u8 & 7) }).collect();
            let types: Vec<u8> = (0..g.rows).map(|r| (r % 5) as u8).collect();
            let mut filtered = vec![];
            refilter(&types, &pix, &g, &mut filtered).unwrap();
            assert_eq!(filtered.len(), g.filtered_len());
            let (t2, p2) = unfilter(&filtered, &g).unwrap();
            assert_eq!((t2, p2), (types, pix));
        }
        assert!(Geom::new(10, 10, 8, 2, 1).is_none(), "interlaced images are not unfiltered");
        assert!(unfilter(&[5, 0, 0, 0], &Geom::new(1, 1, 8, 2, 0).unwrap()).is_none());
    }
}
