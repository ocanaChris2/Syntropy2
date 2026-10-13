//! Model for pictures of colour indices (the frames of a GIF).
//!
//! An index is a name, not a quantity: nothing can be averaged or differenced, and a pixel is
//! either one of the colours around it or something else. So a pixel is coded as a few yes/no
//! questions — is it its left neighbour? the one above? transparent? the colour of its palette
//! nearest to what the screen showed at this place? the upper right, the upper left, the
//! neighbours two away? then, one by one, the palette's colours nearest to the colour expected
//! (what the screen showed; for a first picture, what the neighbours suggest) — and only when
//! all say no, as its index. Each question is predicted
//! from the neighbourhood: the colours themselves, which of them are equal, how long the run to
//! the left is and how far the run above still goes, how long ago this place of the screen
//! last changed. In a flat area that is one well-predicted question a pixel.
//!
//! The model thinks in colours, not indices, because an animation may bring a palette of its
//! own with every frame: the same picture then has other indices each time, and what was drawn
//! before says nothing through them, but nearly everything through its colour.
//!
//! A leaf is a stream of frames: a header (a version, the screen's size, the global palette),
//! then each frame's rectangle, its flags and transparent index, its palette if it has one, and
//! its pixels row by row. The model keeps the screen: the colour each place shows after the
//! frames so far.
use super::binary::Bins;
use super::coder::Coder;
use super::predictor::ByteBuf;

/// The stream's version byte.
const VERSION: u8 = 1;
/// A frame's fixed header: left, top, width, height (16 bits each, low byte first), flags, the
/// transparent index.
pub const FRAME: usize = 10;
/// Frame flags: it has a transparent index; the screen under it is cleared when it is done; it
/// brings a palette.
pub const TRANSPARENT: u8 = 1;
pub const CLEAR: u8 = 2;
pub const LOCAL: u8 = 4;

/// The stream's header for a screen of `w` by `h` and a global palette of `rgb` (3 bytes a
/// colour, at most 256 colours; empty: none).
pub fn header(w: u16, h: u16, rgb: &[u8]) -> Vec<u8> {
    let mut o = vec![VERSION];
    o.extend_from_slice(&w.to_le_bytes());
    o.extend_from_slice(&h.to_le_bytes());
    o.extend_from_slice(&((rgb.len() / 3) as u16).to_le_bytes());
    o.extend_from_slice(rgb);
    o
}

/// A frame's header (with its palette `rgb`, if `flags` say [`LOCAL`]).
pub fn frame(left: u16, top: u16, w: u16, h: u16, flags: u8, transparent: u8, rgb: &[u8]) -> Vec<u8> {
    let mut f = vec![];
    for v in [left, top, w, h] { f.extend_from_slice(&v.to_le_bytes()); }
    f.extend_from_slice(&[flags, transparent]);
    if flags & LOCAL != 0 {
        f.extend_from_slice(&((rgb.len() / 3) as u16).to_le_bytes());
        f.extend_from_slice(rgb);
    }
    f
}

/// Walks a stream of frames by its headers (what a format's rebuild needs of it).
pub struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

/// A frame of a stream: its header's fields, its palette (empty unless it brings one), its
/// pixels.
pub struct Frame<'a> {
    pub left: usize,
    pub top: usize,
    pub width: usize,
    pub height: usize,
    pub flags: u8,
    pub transparent: u8,
    pub palette: &'a [u8],
    pub pixels: &'a [u8],
}

impl<'a> Reader<'a> {
    /// The stream `b`: (the screen's size, the global palette, the frames to come).
    pub fn new(b: &'a [u8]) -> Option<((usize, usize), &'a [u8], Reader<'a>)> {
        if b.len() < 7 || b[0] != VERSION { return None; }
        let at16 = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]) as usize;
        let n = at16(5);
        if n > 256 { return None; }
        Some(((at16(1), at16(3)), b.get(7..7 + 3 * n)?, Reader { b, at: 7 + 3 * n }))
    }

    /// The next frame (None: the stream ends, or is cut short — [`Reader::done`] tells).
    pub fn next(&mut self) -> Option<Frame<'a>> {
        let h = self.b.get(self.at..self.at + FRAME)?;
        let at16 = |o: usize| u16::from_le_bytes([h[o], h[o + 1]]) as usize;
        let mut at = self.at + FRAME;
        let mut palette = &self.b[..0];
        if h[8] & LOCAL != 0 {
            let n = u16::from_le_bytes([*self.b.get(at)?, *self.b.get(at + 1)?]) as usize;
            if n > 256 { return None; }
            palette = self.b.get(at + 2..at + 2 + 3 * n)?;
            at += 2 + 3 * n;
        }
        let (width, height) = (at16(4), at16(6));
        let pixels = self.b.get(at..at + width * height)?;
        self.at = at + width * height;
        Some(Frame { left: at16(0), top: at16(2), width, height, flags: h[8], transparent: h[9], palette, pixels })
    }

    pub fn done(&self) -> bool { self.at == self.b.len() }
}

/// No pixel: outside the frame.
const NONE: u16 = 256;
/// A screen place nothing was drawn at; and the colour key of a transparent pixel.
const BLANK: u32 = 1 << 30;
const CLEARKEY: u64 = 1 << 25;
/// Largest screen kept (pixels).
const MAX_SCREEN: usize = 1 << 24;

const F_HEAD: usize = 0;
const F_FIRST: usize = 1;
const F_SECOND: usize = 2;
const F_MORE: usize = 3;
const F_RANK: usize = 4;
const F_VALUE: usize = 5;
const FAMILIES: usize = 6;
/// Table sizes (log2, per context) by the size of the first leaf a model codes: a stream of a
/// few kilobytes does not fill, and should not pay for, the tables of a long animation.
static BITS: [[u32; FAMILIES]; 4] = [[12, 17, 16, 16, 16, 15], [13, 18, 17, 17, 17, 16], [14, 19, 18, 18, 18, 17], [14, 20, 19, 19, 19, 18]];
static NIN: [usize; FAMILIES] = [6, 12, 10, 7, 9, 12];
/// Colours nearest to the expected one that are asked about, and kept per expected colour (the
/// neighbours asked about before may be among them).
const ASKED: usize = 8;
const LIST: usize = 18;

/// Which candidate comes from where: the left and upper neighbours are 0 and 1, then these two,
/// then the ring around.
const S_CLEAR: u64 = 2;
const S_NEAR: u64 = 3;

#[inline]
fn mix(h: u64, v: u64) -> u64 { (h ^ v).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(27) }

#[inline]
fn hash(vals: &[u64]) -> u64 { vals.iter().fold(0x2545_F491_4F6C_DD1D, |h, &v| mix(h, v)) }

/// Squared distance of two colours (0xRRGGBB).
#[inline]
fn dist(a: u32, b: u32) -> u32 {
    let d = |s: u32| {
        let x = (a >> s & 255) as i32 - (b >> s & 255) as i32;
        (x * x) as u32
    };
    d(16) + d(8) + d(0)
}

/// A colour to four bits a channel, as a key.
#[inline]
fn coarse(col: u32) -> u64 { 1 << 13 | ((col >> 20 & 15) << 8 | (col >> 12 & 15) << 4 | (col >> 4 & 15)) as u64 }

/// What a pixel is predicted from: colour keys of the pixels around (0: none), exact and coarse.
struct Near {
    /// Exact keys: W, N, NE, NW, WW, NN, NEE, NNW, NNE, WWW.
    x: [u64; 10],
    /// Coarse keys of W, N, NE, NW.
    q: [u64; 4],
    /// What the screen shows here (exact and coarse key; 0: nothing), and how many frames ago it
    /// was drawn (3: longer, or never).
    px: u64,
    pq: u64,
    age: u64,
    /// Pixels to the left equal to the left neighbour; pixels above to the right equal to the
    /// one above.
    run: u64,
    ahead: u64,
    /// Which of the nine places of the screen around this one the frame before drew at (a bit
    /// each, row by row from the upper left).
    before: u64,
}

pub struct IndexedModel {
    bins: Bins,
    /// The screen's colours, and the number of the frame that drew each.
    screen: Vec<u32>,
    drawn: Vec<u32>,
    /// Three rows of indices, the row being coded last; how far the runs of the row above go.
    rows: [Vec<u16>; 3],
    ahead: Vec<u8>,
    /// The global palette and the one in force: colours, and their keys by index (the last
    /// entry: no pixel).
    global: Vec<u32>,
    colours: Vec<u32>,
    exact: [u64; 257],
    coarse: [u64; 257],
    /// The palette entry nearest to a colour, and the entries nearest to one in order, as found
    /// before in this frame.
    nearest: Vec<(u32, u16)>,
    ranked: Vec<(u32, [u8; LIST], u8)>,
    order: Vec<(u32, u16)>,
    /// Developer statistics of the first question: (count, bits·65536) by whether the pixels
    /// around are one colour, whether the left one is transparent, and the answer.
    pub detail: [(u64, u64); 8],
}

impl IndexedModel {
    /// A model for a lane whose first leaf of frames has `bytes` bytes.
    pub fn new(bytes: usize) -> Self {
        IndexedModel {
            // (the maps adapt fast: what a context saw sixteen pixels ago counts for little)
            bins: Bins::adapting(&BITS[match bytes { 0..=0x3_FFFF => 0, 0x4_0000..=0xF_FFFF => 1, 0x10_0000..=0x3F_FFFF => 2, _ => 3 }], &NIN, 15, true), screen: vec![], drawn: vec![], rows: [vec![], vec![], vec![]], ahead: vec![],
            global: vec![], colours: vec![], exact: [0; 257], coarse: [0; 257], nearest: vec![(u32::MAX, 0); 1024], ranked: vec![(u32::MAX, [0; LIST], 0); 1024], order: Vec::with_capacity(256), detail: [(0, 0); 8],
        }
    }

    /// Developer statistics: the cost of each kind of decision (in bits·65536: the headers and
    /// palettes, "it is the first candidate", the second, a later neighbour, one of the nearest
    /// colours, the bare index).
    pub fn enable_stats(&mut self) { self.bins.stats = Some(vec![0; FAMILIES]); }
    pub fn stats(&self) -> Vec<u64> { self.bins.stats.clone().unwrap_or_default() }

    /// Code `out.len()` header bytes at `buf[*pos]`, each against the same byte of `before`.
    #[allow(clippy::too_many_arguments)]
    fn head<C: Coder, B: ByteBuf>(&mut self, c: &mut C, buf: &mut B, pos: &mut usize, end: usize, out: &mut [u8], id: u64, before: &[u8]) -> Result<(), String> {
        if out.len() > end - *pos { return Err("indexed: a header cut short (corrupt data)".into()); }
        for k in 0..out.len() {
            let v = if buf.reading() { buf.byte(*pos) as u32 } else { 0 };
            let coded = self.bins.tree(c, v, 8, F_HEAD, id * 16 + k as u64, before.get(k).copied().unwrap_or(0) as u64, if k > 0 { out[k - 1] as u64 } else { 0 }, 0) as u8;
            if !buf.reading() { buf.put(*pos, coded); }
            out[k] = coded;
            *pos += 1;
        }
        Ok(())
    }

    /// Code a palette at `buf[*pos]` (a count, then the colours), returning its colours: each
    /// channel against the same entry of `before`, the colour before, and the channel before.
    /// (Palettes made anew for every frame share little: coding a colour from the next in turn
    /// of the palette before, from a near one a few places off, or its channels as differences
    /// from one another, all cost more than this.)
    fn palette<C: Coder, B: ByteBuf>(&mut self, c: &mut C, buf: &mut B, pos: &mut usize, end: usize, before: &[u32]) -> Result<Vec<u32>, String> {
        let mut n = [0u8; 2];
        self.head(c, buf, pos, end, &mut n, 2, &[before.len() as u8, (before.len() >> 8) as u8])?;
        let n = u16::from_le_bytes(n) as usize;
        if n > 256 || 3 * n > end - *pos { return Err("indexed: a palette out of its leaf (corrupt data)".into()); }
        let mut out = Vec::with_capacity(n);
        let mut last = 0u32;
        for i in 0..n {
            let old = before.get(i).copied().unwrap_or(0);
            let mut rgb = 0u32;
            for ch in 0..3u32 {
                let s = 16 - 8 * ch;
                let v = if buf.reading() { buf.byte(*pos) as u32 } else { 0 };
                let x = self.bins.tree(c, v, 8, F_HEAD, 48 + ch as u64, (old >> s & 255) as u64, (last >> s & 255) as u64, if ch > 0 { (rgb >> (s + 8) & 255) as u64 } else { 0 });
                if !buf.reading() { buf.put(*pos, x as u8); }
                rgb |= x << s;
                *pos += 1;
            }
            out.push(rgb);
            last = rgb;
        }
        Ok(out)
    }

    /// The palette entry nearest to `colour` (not the transparent one; `NONE`: there is none).
    #[inline]
    fn near(&mut self, colour: u32, transparent: Option<u8>) -> u16 {
        let slot = (colour.wrapping_mul(0x9E37_79B1) >> 22) as usize;
        if self.nearest[slot].0 == colour { return self.nearest[slot].1; }
        let best = self.colours.iter().enumerate().filter(|&(i, _)| Some(i as u8) != transparent || i > 255).min_by_key(|&(i, &col)| (dist(col, colour), i)).map_or(NONE, |(i, _)| i as u16);
        self.nearest[slot] = (colour, best);
        best
    }

    /// Code one pixel (`v`: its value, when encoding). `idx`: the indices of W, N, NE, NW, WW,
    /// NN, NEE; `expected`: the colour expected, if any; `shown`: the screen's colour here;
    /// `transparent`: the frame's transparent index.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn pixel<C: Coder>(&mut self, c: &mut C, v: Option<u8>, k: &Near, idx: &[u16], expected: Option<u32>, shown: Option<u32>, transparent: Option<u8>) -> Result<u8, String> {
        let eq = |a: u64, b: u64| (a == b) as u64;
        let [w, n, ne, nw, ww, nn, nee, nnw, nne, www] = k.x;
        let pat = eq(n, w) | eq(nw, w) << 1 | eq(ne, w) << 2 | eq(ww, w) << 3 | eq(nn, n) << 4 | eq(k.px, w) << 5
            | eq(ne, n) << 6 | eq(nw, n) << 7 | eq(k.px, n) << 8 | eq(nee, ne) << 9 | eq(w, CLEARKEY) << 10 | eq(n, CLEARKEY) << 11;
        let runq = match k.run { 0..=3 => k.run, 4..=7 => 4, 8..=15 => 5, 16..=63 => 6, _ => 7 };
        let far = k.ahead.min(3);
        // the wider neighbourhood: its colours as one pattern, and which of them equal the left one
        let wide = hash(&k.x);
        let wide_eq = k.x.iter().enumerate().fold(0u64, |m, (i, &x)| m | eq(x, w) << i) | eq(nn, n) << 10 | eq(nne, ne) << 11 | eq(nnw, nw) << 12 | eq(www, ww) << 13;
        // the colours it may be, each once
        let mut tried = [NONE; 9];
        let mut nt = 0usize;
        for src in 0..9u64 {
            let cand = match src {
                0 | 1 => idx[src as usize],
                S_CLEAR => transparent.map_or(NONE, |t| t as u16),
                S_NEAR => shown.map_or(NONE, |col| self.near(col, transparent)),
                _ => idx[src as usize - 2],
            };
            if cand == NONE || tried[..nt].contains(&cand) { continue; }
            let r = nt as u64;
            tried[nt] = cand;
            nt += 1;
            let bit = v.map_or(0, |v| (v as u16 == cand) as u32);
            let (cx, cq) = (self.exact[cand as usize], self.coarse[cand as usize]);
            let coded = match r {
                0 => {
                    let ctx = [
                        hash(&[src, w, n, ne]),
                        hash(&[src, w, n, ne, nw]),
                        hash(&[w, n, ne, nw, ww, nn]),
                        pat | k.age << 12 | runq << 14 | far << 17 | src << 19,
                        hash(&[k.q[0], k.pq, k.age, eq(n, w), runq]),
                        hash(&[src, w, ne, nee, k.ahead]),
                        k.run.min(255) | k.ahead << 8 | eq(n, w) << 12 | k.q[0] << 13 | src << 40,
                        hash(&[k.q[0], k.q[1], k.q[2], k.pq, k.age]),
                        wide ^ src << 60,
                        wide_eq | k.q[0] << 14 | src << 40,
                        // where the frame before drew: animations draw again where they drew
                        k.before | (pat & 0xFFF) << 9 | k.age << 21 | src << 23,
                        hash(&[k.before, src, w, n, runq]),
                    ];
                    let before = self.bins.cost(F_FIRST);
                    let coded = self.bins.bit2(c, bit, F_FIRST, &ctx, (pat & 63) as usize, (k.age | runq << 2) as usize, ((pat & 7) | runq << 3) as usize);
                    if self.bins.stats.is_some() {
                        // developer statistics: where the first question's bits go
                        let flat = (pat & 7 == 7) as usize;
                        let clear = (w == CLEARKEY) as usize;
                        let class = flat | clear << 1 | (coded as usize) << 2;
                        self.detail[class].0 += 1;
                        self.detail[class].1 += self.bins.cost(F_FIRST) - before;
                    }
                    coded
                }
                1 => {
                    let ctx = [
                        hash(&[src, w, n, ne]),
                        hash(&[src, w, n, ne, nw]),
                        hash(&[src, cx, ne, nn]),
                        pat | k.age << 12 | src << 14 | far << 18,
                        hash(&[src, cq, k.pq, k.age]),
                        hash(&[src, n, ne, nee, k.ahead]),
                        hash(&[src, k.q[0], k.q[1], k.q[2], k.pq]),
                        wide ^ src << 60,
                        wide_eq | cq << 14 | src << 40,
                        k.before | (pat & 0xFFF) << 9 | k.age << 21 | src << 23,
                    ];
                    self.bins.bit2(c, bit, F_SECOND, &ctx, (pat >> 4 & 63) as usize, (src | k.age << 4) as usize, (src | (pat & 3) << 4) as usize)
                }
                _ => {
                    // how far the candidate's colour is from what the screen shows
                    let gap = match (shown, self.colours.get(cand as usize)) {
                        (Some(col), Some(&own)) if src != S_CLEAR => (32 - dist(own, col).leading_zeros()) as u64,
                        _ => 31,
                    };
                    let ctx = [
                        hash(&[src, r, cx, w, n]),
                        pat | r << 12 | src << 16 | k.age << 20,
                        hash(&[src, r, cq, k.pq]),
                        hash(&[src, r, w, n, ne, nw]),
                        hash(&[src, r, cx, ne, nw]),
                        src | r << 4 | gap << 8 | k.age << 13,
                        wide ^ (src | r << 4) << 56,
                    ];
                    self.bins.bit2(c, bit, F_MORE, &ctx, (src | r.min(3) << 4) as usize, (pat & 63) as usize, (src | k.age << 4) as usize)
                }
            };
            if coded == 1 { return Ok(cand as u8); }
        }
        // then the palette's colours nearest to the expected one, in order
        if let Some(expected) = expected {
            let slot = (expected.wrapping_mul(0x9E37_79B1) >> 22) as usize;
            if self.ranked[slot].0 != expected {
                self.order.clear();
                for (i, &col) in self.colours.iter().enumerate().take(256) {
                    if Some(i as u8) != transparent { self.order.push((dist(col, expected), i as u16)); }
                }
                let n = self.order.len().min(LIST);
                if n > 0 && n < self.order.len() { self.order.select_nth_unstable(n - 1); }
                self.order[..n].sort_unstable();
                let mut list = [0u8; LIST];
                for (l, o) in list.iter_mut().zip(&self.order[..n]) { *l = o.1 as u8; }
                self.ranked[slot] = (expected, list, n as u8);
            }
            let (_, list, have) = self.ranked[slot];
            let mut asked = 0u64;
            for &cand in &list[..have as usize] {
                let cand = cand as u16;
                if tried[..nt].contains(&cand) { continue; }
                let bit = v.map_or(0, |v| (v as u16 == cand) as u32);
                let (cx, cq) = (self.exact[cand as usize], self.coarse[cand as usize]);
                let gap = (32 - dist(self.colours[cand as usize], expected).leading_zeros()) as u64;
                let here = shown.is_some() as u64;
                let ctx = [
                    asked | gap << 4 | k.age << 10 | here << 12 | (nt as u64) << 13,
                    hash(&[asked, cx]),
                    hash(&[cx, w]),
                    hash(&[cx, n]),
                    hash(&[asked, w, n]),
                    hash(&[asked, k.px, here]),
                    hash(&[cx, k.px]),
                    hash(&[asked, k.q[0], k.pq, gap]),
                    hash(&[cx, w, n]),
                ];
                let coded = self.bins.bit2(c, bit, F_RANK, &ctx, (asked | here << 3) as usize, (gap.min(15) | k.age << 4) as usize, (asked | here << 3 | k.age << 4) as usize);
                if coded == 1 { return Ok(cand as u8); }
                asked += 1;
                if asked as usize == ASKED { break; }
            }
        }
        // none of them: the index itself (where indices are an order of colours, the colours
        // around predict a number)
        let num = |j: u16| if j == NONE { 0 } else { j as i64 };
        let (a, b, d) = (num(idx[0]), num(idx[1]), num(idx[3]));
        let med = if d >= a.max(b) { a.min(b) } else if d <= a.min(b) { a.max(b) } else { a + b - d } as u64;
        let v = v.unwrap_or(0) as u32;
        let mut node = 1u64;
        for i in (0..8).rev() {
            let ctx = [
                node, node | w << 8, node | n << 8, hash(&[node, w, n]), node | k.px << 8, node | ne << 8,
                hash(&[node, w, n, ne]), hash(&[node, w, nw, n]), hash(&[node, ww, w]), hash(&[node, nn, n]), hash(&[node, w, n, k.px]),
                node | med << 8 | 1 << 40,
            ];
            let bit = self.bins.bit2(c, (v >> i) & 1, F_VALUE, &ctx, 7 - i, nt, (node & 63) as usize);
            node = node << 1 | bit as u64;
        }
        Ok((node & 255) as u8)
    }

    /// Code (encode, or decode when `buf` is the decoder's) one leaf of frames in `buf[range]`.
    pub fn code_leaf<C: Coder, B: ByteBuf>(&mut self, c: &mut C, buf: &mut B, range: std::ops::Range<usize>, _params: &[u8]) -> Result<(), String> {
        let enc = buf.reading();
        let (mut pos, end) = (range.start, range.end);
        let mut scr = [0u8; 5];
        self.head(c, buf, &mut pos, end, &mut scr, 0, &[])?;
        if scr[0] != VERSION { return Err("indexed: a stream of another version".into()); }
        let (sw, sh) = (u16::from_le_bytes([scr[1], scr[2]]) as usize, u16::from_le_bytes([scr[3], scr[4]]) as usize);
        self.global = self.palette(c, buf, &mut pos, end, &[])?;
        let kept = sw * sh > 0 && sw * sh <= MAX_SCREEN;
        self.screen.clear();
        self.screen.resize(if kept { sw * sh } else { 0 }, BLANK);
        self.drawn.clear();
        self.drawn.resize(self.screen.len(), 0);
        let mut before = [0u8; FRAME];
        let mut last_palette: Vec<u32> = vec![];
        let mut number = 0u32;
        while pos < end {
            let mut h = [0u8; FRAME];
            self.head(c, buf, &mut pos, end, &mut h, 1, &before)?;
            before = h;
            let at16 = |o: usize| u16::from_le_bytes([h[o], h[o + 1]]) as usize;
            let (left, top, w, hgt) = (at16(0), at16(2), at16(4), at16(6));
            let flags = h[8];
            let transparent = (flags & TRANSPARENT != 0).then_some(h[9]);
            if flags & LOCAL != 0 {
                last_palette = self.palette(c, buf, &mut pos, end, &last_palette)?;
                self.colours.clone_from(&last_palette);
            } else {
                self.colours.clone_from(&self.global);
            }
            if w == 0 || hgt == 0 || w * hgt > end - pos { return Err("indexed: a frame out of its leaf (corrupt data)".into()); }
            number += 1;
            // the colours' keys by index: exact, and to four bits a channel
            for i in 0..256 {
                (self.exact[i], self.coarse[i]) = match self.colours.get(i) {
                    _ if Some(i as u8) == transparent => (CLEARKEY, 1 << 12),
                    Some(&col) => (1 << 24 | col as u64, coarse(col)),
                    None => (1 << 26 | i as u64, 1 << 14 | i as u64),
                };
            }
            (self.exact[256], self.coarse[256]) = (0, 0);
            self.nearest.fill((u32::MAX, 0));
            for r in self.ranked.iter_mut() { r.0 = u32::MAX; }
            // three rows, a margin of three on the left and two on the right
            for r in self.rows.iter_mut() {
                r.clear();
                r.resize(w + 5, NONE);
            }
            self.ahead.clear();
            self.ahead.resize(w + 5, 0);
            for y in 0..hgt {
                let mut run = 0u64;
                for x in 0..w {
                    let i = x + 3;
                    let [up2, up, cur] = &self.rows;
                    // W, N, NE, NW, WW, NN, NEE, NNW, NNE, WWW
                    let idx = [cur[i - 1], up[i], up[i + 1], up[i - 1], cur[i - 2], up2[i], up[i + 2], up2[i - 1], up2[i + 1], cur[i - 3]];
                    let at = (top + y) * sw + left + x;
                    let on = kept && left + x < sw && top + y < sh;
                    let shown = if on { Some(self.screen[at]).filter(|&col| col != BLANK) } else { None };
                    let k = Near {
                        x: idx.map(|j| self.exact[j as usize]),
                        q: [self.coarse[idx[0] as usize], self.coarse[idx[1] as usize], self.coarse[idx[2] as usize], self.coarse[idx[3] as usize]],
                        px: shown.map_or(0, |col| 1 << 24 | col as u64),
                        pq: shown.map_or(0, coarse),
                        age: if on && self.drawn[at] != 0 { ((number - self.drawn[at]) as u64).min(3) } else { 3 },
                        run,
                        ahead: self.ahead[i] as u64,
                        before: if kept && number > 1 {
                            let (cx, cy) = ((left + x) as isize, (top + y) as isize);
                            let mut m = 0u64;
                            for (bit, (dx, dy)) in [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)].into_iter().enumerate() {
                                let (px, py) = (cx + dx, cy + dy);
                                if px >= 0 && py >= 0 && (px as usize) < sw && (py as usize) < sh && self.drawn[py as usize * sw + px as usize] == number - 1 { m |= 1 << bit; }
                            }
                            m
                        } else { 0 },
                    };
                    // the colour expected: what the screen shows; else what the neighbours do
                    // (the left, upper and upper left ones: their median, channel by channel)
                    let colour = |j: u16| self.colours.get(j as usize).copied().filter(|_| Some(j as u8) != transparent);
                    let expected = shown.or_else(|| match (colour(idx[0]), colour(idx[1]), colour(idx[3])) {
                        (Some(a), Some(b), Some(d)) => Some((0..3).fold(0, |m, ch| {
                            let s = 8 * ch;
                            let (a, b, d) = ((a >> s & 255) as i32, (b >> s & 255) as i32, (d >> s & 255) as i32);
                            m | ((a + b - d).clamp(a.min(b), a.max(b)) as u32) << s
                        })),
                        (a, b, _) => a.or(b),
                    });
                    let v = self.pixel(c, enc.then(|| buf.byte(pos)), &k, &idx[..7], expected, shown, transparent)?;
                    if !enc { buf.put(pos, v); }
                    pos += 1;
                    run = if x > 0 && v as u16 == idx[0] { run + 1 } else { 0 };
                    self.rows[2][i] = v as u16;
                    if on && Some(v) != transparent {
                        // (an index the palette has no colour for shows as itself)
                        self.screen[at] = self.colours.get(v as usize).copied().unwrap_or(1 << 24 | v as u32);
                        self.drawn[at] = number;
                    }
                }
                // the row becomes the one above: how far each of its runs goes on to the right
                let cur = &self.rows[2];
                for i in (3..w + 3).rev() { self.ahead[i] = if cur[i + 1] == cur[i] { (self.ahead[i + 1] + 1).min(7) } else { 0 }; }
                self.rows.rotate_left(1);
            }
            if flags & CLEAR != 0 && kept {
                for y in top..(top + hgt).min(sh) {
                    for x in left..(left + w).min(sw) {
                        self.screen[y * sw + x] = BLANK;
                        self.drawn[y * sw + x] = number;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::cm::coder::{Decoder, Encoder};
    use crate::cm::predictor::{ReadBuf, WriteBuf};

    fn rng(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }

    /// A stream of `frames` frames on a screen of `w` by `h`: flat shapes that move a little
    /// from frame to frame; from the second on, only what changed, the rest transparent. Every
    /// third frame brings a palette of its own (the same colours in another order).
    pub fn frames(n: usize, w: usize, h: usize, colours: u64) -> Vec<u8> {
        let mut s = 31u64;
        let palette = |shift: u64| -> Vec<u8> {
            (0..colours).flat_map(|i| {
                let j = (i + shift) % colours;
                [(j * 37) as u8, (j * 91 + 5) as u8, (255 - j * 11) as u8]
            }).collect()
        };
        let mut out = header(w as u16, h as u16, &palette(0));
        let mut screen = vec![0u8; w * h];
        for f in 0..n {
            // a few rectangles of one colour each, and some speckle
            let mut next = screen.clone();
            for _ in 0..4 {
                let (x0, y0) = ((rng(&mut s) % w as u64) as usize, (rng(&mut s) % h as u64) as usize);
                let (x1, y1) = ((x0 + 3 + (rng(&mut s) % 30) as usize).min(w), (y0 + 3 + (rng(&mut s) % 20) as usize).min(h));
                let col = 1 + (rng(&mut s) % (colours - 1)) as u8;
                for y in y0..y1 { next[y * w + x0..y * w + x1].fill(col); }
            }
            for _ in 0..w * h / 300 { next[(rng(&mut s) % (w * h) as u64) as usize] = (rng(&mut s) % colours) as u8; }
            // the smallest rectangle that holds the changes (the whole screen first)
            let changed: Vec<usize> = (0..w * h).filter(|&i| f == 0 || next[i] != screen[i]).collect();
            let (x0, x1) = (changed.iter().map(|i| i % w).min().unwrap_or(0), changed.iter().map(|i| i % w).max().unwrap_or(0) + 1);
            let (y0, y1) = (changed.iter().map(|i| i / w).min().unwrap_or(0), changed.iter().map(|i| i / w).max().unwrap_or(0) + 1);
            // (the transparent index: one no colour has)
            let transparent = colours as u8;
            let local = f % 3 == 2;
            let shift = if local { 1 + f as u64 % (colours - 1) } else { 0 };
            let flags = if f > 0 { TRANSPARENT } else { 0 } | if f % 7 == 6 { CLEAR } else { 0 } | if local { LOCAL } else { 0 };
            out.extend(frame(x0 as u16, y0 as u16, (x1 - x0) as u16, (y1 - y0) as u16, flags, transparent, &palette(shift)));
            for y in y0..y1 {
                // the colour's index in this frame's palette
                for x in x0..x1 { out.push(if f > 0 && next[y * w + x] == screen[y * w + x] { transparent } else { ((next[y * w + x] as u64 + colours - shift) % colours) as u8 }); }
            }
            screen = next;
        }
        out
    }

    #[test]
    fn frames_roundtrip_and_compress() {
        for (n, w, h, colours) in [(12usize, 90usize, 60usize, 16u64), (1, 300, 200, 100), (30, 17, 9, 4), (3, 1, 1, 2), (2, 500, 3, 40)] {
            let data = frames(n, w, h, colours);
            // the stream reads back by its headers
            let (screen, global, mut r) = Reader::new(&data).unwrap();
            assert_eq!((screen, global.len()), ((w, h), 3 * colours as usize));
            let mut seen = 0;
            while let Some(f) = r.next() {
                assert!(f.width > 0 && f.pixels.len() == f.width * f.height && (f.palette.is_empty() == (f.flags & LOCAL == 0)));
                seen += 1;
            }
            assert!(seen == n && r.done());
            let mut enc = Encoder::new();
            IndexedModel::new(data.len()).code_leaf(&mut enc, &mut ReadBuf(&data), 0..data.len(), &[]).unwrap();
            let coded = enc.finish();
            let mut out = vec![0u8; data.len()];
            IndexedModel::new(data.len()).code_leaf(&mut Decoder::new(&coded), &mut WriteBuf(&mut out), 0..data.len(), &[]).unwrap();
            assert!(out == data, "{n} frames of {w}x{h}: roundtrip");
            if data.len() > 4000 { assert!(coded.len() * 6 < data.len(), "{n} frames of {w}x{h}: {} -> {}", data.len(), coded.len()); }
            // any byte may be a pixel: also one the palette has no colour for (in the first
            // frame, which is the whole screen)
            let mut wild = data.clone();
            let mut s = 3u64;
            let first = 7 + 3 * colours as usize + FRAME;
            for _ in 0..40 { wild[first + (rng(&mut s) % (w * h) as u64) as usize] = rng(&mut s) as u8; }
            let mut enc = Encoder::new();
            IndexedModel::new(wild.len()).code_leaf(&mut enc, &mut ReadBuf(&wild), 0..wild.len(), &[]).unwrap();
            let coded_wild = enc.finish();
            let mut out = vec![0u8; wild.len()];
            IndexedModel::new(wild.len()).code_leaf(&mut Decoder::new(&coded_wild), &mut WriteBuf(&mut out), 0..wild.len(), &[]).unwrap();
            assert!(out == wild, "{n} frames of {w}x{h}: roundtrip of pixels beyond the palette");
            // damage: an error, or other pixels
            for _ in 0..20 {
                let mut bad = coded.clone();
                let i = (rng(&mut s) % bad.len() as u64) as usize;
                bad[i] ^= 1 << (rng(&mut s) % 8);
                let mut out = vec![0u8; data.len()];
                let _ = IndexedModel::new(data.len()).code_leaf(&mut Decoder::new(&bad), &mut WriteBuf(&mut out), 0..data.len(), &[]);
            }
        }
        // a leaf that is not a stream of frames: another version, a frame larger than the leaf,
        // headers cut short
        let whole = frames(2, 20, 10, 8);
        let mut other = whole.clone();
        other[0] = 9;
        let mut large = whole.clone();
        large[7 + 24 + 6] = 200;
        for bad in [&other[..], &large[..], &whole[..3], &whole[..7 + 24 + 4]] {
            let mut enc = Encoder::new();
            assert!(IndexedModel::new(bad.len()).code_leaf(&mut enc, &mut ReadBuf(bad), 0..bad.len(), &[]).is_err());
            assert!(Reader::new(bad).is_none_or(|(_, _, mut r)| {
                while r.next().is_some() {}
                !r.done()
            }));
        }
    }
}
