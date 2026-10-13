//! Model for JPEG DCT coefficients (the coefficient leaves of JPEG containers).
//!
//! The leaf holds a compact serialization of every block's coefficients (see
//! `transform::jpeg::write_coefs`). Instead of modelling those bytes, this model codes the
//! coefficients themselves, block by block (component planes, raster order):
//! 1. the number of nonzero AC coefficients, as a 6-bit binary tree;
//! 2. the AC coefficients in zigzag order until no nonzero one is left: a zero flag, then the
//!    magnitude's bit length (unary), the sign and the bits below the leading one;
//! 3. the DC coefficient as a residual from a pixel-domain prediction: the DC offset that makes
//!    the block's edge pixels continue the gradients of its left and upper neighbours' edges
//!    (boundary pixels from an integer IDCT of the dequantized coefficients).
//!
//! Every binary decision mixes several adaptive context predictions (neighbouring blocks above and
//! left at the same frequency, nonzeros left, position, already-coded neighbours in the block)
//! with an online mixer and refines the result with an APM. Integer-only, like the rest of the
//! engine, so files decode identically everywhere.
use super::coder::Coder;
use super::mixer::Mixer;
use super::predictor::ByteBuf;
use super::squash::stretch;
use super::statemap::{Apm, StateMap};
use crate::transform::jpeg::ZIGZAG;

/// Decision families (mixer weight sets are chosen per family and band).
const F_NZ: usize = 0;
const F_ZF: usize = 1;
const F_EXP: usize = 2;
const F_SIGN: usize = 3;
const F_MANT: usize = 4;
const F_DCZ: usize = 5;
const F_DCEXP: usize = 6;
const F_DCSIGN: usize = 7;
const F_DCMANT: usize = 8;
const F_NE: usize = 9;
pub const FAMILIES: usize = 10;
/// Context inputs per decision.
const INPUTS: usize = 5;

/// Zigzag index of natural position `n`.
const fn zz_index(n: usize) -> usize {
    let mut k = 0;
    while ZIGZAG[k] != n { k += 1; }
    k
}

/// AC coding order (zigzag indices): the 7×7 interior in zigzag order, then the first row, then
/// the first column, so the edge coefficients can be predicted from the interior and the
/// neighbouring blocks.
const ORDER: [usize; 63] = {
    let mut o = [0usize; 63];
    let (mut n, mut k) = (0, 1);
    while k < 64 {
        let p = ZIGZAG[k];
        if p / 8 >= 1 && p % 8 >= 1 { o[n] = k; n += 1; }
        k += 1;
    }
    let mut i = 1;
    while i < 8 { o[n] = zz_index(i); n += 1; i += 1; }
    i = 1;
    while i < 8 { o[n] = zz_index(i * 8); n += 1; i += 1; }
    o
};
/// Entries per context table (direct-indexed; each family packs its contexts below this).
const TABLE: usize = 1 << 16;

/// Image geometry and quantization from the leaf params.
pub struct Geom {
    pub comps: Vec<(usize, usize)>, // (blocks wide, blocks high) per component, padded grid
    pub quant: Vec<[u16; 64]>,      // per component, natural order
}

const PARAMS_VERSION: u8 = 2;
const COMP_PARAMS: usize = 8 + 128;

/// Leaf params: version, component count, then per component: blocks wide u32, blocks high u32,
/// quantization table (64 × u16, natural order).
pub fn params(f: &crate::transform::jpeg::Frame, quant: &[[u16; 64]]) -> Vec<u8> {
    let mut p = vec![PARAMS_VERSION, f.comps.len() as u8];
    for (c, q) in f.comps.iter().zip(quant) {
        p.extend_from_slice(&(c.bw as u32).to_le_bytes());
        p.extend_from_slice(&(c.bh as u32).to_le_bytes());
        for v in q { p.extend_from_slice(&v.to_le_bytes()); }
    }
    p
}

pub fn parse_params(p: &[u8]) -> Result<Geom, String> {
    if p.len() < 2 || p[0] != PARAMS_VERSION || !(1..=4).contains(&p[1]) || p.len() != 2 + COMP_PARAMS * p[1] as usize {
        return Err("bad jpeg coefficient parameters".into());
    }
    let (mut comps, mut quant) = (vec![], vec![]);
    let mut blocks = 0usize;
    for i in 0..p[1] as usize {
        let o = 2 + COMP_PARAMS * i;
        let r = |o: usize| u32::from_le_bytes(p[o..o + 4].try_into().unwrap()) as usize;
        let (w, h) = (r(o), r(o + 4));
        if w == 0 || h == 0 || w > 1 << 16 || h > 1 << 16 { return Err("bad jpeg coefficient parameters".into()); }
        blocks += w * h;
        comps.push((w, h));
        let mut q = [0u16; 64];
        for (k, v) in q.iter_mut().enumerate() { *v = u16::from_le_bytes([p[o + 8 + 2 * k], p[o + 9 + 2 * k]]).max(1); }
        quant.push(q);
    }
    if blocks > 1 << 23 { return Err("bad jpeg coefficient parameters".into()); }
    Ok(Geom { comps, quant })
}

/// IDCT basis: IDCT[k][x] = round(4096 · C(k)/2 · cos((2x+1)kπ/16)), C(0) = 1/√2.
const IDCT: [[i64; 8]; 8] = [
    [1448, 1448, 1448, 1448, 1448, 1448, 1448, 1448],
    [2009, 1703, 1138, 400, -400, -1138, -1703, -2009],
    [1892, 784, -784, -1892, -1892, -784, 784, 1892],
    [1703, -400, -2009, -1138, 1138, 2009, 400, -1703],
    [1448, -1448, -1448, 1448, 1448, -1448, -1448, 1448],
    [1138, -2009, 400, 1703, -1703, -400, 2009, -1138],
    [784, -1892, 1892, -784, -784, 1892, -1892, 784],
    [400, -1138, 1703, -2009, 2009, -1703, 1138, -400],
];
/// Pixel offset (×2^24) of one dequantized DC unit.
const DC_UNIT: i64 = 1448 * 1448;

/// Pixels (×2^24) of the dequantized block `f` (natural order) along column `x`, y = 0..7.
fn col_pixels(f: &[i64; 64], x: usize) -> [i64; 8] {
    let mut g = [0i64; 8];
    for (v, gv) in g.iter_mut().enumerate() { *gv = (0..8).map(|u| IDCT[u][x] * f[v * 8 + u]).sum(); }
    let mut out = [0i64; 8];
    for (y, o) in out.iter_mut().enumerate() { *o = (0..8).map(|v| IDCT[v][y] * g[v]).sum(); }
    out
}

/// Pixels (×2^24) of `f` along row `y`, x = 0..7.
fn row_pixels(f: &[i64; 64], y: usize) -> [i64; 8] {
    let mut g = [0i64; 8];
    for (u, gu) in g.iter_mut().enumerate() { *gu = (0..8).map(|v| IDCT[v][y] * f[v * 8 + u]).sum(); }
    let mut out = [0i64; 8];
    for (x, o) in out.iter_mut().enumerate() { *o = (0..8).map(|u| IDCT[u][x] * g[u]).sum(); }
    out
}

#[inline]
fn div_round(a: i64, b: i64) -> i64 { if a >= 0 { (a + b / 2) / b } else { -((-a + b / 2) / b) } }

#[inline]
fn lg(x: u32) -> usize { (32 - x.leading_zeros()) as usize }

/// Bucket for counts 0..=64 (16 buckets).
#[inline]
fn nzb(n: usize) -> usize {
    const B: [u8; 65] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 9, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 14, 14,
        14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15,
    ];
    B[n.min(64)] as usize
}

/// Per-image neighbour state for one component: the previous and current block rows (zigzag
/// coefficients and nonzero counts).
struct Rows {
    prev: Vec<[i16; 64]>,
    cur: Vec<[i16; 64]>,
    nz_prev: Vec<u8>,
    nz_cur: Vec<u8>,
    /// Nonzero counts of the edge coefficients (first row and column).
    ne_prev: Vec<u8>,
    ne_cur: Vec<u8>,
    /// DC prediction residual magnitudes (bit lengths) of the previous / current row.
    r_prev: Vec<u8>,
    r_cur: Vec<u8>,
    /// Full pixels (×2^24) of each block's two bottom rows (previous / current block row).
    bot_prev: Vec<[[i64; 8]; 2]>,
    bot_cur: Vec<[[i64; 8]; 2]>,
    /// Full pixels of the previous block's two right columns.
    right: [[i64; 8]; 2],
}

impl Rows {
    fn new(w: usize) -> Rows {
        Rows {
            prev: vec![[0; 64]; w], cur: vec![[0; 64]; w], nz_prev: vec![0; w], nz_cur: vec![0; w], ne_prev: vec![0; w], ne_cur: vec![0; w],
            r_prev: vec![0; w], r_cur: vec![0; w],
            bot_prev: vec![[[0; 8]; 2]; w], bot_cur: vec![[[0; 8]; 2]; w], right: [[0; 8]; 2],
        }
    }
    fn next_row(&mut self) {
        std::mem::swap(&mut self.prev, &mut self.cur);
        std::mem::swap(&mut self.nz_prev, &mut self.nz_cur);
        std::mem::swap(&mut self.ne_prev, &mut self.ne_cur);
        std::mem::swap(&mut self.r_prev, &mut self.r_cur);
        std::mem::swap(&mut self.bot_prev, &mut self.bot_cur);
    }
}

pub struct JpegModel {
    sm: Vec<StateMap>, // FAMILIES × INPUTS
    mixer: Mixer,
    apm: Apm,
    ctx: [usize; INPUTS],
    fam: usize,
    /// Developer statistics: coding cost per family (bits·65536), when enabled.
    pub stats: Option<[u64; FAMILIES]>,
}

/// Names of the decision families, for statistics.
pub const FAMILY_NAMES: [&str; FAMILIES] = ["nz-inner", "ac-zero", "ac-exp", "ac-sign", "ac-mant", "dc-zero", "dc-exp", "dc-sign", "dc-mant", "nz-edge"];

impl Default for JpegModel {
    fn default() -> Self { Self::new() }
}

impl JpegModel {
    pub fn new() -> Self {
        JpegModel {
            sm: (0..FAMILIES * INPUTS).map(|_| StateMap::new(TABLE, 255)).collect(),
            mixer: Mixer::new(INPUTS + 1, &[FAMILIES * 16], 1, 6, 6),
            apm: Apm::new(FAMILIES * 64, 7),
            ctx: [0; INPUTS],
            fam: 0,
            stats: None,
        }
    }

    /// Code one binary decision of family `fam` with the given context indices (each below
    /// TABLE), mixer band `band` (< 16) and APM context `actx` (< 64).
    #[inline]
    fn bit<C: Coder>(&mut self, c: &mut C, bit: u32, fam: usize, ctx: [usize; INPUTS], band: usize, actx: usize) -> u32 {
        self.fam = fam;
        self.ctx = ctx;
        for i in 0..INPUTS {
            let p = self.sm[fam * INPUTS + i].p_at(ctx[i] & (TABLE - 1));
            self.mixer.add(stretch(p));
        }
        self.mixer.add(256);
        self.mixer.set(0, fam * 16 + band.min(15));
        let pm = self.mixer.p();
        let pa = self.apm.p(pm, fam * 64 + actx.min(63));
        let p = ((pm + 3 * pa) / 4).clamp(1, 65535);
        let b = c.code(bit, p);
        if let Some(st) = self.stats.as_mut() {
            let pb = if b == 1 { p } else { 65536 - p } as u64;
            st[fam] += (16 << 16) - crate::nsc3::block::log2_fx(pb);
        }
        for i in 0..INPUTS { self.sm[fam * INPUTS + i].update_at(ctx[i] & (TABLE - 1), b); }
        self.mixer.update(b);
        self.apm.update(b);
        b
    }

    /// Code a magnitude ≥ 1 as its bit length (unary, `base` families) then the bits below the
    /// leading one. Returns the magnitude.
    #[allow(clippy::too_many_arguments)]
    fn magnitude<C: Coder>(&mut self, c: &mut C, v: u32, f_exp: usize, f_mant: usize, ctx: [usize; INPUTS], band: usize, actx: usize) -> u32 {
        let e = lg(v);
        let mut got = 1usize;
        while got < 15 {
            let more = self.bit(c, (e > got) as u32, f_exp, ctx.map(|x| x * 16 + got), band, actx.wrapping_mul(4).wrapping_add(got.min(3)));
            if more == 0 { break; }
            got += 1;
        }
        let mut m = 1u32;
        for i in (0..got.saturating_sub(1)).rev() {
            let bitv = (v >> i) & 1;
            let top = (got - 2 - i).min(2); // position below the leading one: 0, 1, 2+
            let b = self.bit(c, bitv, f_mant, [got * 16 + i, (got * 16 + i) * 4 + top, (band.min(15) * 16 + got) * 16 + i, (actx.min(15) * 16 + got) * 16 + i,
                ((ctx[4] % 4096) * 16 + got) * 16 + i], got.min(15), top);
            m = (m << 1) | b;
        }
        m
    }

    /// Code (encode, or decode when `buf` is the decoder's) one coefficient leaf in `buf[range]`.
    pub fn code_leaf<C: Coder, B: ByteBuf>(&mut self, c: &mut C, buf: &mut B, range: std::ops::Range<usize>, params: &[u8]) -> Result<(), String> {
        let g = parse_params(params)?;
        let enc = buf.reading();
        let mut pos = range.start;
        let mut luma_nz: Vec<u8> = vec![];
        let luma_w = g.comps[0].0;
        let (hmax, vmax) = (g.comps[0].0, g.comps[0].1);
        for (ci, &(bw, bh)) in g.comps.iter().enumerate() {
            let cls = (ci > 0) as usize;
            let q = g.quant[ci];
            let unit = q[0] as i64 * DC_UNIT;
            let mut rows = Rows::new(bw);
            for by in 0..bh {
                for bx in 0..bw {
                    // the block's coefficients (zigzag order): parsed when encoding
                    let mut blk = [0i16; 64];
                    let mut parse_end = 0usize;
                    if enc {
                        let n = buf.byte(pos) as usize;
                        if n > 64 { return Err("jpeg coefficients: bad block".into()); }
                        let mut o = pos + 1;
                        for z in blk.iter_mut().take(n) {
                            let mut v = 0u32;
                            for sh in [0, 7, 14] {
                                if o >= range.end { return Err("jpeg coefficients: truncated".into()); }
                                let x = buf.byte(o);
                                o += 1;
                                v |= ((x & 0x7f) as u32) << sh;
                                if x & 0x80 == 0 { break; }
                            }
                            *z = ((v >> 1) as i32 ^ -((v & 1) as i32)) as i16;
                        }
                        parse_end = o;
                    }
                    let (above, left) = (by > 0, bx > 0);
                    let a = if above { rows.prev[bx] } else { [0; 64] };
                    let l = if left { rows.cur[bx - 1] } else { [0; 64] };
                    let al = if above && left { rows.prev[bx - 1] } else { [0; 64] };
                    let nza = if above { rows.nz_prev[bx] as usize } else { 0 };
                    let nzl = if left { rows.nz_cur[bx - 1] as usize } else { 0 };
                    let nzpred = match (above, left) { (true, true) => (nza + nzl + 1) / 2, (true, false) => nza, (false, true) => nzl, _ => 0 };
                    // co-located luma block for chroma planes
                    let nzy = if ci > 0 && !luma_nz.is_empty() {
                        let ly = (by * vmax / bh.max(1)).min(luma_nz.len() / luma_w - 1);
                        let lx = (bx * hmax / bw.max(1)).min(luma_w - 1);
                        luma_nz[ly * luma_w + lx] as usize
                    } else { nzpred };

                    // 1. nonzero count of the 7×7 interior
                    let nz_true = ORDER[..49].iter().filter(|&&k| blk[k] != 0).count();
                    let mut node = 1usize;
                    let nzar = if above && bx + 1 < bw { rows.nz_prev[bx + 1] as usize } else { nza };
                    for i in (0..6).rev() {
                        let b = self.bit(c, ((nz_true >> i) & 1) as u32, F_NZ,
                            [(cls * 16 + nzb(nzpred)) * 64 + node, ((cls * 16 + nzb(nza)) * 16 + nzb(nzl)) * 64 + node,
                             (cls * 16 + nzb(nzy)) * 64 + node, (cls * 16 + nzb(nza.max(nzl))) * 64 + node,
                             ((cls * 16 + nzb((nza + nzl + nzar + 1) / 3)) * 16 + nzb(nzar)) * 64 + node],
                            node.min(15), nzb(nzpred) * 4 + (5 - i).min(3));
                        node = node * 2 + b as usize;
                    }
                    let nz_int = node - 64;
                    if nz_int > 49 { return Err("jpeg coefficients: bad count".into()); }
                    let mut nz = nz_int;

                    // 2. AC coefficients: interior in zigzag order, then the predicted edges
                    let mut left_nz = nz_int;
                    let mut coded_nz = 0usize;
                    let mut prev_mag = 0u32;
                    let mut edge_pred = [0i32; 64]; // predicted value per zigzag index (edges)
                    for (i, &k) in ORDER.iter().enumerate() {
                        if i == 49 {
                            // interior done: predict the first row from the upper block and the
                            // first column from the left block (linear extrapolation of the
                            // neighbour's boundary, minus this block's interior contribution)
                            let mut fb = [0i64; 64];
                            for &kk in &ORDER[..49] { let p = ZIGZAG[kk]; fb[p] = blk[kk] as i64 * q[p] as i64; }
                            if above {
                                let mut fa = [0i64; 64];
                                for kk in 0..64 { let p = ZIGZAG[kk]; fa[p] = a[kk] as i64 * q[p] as i64; }
                                for u in 1..8 {
                                    let (mut s7, mut s6, mut own) = (0i64, 0i64, 0i64);
                                    for v in 0..8 { s7 += fa[v * 8 + u] * IDCT[v][7]; s6 += fa[v * 8 + u] * IDCT[v][6]; }
                                    for v in 1..8 { own += fb[v * 8 + u] * IDCT[v][0]; }
                                    let pv = div_round(2 * s7 - s6 - own, IDCT[0][0] * q[u] as i64);
                                    edge_pred[zz_index(u)] = pv.clamp(-2047, 2047) as i32;
                                }
                            }
                            if left {
                                let mut fl = [0i64; 64];
                                for kk in 0..64 { let p = ZIGZAG[kk]; fl[p] = l[kk] as i64 * q[p] as i64; }
                                for v in 1..8 {
                                    let (mut s7, mut s6, mut own) = (0i64, 0i64, 0i64);
                                    for u in 0..8 { s7 += fl[v * 8 + u] * IDCT[u][7]; s6 += fl[v * 8 + u] * IDCT[u][6]; }
                                    for u in 1..8 { own += fb[v * 8 + u] * IDCT[u][0]; }
                                    let pv = div_round(2 * s7 - s6 - own, IDCT[0][0] * q[v * 8] as i64);
                                    edge_pred[zz_index(v * 8)] = pv.clamp(-2047, 2047) as i32;
                                }
                            }
                            // edges: zero flags only (no count)
                            left_nz = usize::MAX;
                        }
                        if left_nz == 0 { if !enc { blk[k] = 0; } continue; }
                        let remaining = if i < 49 { 49 - i } else { usize::MAX - 1 };
                        let ne_so_far = if i >= 49 { coded_nz - (nz_int - 0) } else { 0 };
                        let pos_n = ZIGZAG[k];
                        let edge = i >= 49;
                        let ma = a[k].unsigned_abs() as u32;
                        let ml = l[k].unsigned_abs() as u32;
                        let mal = al[k].unsigned_abs() as u32;
                        let pm = (2 * (ma + ml) + mal + 2) / 5; // neighbour-predicted magnitude
                        let (ba, bl) = (lg(ma).min(6), lg(ml).min(6));
                        let kb = k.min(63);
                        let pe = edge_pred[k];
                        let bpe = lg(pe.unsigned_abs()).min(11);
                        // the fifth context: edge prediction for edges, neighbour sum inside
                        let c5 = if edge { 4096 + ((cls * 14 + (i - 49)) * 12 + bpe) * 3 + (pe.signum() + 1) as usize } else { (cls * 64 + kb) * 12 + lg(ma + ml).min(11) };
                        let v = blk[k] as i32;
                        let lz = if edge { (16 + nzb(nz_int)).min(31) } else { left_nz.min(15) };
                        let nonzero = if left_nz == remaining { 1 } else {
                            self.bit(c, (v != 0) as u32, F_ZF,
                                [(cls * 64 + kb) * 32 + lz, ((cls * 64 + kb) * 7 + ba) * 7 + bl,
                                 ((cls * 64 + kb) * 8 + lg(prev_mag).min(7)) * 8 + if edge { ne_so_far.min(7) } else { coded_nz.min(7) },
                                 if edge { 2048 + ((cls * 14 + (i - 49)) * 12 + bpe) * 4 + ne_so_far.min(3) } else { (cls * 32 + left_nz.min(31)) * 16 + lg(pm).min(15) }, c5],
                                if edge { 12 + (i - 49) / 5 } else { kb / 6 }, if edge { 48 + bpe.min(15) } else { lg(pm).min(5) * 8 + left_nz.min(7) })
                        };
                        if nonzero == 0 { if !enc { blk[k] = 0; } prev_mag = 0; continue; }
                        let mag = self.magnitude(c, v.unsigned_abs(), F_EXP, F_MANT,
                            [(cls * 64 + kb) * 12 + lg(pm).min(11), (cls * 64 + kb) * 16 + left_nz.min(15),
                             (cls * 64 + kb) * 8 + lg(prev_mag).min(7), ((cls * 7 + ba) * 7 + bl) * 4 + (kb / 16), c5 / 3],
                            if edge { 12 + (i - 49) / 5 } else { kb / 6 }, if edge { bpe.min(15) } else { lg(pm).min(15) });
                        let sa = (a[k].signum() + 1) as usize;
                        let sl = (l[k].signum() + 1) as usize;
                        let spe = (pe.signum() + 1) as usize;
                        let neg = self.bit(c, (v < 0) as u32, F_SIGN,
                            [((cls * 64 + kb) * 3 + sa) * 3 + sl, cls * 64 + kb, ((cls * 3 + sa) * 3 + sl) * 8 + lg(mag).min(7),
                             (cls * 64 + pos_n) * 2 + (prev_mag > 0) as usize, ((cls * 64 + kb) * 3 + spe) * 12 + bpe],
                            if edge { 12 + spe } else { kb / 6 }, if edge { spe * 16 + bpe } else { sa * 3 + sl });
                        let val = if neg == 1 { -(mag as i32) } else { mag as i32 };
                        if !enc { blk[k] = val as i16; }
                        prev_mag = mag;
                        coded_nz += 1;
                        if !edge { left_nz -= 1; } else { nz += 1; }
                    }
                    rows.ne_cur[bx] = (nz - nz_int) as u8;
                    if enc && blk[1..].iter().filter(|&&v| v != 0).count() != nz { return Err("jpeg coefficients: count mismatch".into()); }

                    // 3. DC: residual from the pixel-domain edge-continuity prediction
                    let (da, dl, dal) = (a[0] as i32, l[0] as i32, al[0] as i32);
                    let med = match (above, left) {
                        (true, true) => {
                            let (mn, mx) = (da.min(dl), da.max(dl));
                            if dal >= mx { mn } else if dal <= mn { mx } else { da + dl - dal }
                        }
                        (true, false) => da,
                        (false, true) => dl,
                        _ => 0,
                    };
                    let mut f = [0i64; 64];
                    for k in 1..64 { f[ZIGZAG[k]] = blk[k] as i64 * q[ZIGZAG[k]] as i64; }
                    let (c0, c1, r0, r1) = (col_pixels(&f, 0), col_pixels(&f, 1), row_pixels(&f, 0), row_pixels(&f, 1));
                    let mut ds = [0i64; 16];
                    let mut nd = 0;
                    if left {
                        let [l6, l7] = rows.right;
                        for y in 0..8 { ds[nd] = l7[y] - c0[y] + ((l7[y] - l6[y]) + (c1[y] - c0[y])) / 2; nd += 1; }
                    }
                    if above {
                        let [a6, a7] = rows.bot_prev[bx];
                        for x in 0..8 { ds[nd] = a7[x] - r0[x] + ((a7[x] - a6[x]) + (r1[x] - r0[x])) / 2; nd += 1; }
                    }
                    let (pred, act) = if nd == 0 { (med, 15) } else {
                        let ds = &ds[..nd];
                        let (mn, mx) = (*ds.iter().min().unwrap(), *ds.iter().max().unwrap());
                        let p = div_round(ds.iter().sum::<i64>() / nd as i64, unit);
                        (p.clamp(-32768, 32767) as i32, lg(((mx - mn) / unit).min(1 << 20) as u32).min(15))
                    };
                    let dev = lg((pred - med).unsigned_abs()).min(7);
                    let (ra, rl) = (if above { rows.r_prev[bx] as usize } else { 0 }, if left { rows.r_cur[bx - 1] as usize } else { 0 });
                    let r = blk[0] as i32 - pred;
                    let nzc = nzb(nz);
                    let dctx = [(cls * 16 + act) * 8 + dev, (cls * 16 + ra.min(15)) * 16 + rl.min(15), (cls * 16 + act) * 16 + nzc,
                                (cls * 16 + (ra + rl).min(15)) * 16 + act, (cls * 16 + lg(pred.unsigned_abs()).min(15)) * 16 + act];
                    let nonzero = self.bit(c, (r != 0) as u32, F_DCZ, dctx, act.min(15), act * 4 + nzc.min(3));
                    let mut dcr = 0i32;
                    if nonzero == 1 {
                        let mag = self.magnitude(c, r.unsigned_abs(), F_DCEXP, F_DCMANT, dctx, act.min(15), act);
                        let toward = (med > pred) as usize; // the median predictor's side
                        let neg = self.bit(c, (r < 0) as u32, F_DCSIGN,
                            [cls * 2 + toward, (cls * 8 + dev) * 2 + toward, (cls * 16 + act) * 2 + toward, (cls * 16 + lg(mag).min(15)) * 2 + toward,
                             ((cls * 8 + dev) * 16 + act) * 2 + toward], dev, toward);
                        dcr = if neg == 1 { -(mag as i32) } else { mag as i32 };
                    }
                    let dc = pred + dcr;
                    if !(-32768..=32767).contains(&dc) { return Err("jpeg coefficients: DC out of range".into()); }
                    if !enc { blk[0] = dc as i16; }

                    // serialize (decoder) / check position (encoder)
                    let n = (0..64).rev().find(|&k| blk[k] != 0).map(|k| k + 1).unwrap_or(0);
                    let mut o = pos;
                    let mut put = |buf: &mut B, x: u8| -> Result<(), String> {
                        if o >= range.end { return Err("jpeg coefficients: leaf overflow".into()); }
                        buf.put(o, x);
                        o += 1;
                        Ok(())
                    };
                    put(buf, n as u8)?;
                    for &v in blk.iter().take(n) {
                        let mut z = ((v as i32) << 1 ^ ((v as i32) >> 31)) as u32;
                        while z >= 0x80 { put(buf, z as u8 | 0x80)?; z >>= 7; }
                        put(buf, z as u8)?;
                    }
                    if enc && o != parse_end { return Err("jpeg coefficients: non-canonical block".into()); }
                    pos = o;
                    // full edge pixels for the right and lower neighbours
                    let dcp = blk[0] as i64 * unit;
                    let (mut c6, mut c7, mut r6, mut r7) = (col_pixels(&f, 6), col_pixels(&f, 7), row_pixels(&f, 6), row_pixels(&f, 7));
                    for v in c6.iter_mut().chain(c7.iter_mut()).chain(r6.iter_mut()).chain(r7.iter_mut()) { *v += dcp; }
                    rows.right = [c6, c7];
                    rows.bot_cur[bx] = [r6, r7];
                    rows.cur[bx] = blk;
                    rows.nz_cur[bx] = nz_int as u8;
                    rows.r_cur[bx] = lg(dcr.unsigned_abs()) as u8;
                }
                if ci == 0 { luma_nz.extend_from_slice(&rows.nz_cur); }
                rows.next_row();
            }
        }
        if pos != range.end { return Err("jpeg coefficients: leaf length mismatch".into()); }
        Ok(())
    }
}
