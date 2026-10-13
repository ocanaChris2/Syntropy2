//! The model of H.264 slice data: it codes the bins of CABAC slices (`recode::h264`) with
//! probabilities drawn from far more context than CABAC's own 460 adaptive states.
//!
//! CABAC picks a context from the syntax element and a little of its neighbourhood, and adapts
//! one probability per context. Here every bin gets several predictions — from CABAC's context,
//! from what the walker tells about the element (neighbouring coefficient counts, intra modes,
//! motion differences, quantisers), and from what the model itself remembers (the macroblocks
//! around it, the same macroblock in the pictures before, the block being coded) — mixed online.
//! The bins and their order stay CABAC's, so the slice re-encodes exactly.
use super::binary::{ql, Bins as Core};
use super::coder::Coder;
use crate::bits::{BitReader, BitWriter};
use super::predictor::ByteBuf;
use crate::recode::h264::{self, class, se, Info, Params, Pps, Scratch, SliceHeader, Sps, SLICE_B, TERMINATE};

const MAXIN: usize = 11;
/// Families: the syntax elements (`se`), then record fields and raw bytes.
const F_REC: usize = se::COUNT;
const F_RAW: usize = se::COUNT + 1;
pub const FAMILIES: usize = se::COUNT + 2;
const BITS: [u32; FAMILIES] = [
    14, // mb skip
    14, // mb type
    12, // sub mb type
    14, // ref idx
    16, // mvd
    12, // mvd suffix
    14, // mvd sign
    16, // qp delta
    14, // intra pred flag
    16, // intra pred mode
    12, // chroma pred
    16, // cbp luma
    14, // cbp chroma
    12, // transform 8x8
    16, // coded block flag
    16, // significance
    16, // last
    16, // level
    12, // level suffix
    16, // sign
    10, // end of slice
    14, // records
    16, // raw bytes
];
/// Count limit of the probability maps: video statistics drift, so they keep adapting.
const LIMIT: u32 = 127;
/// Context inputs per family in the full profile, and in the lean one (the first of the same
/// list): about half the inputs for about 0.4% more output, at nearly twice the speed.
pub const NIN: [usize; FAMILIES] = [6, 6, 5, 6, 7, 5, 5, 7, 6, 4, 4, 6, 6, 3, 8, 11, 11, 10, 3, 6, 3, 6, 6];
pub const NIN_LEAN: [usize; FAMILIES] = [5, 4, 4, 4, 4, 2, 3, 4, 2, 3, 2, 3, 4, 2, 5, 5, 6, 6, 3, 3, 3, 6, 6];

/// Blocks per macroblock in the coefficient-count memory: 16 luma 4x4 (raster), 8 chroma AC,
/// luma DC, 2 chroma DC.
const NB: usize = 27;

/// Raster position (u + 4v) of the 4x4 zigzag scan positions; the 8x8 zigzag (u + 8v).
const ZZ4: [u8; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
const ZZ8: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29,
    22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Per scan position of an `n`x`n` zigzag: the scan positions of the coefficients left of, above,
/// right of and below it in the block (64: none).
const fn neighbours<const N: usize>(zz: &[u8; N], n: usize) -> [[u8; 4]; N] {
    let mut inv = [0u8; N];
    let mut s = 0;
    while s < N {
        inv[zz[s] as usize] = s as u8;
        s += 1;
    }
    let mut out = [[64u8; 4]; N];
    let mut s = 0;
    while s < N {
        let (u, v) = (zz[s] as usize % n, zz[s] as usize / n);
        if u > 0 { out[s][0] = inv[u - 1 + n * v]; }
        if v > 0 { out[s][1] = inv[u + n * (v - 1)]; }
        if u + 1 < n { out[s][2] = inv[u + 1 + n * v]; }
        if v + 1 < n { out[s][3] = inv[u + n * (v + 1)]; }
        s += 1;
    }
    out
}
/// Per scan position: the scan positions of the lower-frequency coefficients around it (left,
/// above, above left, two left, two above) as a mask.
const fn template<const N: usize>(zz: &[u8; N], n: usize) -> [u64; N] {
    let mut inv = [0u8; N];
    let mut s = 0;
    while s < N {
        inv[zz[s] as usize] = s as u8;
        s += 1;
    }
    let mut out = [0u64; N];
    let mut s = 0;
    while s < N {
        let (u, v) = (zz[s] as usize % n, zz[s] as usize / n);
        if u > 0 { out[s] |= 1 << inv[u - 1 + n * v]; }
        if v > 0 { out[s] |= 1 << inv[u + n * (v - 1)]; }
        if u > 0 && v > 0 { out[s] |= 1 << inv[u - 1 + n * (v - 1)]; }
        if u > 1 { out[s] |= 1 << inv[u - 2 + n * v]; }
        if v > 1 { out[s] |= 1 << inv[u + n * (v - 2)]; }
        s += 1;
    }
    out
}
static TPL4: [u64; 16] = template(&ZZ4, 4);
static TPL8: [u64; 64] = template(&ZZ8, 8);
static NEAR4: [[u8; 4]; 16] = neighbours(&ZZ4, 4);
static NEAR8: [[u8; 4]; 64] = neighbours(&ZZ8, 8);
/// The 2x2 chroma DC block is scanned in raster order.
static NEAR2: [[u8; 4]; 4] = [[64, 64, 1, 2], [0, 64, 64, 3], [64, 0, 3, 64], [2, 1, 64, 64]];

/// What the model remembers of a macroblock.
#[derive(Clone, Copy, Default)]
struct M {
    /// Class + 1 (0: nothing known).
    k: u8,
    /// Coded block pattern: luma bits 0..4, chroma in bits 4..6.
    cbp: u8,
    qp: u8,
    dqp: i8,
    t8: u8,
    /// Class of the motion vector differences' sum.
    mv: u8,
    coefs: u16,
}

impl M {
    /// Stored as a number, so a picture's worth of them is allocated zeroed (and costs nothing
    /// until used).
    fn pack(self) -> u64 {
        self.k as u64 | (self.cbp as u64) << 8 | (self.qp as u64) << 16 | (self.dqp as u8 as u64) << 24 | (self.t8 as u64) << 32 | (self.mv as u64) << 40 | (self.coefs as u64) << 48
    }
    fn unpack(v: u64) -> M {
        M { k: v as u8, cbp: (v >> 8) as u8, qp: (v >> 16) as u8, dqp: (v >> 24) as u8 as i8, t8: (v >> 32) as u8, mv: (v >> 40) as u8, coefs: (v >> 48) as u16 }
    }
}

pub struct H264Model {
    b: Core,
    /// Per (slice type, context): the last bins coded in it.
    hist: Vec<u8>,
    // the slice
    st: u8,
    w: usize,
    qp: i32,
    // the pictures: the one being coded (entries not coded yet: two pictures back), and the last
    cur: Vec<u64>,
    prev: Vec<u64>,
    curnz: Vec<u8>,
    prevnz: Vec<u8>,
    /// Per block: which scan positions hold a coefficient, a negative one, one above 1 (an 8x8
    /// block's 64 positions take its four slots).
    cursig: Vec<[u16; 3]>,
    prevsig: Vec<[u16; 3]>,
    /// Where slices ended in the last picture, and in this one so far.
    prev_ends: Vec<u32>,
    cur_ends: Vec<u32>,
    // the macroblock: its neighbours (left, top, top left, top right), the one at its place in
    // the last picture, and in the one before
    addr: usize,
    l: M,
    t: M,
    tl: M,
    tr: M,
    p: M,
    o: M,
    skiprun: u32,
    /// Bins of the macroblock and sub-macroblock types so far.
    tybits: u32,
    lastref: u32,
    lastmode: u32,
    rem: u32,
    mvsum: u32,
    /// The x component just coded of the partition's motion vector difference, and its sign.
    mvx: u32,
    mvxs: u32,
    /// The motion vector difference component being coded: its bins so far; the bins of each
    /// candidate value (left-aligned) and their number (0: no candidate); which candidates the
    /// x component was equal to (bits).
    mvb: u64,
    mvn: u32,
    cand: [(u64, u32); 4],
    candx: [i16; 4],
    hitx: u64,
    mbcoefs: u32,
    qpv: u32,
    // the block
    bi: usize,
    sig: u64,
    mag: [u8; 65],
    cnt: u32,
    msum: u32,
    prevmag: u32,
    lastsign: u32,
    /// Negative coefficients of the block so far.
    neg: u64,
    /// Per plane and scan position (the first 16): the balance of the signs coded lately.
    bal: [[i16; 16]; 3],
    /// The maps (significant, negative, above 1) of the blocks left of and above the block, of
    /// the block at its place in the last picture, and of the block of the other chroma plane.
    side: [[u64; 3]; 4],
    /// Intra prediction mode per 4x4 block of the macroblock (raster; 15: none).
    modes: [u8; 16],
    /// Coefficients in the blocks left of and above the block (classes).
    near: [u64; 2],
    /// The last record's tag, and the last slice record's NAL header byte, header length and
    /// alignment bits (contexts for the next).
    rec: [u8; 4],
    /// The last slice header, the last parameter sets (sequence, picture), the last numbers.
    last_hdr: Vec<u8>,
    last_ps: [Vec<u8>; 2],
    last_var: [[u8; 5]; 5],
    scratch: Scratch,
    /// The full profile: all context inputs (else the lean set).
    full: bool,
    /// Developer ablation: the context inputs (family, index) to blank.
    pub drop: Vec<(usize, usize)>,
}


/// The bins of a motion vector difference `v` as CABAC writes them — its magnitude in unary up to
/// 9, the rest in exp-Golomb code (k = 3), its sign — left-aligned, and how many (none for
/// [`h264::NO_MV`] or a value too large to matter).
fn mvd_bins(v: i16) -> (u64, u32) {
    let a = v.unsigned_abs() as u64;
    if v == h264::NO_MV || a > 4000 { return (0, 0); }
    let (mut bits, mut n) = (0u64, 0u32);
    let mut put = |b: u64, k: u32| { bits = bits << k | b; n += k; };
    put((1 << a.min(9)) - 1, a.min(9) as u32);
    if a < 9 {
        put(0, 1);
    } else {
        let (mut s, mut k) = (a - 9, 3u32);
        while s >= 1 << k {
            put(1, 1);
            s -= 1 << k;
            k += 1;
        }
        put(0, 1);
        put(s, k);
    }
    if a != 0 { put((v < 0) as u64, 1); }
    (bits << (64 - n), n)
}

/// How far apart a partition's candidate motion vector differences (of its neighbours) are, and
/// how far one is from the prediction: classes, 15 for none.
#[inline]
fn spread(x: &[i16; 4]) -> u64 { x[..3].iter().filter(|&&v| v != h264::NO_MV).map(|v| v.unsigned_abs() as u32).max().map_or(15, |v| ql(v).min(14)) }

#[inline]
fn far(v: i16) -> u64 { if v == h264::NO_MV { 15 } else { ql(v.unsigned_abs() as u32).min(14) } }

/// 0, 1, 2, 3-4, 5-8, 9-16, 17+: classes of a block's coefficient count.
#[inline]
fn qc(x: u32) -> u64 {
    match x {
        0..=2 => x as u64,
        3..=4 => 3,
        5..=8 => 4,
        9..=16 => 5,
        _ => 6,
    }
}

impl H264Model {
    /// A model of the full profile (every context input) or the lean one.
    pub fn new(full: bool) -> Self {
        H264Model {
            b: Core::adapting(&BITS, if full { &NIN } else { &NIN_LEAN }, LIMIT, true), full, hist: vec![0; 3 * 1024], st: 0, w: 1, qp: 26, cur: vec![], prev: vec![], curnz: vec![], prevnz: vec![],
            cursig: vec![], prevsig: vec![], side: [[0; 3]; 4], neg: 0, bal: [[0; 16]; 3], modes: [15; 16], near: [0; 2],
            prev_ends: vec![], cur_ends: vec![], addr: usize::MAX, l: M::default(), t: M::default(), tl: M::default(), tr: M::default(),
            p: M::default(), o: M::default(), skiprun: 0, tybits: 1, lastref: 0, lastmode: 0, rem: 0, mvsum: 0, mvx: 0, mvxs: 0, mvb: 0, mvn: 0, cand: [(0, 0); 4], candx: [h264::NO_MV; 4], hitx: 0, mbcoefs: 0, qpv: 0,
            bi: 0, sig: 0, mag: [0; 65], cnt: 0, msum: 0, prevmag: 0, lastsign: 0, rec: [0; 4], last_hdr: vec![], last_ps: [vec![], vec![]],
            last_var: [[0; 5]; 5], scratch: Scratch::default(), drop: vec![],
        }
    }

    /// Is this the full profile's model?
    pub fn full(&self) -> bool { self.full }

    pub fn enable_stats(&mut self) { self.b.stats = Some(vec![0; FAMILIES]); }

    /// Bits·65536 spent so far, per family (zeros without statistics).
    pub fn stats(&self) -> Vec<u64> { self.b.stats.clone().unwrap_or_else(|| vec![0; FAMILIES]) }

    /// A slice of a picture `sps` describes begins.
    fn begin_slice(&mut self, h: &SliceHeader, sps: &Sps) {
        let total = sps.width_mbs * sps.height_mbs;
        if self.cur.len() != total || self.w != sps.width_mbs {
            self.w = sps.width_mbs;
            self.cur = vec![0; total];
            self.prev = vec![0; total];
            self.curnz = vec![0; total * NB];
            self.prevnz = vec![0; total * NB];
            self.cursig = vec![[0; 3]; total * NB];
            self.prevsig = vec![[0; 3]; total * NB];
            self.prev_ends.clear();
            self.cur_ends.clear();
        } else if h.first_mb == 0 {
            // a new picture
            std::mem::swap(&mut self.cur, &mut self.prev);
            std::mem::swap(&mut self.curnz, &mut self.prevnz);
            std::mem::swap(&mut self.cursig, &mut self.prevsig);
            std::mem::swap(&mut self.cur_ends, &mut self.prev_ends);
            self.cur_ends.clear();
        }
        self.st = h.slice_type;
        self.qp = h.qp;
        self.addr = usize::MAX;
        self.skiprun = 0;
    }

    fn enter_mb(&mut self, addr: usize) {
        let (w, x) = (self.w, addr % self.w);
        let none = M::default();
        self.addr = addr;
        self.l = if x > 0 { M::unpack(self.cur[addr - 1]) } else { none };
        self.t = if addr >= w { M::unpack(self.cur[addr - w]) } else { none };
        self.tl = if x > 0 && addr >= w { M::unpack(self.cur[addr - w - 1]) } else { none };
        self.tr = if x + 1 < w && addr >= w { M::unpack(self.cur[addr - w + 1]) } else { none };
        self.p = M::unpack(self.prev[addr]);
        self.o = M::unpack(self.cur[addr]);
        self.curnz[addr * NB..(addr + 1) * NB].fill(0);
        self.cursig[addr * NB..(addr + 1) * NB].fill([0; 3]);
        self.tybits = 1;
        self.lastref = 0;
        self.lastmode = 2;
        self.modes = [15; 16];
        self.mvsum = 0;
        self.mbcoefs = 0;
        self.qpv = 0;
    }

    /// The state of the candidates of the motion vector difference component being coded, given
    /// its bins so far: per candidate (a base-3 digit) 0 (none, or ruled out), 1 (its next bin is
    /// 0) or 2 (is 1).
    #[inline]
    fn cands(&self) -> u64 {
        let mut out = 0u64;
        for &(bits, len) in self.cand.iter().rev() {
            let n = self.mvn;
            out = out * 3 + if len > n && (n == 0 || bits >> (64 - n) == self.mvb) { 1 + (bits >> (63 - n) & 1) } else { 0 };
        }
        out
    }

    /// Remember the prediction mode just coded (`lastmode`) for block `a & 15` (in coding order;
    /// bit 4: an 8x8 block) of the macroblock.
    fn set_mode(&mut self, a: u64) {
        const BLK: [usize; 16] = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];
        let (k, m) = ((a & 15) as usize, self.lastmode.min(15) as u8);
        if a >> 4 & 1 == 1 {
            let p = BLK[(k & 3) * 4];
            for q in [p, p + 1, p + 4, p + 5] { self.modes[q] = m; }
        } else {
            self.modes[BLK[k]] = m;
        }
    }

    /// The maps of the block in `slot` of memory `m` (`wide`: an 8x8 block, in four slots).
    fn maps(m: &[[u16; 3]], slot: Option<usize>, wide: bool) -> [u64; 3] {
        let Some(s) = slot else { return [0; 3] };
        let mut out = [0u64; 3];
        for (k, o) in out.iter_mut().enumerate() {
            *o = m[s][k] as u64;
            if wide { *o |= (m[s + 1][k] as u64) << 16 | (m[s + 4][k] as u64) << 32 | (m[s + 5][k] as u64) << 48; }
        }
        out
    }

    fn new_block(&mut self, cat: usize, blk: usize) {
        self.bi = match cat {
            0 => 24,
            1 | 2 => blk & 15,
            3 => 25 + (blk & 1),
            4 => 16 + (blk & 7),
            _ => (blk & 1) * 2 + (blk >> 1 & 1) * 8,
        };
        // the blocks left of and above it: in this macroblock, or the nearest of the next one
        let (addr, w, bi) = (self.addr, self.w, self.bi);
        let (row, left, top) = (addr * NB, (addr % w > 0).then(|| (addr - 1) * NB), (addr >= w).then(|| (addr - w) * NB));
        let (sa, sb) = match cat {
            1 | 2 => (if bi % 4 > 0 { Some(row + bi - 1) } else { left.map(|r| r + bi + 3) }, if bi >= 4 { Some(row + bi - 4) } else { top.map(|r| r + bi + 12) }),
            5 => (if bi % 4 > 0 { Some(row + bi - 2) } else { left.map(|r| r + bi + 2) }, if bi >= 8 { Some(row + bi - 8) } else { top.map(|r| r + bi + 8) }),
            4 => {
                let k = (bi - 16) % 4;
                (if k % 2 > 0 { Some(row + bi - 1) } else { left.map(|r| r + bi + 1) }, if k >= 2 { Some(row + bi - 2) } else { top.map(|r| r + bi + 2) })
            }
            _ => (left.map(|r| r + bi), top.map(|r| r + bi)),
        };
        let other = match bi { 20..=23 => Some(row + bi - 4), 26 => Some(row + 25), _ => None };
        self.side = [Self::maps(&self.cursig, sa, cat == 5), Self::maps(&self.cursig, sb, cat == 5), Self::maps(&self.prevsig, Some(row + bi), cat == 5),
            Self::maps(&self.cursig, other, false)];
        self.neg = 0;
        self.near = [qc(self.side[0][0].count_ones()), qc(self.side[1][0].count_ones())];
        self.sig = 0;
        self.mag = [0; 65];
        self.cnt = 0;
        self.msum = 0;
        self.prevmag = 0;
        self.lastsign = 0;
    }

    /// Code one bin of the slice data: `ctx` is CABAC's context (or bypass / terminate), `info`
    /// what the walker tells about it. Returns the bin coded.
    #[inline]
    pub fn bin<C: Coder>(&mut self, c: &mut C, bit: u32, ctx: u16, info: &Info) -> u32 {
        let addr = info.addr as usize;
        if addr != self.addr { self.enter_mb(addr); }
        let st = self.st as u64;
        let hi = self.st as usize * 1024 + (ctx as usize).min(1023);
        let hv = self.hist[hi] as u64;
        let key = st << 10 | ctx as u64;
        let (a, b, cc, d) = (info.a as u64, info.b as u64, info.c as u64, info.d as u64);
        let (l, t, p, o) = (self.l, self.t, self.p, self.o);
        let (lk, tk, pk, ok) = (l.k as u64, t.k as u64, p.k as u64, o.k as u64);
        let qp = self.qp.clamp(0, 51) as u64;
        let mut cx = [0u64; MAXIN];
        let fam = info.se as usize;
        let full = self.full;
        let (sel, sel2, actx): (usize, usize, usize);
        // Per element: the contexts every profile uses, then those of the full profile only (the
        // order they were found worth their time in: `bench/h264_prune.py`).
        match info.se {
            se::SKIP => {
                let sk = |m: M| (m.k == class::SKIP + 1) as u64;
                let pat = sk(l) | sk(t) << 1 | sk(self.tl) << 2 | sk(self.tr) << 3 | sk(p) << 4 | sk(o) << 5;
                let run = self.skiprun.min(15) as u64;
                cx[0] = key << 8 | hv;
                cx[1] = st << 12 | pk << 8 | lk << 4 | tk;
                cx[2] = st << 12 | ql(l.coefs as u32) << 8 | ql(t.coefs as u32) << 4 | ql(p.coefs as u32);
                cx[3] = st << 12 | run << 8 | pk << 4 | ok;
                cx[4] = st << 16 | (qp / 3) << 8 | pat;
                if full { cx[5] = st << 8 | lk << 4 | tk; }
                sel = (pat & 3) as usize + 4 * st as usize + 8 * run.min(7) as usize;
                sel2 = pat as usize;
                actx = (pk.min(7) * 8 + lk.min(7)) as usize;
            }
            se::MB_TYPE => {
                let node = key << 10 | a << 5 | b.min(31);
                cx[0] = node << 8 | hv;
                cx[1] = node << 8 | lk << 4 | tk;
                cx[2] = node << 8 | pk << 4 | ok;
                cx[3] = node << 6 | qp / 3;
                if full {
                    cx[4] = node << 4 | pk;
                    cx[5] = node << 12 | (self.tybits as u64 & 0xFFF);
                }
                sel = a.min(31) as usize + 32 * (self.st == SLICE_B) as usize;
                sel2 = (lk.min(7) * 8 + tk.min(7)) as usize;
                actx = (a.min(15) * 4 + b.min(3)) as usize;
            }
            se::SUB_TYPE => {
                let node = key << 8 | a << 4 | b;
                cx[0] = node << 8 | hv;
                cx[1] = node << 16 | (self.tybits as u64 & 0xFFFF);
                cx[2] = node << 8 | lk << 4 | tk;
                cx[3] = node << 4 | pk;
                if full { cx[4] = node; }
                sel = (a * 12 + b.min(11)) as usize;
                sel2 = (lk.min(7) * 8 + tk.min(7)) as usize;
                actx = (b.min(15) * 4 + a) as usize;
            }
            se::REF => {
                let v = b.min(15);
                let node = key << 4 | v;
                cx[0] = node;
                cx[1] = node << 8 | hv;
                cx[2] = v << 12 | cc << 4 | a;
                cx[3] = v << 12 | lk << 8 | tk << 4 | pk;
                if full {
                    cx[4] = v << 12 | d << 8 | a << 4 | self.lastref.min(15) as u64;
                    cx[5] = node << 12 | cc << 4 | self.lastref.min(15) as u64;
                }
                sel = (v.min(7) * 8 + (a & 7)) as usize;
                sel2 = (d.min(15) * 4 + (cc >> 4).min(3)) as usize;
                actx = (v.min(7) * 8 + (cc & 7)) as usize;
            }
            se::MVD => {
                let (comp, size) = (a & 1, a >> 2 & 7);
                let (am, bm) = (b & 255, cc & 255);
                let sum = (am + bm) as u32;
                let node = comp << 4 | d.min(15);
                let (mx, mxs) = if comp == 1 { (ql(self.mvx), self.mvxs as u64) } else { (0, 3) };
                if d == 0 {
                    // a new component: its candidates' bins
                    if comp == 0 { self.candx = info.x; }
                    self.cand = info.x.map(mvd_bins);
                    (self.mvb, self.mvn) = (0, 0);
                }
                let state = self.cands();
                let hx = if comp == 1 { self.hitx } else { 0 };
                cx[0] = node << 12 | mx << 8 | mxs << 4 | size;
                cx[1] = node << 12 | (p.mv as u64) << 8 | (l.mv as u64) << 4 | t.mv as u64;
                cx[2] = ((node << 7 | state) << 4 | hx) << 3 | size;
                cx[3] = (node << 4 | spread(&info.x)) << 4 | far(info.x[3]);
                if full {
                    cx[4] = (key << 4 | d.min(15)) << 8 | hv;
                    cx[5] = node << 8 | ql(am as u32) << 4 | ql(bm as u32);
                    cx[6] = node << 12 | (sum.min(63) as u64) << 4 | ql(self.mvsum);
                }
                sel = (d.min(8) + 9 * comp + 18 * ql(sum).min(2)) as usize;
                sel2 = (size * 8 + mx.min(7)) as usize;
                actx = (ql(sum) * 4 + d.min(3)) as usize;
            }
            se::MVD_SUF => {
                let comp = a & 1;
                let sum = ((b & 255) + (cc & 255)) as u32;
                let node = comp << 6 | d.min(63);
                let state = self.cands();
                cx[0] = node << 7 | state;
                cx[1] = (node << 4 | spread(&info.x)) << 4 | far(info.x[3]);
                if full {
                    cx[2] = node << 4 | ql(sum);
                    cx[3] = node << 8 | (hv & 15);
                    cx[4] = (d.min(63) << 7 | state) << 4 | if comp == 1 { self.hitx } else { 0 };
                }
                sel = d.min(63) as usize;
                sel2 = (comp * 16 + ql(sum)) as usize;
                actx = d.min(63) as usize;
            }
            se::MVD_SIGN => {
                let (comp, size) = (a & 1, a >> 2 & 7);
                let (sa, sb) = (b >> 8, cc >> 8);
                let mxs = if comp == 1 { self.mvxs as u64 } else { 3 };
                let state = self.cands();
                let hx = if comp == 1 { self.hitx } else { 0 };
                cx[0] = comp << 12 | (a >> 5) << 8 | st << 4 | (a >> 1 & 1);
                cx[1] = comp << 7 | state;
                cx[2] = ((comp << 7 | state) << 4 | hx) << 4 | ql(d as u32);
                if full {
                    cx[3] = comp << 8 | mxs << 4 | size;
                    cx[4] = (comp << 7 | state) << 4 | sa << 2 | sb;
                }
                sel = (comp << 4 | sa << 2 | sb) as usize;
                sel2 = (mxs * 16 + ql(d as u32)) as usize;
                actx = (comp << 4 | sa << 2 | sb) as usize;
            }
            se::QP => {
                let v = (self.qpv as u64).min(15);
                let cls = d >> 8;
                let near = |m: M| if m.k == 0 { 0 } else { (9 + (m.qp as i32 - self.qp).clamp(-8, 8)) as u64 };
                let pd = (p.dqp as i32 + 64) as u64;
                cx[0] = v << 8 | b;
                cx[1] = v << 12 | near(self.tr) << 6 | near(self.tl);
                cx[2] = v << 16 | b << 8 | pd;
                cx[3] = (v << 8 | hv) << 4 | cls;
                if full {
                    cx[4] = key << 4 | v;
                    cx[5] = v << 8 | cc;
                    cx[6] = v << 12 | pd << 4 | pk;
                }
                sel = (v.min(7) * 8 + b.min(66) / 9) as usize;
                sel2 = (v.min(3) * 16 + cls) as usize;
                actx = (v.min(7) * 8 + (cc / 17).min(7)) as usize;
            }
            se::IPRED_FLAG => {
                let (k, t8) = (a & 15, a >> 4 & 1);
                cx[0] = key << 8 | hv;
                cx[1] = (t8 << 4 | k) << 8 | b << 4 | cc;
                if full {
                    cx[2] = key;
                    cx[3] = t8 << 8 | b << 4 | cc;
                    cx[4] = (t8 << 4 | k) << 8 | d << 4 | self.lastmode as u64;
                    cx[5] = (qp / 3) << 8 | t8 << 4 | d;
                }
                sel = (t8 * 32 + k * 2 + (b == cc) as u64) as usize;
                sel2 = (d * 4 + (hv & 3)) as usize;
                actx = (b.min(7) * 8 + cc.min(7)) as usize;
            }
            se::IPRED_REM => {
                let (k, t8, node) = (a & 15, a >> 4 & 1, a >> 5);
                cx[0] = node << 8 | b << 4 | cc;
                cx[1] = (node << 4 | d) << 5 | t8 << 4 | k;
                cx[2] = (node << 4 | d) << 8 | hv;
                if full { cx[3] = (node << 8 | b << 4 | cc) << 4 | self.lastmode as u64; }
                sel = (node & 63) as usize;
                sel2 = (d * 4 + t8 * 2) as usize;
                actx = (node & 63) as usize;
            }
            se::CHROMA_PRED => {
                cx[0] = a << 8 | b << 4 | cc;
                cx[1] = (a << 8 | b << 4 | cc) << 4 | d;
                if full {
                    cx[2] = (key << 2 | a) << 8 | hv;
                    cx[3] = a << 8 | pk << 4 | d;
                }
                sel = (a * 16 + d) as usize;
                sel2 = (b * 8 + cc) as usize;
                actx = (a * 16 + d) as usize;
            }
            se::CBP_LUMA => {
                let (b8, cls, sofar) = (a, b, cc);
                let node = b8 << 4 | sofar;
                cx[0] = node << 12 | (l.cbp as u64) << 6 | t.cbp as u64;
                cx[1] = (node << 4 | cls) << 5 | qp / 2;
                cx[2] = node << 8 | ql(l.coefs as u32) << 4 | ql(t.coefs as u32);
                if full {
                    cx[3] = (key << 2 | b8) << 8 | hv;
                    cx[4] = (node << 4 | (p.cbp as u64 & 15)) << 4 | cls;
                    cx[5] = (node << 4 | cls) << 4 | ql(self.mvsum);
                }
                sel = node as usize;
                sel2 = (cls * 4 + (p.cbp as u64 >> b8 & 1) * 2 + (o.cbp as u64 >> b8 & 1)) as usize;
                actx = ((ctx as u64 - 73) * 16 + cls) as usize;
            }
            se::CBP_CHROMA => {
                let (cls, luma) = (b, cc);
                cx[0] = (key << 1 | a) << 8 | hv;
                cx[1] = a << 8 | (l.cbp as u64 >> 4) << 4 | (t.cbp as u64 >> 4) << 2 | (luma == 0) as u64;
                cx[2] = (a << 4 | (p.cbp as u64 >> 4)) << 5 | cls << 1 | (luma == 0) as u64;
                cx[3] = ((a << 4 | cls) << 5 | qp / 2) << 3 | luma.count_ones() as u64;
                if full {
                    cx[4] = key << 1 | a;
                    cx[5] = (a << 4 | luma) << 4 | cls;
                }
                sel = (a * 32 + luma * 2 + (cls == class::I16 as u64) as u64) as usize;
                sel2 = (cls * 4 + (p.cbp as u64 >> 4)) as usize;
                actx = ((ctx as u64 - 77) * 8 + (p.cbp as u64 >> 4) * 2 + a) as usize;
            }
            se::T8X8 => {
                cx[0] = key;
                cx[1] = b << 4 | pk;
                if full { cx[2] = key << 8 | hv; }
                sel = (b * 4 + l.t8 as u64 * 2 + t.t8 as u64) as usize;
                sel2 = cc as usize;
                actx = (b * 4 + p.t8 as u64 * 2 + o.t8 as u64) as usize;
            }
            se::CBF => {
                let (cat, blk, nb, cls) = (a >> 8, a & 255, b as u32, cc);
                self.new_block(cat as usize, blk as usize);
                let pnz = self.prevnz[addr * NB + self.bi] as u64;
                let node = cat << 5 | blk;
                // chroma: the block of the other plane
                let other = if matches!(self.bi, 20..=23 | 26) { 1 + (self.side[3][0] != 0) as u64 } else { 0 };
                cx[0] = key << 8 | hv;
                cx[1] = (node << 4 | cls) << 6 | d;
                cx[2] = ((cat << 4 | cls) << 5 | qp / 2) << 4 | qc(nb);
                cx[3] = node << 4 | ql(self.mbcoefs);
                cx[4] = ((node << 2 | other) << 4 | qc(nb)) << 2 | pnz.min(3);
                if full {
                    cx[5] = key;
                    cx[6] = node << 6 | nb.min(33) as u64;
                    cx[7] = ((cat << 4 | cls) << 4 | qc(nb)) << 4 | ql(self.mbcoefs);
                }
                sel = (cat * 8 + qc(nb)) as usize;
                sel2 = (cls * 4 + pnz.min(3)) as usize;
                actx = ((ctx as u64 - 85) % 4 * 16 + cat * 2 + (pnz > 0) as u64) as usize;
            }
            se::SIG | se::LAST => {
                let (cat, i, coefs, nb, cls, blk) = (a >> 8, a & 255, b as u32, cc as u32, d & 15, d >> 4);
                if info.se == se::SIG && i == 0 { self.new_block(cat as usize, blk as usize); }
                let s = (i + (cat == 1 || cat == 4) as u64) as usize;
                let near = match cat { 3 => NEAR2[s & 3], 5 => NEAR8[s & 63], _ => NEAR4[s & 15] };
                let at = |k: u8| if k < 64 { self.sig >> k & 1 } else { 0 };
                let (fl, ft) = (at(near[0]), at(near[1]));
                let last2 = if s >= 2 { self.sig >> (s - 2) & 3 } else { self.sig & 1 } | (s.min(2) as u64) << 2;
                let node = cat << 6 | i;
                let n = coefs.min(15) as u64;
                // the blocks beside it and before it: their coefficients further on
                let ahead = |m: u64| (m.checked_shr(s as u32 + 1).unwrap_or(0).count_ones() as u64).min(3);
                let (aa, ab, ap) = (ahead(self.side[0][0]), ahead(self.side[1][0]), ahead(self.side[2][0]));
                let chroma2 = matches!(self.bi, 20..=23 | 26);
                if info.se == se::SIG {
                    // coefficients among the lower frequencies around it; the same position in
                    // the blocks beside it and before it; the block's prediction mode
                    let tc = (self.sig & match cat { 3 => 0, 5 => TPL8[s & 63], _ => TPL4[s & 15] }).count_ones() as u64;
                    let same = (self.side[0][0] >> s & 1) << 2 | (self.side[1][0] >> s & 1) << 1 | (self.side[2][0] >> s & 1);
                    let bigs = (self.side[0][2] >> s & 1) + (self.side[1][2] >> s & 1);
                    let mode = if cat == 2 || cat == 5 { self.modes[self.bi & 15] as u64 } else { 15 };
                    let other = if chroma2 { 1 + (self.side[3][0] >> s & 1) } else { 0 };
                    cx[0] = (node << 4 | cls) << 5 | qp / 2;
                    cx[1] = ((node << 3 | tc) << 3 | same) << 2 | bigs;
                    cx[2] = ((node << 4 | mode) << 3 | tc) << 2 | n.min(3);
                    cx[3] = (((node << 2 | aa) << 2 | ab) << 2 | ap) << 2 | n.min(3);
                    cx[4] = ((node << 2 | other) << 4 | ql(self.mvsum)) << 2 | n.min(3);
                    if full {
                        cx[5] = key << 8 | hv;
                        cx[6] = node << 4 | n;
                        cx[7] = (node << 4 | last2) << 2 | fl << 1 | ft;
                        cx[8] = (node << 4 | cls) << 5 | blk;
                        cx[9] = (node << 3 | same) << 2 | fl << 1 | ft;
                        cx[10] = (node << 4 | n) << 3 | same;
                    }
                    sel = (i.min(15) * 4 + n.min(3)) as usize;
                } else {
                    // do those blocks have coefficients further on?
                    let more = |m: u64| (m.checked_shr(s as u32 + 1).unwrap_or(0) != 0) as u64;
                    let on = more(self.side[0][0]) << 2 | more(self.side[1][0]) << 1 | more(self.side[2][0]);
                    let other = if chroma2 { 1 + more(self.side[3][0]) } else { 0 };
                    cx[0] = key << 8 | hv;
                    cx[1] = (node << 4 | last2) << 2 | fl << 1 | ft;
                    cx[2] = (node << 4 | cls) << 5 | qp / 2;
                    cx[3] = (node << 3 | on) << 3 | qc(nb);
                    cx[4] = ((node << 2 | aa) << 2 | ab) << 3 | n.min(7);
                    cx[5] = ((node << 2 | other) << 2 | ap) << 4 | n;
                    if full {
                        let pnz = self.prevnz[addr * NB + self.bi] as u64;
                        cx[6] = node << 4 | n;
                        cx[7] = (node << 5 | pnz) << 2 | n.min(3);
                        cx[8] = (node << 4 | cls) << 5 | blk;
                        cx[9] = (node << 3 | on) << 4 | n;
                        cx[10] = ((node << 3 | self.near[0]) << 3 | self.near[1]) << 4 | n;
                    }
                    sel = (i.min(15) * 4 + n.min(4).saturating_sub(1)) as usize;
                }
                sel2 = (cat * 8 + qc(nb)) as usize;
                actx = ((last2 << 2 | fl << 1 | ft) & 15) as usize * 4 + n.min(3) as usize;
            }
            se::LEVEL => {
                let (cat, pos, k, nb, cls, coefs, abs) = (a >> 8, a & 255, b & 15, (cc & 255) as u32, cc >> 8, d & 255, (d >> 8).min(15));
                let s = (pos + (cat == 1 || cat == 4) as u64) as usize;
                let near = match cat { 3 => NEAR2[s & 3], 5 => NEAR8[s & 63], _ => NEAR4[s & 15] };
                // the magnitudes beside it at higher frequencies; above 1 at its position in the
                // blocks beside it and before it
                let (m1, m2) = (self.mag[near[2] as usize] as u32, self.mag[near[3] as usize] as u32);
                let bigs = (self.side[0][2] >> s & 1) << 2 | (self.side[1][2] >> s & 1) << 1 | (self.side[2][2] >> s & 1);
                let node = cat << 4 | abs;
                cx[0] = key << 4 | abs;
                cx[1] = (node << 6 | pos) << 5 | coefs.min(31);
                cx[2] = node << 12 | ql(self.prevmag) << 8 | ql(m1) << 4 | ql(m2);
                cx[3] = (node << 6 | pos) << 4 | ql(self.msum);
                cx[4] = ((node << 6 | pos) << 3 | bigs) << 2 | k.min(3);
                cx[5] = ((node << 4 | ql(self.mvsum)) << 5 | qp / 2) << 2 | k.min(3);
                if full {
                    let pnz = self.prevnz[addr * NB + self.bi] as u64;
                    cx[6] = (key << 4 | abs) << 8 | hv;
                    cx[7] = (node << 3 | qc(nb)) << 4 | coefs.min(15);
                    cx[8] = (node << 4 | cls) << 5 | qp / 2;
                    cx[9] = (node << 5 | pnz) << 2 | k.min(3);
                }
                sel = (abs.min(3) * 16 + k.min(15)) as usize;
                sel2 = (cat * 8 + ql(self.prevmag).min(7)) as usize;
                actx = (ql(m1 + m2).min(7) * 8 + abs.min(7)) as usize;
            }
            se::LEVEL_SUF => {
                let cat = a >> 8;
                cx[0] = cat << 6 | d.min(63);
                cx[1] = d.min(63) << 8 | hv;
                cx[2] = (cat << 6 | d.min(63)) << 5 | qp / 2;
                sel = d.min(63) as usize;
                sel2 = cat as usize;
                actx = d.min(63) as usize;
            }
            se::SIGN => {
                let (cat, pos) = (a >> 8, a & 255);
                let s = (pos + (cat == 1 || cat == 4) as u64) as usize & 63;
                // the sign at the same position in the blocks beside it and before it, and in
                // the other chroma plane's: none, +, -
                let sg = |m: &[u64; 3]| (m[0] >> s & 1) + (m[1] >> s & 1);
                let (sa, sb, sp, so) = (sg(&self.side[0]), sg(&self.side[1]), sg(&self.side[2]), sg(&self.side[3]));
                let node = cat << 6 | pos;
                // the signs of the higher frequencies beside it in the block
                let near = match cat { 3 => NEAR2[s & 3], 5 => NEAR8[s & 63], _ => NEAR4[s & 15] };
                let own = |k: u8| if k < 64 && self.mag[k as usize] != 0 { 1 + (self.neg >> k & 1) } else { 0 };
                let (f1, f2) = (own(near[2]), own(near[3]));
                let cls = cc >> 8;
                let mode = if cat == 2 || cat == 5 { self.modes[self.bi & 15] as u64 } else { 15 };
                let abs = (b >> 4).min(3);
                cx[0] = ((node << 4 | sa << 2 | sb) << 4 | mode) << 2 | abs;
                cx[1] = ((node << 2 | so) << 4 | sa << 2 | sb) << 4 | cls;
                cx[2] = ((node << 4 | f1 << 2 | f2) << 2 | self.lastsign as u64) << 2 | sp;
                if full {
                    let plane = match self.bi { 20..=23 | 26 => 2, 16..=19 | 25 => 1, _ => 0 };
                    let bal = if s < 16 { ((self.bal[plane][s] as i32 + 256) / 74) as u64 } else { 7 };
                    cx[3] = (node << 4 | sa << 2 | sb) << 2 | sp;
                    cx[4] = (node << 4 | f1 << 2 | f2) << 4 | sa << 2 | sb;
                    cx[5] = node << 3 | bal;
                }
                sel = (cat * 9 + sa * 3 + sb) as usize;
                sel2 = (self.lastsign as u64 * 16 + pos.min(15)) as usize;
                actx = (pos.min(15) * 4 + sp) as usize;
            }
            _ => {
                // the end-of-slice bin: at the end of the picture, of a row, where the last
                // picture's slices ended
                let total = self.cur.len();
                let ends = (addr + 1 == total) as u64 | ((addr + 1) % self.w == 0) as u64 * 2 | self.prev_ends.contains(&(addr as u32)) as u64 * 4;
                cx[0] = ends;
                cx[1] = ends << 4 | (self.cur_ends.len() as u64).min(15);
                cx[2] = ends << 8 | hv;
                sel = ends as usize;
                sel2 = 0;
                actx = ends as usize;
            }
        }
        for &(f, i) in &self.drop {
            if f == fam { cx[i] = 0; }
        }
        let out = self.b.bit2(c, bit, fam, &cx[..if full { NIN[fam] } else { NIN_LEAN[fam] }], sel, sel2, actx);
        self.hist[hi] = (hv as u8) << 1 | out as u8;
        // what the bin tells about the element being coded
        match info.se {
            se::MB_TYPE | se::SUB_TYPE => self.tybits = (self.tybits << 1 | out).min(1 << 20),
            se::REF => if out == 0 { self.lastref = info.b as u32 },
            se::MVD => {
                self.mvb = self.mvb << 1 | out as u64;
                self.mvn += 1;
                if out == 0 && d == 0 && a & 1 == 0 {
                    self.mvx = 0;
                    self.mvxs = 0;
                    self.hitx = (0..4).fold(0, |m, k| m | ((self.candx[k] == 0) as u64) << k);
                }
            }
            se::MVD_SUF => {
                self.mvb = self.mvb << 1 | out as u64;
                self.mvn += 1;
            }
            se::MVD_SIGN => {
                if a & 1 == 0 {
                    self.mvx = d as u32;
                    self.mvxs = 1 + out;
                    let v = if out == 1 { -(d as i32) } else { d as i32 };
                    self.hitx = (0..4).fold(0, |m, k| m | ((d < 255 && self.candx[k] as i32 == v) as u64) << k);
                }
                self.mvsum += d as u32;
            }
            se::QP => {
                if out == 1 {
                    self.qpv += 1;
                } else {
                    let v = self.qpv as i32;
                    let dqp = if v & 1 == 1 { (v + 1) / 2 } else { -(v / 2) };
                    self.qp = (self.qp + dqp + 52).rem_euclid(52);
                }
            }
            se::IPRED_FLAG => {
                if out == 1 {
                    self.lastmode = d as u32;
                    self.set_mode(a);
                }
                self.rem = 0;
            }
            se::IPRED_REM => {
                let bitno = (a >> 5 & 3) as u32;
                self.rem |= out << bitno;
                if bitno == 2 {
                    self.lastmode = if self.rem < d as u32 { self.rem } else { self.rem + 1 };
                    self.set_mode(a);
                }
            }
            se::SIG => {
                if out == 1 {
                    let s = (a & 255) + ((a >> 8) == 1 || (a >> 8) == 4) as u64;
                    self.sig |= 1 << (s & 63);
                }
            }
            se::SIGN => {
                let (cat, pos, abs) = (a >> 8, a & 255, (b >> 4) as u32);
                let s = (pos + (cat == 1 || cat == 4) as u64) as usize & 63;
                self.mag[s] = abs.min(255) as u8;
                self.neg |= (out as u64) << s;
                if s < 16 {
                    let plane = match self.bi { 20..=23 | 26 => 2, 16..=19 | 25 => 1, _ => 0 };
                    let v = &mut self.bal[plane][s];
                    *v = *v - *v / 4 + if out == 1 { -64 } else { 64 };
                }
                // an 8x8 block's maps take four slots, 16 scan positions each
                let slot = addr * NB + self.bi + if cat == 5 { [0, 1, 4, 5][s >> 4] } else { 0 };
                let m = &mut self.cursig[slot];
                m[0] |= 1 << (s & 15);
                m[1] |= (out as u16) << (s & 15);
                m[2] |= ((abs > 1) as u16) << (s & 15);
                self.prevmag = abs;
                self.msum += abs;
                self.lastsign = 1 + out;
                self.cnt += 1;
                self.mbcoefs += 1;
                let row = addr * NB;
                if cat == 5 {
                    let v = self.cnt.div_ceil(4).min(16) as u8;
                    for k in [0, 1, 4, 5] { self.curnz[row + self.bi + k] = v; }
                } else {
                    self.curnz[row + self.bi] = self.cnt.min(16) as u8;
                }
            }
            se::END => {
                let k = (a & 15) as u8;
                self.cur[addr] = M {
                    k: k + 1, cbp: (b & 63) as u8, qp: (cc & 63) as u8, dqp: ((b >> 6) as i32 - 64) as i8, t8: (a >> 4 & 1) as u8,
                    mv: ql((a >> 8) as u32) as u8, coefs: d as u16,
                }.pack();
                self.qp = cc as i32;
                self.skiprun = if k == class::SKIP { self.skiprun + 1 } else { 0 };
                if out == 1 && self.cur_ends.len() < 64 { self.cur_ends.push(addr as u32); }
            }
            _ => {}
        }
        out
    }

    /// Code a raw byte (PCM samples).
    fn raw<C: Coder>(&mut self, c: &mut C, v: u8, id: u64, a: u64, b: u64) -> u8 { self.b.tree(c, v as u32, 8, F_RAW, id, a, b, 0) as u8 }

    /// Code the slice data of one slice: its packed bins, read from `src` (the encoder) or
    /// appended to `out` (the decoder, byte-aligned; at most `room` bytes). Returns the bytes of
    /// packed bins.
    #[allow(clippy::too_many_arguments)]
    pub fn slice<C: Coder>(&mut self, c: &mut C, src: Option<&[u8]>, h: &SliceHeader, sps: &Sps, pps: &Pps, scratch: &mut Scratch, out: &mut BitWriter, room: usize)
        -> Result<usize, String> {
        self.begin_slice(h, sps);
        let start = out.len();
        let limit = src.map_or(start + room * 8, |s| s.len() * 8);
        let mut d = Drive { m: self, c, r: src.map(|s| BitReader::new(s, 0)), w: out, limit, over: false };
        h264::walk(&mut d, h, sps, pps, scratch)?;
        if d.over { return Err("h264 records: slice overflows its leaf (corrupt file)".into()); }
        match d.r.as_mut() {
            Some(r) => {
                // the decoder pads with zero bits
                let pad = (8 - r.pos() % 8) % 8;
                if r.get(pad as u32) != 0 { return Err("h264 records: non-canonical record".into()); }
                Ok(r.pos() / 8)
            }
            None => {
                out.align();
                Ok((out.len() - start) / 8)
            }
        }
    }

    /// Code a record field byte.
    fn field<C: Coder>(&mut self, c: &mut C, v: u8, id: u64, a: u64, b: u64, x: u64) -> u8 { self.b.tree(c, v as u32, 8, F_REC, id, a, b, x) as u8 }

    /// Code a varint (`src`: its bytes on the encoder side), appending its bytes to `out`.
    fn var<C: Coder>(&mut self, c: &mut C, src: Option<&[u8]>, which: usize, out: &mut Vec<u8>) -> Result<usize, String> {
        let bad = || "h264 records: bad record (corrupt file)".to_string();
        let mut v = 0u64;
        for k in 0..5 {
            let s = match src { Some(s) => *s.get(k).ok_or_else(bad)?, None => 0 };
            let x = self.field(c, s, 40 + (which * 5 + k) as u64, self.last_var[which][k] as u64, 0, 0);
            self.last_var[which][k] = x;
            out.push(x);
            v |= ((x & 0x7F) as u64) << (7 * k);
            if x & 0x80 == 0 {
                if k > 0 && x == 0 { return Err(bad()); }
                for j in k + 1..5 { self.last_var[which][j] = 0; }
                return Ok(v as usize);
            }
        }
        Err(bad())
    }

    /// Code a leaf of stream records (see `recode::h264`): encode them from `buf`, or decode
    /// them into it.
    pub fn code_leaf<C: Coder, B: ByteBuf>(&mut self, c: &mut C, buf: &mut B, range: std::ops::Range<usize>, params: &[u8]) -> Result<(), String> {
        let bad = || "h264 records: bad record (corrupt file)".to_string();
        if !matches!(params, [1 | 2 | 4, p] if *p == self.full as u8) { return Err("bad h264 record parameters".into()); }
        let enc = buf.reading();
        let all: Vec<u8> = if enc { buf.hist(range.end)[range.start..].to_vec() } else { vec![] };
        let len = range.end - range.start;
        let mut ps = Params::default();
        let mut scratch = std::mem::take(&mut self.scratch);
        let mut pos = 0usize;
        let mut rec: Vec<u8> = Vec::new();
        while pos < len {
            // the encoder's record: the bytes from here on
            let src = enc.then(|| &all[pos..]);
            let at = |k: usize| -> Result<u8, String> { match src { Some(s) => s.get(k).copied().ok_or_else(bad), None => Ok(0) } };
            rec.clear();
            let tag = self.b.tree(c, at(0)? as u32, 8, F_REC, 1, self.rec[0] as u64, 0, 0) as u8;
            self.rec[0] = tag;
            rec.push(tag);
            match tag {
                h264::REC_RAW => {
                    self.var(c, src.map(|s| &s[1..]), 0, &mut rec)?;
                }
                h264::REC_CONFIG | h264::REC_PARAM => {
                    let n = self.var(c, src.map(|s| &s[1..]), 1, &mut rec)?;
                    if n == 0 || n > h264::MAX_PARAM || rec.len() + n > len - pos { return Err(bad()); }
                    let from = rec.len();
                    // against the last parameter set of its type
                    let mut which = 0;
                    for k in 0..n {
                        let old = self.last_ps[which].get(k).map_or(256, |&x| x as u64);
                        let prev = if k > 0 { rec[from + k - 1] } else { 0 };
                        let x = self.field(c, at(from + k)?, 2 + which as u64, old, k.min(255) as u64, prev as u64);
                        if k == 0 { which = (x & 31 == 8) as usize; }
                        rec.push(x);
                    }
                    self.last_ps[which] = rec[from..].to_vec();
                    if !ps.add(&rec[from..]) { return Err(bad()); }
                }
                h264::REC_SLICE => {
                    let nal = self.field(c, at(1)?, 8, self.rec[1] as u64, 0, 0);
                    let hlen = self.field(c, at(2)?, 9, self.rec[2] as u64, nal as u64, 0);
                    (self.rec[1], self.rec[2]) = (nal, hlen);
                    rec.extend_from_slice(&[nal, hlen]);
                    if hlen == 0 || 3 + hlen as usize > len - pos { return Err(bad()); }
                    // the header, against the last slice's
                    let mut header = std::mem::take(&mut self.last_hdr);
                    for k in 0..hlen as usize {
                        let old = header.get(k).map_or(256, |&x| x as u64);
                        let prev = if k > 0 { rec[2 + k] } else { nal };
                        let x = self.field(c, at(3 + k)?, 10 + k.min(15) as u64, old, prev as u64, nal as u64 & 31);
                        rec.push(x);
                        if k < header.len() { header[k] = x; } else { header.push(x); }
                    }
                    header.truncate(hlen as usize);
                    self.last_hdr = header;
                    let n = rec.len();
                    self.var(c, src.map(|s| &s[n..]), 2, &mut rec)?;
                    let align = self.field(c, at(rec.len())?, 30, self.rec[3] as u64, 0, 0);
                    self.rec[3] = align;
                    rec.push(align);
                    let (h, sps, pps) = h264::record_header(nal, &rec[3..3 + hlen as usize], &ps).ok_or_else(bad)?;
                    if !pps.cabac {
                        // a CAVLC slice: its P_8x8ref0 macroblocks, a count and address differences
                        let n = rec.len();
                        let count = self.var(c, src.map(|s| &s[n..]), 3, &mut rec)?;
                        if count > h264::MAX_MBS { return Err(bad()); }
                        for _ in 0..count {
                            let n = rec.len();
                            self.var(c, src.map(|s| &s[n..]), 4, &mut rec)?;
                            if rec.len() > len - pos { return Err(bad()); }
                        }
                    }
                    let room = (len - pos).checked_sub(rec.len()).ok_or_else(bad)?;
                    let mut w = BitWriter::new();
                    let used = self.slice(c, src.map(|s| &s[rec.len()..]), &h, &sps, &pps, &mut scratch, &mut w, room)?;
                    if !enc { rec.extend_from_slice(w.bytes()); } else { rec.resize(rec.len() + used, 0); }
                }
                _ => return Err(bad()),
            }
            if rec.len() > len - pos { return Err("h264 records: leaf overflow (corrupt file)".into()); }
            if !enc {
                for (k, &x) in rec.iter().enumerate() { buf.put(range.start + pos + k, x); }
            }
            pos += rec.len();
        }
        self.scratch = scratch;
        Ok(())
    }
}

/// The walker's bins, coded by the model: read from the leaf (encoder) or decoded and written.
struct Drive<'a, 'b, C: Coder> {
    m: &'a mut H264Model,
    c: &'a mut C,
    r: Option<BitReader<'b>>,
    w: &'a mut BitWriter,
    /// Encoder: bits of packed bins available; decoder: the bit position `w` may not pass.
    limit: usize,
    over: bool,
}

impl<C: Coder> Drive<'_, '_, C> {
    #[inline]
    fn step(&mut self, ctx: u16, info: &Info) -> u32 {
        match self.r.as_mut() {
            Some(r) => {
                if r.pos() >= self.limit {
                    self.over = true;
                    return (ctx == TERMINATE) as u32;
                }
                let bit = r.get(1);
                self.m.bin(self.c, bit, ctx, info)
            }
            None => {
                if self.over || self.w.len() >= self.limit {
                    // the leaf is full (a damaged file): end the slice at once
                    self.over = true;
                    return (ctx == TERMINATE) as u32;
                }
                let b = self.m.bin(self.c, 0, ctx, info);
                self.w.put(b, 1);
                b
            }
        }
    }
}

impl<C: Coder> h264::Bins for Drive<'_, '_, C> {
    #[inline]
    fn decision(&mut self, ctx: usize, info: &Info) -> u32 { self.step(ctx as u16, info) }
    #[inline]
    fn bypass(&mut self, info: &Info) -> u32 { self.step(h264::BYPASS, info) }
    #[inline]
    fn terminate(&mut self, info: &Info) -> u32 { self.step(TERMINATE, info) }

    fn pcm(&mut self, n: usize) -> Result<(), String> {
        // the slice's alignment bits before the samples (a byte), then the samples
        let (mut prev, mut prev2) = (0u8, 0u8);
        match self.r.as_mut() {
            Some(r) => {
                r.seek(r.pos().div_ceil(8) * 8);
                if r.pos() + 8 * (n + 1) > self.limit {
                    self.over = true;
                    return Err("h264 records: PCM samples missing".into());
                }
                for k in 0..n + 1 {
                    let v = r.get(8) as u8;
                    self.m.raw(self.c, v, (k > 0) as u64, prev as u64, prev2 as u64);
                    (prev, prev2) = (v, prev);
                }
            }
            None => {
                self.w.align();
                if self.over || self.w.len() + 8 * (n + 1) > self.limit {
                    self.over = true;
                    return Err("h264 records: slice overflows its leaf (corrupt file)".into());
                }
                for k in 0..n + 1 {
                    let v = self.m.raw(self.c, 0, (k > 0) as u64, prev as u64, prev2 as u64);
                    self.w.extend(&[v]);
                    (prev, prev2) = (v, prev);
                }
            }
        }
        Ok(())
    }
}

/// A coder that codes nothing: the model's statistics give the cost.
pub struct Count;
impl Coder for Count {
    fn code(&mut self, bit: u32, _p: u32) -> u32 { bit }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cm::coder::{Decoder, Encoder};
    use crate::cm::predictor::{ReadBuf, WriteBuf};
    use crate::recode::h264::tests::{track, FIXTURES};

    fn rng(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }

    #[test]
    fn records_roundtrip_and_reject_corruption() {
        for (k, mp4) in FIXTURES.iter().enumerate().flat_map(|x| [(x.0, x.1, false), (x.0 + 10, x.1, true)]).map(|(k, m, full)| ((k, full), m)) {
            let ((k, full), mp4) = (k, mp4);
            let (stream, sizes, nal_len, config) = track(mp4);
            let sp = h264::split(&stream, &sizes, nal_len, &config).unwrap();
            let leaf = &sp.records;
            let params = [nal_len as u8, full as u8];
            let mut enc = Encoder::new();
            H264Model::new(full).code_leaf(&mut enc, &mut ReadBuf(leaf), 0..leaf.len(), &params).unwrap();
            let coded = enc.finish();
            assert!(coded.len() + sp.raw.len() < stream.len(), "fixture {k}: {} bytes of video coded to {} + {} raw", stream.len(), coded.len(), sp.raw.len());
            let mut back = vec![0u8; leaf.len()];
            H264Model::new(full).code_leaf(&mut Decoder::new(&coded), &mut WriteBuf(&mut back), 0..leaf.len(), &params).unwrap();
            assert!(&back == leaf, "fixture {k}: the model did not decode its own output");
            // a model that has seen other leaves decodes it too, when the encoder had seen the same
            let (mut a, mut b) = (H264Model::new(full), H264Model::new(full));
            let mut enc = Encoder::new();
            for _ in 0..2 { a.code_leaf(&mut enc, &mut ReadBuf(leaf), 0..leaf.len(), &params).unwrap(); }
            let twice = enc.finish();
            assert!(twice.len() < coded.len() * 2);
            let mut dec = Decoder::new(&twice);
            for _ in 0..2 {
                back.fill(0);
                b.code_leaf(&mut dec, &mut WriteBuf(&mut back), 0..leaf.len(), &params).unwrap();
                assert!(&back == leaf);
            }
            // damaged coded bytes: an error or other bytes, never a panic or a write past the leaf
            let mut s = 4u64;
            for _ in 0..60 {
                let mut bad = coded.clone();
                let at = (rng(&mut s) % bad.len() as u64) as usize;
                bad[at] ^= 1 << (rng(&mut s) % 8);
                let mut out = vec![0u8; leaf.len()];
                let _ = H264Model::new(full).code_leaf(&mut Decoder::new(&bad), &mut WriteBuf(&mut out), 0..leaf.len(), &params);
            }
            // a leaf of the wrong length, records that are not canonical, bad parameters
            let mut out = vec![0u8; leaf.len() - 1];
            assert!(H264Model::new(full).code_leaf(&mut Decoder::new(&coded), &mut WriteBuf(&mut out), 0..leaf.len() - 1, &params).is_err());
            assert!(H264Model::new(full).code_leaf(&mut Encoder::new(), &mut ReadBuf(&leaf[..leaf.len() - 1]), 0..leaf.len() - 1, &params).is_err());
            assert!(H264Model::new(full).code_leaf(&mut Encoder::new(), &mut ReadBuf(leaf), 0..leaf.len(), &[3, full as u8]).is_err());
            // a leaf of the other profile is refused by this model
            assert!(H264Model::new(full).code_leaf(&mut Encoder::new(), &mut ReadBuf(leaf), 0..leaf.len(), &[nal_len as u8, !full as u8]).is_err());
            let mut s = 8u64;
            for _ in 0..40 {
                let mut bad = leaf.clone();
                let at = (rng(&mut s) % bad.len() as u64) as usize;
                bad[at] ^= 1 << (rng(&mut s) % 8);
                // the encoder refuses, or codes records the decoder gives back
                let mut enc = Encoder::new();
                if H264Model::new(full).code_leaf(&mut enc, &mut ReadBuf(&bad), 0..bad.len(), &params).is_ok() {
                    let coded = enc.finish();
                    let mut out = vec![0u8; bad.len()];
                    H264Model::new(full).code_leaf(&mut Decoder::new(&coded), &mut WriteBuf(&mut out), 0..bad.len(), &params).unwrap();
                    assert!(out == bad, "fixture {k}: damaged records were coded but not decoded back");
                }
            }
        }
    }
}
