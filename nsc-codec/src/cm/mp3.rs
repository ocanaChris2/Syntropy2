//! Model for MP3 frame records (the frame leaves of MP3 containers; see `recode::mp3`).
//!
//! The leaf holds each frame as a parsed record. Instead of modelling those bytes, this model
//! codes the frames' fields themselves:
//! * header bytes and side information against the previous frame's and granule's values (gains,
//!   `big_values` and the count1 length as differences, tables and flags by their predecessors);
//! * scale factors against the same band of the previous granule and the band before;
//! * spectral values line by line: a zero flag, then "larger than k" steps up to the largest
//!   value of the Huffman table the encoder chose for the region (itself a strong context), then
//!   escape bits. Contexts are the lines just coded, the same and neighbouring lines of the
//!   channel's previous granule, the other channel's line, the position and the first value of the
//!   pair. Signs are close to random and coded flat.
//!
//! Every binary decision mixes its context predictions with an online mixer and refines the
//! result with an APM. Integer-only, like the rest of the engine.
use super::binary::{q3, q8, ql, sc, Bins, GAIN_SCALE, INPUTS};
use super::coder::Coder;
use super::predictor::ByteBuf;
use crate::recode::mp3::{self, Body, Frame, Granule, Hdr, PairTables, Parsed, LINES};
use crate::recode::mp3_tables::{LINBITS, SFB_SHORT, TREE_MAX, TREE_OF};

/// The most context inputs any decision uses.
const MAXIN: usize = 14;

const F_HDR: usize = 0;
const F_SIDE: usize = 1;
const F_GAIN: usize = 2;
const F_BV: usize = 3;
const F_TAB: usize = 4;
const F_SCF: usize = 5;
const F_ZERO: usize = 6;
const F_MAG: usize = 7;
const F_ESC: usize = 8;
const F_SIGN: usize = 9;
const F_NQ: usize = 10;
const F_QUAD: usize = 11;
const F_RAW: usize = 12;
pub const FAMILIES: usize = 13;
pub const FAMILY_NAMES: [&str; FAMILIES] = ["header", "side", "gain", "big-values", "tables", "scalefac", "zero", "magnitude", "escape", "sign", "count1-len", "count1", "raw"];
/// log2 of each family's context table size (per input).
const BITS: [u32; FAMILIES] = [12, 14, 14, 14, 16, 16, 16, 16, 14, 16, 14, 16, 12];
/// Context inputs each family uses.
const NIN: [usize; FAMILIES] = [INPUTS, INPUTS, INPUTS, INPUTS, INPUTS, INPUTS, MAXIN, MAXIN, INPUTS, INPUTS, INPUTS, 10, INPUTS];

/// What the next granule of a channel conditions on: the previous granule's side information.
#[derive(Clone, Copy, Default)]
struct Side {
    ws: bool,
    bt: u8,
    layout: u8,
    /// Layout of the granule before.
    layout2: u8,
    bv: u16,
    gain: u8,
    sfc: u16,
    ts: [u8; 3],
    aux: [u8; 3],
    pre: bool,
    scale: bool,
    c1t: bool,
    /// Line after the last coded one (2·big_values + 4·quads).
    end: u16,
    dgain: i32,
    dbv: i32,
    dnq: i32,
}

pub struct Mp3Model {
    b: Bins,
    side: [Side; 2],
    /// Magnitudes (capped at 255) of each channel's previous and current granule.
    prev: [[u8; LINES]; 2],
    cur: [[u8; LINES]; 2],
    /// The granule before the previous one.
    prev2: [[u8; LINES]; 2],
    /// Signs (0 zero, 1 positive, 2 negative) of each channel's previous and current granule.
    sprev: [[u8; LINES]; 2],
    scur: [[u8; LINES]; 2],
    /// Scale factors by slot of each channel's last granule.
    sf: [[u8; 40]; 2],
    hdr: [u8; 4],
    private: u8,
    scfsi: [u8; 2],
    flags: u32,
    /// Developer statistics: cost of the magnitude steps by threshold, and of each (granule
    /// layout: long, mixed, short), when statistics are enabled.
    pub step_cost: [u64; 16],
    pub layout_cost: [(u64, u64); 3],
}

impl Default for Mp3Model {
    fn default() -> Self { Self::new() }
}

/// For each line of a short-block granule (band table `row`): (window 0..3, band width). Lines run
/// band by band, each band as three windows of `width` lines.
fn short_lines(row: usize) -> &'static [(u8, u8); LINES] {
    static T: std::sync::OnceLock<Vec<[(u8, u8); LINES]>> = std::sync::OnceLock::new();
    &T.get_or_init(|| {
        (0..8).map(|r| {
            let mut t = [(0u8, 1u8); LINES];
            let mut i = 0;
            for &w in &SFB_SHORT[r] {
                for win in 0..3u8 {
                    for _ in 0..w {
                        if i < LINES { t[i] = (win, w); }
                        i += 1;
                    }
                }
            }
            t
        }).collect()
    })[row]
}

/// `a[i + d]`, zero outside the granule.
#[inline]
fn at(a: &[u8; LINES], i: usize, d: isize) -> u32 {
    let j = i as isize + d;
    if (0..LINES as isize).contains(&j) { a[j as usize] as u32 } else { 0 }
}

/// Position classes: every pair below line 64, then 16-line steps (64 classes).
#[inline]
fn posb(i: usize) -> u64 { if i < 64 { (i / 2) as u64 } else { 32 + ((i - 64) / 16) as u64 } }

impl Mp3Model {
    pub fn new() -> Self {
        Mp3Model {
            b: Bins::new(&BITS, &NIN),
            side: [Side::default(); 2],
            prev: [[0; LINES]; 2],
            cur: [[0; LINES]; 2],
            prev2: [[0; LINES]; 2],
            sprev: [[0; LINES]; 2],
            scur: [[0; LINES]; 2],
            sf: [[0; 40]; 2],
            hdr: [0; 4],
            private: 0,
            scfsi: [0; 2],
            flags: 0,
            step_cost: [0; 16],
            layout_cost: [(0, 0); 3],
        }
    }

    /// Collect coding cost per family (developer statistics).
    pub fn enable_stats(&mut self) { self.b.stats = Some(vec![0; FAMILIES]); }

    /// The coding cost per family so far (bits·65536), with statistics enabled.
    pub fn stats(&self) -> [u64; FAMILIES] {
        let mut out = [0; FAMILIES];
        if let Some(s) = &self.b.stats { out.copy_from_slice(s); }
        out
    }

    #[inline]
    fn bit<C: Coder>(&mut self, c: &mut C, bit: u32, fam: usize, ctx: [u64; INPUTS], sel: usize, actx: usize) -> u32 { self.b.bit(c, bit, fam, ctx, sel, actx) }

    #[inline]
    fn bit2<C: Coder>(&mut self, c: &mut C, bit: u32, fam: usize, ctx: &[u64], sel: usize, sel2: usize, actx: usize) -> u32 { self.b.bit2(c, bit, fam, ctx, sel, sel2, actx) }

    #[allow(clippy::too_many_arguments)]
    fn tree<C: Coder>(&mut self, c: &mut C, v: u32, n: u32, fam: usize, id: u64, a: u64, b: u64, x: u64) -> u32 { self.b.tree(c, v, n, fam, id, a, b, x) }

    #[allow(clippy::too_many_arguments)]
    fn gamma<C: Coder>(&mut self, c: &mut C, n: u32, fam: usize, id: u64, a: u64, b: u64, x: u64) -> u32 { self.b.gamma(c, n, fam, id, a, b, x) }

    #[allow(clippy::too_many_arguments)]
    fn delta<C: Coder>(&mut self, c: &mut C, d: i32, fam: usize, id: u64, a: u64, b: u64, x: u64) -> i32 { self.b.delta(c, d, fam, id, a, b, x) }

    /// One byte of raw data (little to predict).
    fn byte<C: Coder>(&mut self, c: &mut C, v: u8, id: u64, a: u64) -> u8 { self.tree(c, v as u32, 8, F_RAW, id, a, 0, 0) as u8 }

    /// What came before line `i` of channel `ch` in time: the magnitudes at the line and its two
    /// neighbours, a weighted sum of five lines around it, and a sum of three lines one step
    /// further back. For long blocks "before" is the previous granule (and the one before it),
    /// scaled to this granule's gain; for short blocks it is the previous short window of the
    /// same band, in this granule or at the end of the previous one.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn history(&self, ch: usize, i: usize, short: Option<&[(u8, u8); LINES]>, same: bool, same2: bool, scale1: u32, scale2: u32) -> [u32; 5] {
        let Some(tab) = short else {
            if !same { return [0; 5]; }
            let p = &self.prev[ch];
            let (b0, b1, bm) = (p[i] as u32, at(p, i, 1), at(p, i, -1));
            let s_prev = (at(p, i, -2) + bm + 2 * b0 + b1 + at(p, i, 2)) * scale1 >> 8;
            let q = &self.prev2[ch];
            let s_two = if same2 { (at(q, i, -1) + 2 * q[i] as u32 + at(q, i, 1)) * scale2 >> 8 } else { 0 };
            return [b0, b1, bm, s_prev, s_two];
        };
        // the same frequency `back` windows earlier
        let earlier = |j: isize, back: usize| -> u32 {
            if !(0..LINES as isize).contains(&j) { return 0; }
            let j = j as usize;
            let (win, w) = (tab[j].0 as usize, tab[j].1 as usize);
            if win >= back {
                self.cur[ch][j - back * w] as u32
            } else if same {
                // the previous granule's last windows: window 3 + win - back of the same band
                let k = j + (3 - back) * w;
                if k < LINES { self.prev[ch][k] as u32 } else { 0 }
            } else {
                0
            }
        };
        let i = i as isize;
        let (b0, b1, bm) = (earlier(i, 1), earlier(i + 1, 1), earlier(i - 1, 1));
        [b0, b1, bm, earlier(i - 2, 1) + bm + 2 * b0 + b1 + earlier(i + 2, 1), earlier(i - 1, 2) + 2 * earlier(i, 2) + earlier(i + 1, 2)]
    }

    /// Code the sign of the nonzero value (magnitude `val`) at line `i` of channel `ch`, and
    /// record it.
    #[inline]
    fn sign<C: Coder>(&mut self, c: &mut C, neg: bool, h: &Hdr, ch: usize, i: usize, val: u32, same: bool) -> bool {
        let so = if ch == 1 { self.scur[0][i] as u64 } else { 3 };
        let sp = if same { self.sprev[ch][i] as u64 } else { 3 };
        let s1 = if i > 0 { self.scur[ch][i - 1] as u64 } else { 3 };
        let s2 = if i > 1 { self.scur[ch][i - 2] as u64 } else { 3 };
        let ms = h.ms as u64;
        let mv = (val.min(3)) as u64;
        let mo = if ch == 1 { q3(self.cur[0][i]) } else { 0 };
        // do the two channels' signs agree (2), differ (1), or is one zero (0)? At the lines just
        // coded and at this line of the previous granule: where the sound sits between the channels
        let rel = |a: u8, b: u8| -> u64 { if a == 0 || b == 0 { 0 } else { 1 + (a == b) as u64 } };
        let (r1, r2, r3, rp) = if ch == 1 {
            (if i > 0 { rel(self.scur[0][i - 1], self.scur[1][i - 1]) } else { 0 },
             if i > 1 { rel(self.scur[0][i - 2], self.scur[1][i - 2]) } else { 0 },
             if i > 2 { rel(self.scur[0][i - 3], self.scur[1][i - 3]) } else { 0 },
             if same { rel(self.sprev[0][i], self.sprev[1][i]) } else { 0 })
        } else { (0, 0, 0, 0) };
        let ctx = [
            ((ms * 4 + so) * 4 + mv) * 4 + mo,
            (sp * 4 + s1) * 4 + mv,
            (((so * 3 + r1) * 3 + r2) * 3 + r3) * 2 + ms,
            (ms * 4 + so) * 64 + posb(i),
            (s1 * 4 + s2) * 4 + mv,
            (((ms * 4 + so) * 3 + rp) * 3 + r1) * 4 + mo,
        ];
        let b = self.bit(c, neg as u32, F_SIGN, ctx, (ms * 4 + so) as usize * 4 + mv as usize, (so * 16 + sp * 4 + s1) as usize) == 1;
        self.scur[ch][i] = 1 + b as u8;
        b
    }

    /// Code one granule of channel `ch`; `src` is the granule on the encoder side (ignored when
    /// decoding). Returns the granule coded.
    fn granule<C: Coder>(&mut self, c: &mut C, h: &Hdr, src: &Granule, second: bool, ch: usize, scfsi: u8) -> Result<Granule, String> {
        let bad = || "mp3 frames: bad granule (corrupt file)".to_string();
        let pg = self.side[ch];
        // the other channel's side information of this granule (0 = none)
        let og = (ch == 1).then_some(self.side[0]);
        let chx = ch as u64;
        let mut g = Granule::default();

        // ---- side information
        let o_ws = og.map_or(0, |o| 1 + o.ws as u64 * 4 + o.bt as u64);
        g.window_switching = self.bit(c, src.window_switching as u32, F_SIDE, sc(1, chx, pg.ws as u64 * 4 + pg.bt as u64, o_ws, 0), 1, 0) == 1;
        if g.window_switching {
            g.block_type = self.tree(c, src.block_type as u32, 2, F_SIDE, 2, pg.bt as u64 + 4 * pg.ws as u64, o_ws, chx) as u8;
            g.mixed = self.bit(c, src.mixed as u32, F_SIDE, sc(3, chx, g.block_type as u64, 0, 0), 3, 1) == 1;
        }
        let lay = mp3::layout(&g) as u8;
        let d0 = og.map_or(0, |o| (o.dgain.clamp(-8, 8) + 9) as u64);
        let dg = self.delta(c, src.global_gain as i32 - pg.gain as i32, F_GAIN, 4 + chx, (pg.dgain.clamp(-8, 8) + 8) as u64, d0, g.window_switching as u64 + 2 * second as u64);
        g.global_gain = u8::try_from(pg.gain as i32 + dg).map_err(|_| bad())?;
        let obv = og.map_or(0, |o| (o.dbv.clamp(-32, 32) / 4 + 9) as u64);
        let dbv = self.delta(c, src.big_values as i32 - pg.bv as i32, F_BV, 6 + chx, (pg.dbv.clamp(-32, 32) / 4 + 8) as u64, obv, (dg.clamp(-4, 4) + 4) as u64 + 16 * lay as u64);
        let bv = pg.bv as i32 + dbv;
        if !(0..=LINES as i32 / 2).contains(&bv) { return Err(bad()); }
        g.big_values = bv as u16;
        let bvb = (bv as u64 / 12).min(23);
        let o_sfc = og.map_or(0, |o| 1 + o.sfc as u64);
        g.scalefac_compress = if !h.lsf {
            self.tree(c, src.scalefac_compress as u32, 4, F_TAB, 8 + chx, pg.sfc as u64, o_sfc, lay as u64) as u16
        } else if self.bit(c, (src.scalefac_compress == pg.sfc) as u32, F_TAB, sc(10, chx, lay as u64, (o_sfc == 1 + pg.sfc as u64) as u64, 0), 10, 2) == 1 {
            pg.sfc
        } else {
            self.tree(c, src.scalefac_compress as u32, 9, F_TAB, 11, 0, 0, chx) as u16
        };
        let nt = if g.window_switching { 2 } else { 3 };
        for k in 0..nt {
            let o_t = og.map_or(0, |o| 1 + o.ts[k] as u64);
            // the table follows the previous granule's, shifted by how the gain moved (a coarser
            // step means smaller values) and by the table just chosen for the region below
            let below = if k > 0 { 1 + g.table_select[k - 1] as u64 } else { 0 };
            g.table_select[k] = self.tree(c, src.table_select[k] as u32, 5, F_TAB, 12 + k as u64, pg.ts[k] as u64 + 32 * (pg.ws != g.window_switching) as u64,
                o_t * 8 + (dg.clamp(-3, 3) + 3) as u64, below * 32 + bvb) as u8;
        }
        if g.window_switching {
            for k in 0..3 {
                g.aux[k] = self.tree(c, src.aux[k] as u32, 3, F_TAB, 16 + k as u64, if pg.ws { pg.aux[k] as u64 } else { 8 }, og.map_or(0, |o| 1 + o.aux[k] as u64), 0) as u8;
            }
        } else {
            for (k, n) in [(0usize, 4u32), (1, 3)] {
                g.aux[k] = self.tree(c, src.aux[k] as u32, n, F_TAB, 20 + k as u64, if pg.ws { 16 } else { pg.aux[k] as u64 }, og.map_or(0, |o| 1 + o.aux[k] as u64), bvb) as u8;
            }
        }
        if !h.lsf { g.preflag = self.bit(c, src.preflag as u32, F_SIDE, sc(24, chx, pg.pre as u64, og.map_or(0, |o| 1 + o.pre as u64), 0), 24, 3) == 1; }
        g.scalefac_scale = self.bit(c, src.scalefac_scale as u32, F_SIDE, sc(25, chx, pg.scale as u64, og.map_or(0, |o| 1 + o.scale as u64), 0), 25, 4) == 1;
        g.count1table = self.bit(c, src.count1table as u32, F_SIDE, sc(26, chx, pg.c1t as u64, og.map_or(0, |o| 1 + o.c1t as u64), bvb), 26, 5) == 1;
        if g.scalefac_compress >= if h.lsf { 512 } else { 16 } || !mp3::side_ok(h, &g) { return Err(bad()); }

        // ---- scale factors
        let same = lay == pg.layout;
        let same2 = same && lay == pg.layout2;
        let sfprev = self.sf[ch];
        let mut sfcur = if second && lay == 0 { sfprev } else { [0u8; 40] };
        let mut j = 0;
        for (slot, w) in mp3::scf_slots(h, &g, second, ch, scfsi) {
            let slot = slot as usize;
            if w == 0 {
                sfcur[slot] = 0;
                continue;
            }
            let pv = if same { sfprev[slot] as u64 } else { 16 };
            let left = if slot > 0 { sfcur[slot - 1] as u64 } else { 16 };
            let ov = if ch == 1 { 1 + self.sf[0][slot] as u64 } else { 0 };
            // how the band before moved since the previous granule: this one tends to follow
            let dl = if slot > 0 && same { (sfcur[slot - 1] as i32 - sfprev[slot - 1] as i32).clamp(-3, 3) + 4 } else { 0 } as u64;
            let v = self.tree(c, src.scf.get(j).copied().unwrap_or(0) as u32, w as u32, F_SCF, 32 + w as u64, pv * 32 + left, (slot as u64) * 32 + pv, (dl * 18 + ov) * 32 + pv) as u8;
            sfcur[slot] = v;
            g.scf.push(v);
            j += 1;
        }
        self.sf[ch] = sfcur;

        // ---- the big_values pairs
        let short = (lay == 2).then(|| short_lines(h.row));
        let cost_before = self.b.total_cost();
        // how the previous two granules' magnitudes scale to this granule's gain
        let scale1 = GAIN_SCALE[(dg.clamp(-32, 32) + 32) as usize];
        let scale2 = GAIN_SCALE[((dg + pg.dgain).clamp(-32, 32) + 32) as usize];
        let pt = PairTables::new(h, &g);
        g.pairs.reserve(bv as usize);
        for p in 0..bv as usize {
            let tbl = pt.table(p) as usize;
            let mut v = [0i16; 2];
            for k in 0..2 {
                let i = 2 * p + k;
                if tbl == 0 {
                    self.cur[ch][i] = 0;
                    self.scur[ch][i] = 0;
                    continue;
                }
                let (maxv, lb) = (TREE_MAX[TREE_OF[tbl] as usize] as u32, LINBITS[tbl] as u32);
                let truth = src.pairs.get(p).map_or(0, |s| s[k].unsigned_abs() as u32);
                let a1 = if i > 0 { self.cur[ch][i - 1] } else { 0 };
                let a2 = if i > 1 { self.cur[ch][i - 2] } else { 0 };
                let hist = self.history(ch, i, short, same, same2, scale1, scale2);
                let (b0, b1, bm, s_prev, s_two) = (hist[0] as u8, hist[1] as u8, hist[2] as u8, hist[3], hist[4]);
                let o0 = if ch == 1 { self.cur[0][i] } else { bm };
                let first = if k == 1 { 1 + v[0].unsigned_abs().min(15) as u64 } else { 0 };
                let t = tbl as u64;
                // neighbourhood sums: the lines just coded, a window of the previous granule (or
                // short window), the one before it, and the other channel
                let s_cur = a1 as u32 + a2 as u32 + at(&self.cur[ch], i, -3) + at(&self.cur[ch], i, -4);
                // 24 × the magnitude the neighbourhood suggests
                let expect = 3 * s_cur + 2 * s_prev;
                let s_oth = if ch == 1 { at(&self.cur[0], i, -1) + 2 * o0 as u32 + at(&self.cur[0], i, 1) } else { s_two };
                let left = (bv as usize - 1 - p).min(3) as u64;
                let zrun = (1..=15usize).take_while(|&d| d <= i && self.cur[ch][i - d] == 0).count() as u64;
                let mut a = 0u32;
                while a < maxv {
                    let dk = a as u64;
                    let fam = if a == 0 { F_ZERO } else { F_MAG };
                    let td = t * 16 + dk;
                    let ctx = [
                        (td * 4 + q3(a1)) * 4 + q3(a2),
                        ((td * 8 + q8(b0)) * 4 + q3(b1)) * 2 + same as u64,
                        (dk * 64 + posb(i)) * 8 + q8(b0),
                        ((td * 8 + q8(o0)) * 4 + q3(a1)) * 2 + chx,
                        (dk * 16 + ql(s_cur)) * 16 + ql(s_prev),
                        (td * 17 + first) * 8 + q8(a1),
                        td * 16 + ql(s_cur + s_prev),
                        (dk * 16 + ql(s_prev)) * 16 + ql(s_two),
                        (td * 16 + ql(s_oth)) * 4 + q3(a1),
                        (td * 4 + left) * 16 + posb(i) / 4,
                        td * 17 + first,
                        (dk * 16 + zrun) * 8 + q8(b0),
                        td * 64 + posb(i),
                        // the expected magnitude against this step's threshold, shared by all steps
                        (ql(expect * 8 / (24 * a + 12)) * 4 + dk.min(3)) * 2 + (tbl >= 16) as u64,
                    ];
                    let sel = if a == 0 { TREE_OF[tbl] as usize * 2 + k } else { (a as usize).min(15) + 16 * (tbl >= 16) as usize + 32 * k };
                    let sel2 = (ql(s_cur + s_prev) * 4 + dk.min(3)) as usize;
                    let actx = if a == 0 { q8(b0) as usize * 8 + q8(a1) as usize } else { (a as usize).min(7) * 8 + q8(b0) as usize };
                    let before = self.b.cost(fam);
                    let more = self.bit2(c, (truth > a) as u32, fam, &ctx, sel, sel2, actx);
                    self.step_cost[a as usize] += self.b.cost(fam) - before;
                    if more == 0 { break; }
                    a += 1;
                }
                let mut val = a;
                if a == 15 && lb > 0 {
                    let e = self.gamma(c, truth.saturating_sub(15) + 1, F_ESC, 40 + lb as u64, ql(a1 as u32 + a2 as u32), ql(s_prev), first.min(16)) - 1;
                    if e >= 1 << lb { return Err(bad()); }
                    val += e;
                }
                self.cur[ch][i] = val.min(255) as u8;
                self.scur[ch][i] = 0;
                let neg = val != 0 && self.sign(c, src.pairs.get(p).is_some_and(|s| s[k] < 0), h, ch, i, val, same);
                v[k] = if neg { -(val as i16) } else { val as i16 };
            }
            g.pairs.push(v);
        }

        // ---- the count1 quadruples
        let maxq = (LINES as i32 - 2 * bv) / 4;
        let pred = ((pg.end as i32 - 2 * bv) / 4).clamp(0, maxq);
        let oq = og.map_or(0u64, |o| ((((o.end as i32 - 2 * bv) / 4).clamp(0, maxq) - pred).clamp(-8, 8) + 9) as u64);
        let dnq = self.delta(c, src.quads.len() as i32 - pred, F_NQ, 50 + chx, (pg.dnq.clamp(-8, 8) + 8) as u64, oq, (maxq as u64).min(15) + 16 * g.count1table as u64);
        let nq = pred + dnq;
        if !(0..=maxq).contains(&nq) { return Err(bad()); }
        let c1t = g.count1table as u64;
        for q in 0..nq as usize {
            let (mut flags, mut signs, mut nz) = (0u8, 0u8, 0u64);
            for k in 0..4 {
                let i = 2 * bv as usize + 4 * q + k;
                let a1 = if i > 0 { self.cur[ch][i - 1] } else { 0 };
                let a2 = if i > 1 { self.cur[ch][i - 2] } else { 0 };
                let a3 = if i > 2 { self.cur[ch][i - 3] } else { 0 };
                let hist = self.history(ch, i, short, same, same2, 256, 256);
                let (b0, b1, bm, s_prev, s_two) = (hist[0] as u8, hist[1] as u8, hist[2] as u8, hist[3], hist[4]);
                let o0 = if ch == 1 { self.cur[0][i] } else { 0 };
                let recent: u64 = (1..=8).map(|d| if i >= d { (self.cur[ch][i - d] != 0) as u64 } else { 0 }).sum();
                let around: u64 = (0..5).map(|d| if i + d >= 2 && i + d - 2 < LINES { (self.prev[ch][i + d - 2] != 0 && same) as u64 } else { 0 }).sum();
                let truth = (src.quads.get(q).copied().unwrap_or(0) >> (3 - k)) & 1;
                let s_oth = if ch == 1 { at(&self.cur[0], i, -1) + 2 * o0 as u32 + at(&self.cur[0], i, 1) } else { s_two };
                let left = (nq as usize - 1 - q).min(3) as u64;
                let ctx = [
                    ((c1t * 4 + q3(a1)) * 4 + q3(a2)) * 4 + q3(a3),
                    ((q8(b0) * 4 + q3(b1)) * 4 + q3(bm)) * 2 + same as u64,
                    (posb(i) * 4 + q3(b0)) * 2 + (a1 != 0) as u64,
                    (((k as u64 * 4 + nz) * 8 + q8(o0)) * 4 + q3(a1)) * 2 + chx,
                    ((k as u64 * 4 + nz) * 4 + q3(b0)) * 2 + c1t,
                    (recent * 8 + around) * 4 + q3(a1),
                    (ql(s_prev) * 16 + ql(s_two)) * 2 + c1t,
                    (ql(s_oth) * 9 + recent) * 2 + c1t,
                    ((left * 4 + k as u64) * 4 + nz) * 9 + recent,
                    (ql(s_prev + s_two) * 9 + recent) * 4 + q3(a1),
                ];
                let b = self.bit2(c, truth as u32, F_QUAD, &ctx, k * 8 + nz as usize * 2 + c1t as usize, (ql(s_prev) * 4 + q3(a1)) as usize, q3(b0) as usize * 16 + q3(a1) as usize * 4 + q3(a2) as usize);
                self.cur[ch][i] = b as u8;
                self.scur[ch][i] = 0;
                if b == 1 {
                    flags |= 8 >> k;
                    nz += 1;
                    if self.sign(c, (src.quads.get(q).copied().unwrap_or(0) >> (7 - k)) & 1 == 1, h, ch, i, 1, same) { signs |= 8 >> k; }
                }
            }
            g.quads.push(flags | signs << 4);
        }
        let end = 2 * bv as usize + 4 * nq as usize;
        for x in &mut self.cur[ch][end..] { *x = 0; }
        for x in &mut self.scur[ch][end..] { *x = 0; }

        if self.b.stats.is_some() {
            self.layout_cost[lay as usize].0 += self.b.total_cost() - cost_before;
            self.layout_cost[lay as usize].1 += 1;
        }

        // ---- stuffing bits
        if self.bit(c, (src.stuff_bits != 0) as u32, F_SIDE, sc(27, chx, 0, 0, 0), 27, 6) == 1 {
            g.stuff_bits = self.tree(c, src.stuff_bits as u32, 12, F_RAW, 1, 0, 0, 0) as u16;
            if g.stuff_bits == 0 { return Err(bad()); }
            for k in 0..(g.stuff_bits as usize).div_ceil(8) {
                let s = self.byte(c, src.stuff.get(k).copied().unwrap_or(0), 2, 0);
                g.stuff.push(s);
            }
        }
        self.side[ch] = Side {
            ws: g.window_switching, bt: g.block_type, layout: lay, layout2: pg.layout, bv: g.big_values, gain: g.global_gain, sfc: g.scalefac_compress, ts: g.table_select,
            aux: g.aux, pre: g.preflag, scale: g.scalefac_scale, c1t: g.count1table, end: end as u16, dgain: dg, dbv, dnq,
        };
        Ok(g)
    }

    /// Code one frame; `src` is the frame on the encoder side. Returns the frame coded and its
    /// header.
    fn frame<C: Coder>(&mut self, c: &mut C, src: Option<&Frame>) -> Result<(Frame, Hdr), String> {
        let bad = || "mp3 frames: bad frame (corrupt file)".to_string();
        let blank = Frame { hdr: [0xFF, 0, 0, 0], crc: None, body: Body::Raw(vec![]) };
        let src = src.unwrap_or(&blank);
        let mut hdr = [0xFFu8; 4];
        for k in 1..4 { hdr[k] = self.tree(c, src.hdr[k] as u32, 8, F_HDR, k as u64, self.hdr[k] as u64, 0, 0) as u8; }
        let h = mp3::header(&hdr).ok_or_else(bad)?;
        self.hdr = hdr;
        // a protected frame's CRC is recomputed unless it is not the one the frame gives
        let mut crc = None;
        if h.protected && self.bit(c, src.crc.is_some() as u32, F_SIDE, sc(29, 0, (self.flags >> 2 & 1) as u64, 0, 0), 29, 10) == 1 {
            let raw = src.crc.unwrap_or_default();
            crc = Some([self.byte(c, raw[0], 3, 0), self.byte(c, raw[1], 4, 0)]);
        }
        self.flags = self.flags & !4 | (crc.is_some() as u32) << 2;
        let src_parsed = match &src.body { Body::Parsed(p) => Some(&**p), Body::Raw(_) => None };
        let parsed = self.bit(c, src_parsed.is_some() as u32, F_SIDE, sc(30, 0, (self.flags & 1) as u64, 0, 0), 30, 7) == 1;
        self.flags = self.flags & !1 | parsed as u32;
        if !parsed {
            let raw = match &src.body { Body::Raw(s) => &s[..], _ => &[] };
            let side = (0..h.side_len).map(|k| self.byte(c, raw.get(k).copied().unwrap_or(0), 5, k as u64)).collect();
            return Ok((Frame { hdr, crc, body: Body::Raw(side) }, h));
        }
        let blank = Parsed::default();
        let sp = src_parsed.unwrap_or(&blank);
        let mut p = Box::new(Parsed::default());
        if self.bit(c, (sp.gap != 0) as u32, F_SIDE, sc(31, 0, (self.flags >> 1 & 1) as u64, 0, 0), 31, 8) == 1 {
            p.gap = u16::try_from(self.gamma(c, sp.gap as u32, F_RAW, 6, 0, 0, 0)).map_err(|_| bad())?;
            if p.gap == 0 { return Err(bad()); }
        }
        self.flags = self.flags & !2 | ((p.gap != 0) as u32) << 1;
        let nbits = match (h.lsf, h.mono) { (false, true) => 5, (false, false) => 3, (true, true) => 1, (true, false) => 2 };
        p.private = self.tree(c, sp.private as u32, nbits, F_SIDE, 32, self.private as u64, 0, 0) as u8;
        self.private = p.private;
        if !h.lsf {
            for ch in 0..h.channels() {
                p.scfsi[ch] = self.tree(c, sp.scfsi[ch] as u32, 4, F_SIDE, 33 + ch as u64, self.scfsi[ch] as u64, if ch == 1 { 1 + p.scfsi[0] as u64 } else { 0 }, 0) as u8;
                self.scfsi[ch] = p.scfsi[ch];
            }
        }
        for gr in 0..h.granules() {
            for ch in 0..h.channels() {
                p.gr[gr][ch] = self.granule(c, &h, &sp.gr[gr][ch], gr == 1, ch, p.scfsi[ch])?;
            }
            self.prev2 = self.prev;
            self.prev = self.cur;
            self.sprev = self.scur;
        }
        if self.bit(c, (sp.padbits != 0) as u32, F_SIDE, sc(35, 0, 0, 0, 0), 35, 9) == 1 {
            p.padbits = self.tree(c, sp.padbits as u32, 7, F_RAW, 7, 0, 0, 0) as u8;
            if p.padbits == 0 { return Err(bad()); }
        }
        Ok((Frame { hdr, crc, body: Body::Parsed(p) }, h))
    }

    /// Code (encode, or decode when `buf` is the decoder's) one leaf of frame records in
    /// `buf[range]`.
    pub fn code_leaf<C: Coder, B: ByteBuf>(&mut self, c: &mut C, buf: &mut B, range: std::ops::Range<usize>, _params: &[u8]) -> Result<(), String> {
        let enc = buf.reading();
        let mut pos = range.start;
        let mut rec = Vec::new();
        while pos < range.end {
            let src = if enc { Some(Frame::read(&buf.hist(range.end)[pos..])?) } else { None };
            let (f, h) = self.frame(c, src.as_ref().map(|s| &s.0))?;
            rec.clear();
            f.write(&h, &mut rec);
            if pos + rec.len() > range.end { return Err("mp3 frames: leaf overflow (corrupt file)".into()); }
            if let Some((sf, _, n)) = &src {
                // the record must be the one the decoder will write
                if *n != rec.len() || *sf != f { return Err("mp3 frames: non-canonical record".into()); }
            } else {
                for (k, &x) in rec.iter().enumerate() { buf.put(pos + k, x); }
            }
            pos += rec.len();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cm::coder::{Decoder, Encoder};
    use crate::cm::predictor::{ReadBuf, WriteBuf};

    fn rng(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }

    /// The model decodes its own output back to the frame records, beats their size by far, and
    /// corrupt input is an error or other bytes — never a panic.
    #[test]
    fn frames_roundtrip_and_reject_corruption() {
        for (file, most) in [(&include_bytes!("../../tests/fixtures/tone_stereo.mp3")[..], 80), (&include_bytes!("../../tests/fixtures/tone_mono_lsf.mp3")[..], 90)] {
            let start = file.iter().position(|&b| b == 0xFF).unwrap();
            let sp = mp3::split(&file[start..]).unwrap();
            let leaf = &sp.frames;
            let mut enc = Encoder::new();
            Mp3Model::new().code_leaf(&mut enc, &mut ReadBuf(leaf), 0..leaf.len(), &[]).unwrap();
            let coded = enc.finish();
            assert!(coded.len() * 100 < sp.len * most, "{} bytes of frames coded to {}", sp.len, coded.len());
            let mut back = vec![0u8; leaf.len()];
            Mp3Model::new().code_leaf(&mut Decoder::new(&coded), &mut WriteBuf(&mut back), 0..leaf.len(), &[]).unwrap();
            assert!(&back == leaf, "the model did not decode its own output");
            let mut s = 3u64;
            for _ in 0..60 {
                let mut bad = coded.clone();
                let at = (rng(&mut s) % bad.len() as u64) as usize;
                bad[at] ^= 1 << (rng(&mut s) % 8);
                let mut out = vec![0u8; leaf.len()];
                let _ = Mp3Model::new().code_leaf(&mut Decoder::new(&bad), &mut WriteBuf(&mut out), 0..leaf.len(), &[]);
            }
        }
    }
}
