//! The coding core shared by the dedicated models that code parsed streams (MP3, AAC, …): binary
//! decisions grouped in families, each mixing a few hashed-context predictions with an online
//! mixer (two weight sets, blended per family) and refining the result with an APM; and on top of
//! that, binary trees, Elias-gamma numbers and signed differences. Integer-only.
use super::coder::Coder;
use super::mixer::Mixer;
use super::squash::stretch;
use super::statemap::{Apm, StateMap};

/// Context inputs of a side-information decision ([`sc`]).
pub const INPUTS: usize = 6;
/// Mixer weight sets (per selector) and APM contexts per family.
pub const SEL: usize = 64;
pub const ACTX: usize = 64;

pub struct Bins {
    sm: Vec<StateMap>, // families × maxin
    mixer: Mixer,
    apm: Apm,
    /// A second refinement, by the second weight-set selector (models that ask for it).
    apm2: Option<Apm>,
    bits: &'static [u32],
    nin: &'static [usize],
    maxin: usize,
    /// Developer statistics: coding cost per family (bits·65536), when enabled.
    pub stats: Option<Vec<u64>>,
}

/// Magnitude classes: 0, 1, 2, 3, 4-5, 6-8, 9-14, 15+.
#[inline]
pub fn q8(x: u8) -> u64 {
    const Q: [u8; 16] = [0, 1, 2, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7];
    Q[x.min(15) as usize] as u64
}

#[inline]
pub fn q3(x: u8) -> u64 { x.min(3) as u64 }

/// Sum classes (16): exact up to 5, then two per octave.
#[inline]
pub fn ql(x: u32) -> u64 {
    if x < 5 { return x as u64; }
    let l = 31 - x.leading_zeros();
    (5 + 2 * (l - 2) + ((x >> (l - 1)) & 1)).min(15) as u64
}

/// 256 · 2^(-3·d/16) for a step-size change of `d` (index `d + 32`, d clamped to ±32): how
/// quantized magnitudes scale when the gain or scale factor moves.
const fn build_gain_scale() -> [u32; 65] {
    // 2^(-3/16) in 1/2^16 units, applied repeatedly from the middle outwards
    let mut t = [256u32 << 8; 65];
    let mut k = 1;
    while k <= 32 {
        t[32 + k] = ((t[32 + k - 1] as u64 * 57_549) >> 16) as u32; // × 2^(-3/16)
        t[32 - k] = ((t[32 - k + 1] as u64 * 74_632) >> 16) as u32; // × 2^(+3/16)
        k += 1;
    }
    let mut i = 0;
    while i < 65 {
        t[i] >>= 8;
        i += 1;
    }
    t
}
pub static GAIN_SCALE: [u32; 65] = build_gain_scale();

/// Contexts of a side-information decision: field `id`, tree node, and three small values
/// (below 2^16).
#[inline]
pub fn sc(id: u64, node: u64, a: u64, b: u64, x: u64) -> [u64; INPUTS] {
    let k = id << 56 | node << 40;
    [k, k | a, k | b, k | a << 16 | b, k | x, k | a << 16 | x]
}

impl Bins {
    /// `bits[f]`: log2 of family f's context table size (per input); `nin[f]`: how many context
    /// inputs its decisions give.
    pub fn new(bits: &'static [u32], nin: &'static [usize]) -> Self { Self::adapting(bits, nin, 1023, false) }

    /// [`Bins::new`] with the probability maps' count limit (lower adapts faster), and
    /// optionally a second refinement stage keyed by the second selector.
    pub fn adapting(bits: &'static [u32], nin: &'static [usize], limit: u32, two_stage: bool) -> Self {
        let fams = bits.len();
        let maxin = nin.iter().copied().max().unwrap_or(INPUTS);
        Bins {
            sm: (0..fams * maxin).map(|i| StateMap::new(if i % maxin < nin[i / maxin] { 1 << bits[i / maxin] } else { 1 }, limit)).collect(),
            mixer: Mixer::new(maxin + 1, &[fams * SEL, fams * SEL], fams, 6, 6),
            apm: Apm::new(fams * ACTX, 7),
            apm2: two_stage.then(|| Apm::new(fams * SEL, 7)),
            bits,
            nin,
            maxin,
            stats: None,
        }
    }

    /// Code one binary decision of family `fam` with [`INPUTS`] contexts, mixer weight set `sel`
    /// and APM context `actx`; returns the bit coded.
    #[inline]
    pub fn bit<C: Coder>(&mut self, c: &mut C, bit: u32, fam: usize, ctx: [u64; INPUTS], sel: usize, actx: usize) -> u32 {
        self.bit2(c, bit, fam, &ctx, sel, sel, actx)
    }

    /// [`Bins::bit`] with the family's own number of contexts and a second weight set.
    #[inline]
    pub fn bit2<C: Coder>(&mut self, c: &mut C, bit: u32, fam: usize, ctx: &[u64], sel: usize, sel2: usize, actx: usize) -> u32 {
        debug_assert!(ctx.len() == self.nin[fam]);
        let sh = 64 - self.bits[fam];
        let base = fam * self.maxin;
        let mut idx = [0usize; 24];
        for (i, &cx) in ctx.iter().enumerate() {
            idx[i] = (cx.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> sh) as usize;
            self.mixer.add(stretch(self.sm[base + i].p_at(idx[i])));
        }
        self.mixer.add(256);
        self.mixer.set(0, fam * SEL + sel.min(SEL - 1));
        self.mixer.set(1, fam * SEL + sel2.min(SEL - 1));
        self.mixer.set_final(fam);
        let pm = self.mixer.p();
        let pa = self.apm.p(pm, fam * ACTX + actx.min(ACTX - 1));
        let p = match self.apm2.as_mut() {
            Some(a) => (2 * pm + 3 * pa + 3 * a.p(pm, fam * SEL + sel2.min(SEL - 1))) / 8,
            None => (pm + 3 * pa) / 4,
        }.clamp(1, 65535);
        let b = c.code(bit, p);
        if let Some(st) = self.stats.as_mut() {
            let pb = if b == 1 { p } else { 65536 - p } as u64;
            st[fam] += (16 << 16) - crate::nsc3::block::log2_fx(pb);
        }
        for i in 0..ctx.len() { self.sm[base + i].update_at(idx[i], b); }
        self.mixer.update(b);
        self.apm.update(b);
        if let Some(a) = self.apm2.as_mut() { a.update(b); }
        b
    }

    /// Code the `n`-bit value `v` as a binary tree, most significant bit first.
    #[allow(clippy::too_many_arguments)]
    pub fn tree<C: Coder>(&mut self, c: &mut C, v: u32, n: u32, fam: usize, id: u64, a: u64, b: u64, x: u64) -> u32 {
        let mut node = 1u32;
        for i in (0..n).rev() {
            let bit = self.bit(c, (v >> i) & 1, fam, sc(id, node as u64, a, b, x), (id as usize * 5 + (n - 1 - i) as usize) % SEL, (node as usize) % ACTX);
            node = node << 1 | bit;
        }
        node - (1 << n)
    }

    /// Code `n >= 1` (below 2^15) as its bit length in unary, then the bits under the leading one.
    #[allow(clippy::too_many_arguments)]
    pub fn gamma<C: Coder>(&mut self, c: &mut C, n: u32, fam: usize, id: u64, a: u64, b: u64, x: u64) -> u32 {
        let len = 31 - n.max(1).leading_zeros();
        let mut got = 0u32;
        while got < 15 {
            let more = self.bit(c, (len > got) as u32, fam, sc(id, 2 + got as u64, a, b, x), (id as usize * 5 + got as usize) % SEL, got as usize);
            if more == 0 { break; }
            got += 1;
        }
        let mut m = 1u32;
        for i in (0..got).rev() {
            let top = if i + 1 == got { m as u64 } else { 0 };
            let bit = self.bit(c, (n >> i) & 1, fam, sc(id, 32 + (got * 16 + i) as u64, a * (i + 1 == got) as u64, top, x * (i + 1 == got) as u64), (id as usize * 5 + 20 + got as usize) % SEL, 32 + i as usize);
            m = m << 1 | bit;
        }
        m
    }

    /// Code a signed integer: a zero flag, the sign, the magnitude ([`Bins::gamma`]).
    #[allow(clippy::too_many_arguments)]
    pub fn delta<C: Coder>(&mut self, c: &mut C, d: i32, fam: usize, id: u64, a: u64, b: u64, x: u64) -> i32 {
        let sel = id as usize * 5 % SEL;
        if self.bit(c, (d != 0) as u32, fam, sc(id, 0, a, b, x), sel, 62) == 0 { return 0; }
        let neg = self.bit(c, (d < 0) as u32, fam, sc(id, 1, a, b, x), sel + 1, 63);
        let m = self.gamma(c, d.unsigned_abs(), fam, id, a, b, x) as i32;
        if neg == 1 { -m } else { m }
    }

    /// The cost so far of family `fam` (0 without statistics).
    pub fn cost(&self, fam: usize) -> u64 { self.stats.as_ref().map_or(0, |s| s[fam]) }

    /// The cost so far of all families.
    pub fn total_cost(&self) -> u64 { self.stats.as_ref().map_or(0, |s| s.iter().sum()) }
}
