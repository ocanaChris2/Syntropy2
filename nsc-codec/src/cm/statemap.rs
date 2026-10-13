//! Adaptive probability maps (integer-only).
//!
//! * [`StateMap`]: context → probability, learned online with a count-based rate 1/(n+1.5) that
//!   settles to a fixed floor (`limit`), so early observations move it fast and it stays adaptive.
//! * [`Apm`]: secondary estimation (SSE). It refines an input probability given a context by
//!   interpolating between 33 buckets laid out on the stretch axis; both neighbouring buckets learn.

use super::squash::{squash, stretch};
use super::state::TABLE;

/// round(65536 / (n + 1.5)) for the StateMap learning rate.
const fn build_recip() -> [u32; 1024] {
    let mut t = [0u32; 1024];
    let mut n = 0;
    while n < 1024 {
        t[n] = ((131072 + (2 * n as u64 + 3) / 2) / (2 * n as u64 + 3)) as u32;
        n += 1;
    }
    t
}
pub static RECIP: [u32; 1024] = build_recip();

/// Entries pack a 22-bit probability (P(1)·2^22) and a 10-bit count: `p22 << 10 | n`.
pub struct StateMap {
    t: Vec<u32>,
    limit: u32,
    cx: usize,
}

impl StateMap {
    /// `n` contexts, all starting at p = 0.5.
    pub fn new(n: usize, limit: u32) -> Self {
        StateMap { t: vec![(1u32 << 31) & !1023; n], limit: limit.min(1023), cx: 0 }
    }

    /// A map over bit-history states (`n` must be a multiple of 256): each entry starts at the
    /// (n1 + 1/2)/(n0 + n1 + 1) estimate implied by its state.
    pub fn new_states(n: usize, limit: u32) -> Self {
        let mut t = vec![0u32; n];
        for (i, e) in t.iter_mut().enumerate() {
            let s = i & 255;
            let (n0, n1) = (TABLE.n0[s] as u64, TABLE.n1[s] as u64);
            let p22 = ((2 * n1 + 1) << 22) / (2 * (n0 + n1) + 2);
            *e = ((p22 as u32).min((1 << 22) - 1) << 10) | 0;
        }
        StateMap { t, limit: limit.min(1023), cx: 0 }
    }

    /// Probability (P(1)·65536) for context `cx`; remembers `cx` for the next `update`.
    #[inline]
    pub fn p(&mut self, cx: usize) -> u32 {
        self.cx = cx;
        (self.t[cx] >> 16).clamp(1, 65535)
    }

    /// Probability for context `cx` without remembering it (pair with [`StateMap::update_at`]).
    #[inline]
    pub fn p_at(&self, cx: usize) -> u32 { (self.t[cx] >> 16).clamp(1, 65535) }

    #[inline]
    pub fn update(&mut self, bit: u32) { self.update_at(self.cx, bit) }

    #[inline]
    pub fn update_at(&mut self, cx: usize, bit: u32) {
        let e = &mut self.t[cx];
        let n = *e & 1023;
        let p22 = (*e >> 10) as i64;
        let target: i64 = if bit != 0 { (1 << 22) - 1 } else { 0 };
        let p22 = p22 + (((target - p22) * RECIP[n as usize] as i64) >> 16);
        let n = if n < self.limit { n + 1 } else { self.limit };
        *e = ((p22 as u32) << 10) | n;
    }
}

/// Secondary symbol estimation over `n` contexts, 33 interpolation buckets each.
pub struct Apm {
    t: Vec<u16>,
    idx: usize,
    rate: u32,
}

impl Apm {
    pub fn new(n: usize, rate: u32) -> Self {
        let mut t = vec![0u16; n * 33];
        for i in 0..n {
            for j in 0..33 {
                t[i * 33 + j] = squash((j as i32 - 16) * 128).min(65535) as u16;
            }
        }
        Apm { t, idx: 0, rate }
    }

    /// Prefetch context `cx`'s row (its 33 buckets span at most two cache lines).
    #[inline]
    pub fn prefetch(&self, cx: usize) {
        #[cfg(target_arch = "x86_64")]
        {
            let at = cx * 33;
            if at + 33 <= self.t.len() {
                // SAFETY: prefetch is only a hint, and both addresses are inside `t`.
                unsafe {
                    let p = self.t.as_ptr().add(at) as *const i8;
                    std::arch::x86_64::_mm_prefetch(p, std::arch::x86_64::_MM_HINT_T0);
                    std::arch::x86_64::_mm_prefetch(p.add(64), std::arch::x86_64::_MM_HINT_T0);
                }
            }
        }
    }

    /// Refine probability `p` (P(1)·65536) in context `cx`.
    #[inline]
    pub fn p(&mut self, p: u32, cx: usize) -> u32 {
        let s = (stretch(p) + 2048) as usize; // 1..=4095
        let lo = s >> 7;
        let w = (s & 127) as u32;
        self.idx = cx * 33 + lo;
        let a = self.t[self.idx] as u32;
        let b = self.t[self.idx + 1] as u32;
        ((a * (128 - w) + b * w) >> 7).clamp(1, 65535)
    }

    /// Both buckets move toward the bit. The target overshoots by `2^rate - 1` so the integer step
    /// never stalls short of the extremes (a plain `(g - v) >> rate` would stop 2^rate - 1 away).
    #[inline]
    pub fn update(&mut self, bit: u32) {
        let g: i32 = if bit != 0 { 65535 + (1 << self.rate) - 1 } else { -(1 << self.rate) + 1 };
        for k in 0..2 {
            let e = &mut self.t[self.idx + k];
            let v = *e as i32;
            *e = (v + ((g - v) >> self.rate)).clamp(0, 65535) as u16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statemap_learns() {
        let mut m = StateMap::new(4, 1023);
        for _ in 0..2000 { m.p(1); m.update(1); }
        assert!(m.p(1) > 64000);
        for _ in 0..2000 { m.p(2); m.update(0); }
        assert!(m.p(2) < 1500);
        assert_eq!(m.p(3), 32768);
    }

    #[test]
    fn apm_identity_then_adapts() {
        let mut a = Apm::new(2, 6);
        let p = a.p(40000, 0);
        assert!((p as i32 - 40000).abs() < 600, "{p}");
        for _ in 0..500 { a.p(40000, 1); a.update(1); }
        assert!(a.p(40000, 1) > 60000);
    }
}
