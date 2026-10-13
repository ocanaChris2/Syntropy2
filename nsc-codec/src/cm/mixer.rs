//! Gated linear mixer: an online-trained single-layer network per selected weight set, plus a
//! second layer that blends the first layer's outputs.
//!
//! Inputs are logits (1/256 nat, clamped to ±2047) stored as i16; weights are i16 with 1.0 = 8192.
//! Each "selector" picks one weight set by a small context (partial byte, match state, ...), and
//! every chosen set produces its own prediction. Those predictions are mixed again by a final
//! weight set chosen by its own context. All sets learn by gradient descent on coding cost.
//!
//! Exactness: the kernels are defined by what the AVX2 lanes compute, and the scalar twins
//! reproduce that integer arithmetic exactly:
//! * dot: per pair `(x0·w0 + x1·w1) >> 8`, summed with wrapping i32 adds. With |x| ≤ 2047 no pair
//!   or sum can overflow.
//! * train: `w = sat16(w + ((mulhi16(sat16(2x), err) + 1) >> 1))`.
use crate::simd::{self, Level};

pub const MAX_ST: i32 = 2047;

#[inline]
fn pad16(n: usize) -> usize { (n + 15) & !15 }

pub struct Mixer {
    n: usize,
    x: Vec<i16>,
    nx: usize,
    w: Vec<i16>,
    sel_base: Vec<usize>,
    sel_size: Vec<usize>,
    chosen: Vec<usize>,
    st: Vec<i32>,
    pr: Vec<u32>,
    // second layer
    n2: usize,
    x2: Vec<i16>,
    w2: Vec<i16>,
    sets2: usize,
    chosen2: usize,
    pr_final: u32,
    lr: i32,
    lr2: i32,
    // Warm-up: the effective rates start BOOST× higher and decay toward `lr`/`lr2` over roughly
    // WARM bits, so the mixer converges fast at first (highly redundant data) yet settles at the
    // low steady-state rate that suits typical data. Recomputed every 1024 updates.
    updates: u64,
    lr_eff: i32,
    lr2_eff: i32,
    simd: Level,
}

const BOOST: i64 = 8;
const WARM: i64 = 1 << 17;

impl Mixer {
    /// `max_inputs`: inputs added per bit (upper bound). `selectors`: number of weight sets per
    /// selector. `final_sets`: weight sets of the second layer.
    pub fn new(max_inputs: usize, selectors: &[usize], final_sets: usize, lr: i32, lr2: i32) -> Self {
        let n = pad16(max_inputs.max(1));
        let mut sel_base = vec![];
        let mut total = 0;
        for &s in selectors {
            sel_base.push(total);
            total += s.max(1);
        }
        let k = selectors.len().max(1);
        let n2 = pad16(k);
        let mut w2 = vec![0i16; final_sets.max(1) * n2];
        for set in 0..final_sets.max(1) {
            for j in 0..k {
                w2[set * n2 + j] = (8192 / k as i32) as i16;
            }
        }
        Mixer {
            n,
            x: vec![0; n],
            nx: 0,
            w: vec![0; total * n],
            sel_base,
            sel_size: selectors.iter().map(|&s| s.max(1)).collect(),
            chosen: vec![0; k],
            st: vec![0; k],
            pr: vec![32768; k],
            n2,
            x2: vec![0; n2],
            w2,
            sets2: final_sets.max(1),
            chosen2: 0,
            pr_final: 32768,
            lr,
            lr2,
            updates: 0,
            lr_eff: lr * BOOST as i32,
            lr2_eff: lr2 * BOOST as i32,
            simd: simd::level(),
        }
    }

    #[inline]
    pub fn add(&mut self, v: i32) {
        debug_assert!(self.nx < self.n, "mixer input overflow");
        self.x[self.nx] = v.clamp(-MAX_ST, MAX_ST) as i16;
        self.nx += 1;
    }

    /// Reserve the next `k` input slots and return them for direct writing (values must already
    /// lie within ±2047).
    #[inline]
    pub fn take(&mut self, k: usize) -> &mut [i16] {
        let start = self.nx;
        self.nx += k;
        &mut self.x[start..start + k]
    }

    /// Choose the weight set for selector `s`. Callers pass `cx` below the selector size; the clamp
    /// only guards memory safety (a `%` here would cost an integer division per selector per bit).
    #[inline]
    pub fn set(&mut self, s: usize, cx: usize) {
        debug_assert!(cx < self.sel_size[s]);
        self.chosen[s] = self.sel_base[s] + cx.min(self.sel_size[s] - 1);
    }

    #[inline]
    pub fn set_final(&mut self, cx: usize) {
        debug_assert!(cx < self.sets2);
        self.chosen2 = cx.min(self.sets2 - 1);
    }

    pub fn p(&mut self) -> u32 {
        for v in &mut self.x[self.nx..] { *v = 0; }
        let n = self.n;
        let k = self.chosen.len();
        for s in 0..k {
            let off = self.chosen[s] * n;
            let d = dot(self.simd, &self.x, &self.w[off..off + n]);
            let st = (d >> 5).clamp(-MAX_ST, MAX_ST);
            self.st[s] = st;
            self.pr[s] = super::squash::squash(st);
        }
        if k == 1 {
            self.pr_final = self.pr[0];
            return self.pr_final;
        }
        for s in 0..k { self.x2[s] = self.st[s] as i16; }
        let off = self.chosen2 * self.n2;
        let d = dot(self.simd, &self.x2, &self.w2[off..off + self.n2]);
        self.pr_final = super::squash::squash((d >> 5).clamp(-MAX_ST, MAX_ST));
        self.pr_final
    }

    pub fn update(&mut self, bit: u32) {
        if self.updates & 1023 == 0 {
            let t = self.updates as i64;
            let f = |lr: i32| (lr as i64 * (WARM + t + (BOOST - 1) * WARM) / (WARM + t)) as i32;
            self.lr_eff = f(self.lr);
            self.lr2_eff = f(self.lr2);
        }
        self.updates += 1;
        let n = self.n;
        let target = (bit as i32) << 12;
        let k = self.chosen.len();
        for s in 0..k {
            let err = (((target - (self.pr[s] >> 4) as i32) * self.lr_eff) >> 3).clamp(-32767, 32767);
            if err != 0 {
                let off = self.chosen[s] * n;
                train(self.simd, &self.x, &mut self.w[off..off + n], err);
            }
        }
        if k > 1 {
            let err = (((target - (self.pr_final >> 4) as i32) * self.lr2_eff) >> 3).clamp(-32767, 32767);
            if err != 0 {
                let off = self.chosen2 * self.n2;
                let n2 = self.n2;
                train(self.simd, &self.x2, &mut self.w2[off..off + n2], err);
            }
        }
        self.nx = 0;
    }
}

// ------------------------------------------------------------------ kernels
#[inline]
pub fn dot(level: Level, x: &[i16], w: &[i16]) -> i32 {
    debug_assert_eq!(x.len() % 16, 0);
    debug_assert!(w.len() >= x.len());
    #[cfg(target_arch = "x86_64")]
    {
        if level == Level::Avx2 {
            // SAFETY: Level::Avx2 is only selected when the CPU supports AVX2; lengths are
            // multiples of 16 and `w` is at least as long as `x`.
            return unsafe { dot_avx2(x, w) };
        }
    }
    let _ = level;
    dot_scalar(x, w)
}

#[inline]
pub fn train(level: Level, x: &[i16], w: &mut [i16], err: i32) {
    debug_assert_eq!(x.len() % 16, 0);
    #[cfg(target_arch = "x86_64")]
    {
        if level == Level::Avx2 {
            // SAFETY: as in `dot`.
            unsafe { train_avx2(x, w, err) };
            return;
        }
    }
    let _ = level;
    train_scalar(x, w, err)
}

pub fn dot_scalar(x: &[i16], w: &[i16]) -> i32 {
    let mut sum = 0i32;
    let mut k = 0;
    while k + 1 < x.len() {
        let pair = x[k] as i32 * w[k] as i32 + x[k + 1] as i32 * w[k + 1] as i32;
        sum = sum.wrapping_add(pair >> 8);
        k += 2;
    }
    sum
}

pub fn train_scalar(x: &[i16], w: &mut [i16], err: i32) {
    let e = err.clamp(-32768, 32767);
    for k in 0..x.len() {
        let t = (x[k] as i32 * 2).clamp(-32768, 32767);
        let m = (t * e) >> 16;
        let m = (m + 1).clamp(-32768, 32767) >> 1;
        w[k] = (w[k] as i32 + m).clamp(-32768, 32767) as i16;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(x: &[i16], w: &[i16]) -> i32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_si256();
    let mut k = 0;
    while k < x.len() {
        // SAFETY: k + 16 <= x.len() <= w.len() (lengths are multiples of 16).
        let xv = unsafe { _mm256_loadu_si256(x.as_ptr().add(k) as *const __m256i) };
        let wv = unsafe { _mm256_loadu_si256(w.as_ptr().add(k) as *const __m256i) };
        acc = _mm256_add_epi32(acc, _mm256_srai_epi32(_mm256_madd_epi16(xv, wv), 8));
        k += 16;
    }
    let hi = _mm256_extracti128_si256(acc, 1);
    let lo = _mm256_castsi256_si128(acc);
    let s = _mm_add_epi32(lo, hi);
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_00_00_01));
    _mm_cvtsi128_si32(s)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn train_avx2(x: &[i16], w: &mut [i16], err: i32) {
    use std::arch::x86_64::*;
    let e = _mm256_set1_epi16(err.clamp(-32768, 32767) as i16);
    let one = _mm256_set1_epi16(1);
    let mut k = 0;
    while k < x.len() {
        // SAFETY: k + 16 <= x.len() <= w.len().
        unsafe {
            let xv = _mm256_loadu_si256(x.as_ptr().add(k) as *const __m256i);
            let t = _mm256_adds_epi16(xv, xv);
            let m = _mm256_mulhi_epi16(t, e);
            let m = _mm256_srai_epi16(_mm256_adds_epi16(m, one), 1);
            let wp = w.as_mut_ptr().add(k) as *mut __m256i;
            _mm256_storeu_si256(wp, _mm256_adds_epi16(_mm256_loadu_si256(wp), m));
        }
        k += 16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }

    #[test]
    fn simd_equals_scalar() {
        if simd::level() != Level::Avx2 { return; }
        let mut s = 7u64;
        for round in 0..2000 {
            let n = 16 * (1 + (round % 12));
            let x: Vec<i16> = (0..n).map(|_| ((rng(&mut s) % 4095) as i32 - 2047) as i16).collect();
            let mut w: Vec<i16> = (0..n).map(|_| rng(&mut s) as i16).collect();
            // include extreme weights
            if round % 7 == 0 { for v in w.iter_mut().step_by(3) { *v = if rng(&mut s) & 1 == 0 { i16::MAX } else { i16::MIN }; } }
            assert_eq!(dot_scalar(&x, &w), unsafe { dot_avx2(&x, &w) }, "dot round {round}");
            let err = ((rng(&mut s) % 65535) as i32) - 32767;
            let mut w1 = w.clone();
            let mut w2 = w.clone();
            train_scalar(&x, &mut w1, err);
            unsafe { train_avx2(&x, &mut w2, err) };
            assert_eq!(w1, w2, "train round {round}");
        }
    }

    #[test]
    fn learns_a_constant_input() {
        let mut m = Mixer::new(2, &[1], 1, 24, 8);
        for _ in 0..3000 {
            m.add(512);
            m.add(-300);
            m.set(0, 0);
            m.p();
            m.update(1);
        }
        m.add(512);
        m.add(-300);
        m.set(0, 0);
        assert!(m.p() > 60000);
    }
}
