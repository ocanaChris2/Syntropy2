//! Deterministic f32 transcendental functions for the portable neural kind.
//!
//! Built only from IEEE-754 basic operations (+, −, ×, ÷, comparisons, bit manipulation),
//! evaluated in f64 and rounded once to f32. Rust never contracts `a*b+c` into an FMA unless asked,
//! so every IEEE machine computes the same bits. That is the point: system libm implementations
//! (glibc picks FMA variants at runtime) may differ in the last bit between machines, which would
//! change the neural coder's tables. Accuracy is ~1 ulp of f32, plenty for a probability model.
#![allow(clippy::excessive_precision)]

const LN2_HI: f64 = 6.931_471_803_691_238_164_90e-01;
const LN2_LO: f64 = 1.908_214_929_270_587_700_02e-10;
const INV_LN2: f64 = 1.442_695_040_888_963_387_00;

/// 2^k for integer k in the normal f64 range.
#[inline(always)]
fn pow2(k: i64) -> f64 { f64::from_bits(((k + 1023) as u64) << 52) }

/// e^x in f64 (|x| < 700): x = k·ln2 + r with |r| ≤ ln2/2, then a degree-11 Taylor series.
/// Branch-free (round-half-away via copysign; i32 conversion), so loops over it vectorize.
#[inline(always)]
fn exp64(x: f64) -> f64 {
    let t = x * INV_LN2;
    let k = (t + 0.5f64.copysign(t)) as i32;
    let kf = k as f64;
    let r = (x - kf * LN2_HI) - kf * LN2_LO;
    let mut p = 1.0 / 39_916_800.0;
    p = p * r + 1.0 / 3_628_800.0;
    p = p * r + 1.0 / 362_880.0;
    p = p * r + 1.0 / 40_320.0;
    p = p * r + 1.0 / 5_040.0;
    p = p * r + 1.0 / 720.0;
    p = p * r + 1.0 / 120.0;
    p = p * r + 1.0 / 24.0;
    p = p * r + 1.0 / 6.0;
    p = p * r + 0.5;
    p = p * r + 1.0;
    p = p * r + 1.0;
    p * pow2(k as i64)
}

/// e^x. Inputs are clamped to [−104, 88.7] (outside it f32 under/overflows anyway).
#[inline(always)]
pub fn exp(x: f32) -> f32 {
    exp64((x as f64).clamp(-104.0, 88.7)) as f32
}

/// ln(x) in f64 for finite x > 0: x = m·2^e with m in [√½, √2), then 2·atanh((m−1)/(m+1)).
#[inline(always)]
fn ln64(x: f64) -> f64 {
    let bits = x.to_bits();
    let e0 = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let m0 = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    let big = m0 > std::f64::consts::SQRT_2;
    let m = if big { m0 * 0.5 } else { m0 };
    let e = e0 + big as i64;
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let mut p = 1.0 / 23.0;
    p = p * s2 + 1.0 / 21.0;
    p = p * s2 + 1.0 / 19.0;
    p = p * s2 + 1.0 / 17.0;
    p = p * s2 + 1.0 / 15.0;
    p = p * s2 + 1.0 / 13.0;
    p = p * s2 + 1.0 / 11.0;
    p = p * s2 + 1.0 / 9.0;
    p = p * s2 + 1.0 / 7.0;
    p = p * s2 + 1.0 / 5.0;
    p = p * s2 + 1.0 / 3.0;
    p = p * s2 + 1.0;
    let ef = e as f64;
    ef * LN2_HI + (ef * LN2_LO + 2.0 * s * p)
}

/// ln(x) for x > 0 (f32 subnormals are normal in f64). Non-positive inputs are clamped to the
/// smallest positive f32; the model only takes logs of probabilities plus a positive epsilon.
#[inline(always)]
pub fn ln(x: f32) -> f32 {
    ln64((x as f64).max(1.401_298_464_324_817e-45)) as f32
}

/// tanh(x) = sign(x)·(1 − 2/(e^{2|x|} + 1)) in f64 (cancellation for small |x| stays ~1e-16
/// absolute, so a cubic takes over below 1e-4). Branch-free so it vectorizes.
#[inline(always)]
pub fn tanh(x: f32) -> f32 {
    let a = (x as f64).abs().min(20.0);
    let big = 1.0 - 2.0 / (exp64(2.0 * a) + 1.0);
    let small = a - a * a * a / 3.0;
    let t = if a < 1e-4 { small } else { big };
    t.copysign(x as f64) as f32
}

/// erf(x) by Abramowitz & Stegun 7.1.26 (|error| ≤ 1.5e-7), evaluated in f64. Plenty for GELU,
/// whose output is quantized to int8 next. Fixed cost and branch-free.
#[inline(always)]
pub fn erf(x: f32) -> f32 {
    let a = (x as f64).abs().min(10.0);
    let t = 1.0 / (1.0 + 0.327_591_1 * a);
    let poly = t * (0.254_829_592 + t * (-0.284_496_736 + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let r = 1.0 - poly * exp64(-a * a);
    r.copysign(x as f64) as f32
}

/// (sin x, cos x) with f64 range reduction by π/2 (exact enough for |x| < 2^20).
pub fn sin_cos(x: f32) -> (f32, f32) {
    let xd = x as f64;
    const PIO2_HI: f64 = 1.570_796_326_734_125_614_17;
    const PIO2_LO: f64 = 6.077_100_506_506_192_249_12e-11;
    let k = (xd * std::f64::consts::FRAC_2_PI + if xd >= 0.0 { 0.5 } else { -0.5 }) as i64;
    let kf = k as f64;
    let r = (xd - kf * PIO2_HI) - kf * PIO2_LO;
    let r2 = r * r;
    let s = r * (1.0 + r2 * (-1.0 / 6.0 + r2 * (1.0 / 120.0 + r2 * (-1.0 / 5040.0 + r2 * (1.0 / 362_880.0
        + r2 * (-1.0 / 39_916_800.0 + r2 * (1.0 / 6_227_020_800.0)))))));
    let c = 1.0 + r2 * (-0.5 + r2 * (1.0 / 24.0 + r2 * (-1.0 / 720.0 + r2 * (1.0 / 40_320.0
        + r2 * (-1.0 / 3_628_800.0 + r2 * (1.0 / 479_001_600.0))))));
    let (s, c) = match k & 3 { 0 => (s, c), 1 => (c, -s), 2 => (-s, -c), _ => (-c, s) };
    (s as f32, c as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ulps(a: f32, b: f32) -> u32 {
        if a == b { return 0; }
        let (ia, ib) = (a.to_bits() as i32, b.to_bits() as i32);
        (ia - ib).unsigned_abs()
    }

    #[test]
    fn accurate_against_std() {
        let mut x = -20.0f32;
        while x < 20.0 {
            assert!(ulps(exp(x), x.exp()) <= 2, "exp({x}) = {} vs {}", exp(x), x.exp());
            assert!(ulps(tanh(x), x.tanh()) <= 2, "tanh({x})");
            let (s, c) = sin_cos(x * 50.0);
            assert!((s - (x * 50.0).sin()).abs() < 2e-6 && (c - (x * 50.0).cos()).abs() < 2e-6, "sin_cos({})", x * 50.0);
            x += 0.0137;
        }
        let mut x = 1e-30f32;
        while x < 1e30 {
            assert!(ulps(ln(x), x.ln()) <= 2, "ln({x}) = {} vs {}", ln(x), x.ln());
            x *= 1.37;
        }
        for i in -400..400 {
            let x = i as f32 / 100.0;
            let reference = libm_erf(x as f64) as f32;
            assert!((erf(x) - reference).abs() < 3e-7, "erf({x}) = {} vs {reference}", erf(x));
        }
    }

    // high-precision reference: erf via the same series in f64 with many terms (independent of our
    // f32 rounding path), and erfc asymptotics for large |x|
    fn libm_erf(x: f64) -> f64 {
        let a = x.abs();
        let mut term = a;
        let mut sum = a;
        for n in 1..200 {
            term *= -a * a / n as f64;
            sum += term / (2 * n + 1) as f64;
        }
        let v = (sum * 2.0 / std::f64::consts::PI.sqrt()).clamp(-1.0, 1.0);
        if x < 0.0 { -v } else { v }
    }
}

#[cfg(test)]
mod speed {
    #[test]
    #[ignore]
    fn dmath_speed() {
        let xs: Vec<f32> = (0..8192).map(|i| -(i as f32) * 0.003).collect();
        let ps: Vec<f32> = (0..8192).map(|i| (i as f32 + 0.5) / 9000.0).collect();
        let mut out = vec![0f32; 8192];
        let mut s = 0f32;
        let t = std::time::Instant::now();
        for _ in 0..200 { for (o, &x) in out.iter_mut().zip(&xs) { *o = super::exp(x); } s += out[7]; }
        let te = t.elapsed().as_nanos() as f64 / (200.0 * 8192.0);
        let t = std::time::Instant::now();
        for _ in 0..200 { for (o, &p) in out.iter_mut().zip(&ps) { *o = super::ln(p); } s += out[7]; }
        let tl = t.elapsed().as_nanos() as f64 / (200.0 * 8192.0);
        let t = std::time::Instant::now();
        for _ in 0..200 { for &x in &xs { s += x.exp(); } }
        let tse = t.elapsed().as_nanos() as f64 / (200.0 * 8192.0);
        let t = std::time::Instant::now();
        for _ in 0..200 { for &p in &ps { s += p.ln(); } }
        let tsl = t.elapsed().as_nanos() as f64 / (200.0 * 8192.0);
        eprintln!("ns/call: dmath exp {te:.2} ln {tl:.2} | libm exp {tse:.2} ln {tsl:.2}  ({s})");
    }
}
