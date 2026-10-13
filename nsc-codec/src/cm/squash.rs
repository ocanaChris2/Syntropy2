//! Logistic domain conversions, integer-only.
//!
//! `stretch(p) = ln(p / (1 - p))` and `squash(x) = 1 / (1 + e^-x)`, with probabilities as 16-bit
//! integers (P·65536) and the logistic domain in units of 1/256 nat clamped to ±2047 (|p| up to
//! 0.99966). Squash interpolates a 33-point table of `65536 / (1 + e^(-x/256))` sampled every 128
//! units; stretch is its exact inverse over a 4096-entry table (indexed by `p >> 4`), built at
//! compile time.

/// 65536 / (1 + exp(-x/256)) at x = -2048 + 128·i, rounded.
const SQ: [i32; 33] = [
    22, 36, 60, 98, 162, 267, 439, 720, 1179, 1921, 3108, 4971, 7812, 11955, 17625, 24743, 32768,
    40793, 47911, 53581, 57724, 60565, 62428, 63615, 64357, 64816, 65097, 65269, 65374, 65438,
    65476, 65500, 65514,
];

/// Interpolated squash: the reference definition, used to build the lookup table.
const fn squash_interp(x: i32) -> u32 {
    let x = if x > 2047 { 2047 } else if x < -2047 { -2047 } else { x };
    let i = ((x + 2048) >> 7) as usize;
    let w = (x + 2048) & 127;
    ((SQ[i] * (128 - w) + SQ[i + 1] * w + 64) >> 7) as u32
}

const fn build_squash() -> [u16; 4096] {
    let mut t = [0u16; 4096];
    let mut i = 0;
    while i < 4096 {
        t[i] = squash_interp(i as i32 - 2048) as u16;
        i += 1;
    }
    t
}

static SQUASH: [u16; 4096] = build_squash();

/// Logistic → probability (P(1)·65536), `x` in 1/256 nat, clamped to ±2047. Result in 22..=65514.
/// A single table lookup (identical values to the interpolated definition).
#[inline]
pub fn squash(x: i32) -> u32 { SQUASH[(x.clamp(-2047, 2047) + 2048) as usize] as u32 }

const fn build_stretch() -> [i16; 4096] {
    let mut t = [0i16; 4096];
    let mut pi = 0usize;
    let mut x = -2047i32;
    while x <= 2047 {
        let v = (squash_interp(x) >> 4) as usize;
        let mut i = pi;
        while i <= v {
            t[i] = x as i16;
            i += 1;
        }
        pi = v + 1;
        x += 1;
    }
    let mut i = pi;
    while i < 4096 {
        t[i] = 2047;
        i += 1;
    }
    t
}

static STRETCH: [i16; 4096] = build_stretch();

/// Probability (P(1)·65536, any u32 < 65536) → logistic in 1/256 nat (±2047).
#[inline]
pub fn stretch(p: u32) -> i32 { STRETCH[(p >> 4) as usize & 4095] as i32 }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squash_monotone_and_bounded() {
        let mut prev = 0;
        for x in -2047..=2047 {
            let p = squash(x);
            assert!(p >= prev && (1..=65535).contains(&p));
            prev = p;
        }
        assert_eq!(squash(0), 32768);
    }

    #[test]
    fn stretch_inverts_squash() {
        // stretch is indexed by p >> 4, so compare round trips in the probability domain
        for x in -2047..=2047 {
            let p = squash(x);
            let back = squash(stretch(p));
            assert!((back as i32 - p as i32).abs() <= 24, "x={x} p={p} back={back}");
        }
    }
}
