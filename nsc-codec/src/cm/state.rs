//! Bit-history state machine: a byte that summarises the sequence of 0/1 bits seen in a context.
//!
//! * States 0..=30 hold the exact last 0–4 bits (state 0 = nothing seen yet).
//! * The rest hold bounded counts `(n0, n1)`, plus the most recent bit while both counts are
//!   non-zero and the total is below 16.
//! * When a bit arrives, its count grows. The opposite count is discounted roughly
//!   logarithmically, favouring recent behaviour in non-stationary data.
//! * Counts that leave the allowed region are rescaled toward it, preserving their ratio.
//!
//! States are numbered in order of increasing total count, so the state byte doubles as a
//! "how much has this context been used" priority for hash-table replacement.
//!
//! The table is generated at compile time by [`build`]; `tests::checksum` freezes it, because
//! archives depend on every transition.

const MAXC: usize = 41;
/// For the smaller count `lo` (index), the largest allowed larger count.
const BOUND: [usize; 6] = [40, 30, 12, 8, 6, 5];
/// round(16·log2(n)) for n = 0..=41 (entry 0 unused).
const LOG16: [usize; 42] = [
    0, 0, 16, 25, 32, 37, 41, 45, 48, 51, 53, 55, 57, 59, 61, 63, 64, 65, 67, 68, 69, 70, 71, 72,
    73, 74, 75, 76, 77, 78, 79, 79, 80, 81, 81, 82, 83, 83, 84, 85, 85, 86,
];

pub struct StateTable {
    /// next[state][bit]
    pub next: [[u8; 2]; 256],
    /// Counts represented by each state (for initialising adaptive probabilities).
    pub n0: [u8; 256],
    pub n1: [u8; 256],
    pub count: usize,
}

pub static TABLE: StateTable = build();

const fn allowed(n0: usize, n1: usize) -> bool {
    let (lo, hi) = if n0 < n1 { (n0, n1) } else { (n1, n0) };
    lo < BOUND.len() && hi <= BOUND[lo]
}

const fn discount(n: usize) -> usize { if n <= 2 { n } else { LOG16[n] / 6 - 1 } }

const fn reduce(mut n0: usize, mut n1: usize) -> (usize, usize) {
    while !allowed(n0, n1) {
        let swap = n1 > n0;
        let (mut hi, mut lo) = if swap { (n1, n0) } else { (n0, n1) };
        if lo < 2 {
            hi -= 1;
        } else {
            hi = (hi * (lo - 1) + lo / 2) / lo;
            lo -= 1;
        }
        if swap { n1 = hi; n0 = lo; } else { n0 = hi; n1 = lo; }
    }
    (n0, n1)
}

pub const fn build() -> StateTable {
    let mut hist_len = [0usize; 256];
    let mut hist_bits = [0usize; 256];
    let mut is_hist = [false; 256];
    let mut c0 = [0usize; 256];
    let mut c1 = [0usize; 256];
    // index of (n0, n1, last) with last in {0, 1, 2 = not tracked}
    let mut idx = [[[0u8; 3]; MAXC + 1]; MAXC + 1];
    let mut ns = 0usize;

    let mut len = 0usize;
    while len <= 4 {
        let mut b = 0usize;
        while b < (1 << len) {
            is_hist[ns] = true;
            hist_len[ns] = len;
            hist_bits[ns] = b;
            let mut ones = 0;
            let mut k = 0;
            while k < len { ones += (b >> k) & 1; k += 1; }
            c0[ns] = len - ones;
            c1[ns] = ones;
            ns += 1;
            b += 1;
        }
        len += 1;
    }
    let mut tot = 5usize;
    while tot <= 2 * MAXC {
        let mut n1 = 0usize;
        while n1 <= tot {
            let n0 = tot - n1;
            if n0 <= MAXC && n1 <= MAXC && allowed(n0, n1) {
                if n0 > 0 && n1 > 0 && tot < 16 {
                    c0[ns] = n0; c1[ns] = n1; idx[n0][n1][0] = ns as u8; ns += 1;
                    c0[ns] = n0; c1[ns] = n1; idx[n0][n1][1] = ns as u8; ns += 1;
                } else {
                    c0[ns] = n0; c1[ns] = n1; idx[n0][n1][2] = ns as u8; ns += 1;
                }
            }
            n1 += 1;
        }
        tot += 1;
    }

    let mut next = [[0u8; 2]; 256];
    let mut s = 0usize;
    while s < ns {
        let mut y = 0usize;
        while y < 2 {
            let t = if is_hist[s] {
                if hist_len[s] < 4 {
                    // history states are laid out by length: offset of length L is 2^L - 1
                    let l = hist_len[s] + 1;
                    (1 << l) - 1 + ((hist_bits[s] << 1) | y)
                } else {
                    let ones = c1[s] + y;
                    let (a, b) = reduce(5 - ones, ones);
                    lookup(&idx, a, b, y)
                }
            } else {
                let (mut a, mut b) = (c0[s], c1[s]);
                if y == 1 { b += 1; a = discount(a); } else { a += 1; b = discount(b); }
                let (a, b) = reduce(a, b);
                lookup(&idx, a, b, y)
            };
            next[s][y] = t as u8;
            y += 1;
        }
        s += 1;
    }
    let mut n0 = [0u8; 256];
    let mut n1 = [0u8; 256];
    let mut i = 0;
    while i < ns { n0[i] = c0[i] as u8; n1[i] = c1[i] as u8; i += 1; }
    StateTable { next, n0, n1, count: ns }
}

const fn lookup(idx: &[[[u8; 3]; MAXC + 1]; MAXC + 1], n0: usize, n1: usize, last: usize) -> usize {
    if n0 > 0 && n1 > 0 && n0 + n1 < 16 { idx[n0][n1][last] as usize } else { idx[n0][n1][2] as usize }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_and_closed() {
        assert!(TABLE.count <= 256, "{} states", TABLE.count);
        assert_eq!(TABLE.count, 253);
        for s in 0..TABLE.count {
            for y in 0..2 {
                assert!((TABLE.next[s][y] as usize) < TABLE.count);
            }
        }
        // a long run of ones must saturate in a confident all-ones state
        let mut s = 0usize;
        for _ in 0..200 { s = TABLE.next[s][1] as usize; }
        assert_eq!(TABLE.n0[s], 0);
        assert!(TABLE.n1[s] >= 30);
    }

    #[test]
    fn checksum() {
        // Archive format depends on this table: any change must bump the engine revision.
        let mut h: u64 = 0xcbf29ce484222325;
        for s in 0..256 {
            for b in [TABLE.next[s][0], TABLE.next[s][1], TABLE.n0[s], TABLE.n1[s]] {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        assert_eq!(h, STATE_TABLE_FNV, "state table changed: {h:#x}");
    }

    const STATE_TABLE_FNV: u64 = 0x7925_19ad_1192_a4c2;
}
