//! Outright guesses of the next byte in binary data.
//!
//! Tables of numbers are full of bytes that are *computable* from the bytes at a fixed distance
//! before them — the same as in the record above, that plus the step between the two records
//! above, that plus a carry from the byte before — yet look new to a model that only remembers
//! which byte followed which context: a counter never repeats. Each guesser here names one
//! byte; while the bits coded so far agree with it, it gives the mixer its next bit with a
//! confidence learned from how its last guesses in the same column went.
use super::mixer::Mixer;
use super::squash::stretch;
use super::statemap::StateMap;

/// How many guessers (inputs to the mixer): the byte above in the record before; that plus the
/// step from the record before it; that plus a carry from the byte before. (Guessers on fixed
/// strides of 4 and 8 bytes were tried too: together a tenth of these three's gain, for more
/// than their cost.)
pub const GUESSERS: usize = 3;

pub struct Guess {
    /// The guesses for the byte being coded, and which are still possible.
    e: [u8; GUESSERS],
    live: [bool; GUESSERS],
    /// Per guesser and column: how its last guesses there went (a bit each, the latest lowest).
    hits: Vec<u8>,
    col: usize,
    sm: StateMap,
    cx: [usize; GUESSERS],
}

const COLS: usize = 256;

impl Guess {
    pub fn new() -> Self {
        Guess { e: [0; GUESSERS], live: [false; GUESSERS], hits: vec![0; GUESSERS * COLS], col: 0, sm: StateMap::new(GUESSERS * 256 * 8 * 2, 255), cx: [0; GUESSERS] }
    }

    /// At a byte boundary: `hist` is everything before the byte to code, `r` the record length
    /// in force (0: none), `on` whether this is data the guessers are for.
    pub fn byte_update(&mut self, hist: &[u8], r: usize, on: bool) {
        let pos = hist.len();
        // how the guesses for the byte just coded went
        if pos > 0 {
            let c = hist[pos - 1];
            for k in 0..GUESSERS {
                let h = &mut self.hits[k * COLS + self.col];
                *h = *h << 1 | (self.e[k] == c) as u8;
            }
        }
        let at = |d: usize| -> i32 { if d > 0 && d <= pos { hist[pos - d] as i32 } else { 0 } };
        let have = |d: usize| on && d > 0 && d <= pos;
        // a field that counts: the byte before it (the lower one) passed 255 or 0 since the
        // record above, going the way it went before
        let carry = |d: usize| -> i32 {
            let (low, above, step) = (at(1), at(d + 1), at(d + 1) - at(2 * d + 1));
            if step >= 0 && low < above { 1 } else if step < 0 && low > above { -1 } else { 0 }
        };
        self.col = if r > 0 { pos % r } else { pos } % COLS;
        let guesses = [(have(2 * r), at(r)), (have(2 * r), 2 * at(r) - at(2 * r)), (have(2 * r + 1), at(r) + carry(r))];
        for (k, (ok, v)) in guesses.into_iter().enumerate() {
            self.live[k] = ok;
            self.e[k] = v as u8;
        }
    }

    /// One input per guesser for the bit at `bpos` (0: the first) given the partial byte `c0`
    /// (with its leading 1).
    #[inline]
    pub fn mix(&mut self, bpos: u32, c0: u32, m: &mut Mixer) {
        for k in 0..GUESSERS {
            if self.live[k] && ((self.e[k] as u32 | 0x100) >> (8 - bpos)) != c0 { self.live[k] = false; }
            if !self.live[k] {
                m.add(0);
                continue;
            }
            let bit = (self.e[k] as usize >> (7 - bpos)) & 1;
            self.cx[k] = ((k * 256 + self.hits[k * COLS + self.col] as usize) * 8 + bpos as usize) * 2 + bit;
            m.add(stretch(self.sm.p_at(self.cx[k])));
        }
    }

    #[inline]
    pub fn update(&mut self, bit: u32) {
        for k in 0..GUESSERS {
            if self.live[k] { self.sm.update_at(self.cx[k], bit); }
        }
    }
}
