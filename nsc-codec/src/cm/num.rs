//! Numbers in text.
//!
//! A number in a log line, a table row or a list is rarely new: it is the number that stood in
//! the same place a line before, or that plus what was added the last time — a time, a counter,
//! a size, an identifier. The contexts of the bytes before it cannot say so: the digits that
//! change are the ones those contexts would have to end in, and every other digit run in between
//! breaks them.
//!
//! This model keeps the line as a template (its bytes with every number as one mark), and for
//! each place a number can stand in — what the template says before it — the number last seen
//! there. While a number is coded, its contexts are that reference number's digit at the same
//! index, whether the digits so far were the reference's, and the digits of the reference plus
//! the last difference; outside numbers, the template itself, which is the line's text whatever
//! the numbers in it were.
//!
//! The contexts are what a greedy selection over twenty candidates kept (`bench/num_dev.py`, on
//! logs, tables, JSON and XML). Text with few numbers gains under one percent from them, and
//! pays their time: a writer switches the model on for text that is numbers to a fair part
//! ([`wanted`]).
use std::sync::atomic::{AtomicU64, Ordering};

/// The model's contexts ([`Num::context`]).
pub const CONTEXTS: u32 = 5;
/// The ones in use (a bit each), and two switches: bit 62, numbers select a weight set of the
/// mixer; bit 63, they key the second probability refinement. Developer experiments set this
/// (`nsc cm-test --num HEX`).
pub static KEEP: AtomicU64 = AtomicU64::new(DEFAULT);
pub const DEFAULT: u64 = (1 << CONTEXTS) - 1 | 3 << 62;

pub fn kept(i: u32) -> bool { KEEP.load(Ordering::Relaxed) >> i & 1 == 1 }

/// Is the text `b` worth the model: are 6% of its bytes or more digits?
pub fn wanted(digits: usize, bytes: usize) -> bool { bytes > 0 && digits * 100 >= bytes * 6 }

/// The digits in `b`.
pub fn digits(b: &[u8]) -> usize { b.iter().filter(|c| c.is_ascii_digit()).count() }

/// What a place is known by: the whole line before it; the four symbols before it (which finds
/// a number said twice in a line); the whole line before it with every word as one mark too
/// (rows of a table, whatever their words).
const KEYS: usize = 3;
const SLOTS: usize = 1 << 12;
/// The symbol a number is in a template.
const MARK: u64 = 1;
/// Digits a number's value is kept for.
const DIGITS: u32 = 18;

/// The number last seen in a place.
#[derive(Clone, Copy, Default)]
struct Slot {
    check: u32,
    /// Where it is, its digits, and its value if that fits.
    pos: u32,
    len: u8,
    /// How the numbers before went, two bits each, the latest lowest: it was the number before
    /// it; it was that plus the difference before.
    went: u8,
    value: u64,
    /// What it added to the number before it.
    delta: i64,
}

/// The reference for the number being coded, by one key.
#[derive(Clone, Copy, Default)]
struct Ref {
    slot: u32,
    check: u32,
    /// The reference (`len` 0: none) and the guess (the reference plus the last difference;
    /// `plen` 0: none, or no difference).
    pos: u32,
    len: u8,
    went: u8,
    value: u64,
    pred: [u8; 24],
    plen: u8,
    /// So far the digits were the reference's; the guess's; the first other digit was greater.
    eq: bool,
    on: bool,
    gt: bool,
}

pub struct Num {
    table: Vec<Slot>,
    refs: [Ref; KEYS],
    /// The line so far as a template (hashed), and its last eight symbols.
    line: u64,
    near: u64,
    /// The line as a template of numbers and words, and whether its last symbol is a word.
    shape: u64,
    word: bool,
    /// Of the number being coded: its digits so far (0: not in one), value and start.
    digits: u32,
    value: u64,
    start: u32,
}

#[inline]
fn mix(h: u64, v: u64) -> u64 { (h ^ v).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29) }

impl Num {
    pub fn new() -> Self {
        Num { table: vec![Slot::default(); KEYS * SLOTS], refs: [Ref::default(); KEYS], line: 0, near: 0, shape: 0, word: false, digits: 0, value: 0, start: 0 }
    }

    /// At a byte boundary: `hist` is everything before the byte to code.
    pub fn byte_update(&mut self, hist: &[u8]) {
        let pos = hist.len();
        let Some(&c) = hist.last() else { return };
        if c.is_ascii_digit() {
            let (d, i) = (c - b'0', self.digits as usize);
            if i == 0 {
                self.start = (pos - 1) as u32;
                self.value = 0;
            }
            for r in &mut self.refs {
                if r.eq {
                    let rd = if i < r.len as usize { hist[r.pos as usize + i] - b'0' } else { 255 };
                    if rd != d {
                        r.eq = false;
                        r.gt = rd != 255 && d > rd;
                    }
                }
                if r.on && (i >= r.plen as usize || r.pred[i] != d) { r.on = false; }
            }
            self.value = if self.digits < DIGITS { self.value * 10 + d as u64 } else { u64::MAX };
            self.digits += 1;
            return;
        }
        if self.digits > 0 {
            self.finish();
            self.line = mix(self.line, MARK);
            self.near = self.near << 8 | MARK;
            self.shape = mix(self.shape, MARK);
            self.word = false;
            self.digits = 0;
        }
        if c == b'\n' {
            (self.line, self.near, self.shape, self.word) = (0, 0, 0, false);
        } else {
            self.line = mix(self.line, c as u64 + 2);
            self.near = self.near << 8 | c as u64;
            let letter = c.is_ascii_alphabetic() || c >= 128;
            if !letter { self.shape = mix(self.shape, c as u64 + 2); } else if !self.word { self.shape = mix(self.shape, 2); }
            self.word = letter;
        }
        self.prepare();
    }

    /// The number that just ended becomes the reference of its places.
    fn finish(&mut self) {
        let len = self.digits.min(255) as u8;
        let known = self.digits <= DIGITS;
        for r in &self.refs {
            let had = r.len > 0;
            let same = had && r.eq && r.len == len;
            let step = r.plen > 0 && r.on && r.plen == len;
            let delta = if had && known && r.len as u32 <= DIGITS { self.value as i64 - r.value as i64 } else { 0 };
            self.table[r.slot as usize] = Slot {
                check: r.check,
                pos: self.start,
                len,
                went: if had { r.went << 2 | same as u8 | (step as u8) << 1 } else { 0 },
                value: if known { self.value } else { 0 },
                delta,
            };
        }
    }

    /// Look up the places a number starting here would be in.
    fn prepare(&mut self) {
        let keys = [mix(self.line, 0x51), mix(self.near & 0xFFFF_FFFF, 0x52), mix(self.shape, 0x53)];
        for (k, h) in keys.into_iter().enumerate() {
            let slot = k * SLOTS + (h >> 40) as usize % SLOTS;
            let check = h as u32 | 1;
            let s = self.table[slot];
            let mut r = Ref { slot: slot as u32, check, ..Ref::default() };
            if s.check == check {
                (r.pos, r.len, r.went, r.value, r.eq) = (s.pos, s.len, s.went, s.value, true);
                // the reference plus what it added, written as it was written (zeros in front)
                let next = s.value as i64 + s.delta;
                if s.delta != 0 && s.len as u32 <= DIGITS && next >= 0 {
                    let mut digits = [0u8; 24];
                    let (mut n, mut v) = (0usize, next as u64);
                    while v > 0 || n == 0 {
                        digits[n] = (v % 10) as u8;
                        v /= 10;
                        n += 1;
                    }
                    while n < s.len as usize { n += 1; }
                    for i in 0..n { r.pred[i] = digits[n - 1 - i]; }
                    (r.plen, r.on) = (n as u8, true);
                }
            }
            self.refs[k] = r;
        }
    }

    /// The reference's digit for the next byte (10: it ends here; 11: it ended before; 12: no
    /// reference) with how the number compares so far and how many digits the reference has left.
    #[inline]
    fn about(&self, r: &Ref, hist: &[u8]) -> u64 {
        let i = self.digits as usize;
        let len = r.len as usize;
        let d = if len == 0 { 12 } else if i < len { (hist[r.pos as usize + i] - b'0') as u64 } else if i == len { 10 } else { 11 };
        d | (r.eq as u64) << 4 | (r.gt as u64) << 5 | (len.saturating_sub(i).min(7) as u64) << 6
    }

    /// The same for the guess (the reference plus the last difference).
    #[inline]
    fn guess(&self, r: &Ref) -> u64 {
        let i = self.digits as usize;
        let len = r.plen as usize;
        let d = if len == 0 { 12 } else if i < len { r.pred[i] as u64 } else if i == len { 10 } else { 11 };
        d | (r.on as u64) << 4 | (len.saturating_sub(i).min(7) as u64) << 5 | (r.went as u64 & 15) << 8
    }

    /// Context `n` for the next byte.
    pub fn context(&self, n: u32, hist: &[u8]) -> u64 {
        let [a, b, c] = &self.refs;
        match n {
            // the reference of each place
            0 => self.about(a, hist) | (a.check as u64) << 16,
            1 => self.guess(a),
            2 => self.about(b, hist) | (b.check as u64) << 16,
            3 => self.about(c, hist) | (c.check as u64) << 16,
            // the template: its last eight symbols (inside a number, with the number's mark and
            // how many digits it has so far)
            _ => if self.digits > 0 { (self.near << 8 | MARK) ^ (self.digits.min(31) as u64) << 58 } else { self.near },
        }
    }

    /// The state of the number for the mixer and the refinement (0: no number, and none to expect).
    #[inline]
    pub fn state(&self) -> usize {
        let [a, b, _] = &self.refs;
        if self.digits == 0 && a.len == 0 && b.len == 0 { return 0; }
        1 | (self.digits.min(7) as usize) << 1 | (a.eq as usize) << 4 | (b.eq as usize) << 5 | (b.on as usize) << 6 | ((b.len > 0) as usize) << 7
    }
}
