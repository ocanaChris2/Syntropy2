//! Long-range match model.
//!
//! A table maps a hash of the last `MINLEN` bytes to the position that followed them. When no match
//! is active, that position becomes a candidate, verified by comparing history backwards. While the
//! match holds, the byte after the matched context is predicted bit by bit. Confidence grows with
//! the match length and is learned by a StateMap over (length bucket, expected bit). The expected
//! byte is also exported, so other models and the SSE stage can use it as context.
use super::mixer::Mixer;
use super::squash::stretch;
use super::statemap::StateMap;

const MINLEN: usize = 6;
const MAXLEN: usize = 65535;
const MAX_VERIFY: usize = 512;

/// 0 = no match; 1..=15 exact lengths; then one bucket per doubling up to 23.
#[inline]
pub fn len_bucket(len: usize) -> usize {
    if len < 16 { len } else { (12 + (usize::BITS - len.leading_zeros()) as usize).min(23) }
}

pub struct MatchModel {
    ht: Vec<u32>,
    mask: usize,
    ptr: usize,
    len: usize,
    expected: u32,
    miss: bool,
    active: bool,
    ebit: u32,
    sm: StateMap,
    smcx: usize,
}

impl MatchModel {
    pub fn new(bits: u32) -> Self {
        MatchModel {
            ht: vec![0; 1 << bits],
            mask: (1 << bits) - 1,
            ptr: 0,
            len: 0,
            expected: 0,
            miss: false,
            active: false,
            ebit: 0,
            sm: StateMap::new(24 * 2 * 8, 1023),
            smcx: 0,
        }
    }

    /// Called at each byte boundary with the full history before the byte about to be coded.
    pub fn byte_update(&mut self, hist: &[u8]) {
        let pos = hist.len();
        if self.len > 0 {
            if !self.miss {
                self.len = (self.len + 1).min(MAXLEN);
                self.ptr += 1;
            } else {
                self.len = 0;
            }
        }
        self.miss = false;
        if pos >= MINLEN {
            let mut k = 0u64;
            for &b in &hist[pos - MINLEN..pos] { k = (k << 8) | b as u64; }
            let h = (super::hash::hash(0x4d41, k) as usize) & self.mask;
            if self.len == 0 {
                let cand = self.ht[h] as usize;
                if cand > 0 && cand < pos {
                    let mut l = 0;
                    while l < MAX_VERIFY && l < cand && hist[cand - 1 - l] == hist[pos - 1 - l] { l += 1; }
                    if l >= MINLEN {
                        self.len = l;
                        self.ptr = cand;
                    }
                }
            }
            self.ht[h] = pos as u32;
        }
        self.expected = if self.len > 0 { hist[self.ptr] as u32 } else { 0 };
    }

    /// Expected next byte and its length bucket, while the prediction is still consistent.
    #[inline]
    pub fn expected(&self) -> Option<(u32, usize)> {
        if self.len > 0 && !self.miss { Some((self.expected, len_bucket(self.len))) } else { None }
    }

    /// Mixer-selector context: 0 when idle, else bucket·2 + expected bit (1..=47).
    #[inline]
    pub fn mixer_ctx(&self) -> usize { if self.active { len_bucket(self.len) * 2 + self.ebit as usize } else { 0 } }

    /// Two inputs for the bit at `bpos` (0 = MSB) given the partial byte `c0` (leading 1).
    #[inline]
    pub fn mix(&mut self, bpos: u32, c0: u32, m: &mut Mixer) {
        if self.len > 0 && !self.miss && ((self.expected | 0x100) >> (8 - bpos)) != c0 {
            self.miss = true;
        }
        if self.len > 0 && !self.miss {
            self.ebit = (self.expected >> (7 - bpos)) & 1;
            let lb = len_bucket(self.len);
            self.smcx = (lb * 2 + self.ebit as usize) * 8 + bpos as usize;
            let p = self.sm.p_at(self.smcx);
            m.add(stretch(p));
            let s = (lb as i32) * 80;
            m.add(if self.ebit == 1 { s } else { -s });
            self.active = true;
        } else {
            m.add(0);
            m.add(0);
            self.active = false;
        }
    }

    #[inline]
    pub fn update(&mut self, bit: u32) {
        if self.active {
            self.sm.update_at(self.smcx, bit);
            if bit != self.ebit { self.miss = true; }
        }
    }
}
