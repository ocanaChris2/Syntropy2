//! A set of hashed contexts, each mapped to bit-history states in the shared [`HashTable`].
//!
//! At every byte boundary the predictor gives each context a 64-bit hash. Each context then
//! claims one bucket for the high nibble (and, halfway through the byte, another for the low
//! nibble, keyed by the high nibble). For each bit, the state in the slot for the bits coded so
//! far is turned into mixer inputs through a per-context [`StateMap`]; after the bit, the state
//! advances along the state table.
use super::hash::{rehash, HashTable};
use super::mixer::Mixer;
use super::squash::stretch;
use super::state::TABLE;
use super::statemap::RECIP;

pub struct ContextMap {
    n: usize,
    /// Contexts switched off for the current segment kind: no lookups, zero mixer inputs.
    active: Vec<bool>,
    hashes: Vec<u64>,
    bucket: Vec<usize>,
    slot: Vec<usize>,
    /// Per-context StateMap over the 256 bit-history states: `p22 << 10 | count`. Indexing a
    /// fixed `[u32; 256]` with the state byte needs no bounds check in the per-bit loops.
    sm: Vec<[u32; 256]>,
}

const LIMIT: u32 = 1023;

impl ContextMap {
    pub fn new(n: usize) -> Self {
        let mut proto = [0u32; 256];
        for (s, e) in proto.iter_mut().enumerate() {
            let (n0, n1) = (TABLE.n0[s] as u64, TABLE.n1[s] as u64);
            let p22 = (((2 * n1 + 1) << 22) / (2 * (n0 + n1) + 2)).min((1 << 22) - 1);
            *e = (p22 as u32) << 10;
        }
        ContextMap { n, active: vec![true; n], hashes: vec![0; n], bucket: vec![0; n], slot: vec![0; n], sm: vec![proto; n] }
    }

    pub fn set_active(&mut self, i: usize, on: bool) { self.active[i] = on; }

    #[inline]
    pub fn is_active(&self, i: usize) -> bool { self.active[i] }

    /// Set context `i`'s hash for the coming byte.
    #[inline]
    pub fn set(&mut self, i: usize, h: u64) { self.hashes[i] = h; }

    /// Claim the high-nibble buckets (call once per byte after all `set`s).
    pub fn byte_start(&mut self, ht: &mut HashTable) {
        for (&h, &a) in self.hashes.iter().zip(&self.active) { if a { ht.prefetch(h); } }
        for ((b, &h), &a) in self.bucket.iter_mut().zip(&self.hashes).zip(&self.active) { if a { *b = ht.find(h); } }
    }

    /// Claim the low-nibble buckets once the high nibble (`c0` = 1hhhh) is known.
    pub fn nibble(&mut self, ht: &mut HashTable, c0: u32) {
        for (h, &a) in self.hashes.iter_mut().zip(&self.active) {
            if a {
                *h = rehash(*h, c0);
                ht.prefetch(*h);
            }
        }
        for ((b, &h), &a) in self.bucket.iter_mut().zip(&self.hashes).zip(&self.active) { if a { *b = ht.find(h); } }
    }

    /// Add two inputs per context for the bit at in-nibble slot `j` (1..=15): the logit of the
    /// state's learned probability, and a linear confidence term that is zero for unseen contexts.
    /// Returns how many contexts have been seen before (state != 0).
    #[inline]
    pub fn mix(&mut self, ht: &HashTable, j: usize, m: &mut Mixer) -> usize {
        let mut hits = 0;
        let inputs = m.take(2 * self.n);
        for ((((b, sl), sm), pair), &a) in self.bucket.iter().zip(self.slot.iter_mut()).zip(&self.sm).zip(inputs.chunks_exact_mut(2)).zip(&self.active) {
            if !a {
                pair[0] = 0;
                pair[1] = 0;
                continue;
            }
            let at = *b + j;
            *sl = at;
            let s = ht.get(at);
            let p = ((sm[s as usize] >> 16).max(1)) as i32;
            let seen = (s != 0) as i32;
            pair[0] = stretch(p as u32) as i16;
            pair[1] = (((p - 32768) >> 4) * seen) as i16;
            hits += seen as usize;
        }
        hits
    }

    #[inline]
    pub fn update(&mut self, ht: &mut HashTable, bit: u32) {
        let b = (bit & 1) as usize;
        let target: i64 = if b != 0 { (1 << 22) - 1 } else { 0 };
        for ((sl, sm), &a) in self.slot.iter().zip(self.sm.iter_mut()).zip(&self.active) {
            if !a { continue; }
            let s = ht.get(*sl);
            ht.set(*sl, TABLE.next[s as usize][b]);
            let e = &mut sm[s as usize];
            let n = *e & 1023;
            let p22 = (*e >> 10) as i64;
            let p22 = p22 + (((target - p22) * RECIP[n as usize] as i64) >> 16);
            *e = ((p22 as u32) << 10) | (n + (n < LIMIT) as u32);
        }
    }
}
