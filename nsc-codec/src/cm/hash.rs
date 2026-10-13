//! Context hashing and the shared bit-history hash table.
//!
//! The table holds 16-byte buckets: a checksum byte plus 15 bit-history states covering one
//! nibble (slot 1 for the first bit, 2–3 for the second, 4–7 for the third, 8–15 for the fourth).
//! Each context does two lookups per byte, one per nibble. A lookup probes three buckets in the
//! same 64-byte cache line; on a miss it evicts the least-used one (lowest first-slot state,
//! which grows with use).
//!
//! Hashes are 64-bit: the bucket index comes from the low bits and the checksum from the top
//! byte, so the two are independent at any table size. The memory comes from a zeroed `Vec<u8>`
//! (calloc → lazily zero-filled pages), aligned by offset to 64 bytes, so a large table costs
//! nothing until touched.

#[inline]
fn mix64(mut h: u64) -> u64 {
    h = (h ^ (h >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^ (h >> 31)
}

/// Hash a context value together with its context number.
#[inline]
pub fn hash(i: u32, x: u64) -> u64 {
    mix64(x.wrapping_add((i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)))
}

/// Combine a context hash with extra bits (e.g. the high nibble of the current byte).
#[inline]
pub fn rehash(h: u64, x: u32) -> u64 { mix64(h ^ (x as u64 + 1).wrapping_mul(0xD6E8_FEB8_6659_FD93)) }

pub const BUCKET: usize = 16;

pub struct HashTable {
    mem: Vec<u8>,
    off: usize,
    mask: usize, // bucket-index mask
}

impl HashTable {
    /// `1 << bits` buckets (16 bytes each).
    pub fn new(bits: u32) -> Self {
        let n = 1usize << bits;
        let mem = vec![0u8; n * BUCKET + 64];
        let off = mem.as_ptr().align_offset(64);
        HashTable { mem, off, mask: n - 1 }
    }

    pub fn bytes(&self) -> usize { (self.mask + 1) * BUCKET }

    /// Find or claim the bucket for hash `h`; returns the byte offset of its slot 0 (the checksum
    /// byte), so slot `k` (1..=15) lives at `offset + k`.
    #[inline]
    pub fn find(&mut self, h: u64) -> usize {
        let chk = ((h >> 56) as u8) | 1; // never 0, so an empty bucket never matches
        let i = (h as usize) & self.mask;
        let base = self.off;
        let b0 = base + i * BUCKET;
        let b1 = base + (i ^ 1) * BUCKET;
        let b2 = base + (i ^ 2) * BUCKET;
        if self.mem[b0] == chk { return b0; }
        if self.mem[b1] == chk { return b1; }
        if self.mem[b2] == chk { return b2; }
        // evict the least-used of the three (priority = state in slot 1)
        let mut v = b0;
        if self.mem[b1 + 1] < self.mem[v + 1] { v = b1; }
        if self.mem[b2 + 1] < self.mem[v + 1] { v = b2; }
        // two 8-byte stores (a `fill` here compiles to a libc memset call)
        let b: &mut [u8; BUCKET] = (&mut self.mem[v..v + BUCKET]).try_into().unwrap();
        *b = [0; BUCKET];
        b[0] = chk;
        v
    }

    #[inline]
    pub fn get(&self, at: usize) -> u8 { self.mem[at] }

    #[inline]
    pub fn set(&mut self, at: usize, v: u8) { self.mem[at] = v; }

    /// Prefetch the cache line holding the bucket group for hash `h`.
    #[inline]
    pub fn prefetch(&self, h: u64) {
        #[cfg(target_arch = "x86_64")]
        {
            let i = (h as usize) & self.mask;
            let at = self.off + (i & !3) * BUCKET;
            if at < self.mem.len() {
                // SAFETY: prefetch is only a hint, and the address is inside `mem`.
                unsafe {
                    std::arch::x86_64::_mm_prefetch(self.mem.as_ptr().add(at) as *const i8, std::arch::x86_64::_MM_HINT_T0);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_is_stable_and_aligned() {
        let mut t = HashTable::new(10);
        assert_eq!((t.mem.as_ptr() as usize + t.off) % 64, 0);
        let a = t.find(hash(1, 42));
        t.set(a + 3, 7);
        let b = t.find(hash(1, 42));
        assert_eq!(a, b);
        assert_eq!(t.get(b + 3), 7);
    }

    #[test]
    fn hash_spreads() {
        // consecutive inputs must not collide in the low bits
        let mut seen = std::collections::BTreeSet::new();
        for x in 0..4096u64 { seen.insert(hash(3, x) & 0xffff); }
        assert!(seen.len() > 3900);
    }
}
