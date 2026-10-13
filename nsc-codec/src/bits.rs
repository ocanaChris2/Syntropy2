//! Bit-level I/O and prefix codes shared by the format recoders (MSB-first, as MPEG streams are).

/// Reads bits from a byte slice, most significant bit first. Reading past the end yields zeros
/// (callers compare [`BitReader::pos`] with their own limit).
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    /// A reader over `data` starting at bit `pos`.
    pub fn new(data: &'a [u8], pos: usize) -> Self { BitReader { data, pos } }

    #[inline]
    pub fn pos(&self) -> usize { self.pos }

    #[inline]
    pub fn seek(&mut self, pos: usize) { self.pos = pos; }

    /// A second reader over the same data, at bit `pos`.
    pub fn at(&self, pos: usize) -> BitReader<'a> { BitReader { data: self.data, pos } }

    /// The next `n` (0..=32) bits, without consuming them.
    #[inline]
    pub fn peek(&self, n: u32) -> u32 {
        debug_assert!(n <= 32);
        if n == 0 { return 0; }
        let byte = self.pos >> 3;
        let mut v = 0u64;
        for i in 0..5 {
            v = (v << 8) | self.data.get(byte + i).copied().unwrap_or(0) as u64;
        }
        ((v << (24 + (self.pos & 7))) >> (64 - n)) as u32
    }

    #[inline]
    pub fn skip(&mut self, n: u32) { self.pos += n as usize; }

    /// Consume and return the next `n` (0..=32) bits.
    #[inline]
    pub fn get(&mut self, n: u32) -> u32 {
        let v = self.peek(n);
        self.pos += n as usize;
        v
    }
}

/// Appends bits to a byte vector, most significant bit first.
#[derive(Default)]
pub struct BitWriter {
    bytes: Vec<u8>,
    /// Bits written.
    len: usize,
}

impl BitWriter {
    pub fn new() -> Self { Self::default() }

    #[inline]
    pub fn len(&self) -> usize { self.len }

    /// Append the low `n` (0..=32) bits of `v`.
    #[inline]
    pub fn put(&mut self, v: u32, n: u32) {
        debug_assert!(n <= 32);
        let mut left = n;
        while left > 0 {
            let used = self.len & 7;
            if used == 0 { self.bytes.push(0); }
            let take = (8 - used as u32).min(left);
            let chunk = ((v as u64 >> (left - take)) & ((1u64 << take) - 1)) as u8;
            *self.bytes.last_mut().unwrap() |= chunk << (8 - used as u32 - take);
            self.len += take as usize;
            left -= take;
        }
    }

    /// Pad with zero bits to a byte boundary.
    #[inline]
    pub fn align(&mut self) { self.len = self.bytes.len() * 8; }

    /// Append whole bytes (the writer must be byte-aligned).
    pub fn extend(&mut self, b: &[u8]) {
        debug_assert!(self.len % 8 == 0);
        self.bytes.extend_from_slice(b);
        self.len += b.len() * 8;
    }

    /// The bytes written so far; the last one is zero-padded.
    pub fn bytes(&self) -> &[u8] { &self.bytes }

    /// The bytes written; the last one is zero-padded.
    pub fn into_bytes(self) -> Vec<u8> { self.bytes }
}

const ROOT: u32 = 9;
/// Root entry marking a longer code: its low bits are the trie node reached after `ROOT` bits.
const LONG: u32 = 0xFF;

/// A prefix-code decoder built from (code, length, symbol) triples: a `ROOT`-bit lookup table,
/// then a bit-by-bit walk for the longer codes.
pub struct PrefixDec {
    root: Vec<u32>,       // length << 24 | symbol; LONG << 24 | node; 0 = no code
    nodes: Vec<[u32; 2]>, // child: 0 = none, 1 + symbol << 1 for a leaf, node << 1 otherwise
}

impl PrefixDec {
    /// `codes`: (code, length in 1..=32, symbol < 2^24). The codes must be prefix-free.
    pub fn new(codes: impl IntoIterator<Item = (u32, u8, u32)>) -> Self {
        let mut nodes: Vec<[u32; 2]> = vec![[0, 0]];
        for (code, len, sym) in codes {
            debug_assert!((1..=32).contains(&len) && sym < 1 << 24);
            let mut n = 0usize;
            for i in (0..len).rev() {
                let bit = ((code >> i) & 1) as usize;
                if i == 0 {
                    nodes[n][bit] = 1 | sym << 1;
                } else {
                    if nodes[n][bit] == 0 {
                        nodes.push([0, 0]);
                        nodes[n][bit] = ((nodes.len() - 1) as u32) << 1;
                    }
                    n = (nodes[n][bit] >> 1) as usize;
                }
            }
        }
        let mut root = vec![0u32; 1 << ROOT];
        for (i, e) in root.iter_mut().enumerate() {
            let mut n = 0usize;
            for d in 0..ROOT {
                let c = nodes[n][(i >> (ROOT - 1 - d)) & 1];
                if c == 0 { break; }
                if c & 1 == 1 { *e = (d + 1) << 24 | c >> 1; break; }
                n = (c >> 1) as usize;
                if d == ROOT - 1 { *e = LONG << 24 | n as u32; }
            }
        }
        PrefixDec { root, nodes }
    }

    /// The symbol at the reader's position and its code length, without consuming it. None for
    /// bits that are no code word.
    #[inline]
    pub fn peek(&self, r: &BitReader) -> Option<(u32, u32)> {
        let e = self.root[r.peek(ROOT) as usize];
        match e >> 24 {
            0 => None,
            LONG => {
                let mut n = (e & 0xFF_FFFF) as usize;
                let tail = r.peek(32);
                for d in ROOT..32 {
                    let c = self.nodes[n][((tail >> (31 - d)) & 1) as usize];
                    if c == 0 { return None; }
                    if c & 1 == 1 { return Some((c >> 1, d + 1)); }
                    n = (c >> 1) as usize;
                }
                None
            }
            len => Some((e & 0xFF_FFFF, len)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_roundtrip() {
        let mut s = 9u64;
        let mut rng = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let fields: Vec<(u32, u32)> = (0..2000).map(|_| { let n = (rng() % 33) as u32; (if n == 32 { rng() as u32 } else { rng() as u32 & ((1u32 << n) - 1) }, n) }).collect();
        let mut w = BitWriter::new();
        for &(v, n) in &fields { w.put(v, n); }
        let total = w.len();
        let bytes = w.into_bytes();
        assert_eq!(bytes.len(), total.div_ceil(8));
        let mut r = BitReader::new(&bytes, 0);
        for &(v, n) in &fields { assert_eq!(r.get(n), v); }
        assert_eq!(r.pos(), total);
        assert_eq!(r.get(32), 0, "zeros past the end");
    }

    #[test]
    fn prefix_codes_decode() {
        // lengths 1, 2, 3, 12, 12 (incomplete on purpose: some bit strings are no code)
        let codes = [(0b0u32, 1u8, 7u32), (0b10, 2, 8), (0b110, 3, 9), (0b1110_0000_0001, 12, 10), (0b1110_0000_0000, 12, 11)];
        let d = PrefixDec::new(codes);
        let mut w = BitWriter::new();
        let order = [9u32, 7, 10, 8, 11, 7];
        for s in order { let (c, n, _) = codes.iter().find(|c| c.2 == s).unwrap(); w.put(*c, *n as u32); }
        w.put(0b1111, 4);
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes, 0);
        for s in order {
            let (sym, n) = d.peek(&r).unwrap();
            assert_eq!(sym, s);
            r.skip(n);
        }
        assert!(d.peek(&r).is_none(), "1111… is not a code word");
    }
}
