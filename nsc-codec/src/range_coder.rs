//! Witten–Neal–Cleary range coder — a bit-exact port of the Python reference
//! (test_compression_integrity.RangeEncoder / RangeDecoder). Encoder and decoder must
//! produce/consume identical bits given identical cumulative-frequency tables, so this
//! reproduces the Python integer arithmetic exactly. Values fit in u64: rng ≤ 2^32,
//! total ≤ 2^30 ⇒ rng*ch ≤ 2^62.

const MASK: u64 = (1 << 32) - 1;
const HALF: u64 = 1 << 31;
const QUARTER: u64 = 1 << 30;
const THREEQ: u64 = 3 << 30;

pub struct RangeEncoder {
    low: u64,
    high: u64,
    pending: u64,
    bits: Vec<u8>, // one entry per bit (0/1); packed MSB-first at finish
}

impl RangeEncoder {
    pub fn new() -> Self {
        Self { low: 0, high: MASK, pending: 0, bits: Vec::new() }
    }

    #[inline]
    fn emit(&mut self, bit: u8) {
        self.bits.push(bit);
        for _ in 0..self.pending {
            self.bits.push(1 - bit);
        }
        self.pending = 0;
    }

    pub fn encode(&mut self, cl: u64, ch: u64, total: u64) {
        let rng = self.high - self.low + 1;
        self.high = self.low + (rng * ch) / total - 1;
        self.low = self.low + (rng * cl) / total;
        loop {
            if self.high < HALF {
                self.emit(0);
            } else if self.low >= HALF {
                self.emit(1);
                self.low -= HALF;
                self.high -= HALF;
            } else if self.low >= QUARTER && self.high < THREEQ {
                self.pending += 1;
                self.low -= QUARTER;
                self.high -= QUARTER;
            } else {
                break;
            }
            self.low = (self.low << 1) & MASK;
            self.high = ((self.high << 1) | 1) & MASK;
        }
    }

    /// Returns (packed payload bytes, number of bits). Matches numpy.packbits (MSB-first).
    pub fn finish(mut self) -> (Vec<u8>, usize) {
        self.pending += 1;
        let b = if self.low < QUARTER { 0 } else { 1 };
        self.emit(b);
        let n_bits = self.bits.len();
        let mut payload = vec![0u8; (n_bits + 7) / 8];
        for (i, &bit) in self.bits.iter().enumerate() {
            if bit != 0 {
                payload[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        (payload, n_bits)
    }
}

pub struct RangeDecoder {
    payload: Vec<u8>,
    n_bits: usize,
    i: usize,
    low: u64,
    high: u64,
    value: u64,
}

impl RangeDecoder {
    pub fn new(payload: Vec<u8>, n_bits: usize) -> Self {
        let mut d = Self { payload, n_bits, i: 0, low: 0, high: MASK, value: 0 };
        for _ in 0..32 {
            let b = d.next_bit();
            d.value = ((d.value << 1) | b) & MASK;
        }
        d
    }

    #[inline]
    fn next_bit(&mut self) -> u64 {
        let i = self.i;
        self.i += 1;
        if i < self.n_bits {
            ((self.payload[i / 8] >> (7 - (i % 8))) & 1) as u64
        } else {
            0 // zero tail past the end, matching the reference
        }
    }

    /// `cum` is the cumulative table (len V+1, cum[V] == total). Returns the symbol.
    pub fn decode(&mut self, cum: &[u64]) -> usize {
        let total = cum[cum.len() - 1];
        let scaled = self.target(total);
        // np.searchsorted(cum, scaled, side="right") - 1, then clamp to [0, V-1]
        let sr = cum.partition_point(|&x| x <= scaled) as i64 - 1;
        let s = sr.clamp(0, cum.len() as i64 - 2) as usize;
        self.consume(cum[s], cum[s + 1], total);
        s
    }

    /// Whether [`RangeDecoder::decode`] would return the symbol whose interval is `[cl, ch)` of
    /// `total` — given a strictly increasing table, `decode` picks the unique interval holding the
    /// target — and if so, consumes it exactly as `decode` does. Lets a verifier that already knows
    /// every symbol's interval check a payload without materialising the full tables.
    pub fn check(&mut self, cl: u64, ch: u64, total: u64) -> bool {
        let scaled = self.target(total);
        if scaled < cl || scaled >= ch { return false; }
        self.consume(cl, ch, total);
        true
    }

    /// The scaled code value the next symbol's interval must contain.
    #[inline]
    fn target(&self, total: u64) -> u64 {
        let rng = self.high - self.low + 1;
        ((self.value - self.low + 1) * total - 1) / rng
    }

    /// Narrow to the symbol interval `[cl, ch)` of `total` and renormalise.
    #[inline]
    fn consume(&mut self, cl: u64, ch: u64, total: u64) {
        let rng = self.high - self.low + 1;
        self.high = self.low + (rng * ch) / total - 1;
        self.low = self.low + (rng * cl) / total;
        loop {
            if self.high < HALF {
            } else if self.low >= HALF {
                self.low -= HALF;
                self.high -= HALF;
                self.value -= HALF;
            } else if self.low >= QUARTER && self.high < THREEQ {
                self.low -= QUARTER;
                self.high -= QUARTER;
                self.value -= QUARTER;
            } else {
                break;
            }
            self.low = (self.low << 1) & MASK;
            self.high = ((self.high << 1) | 1) & MASK;
            let b = self.next_bit();
            self.value = ((self.value << 1) | b) & MASK;
        }
    }
}
