//! Binary arithmetic coder (carry-less, 32-bit range, byte-wise output).
//!
//! Probabilities are 16-bit: `p` = P(bit = 1) · 65536, valid in `1..=65535`. The interval
//! `[x1, x2]` always spans at least 2^24 values (its top bytes differ), so with `p < 65536` the split
//! point satisfies `x1 <= xmid < x2` and neither side can become empty.
//!
//! Encoder and decoder share one model loop through the [`Coder`] trait: the encoder codes the bit
//! it is given, the decoder ignores it and returns the decoded bit. Driving both directions through
//! the same generic function makes encoder/decoder drift impossible by construction.

pub trait Coder {
    /// Code one bit with probability `p` (P(1)·65536) and return the bit actually coded.
    fn code(&mut self, bit: u32, p: u32) -> u32;
}

pub struct Encoder {
    x1: u32,
    x2: u32,
    out: Vec<u8>,
    /// Concurrent verification: output is also sent down this channel as it is produced (with the
    /// count already sent).
    tap: Option<(std::sync::mpsc::Sender<Vec<u8>>, usize)>,
    /// Bits left before pending output is sent to the tap regardless of its size, so a reader
    /// never waits long on highly compressible data (which emits few bytes).
    tick: u32,
}

/// Output bytes per chunk sent to a tap, and the coded bits after which pending output is sent
/// anyway (4 KB of input).
const TAP_CHUNK: usize = 4096;
const TAP_BITS: u32 = 1 << 15;

impl Encoder {
    pub fn new() -> Self { Encoder { x1: 0, x2: u32::MAX, out: Vec::new(), tap: None, tick: u32::MAX } }

    /// An encoder whose output is also streamed to `tx` while it is produced; the channel closes
    /// when the encoder finishes (or is dropped). Never changes the output.
    pub fn tapped(tx: std::sync::mpsc::Sender<Vec<u8>>) -> Self { Encoder { tap: Some((tx, 0)), tick: TAP_BITS, ..Encoder::new() } }

    #[inline]
    pub fn encode(&mut self, bit: u32, p: u32) {
        debug_assert!((1..=65535).contains(&p));
        let xmid = self.x1 + (((self.x2 - self.x1) as u64 * p as u64) >> 16) as u32;
        if bit != 0 { self.x2 = xmid } else { self.x1 = xmid + 1 }
        while (self.x1 ^ self.x2) & 0xff00_0000 == 0 {
            self.out.push((self.x2 >> 24) as u8);
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 255;
            if self.tap.as_ref().is_some_and(|t| self.out.len() - t.1 >= TAP_CHUNK) { self.flush_tap(); }
        }
        self.tick -= 1;
        if self.tick == 0 { self.flush_tap(); }
    }

    #[cold]
    fn flush_tap(&mut self) {
        self.tick = if self.tap.is_some() { TAP_BITS } else { u32::MAX };
        if let Some((tx, sent)) = self.tap.as_mut() {
            if *sent < self.out.len() {
                // a verifier that has stopped listening only means no verification: keep coding
                let _ = tx.send(self.out[*sent..].to_vec());
                *sent = self.out.len();
            }
        }
    }

    /// Flush: emitting all four bytes of `x1` makes the decoder's value exactly `x1` (it reads zeros
    /// past the end), which always lies inside the final interval.
    pub fn finish(mut self) -> Vec<u8> {
        self.out.extend_from_slice(&self.x1.to_be_bytes());
        self.flush_tap();
        std::mem::take(&mut self.out)
    }

    pub fn len(&self) -> usize { self.out.len() }
}

impl Coder for Encoder {
    #[inline]
    fn code(&mut self, bit: u32, p: u32) -> u32 { self.encode(bit, p); bit }
}

/// Where a [`DecoderOf`] reads coded bytes from.
pub trait Source {
    /// The next coded byte; zero past the end of the stream.
    fn next(&mut self) -> u8;
}

/// A coded stream held in memory.
pub struct Slice<'a> {
    src: &'a [u8],
    pos: usize,
}

impl Source for Slice<'_> {
    #[inline]
    fn next(&mut self) -> u8 {
        let b = self.src.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }
}

/// A coded stream arriving in chunks from a [`Encoder::tapped`] encoder while it is produced:
/// reads wait for the bytes they need, and read zeros once the finished stream is exhausted —
/// exactly what a [`Decoder`] over the finished stream reads.
pub struct Stream {
    rx: std::sync::mpsc::Receiver<Vec<u8>>,
    buf: Vec<u8>,
    pos: usize,
    closed: bool,
}

impl Stream {
    pub fn new(rx: std::sync::mpsc::Receiver<Vec<u8>>) -> Self { Stream { rx, buf: Vec::new(), pos: 0, closed: false } }

    #[cold]
    fn fill(&mut self) {
        while self.pos >= self.buf.len() && !self.closed {
            match self.rx.recv() {
                Ok(c) => self.buf.extend_from_slice(&c),
                Err(_) => self.closed = true,
            }
        }
    }

    /// The whole stream, once the encoder has finished (waits for it).
    pub fn into_bytes(mut self) -> Vec<u8> {
        while let Ok(c) = self.rx.recv() { self.buf.extend_from_slice(&c); }
        self.buf
    }
}

impl Source for Stream {
    #[inline]
    fn next(&mut self) -> u8 {
        if self.pos >= self.buf.len() { self.fill(); }
        let b = self.buf.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }
}

pub struct DecoderOf<S: Source> {
    x1: u32,
    x2: u32,
    x: u32,
    src: S,
}

/// The decoder over an in-memory stream.
pub type Decoder<'a> = DecoderOf<Slice<'a>>;

impl<'a> DecoderOf<Slice<'a>> {
    pub fn new(src: &'a [u8]) -> Self { DecoderOf::with_source(Slice { src, pos: 0 }) }

    /// Bytes consumed so far (may exceed the input length by the zero padding read at the end).
    pub fn consumed(&self) -> usize { self.src.pos }
}

impl<S: Source> DecoderOf<S> {
    pub fn with_source(src: S) -> Self {
        let mut d = DecoderOf { x1: 0, x2: u32::MAX, x: 0, src };
        for _ in 0..4 { d.x = (d.x << 8) | d.src.next() as u32; }
        d
    }

    pub fn into_source(self) -> S { self.src }

    #[inline]
    pub fn decode(&mut self, p: u32) -> u32 {
        debug_assert!((1..=65535).contains(&p));
        let xmid = self.x1 + (((self.x2 - self.x1) as u64 * p as u64) >> 16) as u32;
        let bit = (self.x <= xmid) as u32;
        if bit != 0 { self.x2 = xmid } else { self.x1 = xmid + 1 }
        while (self.x1 ^ self.x2) & 0xff00_0000 == 0 {
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 255;
            self.x = (self.x << 8) | self.src.next() as u32;
        }
        bit
    }
}

impl<S: Source> Coder for DecoderOf<S> {
    #[inline]
    fn code(&mut self, _bit: u32, p: u32) -> u32 { self.decode(p) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }

    #[test]
    fn roundtrip_random_probabilities() {
        let mut s = 0x1234_5678_9abc_def0u64;
        let n = 200_000;
        let mut bits = Vec::with_capacity(n);
        let mut probs = Vec::with_capacity(n);
        for _ in 0..n {
            let r = xorshift(&mut s);
            let p = ((r >> 16) as u32 % 65535) + 1;
            // draw the bit roughly according to p so the stream is realistic
            let bit = (((r >> 40) as u32 & 0xffff) < p) as u32;
            bits.push(bit);
            probs.push(p);
        }
        let mut e = Encoder::new();
        for i in 0..n { e.encode(bits[i], probs[i]); }
        let out = e.finish();
        let mut d = Decoder::new(&out);
        for i in 0..n { assert_eq!(d.decode(probs[i]), bits[i], "mismatch at {i}"); }
    }

    /// A tapped encoder streams its output to a decoder running at the same time: the decoder
    /// recovers every bit, sees exactly the finished stream, and the output is unchanged.
    #[test]
    fn tapped_encoder_feeds_a_concurrent_decoder() {
        let mut s = 0x5eed_u64;
        let n = 300_000;
        let probs: Vec<u32> = (0..n).map(|_| ((xorshift(&mut s) >> 16) as u32 % 65535) + 1).collect();
        let bits: Vec<u32> = probs.iter().map(|&p| (((xorshift(&mut s) >> 40) as u32 & 0xffff) < p) as u32).collect();
        let (tx, rx) = std::sync::mpsc::channel();
        let p2 = probs.clone();
        let h = std::thread::spawn(move || {
            let mut d = DecoderOf::with_source(Stream::new(rx));
            let got: Vec<u32> = p2.iter().map(|&p| d.decode(p)).collect();
            (got, d.into_source().into_bytes())
        });
        let mut e = Encoder::tapped(tx);
        for i in 0..n { e.encode(bits[i], probs[i]); }
        let out = e.finish();
        let (got, seen) = h.join().unwrap();
        assert!(out.len() > 4 * TAP_CHUNK, "stream too short to arrive in several chunks");
        assert!(got == bits, "concurrent decode differs");
        assert!(seen == out, "streamed bytes differ from the finished stream");
        let mut plain = Encoder::new();
        for i in 0..n { plain.encode(bits[i], probs[i]); }
        assert!(plain.finish() == out, "tapping changed the output");
    }

    #[test]
    fn extreme_probabilities_and_cost() {
        // p = 65535 on an all-ones stream must cost ~2.2e-5 bits per bit.
        let n = 1_000_000;
        let mut e = Encoder::new();
        for _ in 0..n { e.encode(1, 65535); }
        let out = e.finish();
        assert!(out.len() < 12, "all-ones stream at p=65535 took {} bytes", out.len());
        let mut d = Decoder::new(&out);
        for _ in 0..n { assert_eq!(d.decode(65535), 1); }
        // and the worst case: always coding the unlikely symbol
        let mut e = Encoder::new();
        for _ in 0..1000 { e.encode(0, 65535); }
        let out = e.finish();
        let mut d = Decoder::new(&out);
        for _ in 0..1000 { assert_eq!(d.decode(65535), 0); }
    }

    #[test]
    fn cost_close_to_entropy() {
        // Code bits drawn with P(1)=0.1 at the true probability; overhead must be < 0.1%.
        let mut s = 99u64;
        let n = 400_000;
        let p = 6554u32; // 0.1 * 65536
        let mut e = Encoder::new();
        let mut ideal = 0f64;
        for _ in 0..n {
            let bit = ((xorshift(&mut s) & 0xffff) < p as u64) as u32;
            ideal += if bit == 1 { -(p as f64 / 65536.0).log2() } else { -(1.0 - p as f64 / 65536.0).log2() };
            e.encode(bit, p);
        }
        let got = e.finish().len() as f64 * 8.0;
        assert!(got < ideal * 1.001 + 64.0, "coded {got} bits vs ideal {ideal}");
    }
}
