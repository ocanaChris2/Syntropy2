//! Deflate decoder that records the complete parse (see the module docs).
use super::*;

struct Bits<'a> {
    d: &'a [u8],
    pos: usize, // byte position of the next unread byte
    buf: u64,
    n: u32,
    /// Set when a read ran past the end of the data (a truncated stream, not a malformed one).
    eof: bool,
}

impl<'a> Bits<'a> {
    fn new(d: &'a [u8]) -> Self { Bits { d, pos: 0, buf: 0, n: 0, eof: false } }

    #[inline]
    fn need(&mut self, k: u32) -> Result<(), String> {
        while self.n < k {
            let Some(&b) = self.d.get(self.pos) else {
                self.eof = true;
                return Err("deflate: unexpected end of stream".into());
            };
            self.buf |= (b as u64) << self.n;
            self.pos += 1;
            self.n += 8;
        }
        Ok(())
    }

    /// Buffer up to `k` bits without failing at the end of the data.
    #[inline]
    fn fill(&mut self, k: u32) {
        while self.n < k {
            let Some(&b) = self.d.get(self.pos) else { return };
            self.buf |= (b as u64) << self.n;
            self.pos += 1;
            self.n += 8;
        }
    }

    #[inline]
    fn get(&mut self, k: u32) -> Result<u32, String> {
        if k == 0 { return Ok(0); }
        self.need(k)?;
        let v = (self.buf & ((1u64 << k) - 1)) as u32;
        self.buf >>= k;
        self.n -= k;
        Ok(v)
    }

    /// Bits remaining in the current partial byte (dropped by byte alignment).
    fn align(&mut self) -> (u8, u8) {
        let k = self.n % 8;
        let v = (self.buf & ((1u64 << k) - 1)) as u8;
        self.buf >>= k;
        self.n -= k;
        (k as u8, v)
    }

    /// Bytes consumed, counting a partially used byte as consumed.
    fn consumed(&self) -> usize { self.pos - (self.n / 8) as usize }
}

/// Canonical Huffman decoder: a table for codes up to `FAST_BITS` long, bit-by-bit beyond.
struct Huff {
    count: [u16; 16],
    syms: Vec<u16>,
    /// Indexed by the next FAST_BITS input bits (LSB first): symbol << 4 | code length, 0 = slow path.
    fast: Vec<u16>,
}

const FAST_BITS: u32 = 10;

impl Huff {
    fn new(lens: &[u8]) -> Result<Huff, String> {
        let mut h = Self::canonical(lens)?;
        let mut fast = vec![0u16; 1 << FAST_BITS];
        for (sym, &(code, len)) in canonical_codes(lens).iter().enumerate() {
            if len == 0 || len as u32 > FAST_BITS { continue; }
            let mut i = code as usize;
            while i < fast.len() {
                fast[i] = (sym as u16) << 4 | len as u16;
                i += 1 << len;
            }
        }
        h.fast = fast;
        Ok(h)
    }

    #[inline]
    fn decode(&self, b: &mut Bits) -> Result<u16, String> {
        b.fill(FAST_BITS);
        let e = self.fast[(b.buf & ((1 << FAST_BITS) - 1)) as usize];
        let len = (e & 15) as u32;
        if e != 0 && len <= b.n {
            b.buf >>= len;
            b.n -= len;
            return Ok(e >> 4);
        }
        self.decode_slow(b)
    }

    fn canonical(lens: &[u8]) -> Result<Huff, String> {
        let mut count = [0u16; 16];
        for &l in lens { count[l as usize] += 1; }
        count[0] = 0;
        // reject over-subscribed codes (incomplete codes are allowed, as in zlib's single-code case)
        let mut left = 1i32;
        for l in 1..16 {
            left <<= 1;
            left -= count[l] as i32;
            if left < 0 { return Err("deflate: over-subscribed Huffman code".into()); }
        }
        let mut offs = [0u16; 16];
        for l in 1..15 { offs[l + 1] = offs[l] + count[l]; }
        let mut syms = vec![0u16; lens.len()];
        for (s, &l) in lens.iter().enumerate() {
            if l > 0 {
                syms[offs[l as usize] as usize] = s as u16;
                offs[l as usize] += 1;
            }
        }
        Ok(Huff { count, syms, fast: vec![] })
    }

    fn decode_slow(&self, b: &mut Bits) -> Result<u16, String> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for l in 1..16 {
            code |= b.get(1)? as i32;
            let c = self.count[l] as i32;
            if code - c < first {
                return Ok(self.syms[(index + (code - first)) as usize]);
            }
            index += c;
            first += c;
            first <<= 1;
            code <<= 1;
        }
        Err("deflate: invalid Huffman code".into())
    }
}

/// Decode a raw deflate stream from the start of `data`, producing at most `max_out` bytes.
pub fn inflate(data: &[u8], max_out: usize) -> Result<(Vec<u8>, Parse), String> {
    let mut b = Bits::new(data);
    let mut out: Vec<u8> = Vec::new();
    let mut blocks = vec![];
    loop {
        let last = b.get(1)? == 1;
        let bt = b.get(2)?;
        match bt {
            0 => {
                let (pad_bits, pad_val) = b.align();
                let len = b.get(16)? as u16;
                let nlen = b.get(16)? as u16;
                if len != !nlen { return Err("deflate: stored block length check failed".into()); }
                // bits are byte aligned here: copy the raw bytes
                let start = b.consumed();
                let end = start + len as usize;
                if end > data.len() { return Err("deflate: stored block past end".into()); }
                if out.len() + len as usize > max_out { return Err("deflate: output limit".into()); }
                out.extend_from_slice(&data[start..end]);
                b.pos = end;
                b.buf = 0;
                b.n = 0;
                blocks.push(Block { last, kind: Kind::Stored { pad_bits, pad_val, len }, toks: vec![] });
            }
            1 | 2 => {
                let (kind, lit, dist) = if bt == 1 {
                    (Kind::Fixed, Huff::new(&fixed_lit_lens())?, Huff::new(&[5u8; 30])?)
                } else {
                    let hlit = b.get(5)? as u16 + 257;
                    let hdist = b.get(5)? as u16 + 1;
                    let hclen = b.get(4)? as u16 + 4;
                    let mut cl_lens = [0u8; 19];
                    for &i in CL_ORDER.iter().take(hclen as usize) { cl_lens[i] = b.get(3)? as u8; }
                    let clh = Huff::new(&cl_lens)?;
                    let total = (hlit + hdist) as usize;
                    let mut cl_syms = vec![];
                    let mut n = 0usize;
                    let mut have_prev = false;
                    while n < total {
                        let s = clh.decode(&mut b)? as u8;
                        let (x, run) = match s {
                            0..=15 => (0u8, 1usize),
                            16 => { if !have_prev { return Err("deflate: repeat with no previous length".into()); } let x = b.get(2)? as u8; (x, 3 + x as usize) }
                            17 => { let x = b.get(3)? as u8; (x, 3 + x as usize) }
                            _ => { let x = b.get(7)? as u8; (x, 11 + x as usize) }
                        };
                        cl_syms.push((s, x));
                        have_prev = true;
                        n += run;
                    }
                    if n != total { return Err("deflate: code lengths overflow".into()); }
                    let lens = expand_cl_syms(&cl_syms, total)?;
                    if lens[256] == 0 { return Err("deflate: no end-of-block code".into()); }
                    let lit = Huff::new(&lens[..hlit as usize])?;
                    let dist = Huff::new(&lens[hlit as usize..])?;
                    (Kind::Dynamic { hlit, hdist, hclen, cl_lens, cl_syms }, lit, dist)
                };
                let mut toks = vec![];
                loop {
                    let s = lit.decode(&mut b)?;
                    if s < 256 {
                        if out.len() >= max_out { return Err("deflate: output limit".into()); }
                        out.push(s as u8);
                        toks.push(Tok::Lit(s as u8));
                    } else if s == 256 {
                        break;
                    } else {
                        let li = (s - 257) as usize;
                        if li >= 29 { return Err("deflate: bad length code".into()); }
                        let x = b.get(LEN_EXTRA[li] as u32)? as u16;
                        // code 284 + extra 31 is a second spelling of length 258
                        let alt_flag = li == 27 && x == 31;
                        let len = if alt_flag { 258 } else { LEN_BASE[li] + x };
                        let ds = dist.decode(&mut b)? as usize;
                        if ds >= 30 { return Err("deflate: bad distance code".into()); }
                        let d = DIST_BASE[ds] + b.get(DIST_EXTRA[ds] as u32)? as u16;
                        if d as usize > out.len() { return Err("deflate: distance before start".into()); }
                        if out.len() + len as usize > max_out { return Err("deflate: output limit".into()); }
                        let start = out.len() - d as usize;
                        for k in 0..len as usize { let v = out[start + k]; out.push(v); }
                        toks.push(Tok::Match { len, dist: d, alt: alt_flag });
                    }
                }
                blocks.push(Block { last, kind, toks });
            }
            _ => return Err("deflate: reserved block type".into()),
        }
        if last { break; }
    }
    let end_pad = b.align();
    let consumed = b.consumed();
    Ok((out, Parse { blocks, end_pad, consumed }))
}

/// Why [`extent`] stopped.
#[derive(Debug, PartialEq, Eq)]
pub enum ExtentError {
    /// The data ended inside the stream.
    Truncated,
    /// Not a valid deflate stream (or over the output limit).
    Invalid,
}

/// Length of the deflate stream at the start of `data` and of its decompressed output, found by
/// walking the stream's structure without producing output (distances are checked against the
/// output length). Used to place block cuts around streams.
pub fn extent(data: &[u8], max_out: usize) -> Result<(usize, usize), ExtentError> {
    let mut b = Bits::new(data);
    let r = walk(&mut b, max_out);
    match r {
        Ok(out) => Ok((b.consumed(), out)),
        Err(_) if b.eof => Err(ExtentError::Truncated),
        Err(_) => Err(ExtentError::Invalid),
    }
}

fn walk(b: &mut Bits, max_out: usize) -> Result<usize, String> {
    let mut out = 0usize;
    let (fixed_lit, fixed_dist) = (Huff::new(&fixed_lit_lens())?, Huff::new(&[5u8; 30])?);
    loop {
        let last = b.get(1)? == 1;
        match b.get(2)? {
            0 => {
                b.align();
                let len = b.get(16)? as usize;
                if len as u16 != !(b.get(16)? as u16) { return Err("stored length check".into()); }
                let start = b.consumed();
                if start + len > b.d.len() { b.eof = true; return Err("stored block past end".into()); }
                out += len;
                if out > max_out { return Err("output limit".into()); }
                b.pos = start + len;
                b.buf = 0;
                b.n = 0;
            }
            bt @ (1 | 2) => {
                let dynamic;
                let (lit, dist) = if bt == 1 { (&fixed_lit, &fixed_dist) } else {
                    let hlit = b.get(5)? as usize + 257;
                    let hdist = b.get(5)? as usize + 1;
                    let hclen = b.get(4)? as usize + 4;
                    let mut cl_lens = [0u8; 19];
                    for &i in CL_ORDER.iter().take(hclen) { cl_lens[i] = b.get(3)? as u8; }
                    let clh = Huff::new(&cl_lens)?;
                    let mut lens = Vec::with_capacity(hlit + hdist);
                    while lens.len() < hlit + hdist {
                        match clh.decode(b)? {
                            s @ 0..=15 => lens.push(s as u8),
                            16 => { let p = *lens.last().ok_or("repeat without length")?; for _ in 0..3 + b.get(2)? { lens.push(p); } }
                            17 => for _ in 0..3 + b.get(3)? { lens.push(0); },
                            _ => for _ in 0..11 + b.get(7)? { lens.push(0); },
                        }
                    }
                    if lens.len() != hlit + hdist || lens[256] == 0 { return Err("bad code lengths".into()); }
                    dynamic = (Huff::new(&lens[..hlit])?, Huff::new(&lens[hlit..])?);
                    (&dynamic.0, &dynamic.1)
                };
                loop {
                    let s = lit.decode(b)?;
                    if s < 256 {
                        out += 1;
                    } else if s == 256 {
                        break;
                    } else {
                        let li = (s - 257) as usize;
                        if li >= 29 { return Err("bad length code".into()); }
                        let len = LEN_BASE[li] as usize + b.get(LEN_EXTRA[li] as u32)? as usize;
                        let ds = dist.decode(b)? as usize;
                        if ds >= 30 { return Err("bad distance code".into()); }
                        let d = DIST_BASE[ds] as usize + b.get(DIST_EXTRA[ds] as u32)? as usize;
                        if d > out { return Err("distance before start".into()); }
                        out += len.min(258);
                    }
                    if out > max_out { return Err("output limit".into()); }
                }
            }
            _ => return Err("reserved block type".into()),
        }
        if last { break; }
    }
    b.align();
    Ok(out)
}
