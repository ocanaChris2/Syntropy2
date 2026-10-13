//! LZW as GIF and TIFF write it: codes back to bytes, and the same codes again.
//!
//! An LZW stream says nothing but its bytes, if its encoder did the usual: the longest phrase of
//! its dictionary each time, the dictionary cleared when it is full. [`decode`] reads the codes
//! and then runs that usual encoder over the bytes they stand for; where the stream differs —
//! a clear code (encoders disagree by a code or two on when the dictionary is full, some never
//! clear, some clear early), a phrase that is not the longest — it notes the event. From the
//! bytes and the events [`encode`] writes the stream again, bit for bit. A stream of an encoder
//! that does not use the dictionary at all (every code a single byte) would be all events: it
//! is refused, and stays as it is.

/// The two dialects.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Bits of a byte value: a GIF's minimum code size (2 to 8); 8 in TIFF and PDF.
    pub bits: u32,
    /// Codes are packed from the high bit down (TIFF, PDF) instead of from the low bit up (GIF).
    pub msb: bool,
    /// The code width grows one code early (TIFF, PDF).
    pub early: bool,
}

impl Params {
    pub const TIFF: Params = Params { bits: 8, msb: true, early: true };
    pub fn gif(bits: u32) -> Params { Params { bits, msb: false, early: false } }
}

const MAX: usize = 4096;
/// Record flags: the stream starts with a clear code; ends with the end code.
const STARTS_CLEAR: u8 = 1;
const ENDS: u8 = 2;

/// The decoder's state as codes are read: how many codes exist and how wide the next one is.
struct Width {
    clear: usize,
    next: usize,
    size: u32,
    early: bool,
    min: u32,
}

impl Width {
    fn new(p: &Params) -> Width { Width { clear: 1 << p.bits, next: (1 << p.bits) + 2, size: p.bits + 1, early: p.early, min: p.bits + 1 } }

    fn reset(&mut self) {
        self.next = self.clear + 2;
        self.size = self.min;
    }

    /// A code was read that adds an entry (every code but the first after a clear).
    fn added(&mut self) {
        if self.next < MAX {
            self.next += 1;
            if self.next + self.early as usize == 1 << self.size && self.size < 12 { self.size += 1; }
        }
    }
}

/// The usual encoder's dictionary: for a phrase and a byte, the code of the phrase one longer.
struct Trie {
    child: Vec<u16>,
    used: Vec<u32>,
    len: Vec<u16>,
}

impl Trie {
    fn new() -> Trie { Trie { child: vec![0; MAX * 256], used: Vec::with_capacity(MAX), len: vec![1; MAX + 1] } }

    fn reset(&mut self) {
        for &k in &self.used { self.child[k as usize] = 0; }
        self.used.clear();
    }

    /// The phrase of `code` followed by `byte` gets code `new` (unless it has one already).
    fn add(&mut self, code: usize, byte: u8, new: usize) {
        let k = code << 8 | byte as usize;
        self.len[new] = self.len[code] + 1;
        if self.child[k] == 0 {
            self.child[k] = new as u16;
            self.used.push(k as u32);
        }
    }

    /// The longest phrase at the start of `b`: its code.
    #[inline]
    fn longest(&self, b: &[u8]) -> usize {
        let mut cur = b[0] as usize;
        for &x in &b[1..] {
            match self.child[cur << 8 | x as usize] {
                0 => break,
                c => cur = c as usize,
            }
        }
        cur
    }
}

struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    msb: bool,
}

impl Bits<'_> {
    /// The next `n` bits as a code (None: fewer are left).
    #[inline]
    fn get(&mut self, n: u32) -> Option<usize> {
        if self.pos + n as usize > self.data.len() * 8 { return None; }
        let byte = self.pos >> 3;
        let mut v = 0u32;
        for i in 0..3 {
            let x = self.data.get(byte + i).copied().unwrap_or(0) as u32;
            if self.msb { v = v << 8 | x } else { v |= x << (8 * i) }
        }
        let k = (self.pos & 7) as u32;
        self.pos += n as usize;
        Some((if self.msb { v >> (24 - k - n) } else { v >> k }) as usize & ((1 << n) - 1))
    }
}

#[derive(Default)]
struct Packer {
    out: Vec<u8>,
    acc: u64,
    n: u32,
}

impl Packer {
    #[inline]
    fn put(&mut self, code: usize, size: u32, msb: bool) {
        if msb {
            self.acc = self.acc << size | code as u64;
            self.n += size;
            while self.n >= 8 {
                self.n -= 8;
                self.out.push((self.acc >> self.n) as u8);
            }
            self.acc &= (1 << self.n) - 1;
        } else {
            self.acc |= (code as u64) << self.n;
            self.n += size;
            while self.n >= 8 {
                self.out.push(self.acc as u8);
                self.acc >>= 8;
                self.n -= 8;
            }
        }
    }

    /// The last byte, its unused bits filled from `pad`.
    fn finish(mut self, pad: u8, msb: bool) -> Vec<u8> {
        if self.n > 0 {
            let used = self.n;
            self.out.push(if msb { (self.acc << (8 - used)) as u8 | pad & (0xFF >> used) } else { self.acc as u8 | pad & (0xFFu8 << used) });
        }
        self.out
    }
}

fn put_var(o: &mut Vec<u8>, mut v: usize) {
    while v >= 0x80 {
        o.push(v as u8 | 0x80);
        v >>= 7;
    }
    o.push(v as u8);
}

fn get_var(b: &[u8], p: &mut usize) -> Option<usize> {
    let mut v = 0usize;
    for k in 0..5 {
        let x = *b.get(*p)?;
        *p += 1;
        v |= ((x & 0x7F) as usize) << (7 * k);
        if x & 0x80 == 0 { return Some(v); }
    }
    None
}

/// A stream read back.
pub struct Decoded {
    pub bytes: Vec<u8>,
    /// What [`encode`] needs beside the bytes.
    pub record: Vec<u8>,
    /// Bytes of the stream its codes take (what follows is not LZW's).
    pub consumed: usize,
}

/// Read the LZW stream at the start of `data` (at most `most` bytes of content): None if it is
/// not one, or one no usual encoder wrote.
pub fn decode(data: &[u8], p: &Params, most: usize) -> Option<Decoded> {
    if !(2..=8).contains(&p.bits) { return None; }
    let mut w = Width::new(p);
    let (clear, eoi) = (w.clear, w.clear + 1);
    let mut r = Bits { data, pos: 0, msb: p.msb };
    // the decoder's dictionary: each code's phrase as the code before and a last byte
    let mut prefix = vec![0u16; MAX];
    let mut last = vec![0u8; MAX];
    let mut first = vec![0u8; MAX];
    let mut len = vec![1u32; MAX];
    for c in 0..clear { (last[c], first[c]) = (c as u8, c as u8); }
    let mut bytes: Vec<u8> = vec![];
    // every code read: a phrase's code, or `clear`
    let mut codes: Vec<u16> = vec![];
    let mut prev: Option<usize> = None;
    let mut flags = 0u8;
    loop {
        let before = r.pos;
        let Some(c) = r.get(w.size) else { break };
        if c == clear {
            codes.push(clear as u16);
            w.reset();
            prev = None;
            continue;
        }
        if c == eoi {
            flags |= ENDS;
            break;
        }
        // the phrase's length; once the content is complete, bits that read as one more phrase
        // are the last byte's padding (a stream need not end with the end code)
        let l = match prev {
            None if c < clear => 1,
            Some(_) if c < w.next => len[c] as usize,
            Some(pv) if c == w.next && w.next < MAX => len[pv] as usize + 1,
            _ => return None,
        };
        if bytes.len() + l > most {
            r.pos = before;
            break;
        }
        if let Some(pv) = prev {
            // the phrase the code before makes with this one's first byte
            if w.next < MAX {
                let n = w.next;
                (prefix[n], len[n], first[n]) = (pv as u16, len[pv] + 1, first[pv]);
                last[n] = if c == n { first[pv] } else { first[c] };
            }
            w.added();
        }
        // the phrase, written from its end
        let at = bytes.len();
        bytes.resize(at + l, 0);
        let mut k = c;
        for i in (0..l).rev() {
            bytes[at + i] = last[k];
            k = prefix[k] as usize;
        }
        codes.push(c as u16);
        prev = Some(c);
    }
    // what the stream's last byte holds beside codes
    let bits = r.pos;
    let consumed = bits.div_ceil(8);
    let pad = if bits % 8 == 0 { 0 } else if p.msb { data[consumed - 1] & (0xFF >> (bits % 8)) } else { data[consumed - 1] & (0xFFu8 << (bits % 8)) };
    if flags & ENDS == 0 && data.len() > consumed { return None; }
    // the usual encoder over the bytes: where does the stream differ?
    let mut record = vec![0u8, pad];
    let mut t = Trie::new();
    let mut w = Width::new(p);
    let (mut pos, mut skip, mut events) = (0usize, 0usize, 0usize);
    let mut prev: Option<usize> = None;
    for (i, &c) in codes.iter().enumerate() {
        let c = c as usize;
        if c == clear {
            if i == 0 {
                record[0] |= STARTS_CLEAR;
            } else {
                put_var(&mut record, skip);
                put_var(&mut record, 0);
                (skip, events) = (0, events + 1);
            }
            t.reset();
            w.reset();
            prev = None;
            continue;
        }
        if let Some(pv) = prev {
            if w.next < MAX { t.add(pv, bytes[pos], w.next); }
            w.added();
        }
        if t.longest(&bytes[pos..]) == c {
            skip += 1;
        } else {
            put_var(&mut record, skip);
            put_var(&mut record, c + 1);
            (skip, events) = (0, events + 1);
        }
        // (the code's length as of this dictionary: the decoder's table above is the last one's)
        pos += t.len[c] as usize;
        prev = Some(c);
    }
    record[0] |= flags;
    // an encoder that does not use the dictionary: its codes are the content, as they are
    if events * 8 > codes.len() + 64 { return None; }
    Some(Decoded { bytes, record, consumed })
}

/// Write the stream of `bytes` as `record` says ([`decode`]'s), appending to `out`.
pub fn encode(bytes: &[u8], record: &[u8], p: &Params, out: &mut Vec<u8>) -> Result<(), String> {
    let bad = || "lzw: bad record (corrupt file)".to_string();
    if !(2..=8).contains(&p.bits) || record.len() < 2 { return Err(bad()); }
    if bytes.iter().any(|&x| x as usize >> p.bits != 0) { return Err(bad()); }
    let (flags, pad) = (record[0], record[1]);
    let mut w = Width::new(p);
    let (clear, eoi) = (w.clear, w.clear + 1);
    let mut t = Trie::new();
    let mut o = Packer::default();
    if flags & STARTS_CLEAR != 0 { o.put(clear, w.size, p.msb); }
    let mut rp = 2usize;
    // codes until the next event, and the event
    let mut next_event = |rp: &mut usize| -> Result<Option<(usize, usize)>, String> {
        if *rp >= record.len() { return Ok(None); }
        let skip = get_var(record, rp).ok_or_else(bad)?;
        let what = get_var(record, rp).ok_or_else(bad)?;
        Ok(Some((skip, what)))
    };
    let mut event = next_event(&mut rp)?;
    let mut pos = 0usize;
    let mut prev: Option<usize> = None;
    loop {
        // a clear code stands between two phrases, or after the last
        if let Some((0, 0)) = event {
            o.put(clear, w.size, p.msb);
            t.reset();
            w.reset();
            prev = None;
            event = next_event(&mut rp)?;
            continue;
        }
        if pos == bytes.len() { break; }
        // the entry the decoder makes when it reads this code is one the code may be: it is in
        // the dictionary before the phrase is chosen, and counts once the code is written
        let pending = prev.is_some() && w.next < MAX;
        if let (true, Some(pv)) = (pending, prev) { t.add(pv, bytes[pos], w.next); }
        let c = match event.as_mut() {
            Some((0, what)) => {
                // the stream's own code: it must be a phrase that stands here
                let c = *what - 1;
                if c >= w.next + pending as usize || c == clear || c == eoi || pos + t.len[c] as usize > bytes.len() { return Err(bad()); }
                event = next_event(&mut rp)?;
                c
            }
            Some((skip, _)) => {
                *skip -= 1;
                t.longest(&bytes[pos..])
            }
            None => t.longest(&bytes[pos..]),
        };
        o.put(c, w.size, p.msb);
        if prev.is_some() { w.added(); }
        pos += t.len[c] as usize;
        prev = Some(c);
    }
    if event.is_some() { return Err(bad()); }
    if flags & ENDS != 0 { o.put(eoi, w.size, p.msb); }
    out.extend(o.finish(pad, p.msb));
    Ok(())
}

#[cfg(test)]
pub mod tests {
    use super::*;

    fn rng(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }

    /// How an encoder of the tests behaves: when it clears (at this many codes; 0: never), and
    /// how often it takes a phrase shorter than the longest (one in this many; 0: never).
    #[derive(Clone, Copy)]
    pub struct Style {
        pub clear_at: usize,
        pub lazy: u64,
        pub start_clear: bool,
        pub end: bool,
    }

    pub const USUAL: Style = Style { clear_at: 4096, lazy: 0, start_clear: true, end: true };

    /// An LZW stream of `bytes` by an encoder of `style`.
    pub fn compress(bytes: &[u8], p: &Params, style: Style) -> Vec<u8> {
        let mut s = 77u64;
        let mut w = Width::new(p);
        let (clear, eoi) = (w.clear, w.clear + 1);
        let mut t = Trie::new();
        let mut o = Packer::default();
        if style.start_clear { o.put(clear, w.size, p.msb); }
        let (mut pos, mut prev) = (0usize, None::<usize>);
        while pos < bytes.len() {
            if let Some(pv) = prev {
                if w.next < MAX { t.add(pv, bytes[pos], w.next); }
            }
            let mut c = t.longest(&bytes[pos..]);
            if style.lazy > 0 && rng(&mut s) % style.lazy == 0 { c = bytes[pos] as usize; }
            o.put(c, w.size, p.msb);
            if prev.is_some() { w.added(); }
            pos += t.len[c] as usize;
            prev = Some(c);
            if style.clear_at > 0 && w.next >= style.clear_at && pos < bytes.len() {
                // the entry this code will make is the last: clear
                o.put(clear, w.size, p.msb);
                t.reset();
                w.reset();
                prev = None;
            }
        }
        if style.end { o.put(eoi, w.size, p.msb); }
        o.finish(0, p.msb)
    }

    /// Bytes of `bits` bits with runs and repeats, as a picture's rows have.
    pub fn picture(n: usize, bits: u32, seed: u64) -> Vec<u8> {
        let mut s = seed;
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let v = (rng(&mut s) % (1 << bits)) as u8;
            match rng(&mut s) % 4 {
                0 => out.extend(std::iter::repeat_n(v, 1 + (rng(&mut s) % 40) as usize)),
                1 if out.len() > 64 => {
                    let from = out.len() - 1 - (rng(&mut s) % 60) as usize;
                    for k in 0..1 + (rng(&mut s) % 30) as usize { out.push(out[from + k]); }
                }
                _ => out.push(v),
            }
        }
        out.truncate(n);
        out
    }

    #[test]
    fn streams_come_back() {
        let styles = [USUAL, Style { clear_at: 4095, ..USUAL }, Style { clear_at: 0, ..USUAL }, Style { clear_at: 700, start_clear: false, ..USUAL },
            Style { lazy: 40, ..USUAL }, Style { end: false, ..USUAL }, Style { clear_at: 4094, lazy: 300, start_clear: false, end: false }];
        for p in [Params::gif(8), Params::gif(2), Params::gif(5), Params::TIFF] {
            for (k, &style) in styles.iter().enumerate() {
                for n in [0usize, 1, 2, 300, 70_000] {
                    let what = format!("{p:?}, style {k}, {n} bytes");
                    let bytes = picture(n, p.bits, 3 + k as u64);
                    let mut stream = compress(&bytes, &p, style);
                    // what follows an end code is not the stream's
                    if style.end { stream.extend_from_slice(b"after"); }
                    let d = decode(&stream, &p, n).unwrap_or_else(|| panic!("{what}: not decoded"));
                    assert!(d.bytes == bytes, "{what}: bytes");
                    assert_eq!(d.consumed, stream.len() - if style.end { 5 } else { 0 }, "{what}: consumed");
                    let mut back = vec![];
                    encode(&d.bytes, &d.record, &p, &mut back).unwrap_or_else(|e| panic!("{what}: {e}"));
                    assert!(back == stream[..d.consumed], "{what}: stream");
                    // the usual encoder leaves a record of a few bytes
                    if k == 0 && n == 70_000 { assert!(d.record.len() < 80, "{what}: record of {} bytes", d.record.len()); }
                    // never more content than the most allowed
                    assert!(n == 0 || decode(&stream, &p, n - 1).is_none_or(|d| d.bytes.len() < n && !style.end), "{what}: more than the most allowed");
                }
            }
            // every byte a code of its own: not worth a record
            let bytes = picture(20_000, p.bits, 9);
            assert!(decode(&compress(&bytes, &p, Style { lazy: 1, ..USUAL }), &p, 20_000).is_none());
        }
        // bits in the last byte that are no code's are kept
        let bytes = picture(500, 8, 5);
        for p in [Params::gif(8), Params::TIFF] {
            let mut stream = compress(&bytes, &p, Style { end: false, ..USUAL });
            let l = stream.len() - 1;
            stream[l] ^= if p.msb { 1 } else { 0x80 };
            if let Some(d) = decode(&stream, &p, 500) {
                let mut back = vec![];
                encode(&d.bytes, &d.record, &p, &mut back).unwrap();
                assert!(back == stream[..d.consumed], "{p:?}: padding bits");
            }
        }
    }

    /// Damaged streams and records: None or an error, or other bytes; never a panic, and
    /// never more output than the bytes can be codes for.
    #[test]
    fn damaged_streams_and_records_do_not_panic() {
        let mut s = 5u64;
        for p in [Params::gif(8), Params::gif(3), Params::TIFF] {
            let bytes = picture(30_000, p.bits, 21);
            let stream = compress(&bytes, &p, Style { lazy: 90, clear_at: 4095, ..USUAL });
            let d = decode(&stream, &p, bytes.len()).unwrap();
            for _ in 0..600 {
                let mut bad = stream.clone();
                for _ in 0..1 + rng(&mut s) % 4 {
                    let i = (rng(&mut s) % bad.len() as u64) as usize;
                    if rng(&mut s) % 5 == 0 { bad.truncate(i.max(1)); } else { bad[i] ^= 1 << (rng(&mut s) % 8); }
                }
                if let Some(d) = decode(&bad, &p, 2 * bytes.len()) {
                    let mut back = vec![];
                    encode(&d.bytes, &d.record, &p, &mut back).unwrap();
                    assert!(back == bad[..d.consumed], "{p:?}: a damaged stream that decodes must come back");
                }
                let mut rec = d.record.clone();
                let i = (rng(&mut s) % rec.len() as u64) as usize;
                if rng(&mut s) % 4 == 0 { rec.truncate(i); } else { rec[i] ^= 1 << (rng(&mut s) % 8); }
                let mut back = vec![];
                let _ = encode(&d.bytes, &rec, &p, &mut back);
                assert!(back.len() <= 2 * bytes.len() + 4096 * 2, "{p:?}: {} bytes from a damaged record", back.len());
            }
        }
    }
}
