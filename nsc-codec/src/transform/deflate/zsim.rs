//! An emulation of zlib's match finder and parse decisions (deflate_fast / deflate_slow), used to
//! predict the token stream of a deflate stream from its decompressed data.
//!
//! Most real deflate streams (libpng, zip tools, PDF writers, Java, Python) come from zlib, and
//! zlib's parse is a deterministic function of the data and a few parameters (level, strategy,
//! window size, hash size). When the emulation matches, the token stream costs almost nothing to
//! store. When it does not, the actual token is recorded and the emulator resynchronises to it,
//! so correctness never depends on the emulation being exact.
//!
//! Positions are absolute. zlib's sliding window is equivalent here, except that zlib treats
//! window index 0 as "no match" (NIL). For position 0 that is emulated directly; for later
//! window bases the affected position is already beyond the maximum distance whenever it would
//! be queried.
use super::Tok;

const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 258;
const TOO_FAR: usize = 4096;
const NIL: u32 = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Params {
    pub level: u8,     // 1..=9
    pub filtered: bool, // Z_FILTERED strategy (libpng's default for filtered rows)
    pub wbits: u8,     // window size log2 (8..=15)
    pub mem_level: u8, // hash bits = mem_level + 7
}

impl Params {
    pub fn to_byte(&self) -> [u8; 2] { [self.level | (self.filtered as u8) << 7, self.wbits | (self.mem_level - 1) << 4] }
    pub fn from_bytes(b: [u8; 2]) -> Option<Params> {
        let p = Params { level: b[0] & 0x7f, filtered: b[0] & 0x80 != 0, wbits: b[1] & 0xf, mem_level: (b[1] >> 4) + 1 };
        ((1..=9).contains(&p.level) && (8..=15).contains(&p.wbits) && (1..=9).contains(&p.mem_level)).then_some(p)
    }
}

/// zlib's configuration_table: (good, lazy-or-max_insert, nice, chain, slow)
const CONFIG: [(usize, usize, usize, usize, bool); 10] = [
    (0, 0, 0, 0, false),
    (4, 4, 8, 4, false), (4, 5, 16, 8, false), (4, 6, 32, 32, false),
    (4, 4, 16, 16, true), (8, 16, 32, 32, true), (8, 16, 128, 128, true),
    (8, 32, 128, 256, true), (32, 128, 258, 1024, true), (32, 258, 258, 4096, true),
];

pub struct Zsim<'a> {
    d: &'a [u8],
    good: usize,
    lazy: usize,
    nice: usize,
    chain: usize,
    slow: bool,
    filtered: bool,
    wmask: usize,
    max_dist: usize,
    hmask: u32,
    hshift: u32,
    head: Vec<u32>,
    prev: Vec<u32>,
    ins_h: u32,
    strstart: usize,
    match_length: usize,
    match_start: usize,
    prev_length: usize,
    prev_match: usize,
    match_available: bool,
    /// Undo log of the current prediction: (table 0 = head / 1 = prev, index, old value).
    log: Vec<(u8, u32, u32)>,
    saved: (u32, usize, usize, usize, usize, usize, bool),
}

impl<'a> Zsim<'a> {
    pub fn new(d: &'a [u8], p: Params) -> Self {
        let (good, lazy, nice, chain, slow) = CONFIG[p.level as usize];
        let wsize = 1usize << p.wbits;
        let hbits = p.mem_level as u32 + 7;
        let mut z = Zsim {
            d, good, lazy, nice, chain, slow, filtered: p.filtered,
            wmask: wsize - 1, max_dist: wsize - 262, hmask: (1 << hbits) - 1, hshift: hbits.div_ceil(3),
            head: vec![NIL; 1 << hbits], prev: vec![NIL; wsize], ins_h: 0,
            strstart: 0, match_length: MIN_MATCH - 1, match_start: 0, prev_length: MIN_MATCH - 1, prev_match: 0,
            match_available: false,
            log: Vec::with_capacity(512),
            saved: (0, 0, 0, 0, 0, 0, false),
        };
        z.ins_h = z.byte(0) as u32;
        z.update_hash(z.byte(1));
        z
    }

    #[inline]
    fn byte(&self, i: usize) -> u8 { self.d.get(i).copied().unwrap_or(0) }

    #[inline]
    fn lookahead(&self) -> usize { self.d.len().saturating_sub(self.strstart) }

    #[inline]
    fn update_hash(&mut self, c: u8) { self.ins_h = ((self.ins_h << self.hshift) ^ c as u32) & self.hmask; }

    /// zlib's INSERT_STRING at position `s`: returns the previous head of its hash chain.
    #[inline]
    fn insert(&mut self, s: usize) -> u32 {
        self.update_hash(self.byte(s + MIN_MATCH - 1));
        let h = self.ins_h as usize;
        let head = self.head[h];
        let pi = s & self.wmask;
        self.log.push((1, pi as u32, self.prev[pi]));
        self.log.push((0, h as u32, head));
        self.prev[pi] = head;
        self.head[h] = s as u32;
        head
    }

    fn longest_match(&mut self, mut cur: u32) -> usize {
        let mut chain = self.chain;
        let scan = self.strstart;
        let mut best = self.prev_length;
        let mut nice = self.nice;
        let limit = if scan > self.max_dist { (scan - self.max_dist) as u32 } else { NIL };
        if self.prev_length >= self.good { chain >>= 2; }
        if nice > self.lookahead() { nice = self.lookahead(); }
        // Fast path away from the end of the data: every byte a comparison can touch is in range,
        // so compare without bounds checks: two 2-byte probes (bytes 0-1 and best-1..best, as
        // zlib's UNALIGNED_OK variant), then 8 bytes at a time. Same result as the loop below.
        let d = self.d;
        if scan + MAX_MATCH + 8 <= d.len() {
            // SAFETY: chain entries are positions inserted before `scan` (all < scan), so every
            // read below lies within d[..scan + MAX_MATCH + 8]; `best` stays in 2..=MAX_MATCH.
            unsafe {
                let base = d.as_ptr();
                let sp = base.add(scan);
                let rd16 = |p: *const u8| (p as *const u16).read_unaligned();
                let rd64 = |p: *const u8| (p as *const u64).read_unaligned();
                let s0 = rd16(sp);
                let mut send = rd16(sp.add(best - 1));
                loop {
                    let m = cur as usize;
                    debug_assert!(m < scan);
                    let mp = base.add(m);
                    if rd16(mp.add(best - 1)) == send && rd16(mp) == s0 {
                        let mut len = 2;
                        while len < MAX_MATCH {
                            let x = rd64(mp.add(len)) ^ rd64(sp.add(len));
                            if x != 0 { len += (x.trailing_zeros() / 8) as usize; break; }
                            len += 8;
                        }
                        let len = len.min(MAX_MATCH);
                        if len > best {
                            self.match_start = m;
                            best = len;
                            if len >= nice { break; }
                            send = rd16(sp.add(best - 1));
                        }
                    }
                    cur = *self.prev.get_unchecked(m & self.wmask);
                    if cur <= limit { break; }
                    chain -= 1;
                    if chain == 0 { break; }
                }
            }
            return best.min(self.lookahead());
        }
        loop {
            let m = cur as usize;
            if self.byte(m + best) == self.byte(scan + best) && self.byte(m + best - 1) == self.byte(scan + best - 1)
                && self.byte(m) == self.byte(scan) && self.byte(m + 1) == self.byte(scan + 1) {
                let mut len = 2;
                while len < MAX_MATCH && self.byte(m + len) == self.byte(scan + len) { len += 1; }
                if len > best {
                    self.match_start = m;
                    best = len;
                    if len >= nice { break; }
                }
            }
            cur = self.prev[m & self.wmask];
            if cur <= limit { break; }
            chain -= 1;
            if chain == 0 { break; }
        }
        best.min(self.lookahead())
    }

    /// The next token zlib would emit, advancing the emulator past it. None at the end. Keep it
    /// with [`Zsim::accept`] or undo it with [`Zsim::reject`].
    pub fn next(&mut self) -> Option<Tok> {
        self.log.clear();
        self.saved = (self.ins_h, self.strstart, self.match_length, self.match_start, self.prev_length, self.prev_match, self.match_available);
        if self.slow { self.next_slow() } else { self.next_fast() }
    }

    /// The prediction was right: keep the new state.
    pub fn accept(&mut self) { self.log.clear(); }

    /// The prediction was wrong: restore the state from before [`Zsim::next`], then resynchronise
    /// to the real token `t` (which starts at the next token position).
    pub fn reject(&mut self, t: Tok) {
        while let Some((tab, i, old)) = self.log.pop() {
            if tab == 0 { self.head[i as usize] = old; } else { self.prev[i as usize] = old; }
        }
        let (ih, ss, ml, ms, pl, pm, ma) = self.saved;
        self.ins_h = ih;
        self.strstart = ss;
        self.match_length = ml;
        self.match_start = ms;
        self.prev_length = pl;
        self.prev_match = pm;
        self.match_available = ma;
        let start = self.pos();
        self.force(start, t);
        self.log.clear();
    }

    fn next_fast(&mut self) -> Option<Tok> {
        let look = self.lookahead();
        if look == 0 { return None; }
        let mut head = NIL;
        if look >= MIN_MATCH { head = self.insert(self.strstart); }
        if head != NIL && self.strstart - head as usize <= self.max_dist {
            self.match_length = self.longest_match(head);
        }
        if self.match_length >= MIN_MATCH {
            let t = Tok::Match { len: self.match_length as u16, dist: (self.strstart - self.match_start) as u16, alt: false };
            let look = look - self.match_length;
            if self.match_length <= self.lazy && look >= MIN_MATCH {
                let mut ml = self.match_length - 1;
                while ml > 0 { self.strstart += 1; self.insert(self.strstart); ml -= 1; }
                self.strstart += 1;
            } else {
                self.strstart += self.match_length;
                self.ins_h = self.byte(self.strstart) as u32;
                self.update_hash(self.byte(self.strstart + 1));
            }
            self.match_length = 0;
            Some(t)
        } else {
            let t = Tok::Lit(self.byte(self.strstart));
            self.strstart += 1;
            Some(t)
        }
    }

    fn next_slow(&mut self) -> Option<Tok> {
        loop {
            let look = self.lookahead();
            if look == 0 {
                if self.match_available {
                    self.match_available = false;
                    return Some(Tok::Lit(self.byte(self.strstart - 1)));
                }
                return None;
            }
            let mut head = NIL;
            if look >= MIN_MATCH { head = self.insert(self.strstart); }
            self.prev_length = self.match_length;
            self.prev_match = self.match_start;
            self.match_length = MIN_MATCH - 1;
            if head != NIL && self.prev_length < self.lazy && self.strstart - head as usize <= self.max_dist {
                self.match_length = self.longest_match(head);
                if self.match_length <= 5 && (self.filtered
                    || (self.match_length == MIN_MATCH && self.strstart - self.match_start > TOO_FAR)) {
                    self.match_length = MIN_MATCH - 1;
                }
            }
            if self.prev_length >= MIN_MATCH && self.match_length <= self.prev_length {
                let max_insert = self.strstart + look - MIN_MATCH;
                let t = Tok::Match { len: self.prev_length as u16, dist: (self.strstart - 1 - self.prev_match) as u16, alt: false };
                let mut pl = self.prev_length - 2;
                loop {
                    self.strstart += 1;
                    if self.strstart <= max_insert { self.insert(self.strstart); }
                    pl -= 1;
                    if pl == 0 { break; }
                }
                self.match_available = false;
                self.match_length = MIN_MATCH - 1;
                self.strstart += 1;
                return Some(t);
            } else if self.match_available {
                let t = Tok::Lit(self.byte(self.strstart - 1));
                self.strstart += 1;
                return Some(t);
            } else {
                self.match_available = true;
                self.strstart += 1;
            }
        }
    }

    /// Position of the next token's first byte (accounting for a deferred literal).
    pub fn pos(&self) -> usize { if self.match_available { self.strstart - 1 } else { self.strstart } }

    /// Resynchronise after the real stream emitted `t` (starting at the next token position)
    /// instead of the predicted token: positions it covers are inserted as zlib would, and lazy
    /// state is cleared.
    fn force(&mut self, start: usize, t: Tok) {
        let len = match t { Tok::Lit(_) => 1, Tok::Match { len, .. } => len as usize };
        // rebuild state as if zlib had just finished a token ending at start + len
        let end = start + len;
        // positions in [strstart, end) not yet inserted: insert them (zlib inserts every position
        // it steps over when lookahead allows)
        let mut s = self.strstart;
        while s < end {
            if self.d.len().saturating_sub(s) >= MIN_MATCH { self.insert(s); }
            s += 1;
        }
        self.strstart = end;
        self.match_available = false;
        self.match_length = MIN_MATCH - 1;
        self.prev_length = MIN_MATCH - 1;
        if !self.slow {
            self.ins_h = self.byte(end) as u32;
            self.update_hash(self.byte(end + 1));
        }
    }
}

/// Count how many of `toks` (the first `limit`) the emulator with `p` predicts exactly.
pub fn agreement(d: &[u8], toks: &[Tok], p: Params, limit: usize) -> usize {
    let mut z = Zsim::new(d, p);
    let mut ok = 0;
    let mut pos = 0usize;
    for &t in toks.iter().take(limit) {
        let start = z.pos();
        debug_assert_eq!(start, pos);
        match z.next() {
            Some(pt) if pt == t => { ok += 1; z.accept(); }
            _ => z.reject(t),
        }
        pos += match t { Tok::Lit(_) => 1, Tok::Match { len, .. } => len as usize };
    }
    ok
}
