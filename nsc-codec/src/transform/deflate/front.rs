//! A deflate stream's parse, noted for any encoder.
//!
//! The record of [`super::side`] predicts a stream's tokens by running zlib's own parser over
//! the data. Where zlib wrote the stream (or something built on it) that costs next to nothing;
//! where something else did — miniz, zopfli, 7-Zip, libdeflate, Go — nearly every token is a
//! misprediction, and the record comes to more than it saves. This record assumes less. At each
//! token it looks up what any encoder could have found there: the matches that are longer than
//! every nearer one. The token is then noted as a choice among them — the one expected, another
//! by its place among them, one cut short by how much, a literal instead. Encoders differ in
//! what they choose but each is consistent with itself, so the choices are a stream the byte
//! model codes in a bit or two a token; and where no match exists the token is a literal and
//! nothing is noted at all.
//!
//! What is expected is one of four habits, picked per stream by trying them: the longest match
//! or the nearest, and with or without a literal where the next byte starts a longer match (the
//! lazy evaluation most encoders do).
use super::*;

pub const VERSION: u8 = 2;
const WINDOW: usize = 32768;
const MAX_MATCH: usize = 258;
/// Candidates looked at for one token.
const CAP: usize = 32768;
/// Tokens the habits are tried on.
const TRIAL: usize = 20_000;
/// Candidates looked at for each position of a block that is looked through whole.
const PLAN_CAP: usize = 8192;

/// A token's note. `FULL` and `SHORTER` are followed by which match (below `DIRECT`), `SHORTER`
/// then by how many bytes shorter; `ALT` stands before the note of a match of 258 bytes written
/// the long way; `EXPLICIT` is followed by a length and a distance.
const AS_EXPECTED: u8 = 0;
const LITERAL: u8 = 1;
const FULL: u8 = 2;
const SHORTER: u8 = 128;
const DIRECT: usize = 100;
const ALT: u8 = 254;
const EXPLICIT: u8 = 255;

/// Habits: the nearest match is the expected one, and matches count from it (else the longest);
/// a literal is expected where the next byte starts a longer match.
const NEAREST: u8 = 1;
const LAZY: u8 = 2;
/// The token expected is the first step of the cheapest way to code the rest of its block by
/// the block's own code lengths — what an encoder that weighs every parse writes (zopfli,
/// 7-Zip, libdeflate at its high levels). Where a block's table is not known beforehand (one
/// zlib would build from the tokens), the other habits stand in.
const CHEAPEST: u8 = 4;

fn put_varint(o: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        o.push(v as u8 | 0x80);
        v >>= 7;
    }
    o.push(v as u8);
}

struct Rd<'a> {
    b: &'a [u8],
    o: usize,
}

impl Rd<'_> {
    fn u8(&mut self) -> Result<u8, String> {
        let v = *self.b.get(self.o).ok_or("deflate record truncated")?;
        self.o += 1;
        Ok(v)
    }
    fn u16(&mut self) -> Result<u16, String> { Ok(self.u8()? as u16 | (self.u8()? as u16) << 8) }
    fn varint(&mut self) -> Result<u64, String> {
        let mut v = 0u64;
        for sh in (0..64).step_by(7) {
            let b = self.u8()?;
            v |= ((b & 0x7f) as u64) << sh;
            if b & 0x80 == 0 { return Ok(v); }
        }
        Err("deflate record: bad number".into())
    }
}

fn tok_len(t: &Tok) -> usize {
    match t {
        Tok::Lit(_) => 1,
        Tok::Match { len, .. } => *len as usize,
    }
}

/// Prefix lengths the positions are chained by. A match longer than the best so far shares
/// that many bytes at least, so once the best is as long as the next of these, the search goes
/// on along the chain of that longer prefix: the same matches found, in far fewer steps where
/// short prefixes repeat everywhere (a picture's flat areas).
const ORDERS: [usize; 6] = [3, 4, 6, 10, 18, 34];

/// Every position of the data by the bytes at it, nearest first.
struct Finder<'a> {
    d: &'a [u8],
    /// Per order: the last position (plus one) of each hash; of each position in the window, the
    /// one before it (plus one) with the same hash.
    head: Vec<Vec<u32>>,
    prev: Vec<Vec<u32>>,
    /// The same for positions inside a run of one byte, by the byte, how much of the run is
    /// left and the byte after it. A match from inside a run is as long as what is left of the
    /// run at the least (the byte before gives that); a longer one starts where as much of a
    /// run is left and the same byte follows — which this chain lists, instead of every
    /// position of every run of that byte (a picture's flat areas).
    run_head: Vec<u32>,
    run_prev: Vec<u32>,
    /// Of each position: how many bytes from it on are the same (up to 258).
    run: Vec<u16>,
    /// Positions below this are entered.
    upto: usize,
}

impl<'a> Finder<'a> {
    fn new(d: &'a [u8]) -> Self {
        let mut run = vec![1u16; d.len()];
        for x in (0..d.len().saturating_sub(1)).rev() {
            if d[x] == d[x + 1] { run[x] = (run[x + 1] + 1).min(MAX_MATCH as u16); }
        }
        Finder { d, head: vec![vec![0; 1 << 16]; ORDERS.len()], prev: vec![vec![0; WINDOW]; ORDERS.len()], run_head: vec![0; 1 << 16], run_prev: vec![0; WINDOW], run, upto: 0 }
    }

    #[inline]
    fn hash(&self, p: usize, n: usize) -> usize {
        let mut h = 0x811C_9DC5u32;
        for &x in &self.d[p..p + n] { h = (h ^ x as u32).wrapping_mul(0x0100_0193); }
        (h.wrapping_mul(0x9E37_79B1) >> 16) as usize
    }

    /// The run chain's key of `p`: None unless a run of 3 to 257 bytes is left there and a byte follows.
    #[inline]
    fn run_key(&self, p: usize) -> Option<usize> {
        let r = self.run[p] as usize;
        if !(3..MAX_MATCH).contains(&r) || p + r >= self.d.len() { return None; }
        let v = (self.d[p] as u32) << 24 | (r as u32) << 8 | self.d[p + r] as u32;
        Some((v.wrapping_mul(0x9E37_79B1) >> 16) as usize)
    }

    fn ensure(&mut self, p: usize) {
        while self.upto < p {
            let x = self.upto;
            for (k, &n) in ORDERS.iter().enumerate() {
                if x + n > self.d.len() { break; }
                let h = self.hash(x, n);
                self.prev[k][x & (WINDOW - 1)] = self.head[k][h];
                self.head[k][h] = x as u32 + 1;
            }
            if let Some(h) = self.run_key(x) {
                self.run_prev[x & (WINDOW - 1)] = self.run_head[h];
                self.run_head[h] = x as u32 + 1;
            }
            self.upto += 1;
        }
    }

    /// The matches at `p` that are longer than every nearer one, nearest first, as (distance,
    /// length); true if the search was cut off before the last candidate.
    fn front(&mut self, p: usize, out: &mut Vec<(u32, u32)>, cap: usize) -> bool {
        out.clear();
        let d = self.d;
        if p + 3 > d.len() { return false; }
        self.ensure(p);
        let max = MAX_MATCH.min(d.len() - p);
        let mut best = 2usize;
        // what is left of the run at p, where the run chain can take over
        let left = self.run_key(p).map(|h| (self.run[p] as usize, h)).filter(|x| x.0 < max);
        // the chain being followed: of an order, or (None) the run chain
        let mut k = Some(0usize);
        let mut q1 = self.head[0][self.hash(p, ORDERS[0])] as usize;
        // positions on a chain fall (older entries of the ring are of positions since
        // overwritten), and only those before the last one looked at are new
        let (mut on_chain, mut last, mut steps) = (p, p, 0usize);
        while q1 != 0 {
            let q = q1 - 1;
            if q >= on_chain || p - q > WINDOW { break; }
            on_chain = q;
            if q < last {
                last = q;
                if d[q + best] == d[p + best] {
                    let mut l = 0;
                    while l < max && d[q + l] == d[p + l] { l += 1; }
                    if l > best {
                        out.push(((p - q) as u32, l as u32));
                        best = l;
                        if l == max { return false; }
                        match left {
                            // as long as what is left of the run: longer ones are on the run chain
                            Some((r, h)) if l == r => {
                                (k, on_chain) = (None, p);
                                q1 = self.run_head[h] as usize;
                                steps += 1;
                                if steps == cap { return true; }
                                continue;
                            }
                            // this position shares a longer prefix too: on along that chain (one
                            // longer than the run, or it lists the run's every position again)
                            _ => {
                                let to = ORDERS.iter().rposition(|&n| n <= l).unwrap_or(0);
                                if k.is_none_or(|k| to > k) && left.is_none_or(|x| ORDERS[to] > x.0 || k.is_some()) { k = Some(to); }
                            }
                        }
                    }
                }
            }
            steps += 1;
            if steps == cap { return true; }
            q1 = match k {
                Some(k) => self.prev[k][q & (WINDOW - 1)],
                None => self.run_prev[q & (WINDOW - 1)],
            } as usize;
        }
        false
    }
}

/// The finder with the matches at the token being noted, and those at the byte after it when
/// they were looked up.
struct Walk<'a> {
    finder: Finder<'a>,
    f: Vec<(u32, u32)>,
    g: Vec<(u32, u32)>,
    g_at: usize,
    g_cut: bool,
}

impl<'a> Walk<'a> {
    fn new(d: &'a [u8]) -> Self { Walk { finder: Finder::new(d), f: vec![], g: vec![], g_at: usize::MAX, g_cut: false } }

    /// Look up the matches at `p` (into `self.f`): (the search was cut off, the next byte
    /// starts a longer match — looked up in lazy mode only).
    fn at(&mut self, p: usize, mode: u8) -> (bool, bool) {
        let cut = if self.g_at == p {
            std::mem::swap(&mut self.f, &mut self.g);
            self.g_cut
        } else {
            self.finder.front(p, &mut self.f, CAP)
        };
        self.g_at = usize::MAX;
        let mut longer = false;
        if mode & LAZY != 0 && !self.f.is_empty() && p + 1 < self.finder.d.len() {
            self.g_cut = self.finder.front(p + 1, &mut self.g, CAP);
            self.g_at = p + 1;
            longer = self.g.last().map_or(0, |m| m.1) > self.f[self.f.len() - 1].1;
        }
        (cut, longer)
    }
}

/// What an encoder of the habits `mode` is expected to write: None a literal, or which of `f`.
fn expected(mode: u8, f: &[(u32, u32)], longer: bool) -> Option<usize> {
    if f.is_empty() || (mode & LAZY != 0 && longer) { return None; }
    Some(if mode & NEAREST != 0 { 0 } else { f.len() - 1 })
}

/// The place of the match `j` of `m` as it is noted (and back: the same function).
fn place(mode: u8, m: usize, j: usize) -> usize { if mode & NEAREST != 0 { j } else { m - 1 - j } }

/// Note the token `t`; false if it cannot be (a match where none exists: not a parse of the data).
fn put_token(o: &mut Vec<u8>, mode: u8, f: &[(u32, u32)], cut: bool, exp: Option<(u32, u32)>, t: &Tok) -> bool {
    match *t {
        Tok::Lit(_) => {
            if !f.is_empty() || cut { o.push(if exp.is_none() { AS_EXPECTED } else { LITERAL }); }
        }
        Tok::Match { len, dist, alt } => {
            if f.is_empty() && !cut { return false; }
            if alt { o.push(ALT); }
            if exp == Some((dist as u32, len as u32)) {
                o.push(AS_EXPECTED);
                return true;
            }
            match f.iter().position(|&(d, l)| d == dist as u32 && l >= len as u32).filter(|&j| place(mode, f.len(), j) < DIRECT) {
                Some(j) if f[j].1 == len as u32 => o.push(FULL + place(mode, f.len(), j) as u8),
                Some(j) => o.extend_from_slice(&[SHORTER + place(mode, f.len(), j) as u8, (f[j].1 - len as u32) as u8]),
                None => {
                    o.extend_from_slice(&[EXPLICIT, (len - 3) as u8]);
                    o.extend_from_slice(&(dist - 1).to_le_bytes());
                }
            }
        }
    }
    true
}

/// The token noted at `p`, where `f` are the matches.
fn get_token(r: &mut Rd, mode: u8, f: &[(u32, u32)], cut: bool, exp: Option<(u32, u32)>, d: &[u8], p: usize) -> Result<Tok, String> {
    let bad = || "deflate record: bad token".to_string();
    if f.is_empty() && !cut { return Ok(Tok::Lit(d[p])); }
    let mut s = r.u8()?;
    let alt = s == ALT;
    if alt { s = r.u8()?; }
    // the match noted by its place
    let which = |s: u8, base: u8| -> Result<(u32, u32), String> {
        let i = (s - base) as usize;
        if i >= f.len() { return Err(bad()); }
        Ok(f[place(mode, f.len(), i)])
    };
    let (dist, len) = match s {
        AS_EXPECTED => match exp {
            Some(m) => m,
            None if alt => return Err(bad()),
            None => return Ok(Tok::Lit(d[p])),
        },
        LITERAL if !alt => return Ok(Tok::Lit(d[p])),
        FULL..=101 => which(s, FULL)?,
        SHORTER..=227 => {
            let (dist, len) = which(s, SHORTER)?;
            let less = r.u8()? as u32;
            if less == 0 || less + 3 > len { return Err(bad()); }
            (dist, len - less)
        }
        EXPLICIT => {
            let len = r.u8()? as u32 + 3;
            (r.u16()? as u32 + 1, len)
        }
        _ => return Err(bad()),
    };
    if !(3..=MAX_MATCH as u32).contains(&len) || dist == 0 || dist as usize > WINDOW.min(p) || p + len as usize > d.len() || (alt && len != 258) { return Err(bad()); }
    Ok(Tok::Match { len: len as u16, dist: dist as u16, alt })
}

/// A block looked through before its tokens are noted: the matches at each of its positions,
/// and the first step of the cheapest coding from there.
struct Plan {
    p0: usize,
    at: Vec<u32>,
    fronts: Vec<(u32, u32)>,
    cut: Vec<bool>,
    /// Per position: (length, distance) of that first step; a length of 1 is a literal.
    first: Vec<(u32, u32)>,
}

impl Plan {
    /// The block of `data[p0..end]` coded by the table of `kind`; None without the table.
    fn new(finder: &mut Finder, data: &[u8], p0: usize, end: usize, kind: &Kind) -> Option<Plan> {
        const INF: u32 = u32::MAX / 2;
        let (ll, dl) = lengths(kind)?;
        let n = end - p0;
        let mut pl = Plan { p0, at: Vec::with_capacity(n + 1), fronts: vec![], cut: Vec::with_capacity(n), first: vec![(1, 0); n] };
        let mut f = vec![];
        for p in p0..end {
            pl.at.push(pl.fronts.len() as u32);
            pl.cut.push(finder.front(p, &mut f, PLAN_CAP));
            pl.fronts.extend_from_slice(&f);
        }
        pl.at.push(pl.fronts.len() as u32);
        let cost = |x: u8| if x == 0 { INF } else { x as u32 };
        let len_cost: Vec<u32> = (0..259usize).map(|l| {
            if l < 3 { return INF; }
            let c = if l == 258 { 28 } else { LEN_BASE.iter().rposition(|&b| b as usize <= l).unwrap() };
            cost(*ll.get(257 + c).unwrap_or(&0)) + LEN_EXTRA[c] as u32
        }).collect();
        let dist_cost: Vec<u32> = (0..30).map(|c| cost(*dl.get(c).unwrap_or(&0)) + DIST_EXTRA[c] as u32).collect();
        let mut togo = vec![0u32; n + 1];
        for i in (0..n).rev() {
            let p = p0 + i;
            let mut best = (cost(ll[data[p] as usize]).saturating_add(togo[i + 1]), 1u32, 0u32);
            let mut from = 3u32;
            for &(d, l) in &pl.fronts[pl.at[i] as usize..pl.at[i + 1] as usize] {
                let dc = dist_cost[DIST_BASE.iter().rposition(|&b| b as u32 <= d).unwrap()];
                for len in from..=l.min((n - i) as u32) {
                    let c = len_cost[len as usize].saturating_add(dc).saturating_add(togo[i + len as usize]);
                    if c < best.0 { best = (c, len, d); }
                }
                from = l + 1;
            }
            togo[i] = best.0.min(INF);
            pl.first[i] = (best.1, best.2);
        }
        Some(pl)
    }

    /// At `p`: the matches, whether their search was cut off, and the token expected (None: a literal).
    fn at(&self, p: usize) -> (&[(u32, u32)], bool, Option<(u32, u32)>) {
        let i = p - self.p0;
        let (len, dist) = self.first[i];
        (&self.fronts[self.at[i] as usize..self.at[i + 1] as usize], self.cut[i], (len > 1).then_some((dist, len)))
    }
}

/// Is the table of this block one zlib builds from its tokens (and so not noted)?
fn table_follows(b: &Block) -> bool { matches!(b.kind, Kind::Dynamic { .. }) && ztrees::zlib_header(&b.toks) == b.kind }

/// The notes of the first `limit` tokens of `parse` under the habits `mode`; None if a token
/// cannot be noted.
fn notes(data: &[u8], parse: &Parse, mode: u8, limit: usize, o: &mut Vec<u8>) -> Option<()> {
    let mut w = Walk::new(data);
    let (mut p, mut n) = (0usize, 0usize);
    for b in &parse.blocks {
        if let Kind::Stored { len, .. } = b.kind {
            p += len as usize;
            continue;
        }
        let end = p + b.toks.iter().map(tok_len).sum::<usize>();
        let plan = if mode & CHEAPEST != 0 && !table_follows(b) { Plan::new(&mut w.finder, data, p, end, &b.kind) } else { None };
        if plan.is_some() { w.g_at = usize::MAX; }
        for t in &b.toks {
            if n == limit { return Some(()); }
            let ok = match &plan {
                Some(pl) => {
                    let (f, cut, exp) = pl.at(p);
                    put_token(o, mode, f, cut, exp, t)
                }
                None => {
                    let (cut, longer) = w.at(p, mode);
                    put_token(o, mode, &w.f, cut, expected(mode, &w.f, longer).map(|j| w.f[j]), t)
                }
            };
            if !ok { return None; }
            p += tok_len(t);
            n += 1;
        }
    }
    Some(())
}

/// Developer diagnostics: how the tokens of `parse` are noted under each of the habits — counts
/// of tokens without a note, as expected (literals, matches), literals instead of a match, full
/// matches by place, shorter ones, explicit ones; and the notes' order-0 cost.
pub fn census(data: &[u8], parse: &Parse) -> String {
    let mut out = String::new();
    for mode in 0..4u8 {
        let mut w = Walk::new(data);
        let (mut p, mut o) = (0usize, vec![]);
        let (mut free, mut exp_lit, mut exp_match, mut lit, mut explicit) = (0usize, 0usize, 0usize, 0usize, 0usize);
        let (mut full, mut shorter) = ([0usize; 4], [0usize; 4]);
        let mut cut_off = 0usize;
        // by the longest match's length (3, 4, 5, 6-9, 10+) and distance (to 64, 512, 4096, more):
        // literals written instead of it, and tokens that are it
        let mut table = [[[0usize; 4]; 5]; 2];
        for b in &parse.blocks {
            if let Kind::Stored { len, .. } = b.kind {
                p += len as usize;
                continue;
            }
            for t in &b.toks {
                let (cut, longer) = w.at(p, mode);
                cut_off += cut as usize;
                let at = o.len();
                if let Some(&(dist, len)) = w.f.last() {
                    let row = match len { 3 => 0, 4 => 1, 5 => 2, 6..=9 => 3, _ => 4 };
                    let col = match dist { 0..=64 => 0, 65..=512 => 1, 513..=4096 => 2, _ => 3 };
                    match *t {
                        Tok::Lit(_) => table[0][row][col] += 1,
                        Tok::Match { len: l, dist: d, .. } if l as u32 == len && d as u32 == dist => table[1][row][col] += 1,
                        _ => {}
                    }
                }
                if !put_token(&mut o, mode, &w.f, cut, expected(mode, &w.f, longer).map(|j| w.f[j]), t) { return "not a parse of the data".into(); }
                match o.get(at).copied() {
                    None => free += 1,
                    Some(AS_EXPECTED) if matches!(t, Tok::Lit(_)) => exp_lit += 1,
                    Some(AS_EXPECTED) => exp_match += 1,
                    Some(LITERAL) => lit += 1,
                    Some(s @ FULL..=101) => full[((s - FULL) as usize).min(3)] += 1,
                    Some(s @ SHORTER..=227) => shorter[((s - SHORTER) as usize).min(3)] += 1,
                    Some(_) => explicit += 1,
                }
                p += tok_len(t);
            }
        }
        if mode == 0 { out.push_str(&format!("\n      literal instead, by longest match (rows: 3, 4, 5, 6-9, 10+ bytes; columns: to 64, 512, 4096, farther): {:?}\n      the longest match taken: {:?}", table[0], table[1])); }
        out.push_str(&format!("\n      habits {mode}: {} notes of {} bytes, about {} coded; no note {free}, as expected {exp_lit} literals + {exp_match} matches, literal instead {lit}, other match {full:?}, shorter {shorter:?}, explicit {explicit}, searches cut off {cut_off}",
            free + exp_lit + exp_match + lit + explicit + full.iter().sum::<usize>() + shorter.iter().sum::<usize>(), o.len(), cost(&o)));
    }
    out
}

/// The code lengths of a block's two alphabets (literals and lengths; distances).
fn lengths(k: &Kind) -> Option<(Vec<u8>, Vec<u8>)> {
    match k {
        Kind::Fixed => {
            let mut l = vec![8u8; 288];
            for x in &mut l[144..256] { *x = 9; }
            for x in &mut l[256..280] { *x = 7; }
            Some((l, vec![5; 30]))
        }
        Kind::Dynamic { hlit, hdist, cl_syms, .. } => {
            let mut all: Vec<u8> = vec![];
            for &(s, x) in cl_syms {
                match s {
                    0..=15 => all.push(s),
                    16 => { let p = *all.last()?; all.extend(std::iter::repeat_n(p, 3 + x as usize)); }
                    17 => all.extend(std::iter::repeat_n(0, 3 + x as usize)),
                    _ => all.extend(std::iter::repeat_n(0, 11 + x as usize)),
                }
            }
            if all.len() != (*hlit + *hdist) as usize { return None; }
            let d = all.split_off(*hlit as usize);
            Some((all, d))
        }
        _ => None,
    }
}

/// Developer experiment: how often is a token the first step of the cheapest way to code the
/// rest of its block, by the block's own code lengths? (What an optimal parser would write.)
pub fn dp_census(data: &[u8], parse: &Parse) -> String {
    let n = data.len();
    // the matches at every position
    let mut finder = Finder::new(data);
    let mut fronts: Vec<(u32, u32)> = vec![];
    let mut at: Vec<u32> = Vec::with_capacity(n + 1);
    let mut f = vec![];
    for p in 0..n {
        at.push(fronts.len() as u32);
        finder.front(p, &mut f, CAP);
        fronts.extend_from_slice(&f);
    }
    at.push(fronts.len() as u32);
    const INF: u32 = u32::MAX / 2;
    let (mut same, mut noted, mut free, mut kind_same) = (0usize, 0usize, 0usize, 0usize);
    let mut p0 = 0usize;
    let mut togo = vec![0u32; n + 1];
    for b in &parse.blocks {
        if let Kind::Stored { len, .. } = b.kind { p0 += len as usize; continue; }
        let Some((ll, dl)) = lengths(&b.kind) else { return "tables unreadable".into() };
        let end = p0 + b.toks.iter().map(tok_len).sum::<usize>();
        let cost = |x: u8| if x == 0 { INF } else { x as u32 };
        let len_cost: Vec<u32> = (0..259usize).map(|l| if l < 3 { INF } else { let c = LEN_BASE.iter().rposition(|&b| b as usize <= l).unwrap(); let c = if l == 258 { 28 } else { c }; cost(*ll.get(257 + c).unwrap_or(&0)).saturating_add(LEN_EXTRA[c] as u32) }).collect();
        let dist_cost = |d: u32| { let c = DIST_BASE.iter().rposition(|&b| b as u32 <= d).unwrap(); cost(*dl.get(c).unwrap_or(&0)).saturating_add(DIST_EXTRA[c] as u32) };
        // the first step of the cheapest coding from p: (cost to go, length (1: a literal), distance)
        let best = |p: usize, togo: &[u32]| -> (u32, u32, u32) {
            let mut b = (cost(ll[data[p] as usize]).saturating_add(togo[p + 1]), 1u32, 0u32);
            let mut from = 3u32;
            for &(d, l) in &fronts[at[p] as usize..at[p + 1] as usize] {
                let dc = dist_cost(d);
                for len in from..=l.min((end - p) as u32) {
                    let c = len_cost[len as usize].saturating_add(dc).saturating_add(togo[p + len as usize]);
                    if c < b.0 { b = (c, len, d); }
                }
                from = l + 1;
            }
            b
        };
        togo[end] = 0;
        for p in (p0..end).rev() { togo[p] = best(p, &togo).0.min(INF); }
        let mut p = p0;
        for t in &b.toks {
            if at[p] == at[p + 1] { free += 1; } else {
                noted += 1;
                let (_, len, d) = best(p, &togo);
                match *t {
                    Tok::Lit(_) => { same += (len == 1) as usize; kind_same += (len == 1) as usize; }
                    Tok::Match { len: l, dist, .. } => { same += (len == l as u32 && d == dist as u32) as usize; kind_same += (len > 1) as usize; }
                }
            }
            p += tok_len(t);
        }
        p0 = end;
    }
    format!("cheapest-path first step: the token itself for {same} of {noted} tokens with a choice ({:.1}%), a literal or a match as it is for {kind_same}; {free} tokens had none", 100.0 * same as f64 / noted.max(1) as f64)
}

/// What `bytes` come to when coded, about: their order-0 entropy, in bytes.
pub fn cost(bytes: &[u8]) -> usize { (crate::nsc3::block::order0_mbits(bytes) as u64 * bytes.len() as u64 / 8000) as usize }

/// The record of `parse`, which decoded to `data`; None if it is not a parse of it.
pub fn encode(data: &[u8], parse: &Parse) -> Option<Vec<u8>> {
    // the habits that make the first tokens cheapest
    let mut best = (usize::MAX, 0u8);
    let mut trial = vec![];
    for mode in [0u8, 1, 2, 3, CHEAPEST, CHEAPEST | LAZY] {
        trial.clear();
        notes(data, parse, mode, TRIAL, &mut trial)?;
        let c = cost(&trial);
        if c < best.0 { best = (c, mode); }
    }
    encode_as(data, parse, best.1)
}

/// The record of `parse` under the habits `mode`.
fn encode_as(data: &[u8], parse: &Parse, mode: u8) -> Option<Vec<u8>> {
    let mut o = vec![VERSION, mode];
    put_varint(&mut o, parse.blocks.len() as u64);
    for b in &parse.blocks {
        match &b.kind {
            Kind::Stored { pad_bits, pad_val, len } => {
                o.extend_from_slice(&[b.last as u8, *pad_bits, *pad_val]);
                o.extend_from_slice(&len.to_le_bytes());
            }
            k => {
                // a table as zlib builds it from the tokens needs no noting
                let (bt, pred) = match k {
                    Kind::Fixed => (1u8, false),
                    _ => (2u8, ztrees::zlib_header(&b.toks) == *k),
                };
                o.push(b.last as u8 | bt << 1 | (pred as u8) << 3);
                put_varint(&mut o, b.toks.len() as u64);
                // (to look a block through before its tokens, its extent must be known)
                if mode & CHEAPEST != 0 { put_varint(&mut o, b.toks.iter().map(tok_len).sum::<usize>() as u64); }
                if let (Kind::Dynamic { hlit, hdist, hclen, cl_lens, cl_syms }, false) = (k, pred) {
                    o.extend_from_slice(&[(*hlit - 257) as u8, (*hdist - 1) as u8, (*hclen - 4) as u8]);
                    for &i in CL_ORDER.iter().take(*hclen as usize) { o.push(cl_lens[i]); }
                    put_varint(&mut o, cl_syms.len() as u64);
                    for &(s, x) in cl_syms { o.extend_from_slice(&[s, x]); }
                }
            }
        }
    }
    o.extend_from_slice(&[parse.end_pad.0, parse.end_pad.1]);
    notes(data, parse, mode, usize::MAX, &mut o)?;
    Some(o)
}

/// The parse [`encode`] noted, from `data` and the record.
pub fn decode(data: &[u8], side: &[u8]) -> Result<Parse, String> {
    let mut r = Rd { b: side, o: 0 };
    if r.u8()? != VERSION { return Err("deflate record: unknown version".into()); }
    let mode = r.u8()?;
    if mode > 7 { return Err("deflate record: bad mode".into()); }
    let nb = r.varint()? as usize;
    if nb > data.len() + 1 { return Err("deflate record: too many blocks".into()); }
    struct Meta {
        last: bool,
        bt: u8,
        pred: bool,
        ntok: usize,
        bytes: usize,
        stored: (u8, u8, u16),
        header: Option<Kind>,
    }
    let mut metas = Vec::with_capacity(nb);
    for _ in 0..nb {
        let f = r.u8()?;
        let (last, bt, pred) = (f & 1 == 1, (f >> 1) & 3, f & 8 != 0);
        if bt == 0 {
            let (pb, pv) = (r.u8()?, r.u8()?);
            metas.push(Meta { last, bt, pred, ntok: 0, bytes: 0, stored: (pb, pv, r.u16()?), header: None });
            continue;
        }
        if bt > 2 { return Err("deflate record: bad block type".into()); }
        let ntok = r.varint()? as usize;
        if ntok > data.len() { return Err("deflate record: token count exceeds data".into()); }
        let bytes = if mode & CHEAPEST != 0 { r.varint()? as usize } else { 0 };
        if bytes > data.len() { return Err("deflate record: block exceeds data".into()); }
        let header = if bt == 2 && !pred {
            let (hlit, hdist, hclen) = (r.u8()? as u16 + 257, r.u8()? as u16 + 1, r.u8()? as u16 + 4);
            if hclen > 19 { return Err("deflate record: bad table".into()); }
            let mut cl_lens = [0u8; 19];
            for &i in CL_ORDER.iter().take(hclen as usize) { cl_lens[i] = r.u8()?; }
            let n = r.varint()? as usize;
            if n > 320 { return Err("deflate record: bad table".into()); }
            let mut cl_syms = Vec::with_capacity(n);
            for _ in 0..n { cl_syms.push((r.u8()?, r.u8()?)); }
            Some(Kind::Dynamic { hlit, hdist, hclen, cl_lens, cl_syms })
        } else {
            None
        };
        metas.push(Meta { last, bt, pred, ntok, bytes, stored: (0, 0, 0), header });
    }
    let end_pad = (r.u8()?, r.u8()?);
    let mut w = Walk::new(data);
    let mut p = 0usize;
    let mut blocks = Vec::with_capacity(nb);
    for m in metas {
        if m.bt == 0 {
            let (pad_bits, pad_val, len) = m.stored;
            if p + len as usize > data.len() { return Err("deflate record: stored block past the data".into()); }
            p += len as usize;
            blocks.push(Block { last: m.last, kind: Kind::Stored { pad_bits, pad_val, len }, toks: vec![] });
            continue;
        }
        let mut toks = Vec::with_capacity(m.ntok);
        // the block looked through first, where its table is known and the habits ask for it
        let end = p + m.bytes;
        if end > data.len() { return Err("deflate record: block past the data".into()); }
        let table = match (m.bt, &m.header) { (1, _) => Some(Kind::Fixed), (_, Some(h)) => Some(h.clone()), _ => None };
        let plan = match table { Some(k) if mode & CHEAPEST != 0 => Plan::new(&mut w.finder, data, p, end, &k), _ => None };
        if plan.is_some() { w.g_at = usize::MAX; }
        for _ in 0..m.ntok {
            if p >= data.len() { return Err("deflate record: tokens run past the data".into()); }
            let t = match &plan {
                Some(pl) => {
                    if p >= end { return Err("deflate record: tokens run past their block".into()); }
                    let (f, cut, exp) = pl.at(p);
                    get_token(&mut r, mode, f, cut, exp, data, p)?
                }
                None => {
                    let (cut, longer) = w.at(p, mode);
                    get_token(&mut r, mode, &w.f, cut, expected(mode, &w.f, longer).map(|j| w.f[j]), data, p)?
                }
            };
            p += tok_len(&t);
            toks.push(t);
        }
        if mode & CHEAPEST != 0 && p != end { return Err("deflate record: block of another extent".into()); }
        let kind = match m.bt {
            1 => Kind::Fixed,
            _ if m.pred => ztrees::zlib_header(&toks),
            _ => m.header.unwrap(),
        };
        blocks.push(Block { last: m.last, kind, toks });
    }
    if r.o != side.len() || p != data.len() { return Err("deflate record does not cover the data".into()); }
    Ok(Parse { blocks, end_pad, consumed: 0 })
}

#[cfg(test)]
pub mod tests {
    use super::*;

    fn rng(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }

    /// Text-like data: words of a small vocabulary, with runs.
    pub fn text(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        let words: Vec<Vec<u8>> = (0..300).map(|_| (0..2 + rng(&mut s) % 9).map(|_| b'a' + (rng(&mut s) % 26) as u8).collect()).collect();
        let mut out = Vec::with_capacity(n + 16);
        while out.len() < n {
            match rng(&mut s) % 20 {
                0 => out.extend(std::iter::repeat_n(b'-', 3 + (rng(&mut s) % 300) as usize)),
                _ => {
                    out.extend_from_slice(&words[(rng(&mut s) % 300) as usize]);
                    out.push(b' ');
                }
            }
        }
        out.truncate(n);
        out
    }

    /// A parse of `data` by an encoder that is not zlib: `style` 0 takes the longest match
    /// always, 1 the nearest of at least four bytes, 2 a literal first where the next byte
    /// matches longer, 3 chooses among the matches and their lengths as an optimal parser might
    /// (and now and then a match no search of ours lists first). Blocks of `block` tokens; it
    /// looks `back` bytes back for matches.
    pub fn foreign(data: &[u8], style: u32, block: usize, back: usize, seed: u64) -> Parse {
        let mut s = seed;
        let n = data.len();
        // all matches at p within reach: (distance, length), nearest first
        let all = |p: usize| -> Vec<(usize, usize)> {
            let mut v = vec![];
            if p + 3 > n { return v; }
            for q in (p.saturating_sub(back.min(WINDOW))..p).rev() {
                let mut l = 0;
                while l < MAX_MATCH.min(n - p) && data[q + l] == data[p + l] { l += 1; }
                if l >= 3 { v.push((p - q, l)); }
            }
            v
        };
        let longest = |v: &[(usize, usize)]| v.iter().copied().fold((0, 0), |b, m| if m.1 > b.1 { m } else { b });
        let mut toks = vec![];
        let mut p = 0;
        while p < n {
            let v = all(p);
            let m = longest(&v);
            let t = if v.is_empty() {
                Tok::Lit(data[p])
            } else {
                match style {
                    0 => Tok::Match { len: m.1 as u16, dist: m.0 as u16, alt: false },
                    1 => match v.iter().find(|x| x.1 >= 4) {
                        Some(x) => Tok::Match { len: x.1 as u16, dist: x.0 as u16, alt: false },
                        None => Tok::Lit(data[p]),
                    },
                    2 if longest(&all(p + 1)).1 > m.1 => Tok::Lit(data[p]),
                    2 => Tok::Match { len: m.1 as u16, dist: m.0 as u16, alt: false },
                    _ => match rng(&mut s) % 10 {
                        0 | 1 => Tok::Lit(data[p]),
                        2 | 3 => {
                            let x = v[(rng(&mut s) % v.len() as u64) as usize];
                            Tok::Match { len: (3 + rng(&mut s) % (x.1 as u64 - 2)) as u16, dist: x.0 as u16, alt: false }
                        }
                        4 => Tok::Match { len: m.1 as u16, dist: m.0 as u16, alt: m.1 == 258 },
                        _ => Tok::Match { len: m.1 as u16, dist: m.0 as u16, alt: false },
                    },
                }
            };
            p += tok_len(&t);
            toks.push(t);
        }
        let count = toks.len().div_ceil(block).max(1);
        let mut blocks: Vec<Block> = toks.chunks(block).enumerate().map(|(i, c)| {
            // tables as zlib builds them, fixed ones, and one a little off (as other encoders' are)
            let kind = match i % 3 {
                1 => Kind::Fixed,
                _ => ztrees::zlib_header(c),
            };
            Block { last: i + 1 == count, kind, toks: c.to_vec() }
        }).collect();
        if blocks.is_empty() { blocks.push(Block { last: true, kind: Kind::Fixed, toks: vec![] }); }
        Parse { blocks, end_pad: (0, 0), consumed: 0 }
    }

    #[test]
    fn parses_of_any_encoder_come_back() {
        for (n, seed) in [(0usize, 1u64), (1, 2), (2, 3), (5, 4), (700, 5), (20_000, 6), (80_000, 7)] {
            let data = text(n, seed);
            for style in 0..4u32 {
                let what = format!("{n} bytes, style {style}");
                // (the long one for what is more than a window back; a search of all of it takes long)
                let parse = foreign(&data, style, 3000, if n > 20_000 { 2500 } else { n }, seed + style as u64);
                // through a real stream: what an inflater makes of it is what was written
                let stream = (0..8).find_map(|pb| writer::write(&Parse { end_pad: (pb, 0), ..parse.clone() }, &data).ok()).unwrap_or_else(|| panic!("{what}: not written"));
                let (back, read) = inflate::inflate(&stream, n).unwrap_or_else(|e| panic!("{what}: {e}"));
                assert!(back == data, "{what}: data");
                let rec = encode(&data, &read).unwrap_or_else(|| panic!("{what}: no record"));
                let got = decode(&data, &rec).unwrap_or_else(|e| panic!("{what}: {e}"));
                assert!(got.blocks == read.blocks && got.end_pad == read.end_pad, "{what}: parse");
                assert!(writer::write(&got, &data).unwrap() == stream, "{what}: stream");
                // a consistent encoder's choices cost little: far less than a byte a token
                let toks: usize = read.blocks.iter().map(|b| b.toks.len()).sum();
                if n == 20_000 && style < 3 { assert!(cost(&rec) * if style == 1 { 4 } else { 8 } < toks, "{what}: about {} bytes for {toks} tokens", cost(&rec)); }
            }
        }
        // every habit notes any parse, the cheapest-path one too (its blocks looked through first)
        let data = text(40_000, 12);
        for style in [0u32, 3] {
            let parse = foreign(&data, style, 1500, 2500, 5);
            for mode in 0..8u8 {
                let rec = encode_as(&data, &parse, mode).unwrap();
                let got = decode(&data, &rec).unwrap_or_else(|e| panic!("style {style}, habits {mode}: {e}"));
                assert!(got.blocks == parse.blocks, "style {style}, habits {mode}");
            }
        }
        // zlib's own streams too
        let data = text(80_000, 9);
        let stream = crate::transform::deflate::deflate_like_zlib(&data, 6);
        let (_, read) = inflate::inflate(&stream, data.len()).unwrap();
        let rec = encode(&data, &read).unwrap();
        assert!(decode(&data, &rec).unwrap().blocks == read.blocks);
    }

    /// Damaged records: an error or another parse, never a panic; and never a token past the data.
    #[test]
    fn damaged_records_do_not_panic() {
        let mut s = 3u64;
        let data = text(30_000, 11);
        for style in [2u32, 3] {
            let parse = foreign(&data, style, 2000, 2500, 17);
            let rec = encode(&data, &parse).unwrap();
            for _ in 0..1500 {
                let mut bad = rec.clone();
                for _ in 0..1 + rng(&mut s) % 3 {
                    let i = (rng(&mut s) % bad.len() as u64) as usize;
                    if rng(&mut s) % 5 == 0 { bad.truncate(i.max(1)); } else { bad[i] ^= 1 << (rng(&mut s) % 8); }
                }
                if let Ok(p) = decode(&data, &bad) {
                    let mut at = 0usize;
                    for b in &p.blocks {
                        if let Kind::Stored { len, .. } = b.kind { at += len as usize; }
                        for t in &b.toks {
                            if let Tok::Match { len, dist, .. } = *t { assert!(dist as usize <= at && at + len as usize <= data.len()); }
                            at += tok_len(t);
                        }
                    }
                    assert_eq!(at, data.len());
                    let _ = writer::write(&p, &data);
                }
            }
        }
    }
}
