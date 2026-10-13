//! Compact reconstruction data ("side info") for a deflate stream, given its decompressed data.
//!
//! The token stream is predicted by the zlib emulator ([`super::zsim`]) and each dynamic header
//! by the zlib tree builder ([`super::ztrees`]). Only what they cannot predict is stored:
//! * the emulator parameters;
//! * each block's type, final flag and token count;
//! * headers zlib would not have produced;
//! * mispredicted tokens. A literal correction needs no value: it is the data byte.
//!
//! For zlib-made streams this is a few bytes per block.
use super::zsim::{Params, Zsim};
use super::*;

const VERSION: u8 = 1;
const SEARCH: usize = 4096;

fn put_varint(o: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 { o.push(v as u8 | 0x80); v >>= 7; }
    o.push(v as u8);
}

struct Rd<'a> { b: &'a [u8], o: usize }
impl Rd<'_> {
    fn u8(&mut self) -> Result<u8, String> { let v = *self.b.get(self.o).ok_or("side info truncated")?; self.o += 1; Ok(v) }
    fn u16(&mut self) -> Result<u16, String> { Ok(self.u8()? as u16 | (self.u8()? as u16) << 8) }
    fn varint(&mut self) -> Result<u64, String> {
        let mut v = 0u64;
        for sh in (0..64).step_by(7) {
            let b = self.u8()?;
            v |= ((b & 0x7f) as u64) << sh;
            if b & 0x80 == 0 { return Ok(v); }
        }
        Err("side info: bad varint".into())
    }
}

fn tok_len(t: &Tok) -> usize { match t { Tok::Lit(_) => 1, Tok::Match { len, .. } => *len as usize } }

/// Advance the emulator over a stored block's bytes `[pos, end)` the way zlib's matcher did
/// (its tokens for that region were computed, then discarded in favour of storing).
fn skip_stored(z: &mut Zsim, data: &[u8], pos: usize, end: usize) {
    let mut p = pos;
    while p < end {
        match z.next() {
            Some(t) if p + tok_len(&t) <= end => { p += tok_len(&t); z.accept(); }
            _ => { z.reject(Tok::Lit(data[p])); p += 1; }
        }
    }
}

/// Pick emulator parameters by agreement on the first tokens.
fn choose_params(data: &[u8], toks: &[Tok], wbits: u8) -> Params {
    let mut best = (0usize, Params { level: 6, filtered: false, wbits, mem_level: 8 });
    for level in [6u8, 9, 5, 4, 7, 8, 1, 2, 3] {
        for filtered in [false, true] {
            let p = Params { level, filtered, wbits, mem_level: 8 };
            let a = zsim::agreement(data, toks, p, SEARCH);
            if a > best.0 { best = (a, p); }
            if a == toks.len().min(SEARCH) { return p; }
        }
    }
    best.1
}

/// Encode the side info for `parse` (which decoded to `data`). `wbits` is the zlib window size
/// (15 for raw deflate from gzip/zip).
pub fn encode(data: &[u8], parse: &Parse, wbits: u8) -> Vec<u8> {
    let toks: Vec<Tok> = parse.blocks.iter().flat_map(|b| b.toks.iter().copied()).collect();
    let params = choose_params(data, &toks, wbits);
    let mut o = vec![VERSION];
    o.extend_from_slice(&params.to_byte());
    put_varint(&mut o, parse.blocks.len() as u64);
    let mut z = Zsim::new(data, params);
    let mut corr: Vec<(u64, Tok)> = vec![];
    let (mut ti, mut pos) = (0u64, 0usize);
    for b in &parse.blocks {
        match &b.kind {
            Kind::Stored { pad_bits, pad_val, len } => {
                o.push(b.last as u8);
                o.extend_from_slice(&[*pad_bits, *pad_val]);
                o.extend_from_slice(&len.to_le_bytes());
                skip_stored(&mut z, data, pos, pos + *len as usize);
                pos += *len as usize;
            }
            k => {
                let (bt, pred) = match k {
                    Kind::Fixed => (1u8, false),
                    _ => (2u8, ztrees::zlib_header(&b.toks) == *k),
                };
                o.push(b.last as u8 | bt << 1 | (pred as u8) << 3);
                put_varint(&mut o, b.toks.len() as u64);
                if let (Kind::Dynamic { hlit, hdist, hclen, cl_lens, cl_syms }, false) = (k, pred) {
                    o.extend_from_slice(&[(*hlit - 257) as u8, (*hdist - 1) as u8, (*hclen - 4) as u8]);
                    for &i in CL_ORDER.iter().take(*hclen as usize) { o.push(cl_lens[i]); }
                    put_varint(&mut o, cl_syms.len() as u64);
                    for &(s, x) in cl_syms { o.extend_from_slice(&[s, x]); }
                }
                for t in &b.toks {
                    match z.next() {
                        Some(pt) if pt == *t => z.accept(),
                        _ => { corr.push((ti, *t)); z.reject(*t); }
                    }
                    ti += 1;
                    pos += tok_len(t);
                }
            }
        }
    }
    o.extend_from_slice(&[parse.end_pad.0, parse.end_pad.1]);
    put_varint(&mut o, corr.len() as u64);
    let mut last = 0u64;
    for (i, t) in corr {
        put_varint(&mut o, i - last);
        last = i + 1;
        match t {
            Tok::Lit(_) => o.push(0),
            Tok::Match { len, dist, alt } => {
                o.push(1 + alt as u8);
                o.extend_from_slice(&len.to_le_bytes());
                o.extend_from_slice(&dist.to_le_bytes());
            }
        }
    }
    o
}

/// Rebuild the parse from `data` and its side info (inverse of [`encode`]).
pub fn decode(data: &[u8], side: &[u8]) -> Result<Parse, String> {
    // the record for any encoder is a version of its own
    if side.first() == Some(&front::VERSION) { return front::decode(data, side); }
    let mut r = Rd { b: side, o: 0 };
    if r.u8()? != VERSION { return Err("side info: unknown version".into()); }
    let params = Params::from_bytes([r.u8()?, r.u8()?]).ok_or("side info: bad parameters")?;
    let nb = r.varint()? as usize;
    if nb > data.len() + 1 { return Err("side info: too many blocks".into()); }
    // first pass: block metadata; corrections follow all blocks
    struct Meta { last: bool, bt: u8, pred: bool, ntok: usize, stored: (u8, u8, u16), header: Option<Kind> }
    let mut metas = Vec::with_capacity(nb);
    for _ in 0..nb {
        let f = r.u8()?;
        let (last, bt, pred) = (f & 1 == 1, (f >> 1) & 3, f & 8 != 0);
        if bt == 0 {
            let (pb, pv) = (r.u8()?, r.u8()?);
            metas.push(Meta { last, bt, pred, ntok: 0, stored: (pb, pv, r.u16()?), header: None });
            continue;
        }
        if bt > 2 { return Err("side info: bad block type".into()); }
        let ntok = r.varint()? as usize;
        if ntok > data.len() { return Err("side info: token count exceeds data".into()); }
        let header = if bt == 2 && !pred {
            let (hlit, hdist, hclen) = (r.u8()? as u16 + 257, r.u8()? as u16 + 1, r.u8()? as u16 + 4);
            if hclen > 19 { return Err("side info: bad hclen".into()); }
            let mut cl_lens = [0u8; 19];
            for &i in CL_ORDER.iter().take(hclen as usize) { cl_lens[i] = r.u8()?; }
            let n = r.varint()? as usize;
            if n > 320 { return Err("side info: too many code-length symbols".into()); }
            let mut cl_syms = Vec::with_capacity(n);
            for _ in 0..n { cl_syms.push((r.u8()?, r.u8()?)); }
            Some(Kind::Dynamic { hlit, hdist, hclen, cl_lens, cl_syms })
        } else { None };
        metas.push(Meta { last, bt, pred, ntok, stored: (0, 0, 0), header });
    }
    let end_pad = (r.u8()?, r.u8()?);
    let nc = r.varint()? as usize;
    if nc > data.len() { return Err("side info: too many corrections".into()); }
    let mut corr: Vec<(u64, Option<Tok>)> = Vec::with_capacity(nc);
    let mut idx = 0u64;
    for _ in 0..nc {
        idx += r.varint()?;
        let k = r.u8()?;
        let t = match k {
            0 => None, // literal: value comes from the data
            1 | 2 => { let len = r.u16()?; let dist = r.u16()?; if !(3..=258).contains(&len) || dist == 0 || dist > 32768 { return Err("side info: bad match".into()); } Some(Tok::Match { len, dist, alt: k == 2 }) }
            _ => return Err("side info: bad correction".into()),
        };
        corr.push((idx, t));
        idx += 1;
    }
    if r.o != side.len() { return Err("side info: trailing bytes".into()); }
    // second pass: replay the emulator
    let mut z = Zsim::new(data, params);
    let (mut ti, mut pos, mut ci) = (0u64, 0usize, 0usize);
    let mut blocks = Vec::with_capacity(nb);
    for m in metas {
        if m.bt == 0 {
            let (pb, pv, len) = m.stored;
            if pos + len as usize > data.len() { return Err("side info: stored block past data".into()); }
            skip_stored(&mut z, data, pos, pos + len as usize);
            pos += len as usize;
            blocks.push(Block { last: m.last, kind: Kind::Stored { pad_bits: pb, pad_val: pv, len }, toks: vec![] });
            continue;
        }
        let mut toks = Vec::with_capacity(m.ntok);
        for _ in 0..m.ntok {
            if pos >= data.len() { return Err("side info: tokens run past the data".into()); }
            let t = if ci < corr.len() && corr[ci].0 == ti {
                let t = corr[ci].1.unwrap_or(Tok::Lit(data[pos]));
                ci += 1;
                let _ = z.next();
                z.reject(t);
                t
            } else {
                match z.next() {
                    Some(t) => { z.accept(); t }
                    None => return Err("side info: emulator ran out of data".into()),
                }
            };
            if let Tok::Match { len, dist, .. } = t {
                if dist as usize > pos || pos + len as usize > data.len() { return Err("side info: invalid match".into()); }
            }
            pos += tok_len(&t);
            ti += 1;
            toks.push(t);
        }
        let kind = match m.bt {
            1 => Kind::Fixed,
            _ => if m.pred { ztrees::zlib_header(&toks) } else { m.header.unwrap() },
        };
        blocks.push(Block { last: m.last, kind, toks });
    }
    if pos != data.len() { return Err("side info: parse does not cover the data".into()); }
    Ok(Parse { blocks, end_pad, consumed: 0 })
}
