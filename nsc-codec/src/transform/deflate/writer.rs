//! Rebuild a deflate bitstream from a recorded [`Parse`] and the decompressed data.
use super::*;

struct BitW {
    out: Vec<u8>,
    buf: u64,
    n: u32,
}

impl BitW {
    fn new() -> Self { BitW { out: Vec::new(), buf: 0, n: 0 } }

    #[inline]
    fn put(&mut self, v: u32, k: u32) {
        if k == 0 { return; }
        self.buf |= (v as u64 & ((1u64 << k) - 1)) << self.n;
        self.n += k;
        while self.n >= 8 {
            self.out.push(self.buf as u8);
            self.buf >>= 8;
            self.n -= 8;
        }
    }

    fn flush_partial(&mut self) {
        if self.n > 0 {
            self.out.push(self.buf as u8);
            self.buf = 0;
            self.n = 0;
        }
    }
}

fn len_code(len: u16, alt: bool) -> (usize, u16) {
    if alt { return (27, 31); }
    if len == 258 { return (28, 0); }
    let mut i = 27;
    while LEN_BASE[i] > len { i -= 1; }
    (i, len - LEN_BASE[i])
}

fn dist_code(d: u16) -> (usize, u16) {
    let mut i = 29;
    while DIST_BASE[i] > d { i -= 1; }
    (i, d - DIST_BASE[i])
}

/// Write the stream described by `p`; `data` is the decompressed output (stored blocks copy from
/// it). Errors if the parse is inconsistent (e.g. a symbol without a code).
pub fn write(p: &Parse, data: &[u8]) -> Result<Vec<u8>, String> {
    let mut w = BitW::new();
    let mut pos = 0usize; // position in `data`
    for blk in &p.blocks {
        w.put(blk.last as u32, 1);
        match &blk.kind {
            Kind::Stored { pad_bits, pad_val, len } => {
                w.put(0, 2);
                // the pad must be exactly what brings the writer to a byte boundary
                if (w.n + *pad_bits as u32) % 8 != 0 { return Err("deflate: stored pad mismatch".into()); }
                w.put(*pad_val as u32, *pad_bits as u32);
                w.put(*len as u32, 16);
                w.put(!*len as u32 & 0xffff, 16);
                let end = pos + *len as usize;
                if end > data.len() { return Err("deflate: stored block past data".into()); }
                w.out.extend_from_slice(&data[pos..end]);
                pos = end;
            }
            k => {
                let (lit_lens, dist_lens): (Vec<u8>, Vec<u8>) = match k {
                    Kind::Fixed => { w.put(1, 2); (fixed_lit_lens(), vec![5u8; 30]) }
                    Kind::Dynamic { hlit, hdist, hclen, cl_lens, cl_syms } => {
                        w.put(2, 2);
                        w.put((*hlit - 257) as u32, 5);
                        w.put((*hdist - 1) as u32, 5);
                        w.put((*hclen - 4) as u32, 4);
                        for &i in CL_ORDER.iter().take(*hclen as usize) { w.put(cl_lens[i] as u32, 3); }
                        let clc = canonical_codes(cl_lens);
                        for &(s, x) in cl_syms {
                            let (c, l) = clc[s as usize];
                            if l == 0 { return Err("deflate: code-length symbol without a code".into()); }
                            w.put(c as u32, l as u32);
                            match s { 16 => w.put(x as u32, 2), 17 => w.put(x as u32, 3), 18 => w.put(x as u32, 7), _ => {} }
                        }
                        let lens = expand_cl_syms(cl_syms, (*hlit + *hdist) as usize)?;
                        (lens[..*hlit as usize].to_vec(), lens[*hlit as usize..].to_vec())
                    }
                    Kind::Stored { .. } => unreachable!(),
                };
                let lc = canonical_codes(&lit_lens);
                let dc = canonical_codes(&dist_lens);
                for t in &blk.toks {
                    match *t {
                        Tok::Lit(b) => {
                            let (c, l) = lc[b as usize];
                            if l == 0 { return Err("deflate: literal without a code".into()); }
                            w.put(c as u32, l as u32);
                            pos += 1;
                        }
                        Tok::Match { len, dist, alt } => {
                            let (li, lx) = len_code(len, alt);
                            let (c, l) = *lc.get(257 + li).ok_or("deflate: length code out of table")?;
                            if l == 0 { return Err("deflate: length without a code".into()); }
                            w.put(c as u32, l as u32);
                            w.put(lx as u32, LEN_EXTRA[li] as u32);
                            let (di, dx) = dist_code(dist);
                            let (c, l) = *dc.get(di).ok_or("deflate: distance code out of table")?;
                            if l == 0 { return Err("deflate: distance without a code".into()); }
                            w.put(c as u32, l as u32);
                            w.put(dx as u32, DIST_EXTRA[di] as u32);
                            pos += len as usize;
                        }
                    }
                }
                let (c, l) = lc[256];
                w.put(c as u32, l as u32);
            }
        }
    }
    // final padding bits to the byte boundary
    let (pb, pv) = p.end_pad;
    if (w.n + pb as u32) % 8 != 0 && !(w.n == 0 && pb == 0) { return Err("deflate: end pad mismatch".into()); }
    w.put(pv as u32, pb as u32);
    w.flush_partial();
    Ok(w.out)
}
