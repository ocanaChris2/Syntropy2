//! Deflate (RFC 1951) recompression support.
//!
//! [`inflate::inflate`] decodes a deflate stream and records its complete parse: every block
//! header, the exact run-length symbols that encode each dynamic Huffman table, every
//! literal/match token, and the padding bits. [`writer::write`] rebuilds the identical bitstream
//! from that parse. The compressor therefore stores decompressed data plus a (predicted, mostly
//! redundant) description of the parse instead of the compressed bytes, and always proves the
//! rebuild bit-exact before relying on it.
pub mod front;
pub mod inflate;
pub mod scan;
pub mod side;
pub mod writer;
pub mod zsim;
pub mod ztrees;

pub const LEN_BASE: [u16; 29] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
pub const LEN_EXTRA: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
pub const DIST_BASE: [u16; 30] = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
pub const DIST_EXTRA: [u8; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
/// Order in which code-length-code lengths are transmitted.
pub const CL_ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tok {
    Lit(u8),
    /// `alt`: length 258 sent as code 284 + extra 31 (legal for some decoders, rare encoders).
    Match { len: u16, dist: u16, alt: bool },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    /// pad: bits skipped to reach a byte boundary before LEN (normally zero).
    Stored { pad_bits: u8, pad_val: u8, len: u16 },
    Fixed,
    /// The exact transmitted table: header counts, code-length-code lengths in CL_ORDER, and the
    /// run-length symbols (symbol, extra-bits value) that spell the literal/length + distance code
    /// lengths.
    Dynamic { hlit: u16, hdist: u16, hclen: u16, cl_lens: [u8; 19], cl_syms: Vec<(u8, u8)> },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    pub last: bool,
    pub kind: Kind,
    /// Tokens of a Huffman block, ending implicitly with end-of-block (empty for stored blocks).
    pub toks: Vec<Tok>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Parse {
    pub blocks: Vec<Block>,
    /// Bits after the final block up to the byte boundary (count, value).
    pub end_pad: (u8, u8),
    /// Bytes of the input the stream occupied.
    pub consumed: usize,
}

/// Canonical Huffman codes (bit-reversed for LSB-first writing) from code lengths.
pub fn canonical_codes(lens: &[u8]) -> Vec<(u16, u8)> {
    let mut bl_count = [0u16; 16];
    for &l in lens { if l > 0 { bl_count[l as usize] += 1; } }
    let mut next = [0u16; 16];
    let mut code = 0u16;
    for bits in 1..16 {
        code = (code + bl_count[bits - 1]) << 1;
        next[bits] = code;
    }
    lens.iter().map(|&l| {
        if l == 0 { return (0, 0); }
        let c = next[l as usize];
        next[l as usize] += 1;
        // reverse the l-bit code
        let mut r = 0u16;
        for i in 0..l { r |= ((c >> i) & 1) << (l - 1 - i); }
        (r, l)
    }).collect()
}

/// Decode the run-length symbols of a dynamic block into the literal/length and distance code
/// lengths (hlit + hdist entries). Errors on malformed runs.
pub fn expand_cl_syms(cl_syms: &[(u8, u8)], total: usize) -> Result<Vec<u8>, String> {
    let mut lens = Vec::with_capacity(total);
    for &(s, x) in cl_syms {
        match s {
            0..=15 => lens.push(s),
            16 => {
                let prev = *lens.last().ok_or("deflate: repeat with no previous length")?;
                for _ in 0..3 + x as usize { lens.push(prev); }
            }
            17 => for _ in 0..3 + x as usize { lens.push(0); },
            18 => for _ in 0..11 + x as usize { lens.push(0); },
            _ => return Err("deflate: bad code-length symbol".into()),
        }
        if lens.len() > total { return Err("deflate: code lengths overflow".into()); }
    }
    if lens.len() != total { return Err("deflate: code lengths short".into()); }
    Ok(lens)
}

pub fn fixed_lit_lens() -> Vec<u8> {
    let mut l = vec![8u8; 288];
    for x in l.iter_mut().take(256).skip(144) { *x = 9; }
    for x in l.iter_mut().take(280).skip(256) { *x = 7; }
    l
}

/// A zlib-style deflater assembled from the emulator, tree builder and writer (tests only):
/// raw deflate of `data` at `level`, blocks of 16383 tokens.
#[cfg(test)]
pub fn deflate_like_zlib(data: &[u8], level: u8) -> Vec<u8> {
    let mut z = zsim::Zsim::new(data, zsim::Params { level, filtered: false, wbits: 15, mem_level: 8 });
    let mut toks = vec![];
    while let Some(t) = z.next() { z.accept(); toks.push(t); }
    let n = toks.len().div_ceil(16383).max(1);
    let mut blocks: Vec<Block> = toks.chunks(16383).enumerate()
        .map(|(i, c)| Block { last: i + 1 == n, kind: ztrees::zlib_header(c), toks: c.to_vec() }).collect();
    if blocks.is_empty() { blocks.push(Block { last: true, kind: Kind::Fixed, toks: vec![] }); }
    (0..8).find_map(|pb| writer::write(&Parse { blocks: blocks.clone(), end_pad: (pb, 0), consumed: 0 }, data).ok()).unwrap()
}

#[cfg(test)]
mod tests;
