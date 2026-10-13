//! zlib's dynamic Huffman header construction (trees.c: build_tree, gen_bitlen, scan/send_tree,
//! build_bl_tree), used to predict a dynamic block's transmitted header from its tokens.
//! Following zlib, the literal and distance length runs are encoded separately (runs never
//! cross the boundary).
use super::{Kind, Tok, CL_ORDER, DIST_BASE, LEN_BASE};

const L_CODES: usize = 286;
const D_CODES: usize = 30;
const BL_CODES: usize = 19;
const HEAP_SIZE: usize = 2 * L_CODES + 1;

struct Builder {
    heap: [usize; HEAP_SIZE],
    heap_len: usize,
    heap_max: usize,
    depth: [u8; HEAP_SIZE],
}

impl Builder {
    fn smaller(freq: &[u32], depth: &[u8], n: usize, m: usize) -> bool {
        freq[n] < freq[m] || (freq[n] == freq[m] && depth[n] <= depth[m])
    }

    fn pqdownheap(&mut self, freq: &[u32], mut k: usize) {
        let v = self.heap[k];
        let mut j = k << 1;
        while j <= self.heap_len {
            if j < self.heap_len && Self::smaller(freq, &self.depth, self.heap[j + 1], self.heap[j]) { j += 1; }
            if Self::smaller(freq, &self.depth, v, self.heap[j]) { break; }
            self.heap[k] = self.heap[j];
            k = j;
            j <<= 1;
        }
        self.heap[k] = v;
    }

    /// zlib build_tree + gen_bitlen: code lengths for `freq` (elems symbols), max `max_len` bits.
    /// Returns (lengths, max_code). `freq` may be modified (forced codes) as in zlib.
    fn build(&mut self, freq_in: &[u32], max_len: u8) -> (Vec<u8>, usize) {
        let elems = freq_in.len();
        let mut freq = vec![0u32; 2 * elems + 1];
        freq[..elems].copy_from_slice(freq_in);
        let mut dad = vec![0usize; 2 * elems + 1];
        let mut len = vec![0u8; 2 * elems + 1];
        self.heap_len = 0;
        self.heap_max = HEAP_SIZE;
        let mut max_code: isize = -1;
        for n in 0..elems {
            if freq[n] != 0 {
                self.heap_len += 1;
                self.heap[self.heap_len] = n;
                max_code = n as isize;
                self.depth[n] = 0;
            }
        }
        while self.heap_len < 2 {
            let node = if max_code < 2 { max_code += 1; max_code as usize } else { 0 };
            self.heap_len += 1;
            self.heap[self.heap_len] = node;
            freq[node] = 1;
            self.depth[node] = 0;
        }
        let max_code = max_code as usize;
        for n in (1..=self.heap_len / 2).rev() { self.pqdownheap(&freq, n); }
        let mut node = elems;
        loop {
            // pqremove: n = smallest
            let n = self.heap[1];
            self.heap[1] = self.heap[self.heap_len];
            self.heap_len -= 1;
            self.pqdownheap(&freq, 1);
            let m = self.heap[1];
            self.heap_max -= 1;
            self.heap[self.heap_max] = n;
            self.heap_max -= 1;
            self.heap[self.heap_max] = m;
            freq[node] = freq[n] + freq[m];
            self.depth[node] = self.depth[n].max(self.depth[m]) + 1;
            dad[n] = node;
            dad[m] = node;
            self.heap[1] = node;
            node += 1;
            self.pqdownheap(&freq, 1);
            if self.heap_len < 2 { break; }
        }
        self.heap_max -= 1;
        self.heap[self.heap_max] = self.heap[1];
        // gen_bitlen
        let mut bl_count = [0u16; 16];
        let mut overflow = 0i32;
        len[self.heap[self.heap_max]] = 0;
        for h in self.heap_max + 1..HEAP_SIZE {
            let n = self.heap[h];
            let mut bits = len[dad[n]] + 1;
            if bits > max_len { bits = max_len; overflow += 1; }
            len[n] = bits;
            if n > max_code { continue; }
            bl_count[bits as usize] += 1;
        }
        if overflow > 0 {
            loop {
                let mut bits = max_len as usize - 1;
                while bl_count[bits] == 0 { bits -= 1; }
                bl_count[bits] -= 1;
                bl_count[bits + 1] += 2;
                bl_count[max_len as usize] -= 1;
                overflow -= 2;
                if overflow <= 0 { break; }
            }
            let mut h = HEAP_SIZE;
            for bits in (1..=max_len as usize).rev() {
                let mut n = bl_count[bits];
                while n != 0 {
                    h -= 1;
                    let m = self.heap[h];
                    if m > max_code { continue; }
                    len[m] = bits as u8;
                    n -= 1;
                }
            }
        }
        (len[..elems].to_vec(), max_code)
    }
}

/// zlib scan_tree/send_tree run-length encoding of one tree's lengths (codes 0..=max_code).
fn send_tree(lens: &[u8], max_code: usize, out: &mut Vec<(u8, u8)>) {
    let mut prevlen: i32 = -1;
    let mut nextlen = lens[0] as i32;
    let mut count = 0i32;
    let (mut max_count, mut min_count) = if nextlen == 0 { (138, 3) } else { (7, 4) };
    for n in 0..=max_code {
        let curlen = nextlen;
        nextlen = if n + 1 <= max_code { lens[n + 1] as i32 } else { 0xffff };
        count += 1;
        if count < max_count && curlen == nextlen { continue; }
        if count < min_count {
            for _ in 0..count { out.push((curlen as u8, 0)); }
        } else if curlen != 0 {
            if curlen != prevlen { out.push((curlen as u8, 0)); count -= 1; }
            out.push((16, (count - 3) as u8));
        } else if count <= 10 {
            out.push((17, (count - 3) as u8));
        } else {
            out.push((18, (count - 11) as u8));
        }
        count = 0;
        prevlen = curlen;
        if nextlen == 0 { max_count = 138; min_count = 3; }
        else if curlen == nextlen { max_count = 6; min_count = 3; }
        else { max_count = 7; min_count = 4; }
    }
}

fn len_sym(len: u16, alt: bool) -> usize {
    if alt { return 257 + 27; }
    if len == 258 { return 257 + 28; }
    let mut i = 27;
    while LEN_BASE[i] > len { i -= 1; }
    257 + i
}

fn dist_sym(d: u16) -> usize {
    let mut i = 29;
    while DIST_BASE[i] > d { i -= 1; }
    i
}

/// The dynamic header zlib would transmit for a block with these tokens.
pub fn zlib_header(toks: &[Tok]) -> Kind {
    let mut lf = vec![0u32; L_CODES];
    let mut df = vec![0u32; D_CODES];
    for t in toks {
        match *t {
            Tok::Lit(b) => lf[b as usize] += 1,
            Tok::Match { len, dist, alt } => { lf[len_sym(len, alt)] += 1; df[dist_sym(dist)] += 1; }
        }
    }
    lf[256] = 1;
    let mut b = Builder { heap: [0; HEAP_SIZE], heap_len: 0, heap_max: HEAP_SIZE, depth: [0; HEAP_SIZE] };
    let (llens, lmax) = b.build(&lf, 15);
    let (dlens, dmax) = b.build(&df, 15);
    // bit-length tree from the run-length symbol frequencies of both trees
    let mut syms_l = vec![];
    send_tree(&llens, lmax, &mut syms_l);
    let mut syms_d = vec![];
    send_tree(&dlens, dmax, &mut syms_d);
    let mut blf = vec![0u32; BL_CODES];
    for &(s, _) in syms_l.iter().chain(&syms_d) { blf[s as usize] += 1; }
    let (bllens, _) = b.build(&blf, 7);
    let mut max_blindex = BL_CODES - 1;
    while max_blindex >= 3 && bllens[CL_ORDER[max_blindex]] == 0 { max_blindex -= 1; }
    let mut cl_lens = [0u8; 19];
    cl_lens.copy_from_slice(&bllens);
    let mut cl_syms = syms_l;
    cl_syms.extend(syms_d);
    Kind::Dynamic { hlit: (lmax + 1) as u16, hdist: (dmax + 1) as u16, hclen: (max_blindex + 1) as u16, cl_lens, cl_syms }
}
