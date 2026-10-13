//! JPEG recompression support (8-bit, Huffman coded, baseline/extended sequential and
//! progressive): parse the marker structure, decode every scan to quantized DCT coefficients, and
//! re-encode the coefficients to the identical bytes.
//!
//! A JPEG is split into
//! * a header: the file with the entropy-coded data of each scan removed (all marker segments,
//!   tables and metadata, verbatim);
//! * the coefficients of every component (compact serialization, see [`write_coefs`]);
//! * a small record of the encoder's padding bits when they are not the standard 1-bits.
//!
//! [`rebuild`] reverses the split. Re-encoding follows the canonical rules every common encoder
//! uses (libjpeg's, including its progressive EOB-run and correction-bit buffering), and
//! [`analyze`] proves the rebuild exact before a file is split, so a JPEG from an unusual encoder
//! is simply left as bytes.

/// Zigzag position → natural (row-major) position.
pub const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Largest image (in blocks) accepted, bounding coefficient memory (320 MB; ~100 MP at 4:2:0).
const MAX_BLOCKS: usize = 5 << 19;

#[derive(Clone, Debug)]
pub struct Comp {
    pub id: u8,
    pub h: usize,
    pub v: usize,
    pub tq: usize,
    /// Block grid padded to whole MCUs (the coefficient storage).
    pub bw: usize,
    pub bh: usize,
    /// Blocks actually covering the component (non-interleaved scans visit only these).
    pub cw: usize,
    pub ch: usize,
}

#[derive(Clone, Debug)]
pub struct Frame {
    pub progressive: bool,
    pub width: usize,
    pub height: usize,
    pub comps: Vec<Comp>,
    pub mcux: usize,
    pub mcuy: usize,
}

#[derive(Clone, Debug)]
struct Scan {
    comps: Vec<(usize, usize, usize)>, // (component index, DC table, AC table)
    ss: usize,
    se: usize,
    ah: u32,
    al: u32,
}

/// A Huffman table: canonical decode tables and the encode map.
#[derive(Clone)]
struct Huff {
    maxcode: [i32; 18],
    valptr: [i32; 17],
    mincode: [i32; 17],
    vals: Vec<u8>,
    /// symbol → (code, length); length 0 = no code
    enc: [(u16, u8); 256],
    /// 9-bit lookahead: (symbol, length), length 0 = slow path
    fast: Vec<(u8, u8)>,
}

impl Huff {
    fn new(counts: &[u8; 16], vals: &[u8]) -> Result<Huff, String> {
        let total: usize = counts.iter().map(|&c| c as usize).sum();
        if total != vals.len() || total > 256 { return Err("jpeg: bad huffman table".into()); }
        let mut h = Huff { maxcode: [-1; 18], valptr: [0; 17], mincode: [0; 17], vals: vals.to_vec(), enc: [(0, 0); 256], fast: vec![(0, 0); 512] };
        let (mut code, mut k) = (0i32, 0usize);
        for l in 1..=16 {
            let n = counts[l - 1] as usize;
            h.valptr[l] = k as i32;
            h.mincode[l] = code;
            for _ in 0..n {
                let s = vals[k] as usize;
                if h.enc[s].1 != 0 { return Err("jpeg: duplicate huffman symbol".into()); }
                h.enc[s] = (code as u16, l as u8);
                if l <= 9 {
                    let base = (code as usize) << (9 - l);
                    for e in &mut h.fast[base..base + (1 << (9 - l))] { *e = (s as u8, l as u8); }
                }
                code += 1;
                k += 1;
            }
            h.maxcode[l] = if n > 0 { code - 1 } else { -1 };
            if code > 1 << l { return Err("jpeg: oversubscribed huffman table".into()); }
            code <<= 1;
        }
        h.maxcode[17] = i32::MAX;
        Ok(h)
    }

    #[inline]
    fn decode(&self, r: &mut BitR) -> Result<u8, String> {
        r.fill();
        if r.n >= 9 {
            let (s, l) = self.fast[(r.acc >> (r.n - 9)) as usize & 511];
            if l != 0 { r.n -= l as u32; return Ok(s); }
        }
        let mut code = 0i32;
        for l in 1..=16 {
            code = (code << 1) | r.bit()? as i32;
            if self.maxcode[l] >= 0 && code <= self.maxcode[l] && code >= self.mincode[l] {
                return Ok(self.vals[(self.valptr[l] + code - self.mincode[l]) as usize]);
            }
        }
        Err("jpeg: bad huffman code".into())
    }

    #[inline]
    fn put(&self, w: &mut BitW, s: u8) -> Result<(), String> {
        let (c, l) = self.enc[s as usize];
        if l == 0 { return Err("jpeg: symbol missing from huffman table".into()); }
        w.bits(c as u32, l as u32);
        Ok(())
    }
}

/// Entropy-coded bit reader (MSB first) with byte unstuffing; stops at markers.
struct BitR<'a> {
    d: &'a [u8],
    pos: usize,
    acc: u64,
    n: u32,
}

impl<'a> BitR<'a> {
    fn new(d: &'a [u8], pos: usize) -> Self { BitR { d, pos, acc: 0, n: 0 } }

    #[inline]
    fn fill(&mut self) {
        while self.n <= 56 {
            let Some(&b) = self.d.get(self.pos) else { return };
            if b == 0xff {
                if self.d.get(self.pos + 1) != Some(&0) { return; } // a marker: no more data
                self.pos += 2;
            } else {
                self.pos += 1;
            }
            self.acc = (self.acc << 8) | b as u64;
            self.n += 8;
        }
    }

    #[inline]
    fn bit(&mut self) -> Result<u32, String> {
        if self.n == 0 { self.fill(); if self.n == 0 { return Err("jpeg: scan data ended early".into()); } }
        self.n -= 1;
        Ok((self.acc >> self.n) as u32 & 1)
    }

    #[inline]
    fn get(&mut self, k: u32) -> Result<u32, String> {
        if k == 0 { return Ok(0); }
        if self.n < k { self.fill(); if self.n < k { return Err("jpeg: scan data ended early".into()); } }
        self.n -= k;
        Ok((self.acc >> self.n) as u32 & ((1 << k) - 1))
    }

    /// Byte-align: the bits left in the current byte are padding. Returns (count, value). Errors if
    /// whole unread bytes remain (data the encoder would not have produced).
    fn align(&mut self) -> Result<(u8, u8), String> {
        let k = self.n % 8;
        let v = (self.acc >> (self.n - k)) as u8 & ((1u16 << k) - 1) as u8;
        self.n -= k;
        if self.n != 0 { return Err("jpeg: unread data before marker".into()); }
        Ok((k as u8, v))
    }
}

/// Entropy-coded bit writer with byte stuffing.
struct BitW {
    out: Vec<u8>,
    acc: u64,
    n: u32,
}

impl BitW {
    #[inline]
    fn bits(&mut self, v: u32, k: u32) {
        if k == 0 { return; }
        self.acc = (self.acc << k) | (v as u64 & ((1u64 << k) - 1));
        self.n += k;
        while self.n >= 8 {
            self.n -= 8;
            let b = (self.acc >> self.n) as u8;
            self.out.push(b);
            if b == 0xff { self.out.push(0); }
        }
    }

    /// Pad to a byte boundary: with ones (the standard), or with the recorded bits.
    fn align(&mut self, pad: Option<(u8, u8)>) -> Result<(), String> {
        let k = (8 - self.n % 8) % 8;
        match pad {
            None => self.bits((1 << k) - 1, k),
            Some((n, v)) => {
                if n as u32 != k { return Err("jpeg: padding record out of step".into()); }
                self.bits(v as u32, k);
            }
        }
        Ok(())
    }
}

/// Source of padding bits for the writer: all standard, or an explicit list in order.
struct Pads<'a> {
    explicit: &'a [(u8, u8)],
    i: usize,
}

impl Pads<'_> {
    fn next(&mut self) -> Result<Option<(u8, u8)>, String> {
        if self.explicit.is_empty() { return Ok(None); }
        let v = *self.explicit.get(self.i).ok_or("jpeg: padding record short")?;
        self.i += 1;
        Ok(Some(v))
    }
}

thread_local! {
    /// Developer symbol trace (see `diagnose`): None unless tracing.
    static TRACE: std::cell::RefCell<Option<Vec<String>>> = const { std::cell::RefCell::new(None) };
}

fn trace(f: impl FnOnce() -> String) {
    TRACE.with(|t| if let Some(v) = t.borrow_mut().as_mut() { v.push(f()) });
}

#[inline]
fn extend(v: u32, s: u32) -> i32 {
    if s == 0 { 0 } else if v < 1 << (s - 1) { v as i32 - (1 << s) + 1 } else { v as i32 }
}

#[inline]
fn bitlen(v: u32) -> u32 { 32 - v.leading_zeros() }

/// Value bits of `v` in category `bitlen(|v|)` (ones' complement for negatives).
#[inline]
fn vbits(v: i32) -> (u32, u32) {
    let a = v.unsigned_abs();
    let s = bitlen(a);
    (if v < 0 { (!(a)) & ((1 << s) - 1) } else { a }, s)
}

/// Everything parsed from a JPEG file.
struct Parsed {
    frame: Frame,
    /// Per scan: (scan header, Huffman tables in force, restart interval, data range in the file).
    scans: Vec<(Scan, Vec<Option<Huff>>, usize, std::ops::Range<usize>)>,
    /// Length of the JPEG (through EOI).
    len: usize,
}

fn be16(b: &[u8], o: usize) -> Result<usize, String> {
    Ok(u16::from_be_bytes(b.get(o..o + 2).ok_or("jpeg: truncated")?.try_into().unwrap()) as usize)
}

/// Parse the marker structure from SOI through EOI. With `entropy` false, `b` is a header (the
/// entropy-coded data already removed) and scan data ranges are empty.
fn parse(b: &[u8], entropy: bool) -> Result<Parsed, String> {
    if b.len() < 4 || b[0] != 0xff || b[1] != 0xd8 { return Err("jpeg: no SOI".into()); }
    let mut o = 2usize;
    let mut frame: Option<Frame> = None;
    let mut tables: Vec<Option<Huff>> = vec![None; 8]; // 0..3 DC, 4..7 AC
    let mut dri = 0usize;
    let mut scans = vec![];
    loop {
        if o + 2 > b.len() { return Err("jpeg: truncated".into()); }
        if b[o] != 0xff { return Err("jpeg: expected a marker".into()); }
        let m = b[o + 1];
        if m == 0xff { return Err("jpeg: fill bytes between segments".into()); }
        if m == 0xd9 {
            o += 2;
            break;
        }
        if m == 0xd8 || m == 0x01 || (0xd0..=0xd7).contains(&m) { return Err("jpeg: unexpected standalone marker".into()); }
        let len = be16(b, o + 2)?;
        if o + 2 + len > b.len() { return Err("jpeg: truncated".into()); }
        if len < 2 { return Err("jpeg: bad segment length".into()); }
        let seg = &b[o + 4..o + 2 + len];
        o += 2 + len;
        match m {
            0xc0 | 0xc1 | 0xc2 => {
                if frame.is_some() { return Err("jpeg: several frames".into()); }
                if seg.len() < 6 || seg[0] != 8 { return Err("jpeg: only 8-bit precision".into()); }
                let (height, width, nc) = (be16(seg, 1)?, be16(seg, 3)?, seg[5] as usize);
                if height == 0 || width == 0 || !(1..=4).contains(&nc) || seg.len() != 6 + 3 * nc { return Err("jpeg: bad frame header".into()); }
                let mut comps = vec![];
                for i in 0..nc {
                    let (id, hv, tq) = (seg[6 + 3 * i], seg[7 + 3 * i], seg[8 + 3 * i] as usize);
                    let (h, v) = ((hv >> 4) as usize, (hv & 15) as usize);
                    if !(1..=4).contains(&h) || !(1..=4).contains(&v) || tq > 3 { return Err("jpeg: bad component".into()); }
                    comps.push(Comp { id, h, v, tq, bw: 0, bh: 0, cw: 0, ch: 0 });
                }
                let hmax = comps.iter().map(|c| c.h).max().unwrap();
                let vmax = comps.iter().map(|c| c.v).max().unwrap();
                let (mcux, mcuy) = (width.div_ceil(8 * hmax), height.div_ceil(8 * vmax));
                let mut blocks = 0usize;
                for c in comps.iter_mut() {
                    c.bw = mcux * c.h;
                    c.bh = mcuy * c.v;
                    c.cw = (width * c.h).div_ceil(hmax).div_ceil(8);
                    c.ch = (height * c.v).div_ceil(vmax).div_ceil(8);
                    blocks += c.bw * c.bh;
                }
                if blocks > MAX_BLOCKS { return Err("jpeg: image too large".into()); }
                frame = Some(Frame { progressive: m == 0xc2, width, height, comps, mcux, mcuy });
            }
            0xc3 | 0xc5..=0xc7 | 0xc9..=0xcb | 0xcd..=0xcf => return Err("jpeg: unsupported coding process".into()),
            0xc4 => {
                let mut p = 0;
                while p < seg.len() {
                    let tc_th = seg[p];
                    let (tc, th) = ((tc_th >> 4) as usize, (tc_th & 15) as usize);
                    if tc > 1 || th > 3 || p + 17 > seg.len() { return Err("jpeg: bad DHT".into()); }
                    let counts: [u8; 16] = seg[p + 1..p + 17].try_into().unwrap();
                    let n: usize = counts.iter().map(|&c| c as usize).sum();
                    if p + 17 + n > seg.len() { return Err("jpeg: bad DHT".into()); }
                    tables[tc * 4 + th] = Some(Huff::new(&counts, &seg[p + 17..p + 17 + n])?);
                    p += 17 + n;
                }
            }
            0xdd => { if seg.len() != 2 { return Err("jpeg: bad DRI".into()); } dri = be16(seg, 0)?; }
            0xdc => return Err("jpeg: DNL not supported".into()),
            0xda => {
                let f = frame.as_ref().ok_or("jpeg: scan before frame")?;
                let ns = *seg.first().ok_or("jpeg: bad SOS")? as usize;
                if !(1..=4).contains(&ns) || seg.len() != 4 + 2 * ns { return Err("jpeg: bad SOS".into()); }
                let mut sc = vec![];
                for i in 0..ns {
                    let (cid, t) = (seg[1 + 2 * i], seg[2 + 2 * i]);
                    let ci = f.comps.iter().position(|c| c.id == cid).ok_or("jpeg: scan names an unknown component")?;
                    if sc.iter().any(|&(c, _, _)| c == ci) { return Err("jpeg: component twice in a scan".into()); }
                    let (td, ta) = ((t >> 4) as usize, (t & 15) as usize);
                    if td > 3 || ta > 3 { return Err("jpeg: bad table selector".into()); }
                    sc.push((ci, td, ta));
                }
                let (ss, se, ah, al) = (seg[1 + 2 * ns] as usize, seg[2 + 2 * ns] as usize, (seg[3 + 2 * ns] >> 4) as u32, (seg[3 + 2 * ns] & 15) as u32);
                if f.progressive {
                    if ss > se || se > 63 || (ss == 0) != (se == 0) || (ss > 0 && ns != 1) || al > 13 || ah > 13 { return Err("jpeg: bad progressive scan".into()); }
                } else if ss != 0 || se != 63 || ah != 0 || al != 0 {
                    return Err("jpeg: bad sequential scan".into());
                }
                let scan = Scan { comps: sc, ss, se, ah, al };
                let start = o;
                if entropy {
                    // entropy-coded data runs to the next marker that is not RSTn or a stuffed 0xFF
                    while o + 1 < b.len() && !(b[o] == 0xff && b[o + 1] != 0 && !(0xd0..=0xd7).contains(&b[o + 1])) { o += 1; }
                    if o + 1 >= b.len() { return Err("jpeg: truncated".into()); }
                }
                scans.push((scan, tables.clone(), dri, start..o));
            }
            _ => {} // APPn, COM, DQT, ...: verbatim
        }
    }
    let frame = frame.ok_or("jpeg: no frame")?;
    if scans.is_empty() { return Err("jpeg: no scans".into()); }
    Ok(Parsed { frame, scans, len: o })
}

/// Blocks a scan visits, in coding order, as (component index, block row, block column); MCU
/// boundaries every `per_mcu` entries.
fn scan_blocks(f: &Frame, s: &Scan) -> (Vec<(usize, usize, usize)>, usize) {
    let mut v = vec![];
    if s.comps.len() == 1 {
        let ci = s.comps[0].0;
        let c = &f.comps[ci];
        for r in 0..c.ch { for col in 0..c.cw { v.push((ci, r, col)); } }
        (v, 1)
    } else {
        let per: usize = s.comps.iter().map(|&(ci, _, _)| f.comps[ci].h * f.comps[ci].v).sum();
        for my in 0..f.mcuy {
            for mx in 0..f.mcux {
                for &(ci, _, _) in &s.comps {
                    let c = &f.comps[ci];
                    for by in 0..c.v { for bx in 0..c.h { v.push((ci, my * c.v + by, mx * c.h + bx)); } }
                }
            }
        }
        (v, per)
    }
}

/// Coefficients of every component, in natural order, 64 per block over the padded grid.
pub struct Coefs {
    pub planes: Vec<Vec<i16>>,
}

impl Coefs {
    fn new(f: &Frame) -> Coefs { Coefs { planes: f.comps.iter().map(|c| vec![0i16; c.bw * c.bh * 64]).collect() } }
    #[inline]
    fn block(&mut self, f: &Frame, ci: usize, r: usize, c: usize) -> &mut [i16] {
        let o = (r * f.comps[ci].bw + c) * 64;
        &mut self.planes[ci][o..o + 64]
    }
}

fn table<'t>(t: &'t [Option<Huff>], i: usize) -> Result<&'t Huff, String> { t[i].as_ref().ok_or_else(|| "jpeg: missing huffman table".to_string()) }

/// Decode one scan into `co`; returns the padding bits at each restart and at the end.
fn decode_scan(b: &[u8], f: &Frame, s: &Scan, t: &[Option<Huff>], dri: usize, range: std::ops::Range<usize>, co: &mut Coefs) -> Result<Vec<(u8, u8)>, String> {
    let (blocks, per) = scan_blocks(f, s);
    let n_mcu = blocks.len() / per;
    let mut r = BitR::new(&b[..range.end], range.start);
    let mut pads = vec![];
    let mut pred = [0i32; 4];
    let mut eobrun = 0u32;
    let mut next_rst = 0u8;
    for m in 0..n_mcu {
        if dri > 0 && m > 0 && m % dri == 0 {
            pads.push(r.align()?);
            if r.pos + 2 > range.end || b[r.pos] != 0xff || b[r.pos + 1] != 0xd0 + next_rst { return Err("jpeg: missing restart marker".into()); }
            r.pos += 2;
            next_rst = (next_rst + 1) & 7;
            pred = [0; 4];
            eobrun = 0;
        }
        for &(ci, br, bc) in &blocks[m * per..(m + 1) * per] {
            let k_sc = s.comps.iter().position(|x| x.0 == ci).unwrap();
            let (_, td, ta) = s.comps[k_sc];
            let blk = co.block(f, ci, br, bc);
            if !f.progressive {
                let dc = table(t, td)?;
                let ac = table(t, 4 + ta)?;
                let sz = dc.decode(&mut r)? as u32;
                if sz > 11 { return Err("jpeg: bad DC size".into()); }
                pred[k_sc] += extend(r.get(sz)?, sz);
                if !(-32768..=32767).contains(&pred[k_sc]) { return Err("jpeg: DC out of range".into()); }
                blk[0] = pred[k_sc] as i16;
                let mut k = 1;
                let mut zrl = false;
                while k < 64 {
                    let rs = ac.decode(&mut r)?;
                    let (run, sz) = ((rs >> 4) as usize, (rs & 15) as u32);
                    if sz == 0 {
                        if run == 15 { k += 16; zrl = true; if k > 64 { return Err("jpeg: run past block end".into()); } continue; }
                        if run != 0 || zrl { return Err("jpeg: non-canonical end of block".into()); }
                        break;
                    }
                    k += run;
                    if k > 63 || sz > 10 { return Err("jpeg: bad AC coefficient".into()); }
                    blk[ZIGZAG[k]] = extend(r.get(sz)?, sz) as i16;
                    k += 1;
                    zrl = false;
                }
                if zrl { return Err("jpeg: non-canonical zero run".into()); }
            } else if s.ss == 0 {
                if s.ah == 0 {
                    let sz = table(t, td)?.decode(&mut r)? as u32;
                    if sz > 11 { return Err("jpeg: bad DC size".into()); }
                    pred[k_sc] += extend(r.get(sz)?, sz);
                    let v = pred[k_sc] << s.al;
                    if !(-32768..=32767).contains(&v) { return Err("jpeg: DC out of range".into()); }
                    blk[0] = v as i16;
                } else if r.get(1)? == 1 {
                    blk[0] |= 1 << s.al;
                }
            } else if s.ah == 0 {
                if eobrun > 0 { eobrun -= 1; continue; }
                let ac = table(t, 4 + ta)?;
                let mut k = s.ss;
                while k <= s.se {
                    let rs = ac.decode(&mut r)?;
                    let (run, sz) = ((rs >> 4) as u32, (rs & 15) as u32);
                    if sz == 0 {
                        if run == 15 { k += 16; if k > s.se + 1 { return Err("jpeg: run past band end".into()); } continue; }
                        eobrun = (1 << run) + r.get(run)? - 1;
                        break;
                    }
                    k += run as usize;
                    if k > s.se || sz > 10 { return Err("jpeg: bad AC coefficient".into()); }
                    let v = extend(r.get(sz)?, sz) << s.al;
                    if !(-32768..=32767).contains(&v) { return Err("jpeg: AC out of range".into()); }
                    blk[ZIGZAG[k]] = v as i16;
                    k += 1;
                }
            } else {
                let ac = table(t, 4 + ta)?;
                let (p1, m1) = (1i16 << s.al, -1i16 << s.al);
                let mut k = s.ss;
                if eobrun == 0 {
                    while k <= s.se {
                        let rs = ac.decode(&mut r)?;
                        trace(|| format!("{br},{bc} k{k} sym {rs:02x}"));
                        let (mut run, sz) = ((rs >> 4) as i32, (rs & 15) as u32);
                        let mut val = 0i16;
                        if sz != 0 {
                            if sz != 1 { return Err("jpeg: bad refinement size".into()); }
                            val = if r.get(1)? == 1 { p1 } else { m1 };
                        } else if run != 15 {
                            eobrun = (1 << run) + r.get(run as u32)?;
                            trace(|| format!("EOBRUN {eobrun}"));
                            break;
                        }
                        loop {
                            let c = &mut blk[ZIGZAG[k]];
                            if *c != 0 {
                                if r.get(1)? == 1 && *c & p1 == 0 { *c += if *c >= 0 { p1 } else { m1 }; }
                            } else {
                                run -= 1;
                                if run < 0 { break; }
                            }
                            k += 1;
                            if k > s.se { break; }
                        }
                        if val != 0 {
                            if k > s.se { return Err("jpeg: refinement past band end".into()); }
                            blk[ZIGZAG[k]] = val;
                        }
                        k += 1;
                    }
                }
                if eobrun > 0 {
                    while k <= s.se {
                        let c = &mut blk[ZIGZAG[k]];
                        if *c != 0 && r.get(1)? == 1 && *c & p1 == 0 { *c += if *c >= 0 { p1 } else { m1 }; }
                        k += 1;
                    }
                    eobrun -= 1;
                }
            }
        }
    }
    pads.push(r.align()?);
    if r.pos != range.end { return Err("jpeg: data after the last MCU".into()); }
    Ok(pads)
}

/// Re-encode one scan from final coefficients (libjpeg's canonical choices).
/// EOB-run policies of progressive AC scans: libjpeg's (a block's trailing zeros join the run of
/// following empty blocks), or a block with coefficients ends with its own EOB (runs gather only
/// empty blocks).
const POLICY_LIBJPEG: u8 = 0;
const POLICY_SEPARATE: u8 = 1;

fn encode_scan(f: &Frame, s: &Scan, t: &[Option<Huff>], dri: usize, co: &Coefs, policy: u8, pads: &mut Pads, out: &mut Vec<u8>) -> Result<(), String> {
    let (blocks, per) = scan_blocks(f, s);
    let n_mcu = blocks.len() / per;
    let mut w = BitW { out: std::mem::take(out), acc: 0, n: 0 };
    let mut pred = [0i32; 4];
    let mut eobrun = 0u32;
    let mut be: Vec<u8> = Vec::new(); // buffered correction bits (refinement scans)
    let mut next_rst = 0u8;
    let ac_tab = |k_sc: usize| table(t, 4 + s.comps[k_sc].2);
    fn emit_eobrun(w: &mut BitW, ac: &Huff, eobrun: &mut u32, be: &mut Vec<u8>) -> Result<(), String> {
        if *eobrun > 0 {
            let nb = bitlen(*eobrun) - 1;
            trace(|| format!("EOBRUN {} (be {})", *eobrun, be.len()));
            if nb > 14 { return Err("jpeg: EOB run too long".into()); }
            ac.put(w, (nb << 4) as u8)?;
            w.bits(*eobrun, nb);
            *eobrun = 0;
            for &bit in be.iter() { w.bits(bit as u32, 1); }
            be.clear();
        }
        Ok(())
    }
    for m in 0..n_mcu {
        if dri > 0 && m > 0 && m % dri == 0 {
            if f.progressive && s.ss > 0 { emit_eobrun(&mut w, ac_tab(0)?, &mut eobrun, &mut be)?; }
            w.align(pads.next()?)?;
            w.out.extend_from_slice(&[0xff, 0xd0 + next_rst]);
            next_rst = (next_rst + 1) & 7;
            pred = [0; 4];
        }
        for &(ci, br, bc) in &blocks[m * per..(m + 1) * per] {
            let k_sc = s.comps.iter().position(|x| x.0 == ci).unwrap();
            let td = s.comps[k_sc].1;
            let o = (br * f.comps[ci].bw + bc) * 64;
            let blk = &co.planes[ci][o..o + 64];
            if !f.progressive {
                let (dc, ac) = (table(t, td)?, ac_tab(k_sc)?);
                let v = blk[0] as i32;
                let (bits, sz) = vbits(v - pred[k_sc]);
                pred[k_sc] = v;
                dc.put(&mut w, sz as u8)?;
                w.bits(bits, sz);
                let mut run = 0;
                for k in 1..64 {
                    let v = blk[ZIGZAG[k]] as i32;
                    if v == 0 { run += 1; continue; }
                    while run > 15 { ac.put(&mut w, 0xf0)?; run -= 16; }
                    let (bits, sz) = vbits(v);
                    ac.put(&mut w, ((run << 4) | sz) as u8)?;
                    w.bits(bits, sz);
                    run = 0;
                }
                if run > 0 { ac.put(&mut w, 0)?; }
            } else if s.ss == 0 {
                if s.ah == 0 {
                    let v = (blk[0] as i32) >> s.al;
                    let (bits, sz) = vbits(v - pred[k_sc]);
                    pred[k_sc] = v;
                    table(t, td)?.put(&mut w, sz as u8)?;
                    w.bits(bits, sz);
                } else {
                    w.bits(((blk[0] as i32) >> s.al) as u32 & 1, 1);
                }
            } else if s.ah == 0 {
                let ac = ac_tab(k_sc)?;
                let mut run = 0u32;
                for k in s.ss..=s.se {
                    let v = blk[ZIGZAG[k]] as i32;
                    let a = v.unsigned_abs() >> s.al;
                    if a == 0 { run += 1; continue; }
                    emit_eobrun(&mut w, ac, &mut eobrun, &mut be)?;
                    while run > 15 { ac.put(&mut w, 0xf0)?; run -= 16; }
                    let sz = bitlen(a);
                    ac.put(&mut w, ((run << 4) | sz) as u8)?;
                    w.bits(if v < 0 { !a & ((1 << sz) - 1) } else { a }, sz);
                    run = 0;
                }
                if run > 0 {
                    let had = run <= (s.se - s.ss) as u32;
                    eobrun += 1;
                    if eobrun == 0x7fff || (policy == POLICY_SEPARATE && had) { emit_eobrun(&mut w, ac, &mut eobrun, &mut be)?; }
                }
            } else {
                let ac = ac_tab(k_sc)?;
                let mut abs = [0u32; 64];
                let mut eob = 0;
                for k in s.ss..=s.se {
                    abs[k] = (blk[ZIGZAG[k]] as i32).unsigned_abs() >> s.al;
                    if abs[k] == 1 { eob = k; }
                }
                let mut run = 0u32;
                let mut corr: Vec<u8> = Vec::new();
                for k in s.ss..=s.se {
                    let a = abs[k];
                    if a == 0 { run += 1; continue; }
                    while run > 15 && k <= eob {
                        emit_eobrun(&mut w, ac, &mut eobrun, &mut be)?;
                        trace(|| format!("{br},{bc} k{k} sym f0"));
                        ac.put(&mut w, 0xf0)?;
                        run -= 16;
                        for &bit in &corr { w.bits(bit as u32, 1); }
                        corr.clear();
                    }
                    if a > 1 { corr.push((a & 1) as u8); continue; }
                    emit_eobrun(&mut w, ac, &mut eobrun, &mut be)?;
                    trace(|| format!("{br},{bc} k{k} sym {:02x}", (run << 4) | 1));
                    ac.put(&mut w, ((run << 4) | 1) as u8)?;
                    w.bits((blk[ZIGZAG[k]] >= 0) as u32, 1);
                    for &bit in &corr { w.bits(bit as u32, 1); }
                    corr.clear();
                    run = 0;
                }
                if run > 0 || !corr.is_empty() {
                    if policy == POLICY_SEPARATE && abs[s.ss..=s.se].iter().any(|&a| a != 0) {
                        // a block with content ends with its own EOB, after the pending run
                        emit_eobrun(&mut w, ac, &mut eobrun, &mut be)?;
                        eobrun = 1;
                        be.extend_from_slice(&corr);
                        emit_eobrun(&mut w, ac, &mut eobrun, &mut be)?;
                    } else {
                        eobrun += 1;
                        be.extend_from_slice(&corr);
                        if eobrun == 0x7fff || be.len() > 1000 - 64 + 1 { emit_eobrun(&mut w, ac, &mut eobrun, &mut be)?; }
                    }
                }
            }
        }
    }
    if f.progressive && s.ss > 0 { emit_eobrun(&mut w, ac_tab(0)?, &mut eobrun, &mut be)?; }
    w.align(pads.next()?)?;
    *out = w.out;
    Ok(())
}

// ------------------------------------------------------------------ coefficient serialization

/// Compact form of the coefficients: per component plane, per block in raster order: the count
/// `n` of coefficients up to the last nonzero one (zigzag order, 0..=64), then those `n` values as
/// zigzag-signed varints.
pub fn write_coefs(f: &Frame, co: &Coefs) -> Vec<u8> {
    let mut out = Vec::new();
    for (ci, c) in f.comps.iter().enumerate() {
        for bi in 0..c.bw * c.bh {
            let blk = &co.planes[ci][bi * 64..bi * 64 + 64];
            let n = (0..64).rev().find(|&k| blk[ZIGZAG[k]] != 0).map(|k| k + 1).unwrap_or(0);
            out.push(n as u8);
            for k in 0..n {
                let v = blk[ZIGZAG[k]] as i32;
                let mut z = ((v << 1) ^ (v >> 31)) as u32;
                while z >= 0x80 { out.push(z as u8 | 0x80); z >>= 7; }
                out.push(z as u8);
            }
        }
    }
    out
}

pub fn read_coefs(f: &Frame, b: &[u8]) -> Result<Coefs, String> {
    let mut co = Coefs::new(f);
    let mut o = 0usize;
    let mut next = || -> Result<u8, String> { let v = *b.get(o).ok_or("jpeg: coefficient data truncated")?; o += 1; Ok(v) };
    for (ci, c) in f.comps.iter().enumerate() {
        for bi in 0..c.bw * c.bh {
            let n = next()? as usize;
            if n > 64 { return Err("jpeg: bad coefficient count".into()); }
            for k in 0..n {
                let mut z = 0u32;
                for sh in [0, 7, 14] {
                    let x = next()?;
                    z |= ((x & 0x7f) as u32) << sh;
                    if x & 0x80 == 0 { break; }
                    if sh == 14 { return Err("jpeg: bad coefficient".into()); }
                }
                let v = (z >> 1) as i32 ^ -((z & 1) as i32);
                if !(-32768..=32767).contains(&v) { return Err("jpeg: bad coefficient".into()); }
                co.planes[ci][bi * 64 + ZIGZAG[k]] = v as i16;
            }
        }
    }
    if o != b.len() { return Err("jpeg: coefficient data has trailing bytes".into()); }
    Ok(co)
}

// ------------------------------------------------------------------ split / rebuild

/// A JPEG split for recompression.
pub struct Split {
    /// Bytes of the JPEG (SOI through EOI) this split rebuilds.
    pub len: usize,
    pub header: Vec<u8>,
    pub coefs: Vec<u8>,
    pub pads: Vec<u8>,
    pub frame: Frame,
    /// Quantization tables per component (natural order), for the coefficient model.
    pub quant: Vec<[u16; 64]>,
}

/// Encoder record: empty when every scan uses libjpeg's choices and every pad is standard ones.
/// Else: flags (bit 0 explicit pads, bit 1 per-scan policies), the policy of each scan, then the
/// (count, value) of every pad.
fn side_record(policies: &[u8], pads: &[(u8, u8)]) -> Vec<u8> {
    let odd_pads = !pads.iter().all(|&(n, v)| v as u16 == (1u16 << n) - 1);
    let odd_policy = policies.iter().any(|&p| p != POLICY_LIBJPEG);
    if !odd_pads && !odd_policy { return vec![]; }
    let mut r = vec![odd_pads as u8 | (odd_policy as u8) << 1];
    if odd_policy { r.extend_from_slice(policies); }
    if odd_pads { r.extend(pads.iter().flat_map(|&(n, v)| [n, v])); }
    r
}

/// Rebuild the JPEG from its split parts.
pub fn rebuild(header: &[u8], coefs: &[u8], side: &[u8]) -> Result<Vec<u8>, String> {
    let p = parse(header, false)?;
    let co = read_coefs(&p.frame, coefs)?;
    let (flags, rest) = side.split_first().map(|(f, r)| (*f, r)).unwrap_or((0, &[]));
    if flags > 3 { return Err("jpeg: bad encoder record".into()); }
    let (policies, rest) = if flags & 2 != 0 {
        if rest.len() < p.scans.len() { return Err("jpeg: bad encoder record".into()); }
        rest.split_at(p.scans.len())
    } else { (&[][..], rest) };
    if (flags & 1 == 0 && !rest.is_empty()) || rest.len() % 2 != 0 { return Err("jpeg: bad encoder record".into()); }
    let explicit: Vec<(u8, u8)> = rest.chunks(2).map(|c| (c[0], c[1])).collect();
    let mut pads = Pads { explicit: &explicit, i: 0 };
    let mut out = Vec::with_capacity(header.len() + coefs.len() / 4);
    let mut pos = 0usize;
    for (i, (scan, tabs, dri, r)) in p.scans.iter().enumerate() {
        out.extend_from_slice(&header[pos..r.start]);
        pos = r.start;
        let policy = policies.get(i).copied().unwrap_or(POLICY_LIBJPEG);
        if policy > POLICY_SEPARATE { return Err("jpeg: bad encoder record".into()); }
        encode_scan(&p.frame, scan, tabs, *dri, &co, policy, &mut pads, &mut out)?;
    }
    out.extend_from_slice(&header[pos..]);
    if pads.i != explicit.len() { return Err("jpeg: padding record has extra entries".into()); }
    Ok(out)
}

/// The EOB-run policy that reproduces each scan's original bytes (progressive AC scans may need
/// the non-libjpeg one).
fn find_policies(b: &[u8], p: &Parsed, co: &Coefs, pads: &[(u8, u8)]) -> Result<Vec<u8>, String> {
    let rec = if pads.iter().all(|&(n, v)| v as u16 == (1u16 << n) - 1) { vec![] } else { pads.to_vec() };
    let mut pd = Pads { explicit: &rec, i: 0 };
    let mut policies = vec![];
    for (scan, tabs, dri, r) in &p.scans {
        let orig = &b[r.clone()];
        let mark = pd.i;
        let mut chosen = None;
        for policy in [POLICY_LIBJPEG, POLICY_SEPARATE] {
            if policy != POLICY_LIBJPEG && !(p.frame.progressive && scan.ss > 0) { break; }
            pd.i = mark;
            let mut out = Vec::with_capacity(orig.len());
            if encode_scan(&p.frame, scan, tabs, *dri, co, policy, &mut pd, &mut out).is_ok() && out == orig {
                chosen = Some(policy);
                break;
            }
        }
        policies.push(chosen.ok_or("jpeg: re-encoding differs from the original")?);
    }
    Ok(policies)
}

/// Split the JPEG at the start of `b` (SOI through EOI) if its rebuild is exact.
pub fn analyze(b: &[u8]) -> Result<Split, String> {
    let p = parse(b, true)?;
    let mut co = Coefs::new(&p.frame);
    let mut pads = vec![];
    for (scan, tabs, dri, r) in &p.scans {
        pads.extend(decode_scan(b, &p.frame, scan, tabs, *dri, r.clone(), &mut co)?);
    }
    // the header: the file without the scans' entropy-coded data
    let mut header = Vec::with_capacity(4096);
    let mut pos = 0usize;
    for (_, _, _, r) in &p.scans {
        header.extend_from_slice(&b[pos..r.start]);
        pos = r.end;
    }
    header.extend_from_slice(&b[pos..p.len]);
    let policies = find_policies(b, &p, &co, &pads)?;
    let coefs = write_coefs(&p.frame, &co);
    drop(co); // the end-to-end check below rebuilds its own planes
    let side = side_record(&policies, &pads);
    let back = rebuild(&header, &coefs, &side)?;
    if back != b[..p.len] { return Err("jpeg: re-encoding differs from the original".into()); }
    let quant = quant_tables(&header, &p.frame)?;
    Ok(Split { len: p.len, header, coefs, pads: side, frame: p.frame, quant })
}

/// Quantization table of each component (natural order), from the DQT segments of `header`.
fn quant_tables(h: &[u8], f: &Frame) -> Result<Vec<[u16; 64]>, String> {
    let mut q = [[1u16; 64]; 4];
    let mut o = 2;
    while o + 4 <= h.len() && h[o + 1] != 0xd9 {
        let len = be16(h, o + 2)?;
        if h[o + 1] == 0xdb {
            let seg = &h[o + 4..o + 2 + len];
            let mut p = 0;
            while p < seg.len() {
                let (pq, tq) = ((seg[p] >> 4) as usize, (seg[p] & 15) as usize);
                if tq > 3 || pq > 1 || p + 1 + 64 * (pq + 1) > seg.len() { return Err("jpeg: bad DQT".into()); }
                for k in 0..64 {
                    let v = if pq == 0 { seg[p + 1 + k] as u16 } else { u16::from_be_bytes([seg[p + 1 + 2 * k], seg[p + 2 + 2 * k]]) };
                    q[tq][ZIGZAG[k]] = v.max(1);
                }
                p += 1 + 64 * (pq + 1);
            }
        }
        o += 2 + len;
    }
    Ok(f.comps.iter().map(|c| q[c.tq]).collect())
}

/// Developer diagnostics: where a JPEG's rebuild first differs (scan index, scan parameters,
/// offset within the scan data).
pub fn diagnose(b: &[u8]) -> String {
    let p = match parse(b, true) { Ok(p) => p, Err(e) => return format!("parse: {e}") };
    let mut co = Coefs::new(&p.frame);
    let mut pads = vec![];
    for (i, (scan, tabs, dri, r)) in p.scans.iter().enumerate() {
        match decode_scan(b, &p.frame, scan, tabs, *dri, r.clone(), &mut co) {
            Ok(v) => pads.extend(v),
            Err(e) => return format!("decode scan {i} (ss {} se {} ah {} al {}): {e}", scan.ss, scan.se, scan.ah, scan.al),
        }
    }
    let std_pads = pads.iter().all(|&(n, v)| v as u16 == (1u16 << n) - 1);
    let rec = if std_pads { vec![] } else { pads.clone() };
    let mut pd = Pads { explicit: &rec, i: 0 };
    let mut policies = vec![];
    for (i, (scan, tabs, dri, r)) in p.scans.iter().enumerate() {
        let orig = &b[r.clone()];
        let mark = pd.i;
        let mut first = String::new();
        let mut found = None;
        for policy in [POLICY_LIBJPEG, POLICY_SEPARATE] {
            pd.i = mark;
            let mut out = vec![];
            if let Err(e) = encode_scan(&p.frame, scan, tabs, *dri, &co, policy, &mut pd, &mut out) {
                if first.is_empty() { first = format!("encode scan {i} (policy {policy}): {e}"); }
                continue;
            }
            if out == orig { found = Some(policy); break; }
            if first.is_empty() {
                let at = out.iter().zip(orig).position(|(x, y)| x != y).unwrap_or(out.len().min(orig.len()));
                first = format!("scan {i} (comps {:?} ss {} se {} ah {} al {}) differs at byte {at} of {} (rebuilt {} bytes)",
                    scan.comps.iter().map(|c| c.0).collect::<Vec<_>>(), scan.ss, scan.se, scan.ah, scan.al, orig.len(), out.len());
            }
        }
        match found { Some(pol) => policies.push(pol), None => return first }
    }
    format!("OK ({} scans, {}x{}, {}, pads {}, eob policy {})", p.scans.len(), p.frame.width, p.frame.height,
        if p.frame.progressive { "progressive" } else { "sequential" }, if std_pads { "standard" } else { "explicit" },
        if policies.iter().all(|&x| x == POLICY_LIBJPEG) { "libjpeg" } else { "separate" })
}

/// Developer trace: the decoded symbol events of scan `si` and the events of its re-encoding
/// under `policy`, written to `<out>.dec` / `<out>.enc`.
pub fn trace_scan(b: &[u8], si: usize, policy: u8, out: &str) -> Result<(), String> {
    let p = parse(b, true)?;
    let mut co = Coefs::new(&p.frame);
    let mut pads = vec![];
    for (i, (scan, tabs, dri, r)) in p.scans.iter().enumerate() {
        if i == si { TRACE.with(|t| *t.borrow_mut() = Some(vec![])); }
        pads.extend(decode_scan(b, &p.frame, scan, tabs, *dri, r.clone(), &mut co)?);
        if i == si {
            let v = TRACE.with(|t| t.borrow_mut().take()).unwrap();
            std::fs::write(format!("{out}.dec"), v.join("\n")).map_err(|e| e.to_string())?;
        }
    }
    let (scan, tabs, dri, _) = &p.scans[si];
    TRACE.with(|t| *t.borrow_mut() = Some(vec![]));
    let mut pd = Pads { explicit: &[], i: 0 };
    let mut o = vec![];
    let r = encode_scan(&p.frame, scan, tabs, *dri, &co, policy, &mut pd, &mut o);
    let v = TRACE.with(|t| t.borrow_mut().take()).unwrap();
    std::fs::write(format!("{out}.enc"), v.join("\n")).map_err(|e| e.to_string())?;
    r
}

/// Length of the JPEG at the start of `b` by marker structure (SOI through EOI), for planning.
/// `Err(true)` when `b` ends inside it.
pub fn extent(b: &[u8]) -> Result<usize, bool> {
    match parse(b, true) {
        Ok(p) => Ok(p.len),
        Err(e) => Err(e == "jpeg: truncated"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_jpegs_rebuild_exactly() {
        let Ok(rd) = std::fs::read_dir("bench/corpus/jpeg") else { eprintln!("jpeg test: corpus missing — skipping"); return };
        let mut n = 0;
        for e in rd.flatten() {
            let b = std::fs::read(e.path()).unwrap();
            let s = analyze(&b).unwrap_or_else(|err| panic!("{}: {err}", e.path().display()));
            assert_eq!(s.len, b.len());
            assert_eq!(rebuild(&s.header, &s.coefs, &s.pads).unwrap(), b);
            n += 1;
        }
        assert!(n > 0);
    }
}
