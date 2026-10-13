//! H.264 CAVLC: the other entropy coding of slice data (Baseline profile, and Main or High
//! streams that choose it) — variable-length codes instead of the arithmetic coder.
//!
//! A CAVLC slice holds the same macroblock syntax as a CABAC one, so it is recoded through the
//! same bins: its syntax elements are read ([`Vlc::mb`]), binarized the way CABAC would
//! ([`binarize`]) and modelled like any other slice; on rebuild, [`walk`] turns the bins back
//! into syntax elements and the same [`Vlc::mb`] writes them. The two codings differ in a few
//! things one can express and the other cannot (see [`to_bins`]); a slice using them is left as
//! bytes, like any slice whose rebuild is not exact.
use super::super::h264_vlc_tables::{CBP_INTER, CBP_INTER_GRAY, CBP_INTRA, CBP_INTRA_GRAY, COEFF_TOKEN, RUN_BEFORE, TOTAL_ZEROS, TOTAL_ZEROS_DC};
use super::{walk, Bins, Info, MbSyn, Pps, Scratch, SliceHeader, Sps, BLK, B_PRED, B_SUB, MAX_MBS, PREDICTED, SLICE_I, SLICE_P};
use crate::bits::{BitReader, BitWriter, PrefixDec};

/// A prefix code of the tables.
#[derive(Clone, Copy)]
enum Code {
    /// coeff_token by table.
    Token(usize),
    /// total_zeros by the block's number of coefficients; of a chroma DC block.
    Zeros(usize),
    ZerosDc(usize),
    /// run_before by the zeros left (7: more than 6).
    Run(usize),
}

impl Code {
    fn table(self) -> &'static [(u8, u8)] {
        match self {
            Code::Token(t) => COEFF_TOKEN[t].as_flattened(),
            Code::Zeros(n) => &TOTAL_ZEROS[n],
            Code::ZerosDc(n) => &TOTAL_ZEROS_DC[n],
            Code::Run(n) => &RUN_BEFORE[n],
        }
    }
}

/// Decoders of every table.
struct Decoders {
    token: Vec<PrefixDec>,
    zeros: Vec<PrefixDec>,
    zeros_dc: Vec<PrefixDec>,
    run: Vec<PrefixDec>,
}

fn decoders() -> &'static Decoders {
    static D: std::sync::OnceLock<Decoders> = std::sync::OnceLock::new();
    D.get_or_init(|| {
        let dec = |t: &[(u8, u8)]| PrefixDec::new(t.iter().enumerate().filter(|(_, c)| c.1 > 0).map(|(sym, &(code, len))| (code as u32, len, sym as u32)));
        Decoders {
            token: (0..5).map(|t| dec(Code::Token(t).table())).collect(),
            zeros: (0..16).map(|n| dec(Code::Zeros(n).table())).collect(),
            zeros_dc: (0..4).map(|n| dec(Code::ZerosDc(n).table())).collect(),
            run: (0..8).map(|n| dec(Code::Run(n).table())).collect(),
        }
    })
}

/// The slice data as bits, read or written: a reader returns what it reads, a writer writes the
/// value it is given and returns it — so the syntax is written once, for both directions.
trait Io {
    fn reading(&self) -> bool;
    /// `n` (up to 32) bits.
    fn u(&mut self, v: u32, n: u32) -> Option<u32>;
    /// `n` zero bits and a one (a reader: how many zeros it found).
    fn zeros(&mut self, n: u32) -> Option<u32>;
    /// A symbol of a prefix code.
    fn vlc(&mut self, code: Code, sym: usize) -> Option<usize>;
    /// Zero bits up to the byte boundary.
    fn align(&mut self) -> Option<()>;

    /// Unsigned Exp-Golomb.
    fn ue(&mut self, v: u32) -> Option<u32> {
        let x = v as u64 + 1;
        let n = 63 - x.leading_zeros();
        let n = self.zeros(n)?;
        if n > 31 { return None; }
        let rest = self.u((x & ((1 << n) - 1)) as u32, n)?;
        Some((((1u64 << n) | rest as u64) - 1) as u32)
    }

    /// Signed Exp-Golomb.
    fn se(&mut self, v: i32) -> Option<i32> {
        let k = self.ue(if v > 0 { (2 * v as i64 - 1) as u32 } else { (-2 * v as i64) as u32 })? as i64;
        Some(if k & 1 == 1 { (k + 1) / 2 } else { -(k / 2) } as i32)
    }

    /// Truncated Exp-Golomb for values up to `max` (at least 1).
    fn te(&mut self, v: u32, max: u32) -> Option<u32> {
        if max > 1 { return self.ue(v).filter(|&x| x <= max); }
        Some(1 - self.u(1 - v.min(1), 1)?)
    }
}

struct Rd<'a> {
    r: BitReader<'a>,
    /// Where the slice data ends: the position of the stop bit.
    end: usize,
}

impl Io for Rd<'_> {
    fn reading(&self) -> bool { true }
    fn u(&mut self, _: u32, n: u32) -> Option<u32> { (self.r.pos() + n as usize <= self.end).then(|| self.r.get(n)) }
    fn zeros(&mut self, _: u32) -> Option<u32> {
        let mut n = 0;
        while self.u(0, 1)? == 0 {
            n += 1;
            if n > 32 { return None; }
        }
        Some(n)
    }
    fn vlc(&mut self, code: Code, _: usize) -> Option<usize> {
        let d = decoders();
        let dec = match code { Code::Token(t) => &d.token[t], Code::Zeros(n) => &d.zeros[n], Code::ZerosDc(n) => &d.zeros_dc[n], Code::Run(n) => &d.run[n] };
        let (sym, len) = dec.peek(&self.r)?;
        if self.r.pos() + len as usize > self.end { return None; }
        self.r.skip(len as u32);
        Some(sym as usize)
    }
    fn align(&mut self) -> Option<()> {
        let n = (8 - self.r.pos() % 8) % 8;
        (self.u(0, n as u32)? == 0).then_some(())
    }
}

struct Wr<'a>(&'a mut BitWriter);

impl Io for Wr<'_> {
    fn reading(&self) -> bool { false }
    fn u(&mut self, v: u32, n: u32) -> Option<u32> {
        self.0.put(v, n);
        Some(v)
    }
    fn zeros(&mut self, n: u32) -> Option<u32> {
        if n > 32 { return None; }
        self.0.put(0, n);
        self.0.put(1, 1);
        Some(n)
    }
    fn vlc(&mut self, code: Code, sym: usize) -> Option<usize> {
        let &(bits, len) = code.table().get(sym)?;
        if len == 0 { return None; }
        self.0.put(bits as u32, len as u32);
        Some(sym)
    }
    fn align(&mut self) -> Option<()> {
        let n = (8 - self.0.len() % 8) % 8;
        self.0.put(0, n as u32);
        Some(())
    }
}

/// A level's code number (`levelCode`) with suffix length `sl`: level_prefix and level_suffix.
fn level<I: Io>(io: &mut I, code: u32, sl: u32) -> Option<u32> {
    if io.reading() {
        let prefix = io.zeros(0)?;
        if prefix > 28 { return None; }
        let size = if prefix == 14 && sl == 0 { 4 } else if prefix >= 15 { prefix - 3 } else { sl };
        let mut code = (prefix.min(15) << sl) + io.u(0, size)?;
        if prefix >= 15 && sl == 0 { code += 15; }
        if prefix >= 16 { code += (1 << (prefix - 3)) - 4096; }
        return Some(code);
    }
    // the escape codes start where the plain ones end
    let base = if sl == 0 { 30 } else { 15 << sl };
    let (prefix, size, suffix) = if sl == 0 && code < 14 {
        (code, 0, 0)
    } else if sl == 0 && code < 30 {
        (14, 4, code - 14)
    } else if code < base {
        (code >> sl, sl, code & ((1 << sl) - 1))
    } else if code - base < 4096 {
        (15, 12, code - base)
    } else {
        let rest = (code - base) as u64;
        let p = (16..=28u32).find(|&p| rest < (2u64 << (p - 3)) - 4096)?;
        (p, p - 3, (rest - ((1u64 << (p - 3)) - 4096)) as u32)
    };
    io.zeros(prefix)?;
    io.u(suffix, size)?;
    Some(code)
}

/// One residual block: its coefficients `coef` in scan order (as many as the block can hold),
/// with `nc` choosing the coeff_token table (-1: a chroma DC block). Returns how many are not
/// zero. A reader expects `coef` zeroed.
fn block<I: Io>(io: &mut I, coef: &mut [i32], nc: i32) -> Option<usize> {
    let max = coef.len();
    let reading = io.reading();
    // the coefficients from the highest frequency down, and the zeros before each
    let (mut lev, mut run) = ([0i32; 16], [0usize; 16]);
    let (mut total, mut ones, mut zeros) = (0usize, 0usize, 0usize);
    if !reading {
        let mut above = None;
        for i in (0..max).rev() {
            if coef[i] == 0 { continue; }
            if let Some(a) = above { run[total - 1] = a - i - 1; } else { zeros = i + 1; }
            lev[total] = coef[i];
            above = Some(i);
            total += 1;
        }
        zeros -= total.min(zeros);
        ones = lev[..total.min(3)].iter().take_while(|v| v.abs() == 1).count();
    }
    let table = match nc { -1 => 4, 0..=1 => 0, 2..=3 => 1, 4..=7 => 2, _ => 3 };
    let token = io.vlc(Code::Token(table), total * 4 + ones)?;
    (total, ones) = (token / 4, token % 4);
    if total > max { return None; }
    if total == 0 { return Some(0); }
    for v in lev[..ones].iter_mut() { *v = if io.u((*v < 0) as u32, 1)? == 1 { -1 } else { 1 }; }
    let mut sl = (total > 10 && ones < 3) as u32;
    for i in ones..total {
        // the first level after fewer than three trailing ones cannot be ±1
        let shift = if i == ones && ones < 3 { 2 } else { 0 };
        let code = if reading { 0 } else { (if lev[i] > 0 { 2 * lev[i] as i64 - 2 } else { -2 * lev[i] as i64 - 1 } as u32).checked_sub(shift)? };
        let code = level(io, code, sl)?.checked_add(shift)?;
        if reading { lev[i] = if code & 1 == 0 { (code as i64 + 2) >> 1 } else { (-(code as i64) - 1) >> 1 } as i32; }
        if sl == 0 { sl = 1; }
        if lev[i].unsigned_abs() > 3 << (sl - 1) && sl < 6 { sl += 1; }
    }
    if total < max {
        zeros = io.vlc(if nc == -1 { Code::ZerosDc(total) } else { Code::Zeros(total) }, zeros)?;
        if zeros > max - total { return None; }
    }
    let mut left = zeros;
    for r in run[..total - 1].iter_mut() {
        if left == 0 {
            *r = 0;
        } else {
            *r = io.vlc(Code::Run(left.min(7)), *r)?;
            left = left.checked_sub(*r)?;
        }
    }
    if reading {
        let mut at = total + zeros - 1;
        for i in 0..total {
            coef[at] = lev[i];
            if i + 1 < total { at -= 1 + run[i]; }
        }
    }
    Some(total)
}

/// The macroblock layer of a CAVLC slice, read or written.
struct Vlc<'a> {
    h: &'a SliceHeader,
    sps: &'a Sps,
    pps: &'a Pps,
    /// Coefficients per 4x4 block (16 luma in raster order, 4 Cb, 4 Cr) of the picture's
    /// macroblocks: those of this slice decoded so far are read.
    tc: &'a mut [[u8; 24]],
    addr: usize,
}

impl Vlc<'_> {
    /// The coeff_token table context of block `b` (luma 0..16 in raster order, chroma from 16):
    /// from the coefficient counts of the blocks left of and above it.
    fn nc(&self, b: usize) -> i32 {
        let w = self.sps.width_mbs;
        let left = (self.addr % w != 0 && self.addr > self.h.first_mb).then(|| &self.tc[self.addr - 1]);
        let top = (self.addr >= self.h.first_mb + w).then(|| &self.tc[self.addr - w]);
        let cur = &self.tc[self.addr];
        let (a, t) = if b < 16 {
            (if b % 4 > 0 { Some(cur[b - 1]) } else { left.map(|m| m[b + 3]) }, if b >= 4 { Some(cur[b - 4]) } else { top.map(|m| m[b + 12]) })
        } else {
            let k = (b - 16) % 4;
            (if k % 2 > 0 { Some(cur[b - 1]) } else { left.map(|m| m[b + 1]) }, if k >= 2 { Some(cur[b - 2]) } else { top.map(|m| m[b + 2]) })
        };
        match (a, t) {
            (Some(a), Some(t)) => (a as i32 + t as i32 + 1) >> 1,
            (Some(n), None) | (None, Some(n)) => n as i32,
            (None, None) => 0,
        }
    }

    /// One macroblock that is not skipped: `s` is filled by a reader (which expects it cleared),
    /// written from by a writer; `pcm`: the samples of a PCM macroblock.
    fn mb<I: Io>(&mut self, io: &mut I, s: &mut MbSyn, pcm: &mut [u8; 384]) -> Option<()> {
        let st = self.h.slice_type;
        let chroma = self.sps.chroma_format != 0;
        let reading = io.reading();
        // P_8x8ref0 is type 4 of a P slice: here, type 3 with a flag
        let ty = io.ue(if st == SLICE_P && s.ref0 { 4 } else { s.mb_type as u32 })?;
        if reading {
            s.ref0 = st == SLICE_P && ty == 4;
            s.mb_type = u8::try_from(if s.ref0 { 3 } else { ty }).ok()?;
        }
        let ty = s.mb_type as u32;
        // the inter type, or the intra type (0 NxN, 1..=24 16x16, 25 PCM)
        let first_intra = match st { SLICE_I => 0, SLICE_P => 5, _ => 23 };
        let (inter, intra) = if ty < first_intra { (Some(ty), 0) } else { (None, ty - first_intra) };
        if intra > 25 { return None; }
        self.tc[self.addr] = [0; 24];
        if inter.is_none() && intra == 25 {
            if self.sps.depth > 0 { return None; }
            io.align()?;
            for b in pcm[..if chroma { 384 } else { 256 }].iter_mut() { *b = io.u(*b as u32, 8)? as u8; }
            self.tc[self.addr] = [16; 24];
            return Some(());
        }
        let multi = [self.h.num_ref_idx[0] > 1, self.h.num_ref_idx[1] > 1];
        let (mut nref, mut nmvd) = (0usize, 0usize);
        let mut may_8x8 = true;
        {
            let mut reference = |io: &mut I, s: &mut MbSyn, list: usize| -> Option<()> {
                if !multi[list] { return Some(()); }
                let v = io.te(*s.refs.get(nref)? as u32, self.h.num_ref_idx[list] - 1)?;
                s.refs[nref] = v as u8;
                nref += 1;
                Some(())
            };
            let mut vectors = |io: &mut I, s: &mut MbSyn, n: usize| -> Option<()> {
                for _ in 0..n {
                    let v = s.mvd.get_mut(nmvd)?;
                    *v = [io.se(v[0])?, io.se(v[1])?];
                    nmvd += 1;
                }
                Some(())
            };
            // partitions of a sub-macroblock by its shape (8x8, 8x4, 4x8, 4x4)
            const PARTS: [usize; 4] = [1, 2, 2, 4];
            match (st, inter) {
                (_, None) => {}
                (SLICE_P, Some(t)) => match t {
                    0 => {
                        reference(io, s, 0)?;
                        vectors(io, s, 1)?;
                    }
                    1 | 2 => {
                        reference(io, s, 0)?;
                        reference(io, s, 0)?;
                        vectors(io, s, 2)?;
                    }
                    _ => {
                        for k in 0..4 { s.sub[k] = io.ue(s.sub[k] as u32).filter(|&v| v < 4)? as u8; }
                        if !s.ref0 {
                            for _ in 0..4 { reference(io, s, 0)?; }
                        }
                        for k in 0..4 { vectors(io, s, PARTS[s.sub[k] as usize])?; }
                        may_8x8 = s.sub.iter().all(|&v| v == 0);
                    }
                },
                (_, Some(t)) => match t {
                    0 => may_8x8 = self.sps.direct_8x8_inference,
                    1..=3 => {
                        let uses = [t != 2, t != 1];
                        for list in 0..2 { if uses[list] { reference(io, s, list)?; } }
                        for list in 0..2 { if uses[list] { vectors(io, s, 1)?; } }
                    }
                    4..=21 => {
                        let pred = B_PRED[(t as usize - 4) / 2];
                        for list in 0..2 { for p in pred { if p >> list & 1 == 1 { reference(io, s, list)?; } } }
                        for list in 0..2 { for p in pred { if p >> list & 1 == 1 { vectors(io, s, 1)?; } } }
                    }
                    22 => {
                        for k in 0..4 { s.sub[k] = io.ue(s.sub[k] as u32).filter(|&v| v < 13)? as u8; }
                        for list in 0..2 { for k in 0..4 { if B_SUB[s.sub[k] as usize].0 >> list & 1 == 1 { reference(io, s, list)?; } } }
                        for list in 0..2 {
                            for k in 0..4 {
                                let (lists, shape) = B_SUB[s.sub[k] as usize];
                                if lists >> list & 1 == 1 { vectors(io, s, PARTS[shape as usize])?; }
                            }
                        }
                        may_8x8 = s.sub.iter().all(|&v| if v == 0 { self.sps.direct_8x8_inference } else { B_SUB[v as usize].1 == 0 });
                    }
                    _ => return None,
                },
            }
        }
        (s.nref, s.nmvd) = (nref as u8, nmvd as u8);
        let i16 = inter.is_none() && intra != 0;
        if inter.is_none() {
            if intra == 0 {
                if self.pps.transform_8x8 { s.t8x8 = io.u(s.t8x8 as u32, 1)? == 1; }
                for k in 0..if s.t8x8 { 4 } else { 16 } {
                    let predicted = io.u((s.ipred[k] == PREDICTED) as u32, 1)? == 1;
                    s.ipred[k] = if predicted { PREDICTED } else { io.u(s.ipred[k] as u32 & 7, 3)? as u8 };
                }
            }
            if chroma { s.chroma_pred = io.ue(s.chroma_pred as u32).filter(|&v| v < 4)? as u8; }
        }
        if i16 {
            let k = intra - 1;
            s.cbp = if k >= 12 { 15 } else { 0 } | ((k / 4 % 3) as u8) << 4;
        } else {
            // the coded block pattern as a code number
            let table: &[u8] = match (chroma, inter.is_none()) { (true, true) => &CBP_INTRA, (true, false) => &CBP_INTER, (false, true) => &CBP_INTRA_GRAY, (false, false) => &CBP_INTER_GRAY };
            let code = if reading { 0 } else { table.iter().position(|&c| c == s.cbp)? as u32 };
            s.cbp = *table.get(io.ue(code)? as usize)?;
            if inter.is_some() && s.cbp & 15 != 0 && self.pps.transform_8x8 && may_8x8 { s.t8x8 = io.u(s.t8x8 as u32, 1)? == 1; }
        }
        if s.cbp == 0 && !i16 { return Some(()); }
        s.dqp = io.se(s.dqp)?;
        // the residual
        if i16 { block(io, &mut s.dc, self.nc(0))?; }
        for b8 in 0..4 {
            if s.cbp >> b8 & 1 == 0 { continue; }
            for k in 0..4 {
                let b = BLK[b8 * 4 + k];
                let nc = self.nc(b);
                self.tc[self.addr][b] = block(io, &mut s.luma[b][i16 as usize..], nc)? as u8;
            }
        }
        if chroma && s.cbp >> 4 != 0 {
            for c in 0..2 { block(io, &mut s.cdc[c], -1)?; }
        }
        if chroma && s.cbp >> 4 == 2 {
            for c in 0..2 {
                for k in 0..4 {
                    let nc = self.nc(16 + 4 * c + k);
                    self.tc[self.addr][16 + 4 * c + k] = block(io, &mut s.cac[c][k][1..], nc)? as u8;
                }
            }
        }
        Some(())
    }
}

// ------------------------------------------------------------------ syntax elements as bins
/// The bins CABAC would code a level's magnitude and sign with.
fn level_bins(w: &mut BitWriter, v: i32) {
    let a = v.unsigned_abs();
    w.put((a > 1) as u32, 1);
    if a > 1 {
        if a < 15 {
            w.put((1 << (a - 2)) - 1, a - 2);
            w.put(0, 1);
        } else {
            w.put((1 << 13) - 1, 13);
            exp_golomb(w, a - 15, 0);
        }
    }
    w.put((v < 0) as u32, 1);
}

/// Exp-Golomb code of order `k` as CABAC's bypass bins: ones while the value is taken down by
/// growing powers of two, a zero, the rest.
fn exp_golomb(w: &mut BitWriter, mut s: u32, mut k: u32) {
    while s >= 1 << k {
        w.put(1, 1);
        s -= 1 << k;
        k += 1;
    }
    w.put(0, 1);
    w.put(s, k);
}

/// The bins of a residual block: coded_block_flag (when it has one), the significance map, the
/// levels from the last one back. None for a block CABAC cannot express: one without a flag
/// and without coefficients.
fn block_bins(w: &mut BitWriter, coef: &[i32], flag: bool) -> Option<()> {
    let last = coef.iter().rposition(|&c| c != 0);
    if flag { w.put(last.is_some() as u32, 1); }
    let Some(last) = last else { return (flag).then_some(()) };
    for (i, &c) in coef[..coef.len() - 1].iter().enumerate() {
        w.put((c != 0) as u32, 1);
        if c != 0 {
            w.put((i == last) as u32, 1);
            if i == last { break; }
        }
    }
    for &c in coef[..=last].iter().rev().filter(|&&c| c != 0) {
        if c.unsigned_abs() > 1 << 20 { return None; }
        level_bins(w, c);
    }
    Some(())
}

/// An intra macroblock type (0 NxN, 1..=24 16x16, 25 PCM) after what its slice type puts first.
fn intra_type_bins(w: &mut BitWriter, t: u32) {
    w.put((t != 0) as u32, 1);
    if t == 0 { return; }
    w.put((t == 25) as u32, 1);
    if t == 25 { return; }
    let k = t - 1;
    w.put((k >= 12) as u32, 1);
    let c = k / 4 % 3;
    w.put((c != 0) as u32, 1);
    if c != 0 { w.put((c == 2) as u32, 1); }
    w.put(k % 4, 2);
}

fn unary(w: &mut BitWriter, v: u32) {
    for _ in 0..v { w.put(1, 1); }
    w.put(0, 1);
}

/// The bins of a macroblock, in the order and form [`walk`] reads them (CABAC's binarizations).
/// None for what CABAC cannot express, or [`walk`] does not accept.
fn binarize(s: &MbSyn, pcm: &[u8; 384], h: &SliceHeader, sps: &Sps, pps: &Pps, w: &mut BitWriter) -> Option<()> {
    let st = h.slice_type;
    let chroma = sps.chroma_format != 0;
    if st != SLICE_I { w.put(s.skip as u32, 1); }
    if s.skip {
        w.put(s.end as u32, 1);
        return Some(());
    }
    let ty = s.mb_type as u32;
    let first_intra = match st { SLICE_I => 0, SLICE_P => 5, _ => 23 };
    let (inter, intra) = if ty < first_intra { (Some(ty), 0) } else { (None, ty - first_intra) };
    // the type
    match (st, inter) {
        (SLICE_I, _) => intra_type_bins(w, intra),
        (SLICE_P, None) => {
            w.put(1, 1);
            intra_type_bins(w, intra);
        }
        (SLICE_P, Some(t)) => w.put([0b000, 0b011, 0b010, 0b001][t as usize], 3),
        (_, None) => {
            w.put(0b111101, 6);
            intra_type_bins(w, intra);
        }
        (_, Some(t)) => match t {
            0 => w.put(0, 1),
            1 | 2 => w.put(0b100 | (t - 1), 3),
            3..=10 => w.put(0b110000 | (t - 3), 6),
            11 => w.put(0b111110, 6),
            22 => w.put(0b111111, 6),
            _ => w.put(0b1100000 | (t + 4), 7),
        },
    }
    if inter.is_none() && intra == 25 {
        // the samples follow the bins, after padding and the (empty) alignment
        w.align();
        w.extend(&[0]);
        w.extend(&pcm[..if chroma { 384 } else { 256 }]);
        w.put(s.end as u32, 1);
        return Some(());
    }
    let multi = [h.num_ref_idx[0] > 1, h.num_ref_idx[1] > 1];
    let (mut nref, mut nmvd) = (0usize, 0usize);
    let mut may_8x8 = true;
    {
        let mut reference = |w: &mut BitWriter, list: usize| -> Option<()> {
            if !multi[list] { return Some(()); }
            // a P_8x8ref0 macroblock's references are zero, and here they are written
            let v = if s.ref0 { 0 } else { *s.refs.get(nref)? as u32 };
            nref += 1;
            if v >= 32 { return None; }
            unary(w, v);
            Some(())
        };
        let mut vectors = |w: &mut BitWriter, n: usize| -> Option<()> {
            for _ in 0..n {
                for v in *s.mvd.get(nmvd)? {
                    let a = v.unsigned_abs();
                    if a > 1 << 20 { return None; }
                    w.put((1 << a.min(9)) - 1, a.min(9));
                    if a < 9 { w.put(0, 1); } else { exp_golomb(w, a - 9, 3); }
                    if a != 0 { w.put((v < 0) as u32, 1); }
                }
                nmvd += 1;
            }
            Some(())
        };
        const PARTS: [usize; 4] = [1, 2, 2, 4];
        match (st, inter) {
            (_, None) => {}
            (SLICE_P, Some(t)) => match t {
                0 => {
                    reference(w, 0)?;
                    vectors(w, 1)?;
                }
                1 | 2 => {
                    reference(w, 0)?;
                    reference(w, 0)?;
                    vectors(w, 2)?;
                }
                _ => {
                    for &v in &s.sub {
                        match v {
                            0 => w.put(1, 1),
                            1 => w.put(0b00, 2),
                            2 => w.put(0b011, 3),
                            _ => w.put(0b010, 3),
                        }
                    }
                    for _ in 0..4 { reference(w, 0)?; }
                    for &v in &s.sub { vectors(w, PARTS[v as usize & 3])?; }
                    may_8x8 = s.sub.iter().all(|&v| v == 0);
                }
            },
            (_, Some(t)) => match t {
                0 => may_8x8 = sps.direct_8x8_inference,
                1..=3 => {
                    let uses = [t != 2, t != 1];
                    for list in 0..2 { if uses[list] { reference(w, list)?; } }
                    for list in 0..2 { if uses[list] { vectors(w, 1)?; } }
                }
                4..=21 => {
                    let pred = B_PRED[(t as usize - 4) / 2];
                    for list in 0..2 { for p in pred { if p >> list & 1 == 1 { reference(w, list)?; } } }
                    for list in 0..2 { for p in pred { if p >> list & 1 == 1 { vectors(w, 1)?; } } }
                }
                _ => {
                    for &v in &s.sub {
                        match v {
                            0 => w.put(0, 1),
                            1 | 2 => w.put(0b100 | (v as u32 - 1), 3),
                            3..=6 => w.put(0b11000 | (v as u32 - 3), 5),
                            7..=10 => w.put(0b111000 | (v as u32 - 7), 6),
                            _ => w.put(0b11110 | (v as u32 - 11), 5),
                        }
                    }
                    for list in 0..2 { for &v in &s.sub { if B_SUB[v as usize].0 >> list & 1 == 1 { reference(w, list)?; } } }
                    for list in 0..2 {
                        for &v in &s.sub {
                            let (lists, shape) = B_SUB[v as usize];
                            if lists >> list & 1 == 1 { vectors(w, PARTS[shape as usize])?; }
                        }
                    }
                    may_8x8 = s.sub.iter().all(|&v| if v == 0 { sps.direct_8x8_inference } else { B_SUB[v as usize].1 == 0 });
                }
            },
        }
    }
    let i16 = inter.is_none() && intra != 0;
    let pattern = |w: &mut BitWriter| {
        w.put((0..4).fold(0, |m, b8| m << 1 | (s.cbp as u32 >> b8 & 1)), 4);
        if chroma {
            let c = s.cbp as u32 >> 4;
            w.put((c != 0) as u32, 1);
            if c != 0 { w.put((c == 2) as u32, 1); }
        }
    };
    if inter.is_none() {
        if intra == 0 {
            if pps.transform_8x8 { w.put(s.t8x8 as u32, 1); }
            for &m in &s.ipred[..if s.t8x8 { 4 } else { 16 }] {
                w.put((m == PREDICTED) as u32, 1);
                // the remaining mode, lowest bit first
                if m != PREDICTED { w.put((m as u32 & 1) << 2 | (m as u32 & 2) | (m as u32 >> 2 & 1), 3); }
            }
        }
        if chroma { w.put([0b0, 0b10, 0b110, 0b111][s.chroma_pred as usize & 3], [1, 2, 3, 3][s.chroma_pred as usize & 3]); }
        if !i16 { pattern(w); }
    } else {
        pattern(w);
        if s.cbp & 15 != 0 && pps.transform_8x8 && may_8x8 { w.put(s.t8x8 as u32, 1); }
    }
    if s.cbp != 0 || i16 {
        // the quantiser change: 1, -1, 2, -2 … counted in unary
        let v = if s.dqp > 0 { 2 * s.dqp as i64 - 1 } else { -2 * s.dqp as i64 };
        if v > 104 { return None; }
        unary(w, v as u32);
        if i16 { block_bins(w, &s.dc, true)?; }
        for b8 in 0..4 {
            if s.cbp >> b8 & 1 == 0 { continue; }
            if s.t8x8 {
                // an 8x8 block: its coefficients are dealt out to the four 4x4 blocks in turn
                let mut c = [0i32; 64];
                for (i, v) in c.iter_mut().enumerate() { *v = s.luma[BLK[b8 * 4 + (i & 3)]][i >> 2]; }
                block_bins(w, &c, false)?;
            } else {
                for k in 0..4 { block_bins(w, &s.luma[BLK[b8 * 4 + k]][i16 as usize..], true)?; }
            }
        }
        if chroma && s.cbp >> 4 != 0 {
            for c in 0..2 { block_bins(w, &s.cdc[c], true)?; }
        }
        if chroma && s.cbp >> 4 == 2 {
            for c in 0..2 { for k in 0..4 { block_bins(w, &s.cac[c][k][1..], true)?; } }
        }
    }
    w.put(s.end as u32, 1);
    Some(())
}

// ------------------------------------------------------------------ slices
/// A CAVLC slice as bins.
pub struct Slice {
    /// The packed bins (as [`walk`] reads them; see `Recorder`).
    pub bins: Vec<u8>,
    pub macroblocks: usize,
    /// The macroblocks of type P_8x8ref0 (by address): CABAC has no such type, so the bins hold
    /// them as P_8x8 with their references (all zero) written out.
    pub ref0: Vec<u32>,
}

/// Take the CAVLC slice with RBSP `rbsp` and header `h` apart. None when it does not parse, or
/// uses what the bins cannot express: an 8x8 block flagged as coded but without coefficients,
/// values beyond what [`walk`] accepts, anything but zero bits after the stop bit.
pub fn to_bins(rbsp: &[u8], h: &SliceHeader, sps: &Sps, pps: &Pps, scratch: &mut Scratch) -> Option<Slice> {
    let total = sps.width_mbs * sps.height_mbs;
    if !sps.supported || !pps.supported || total > MAX_MBS || h.first_mb >= total { return None; }
    if scratch.tc.len() < total { scratch.tc.resize(total, [0; 24]); }
    // the slice data ends at the stop bit: the last bit set, in the last byte
    let end = rbsp.len() * 8 - 1 - rbsp.last().filter(|&&x| x != 0)?.trailing_zeros() as usize;
    if end < h.data { return None; }
    let mut io = Rd { r: BitReader::new(rbsp, h.data), end };
    let mut v = Vlc { h, sps, pps, tc: &mut scratch.tc[..total], addr: h.first_mb };
    let mut out = Slice { bins: vec![], macroblocks: 0, ref0: vec![] };
    let mut w = BitWriter::new();
    let mut s = Box::new(MbSyn::default());
    let mut pcm = [0u8; 384];
    loop {
        if h.slice_type != SLICE_I {
            let run = io.ue(0)? as usize;
            if run > total - v.addr { return None; }
            for k in 0..run {
                *s = MbSyn { skip: true, end: k + 1 == run && io.r.pos() >= end, ..Default::default() };
                v.tc[v.addr] = [0; 24];
                binarize(&s, &pcm, h, sps, pps, &mut w)?;
                v.addr += 1;
            }
            out.macroblocks += run;
            if run > 0 && io.r.pos() >= end { break; }
        }
        if v.addr >= total { return None; }
        *s = MbSyn::default();
        v.mb(&mut io, &mut s, &mut pcm)?;
        s.end = io.r.pos() >= end;
        if s.ref0 { out.ref0.push(v.addr as u32); }
        binarize(&s, &pcm, h, sps, pps, &mut w)?;
        v.addr += 1;
        out.macroblocks += 1;
        if s.end { break; }
    }
    if io.r.pos() != end { return None; }
    out.bins = w.into_bytes();
    Some(out)
}

/// Reads packed bins for [`walk`] and writes each macroblock it makes of them as CAVLC.
struct Transcoder<'a, 'b> {
    r: BitReader<'a>,
    limit: usize,
    short: bool,
    v: Vlc<'a>,
    w: &'b mut BitWriter,
    /// Skipped macroblocks not written yet.
    run: u32,
    pcm: [u8; 384],
    ref0: std::slice::Iter<'a, u32>,
    failed: bool,
}

impl Bins for Transcoder<'_, '_> {
    fn decision(&mut self, _: usize, _: &Info) -> u32 { self.r.get(1) }
    fn bypass(&mut self, _: &Info) -> u32 { self.r.get(1) }
    fn terminate(&mut self, _: &Info) -> u32 {
        // past the end of the bins (a damaged record), end the slice at once
        if self.r.pos() >= self.limit {
            self.short = true;
            return 1;
        }
        self.r.get(1)
    }
    fn pcm(&mut self, n: usize) -> Result<(), String> {
        self.r.seek(self.r.pos().div_ceil(8) * 8);
        if self.r.pos() + 8 * (n + 1) > self.limit || self.r.get(8) != 0 { return Err("h264: bad PCM bytes".into()); }
        for b in self.pcm[..n].iter_mut() { *b = self.r.get(8) as u8; }
        Ok(())
    }
    fn collect(&self) -> bool { true }
    fn macroblock(&mut self, addr: usize, s: &mut MbSyn) -> Result<(), String> {
        let mut io = Wr(&mut *self.w);
        let mut done = || -> Option<()> {
            if s.skip {
                self.run += 1;
                self.v.tc[addr] = [0; 24];
            } else {
                if self.v.h.slice_type != SLICE_I { io.ue(std::mem::take(&mut self.run))?; }
                if self.ref0.as_slice().first() == Some(&(addr as u32)) {
                    self.ref0.next();
                    if self.v.h.slice_type != SLICE_P || s.mb_type != 3 || s.refs[..s.nref as usize].iter().any(|&r| r != 0) { return None; }
                    s.ref0 = true;
                }
                self.v.addr = addr;
                self.v.mb(&mut io, s, &mut self.pcm)?;
            }
            if s.end && self.run > 0 { io.ue(std::mem::take(&mut self.run))?; }
            Some(())
        };
        if done().is_none() { self.failed = true; }
        Ok(())
    }
}

/// Write the slice data of a CAVLC slice from its bins (`bins` starts with them) after the
/// header bits already in `w`, through the stop bit and the padding after it. Returns how many
/// bytes of `bins` the slice took.
pub fn from_bins(h: &SliceHeader, bins: &[u8], ref0: &[u32], sps: &Sps, pps: &Pps, scratch: &mut Scratch, w: &mut BitWriter) -> Result<usize, String> {
    let bad = || "h264: bins do not make a CAVLC slice".to_string();
    let total = sps.width_mbs * sps.height_mbs;
    if !sps.supported || !pps.supported || total > MAX_MBS || h.first_mb >= total { return Err(bad()); }
    let mut tc = std::mem::take(&mut scratch.tc);
    if tc.len() < total { tc.resize(total, [0; 24]); }
    let mut t = Transcoder {
        r: BitReader::new(bins, 0), limit: bins.len() * 8, short: false, v: Vlc { h, sps, pps, tc: &mut tc[..total], addr: h.first_mb }, w, run: 0, pcm: [0; 384],
        ref0: ref0.iter(), failed: false,
    };
    let walked = walk(&mut t, h, sps, pps, scratch);
    let ok = !t.failed && !t.short && t.r.pos() <= t.limit && t.ref0.as_slice().is_empty();
    let used = t.r.pos().div_ceil(8);
    scratch.tc = tc;
    walked?;
    if !ok { return Err(bad()); }
    w.put(1, 1);
    w.align();
    Ok(used)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }

    /// What a writer wrote, read back with the same calls.
    fn reread<T>(write: impl Fn(&mut Wr) -> Option<T>, read: impl Fn(&mut Rd) -> Option<T>) -> (Option<T>, Option<T>, bool) {
        let mut w = BitWriter::new();
        let a = write(&mut Wr(&mut w));
        let end = w.len();
        let bytes = w.into_bytes();
        let mut r = Rd { r: BitReader::new(&bytes, 0), end };
        let b = read(&mut r);
        (a, b, r.r.pos() == end)
    }

    #[test]
    fn numbers_and_levels_read_back() {
        for v in (0..2000u32).chain([65534, 65535, 1 << 20, u32::MAX - 1]) {
            let (a, b, whole) = reread(|w| w.ue(v), |r| r.ue(0));
            assert!(a == Some(v) && b == Some(v) && whole, "ue {v}");
        }
        for v in (-1500..1500i32).chain([i32::MAX / 2, -(i32::MAX / 2)]) {
            let (a, b, whole) = reread(|w| w.se(v), |r| r.se(0));
            assert!(a == Some(v) && b == Some(v) && whole, "se {v}");
        }
        for max in [1u32, 2, 5, 31] {
            for v in 0..=max {
                let (a, b, whole) = reread(|w| w.te(v, max), |r| r.te(0, max));
                assert!(a == Some(v) && b == Some(v) && whole, "te {v} of {max}");
            }
        }
        // every level code with every suffix length, through the escapes
        for sl in 0..=6 {
            for code in (0..9000u32).chain([12000, 20000, 50000, 200_000, 1 << 21]) {
                let (a, b, whole) = reread(|w| level(w, code, sl), |r| level(r, 0, sl));
                assert!(a == Some(code) && b == Some(code) && whole, "level code {code}, suffix length {sl}: {a:?} {b:?}");
            }
        }
    }

    #[test]
    fn blocks_read_back() {
        let mut s = 21u64;
        for round in 0..20_000 {
            // 4, 15 or 16 coefficients, sparse or dense, small or large
            let max = [4usize, 15, 16][(rng(&mut s) % 3) as usize];
            let nc = if max == 4 { -1 } else { (rng(&mut s) % 18) as i32 };
            let dense = rng(&mut s) % 4;
            let mut coef = vec![0i32; max];
            for c in coef.iter_mut() {
                if rng(&mut s) % 4 <= dense {
                    let big = if round % 50 == 0 { 40_000 } else if round % 5 == 0 { 300 } else { 4 };
                    *c = (rng(&mut s) % (2 * big + 1)) as i32 - big as i32;
                }
            }
            let (a, b, whole) = reread(|w| block(w, &mut coef.clone(), nc), |r| {
                let mut back = vec![0i32; max];
                block(r, &mut back, nc)?;
                Some(back.iter().filter(|&&c| c != 0).count()).filter(|_| back == coef)
            });
            let n = coef.iter().filter(|&&c| c != 0).count();
            assert!(a == Some(n) && b == Some(n) && whole, "round {round}: {coef:?} with nC {nc}: wrote {a:?}, read {b:?}");
        }
    }

    #[test]
    fn random_bits_are_parsed_or_refused() {
        // CAVLC slices of random bits: taken apart (then they must come back) or refused
        let (_, _, _, config) = crate::recode::h264::tests::track(crate::recode::h264::tests::FIXTURES[2]);
        let mut ps = crate::recode::h264::Params::default();
        for n in &config { assert!(ps.add(n)); }
        let (sps, pps) = ps.of(0).unwrap();
        assert!(!pps.cabac);
        let mut scratch = Scratch::default();
        let mut s = 77u64;
        let mut taken = 0;
        for round in 0..4000 {
            let mut rbsp: Vec<u8> = (0..3 + rng(&mut s) % 60).map(|_| rng(&mut s) as u8).collect();
            *rbsp.last_mut().unwrap() |= 1;
            for slice_type in [SLICE_I, SLICE_P] {
                let h = SliceHeader { first_mb: (round % 7) as usize, slice_type, num_ref_idx: [1 + (round % 3) as u32, 1], qp: 26, data: 3, ..Default::default() };
                if let Some(sl) = to_bins(&rbsp, &h, &sps, &pps, &mut scratch) {
                    let mut w = BitWriter::new();
                    w.put((rbsp[0] >> 5) as u32, 3);
                    let used = from_bins(&h, &sl.bins, &sl.ref0, &sps, &pps, &mut scratch, &mut w).unwrap();
                    assert!(used == sl.bins.len() && w.into_bytes() == rbsp, "round {round}: a slice of random bits did not come back");
                    taken += 1;
                }
            }
        }
        assert!(taken > 20, "{taken} random slices parsed");
    }
}
