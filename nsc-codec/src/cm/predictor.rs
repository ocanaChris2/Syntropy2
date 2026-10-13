//! The bit predictor: generic byte contexts + match model → mixer → SSE chain.
//!
//! [`code_bytes`] drives a [`Predictor`] and a [`Coder`] over a byte range. For each byte it calls
//! `byte_update` with the history (everything before the byte), then for each bit, MSB first:
//! `p()`, code the bit, `update(bit)`. The encoder and decoder run this one function, so they see
//! identical predictions by construction.
use super::coder::Coder;
use super::contextmap::ContextMap;
use super::hash::{hash, HashTable};
use super::levels::{ctx, sel, ModelCfg};
use super::matcher::MatchModel;
use super::mixer::Mixer;
use super::statemap::Apm;

/// Byte storage the coding loop reads history from and writes decoded bytes into.
pub trait ByteBuf {
    fn hist(&self, end: usize) -> &[u8];
    fn byte(&self, i: usize) -> u8;
    fn put(&mut self, i: usize, b: u8);
    /// True on the encoder side (the bytes are already there).
    fn reading(&self) -> bool;
}

/// Encoder side: the input is read-only (writes are no-ops; the byte is already there).
pub struct ReadBuf<'a>(pub &'a [u8]);
impl ByteBuf for ReadBuf<'_> {
    #[inline] fn hist(&self, end: usize) -> &[u8] { &self.0[..end] }
    #[inline] fn byte(&self, i: usize) -> u8 { self.0[i] }
    #[inline] fn put(&mut self, _i: usize, _b: u8) {}
    #[inline] fn reading(&self) -> bool { true }
}

/// Decoder side: decoded bytes are written in place.
pub struct WriteBuf<'a>(pub &'a mut [u8]);
impl ByteBuf for WriteBuf<'_> {
    #[inline] fn hist(&self, end: usize) -> &[u8] { &self.0[..end] }
    #[inline] fn byte(&self, i: usize) -> u8 { self.0[i] }
    #[inline] fn put(&mut self, i: usize, b: u8) { self.0[i] = b; }
    #[inline] fn reading(&self) -> bool { false }
}

/// Code bytes `range` of `buf` with `pred` through `coder`.
pub fn code_bytes<C: Coder, B: ByteBuf>(pred: &mut Predictor, coder: &mut C, buf: &mut B, range: std::ops::Range<usize>) {
    for i in range {
        pred.byte_update(buf.hist(i));
        let input = buf.byte(i) as u32;
        let mut c = 0u32;
        for k in (0..8).rev() {
            let p = pred.p();
            let bit = coder.code((input >> k) & 1, p);
            pred.update(bit);
            c = (c << 1) | bit;
        }
        buf.put(i, c as u8);
    }
}

#[inline]
fn is_word_char(c: u32) -> bool { (c as u8).is_ascii_alphanumeric() || c >= 128 }

/// Per-byte text structure tracked for the text model: sentence position, XML tags, delimited
/// fields, line indentation, number tokens and bracket nesting.
#[derive(Default)]
struct TextState {
    sent_words: u32,
    sent_end: bool,
    in_tag: bool,
    tag: u64,
    tag_done: bool,
    elem: u64,
    field: u32,
    field_pos: u32,
    indent: u32,
    line_start: bool,
    num: u64,
    brackets: [u8; 16],
    depth: usize,
}

impl TextState {
    fn update(&mut self, c: u8, word_ended: bool) {
        if word_ended { self.sent_words += 1; }
        match c {
            b'.' | b'!' | b'?' => self.sent_end = true,
            b' ' | b'\n' | b'\t' | b'\r' if self.sent_end => { self.sent_words = 0; self.sent_end = false; }
            _ if !c.is_ascii_whitespace() => self.sent_end = false,
            _ => {}
        }
        if c == b'<' {
            self.in_tag = true;
            self.tag = 0;
            self.tag_done = false;
        } else if self.in_tag {
            if c == b'>' {
                self.in_tag = false;
                self.elem = self.tag;
            } else if !self.tag_done && (c.is_ascii_alphanumeric() || matches!(c, b':' | b'-' | b'_' | b'/' | b'!' | b'?')) {
                self.tag = (self.tag ^ c as u64).wrapping_mul(0x100_0000_01B3);
            } else {
                self.tag_done = true;
            }
        }
        match c {
            b'\n' => { self.field = 0; self.field_pos = 0; self.indent = 0; self.line_start = true; }
            b',' | b'\t' | b';' | b'|' => { self.field += 1; self.field_pos = 0; }
            _ => self.field_pos += 1,
        }
        if self.line_start {
            match c {
                b' ' => self.indent += 1,
                b'\t' => self.indent += 4,
                b'\n' => {}
                _ => self.line_start = false,
            }
        }
        if c.is_ascii_digit() || (c == b'.' && self.num != 0) {
            self.num = (self.num ^ c as u64).wrapping_mul(0x100_0000_01B3) | 1;
        } else {
            self.num = 0;
        }
        match c {
            b'(' | b'[' | b'{' => { if self.depth < 16 { self.brackets[self.depth] = c; } self.depth += 1; }
            b')' | b']' | b'}' => { if self.depth > 0 { self.depth -= 1; } }
            _ => {}
        }
    }
}

/// Raster geometry of the current image segment (from the detector's image params).
#[derive(Clone, Copy)]
struct ImgGeom {
    start: usize,  // block offset where this segment piece begins (always a row start)
    stride: usize, // bytes per row, including padding
    width: usize,  // pixels per row
    bpp: usize,    // bytes per pixel
    bps: usize,    // bytes per sample (1 or 2)
    be: bool,      // 16-bit samples are big-endian
}

impl ImgGeom {
    fn from_params(p: &[u8], start: usize) -> Option<ImgGeom> {
        use crate::detect::imgfmt::*;
        if p.len() < 11 { return None; }
        let fmt = p[0];
        let width = u32::from_le_bytes(p[1..5].try_into().ok()?) as usize;
        let stride = u32::from_le_bytes(p[5..9].try_into().ok()?) as usize;
        let ch = p[9] as usize;
        let (bps, be) = match fmt { GRAY16LE => (2, false), GRAY16BE | RGB48BE => (2, true), _ => (1, false) };
        let bpp = ch * bps;
        if width == 0 || bpp == 0 || stride < width * bpp || stride > 1 << 24 { return None; }
        Some(ImgGeom { start, stride, width, bpp, bps, be })
    }
}

#[inline]
fn clamp_px(v: i32, max: i32) -> i32 { v.clamp(0, max) }

/// Integer log2 bucket (0..=15) for texture measures.
#[inline]
fn qlog(v: i32) -> u64 { (32 - (v.max(0) as u32 + 1).leading_zeros()).min(15) as u64 }

/// Candidate hashed contexts of the second binary model (`ctx::BIN2`), beside its guessers
/// (`guess`). None is in use: together they help megabyte tables and hurt small sections, and
/// which of them earn their place has not been settled (`bench/bin_dev.py --ablate`).
pub const BIN2: u32 = 24;
/// Developer experiments: the candidates switched on (a bit each; `nsc cm-test --bin`).
pub static BIN2_KEEP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Mixer weight-set groups by segment kind (generic, text, fast, image, audio, exe).
const KGROUPS: usize = 6;

/// Number of image-model contexts (ids `ctx::I_WN..=ctx::I_WNQ`).
const NIMG: usize = (ctx::I_WNQ - ctx::I_WN + 1) as usize;

pub struct Predictor {
    ids: Vec<u32>,
    kgroup: usize,
    img: Option<ImgGeom>,
    /// Per-byte image summary for the mixer selectors and the image SSE stage: (predicted value
    /// (8-bit), activity bucket, channel, high-byte flag).
    img_sum: (u32, u32, u32, u32),
    /// The image contexts' values for the current byte ([`Predictor::image_ctxs`]).
    img_ctx: [u64; NIMG],
    a4: Option<Apm>,
    apm_img: bool,
    sels: Vec<u8>,
    ht: HashTable,
    cm: ContextMap,
    mm: MatchModel,
    mixer: Mixer,
    a1: Option<Apm>,
    a2: Option<Apm>,
    a3: Option<Apm>,
    c0: u32,
    bpos: u32,
    c4: u32,
    c8: u32,
    c12: u32,
    c16: u32,
    pos: usize,
    word: u64,
    word1: u64,
    word2: u64,
    txt: TextState,
    t1: Vec<u16>,
    t2: Vec<u16>,
    cpos: Vec<[u32; 4]>,
    rlen: usize,
    rlen1: usize,
    rlen2: usize,
    rcount1: u32,
    rcount2: u32,
    line_start: usize,
    prev_line_start: usize,
    o24: u64,
    a3cx: usize,
    /// In x86 code (when the x86 contexts are configured): the instruction decoder, and the
    /// block offset its segment starts at.
    x86: Option<Box<crate::x86::X86>>,
    x86_start: usize,
    has_x86: bool,
    /// The record length before the current one; where the last zero byte was (the position
    /// after it).
    rlen_prev: usize,
    last0: usize,
    /// The second binary model's guessers, and whether the segment is data for them.
    guess: Option<Box<super::guess::Guess>>,
    guess_on: bool,
    /// The number model of text, and whether the segment is text.
    num: Option<Box<super::num::Num>>,
    num_on: bool,
}

impl Predictor {
    pub fn new(cfg: &ModelCfg) -> Self {
        let mut p = Self::build(cfg);
        p.set_kind(crate::nsc3::format::kind::GENERIC);
        p
    }

    fn build(cfg: &ModelCfg) -> Self {
        // the instruction model's bit stands for all its contexts
        let has_x86 = cfg.ctx_mask >> ctx::X86 & 1 == 1;
        let mut ids: Vec<u32> = (0..ctx::COUNT).filter(|&i| !matches!(i, ctx::X86 | ctx::BIN2 | ctx::NUM) && cfg.ctx_mask >> i & 1 == 1).collect();
        if has_x86 { ids.extend((0..crate::x86::CONTEXTS).filter(|&i| crate::x86::kept(i)).map(|i| ctx::X_FIRST + i)); }
        let has_bin2 = cfg.ctx_mask >> ctx::BIN2 & 1 == 1;
        if has_bin2 {
            let keep = BIN2_KEEP.load(std::sync::atomic::Ordering::Relaxed);
            ids.extend((0..BIN2).filter(|&i| keep >> i & 1 == 1).map(|i| ctx::B_FIRST + i));
        }
        let has_num = cfg.ctx_mask >> ctx::NUM & 1 == 1;
        if has_num { ids.extend((0..super::num::CONTEXTS).filter(|&i| super::num::kept(i)).map(|i| ctx::N_FIRST + i)); }
        let sels: Vec<u8> = (0..5u8).filter(|s| cfg.sel_mask >> s & 1 == 1).collect();
        let sizes: Vec<usize> = sels.iter().map(|&s| match s {
            sel::C0 => 256 * KGROUPS,
            sel::C1 | sel::C2 => 512,
            sel::HITS => 256,
            sel::MATCH => 48,
            _ => 1,
        }).collect();
        let n_inputs = ids.len() * 2 + 2 + 1 + if has_bin2 { super::guess::GUESSERS } else { 0 };
        Predictor {
            cm: ContextMap::new(ids.len()),
            ids,
            mixer: Mixer::new(n_inputs, &sizes, 256 * KGROUPS, cfg.lr as i32, cfg.lr2.max(1) as i32),
            kgroup: 0,
            img: None,
            img_sum: (0, 0, 0, 1),
            img_ctx: [0; NIMG],
            a4: (cfg.ctx_mask >> ctx::I_MED & 1 == 1).then(|| Apm::new(65536, 7)),
            apm_img: false,
            sels,
            ht: HashTable::new(cfg.hash_bits as u32),
            mm: MatchModel::new(cfg.mm_bits as u32),
            a1: (cfg.apm_mask & 1 != 0).then(|| Apm::new(65536, 7)),
            a2: (cfg.apm_mask & 2 != 0).then(|| Apm::new(65536, 7)),
            a3: (cfg.apm_mask & 4 != 0).then(|| Apm::new(16 * 256 * 8 + 256, 7)),
            c0: 1,
            bpos: 0,
            c4: 0,
            c8: 0,
            c12: 0,
            c16: 0,
            pos: 0,
            word: 0,
            word1: 0,
            word2: 0,
            txt: TextState::default(),
            t1: vec![0; 256],
            t2: vec![0; 65536],
            cpos: vec![[0; 4]; 256],
            rlen: 0,
            rlen1: 0,
            rlen2: 0,
            rcount1: 0,
            rcount2: 0,
            line_start: 0,
            prev_line_start: 0,
            o24: 0,
            a3cx: 0,
            x86: None,
            x86_start: 0,
            has_x86,
            rlen_prev: 0,
            last0: 0,
            guess: has_bin2.then(|| Box::new(super::guess::Guess::new())),
            guess_on: false,
            num: has_num.then(|| Box::new(super::num::Num::new())),
            num_on: false,
        }
    }

    /// Switch the model to a segment kind (at a byte boundary, before its first byte). Each kind
    /// group trains its own first- and second-layer mixer weights; text segments swap the
    /// binary-only contexts for the text model.
    pub fn set_kind(&mut self, k: u8) { self.set_segment(k, &[], 0) }

    /// Switch to a segment of kind `k` with detector params `params`, starting at block offset
    /// `start` (image segments need their raster geometry and row origin).
    pub fn set_segment(&mut self, k: u8, params: &[u8], start: usize) {
        use crate::nsc3::format::kind;
        self.kgroup = match k { kind::TEXT => 1, kind::FAST => 2, kind::IMAGE => 3, kind::AUDIO => 4, kind::EXE => 5, _ => 0 };
        self.img = if k == kind::IMAGE { ImgGeom::from_params(params, start) } else { None };
        let image = self.img.is_some();
        let arch = params.first().map_or(0, |a| a & !crate::detect::EXE_RELOCATABLE);
        self.x86 = (self.has_x86 && k == kind::EXE && matches!(arch, 1 | 2)).then(|| Box::new(crate::x86::X86::new(arch == 2)));
        self.x86_start = start;
        self.guess_on = !image && !matches!(k, kind::TEXT | kind::EXE);
        self.num_on = k == kind::TEXT && self.num.is_some();
        for i in 0..self.ids.len() {
            let id = self.ids[i];
            // High-entropy (fast-kind) data keeps the full binary context set: compressed streams
            // still carry structure the higher orders find. It only trains its own mixer weights.
            // Image segments use the image model plus order-1/2 and the match model.
            let keep = if image {
                ctx::image_only(id) || matches!(id, ctx::O1 | ctx::O2 | ctx::MATCH)
            } else if ctx::x86_only(id) {
                self.x86.is_some()
            } else if ctx::bin2_only(id) {
                !matches!(k, kind::TEXT | kind::EXE)
            } else if ctx::num_only(id) {
                k == kind::TEXT
            } else {
                !ctx::image_only(id) && match k {
                    kind::TEXT => !ctx::binary_only(id),
                    _ => !ctx::text_only(id),
                }
            };
            self.cm.set_active(i, keep);
        }
    }

    /// Per-byte image summary (predicted value, activity, channel, high-byte flag) for selectors
    /// and the image SSE stage.
    fn image_summary(&self, hist: &[u8]) -> (u32, u32, u32, u32) {
        let Some(g) = self.img else { return (0, 0, 0, 1) };
        let p = hist.len();
        let off = p - g.start;
        let (row, col) = (off / g.stride, off % g.stride);
        if col >= g.width * g.bpp { return (0, 0, 0, 1); }
        let within = col % g.bpp;
        let ch = (within / g.bps) as u32;
        let hi = (g.bps == 1 || (within % g.bps == 0) == g.be) as u32;
        let x = col / g.bpp;
        let byte_at = |dx: isize, dy: isize| -> Option<i32> {
            let (xx, yy) = (x as isize + dx, row as isize + dy);
            if xx < 0 || yy < 0 || xx >= g.width as isize || (dy == 0 && dx >= 0) { return None; }
            Some(hist[g.start + yy as usize * g.stride + xx as usize * g.bpp + within] as i32)
        };
        let w = byte_at(-1, 0);
        let n = byte_at(0, -1);
        let wv = w.or(n).unwrap_or(0);
        let nv = n.unwrap_or(wv);
        let nw = byte_at(-1, -1).unwrap_or(nv);
        let ne = byte_at(1, -1).unwrap_or(nv);
        let (lo, hiv) = (wv.min(nv), wv.max(nv));
        let med = if nw >= hiv { lo } else if nw <= lo { hiv } else { wv + nv - nw };
        let act = (wv - nw).abs() + (nv - nw).abs() + (ne - nv).abs();
        (med.clamp(0, 255) as u32, qlog(act) as u32, ch, hi)
    }

    /// Image-model context value for `id` at the next byte (geometry from `self.img`).

    /// Image-model context values at the next byte for every image context id, indexed by
    /// `id - ctx::I_WN` (geometry from `self.img`). The neighbourhood is read once for all of them.
    fn image_ctxs(&self, hist: &[u8]) -> [u64; NIMG] {
        let Some(g) = self.img else { return [0; NIMG] };
        let p = hist.len();
        let off = p - g.start;
        let row = off / g.stride;
        let col = off % g.stride;
        if col >= g.width * g.bpp { return [0xFFFF_0000 | (col - g.width * g.bpp) as u64; NIMG]; } // row padding
        let x = col / g.bpp;
        let within = col % g.bpp;
        let ch = within / g.bps;
        let hi_byte = g.bps == 1 || (within % g.bps == 0) == g.be; // coding the high byte of a sample?
        let max = if g.bps == 2 { 65535 } else { 255 };
        // sample value at pixel (x+dx, row+dy), same channel; None outside the piece
        let at = |dx: isize, dy: isize| -> Option<i32> {
            let xx = x as isize + dx;
            let yy = row as isize + dy;
            if xx < 0 || yy < 0 || xx >= g.width as isize || (dy == 0 && dx >= 0) { return None; }
            let base = g.start + yy as usize * g.stride + xx as usize * g.bpp + ch * g.bps;
            if g.bps == 1 { return Some(hist[base] as i32); }
            if base + 1 >= p { return None; }
            let (a, b) = (hist[base] as i32, hist[base + 1] as i32);
            Some(if g.be { (a << 8) | b } else { (b << 8) | a })
        };
        let w = at(-1, 0);
        let n = at(0, -1);
        let wv = w.or(n).unwrap_or(0);
        let nv = n.unwrap_or(wv);
        let nw = at(-1, -1).unwrap_or(nv);
        let ne = at(1, -1).unwrap_or(nv);
        let ww = at(-2, 0).unwrap_or(wv);
        let nn = at(0, -2).unwrap_or(nv);
        let nne = at(1, -2).unwrap_or(ne);
        let med = { let (lo, hi) = (wv.min(nv), wv.max(nv)); if nw >= hi { lo } else if nw <= lo { hi } else { wv + nv - nw } };
        let grad = clamp_px(wv + nv - nw, max);
        let pa = { let pp = wv + nv - nw; let (da, db, dc) = ((pp - wv).abs(), (pp - nv).abs(), (pp - nw).abs()); if da <= db && da <= dc { wv } else if db <= dc { nv } else { nw } };
        // cross-channel: previous channel of this pixel and of the west / north pixels
        let prev_ch = |dx: isize, dy: isize| -> Option<i32> {
            if ch == 0 || g.bps != 1 { return None; }
            let xx = x as isize + dx;
            let yy = row as isize + dy;
            if xx < 0 || yy < 0 || xx >= g.width as isize { return None; }
            Some(hist[g.start + yy as usize * g.stride + xx as usize * g.bpp + ch - 1] as i32)
        };
        let cp = prev_ch(0, 0);
        // 16-bit samples: the high byte is predicted from predictions >> 8; the low byte from the
        // prediction minus the high byte already coded (c1 when big-endian, c2 otherwise).
        let v = |pred: i32| -> u64 {
            if g.bps == 1 { return pred as u64; }
            if hi_byte { return (pred >> 8) as u64 | 0x1000; }
            let hi = if g.be { hist[p - 1] as i32 } else { hist[p - 1] as i32 };
            (pred - (hi << 8)).clamp(-256, 511) as u64 & 0x3FF | 0x2000
        };
        // error feedback (CALIC-style): how far the median predictor missed at the west and north
        // neighbours, and a prediction corrected by their average bias
        let med_at = |a: i32, b: i32, c: i32| { let (lo, hi) = (a.min(b), a.max(b)); if c >= hi { lo } else if c <= lo { hi } else { a + b - c } };
        let nww = at(-2, -1).unwrap_or(nw);
        let nnw = at(-1, -2).unwrap_or(nn);
        let err_w = wv - med_at(ww, nw, nww);
        let err_n = nv - med_at(nw, nn, nnw);
        let qerr = |e: i32| -> u64 { (if e < 0 { 16 } else { 0 }) | qlog(e.abs()) };
        let tag = (ch as u64) << 24 | (hi_byte as u64) << 28;
        let mut out = [0u64; NIMG];
        for (k, o) in out.iter_mut().enumerate() {
            let val = match ctx::I_WN + k as u32 {
                ctx::I_ERR => v(med) | qerr(err_w) << 12 | qerr(err_n) << 18,
                ctx::I_CCE => match cp {
                    // correct this channel's median by the previous channel's error at this pixel
                    Some(c) => {
                        let (wp, np, nwp) = (prev_ch(-1, 0).unwrap_or(c), prev_ch(0, -1).unwrap_or(c), prev_ch(-1, -1).unwrap_or(c));
                        let e = c - med_at(wp, np, nwp);
                        v(clamp_px(med + e, max)) | qerr(e) << 12
                    }
                    None => v(clamp_px(med + err_w / 2, max)) | 1 << 47,
                },
                ctx::I_WNQ => (v(wv) >> 3) << 16 | (v(nv) >> 3) << 8 | v(ne) >> 3,
                ctx::I_BIAS => v(clamp_px(med + (err_w + err_n) / 2, max)) | 1 << 46,
                ctx::I_WN => v(wv) << 12 | v(nv),
                ctx::I_MED => v(med),
                ctx::I_GRAD => v(grad),
                ctx::I_AVG => v((wv + nv + 1) >> 1) << 8 | qlog((ne - nw).abs()),
                ctx::I_NNE => v(nv) << 12 | v(ne),
                ctx::I_WW => v(wv) << 12 | v(clamp_px(2 * wv - ww, max)),
                ctx::I_NN => v(nv) << 12 | v(clamp_px(2 * nv - nn, max)),
                ctx::I_PAETH => v(pa) | 1 << 40,
                ctx::I_TEX => {
                    let dh = (wv - ww).abs() + (nv - nw).abs() + (ne - nv).abs();
                    let dv = (wv - nw).abs() + (nv - nn).abs() + (ne - nne).abs();
                    qlog(dh) | qlog(dv) << 4 | v(med) >> 3 << 8
                }
                ctx::I_CC => match cp {
                    Some(c) => { let wp = prev_ch(-1, 0).unwrap_or(c); v(clamp_px(c + wv - wp, max)) | (c as u64) << 12 }
                    None => v(med) | 1 << 44,
                },
                ctx::I_CC2 => match cp {
                    Some(c) => { let np = prev_ch(0, -1).unwrap_or(c); v(clamp_px(c + nv - np, max)) | 1 << 44 }
                    None => v(grad) | 3 << 44,
                },
                ctx::I_WNW => v(wv) << 24 | v(nw) << 12 | v(nv),
                _ => 0,
            };
            *o = val ^ tag;
        }
        out
    }

    fn record_update(&mut self, p: usize, c: usize) {
        // p = position of byte c. Look for a distance that byte c has repeated 4 times in a row.
        let cp = self.cpos[c];
        let cur = p as u32 + 1;
        if cp[3] != 0 {
            let r = cur.wrapping_sub(cp[0]);
            if r > 1 && r == cp[0].wrapping_sub(cp[1]) && r == cp[1].wrapping_sub(cp[2]) && r == cp[2].wrapping_sub(cp[3]) && r < 65536 {
                let r = r as usize;
                if r == self.rlen1 { self.rcount1 += 1; }
                else if r == self.rlen2 { self.rcount2 += 1; }
                else if self.rcount1 > self.rcount2 { self.rlen2 = r; self.rcount2 = 1; }
                else { self.rlen1 = r; self.rcount1 = 1; }
            }
        }
        if self.rcount1 > 15 && self.rlen != self.rlen1 { self.rlen_prev = self.rlen; self.rlen = self.rlen1; self.rcount1 = 0; self.rcount2 = 0; }
        if self.rcount2 > 15 && self.rlen != self.rlen2 { self.rlen_prev = self.rlen; self.rlen = self.rlen2; self.rcount1 = 0; self.rcount2 = 0; }
        if c == 0 { self.last0 = p + 1; }
        self.cpos[c] = [cur, cp[0], cp[1], cp[2]];
    }

    pub fn byte_update(&mut self, hist: &[u8]) {
        let pos = hist.len();
        self.pos = pos;
        if pos > 0 {
            let c = hist[pos - 1] as u32;
            self.c16 = (self.c16 << 8) | (self.c12 >> 24);
            self.c12 = (self.c12 << 8) | (self.c8 >> 24);
            self.c8 = (self.c8 << 8) | (self.c4 >> 24);
            self.c4 = (self.c4 << 8) | c;
            let mut word_ended = false;
            if is_word_char(c) {
                let l = (c as u8).to_ascii_lowercase() as u64;
                self.word = (self.word ^ (l + 1)).wrapping_mul(0x100_0000_01B3) | 1;
            } else if self.word != 0 {
                self.word2 = self.word1;
                self.word1 = self.word;
                self.word = 0;
                word_ended = true;
            }
            self.txt.update(c as u8, word_ended);
            if self.num_on {
                if let Some(n) = self.num.as_mut() { n.byte_update(hist); }
            }
            if pos >= 2 {
                let p1 = hist[pos - 2] as usize;
                self.t1[p1] = (self.t1[p1] << 8) | c as u16;
            }
            if pos >= 3 {
                let p2 = ((hist[pos - 3] as usize) << 8) | hist[pos - 2] as usize;
                self.t2[p2] = (self.t2[p2] << 8) | c as u16;
            }
            self.record_update(pos - 1, c as usize);
            if c == b'\n' as u32 {
                self.prev_line_start = self.line_start;
                self.line_start = pos;
            }
            if pos > self.x86_start {
                if let Some(x) = self.x86.as_mut() { x.update(c as u8); }
            }
            if self.ids.contains(&ctx::O24) {
                let start = pos.saturating_sub(24);
                let mut h = 0u64;
                for &b in &hist[start..pos] { h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3); }
                self.o24 = h;
            }
        }
        self.c0 = 1;
        self.bpos = 0;
        if self.img.is_some() {
            self.img_sum = self.image_summary(hist);
            self.img_ctx = self.image_ctxs(hist);
        }
        self.mm.byte_update(hist);
        if let Some(g) = self.guess.as_mut() { g.byte_update(hist, self.rlen, self.guess_on); }
        for i in 0..self.ids.len() {
            if !self.cm.is_active(i) { continue; }
            let id = self.ids[i];
            let h = hash(id, self.ctx_value(id, hist));
            self.cm.set(i, h);
        }
        self.cm.byte_start(&mut self.ht);
    }

    /// Context `n` of the second binary model: the byte's place in a record (of the detected
    /// length `r`: its column, the bytes above it in the records before, and what they suggest
    /// for a field that counts up), and in 2-, 4- and 8-byte words.
    fn bin2(&self, n: u32, hist: &[u8]) -> u64 {
        let pos = self.pos;
        let at = |d: usize| -> u64 { if d > 0 && d <= pos { hist[pos - d] as u64 } else { 0 } };
        let r = self.rlen;
        let (c1, c2) = (self.c4 as u64 & 0xff, self.c4 as u64 >> 8 & 0xff);
        let col = if r > 0 { (pos % r) as u64 } else { 0 };
        let rr = (r as u64).min(0xFFFF);
        let (n1, n2) = (at(r), at(2 * r));
        let p = pos as u64;
        match n {
            0 => rr | col << 16,
            1 => rr | col << 16 | c1 << 32,
            2 => rr | col << 16 | n1 << 32,
            3 => rr | n1 << 16 | at(r + 1) << 24 | c1 << 32,
            4 => rr | n1 << 16 | n2 << 24 | at(3 * r) << 32,
            5 => rr | col << 16 | ((2 * n1 + 256 - n2) & 255) << 32,
            6 => rr | col << 16 | n1 << 32 | ((c1 < at(r + 1)) as u64) << 40 | ((n1 + 256 - n2) & 255) << 41,
            7 => rr | n1 << 16 | (if r > 1 { at(r - 1) } else { 0 }) << 24,
            8 => rr | col << 16 | n1 << 32 | c1 << 40,
            9 => (p & 3) | at(4) << 2 | at(8) << 10,
            10 => (p & 7) | at(8) << 3 | at(16) << 11,
            11 => (p & 3) | at(4) << 2 | c1 << 10,
            12 => (p & 7) | at(8) << 3 | c1 << 11,
            13 => (p & 3) | ((2 * at(4) + 256 - at(8)) & 255) << 2,
            14 => (p & 7) | ((2 * at(8) + 256 - at(16)) & 255) << 3,
            15 => ((pos - self.last0).min(15) as u64) | c1 << 4,
            16 => {
                // how long ago the byte before was last seen
                let before = self.cpos[c1 as usize][1] as usize;
                c1 | (if before > 0 { qlog((pos - before) as i32) + 1 } else { 0 }) << 8
            }
            17 => (p & 1) | c2 << 1 | at(4) << 9,
            18 => (p & 1) | c2 << 1 | at(4) << 9 | at(6) << 17,
            19 => rr | col << 16 | c1 << 32 | c2 << 40,
            20 => {
                let r2 = self.rlen_prev;
                (r2 as u64).min(0xFFFF) | (if r2 > 0 { (pos % r2) as u64 } else { 0 }) << 16 | at(r2) << 32
            }
            21 => (self.rlen_prev as u64).min(0xFFFF) | at(self.rlen_prev) << 16 | at(2 * self.rlen_prev) << 24,
            22 => (p & 3) | at(4) << 2 | at(3) << 10,
            _ => (p & 7) | at(8) << 3 | at(7) << 11,
        }
    }

    fn ctx_value(&self, id: u32, hist: &[u8]) -> u64 {
        let pos = self.pos;
        let c4 = self.c4 as u64;
        let c8 = self.c8 as u64;
        let c1 = c4 & 0xff;
        match id {
            ctx::O1 => c1,
            ctx::O2 => c4 & 0xffff,
            ctx::O3 => c4 & 0xff_ffff,
            ctx::O4 => c4,
            ctx::O5 => c4 | (c8 & 0xff) << 32,
            ctx::O6 => c4 | (c8 & 0xffff) << 32,
            ctx::O7 => c4 | (c8 & 0xff_ffff) << 32,
            ctx::O8 => c4 | c8 << 32,
            ctx::O12 => (c4 | c8 << 32).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ self.c12 as u64,
            ctx::O16 => (c4 | c8 << 32).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (self.c12 as u64 | (self.c16 as u64) << 32),
            ctx::O24 => self.o24,
            ctx::WORD => if self.word != 0 { self.word } else { c1 << 1 },
            ctx::WORD2 => self.word.wrapping_mul(0xA24B_AED4_963E_E407) ^ self.word1,
            ctx::SP_B2 => (c4 >> 8) & 0xff,
            ctx::SP_B3 => (c4 >> 16) & 0xff,
            ctx::SP_B2B3 => (c4 >> 8) & 0xffff,
            ctx::SP_B1B3 => c1 | (c4 & 0xff_0000) >> 8,
            ctx::SP_B4B8 => (c4 >> 24) | (c8 >> 24) << 8,
            ctx::SP_NIB => c4 & 0xf0f0_f0f0,
            ctx::REC => {
                let r = self.rlen;
                if r > 0 && pos >= 2 * r {
                    r as u64 | (hist[pos - r] as u64) << 16 | (hist[pos - 2 * r] as u64) << 24
                } else { 0 }
            }
            ctx::REC_C1 => {
                let r = self.rlen;
                if r > 0 && pos >= r { (hist[pos - r] as u64) | c1 << 8 | (r as u64) << 16 } else { 0 }
            }
            ctx::IND1 => self.t1[c1 as usize] as u64 | c1 << 16,
            ctx::IND2 => self.t2[(c4 & 0xffff) as usize] as u64 | (c4 & 0xffff) << 16,
            ctx::MATCH => match self.mm.expected() { Some((e, lb)) => e as u64 | (lb as u64) << 8 | 1 << 16, None => 0 },
            ctx::COL => {
                let col = pos - self.line_start;
                let above = if self.prev_line_start + col + 1 < self.line_start { hist[self.prev_line_start + col] as u64 } else { 0 };
                col.min(255) as u64 | above << 8
            }
            ctx::POS => (pos & 3) as u64 | c1 << 8,
            ctx::T_W3 => self.word.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ self.word1.wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ self.word2,
            ctx::T_WSKIP => self.word.wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ self.word2,
            ctx::T_W1C2 => self.word1.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (c4 & 0xffff),
            ctx::T_XML => {
                let t = &self.txt;
                if t.in_tag { t.tag.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (t.tag_done as u64) << 8 ^ c1 ^ 1 << 60 }
                else { t.elem.wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ c1 }
            }
            ctx::T_FIELD => (self.txt.field.min(63) as u64) | (self.txt.field_pos.min(15) as u64) << 8 | c1 << 16,
            ctx::T_INDENT => (self.txt.indent.min(63) as u64) | (self.txt.line_start as u64) << 8 | (c4 & 0xffff) << 16,
            ctx::T_NUM => if self.txt.num != 0 { self.txt.num } else { c1 << 1 | 1 << 40 },
            ctx::T_BRACKET => {
                let t = &self.txt;
                let top = if t.depth > 0 && t.depth <= 16 { t.brackets[t.depth - 1] as u64 } else { 0 };
                top | (t.depth.min(7) as u64) << 8 | c1 << 16
            }
            ctx::T_SENT => self.word.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ self.txt.sent_words.min(15) as u64,
            id if ctx::image_only(id) => self.img_ctx[(id - ctx::I_WN) as usize],
            id if ctx::x86_only(id) => self.x86.as_ref().map_or(0, |x| x.context(id - ctx::X_FIRST, self.c4, pos - self.x86_start)),
            id if ctx::bin2_only(id) => self.bin2(id - ctx::B_FIRST, hist),
            id if ctx::num_only(id) => self.num.as_ref().map_or(0, |n| n.context(id - ctx::N_FIRST, hist)),
            _ => 0,
        }
    }

    #[inline]
    pub fn p(&mut self) -> u32 {
        let bpos = self.bpos;
        let c0 = self.c0;
        let c4 = self.c4;
        // SSE contexts are known before mixing: prefetch their rows so the lookups after the
        // mixer find them in cache.
        let a1cx = (c0 | (c4 & 0xff) << 8) as usize;
        // in a number of text (or where one is expected): how it compares with the number last
        // in its place
        let number = if self.num_on { self.num.as_ref().map_or(0, |n| n.state()) } else { 0 };
        let keep = if number != 0 { super::num::KEEP.load(std::sync::atomic::Ordering::Relaxed) } else { 0 };
        let a2cx = match self.x86.as_ref() {
            // in code: what the byte is, and the opcode's low bits
            Some(x) => c0 as usize | (x.state() & 127) << 8 | (x.opcode_class() & 1) << 15,
            None if keep >> 63 & 1 == 1 => c0 as usize | number << 8,
            None => c0 as usize | (((c4 & 0xffff).wrapping_mul(0x9E37_79B1) >> 24) as usize) << 8,
        };
        self.a3cx = match self.mm.expected() {
            Some((e, lb)) => (lb.min(15) << 11) | ((e as usize) << 3) | bpos as usize,
            None => 16 * 256 * 8 + c0 as usize,
        };
        if let Some(a) = self.a1.as_ref() { a.prefetch(a1cx); }
        if let Some(a) = self.a2.as_ref() { a.prefetch(a2cx); }
        if let Some(a) = self.a3.as_ref() { a.prefetch(self.a3cx); }

        let j = if bpos < 4 { c0 as usize } else { ((c0 & ((1 << (bpos - 4)) - 1)) | (1 << (bpos - 4))) as usize };
        let hits = self.cm.mix(&self.ht, j, &mut self.mixer);
        self.mm.mix(bpos, c0, &mut self.mixer);
        if let Some(g) = self.guess.as_mut() { g.mix(bpos, c0, &mut self.mixer); }
        self.mixer.add(256);
        for (k, &s) in self.sels.iter().enumerate() {
            let cx = match s {
                sel::C0 => self.kgroup * 256 + c0 as usize,
                sel::MATCH => self.mm.mixer_ctx(),
                sel::C1 if self.img.is_some() => { let (_, act, ch, hi) = self.img_sum; 256 + ((ch & 3) * 64 + act.min(15) * 4 + hi * 2) as usize }
                sel::C2 if self.img.is_some() => 256 + (self.img_sum.0 >> 1) as usize,
                sel::C2 if self.x86.is_some() => 256 + self.x86.as_ref().map_or(0, |x| x.state()),
                sel::C2 if keep >> 62 & 1 == 1 => 256 + number,
                sel::C1 => (c4 & 0xff) as usize,
                sel::HITS => hits.min(31) * 8 + bpos as usize,
                sel::C2 => ((c4 >> 8) & 0xff) as usize,
                _ => 0,
            };
            self.mixer.set(k, cx);
        }
        self.mixer.set_final(self.kgroup * 256 + c0 as usize);
        let pm = self.mixer.p();
        // SSE refinements carry 3/4 of the weight: they reach the extremes the mixer's logit clamp
        // cannot, which matters on highly redundant data. Weights are fixed per APM count.
        if let (true, Some(a4)) = (self.img.is_some(), self.a4.as_mut()) {
            let a4cx = (self.img_sum.0 as usize) << 8 | c0 as usize;
            let p4 = a4.p(pm, a4cx);
            let p1 = self.a1.as_mut().map(|a| a.p(pm, a1cx)).unwrap_or(p4);
            self.apm_img = true;
            return ((pm + p1 + 2 * p4) >> 2).clamp(1, 65535);
        }
        self.apm_img = false;
        let p = match (self.a1.as_mut(), self.a2.as_mut(), self.a3.as_mut()) {
            (Some(a1), Some(a2), Some(a3)) => (pm + a1.p(pm, a1cx) + a2.p(pm, a2cx) + a3.p(pm, self.a3cx)) >> 2,
            (Some(a1), Some(a2), None) => (2 * pm + 3 * (a1.p(pm, a1cx) + a2.p(pm, a2cx))) >> 3,
            (Some(a1), None, None) => (pm + 3 * a1.p(pm, a1cx)) >> 2,
            _ => pm,
        };
        p.clamp(1, 65535)
    }

    #[inline]
    pub fn update(&mut self, bit: u32) {
        self.cm.update(&mut self.ht, bit);
        self.mm.update(bit);
        if let Some(g) = self.guess.as_mut() { g.update(bit); }
        self.mixer.update(bit);
        if self.apm_img {
            if let Some(a) = self.a1.as_mut() { a.update(bit); }
            if let Some(a) = self.a4.as_mut() { a.update(bit); }
        } else {
            if let Some(a) = self.a1.as_mut() { a.update(bit); }
            if let Some(a) = self.a2.as_mut() { a.update(bit); }
            if let Some(a) = self.a3.as_mut() { a.update(bit); }
        }
        self.c0 = (self.c0 << 1) | bit;
        self.bpos += 1;
        if self.bpos == 4 { self.cm.nibble(&mut self.ht, self.c0); }
    }
}
