//! Model for AAC frame records (the frame leaves of AAC containers; see `recode::aac`).
//!
//! The leaf holds each frame as a parsed record; this model codes the fields themselves:
//! * element kinds, tags and window layouts against the previous frame's;
//! * section code books band by band (a "new section" flag, then the book against the previous
//!   frame's book for the band and the band before);
//! * scale factors as the stream's own differences, with the difference the previous frame's
//!   scale factor would predict as context;
//! * spectral values line by line — a zero flag, "larger than k" steps up to the band's code
//!   book limit, escape bits — with the lines just coded, the same lines of the channel's two
//!   previous frames (scaled by the change of the band's scale factor), the other channel of a
//!   pair, the position and the values already coded in the tuple as contexts; signs against
//!   the other channel's and the previous frame's.
//!
//! Fill and data-stream payloads, noise shaping and pulse data are coded as plain bytes.
use super::binary::{q3, q8, ql, sc, Bins, GAIN_SCALE};
use super::coder::Coder;
use super::predictor::ByteBuf;
use crate::recode::aac::{self, Adts, Body, Element, Frame, Ics, Info, Pulse, Rec, LINES, SECTION_START};
use crate::recode::aac_tables::SHAPE;

const MAXIN: usize = 12;
const F_HDR: usize = 0;
const F_INFO: usize = 1;
const F_GAIN: usize = 2;
const F_CB: usize = 3;
const F_SF: usize = 4;
const F_ZERO: usize = 5;
const F_MAG: usize = 6;
const F_ESC: usize = 7;
const F_SIGN: usize = 8;
const F_RAW: usize = 9;
pub const FAMILIES: usize = 10;
pub const FAMILY_NAMES: [&str; FAMILIES] = ["header", "layout", "gain", "codebooks", "scalefac", "zero", "magnitude", "escape", "sign", "raw"];
const BITS: [u32; FAMILIES] = [14, 14, 14, 16, 16, 16, 16, 14, 16, 14];
const NIN: [usize; FAMILIES] = [6, 6, 6, 6, 6, MAXIN, MAXIN, 6, 6, 6];
/// Channels with their own history (further ones share the last).
const CHANNELS: usize = 8;
/// Slots of per-band state: band index, or 16 · group + band for short windows.
const SLOTS: usize = 128;

#[inline]
fn slot(info: &Info, g: usize, sfb: usize) -> usize { if info.short() { (g * 16 + sfb).min(SLOTS - 1) } else { sfb.min(SLOTS - 1) } }

/// What a channel's next frame conditions on.
#[derive(Clone)]
struct Chan {
    seen: bool,
    info: Info,
    gain: u8,
    dgain: i32,
    /// Per slot: the previous frame's code book byte, absolute scale factor and scale-factor code.
    cb: [u8; SLOTS],
    sf: [i16; SLOTS],
    sfd: [i8; SLOTS],
    pulse: bool,
    tns: bool,
    /// Magnitudes (capped) and signs (0, 1 positive, 2 negative) by line — `128 · window +
    /// frequency` for short windows — of the frame being coded and the two before it.
    cur: [u8; LINES],
    prev: [u8; LINES],
    prev2: [u8; LINES],
    scur: [u8; LINES],
    sprev: [u8; LINES],
    /// Whether `prev` / `prev2` were short-window frames.
    short: [bool; 2],
}

impl Chan {
    fn new() -> Self {
        Chan {
            seen: false, info: Info::default(), gain: 0, dgain: 0, cb: [0; SLOTS], sf: [0; SLOTS], sfd: [0; SLOTS], pulse: false, tns: false,
            cur: [0; LINES], prev: [0; LINES], prev2: [0; LINES], scur: [0; LINES], sprev: [0; LINES], short: [false; 2],
        }
    }
}

pub struct AacModel {
    b: Bins,
    ch: Vec<Chan>,
    /// Per element position: the previous frame's element kind, tag, payload length, pair flags
    /// and M/S mask.
    elem: Vec<[u16; 5]>,
    ms: Vec<[u8; SLOTS]>,
    adts: [u8; 7],
    flags: u32,
}

impl Default for AacModel {
    fn default() -> Self { Self::new() }
}

#[inline]
fn at(a: &[u8; LINES], i: usize, d: isize) -> u32 {
    let j = i as isize + d;
    if (0..LINES as isize).contains(&j) { a[j as usize] as u32 } else { 0 }
}

/// Position classes within a window: every 4 lines below 64, then coarser (48 classes).
#[inline]
fn posb(f: usize) -> u64 { if f < 64 { (f / 4) as u64 } else if f < 256 { 16 + ((f - 64) / 16) as u64 } else { 28 + ((f - 256) / 64).min(19) as u64 } }

impl AacModel {
    pub fn new() -> Self {
        AacModel { b: Bins::new(&BITS, &NIN), ch: (0..CHANNELS).map(|_| Chan::new()).collect(), elem: vec![[0; 5]; 64], ms: vec![[0; SLOTS]; 64], adts: [0; 7], flags: 0 }
    }

    /// Collect coding cost per family (developer statistics).
    pub fn enable_stats(&mut self) { self.b.stats = Some(vec![0; FAMILIES]); }

    pub fn stats(&self) -> [u64; FAMILIES] {
        let mut out = [0; FAMILIES];
        if let Some(s) = &self.b.stats { out.copy_from_slice(s); }
        out
    }

    fn byte<C: Coder>(&mut self, c: &mut C, v: u8, id: u64, a: u64) -> u8 { self.b.tree(c, v as u32, 8, F_RAW, id, a, 0, 0) as u8 }

    /// A flag and, when set, a count of at least one.
    fn count<C: Coder>(&mut self, c: &mut C, n: u32, id: u64, a: u64) -> u32 {
        if self.b.bit(c, (n != 0) as u32, F_RAW, sc(id, 0, a, 0, 0), id as usize, 0) == 0 { return 0; }
        self.b.gamma(c, n, F_RAW, id, a, 0, 0)
    }

    fn bytes<C: Coder>(&mut self, c: &mut C, src: &[u8], n: usize, id: u64) -> Vec<u8> {
        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            let prev = if k > 0 { 1 + (out[k - 1] >> 4) as u64 } else { 0 };
            out.push(self.byte(c, src.get(k).copied().unwrap_or(0), id, prev * 4 + k.min(3) as u64));
        }
        out
    }

    fn info<C: Coder>(&mut self, c: &mut C, src: &Info, ci: usize, sf: usize) -> Result<Info, String> {
        let p = self.ch[ci].info;
        let seen = self.ch[ci].seen as u64;
        let mut i = Info { seq: self.b.tree(c, src.seq as u32, 2, F_INFO, 1, p.seq as u64 + 4 * seen, 0, 0) as u8, ..Default::default() };
        i.shape = self.b.bit(c, src.shape as u32, F_INFO, sc(2, 0, p.shape as u64, i.seq as u64, 0), 2, 0) as u8;
        let same_kind = (i.short() == p.short()) as u64 * seen;
        let nbits = if i.short() { 4 } else { 6 };
        i.max_sfb = if same_kind == 1 && self.b.bit(c, (src.max_sfb == p.max_sfb) as u32, F_INFO, sc(3, 0, i.short() as u64, 0, 0), 3, 1) == 1 {
            p.max_sfb
        } else {
            self.b.tree(c, src.max_sfb as u32, nbits, F_INFO, 4 + i.short() as u64, same_kind * (1 + p.max_sfb as u64), 0, 0) as u8
        };
        if i.short() {
            i.grouping = self.b.tree(c, src.grouping as u32, 7, F_INFO, 6, if p.short() { 1 + p.grouping as u64 } else { 0 }, 0, 0) as u8;
        }
        if i.max_sfb as usize > i.offsets(sf).len() - 1 { return Err("aac frames: bad window layout (corrupt file)".into()); }
        Ok(i)
    }

    /// Code one channel; `common`: the pair's shared layout; `other`: the pair's first channel
    /// (already coded) when this is the second; `ms`: per slot, whether the pair codes the band
    /// as mid/side.
    #[allow(clippy::too_many_arguments)]
    fn ics<C: Coder>(&mut self, c: &mut C, src: &Ics, ci: usize, sf: usize, common: Option<Info>, other: Option<usize>, ms: &[u8; SLOTS]) -> Result<Ics, String> {
        let bad = || "aac frames: bad channel (corrupt file)".to_string();
        let chx = other.is_some() as u64;
        let p_gain = self.ch[ci].gain;
        let o_d = other.map_or(0, |o| (self.ch[o].dgain.clamp(-8, 8) + 9) as u64);
        let dg = self.b.delta(c, src.gain as i32 - p_gain as i32, F_GAIN, 1 + chx, (self.ch[ci].dgain.clamp(-8, 8) + 8) as u64, o_d, self.ch[ci].seen as u64);
        let mut s = Ics { gain: u8::try_from(p_gain as i32 + dg).map_err(|_| bad())?, ..Default::default() };
        s.info = match common { Some(i) => i, None => self.info(c, &src.info, ci, sf)? };
        let info = s.info;
        let (groups, max_sfb, off) = (info.groups(), info.max_sfb as usize, info.offsets(sf));
        let same_kind = self.ch[ci].seen && info.short() == self.ch[ci].info.short();

        // ---- section code books
        let mut cbs = [0u8; SLOTS];
        for g in 0..groups.len() {
            for sfb in 0..max_sfb {
                let sl = slot(&info, g, sfb);
                let truth = src.cb.get(g * max_sfb + sfb).copied().unwrap_or(0);
                let before = if sfb > 0 { 1 + (s.cb[g * max_sfb + sfb - 1] & 15) as u64 } else { 0 };
                let (pcb, pstart) = if same_kind { (1 + (self.ch[ci].cb[sl] & 15) as u64, 1 + (self.ch[ci].cb[sl] >> 7) as u64) } else { (0, 0) };
                let ocb = other.map_or(0, |o| 1 + (self.ch[o].cb[sl] & 15) as u64);
                let start = sfb == 0 || self.b.bit(c, (truth >> 7) as u32, F_CB, sc(1, chx, pstart * 32 + before, (pcb == before) as u64 * 32 + pcb, ocb * 4 + (ocb == before) as u64 + 2 * info.short() as u64), 1, (pstart * 17 + before) as usize) == 1;
                let cb = if start {
                    self.b.tree(c, (truth & 15) as u32, 4, F_CB, 2 + chx, pcb * 17 + before, ocb * 17 + before, (sfb as u64).min(40) * 2 + info.short() as u64) as u8
                } else {
                    s.cb[g * max_sfb + sfb - 1] & 15
                };
                if cb == 12 { return Err(bad()); }
                cbs[sl] = cb | if start { SECTION_START } else { 0 };
                s.cb.push(cbs[sl]);
            }
        }

        // ---- scale factors
        // running values of the three kinds: spectrum, noise energy, intensity position
        let mut run = [s.gain as i32, s.gain as i32 - 90, 0];
        let mut sfa = [0i16; SLOTS];
        let mut sfd = [0i8; SLOTS];
        let (mut noise_pcm, mut dprev, mut j) = (true, 0i32, 0usize);
        for g in 0..groups.len() {
            for sfb in 0..max_sfb {
                let sl = slot(&info, g, sfb);
                let cb = s.cb[g * max_sfb + sfb] & 15;
                if !aac::has_sf(cb) { continue; }
                let truth = src.sf.get(j).copied().unwrap_or(0) as u32;
                j += 1;
                let kind = match cb { 13 => 1, 14 | 15 => 2, _ => 0 };
                if kind == 1 && noise_pcm {
                    noise_pcm = false;
                    let v = self.b.tree(c, truth, 9, F_SF, 1, 0, 0, 0);
                    run[1] += v as i32 - 256;
                    s.sf.push(v as u16);
                    continue;
                }
                // what the previous frame's scale factor for this band would make the difference
                let had = same_kind && aac::has_spec(self.ch[ci].cb[sl] & 15) && kind == 0;
                let pd = if had { ((self.ch[ci].sf[sl] as i32 - run[0]).clamp(-15, 15) + 16) as u64 } else { 0 };
                let od = other.map_or(0, |o| (self.ch[o].sfd[sl] as i32).clamp(-7, 7) as i64 + 8) as u64;
                let d = self.b.delta(c, truth as i32 - 60, F_SF, 2 + kind as u64, pd, ((dprev.clamp(-7, 7) + 8) as u64) * 16 + od, (sfb as u64).min(47) + 48 * info.short() as u64);
                if !(-60..=60).contains(&d) { return Err(bad()); }
                run[kind] += d;
                dprev = d;
                if kind == 0 {
                    sfa[sl] = run[0].clamp(-2000, 2000) as i16;
                    sfd[sl] = d.clamp(-100, 100) as i8;
                }
                s.sf.push((d + 60) as u16);
            }
        }

        // ---- pulse data and temporal noise shaping
        if self.b.bit(c, src.pulse.is_some() as u32, F_INFO, sc(8, chx, self.ch[ci].pulse as u64, info.short() as u64, 0), 8, 2) == 1 {
            if info.short() { return Err(bad()); }
            let blank = Pulse::default();
            let sp = src.pulse.as_ref().unwrap_or(&blank);
            let n = self.b.tree(c, (sp.pulses.len() as u32).wrapping_sub(1) & 3, 2, F_RAW, 1, 0, 0, 0) as usize + 1;
            let mut p = Pulse { start: self.b.tree(c, sp.start as u32, 6, F_RAW, 2, 0, 0, 0) as u8, pulses: Vec::with_capacity(n) };
            for k in 0..n {
                let (o, a) = sp.pulses.get(k).copied().unwrap_or((0, 0));
                p.pulses.push((self.b.tree(c, o as u32, 5, F_RAW, 3, 0, 0, 0) as u8, self.b.tree(c, a as u32, 4, F_RAW, 4, 0, 0, 0) as u8));
            }
            s.pulse = Some(p);
        }
        if self.b.bit(c, src.tns.is_some() as u32, F_INFO, sc(9, chx, self.ch[ci].tns as u64, info.seq as u64, other.map_or(0, |o| 1 + self.ch[o].tns as u64)), 9, 3) == 1 {
            let blank = (0u16, vec![]);
            let (sn, sb) = src.tns.as_ref().unwrap_or(&blank);
            let n = self.b.gamma(c, *sn as u32 + 1, F_RAW, 5, info.short() as u64, 0, 0) - 1;
            if n > 4000 { return Err(bad()); }
            let mut bytes = Vec::with_capacity((n as usize).div_ceil(8));
            for k in 0..(n as usize).div_ceil(8) {
                let width = (n as usize - 8 * k).min(8) as u32;
                bytes.push(self.b.tree(c, sb.get(k).copied().unwrap_or(0) as u32, width, F_RAW, 6, k.min(7) as u64 * 9 + width as u64, 0, 0) as u8);
            }
            s.tns = Some((n as u16, bytes));
        }

        // ---- spectral values
        let short = info.short();
        let hist_ok = self.ch[ci].seen && self.ch[ci].short[0] == short;
        let hist2_ok = hist_ok && self.ch[ci].short[1] == short;
        self.ch[ci].cur = [0; LINES];
        self.ch[ci].scur = [0; LINES];
        let mut k = 0usize; // index into the stream-ordered values
        let mut w0 = 0usize;
        for (g, &glen) in groups.iter().enumerate() {
            for sfb in 0..max_sfb {
                let sl = slot(&info, g, sfb);
                let cb = s.cb[g * max_sfb + sfb] & 15;
                if !aac::has_spec(cb) { continue; }
                let (dim, _, lav) = SHAPE[cb as usize - 1];
                let (lav, t) = (lav as u32, cb as u64);
                // how the previous frame's values in this band scale to this frame's step
                let scale = if hist_ok && !short && aac::has_spec(self.ch[ci].cb[sl] & 15) {
                    GAIN_SCALE[((sfa[sl] as i32 - self.ch[ci].sf[sl] as i32).clamp(-32, 32) + 32) as usize]
                } else { 256 };
                let is_ms = ms[sl] as u64;
                let (f0, width) = (off[sfb] as usize, (off[sfb + 1] - off[sfb]) as usize);
                for w in 0..glen as usize {
                    let base = if short { (w0 + w) * 128 } else { 0 };
                    let mut in_tuple = 0u64; // sum of the magnitudes coded so far in the tuple
                    for x in 0..width {
                        if x % dim == 0 { in_tuple = 0; }
                        let i = base + f0 + x;
                        let truth = src.spec.get(k).copied().unwrap_or(0);
                        k += 1;
                        let ch = &self.ch[ci];
                        // neighbours in frequency (within the window), then history: the previous
                        // frames for long windows, the previous windows for short ones
                        let (a1, a2) = if f0 + x > 0 { (ch.cur[i - 1], if f0 + x > 1 { ch.cur[i - 2] } else { 0 }) } else { (0, 0) };
                        let s_cur = a1 as u32 + a2 as u32 + if f0 + x > 2 { ch.cur[i - 3] as u32 } else { 0 } + if f0 + x > 3 { ch.cur[i - 4] as u32 } else { 0 };
                        let (b0, b1, bm, s_prev, s_two) = if short {
                            let earlier = |back: usize, d: isize| -> u32 {
                                let f = (f0 + x) as isize + d;
                                if !(0..128).contains(&f) { return 0; }
                                if w0 + w >= back { ch.cur[(w0 + w - back) * 128 + f as usize] as u32 }
                                else if hist_ok { ch.prev[(8 + w0 + w - back) * 128 + f as usize] as u32 }
                                else { 0 }
                            };
                            let (b0, b1, bm) = (earlier(1, 0), earlier(1, 1), earlier(1, -1));
                            (b0, b1, bm, earlier(1, -2) + bm + 2 * b0 + b1 + earlier(1, 2), earlier(2, -1) + 2 * earlier(2, 0) + earlier(2, 1))
                        } else if hist_ok {
                            let (b0, b1, bm) = (ch.prev[i] as u32, at(&ch.prev, i, 1), at(&ch.prev, i, -1));
                            let s_two = if hist2_ok { at(&ch.prev2, i, -1) + 2 * ch.prev2[i] as u32 + at(&ch.prev2, i, 1) } else { 0 };
                            (b0 * scale >> 8, b1 * scale >> 8, bm * scale >> 8, (at(&ch.prev, i, -2) + bm + 2 * b0 + b1 + at(&ch.prev, i, 2)) * scale >> 8, s_two * scale >> 8)
                        } else { (0, 0, 0, 0, 0) };
                        let (b0, b1, bm) = (b0.min(255) as u8, b1.min(255) as u8, bm.min(255) as u8);
                        let (o0, s_oth) = match other {
                            Some(o) => (self.ch[o].cur[i], at(&self.ch[o].cur, i, -1) + 2 * self.ch[o].cur[i] as u32 + at(&self.ch[o].cur, i, 1)),
                            None => (bm, s_two),
                        };
                        let zrun = (1..=15usize).take_while(|&d| d <= f0 + x && self.ch[ci].cur[i - d] == 0).count() as u64;
                        let tup = ((x % dim) as u64) * 20 + in_tuple.min(19);
                        let pos = posb(f0 + x);
                        let mut a = 0u32;
                        let target = truth.unsigned_abs() as u32;
                        while a < lav {
                            let dk = a as u64;
                            let fam = if a == 0 { F_ZERO } else { F_MAG };
                            let td = t * 17 + dk;
                            let ctx = [
                                (td * 4 + q3(a1)) * 4 + q3(a2),
                                ((td * 8 + q8(b0)) * 4 + q3(b1)) * 2 + hist_ok as u64,
                                (dk * 64 + pos) * 8 + q8(b0),
                                (((td * 8 + q8(o0)) * 4 + q3(a1)) * 2 + chx) * 2 + is_ms,
                                (dk * 16 + ql(s_cur)) * 16 + ql(s_prev),
                                (td * 80 + tup) * 8 + q8(a1),
                                td * 16 + ql(s_cur + s_prev),
                                (dk * 16 + ql(s_prev)) * 16 + ql(s_two),
                                (td * 16 + ql(s_oth)) * 4 + q3(a1),
                                td * 80 + tup,
                                (dk * 16 + zrun) * 8 + q8(b0),
                                (td * 64 + pos) * 2 + short as u64,
                            ];
                            let sel = if a == 0 { cb as usize * 4 + (x % dim) } else { (a as usize).min(15) + 16 * (cb >= 9) as usize + 32 * (x % 2) };
                            let sel2 = (ql(s_cur + s_prev) * 4 + dk.min(3)) as usize;
                            let actx = if a == 0 { q8(b0) as usize * 8 + q8(a1) as usize } else { (a as usize).min(7) * 8 + q8(b0) as usize };
                            if self.b.bit2(c, (target > a) as u32, fam, &ctx, sel, sel2, actx) == 0 { break; }
                            a += 1;
                        }
                        let mut val = a;
                        if cb == 11 && a == 16 {
                            // the escape: a value of 16 or more
                            let e = self.b.gamma(c, target.saturating_sub(16) + 1, F_ESC, 1, ql(a1 as u32 + a2 as u32), ql(s_prev), tup.min(40)) - 1;
                            val = 16 + e;
                            if val > 8191 { return Err(bad()); }
                        }
                        in_tuple += val.min(19) as u64;
                        self.ch[ci].cur[i] = val.min(255) as u8;
                        let mut v = val as i16;
                        if val != 0 {
                            let ch = &self.ch[ci];
                            let so = other.map_or(3, |o| self.ch[o].scur[i] as u64);
                            let sp = if hist_ok && !short { ch.sprev[i] as u64 } else { 3 };
                            let s1 = if f0 + x > 0 { ch.scur[i - 1] as u64 } else { 3 };
                            let s2 = if f0 + x > 1 { ch.scur[i - 2] as u64 } else { 3 };
                            let rel = |q: usize| -> u64 { other.map_or(0, |o| { let (p, r) = (self.ch[o].scur[q], ch.scur[q]); if p == 0 || r == 0 { 0 } else { 1 + (p == r) as u64 } }) };
                            let (r1, r2) = (if f0 + x > 0 { rel(i - 1) } else { 0 }, if f0 + x > 1 { rel(i - 2) } else { 0 });
                            let mv = val.min(3) as u64;
                            let mo = other.map_or(0, |o| q3(self.ch[o].cur[i]));
                            let ctx = [
                                ((is_ms * 4 + so) * 4 + mv) * 4 + mo,
                                (sp * 4 + s1) * 4 + mv,
                                ((so * 3 + r1) * 3 + r2) * 2 + is_ms,
                                (is_ms * 4 + so) * 64 + pos,
                                (s1 * 4 + s2) * 4 + mv,
                                ((is_ms * 4 + so) * 4 + s1) * 4 + sp,
                            ];
                            let neg = self.b.bit(c, (truth < 0) as u32, F_SIGN, ctx, (is_ms * 4 + so) as usize * 4 + mv as usize, (so * 16 + sp * 4 + s1) as usize) == 1;
                            self.ch[ci].scur[i] = 1 + neg as u8;
                            if neg { v = -v; }
                        }
                        s.spec.push(v);
                    }
                }
            }
            w0 += glen as usize;
        }

        // ---- what the next frame conditions on
        let ch = &mut self.ch[ci];
        ch.seen = true;
        ch.info = info;
        ch.dgain = dg;
        ch.gain = s.gain;
        ch.cb = cbs;
        ch.sf = sfa;
        ch.sfd = sfd;
        ch.pulse = s.pulse.is_some();
        ch.tns = s.tns.is_some();
        Ok(s)
    }

    /// Code one frame's elements; `src` is the frame on the encoder side.
    fn frame<C: Coder>(&mut self, c: &mut C, src: &Frame, sf: usize) -> Result<Frame, String> {
        let bad = || "aac frames: bad frame (corrupt file)".to_string();
        let mut f = Frame::default();
        let mut ci = 0usize;
        let no_ms = [0u8; SLOTS];
        let blank_ics = Ics::default();
        for e in 0..65 {
            let se = src.elements.get(e);
            let id_truth = match se { Some(Element::Single { lfe, .. }) => if *lfe { 3 } else { 0 }, Some(Element::Pair { .. }) => 1, Some(Element::Data { .. }) => 4, Some(Element::Fill { .. }) => 6, None => 7 };
            let pe = self.elem[e.min(63)];
            let id = self.b.tree(c, id_truth, 3, F_HDR, 1, pe[0] as u64 * 2 + (e > 0) as u64, e.min(7) as u64, 0);
            if id == 7 { break; }
            if e == 64 { return Err(bad()); }
            let mut now = [id as u16 + 1, 0, 0, 0, 0];
            match id {
                0 | 3 => {
                    let (stag, sics) = match se { Some(Element::Single { tag, ics, .. }) => (*tag, ics), _ => (0, &blank_ics) };
                    let tag = self.b.tree(c, stag as u32, 4, F_HDR, 2, pe[1] as u64, 0, 0) as u8;
                    let ics = self.ics(c, sics, ci.min(CHANNELS - 1), sf, None, None, &no_ms)?;
                    ci += 1;
                    now[1] = tag as u16;
                    f.elements.push(Element::Single { lfe: id == 3, tag, ics });
                }
                1 => {
                    let blank = [Ics::default(), Ics::default()];
                    let (stag, scommon, sms, smask, sics) = match se {
                        Some(Element::Pair { tag, common, ms_present, ms_used, ics }) => (*tag, *common, *ms_present, &ms_used[..], &**ics),
                        _ => (0, false, 0, &[][..], &blank),
                    };
                    let tag = self.b.tree(c, stag as u32, 4, F_HDR, 2, pe[1] as u64, 0, 0) as u8;
                    let common = self.b.bit(c, scommon as u32, F_INFO, sc(10, 0, pe[3] as u64 & 3, 0, 0), 10, 4) == 1;
                    let (c0, c1) = (ci.min(CHANNELS - 1), (ci + 1).min(CHANNELS - 1));
                    let (mut info, mut ms_present, mut ms_used) = (None, 0u8, vec![]);
                    let mut mask = [0u8; SLOTS];
                    if common {
                        let i = self.info(c, &sics[0].info, c0, sf)?;
                        ms_present = self.b.tree(c, sms as u32, 2, F_INFO, 11, (pe[3] >> 2) as u64, i.short() as u64, 0) as u8;
                        if ms_present == 3 { return Err(bad()); }
                        let (groups, max_sfb) = (i.groups().len(), i.max_sfb as usize);
                        let prev_mask = self.ms[e.min(63)];
                        let same_kind = self.ch[c0].seen && self.ch[c0].info.short() == i.short() && pe[3] >> 2 == 1;
                        for g in 0..groups {
                            for sfb in 0..max_sfb {
                                let sl = slot(&i, g, sfb);
                                if ms_present == 1 {
                                    let before = if sfb > 0 { 1 + mask[slot(&i, g, sfb - 1)] as u64 } else { 0 };
                                    let pm = if same_kind { 1 + prev_mask[sl] as u64 } else { 0 };
                                    let m = self.b.bit(c, smask.get(g * max_sfb + sfb).copied().unwrap_or(0) as u32, F_INFO, sc(12, 0, pm * 3 + before, (sfb as u64).min(47), i.short() as u64), 12, (pm * 3 + before) as usize) as u8;
                                    ms_used.push(m);
                                    mask[sl] = m;
                                } else {
                                    mask[sl] = (ms_present == 2) as u8;
                                }
                            }
                        }
                        info = Some(i);
                    }
                    let a = self.ics(c, &sics[0], c0, sf, info, None, &mask)?;
                    let d = self.ics(c, &sics[1], c1, sf, info, Some(c0), &mask)?;
                    ci += 2;
                    now[1] = tag as u16;
                    now[3] = common as u16 | (ms_present as u16) << 2;
                    self.ms[e.min(63)] = mask;
                    f.elements.push(Element::Pair { tag, common, ms_present, ms_used, ics: Box::new([a, d]) });
                }
                4 => {
                    let (stag, salign, spad, sdata) = match se { Some(Element::Data { tag, align, pad, data }) => (*tag, *align, *pad, &data[..]), _ => (0, false, 0, &[][..]) };
                    let tag = self.b.tree(c, stag as u32, 4, F_HDR, 2, pe[1] as u64, 0, 0) as u8;
                    let align = self.b.bit(c, salign as u32, F_HDR, sc(3, 0, 0, 0, 0), 3, 0) == 1;
                    let pad = if align { self.byte(c, spad, 7, 0) } else { 0 };
                    let n = self.count(c, sdata.len() as u32, 8, (pe[2] != 0) as u64) as usize;
                    if n > 510 { return Err(bad()); }
                    let data = self.bytes(c, sdata, n, 9);
                    now[1] = tag as u16;
                    now[2] = n as u16;
                    f.elements.push(Element::Data { tag, align, pad, data });
                }
                6 => {
                    let sdata = match se { Some(Element::Fill { data }) => &data[..], _ => &[][..] };
                    let n = self.count(c, sdata.len() as u32, 10, (pe[2] != 0) as u64) as usize;
                    if n > 269 { return Err(bad()); }
                    let data = self.bytes(c, sdata, n, 11);
                    now[2] = n as u16;
                    f.elements.push(Element::Fill { data });
                }
                _ => return Err(bad()),
            }
            self.elem[e.min(63)] = now;
        }
        // the frames coded are history for the next ones
        for ch in self.ch.iter_mut().take(ci.min(CHANNELS)) {
            ch.prev2 = ch.prev;
            ch.prev = ch.cur;
            ch.sprev = ch.scur;
            ch.short = [ch.info.short(), ch.short[0]];
        }
        if self.b.bit(c, (src.pad != 0) as u32, F_RAW, sc(12, 0, 0, 0, 0), 12, 0) == 1 {
            f.pad = self.b.tree(c, src.pad as u32, 7, F_RAW, 13, 0, 0, 0) as u8;
            if f.pad == 0 { return Err(bad()); }
        }
        f.tail = u16::try_from(self.count(c, src.tail as u32, 14, 0)).map_err(|_| bad())?;
        Ok(f)
    }

    /// Code (encode, or decode when `buf` is the decoder's) one leaf of frame records in
    /// `buf[range]`. `params`: [1 for ADTS frames, sampling-frequency index of a headerless run].
    pub fn code_leaf<C: Coder, B: ByteBuf>(&mut self, c: &mut C, buf: &mut B, range: std::ops::Range<usize>, params: &[u8]) -> Result<(), String> {
        let bad = || "aac frames: bad record (corrupt file)".to_string();
        let &[mode, sf] = params else { return Err("bad aac frame parameters".into()) };
        if mode > 1 || sf >= 12 { return Err("bad aac frame parameters".into()); }
        let (adts, enc) = (mode == 1, buf.reading());
        let mut pos = range.start;
        let mut out = Vec::new();
        while pos < range.end {
            let src = if enc { Some(Rec::read(&buf.hist(range.end)[pos..], adts, sf as usize)?) } else { None };
            let srec = src.as_ref().map(|s| &s.0);
            let mut fsf = sf as usize;
            let hdr = if adts {
                let sh = srec.and_then(|r| r.adts).unwrap_or_default();
                let mut h = [0xFFu8; 7];
                for k in 1..7 { h[k] = self.b.tree(c, sh.hdr[k] as u32, 8, F_HDR, 8 + k as u64, self.adts[k] as u64, 0, 0) as u8; }
                self.adts = h;
                let crc = if h[1] & 1 == 0 { Some([self.byte(c, sh.crc.unwrap_or_default()[0], 15, 0), self.byte(c, sh.crc.unwrap_or_default()[1], 16, 0)]) } else { None };
                let a = Adts { hdr: h, crc };
                fsf = a.sf();
                if fsf >= 12 { return Err(bad()); }
                Some(a)
            } else {
                None
            };
            let src_frame = match srec.map(|r| &r.body) { Some(Body::Parsed(f)) => Some(f), _ => None };
            let parsed = self.b.bit(c, src_frame.is_some() as u32, F_HDR, sc(20, 0, (self.flags & 1) as u64, 0, 0), 20, 1) == 1;
            self.flags = self.flags & !1 | parsed as u32;
            let body = if parsed {
                let blank = Frame::default();
                Body::Parsed(self.frame(c, src_frame.unwrap_or(&blank), fsf)?)
            } else {
                let len = match srec.map(|r| &r.body) { Some(Body::Raw(n)) => *n, _ => 0 };
                Body::Raw(u16::try_from(self.count(c, len as u32, 21, 0)).map_err(|_| bad())?)
            };
            let rec = Rec { adts: hdr, body };
            out.clear();
            rec.write(sf as usize, &mut out);
            if pos + out.len() > range.end { return Err("aac frames: leaf overflow (corrupt file)".into()); }
            if let Some((sr, n)) = &src {
                // the record must be the one the decoder will write
                if *n != out.len() || *sr != rec { return Err("aac frames: non-canonical record".into()); }
            } else {
                for (k, &x) in out.iter().enumerate() { buf.put(pos + k, x); }
            }
            pos += out.len();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cm::coder::{Decoder, Encoder};
    use crate::cm::predictor::{ReadBuf, WriteBuf};

    fn rng(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }

    /// The model decodes its own output back to the frame records, well under the frames' size;
    /// corrupt input is an error or other bytes, never a panic.
    #[test]
    fn frames_roundtrip_and_reject_corruption() {
        let adts: &[u8] = include_bytes!("../../tests/fixtures/tone_adts.aac");
        let m4a: &[u8] = include_bytes!("../../tests/fixtures/tone.m4a");
        let idx = crate::recode::mp4::index(&mut std::io::Cursor::new(m4a), m4a.len() as u64).unwrap();
        let run = &idx.runs[0];
        for (sp, params) in [(aac::split_adts(adts).unwrap(), [1u8, 0]), (aac::split_run(&m4a[run.start as usize..], &run.sizes, 4).unwrap(), [0, 4])] {
            let leaf = &sp.frames;
            let mut enc = Encoder::new();
            AacModel::new().code_leaf(&mut enc, &mut ReadBuf(leaf), 0..leaf.len(), &params).unwrap();
            let coded = enc.finish();
            assert!(coded.len() * 100 < sp.len * 90, "{} bytes of frames coded to {}", sp.len, coded.len());
            let mut back = vec![0u8; leaf.len()];
            AacModel::new().code_leaf(&mut Decoder::new(&coded), &mut WriteBuf(&mut back), 0..leaf.len(), &params).unwrap();
            assert!(&back == leaf, "the model did not decode its own output");
            let mut s = 4u64;
            for _ in 0..60 {
                let mut bad = coded.clone();
                let at = (rng(&mut s) % bad.len() as u64) as usize;
                bad[at] ^= 1 << (rng(&mut s) % 8);
                let mut out = vec![0u8; leaf.len()];
                let _ = AacModel::new().code_leaf(&mut Decoder::new(&bad), &mut WriteBuf(&mut out), 0..leaf.len(), &params);
            }
            // parameters that are not this leaf's are refused or decode to something else
            let mut out = vec![0u8; leaf.len()];
            assert!(AacModel::new().code_leaf(&mut Decoder::new(&coded), &mut WriteBuf(&mut out), 0..leaf.len(), &[2, 0]).is_err());
        }
    }
}
