//! Model for PCM audio leaves (WAV / AIFF / AU sample data).
//!
//! Samples are coded, not bytes. Stereo frames become mid/side; every channel is predicted by a
//! cascade of integer filters, each fed with the previous stage's residual:
//! 1. a fixed first-order filter,
//! 2. a short adaptive filter,
//! 3. a long and a short sign-sign NLMS filter.
//!
//! That is the model of decoder level 1. The second version (a bit of the leaf's parameters,
//! see [`setting`]) changes three things. Stage 2 becomes a filter of four taps over the
//! channel's last value and differences, and five over the other channel's, whose weights move
//! by one unit a sample: the coarse steps of the sign-sign filters leave a noise in the
//! prediction that a smooth signal (a piano, a quiet room) is quieter than. A filter of 1024
//! taps may stand before the long one. And a stereo pair is coded as mid and side or as left
//! and right, whichever the writer found shorter on a sample: two channels that have little in
//! common are each easier to predict than their sum and difference.
//!
//! The final residual is coded as its bit length (unary), the bits below the leading one and the
//! sign, each decision mixing a few adaptive contexts keyed by the channel's recent loudness.
//! Everything is integer arithmetic, so a file decodes identically on every machine. Bytes that do
//! not fill a whole frame (a segment tail) are coded with a small byte model.
use super::coder::Coder;
use super::mixer::Mixer;
use super::predictor::ByteBuf;
use super::squash::stretch;
use super::statemap::{Apm, StateMap};

const INPUTS: usize = 4;
const TABLE: usize = 1 << 16;
/// Decision families: residual length (unary), mantissa, sign, tail bytes.
const F_LEN: usize = 0;
const F_MANT: usize = 1;
const F_SIGN: usize = 2;
const F_TAIL: usize = 3;
const FAMILIES: usize = 4;

#[inline]
fn lg(x: u32) -> usize { (32 - x.leading_zeros()) as usize }

#[inline]
fn sat16(v: i32) -> i32 { v.clamp(-32768, 32767) }

/// Sign-sign NLMS filter over the stage's own past inputs. Histories live in roll buffers: values
/// are appended, and the newest `order` are copied back to the front when the buffer fills.
struct Nlms {
    order: usize,
    w: Vec<i32>,
    hist: Vec<i32>,
    delta: Vec<i32>, // adaptation step per history position
    pos: usize,      // one past the newest value
    shift: u32,
    avg: i32,
}

const ROLL: usize = 512;

impl Nlms {
    fn new(order: usize, shift: u32) -> Nlms {
        Nlms { order, w: vec![0; order], hist: vec![0; order + ROLL], delta: vec![0; order + ROLL], pos: order, shift, avg: 0 }
    }

    #[inline]
    fn predict(&self) -> i32 {
        let h = &self.hist[self.pos - self.order..self.pos];
        let dot: i64 = self.w.iter().zip(h).map(|(&w, &h)| w as i64 * h as i64).sum();
        ((dot + (1 << (self.shift - 1))) >> self.shift).clamp(-(1 << 24), 1 << 24) as i32
    }

    /// `input` entered the stage and `err` left it.
    #[inline]
    fn update(&mut self, input: i32, err: i32) {
        if err != 0 {
            let s = if err > 0 { 1 } else { -1 };
            let d = &self.delta[self.pos - self.order..self.pos];
            for (w, &d) in self.w.iter_mut().zip(d) { *w = (*w + s * d).clamp(-(1 << 20), 1 << 20); }
        }
        let x = sat16(input);
        let a = x.abs();
        let d = if a > self.avg * 3 { 32 } else if a > self.avg * 4 / 3 { 16 } else if a > 0 { 8 } else { 0 };
        self.avg += (a - self.avg) / 16;
        if self.pos == self.hist.len() {
            let n = self.order;
            self.hist.copy_within(self.pos - n.., 0);
            self.delta.copy_within(self.pos - n.., 0);
            self.pos = n;
        }
        self.hist[self.pos] = x;
        self.delta[self.pos] = d * x.signum();
        self.pos += 1;
        // older inputs adapt more gently
        let p = self.pos;
        if self.order >= 2 { self.delta[p - 2] >>= 1; }
        if self.order >= 3 { self.delta[p - 3] >>= 1; }
        if self.order >= 9 { self.delta[p - 9] >>= 1; }
    }
}

/// A short filter with fine steps, over this channel's last value and differences and the other
/// channel's: what a smooth signal is predicted by, before the long filters see the rest.
struct Fine {
    a: [i32; 4],
    b: [i32; 5],
    ha: [i32; 4],
    hb: [i32; 5],
    last_a: i32,
    last_b: i32,
}

impl Fine {
    fn new() -> Fine { Fine { a: [360, 317, -109, 98], b: [0; 5], ha: [0; 4], hb: [0; 5], last_a: 0, last_b: 0 } }

    /// Before predicting: the other channel's newest value.
    fn other(&mut self, v: i32) {
        self.hb.copy_within(1..4, 2);
        self.hb[1] = v - self.last_b;
        self.hb[0] = v;
        self.last_b = v;
    }

    fn predict(&self) -> i32 {
        let pa: i64 = self.a.iter().zip(&self.ha).map(|(&w, &h)| w as i64 * h as i64).sum();
        let pb: i64 = self.b.iter().zip(&self.hb).map(|(&w, &h)| w as i64 * h as i64).sum();
        ((pa + (pb >> 1)) >> 10).clamp(-(1 << 26), 1 << 26) as i32
    }

    /// `input` entered the stage and `err` left it.
    fn update(&mut self, input: i32, err: i32) {
        if err != 0 {
            let s = err.signum();
            for (w, h) in self.a.iter_mut().zip(&self.ha) { *w = (*w + s * h.signum()).clamp(-(1 << 16), 1 << 16); }
            for (w, h) in self.b.iter_mut().zip(&self.hb) { *w = (*w + s * h.signum()).clamp(-(1 << 16), 1 << 16); }
        }
        self.ha.copy_within(1..3, 2);
        self.ha[1] = input - self.last_a;
        self.ha[0] = input;
        self.last_a = input;
    }
}

/// Prediction cascade and residual statistics of one coded channel.
struct Chan {
    /// Filter inputs are scaled down by this (bits − 16 for 24-bit audio) to stay in 16-bit range.
    shift: u32,
    last: i32,
    /// Stage 2: the short filter of the first version, or the fine one of the second.
    short: Nlms,
    fine: Option<Fine>,
    /// Before the long filter, one of 1024 taps (second version, the higher levels).
    huge: Option<Nlms>,
    long: Nlms,
    tiny: Nlms,
    avg: u32, // running |residual| · 16
    prev_r: i32,
    // stage outputs of the current sample, kept between predict and update
    c1: i32,
    p2: i32,
    ph: i32,
    p3: i32,
    p4: i32,
    /// The last value after the first stage (what the other channel's fine filter sees).
    e1: i32,
}

impl Chan {
    fn new(long_order: usize, shift: u32, flags: u8) -> Chan {
        Chan {
            shift, last: 0, short: Nlms::new(8, 10), fine: (flags & FINE != 0).then(Fine::new),
            huge: (flags & HUGE != 0).then(|| Nlms::new(1024, 15)), long: Nlms::new(long_order, 13), tiny: Nlms::new(16, 11),
            avg: 16 * 64, prev_r: 0, c1: 0, p2: 0, ph: 0, p3: 0, p4: 0, e1: 0,
        }
    }

    /// Total prediction of the next value (all stages), remembering each stage's part.
    fn predict(&mut self) -> i32 {
        self.c1 = (self.last * 31) >> 5;
        self.p2 = match &self.fine { Some(f) => f.predict(), None => self.short.predict() << self.shift };
        self.ph = self.huge.as_ref().map_or(0, |h| h.predict() << self.shift);
        self.p3 = self.long.predict() << self.shift;
        self.p4 = self.tiny.predict() << self.shift;
        self.c1 + self.p2 + self.ph + self.p3 + self.p4
    }

    /// The true value was `x`, coded as residual `r`.
    fn update(&mut self, x: i32, r: i32) {
        let e1 = x - self.c1;
        let e2 = e1 - self.p2;
        let eh = e2 - self.ph;
        let e3 = eh - self.p3;
        let sh = self.shift;
        match self.fine.as_mut() { Some(f) => f.update(e1, e2), None => self.short.update(e1 >> sh, e2) }
        if let Some(h) = self.huge.as_mut() { h.update(e2 >> sh, eh); }
        self.long.update(eh >> sh, e3);
        self.tiny.update(e3 >> sh, r);
        self.e1 = e1;
        self.last = x;
        self.avg = (self.avg as i64 + ((r.unsigned_abs().min(1 << 24) as i64 * 16 - self.avg as i64) >> 4)) as u32;
        self.prev_r = r;
    }
}

pub struct AudioModel {
    sm: Vec<StateMap>,
    mixer: Mixer,
    apm: Apm,
}

impl Default for AudioModel {
    fn default() -> Self { Self::new() }
}

/// The model's setting, the last byte of a leaf's parameters: the long filter's order as
/// log2(order / 16) in its low three bits, and for the second version of the model [`FINE`]
/// with, as the writer chose, [`HUGE`] and [`APART`].
const HUGE: u8 = 8;
const FINE: u8 = 16;
/// A stereo pair is coded left and right, not mid and side.
const APART: u8 = 32;

/// Audio leaf params (see `detect::audio_params`): channels, bits, flags (bit 0 big-endian,
/// bit 1 unsigned), frame bytes, then the model's setting (added by the block encoder).
fn parse(p: &[u8]) -> Result<(usize, usize, bool, bool, usize, u8), String> {
    if p.len() != 5 { return Err("bad audio parameters".into()); }
    let (ch, bits) = (p[0] as usize, p[1] as usize);
    let (code, flags) = (p[4] & 7, p[4] & !7);
    let known = flags == 0 || (flags & FINE != 0 && flags & !(FINE | HUGE | APART) == 0);
    if !(1..=8).contains(&ch) || !matches!(bits, 8 | 16 | 24) || p[3] as usize != ch * bits / 8 || code > 6 || !known { return Err("bad audio parameters".into()); }
    Ok((ch, bits, p[2] & 1 != 0, p[2] & 2 != 0, 16 << code, flags))
}

/// Long-filter order code for a level (see [`parse`]).
pub fn order_code(level: u32) -> u8 { match level { 1 | 2 => 1, 3 | 4 => 3, 5 | 6 => 4, _ => 5 } }

/// A second-version setting with the filters `level` sets (what a recoder's leaf gets: its
/// stereo coding was chosen where the samples were at hand).
pub fn at_level(setting: u8, level: u32) -> u8 {
    if setting & FINE == 0 { return setting; }
    setting & APART | FINE | if level <= 4 { 3 } else { 4 | HUGE }
}

/// Bytes of a leaf the two ways of coding a stereo pair are tried on.
const SAMPLE: usize = 1 << 20;

/// The model's setting for the leaf `pcm` (detector params `params`) at `level`: for a decoder
/// of level 1 the first version's; otherwise the second's, with the stereo coding that is
/// shorter on a sample from the leaf's middle.
pub fn setting(level: u32, level2: bool, pcm: &[u8], params: &[u8]) -> u8 {
    if !level2 { return order_code(level); }
    let base = FINE | if level <= 4 { 3 } else { 4 | HUGE };
    let frame = params.get(3).copied().unwrap_or(0) as usize;
    if params.first() != Some(&2) || frame == 0 || pcm.len() < 64 * frame { return base; }
    let len = pcm.len().min(SAMPLE) / frame * frame;
    let from = (pcm.len() - len) / 2 / frame * frame;
    let cost = |s: u8| {
        let mut p = params.to_vec();
        p.push(s);
        let mut enc = super::coder::Encoder::new();
        match AudioModel::new().code_leaf(&mut enc, &mut super::predictor::ReadBuf(&pcm[from..from + len]), 0..len, &p) {
            Ok(()) => enc.finish().len(),
            Err(_) => usize::MAX,
        }
    };
    if cost(base | APART) < cost(base) { base | APART } else { base }
}

impl AudioModel {
    pub fn new() -> Self {
        AudioModel {
            sm: (0..FAMILIES * INPUTS).map(|_| StateMap::new(TABLE, 1023)).collect(),
            mixer: Mixer::new(INPUTS + 1, &[FAMILIES * 32], 1, 6, 6),
            apm: Apm::new(FAMILIES * 64, 7),
        }
    }

    #[inline]
    fn bit<C: Coder>(&mut self, c: &mut C, bit: u32, fam: usize, ctx: [usize; INPUTS], band: usize, actx: usize) -> u32 {
        for (i, &cx) in ctx.iter().enumerate() { self.mixer.add(stretch(self.sm[fam * INPUTS + i].p_at(cx & (TABLE - 1)))); }
        self.mixer.add(256);
        self.mixer.set(0, fam * 32 + band.min(31));
        let pm = self.mixer.p();
        let pa = self.apm.p(pm, fam * 64 + actx.min(63));
        let p = ((pm + 3 * pa) / 4).clamp(1, 65535);
        let b = c.code(bit, p);
        for (i, &cx) in ctx.iter().enumerate() { self.sm[fam * INPUTS + i].update_at(cx & (TABLE - 1), b); }
        self.mixer.update(b);
        self.apm.update(b);
        b
    }

    /// Code a residual: bit length in unary, the bits below the leading one, the sign.
    fn residual<C: Coder>(&mut self, c: &mut C, r: i32, ch: &Chan, ci: usize, other: usize) -> i32 {
        let k = lg(ch.avg >> 4).min(24);
        let pk = lg(ch.prev_r.unsigned_abs()).min(24);
        let m = r.unsigned_abs();
        let e = lg(m);
        let mut got = 0usize;
        while got < 25 {
            let more = self.bit(c, (e > got) as u32, F_LEN,
                [(ci * 25 + k) * 32 + got, ((k * 25 + pk) * 32) + got, ((k * 25 + other) * 32) + got, (ci * 32 + got) * 8 + (k.saturating_sub(got)).min(7)],
                k, got.min(15) * 4 + ci.min(3));
            if more == 0 { break; }
            got += 1;
        }
        let mut v = if got == 0 { 0 } else { 1u32 };
        for i in (0..got.saturating_sub(1)).rev() {
            let top = (got - 2 - i).min(2);
            let b = self.bit(c, (m >> i) & 1, F_MANT,
                [(got * 32 + i) * 4 + top, ((k * 26 + got) * 3 + top) * 2 + ((v & 1) as usize), (got * 32 + i) * 25 + k, ci * 64 + got],
                got, top * 16 + got.min(15));
            v = (v << 1) | b;
        }
        if v == 0 { return 0; }
        let sp = (ch.prev_r.signum() + 1) as usize;
        let neg = self.bit(c, (r < 0) as u32, F_SIGN, [ci * 3 + sp, (k * 3 + sp) * 26 + got, ci, (ci * 26 + got) * 3 + sp], sp, sp);
        if neg == 1 { -(v as i32) } else { v as i32 }
    }

    /// Code (encode, or decode when `buf` is the decoder's) one audio leaf in `buf[range]`.
    pub fn code_leaf<C: Coder, B: ByteBuf>(&mut self, c: &mut C, buf: &mut B, range: std::ops::Range<usize>, params: &[u8]) -> Result<(), String> {
        let (nch, bits, be, unsigned, long_order, flags) = parse(params)?;
        let bps = bits / 8;
        let frame = nch * bps;
        let enc = buf.reading();
        let stereo = nch == 2 && flags & APART == 0;
        let mut chans: Vec<Chan> = (0..nch).map(|_| Chan::new(long_order, bits.saturating_sub(16) as u32, flags)).collect();
        let n_frames = (range.end - range.start) / frame;
        let lim = 1i64 << (bits - 1);
        let mut vals = [0i32; 8];
        let mut pos = range.start;
        for _ in 0..n_frames {
            if enc {
                for (i, v) in vals.iter_mut().enumerate().take(nch) {
                    let o = pos + i * bps;
                    let mut u = 0u32;
                    for j in 0..bps {
                        let byte = buf.byte(o + if be { j } else { bps - 1 - j }) as u32;
                        u = (u << 8) | byte;
                    }
                    *v = if unsigned { u as i32 - (1 << (bits - 1)) } else { ((u << (32 - bits)) as i32) >> (32 - bits) };
                }
                if stereo {
                    let (l, r) = (vals[0], vals[1]);
                    let s = l - r;
                    vals[0] = r + (s >> 1);
                    vals[1] = s;
                }
            }
            let mut prev_k = 0usize;
            for ci in 0..nch {
                // the fine filter sees the channel before this one after its first stage: of this
                // very sample, but for the first channel
                if nch > 1 {
                    let v = chans[(ci + nch - 1) % nch].e1;
                    if let Some(f) = chans[ci].fine.as_mut() { f.other(v); }
                }
                let ch = &mut chans[ci];
                let p = ch.predict();
                let x = vals[ci];
                let r = self.residual(c, x.wrapping_sub(p), &chans[ci], ci, prev_k);
                let x = p.wrapping_add(r);
                // the side channel's range is one bit wider; anything else must stay in range
                let bound = if stereo && ci == 1 { 2 * lim } else { lim };
                if (x as i64) < -bound || (x as i64) >= bound { return Err("audio: sample out of range (corrupt data)".into()); }
                vals[ci] = x;
                prev_k = lg(chans[ci].avg >> 4).min(24);
                chans[ci].update(x, r);
            }
            if !enc {
                let mut out = vals;
                if stereo {
                    let (m, s) = (vals[0], vals[1]);
                    let r = m - (s >> 1);
                    out[0] = s + r;
                    out[1] = r;
                }
                for (i, &v) in out.iter().enumerate().take(nch) {
                    if (v as i64) < -lim || (v as i64) >= lim { return Err("audio: sample out of range (corrupt data)".into()); }
                    let u = if unsigned { (v + (1 << (bits - 1))) as u32 } else { v as u32 };
                    let o = pos + i * bps;
                    for j in 0..bps {
                        let byte = (u >> (8 * (bps - 1 - j))) as u8;
                        buf.put(o + if be { j } else { bps - 1 - j }, byte);
                    }
                }
            }
            pos += frame;
        }
        // a partial frame at the end
        while pos < range.end {
            let byte = if enc { buf.byte(pos) as u32 } else { 0 };
            let mut node = 1usize;
            for i in (0..8).rev() {
                let b = self.bit(c, (byte >> i) & 1, F_TAIL, [node, (pos - range.start) % frame * 256 + node, 0, node], 0, 0);
                node = node * 2 + b as usize;
            }
            if !enc { buf.put(pos, (node - 256) as u8); }
            pos += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cm::coder::{Decoder, Encoder};
    use crate::cm::predictor::{ReadBuf, WriteBuf};

    /// Synthetic tones with noise in several layouts: exact roundtrip, real compression, and a
    /// clean error (or exact data) on corruption.
    #[test]
    fn layouts_roundtrip_and_compress() {
        let mut s = 7u64;
        let mut noise = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s % 64) as i32 - 32 };
        // (channels, bits, big-endian, unsigned)
        for &(ch, bits, be, uns) in &[(2usize, 16usize, false, false), (1, 8, false, true), (2, 24, true, false), (1, 16, true, false)] {
            let bps = bits / 8;
            let frames = 20_000;
            let mut pcm = vec![];
            for n in 0..frames {
                for c in 0..ch {
                    let amp = (1i64 << (bits - 2)) as f64;
                    let v = (amp * ((n as f64) * 0.031 * (c as f64 + 1.0)).sin()) as i32 + noise() * (1i32 << (bits - 8)) / 64;
                    let u = if uns { (v + (1 << (bits - 1))) as u32 } else { v as u32 };
                    for j in 0..bps {
                        let byte = (u >> (8 * (bps - 1 - j))) as u8;
                        pcm.push(byte);
                    }
                    if !be { let l = pcm.len(); pcm[l - bps..].reverse(); }
                }
            }
            pcm.extend_from_slice(&[1, 2, 3][..(ch * bps - 1).min(3)]); // a partial frame
            let detected = crate::detect::audio_params(ch as u8, bits as u8, be, uns);
            // the first version of the model, and the second as a block of each level sets it
            let mut sizes = vec![];
            for setting in [order_code(5), setting(3, true, &pcm, &detected), setting(5, true, &pcm, &detected), setting(5, true, &pcm, &detected) ^ if ch == 2 { APART } else { 0 }] {
                let mut p = detected.clone();
                p.push(setting);
                let mut enc = Encoder::new();
                AudioModel::new().code_leaf(&mut enc, &mut ReadBuf(&pcm), 0..pcm.len(), &p).unwrap();
                let coded = enc.finish();
                assert!(coded.len() * 10 < pcm.len() * 8, "{ch}ch {bits}-bit, setting {setting:#x}: {} -> {}", pcm.len(), coded.len());
                let mut out = vec![0u8; pcm.len()];
                AudioModel::new().code_leaf(&mut Decoder::new(&coded), &mut WriteBuf(&mut out), 0..pcm.len(), &p).unwrap();
                assert!(out == pcm, "{ch}ch {bits}-bit roundtrip, setting {setting:#x}");
                let mut bad = coded.clone();
                bad[coded.len() / 3] ^= 0x10;
                let mut out2 = vec![0u8; pcm.len()];
                let _ = AudioModel::new().code_leaf(&mut Decoder::new(&bad), &mut WriteBuf(&mut out2), 0..pcm.len(), &p);
                sizes.push(coded.len());
            }
            // the second version is no worse on a tone under noise (what it is for is a signal
            // quieter than the noise the coarse filters make); and the writer took the shorter
            // of the two ways to code a pair
            assert!(sizes[2] <= sizes[0], "{ch}ch {bits}-bit: first version {} bytes, second {}", sizes[0], sizes[2]);
            assert!(sizes[2] <= sizes[3] + sizes[3] / 50, "{ch}ch {bits}-bit: the pair's coding was not the shorter: {} against {}", sizes[2], sizes[3]);
        }
        // settings no decoder knows are refused
        for bad in [7u8, 8, 0x40, 0x80, HUGE | 4, APART | 4] {
            let mut p = crate::detect::audio_params(2, 16, false, false);
            p.push(bad);
            assert!(parse(&p).is_err(), "setting {bad:#x} accepted");
        }
        assert_eq!((setting(5, false, &[], &[2, 16, 0, 4]), setting(3, true, &[], &[2, 16, 0, 4]), setting(9, true, &[], &[1, 16, 0, 2])), (4, FINE | 3, FINE | HUGE | 4));
    }
}
