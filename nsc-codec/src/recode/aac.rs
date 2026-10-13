//! AAC (MPEG-2/4 AAC-LC, 1024-sample frames) recoder.
//!
//! A run of AAC frames — raw data blocks one after another as an MP4 track stores them, or an
//! ADTS stream — is taken apart into
//! * the frames as parsed records ([`Frame`]): their syntactic elements down to section code
//!   books, scale-factor codes and Huffman-decoded spectral values, which the AAC model codes
//!   instead of the bytes;
//! * the bytes left as they are: frames that do not parse (other profiles, coupling or program
//!   configuration elements, damage), and whatever follows a frame's end marker.
//!
//! Fill and data-stream elements keep their payload bytes inside the record (spectral band
//! replication data of HE-AAC lives there). An ADTS header's frame length follows from the
//! re-encoded frame and is not kept. [`super::proven`] accepts a split only when [`rebuild`]
//! reproduces the run exactly, so the parser only has to be a bijection on what it accepts.
use super::aac_tables::{SCALEFACTOR, SHAPE, SPECTRAL, SWB_LONG, SWB_SHORT};
use crate::bits::{BitReader, BitWriter, PrefixDec};
use crate::transform::deflate::inflate::ExtentError;
use std::sync::OnceLock;

/// Fewest consecutive ADTS frames taken as a stream.
const MIN_FRAMES: usize = 4;
/// Frame records per leaf (leaves are what lanes are made of).
const LEAF_FRAMES: usize = 256;
/// Spectral lines per frame.
pub const LINES: usize = 1024;
const ZERO_HCB: u8 = 0;
const ESC_HCB: u8 = 11;
const NOISE_HCB: u8 = 13;
const INTENSITY_HCB: u8 = 15;
/// Bit 7 of a band's code-book byte: a section starts at this band.
pub const SECTION_START: u8 = 0x80;

// ------------------------------------------------------------------ parsed frames
/// Window layout of a channel (`ics_info`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Info {
    /// 0 only-long, 1 long-start, 2 eight-short, 3 long-stop.
    pub seq: u8,
    pub shape: u8,
    pub max_sfb: u8,
    /// Eight-short: bit 6 - k set when window k + 1 continues window k's group.
    pub grouping: u8,
}

impl Info {
    pub fn short(&self) -> bool { self.seq == 2 }

    /// Lengths (in windows) of the window groups.
    pub fn groups(&self) -> Vec<u8> {
        if !self.short() { return vec![1]; }
        let mut g = vec![1u8];
        for k in 0..7 {
            if self.grouping & (0x40 >> k) != 0 { *g.last_mut().unwrap() += 1; } else { g.push(1); }
        }
        g
    }

    /// Scale-factor band offsets for sampling-frequency index `sf`.
    pub fn offsets(&self, sf: usize) -> &'static [u16] { if self.short() { SWB_SHORT[sf] } else { SWB_LONG[sf] } }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Pulse {
    pub start: u8,
    /// (offset, amplitude) of each pulse (1..=4).
    pub pulses: Vec<(u8, u8)>,
}

/// One channel's data (`individual_channel_stream`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Ics {
    pub gain: u8,
    pub info: Info,
    /// Per (group, band below `max_sfb`): the code book, with [`SECTION_START`] where a section
    /// begins.
    pub cb: Vec<u8>,
    /// The scale-factor code of every band that has one (0..=120; the first noise band: 0..=511).
    pub sf: Vec<u16>,
    pub pulse: Option<Pulse>,
    /// Temporal noise shaping data, as its bits (count, bytes with the last one right-aligned).
    pub tns: Option<(u16, Vec<u8>)>,
    /// The spectral values in stream order (group, band, window, line).
    pub spec: Vec<i16>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Element {
    /// A single channel (`lfe`: the low-frequency one).
    Single { lfe: bool, tag: u8, ics: Ics },
    /// A channel pair; with `common` both channels share the first one's window layout and
    /// `ms_used` (one byte per (group, band) when `ms_present` is 1).
    Pair { tag: u8, common: bool, ms_present: u8, ms_used: Vec<u8>, ics: Box<[Ics; 2]> },
    /// Data stream element: its alignment bits (when aligned) and payload.
    Data { tag: u8, align: bool, pad: u8, data: Vec<u8> },
    /// Fill element payload (extension data such as SBR).
    Fill { data: Vec<u8> },
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Frame {
    pub elements: Vec<Element>,
    /// The bits between the end marker and the byte boundary.
    pub pad: u8,
    /// Bytes after the frame's end that belong to it (kept among the raw bytes).
    pub tail: u16,
}

/// An ADTS header with its frame length zeroed, and the CRC bytes of a protected frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Adts {
    pub hdr: [u8; 7],
    pub crc: Option<[u8; 2]>,
}

impl Adts {
    pub fn len(&self) -> usize { 7 + 2 * self.crc.is_some() as usize }
    pub fn sf(&self) -> usize { ((self.hdr[2] >> 2) & 15) as usize }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Body {
    /// The frame's payload is `len` of the raw bytes.
    Raw(u16),
    Parsed(Frame),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Rec {
    pub adts: Option<Adts>,
    pub body: Body,
}

struct Decoders {
    spectral: Vec<PrefixDec>,
    sf: PrefixDec,
}

fn decoders() -> &'static Decoders {
    static D: OnceLock<Decoders> = OnceLock::new();
    D.get_or_init(|| Decoders {
        spectral: SPECTRAL.iter().map(|b| PrefixDec::new(b.iter().enumerate().map(|(i, c)| (c.0, c.1, i as u32)))).collect(),
        sf: PrefixDec::new(SCALEFACTOR.iter().enumerate().map(|(i, c)| (c.0, c.1, i as u32))),
    })
}

/// Does a band with code book `cb` carry a scale factor / spectral values?
pub fn has_sf(cb: u8) -> bool { cb != ZERO_HCB }
pub fn has_spec(cb: u8) -> bool { (1..=ESC_HCB).contains(&cb) }

fn valid_cb(cb: u8) -> bool { cb <= ESC_HCB || (NOISE_HCB..=INTENSITY_HCB).contains(&cb) }

/// The symbol index of a tuple of book `cb` (1..=11) and back.
fn tuple_index(cb: u8, v: &[i16]) -> Option<usize> {
    let (dim, signed, lav) = SHAPE[cb as usize - 1];
    let lav = lav as i32;
    let base = if signed { 2 * lav + 1 } else { lav + 1 } as usize;
    let mut i = 0usize;
    for &x in &v[..dim] {
        let d = if signed { x as i32 + lav } else { (x.unsigned_abs() as i32).min(lav) };
        if !(0..base as i32).contains(&d) { return None; }
        i = i * base + d as usize;
    }
    Some(i)
}

fn tuple_values(cb: u8, mut i: usize, out: &mut [i16; 4]) -> usize {
    let (dim, signed, lav) = SHAPE[cb as usize - 1];
    let base = if signed { 2 * lav as usize + 1 } else { lav as usize + 1 };
    for k in (0..dim).rev() {
        let d = (i % base) as i16;
        out[k] = if signed { d - lav as i16 } else { d };
        i /= base;
    }
    dim
}

// ------------------------------------------------------------------ bit-level parse and write
fn read_info(r: &mut BitReader, sf: usize) -> Option<Info> {
    if r.get(1) != 0 { return None; } // ics_reserved_bit
    let mut i = Info { seq: r.get(2) as u8, shape: r.get(1) as u8, ..Default::default() };
    if i.short() {
        i.max_sfb = r.get(4) as u8;
        i.grouping = r.get(7) as u8;
    } else {
        i.max_sfb = r.get(6) as u8;
        if r.get(1) != 0 { return None; } // predictor data: not AAC-LC
    }
    (i.max_sfb as usize <= i.offsets(sf).len() - 1).then_some(i)
}

fn write_info(w: &mut BitWriter, i: &Info) {
    w.put(0, 1);
    w.put(i.seq as u32, 2);
    w.put(i.shape as u32, 1);
    if i.short() {
        w.put(i.max_sfb as u32, 4);
        w.put(i.grouping as u32, 7);
    } else {
        w.put(i.max_sfb as u32, 6);
        w.put(0, 1);
    }
}

/// Number of lines of band `sfb` in a group of `glen` windows.
#[inline]
fn band_lines(off: &[u16], sfb: usize, glen: u8) -> usize { (off[sfb + 1] - off[sfb]) as usize * glen as usize }

fn read_ics(r: &mut BitReader, sf: usize, common: Option<Info>, limit: usize) -> Option<Ics> {
    let mut s = Ics { gain: r.get(8) as u8, ..Default::default() };
    s.info = match common { Some(i) => i, None => read_info(r, sf)? };
    let (groups, max_sfb, off) = (s.info.groups(), s.info.max_sfb as usize, s.info.offsets(sf));
    // section data
    let bits = if s.info.short() { 3 } else { 5 };
    let esc = (1u32 << bits) - 1;
    for _ in 0..groups.len() {
        let mut k = 0usize;
        while k < max_sfb {
            let cb = r.get(4) as u8;
            if !valid_cb(cb) { return None; }
            let mut len = 0usize;
            loop {
                let incr = r.get(bits);
                len += incr as usize;
                if incr != esc { break; }
                if len > max_sfb { return None; }
            }
            if len == 0 || k + len > max_sfb || r.pos() > limit { return None; }
            for j in 0..len { s.cb.push(cb | if j == 0 { SECTION_START } else { 0 }); }
            k += len;
        }
    }
    // scale factors
    let d = decoders();
    let mut noise_pcm = true;
    for &c in &s.cb {
        let cb = c & 15;
        if !has_sf(cb) { continue; }
        if cb == NOISE_HCB && noise_pcm {
            noise_pcm = false;
            s.sf.push(r.get(9) as u16);
        } else {
            let (v, n) = d.sf.peek(r)?;
            r.skip(n);
            s.sf.push(v as u16);
        }
    }
    if r.pos() > limit { return None; }
    if r.get(1) == 1 {
        if s.info.short() { return None; }
        let n = r.get(2) as usize + 1;
        let mut p = Pulse { start: r.get(6) as u8, pulses: Vec::with_capacity(n) };
        for _ in 0..n { p.pulses.push((r.get(5) as u8, r.get(4) as u8)); }
        s.pulse = Some(p);
    }
    if r.get(1) == 1 {
        // temporal noise shaping: parsed for its length, kept as bits
        let start = r.pos();
        let (windows, nb) = if s.info.short() { (8, [1u32, 4, 3]) } else { (1, [2u32, 6, 5]) };
        for _ in 0..windows {
            let n_filt = r.get(nb[0]);
            if n_filt == 0 { continue; }
            let coef_res = r.get(1);
            for _ in 0..n_filt {
                r.skip(nb[1]);
                let order = r.get(nb[2]);
                if order > 20 { return None; }
                if order != 0 {
                    r.skip(1);
                    let coef_len = coef_res + 3 - r.get(1);
                    r.skip(order * coef_len);
                }
            }
        }
        let n = r.pos() - start;
        if n > 4000 || r.pos() > limit { return None; }
        let mut rr = r.at(start);
        s.tns = Some((n as u16, (0..n.div_ceil(8)).map(|i| rr.get((n - 8 * i).min(8) as u32) as u8).collect()));
    }
    if r.get(1) == 1 { return None; } // gain control: SSR profile
    // spectral data
    let mut tuple = [0i16; 4];
    let mut idx = 0usize;
    for &glen in &groups {
        for sfb in 0..max_sfb {
            let cb = s.cb[idx] & 15;
            idx += 1;
            if !has_spec(cb) { continue; }
            let dec = &d.spectral[cb as usize - 1];
            let (dim, signed, _) = SHAPE[cb as usize - 1];
            for _ in 0..band_lines(off, sfb, glen) / dim {
                let (sym, n) = dec.peek(r)?;
                r.skip(n);
                tuple_values(cb, sym as usize, &mut tuple);
                if !signed {
                    for v in tuple[..dim].iter_mut() {
                        if *v != 0 && r.get(1) == 1 { *v = -*v; }
                    }
                }
                if cb == ESC_HCB {
                    for v in tuple[..2].iter_mut() {
                        if v.unsigned_abs() == 16 {
                            let mut n = 4;
                            while r.get(1) == 1 {
                                n += 1;
                                if n > 12 { return None; }
                            }
                            let e = (r.get(n) | 1 << n) as i16;
                            *v = if *v < 0 { -e } else { e };
                        }
                    }
                }
                s.spec.extend_from_slice(&tuple[..dim]);
            }
            if r.pos() > limit { return None; }
        }
    }
    Some(s)
}

fn write_ics(w: &mut BitWriter, s: &Ics, sf: usize, own_info: bool) -> Result<(), String> {
    let bad = || "aac: values outside the code books (corrupt file)".to_string();
    w.put(s.gain as u32, 8);
    if own_info { write_info(w, &s.info); }
    let (groups, max_sfb, off) = (s.info.groups(), s.info.max_sfb as usize, s.info.offsets(sf));
    if s.cb.len() != groups.len() * max_sfb || max_sfb > off.len() - 1 { return Err(bad()); }
    let bits = if s.info.short() { 3 } else { 5 };
    let esc = (1usize << bits) - 1;
    for g in 0..groups.len() {
        let row = &s.cb[g * max_sfb..(g + 1) * max_sfb];
        let mut k = 0;
        while k < max_sfb {
            if row[k] & SECTION_START == 0 || !valid_cb(row[k] & 15) || row[k] & 0x70 != 0 { return Err(bad()); }
            let mut len = 1;
            while k + len < max_sfb && row[k + len] & SECTION_START == 0 {
                if row[k + len] != row[k] & 15 { return Err(bad()); }
                len += 1;
            }
            w.put((row[k] & 15) as u32, 4);
            let mut left = len;
            while left >= esc {
                w.put(esc as u32, bits);
                left -= esc;
            }
            w.put(left as u32, bits);
            k += len;
        }
    }
    let mut sfs = s.sf.iter();
    let mut noise_pcm = true;
    for &c in &s.cb {
        let cb = c & 15;
        if !has_sf(cb) { continue; }
        let v = *sfs.next().ok_or_else(bad)? as usize;
        if cb == NOISE_HCB && noise_pcm {
            noise_pcm = false;
            if v > 511 { return Err(bad()); }
            w.put(v as u32, 9);
        } else {
            let (code, n) = *SCALEFACTOR.get(v).ok_or_else(bad)?;
            w.put(code, n as u32);
        }
    }
    if sfs.next().is_some() { return Err(bad()); }
    w.put(s.pulse.is_some() as u32, 1);
    if let Some(p) = &s.pulse {
        if s.info.short() || p.pulses.is_empty() || p.pulses.len() > 4 || p.start > 63 { return Err(bad()); }
        w.put(p.pulses.len() as u32 - 1, 2);
        w.put(p.start as u32, 6);
        for &(o, a) in &p.pulses {
            if o > 31 || a > 15 { return Err(bad()); }
            w.put(o as u32, 5);
            w.put(a as u32, 4);
        }
    }
    w.put(s.tns.is_some() as u32, 1);
    if let Some((n, bytes)) = &s.tns {
        let n = *n as usize;
        if bytes.len() != n.div_ceil(8) { return Err(bad()); }
        for (i, &b) in bytes.iter().enumerate() {
            let k = (n - 8 * i).min(8) as u32;
            if k < 8 && b as u32 >> k != 0 { return Err(bad()); }
            w.put(b as u32, k);
        }
    }
    w.put(0, 1);
    let mut spec = s.spec.iter();
    let mut idx = 0usize;
    let mut tuple = [0i16; 4];
    for &glen in &groups {
        for sfb in 0..max_sfb {
            let cb = s.cb[idx] & 15;
            idx += 1;
            if !has_spec(cb) { continue; }
            let (dim, signed, lav) = SHAPE[cb as usize - 1];
            for _ in 0..band_lines(off, sfb, glen) / dim {
                for t in tuple[..dim].iter_mut() { *t = *spec.next().ok_or_else(bad)?; }
                for &v in &tuple[..dim] {
                    let a = v.unsigned_abs();
                    if (cb != ESC_HCB && a > lav as u16) || a > 8191 { return Err(bad()); }
                }
                let (code, n) = SPECTRAL[cb as usize - 1][tuple_index(cb, &tuple).ok_or_else(bad)?];
                w.put(code, n as u32);
                if !signed {
                    for &v in &tuple[..dim] {
                        if v != 0 { w.put((v < 0) as u32, 1); }
                    }
                }
                if cb == ESC_HCB {
                    for &v in &tuple[..2] {
                        let a = v.unsigned_abs() as u32;
                        if a >= 16 {
                            let n = 31 - a.leading_zeros(); // 4..=12
                            w.put((1 << (n - 4)) - 1, n - 4);
                            w.put(0, 1);
                            w.put(a & ((1 << n) - 1), n);
                        }
                    }
                }
            }
        }
    }
    if spec.next().is_some() { return Err(bad()); }
    Ok(())
}

/// Parse one frame (a raw data block and what follows it) in `b`, for sampling-frequency index
/// `sf`. None if it is not elements this recoder knows, ending within `b`.
pub fn read_frame(b: &[u8], sf: usize) -> Option<Frame> {
    if sf >= SWB_LONG.len() { return None; }
    let limit = b.len() * 8;
    let mut r = BitReader::new(b, 0);
    let mut f = Frame::default();
    loop {
        if r.pos() + 3 > limit || f.elements.len() >= 64 { return None; }
        match r.get(3) {
            id @ (0 | 3) => {
                let tag = r.get(4) as u8;
                f.elements.push(Element::Single { lfe: id == 3, tag, ics: read_ics(&mut r, sf, None, limit)? });
            }
            1 => {
                let tag = r.get(4) as u8;
                let common = r.get(1) == 1;
                let (mut info, mut ms_present, mut ms_used) = (None, 0u8, vec![]);
                if common {
                    let i = read_info(&mut r, sf)?;
                    ms_present = r.get(2) as u8;
                    if ms_present == 3 { return None; }
                    if ms_present == 1 {
                        ms_used = (0..i.groups().len() * i.max_sfb as usize).map(|_| r.get(1) as u8).collect();
                    }
                    info = Some(i);
                }
                let a = read_ics(&mut r, sf, info, limit)?;
                let c = read_ics(&mut r, sf, info, limit)?;
                f.elements.push(Element::Pair { tag, common, ms_present, ms_used, ics: Box::new([a, c]) });
            }
            4 => {
                let tag = r.get(4) as u8;
                let align = r.get(1) == 1;
                let mut count = r.get(8) as usize;
                if count == 255 { count += r.get(8) as usize; }
                let mut pad = 0;
                if align {
                    let n = (8 - r.pos() % 8) % 8;
                    pad = r.get(n as u32) as u8;
                }
                if r.pos() + 8 * count > limit { return None; }
                f.elements.push(Element::Data { tag, align, pad, data: (0..count).map(|_| r.get(8) as u8).collect() });
            }
            6 => {
                let mut count = r.get(4) as usize;
                if count == 15 { count = 14 + r.get(8) as usize; }
                if r.pos() + 8 * count > limit { return None; }
                f.elements.push(Element::Fill { data: (0..count).map(|_| r.get(8) as u8).collect() });
            }
            7 => break,
            _ => return None, // coupling channel and program configuration elements
        }
        if r.pos() > limit { return None; }
    }
    let n = (8 - r.pos() % 8) % 8;
    f.pad = r.get(n as u32) as u8;
    if r.pos() > limit { return None; }
    f.tail = u16::try_from(b.len() - r.pos() / 8).ok()?;
    Some(f)
}

/// The bytes of a frame up to its tail.
pub fn write_frame(f: &Frame, sf: usize) -> Result<Vec<u8>, String> {
    let bad = || "aac: bad frame (corrupt file)".to_string();
    if sf >= SWB_LONG.len() { return Err(bad()); }
    let mut w = BitWriter::new();
    for e in &f.elements {
        match e {
            Element::Single { lfe, tag, ics } => {
                w.put(if *lfe { 3 } else { 0 }, 3);
                w.put(*tag as u32 & 15, 4);
                write_ics(&mut w, ics, sf, true)?;
            }
            Element::Pair { tag, common, ms_present, ms_used, ics } => {
                w.put(1, 3);
                w.put(*tag as u32 & 15, 4);
                w.put(*common as u32, 1);
                if *common {
                    if ics[1].info != ics[0].info || *ms_present > 2 { return Err(bad()); }
                    write_info(&mut w, &ics[0].info);
                    w.put(*ms_present as u32, 2);
                    if *ms_present == 1 {
                        if ms_used.len() != ics[0].info.groups().len() * ics[0].info.max_sfb as usize { return Err(bad()); }
                        for &m in ms_used { w.put(m as u32 & 1, 1); }
                    }
                }
                write_ics(&mut w, &ics[0], sf, !*common)?;
                write_ics(&mut w, &ics[1], sf, !*common)?;
            }
            Element::Data { tag, align, pad, data } => {
                if data.len() > 510 { return Err(bad()); }
                w.put(4, 3);
                w.put(*tag as u32 & 15, 4);
                w.put(*align as u32, 1);
                if data.len() >= 255 {
                    w.put(255, 8);
                    w.put(data.len() as u32 - 255, 8);
                } else {
                    w.put(data.len() as u32, 8);
                }
                if *align {
                    let n = (8 - w.len() % 8) % 8;
                    if *pad as u32 >> n != 0 { return Err(bad()); }
                    w.put(*pad as u32, n as u32);
                }
                for &x in data { w.put(x as u32, 8); }
            }
            Element::Fill { data } => {
                if data.len() > 269 { return Err(bad()); }
                w.put(6, 3);
                if data.len() >= 15 {
                    w.put(15, 4);
                    w.put(data.len() as u32 - 14, 8);
                } else {
                    w.put(data.len() as u32, 4);
                }
                for &x in data { w.put(x as u32, 8); }
            }
        }
    }
    w.put(7, 3);
    let n = (8 - w.len() % 8) % 8;
    if f.pad as u32 >> n != 0 { return Err(bad()); }
    w.put(f.pad as u32, n as u32);
    Ok(w.into_bytes())
}

// ------------------------------------------------------------------ the frame records (leaf bytes)
struct Cur<'a> {
    b: &'a [u8],
    o: usize,
}

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let s = self.b.get(self.o..self.o + n).ok_or("aac: truncated frame record (corrupt file)")?;
        self.o += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> { Ok(self.take(1)?[0]) }
    fn u16(&mut self) -> Result<u16, String> { Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap())) }
}

fn put_info(out: &mut Vec<u8>, i: &Info) { out.extend_from_slice(&[i.seq, i.shape, i.max_sfb, i.grouping]); }

fn get_info(c: &mut Cur, sf: usize) -> Result<Info, String> {
    let i = Info { seq: c.u8()?, shape: c.u8()?, max_sfb: c.u8()?, grouping: c.u8()? };
    if i.seq > 3 || i.shape > 1 || i.grouping > 127 || (!i.short() && i.grouping != 0) || i.max_sfb as usize > i.offsets(sf).len() - 1 {
        return Err("aac: bad window layout (corrupt file)".into());
    }
    Ok(i)
}

/// Number of spectral values a channel with these code books carries.
pub fn spec_len(info: &Info, cb: &[u8], sf: usize) -> usize {
    let (off, max_sfb) = (info.offsets(sf), info.max_sfb as usize);
    info.groups().iter().enumerate().map(|(g, &glen)| {
        (0..max_sfb).filter(|&s| has_spec(cb[g * max_sfb + s] & 15)).map(|s| band_lines(off, s, glen)).sum::<usize>()
    }).sum()
}

impl Rec {
    /// Append the record. `sf`: sampling-frequency index of a headerless run (an ADTS frame
    /// carries its own).
    pub fn write(&self, sf: usize, out: &mut Vec<u8>) {
        let sf = self.adts.map_or(sf, |a| a.sf());
        if let Some(a) = &self.adts {
            out.extend_from_slice(&a.hdr);
            out.push(a.crc.is_some() as u8);
            out.extend_from_slice(&a.crc.unwrap_or_default()[..2 * a.crc.is_some() as usize]);
        }
        match &self.body {
            Body::Raw(len) => {
                out.push(0);
                out.extend_from_slice(&len.to_le_bytes());
            }
            Body::Parsed(f) => {
                out.push(1);
                out.push(f.elements.len() as u8);
                for e in &f.elements {
                    match e {
                        Element::Single { lfe, tag, ics } => {
                            out.extend_from_slice(&[if *lfe { 3 } else { 0 }, *tag]);
                            write_ics_rec(out, ics, sf, true);
                        }
                        Element::Pair { tag, common, ms_present, ms_used, ics } => {
                            out.extend_from_slice(&[1, *tag, *common as u8]);
                            if *common {
                                put_info(out, &ics[0].info);
                                out.push(*ms_present);
                                out.extend_from_slice(ms_used);
                            }
                            write_ics_rec(out, &ics[0], sf, !*common);
                            write_ics_rec(out, &ics[1], sf, !*common);
                        }
                        Element::Data { tag, align, pad, data } => {
                            out.extend_from_slice(&[4, *tag, *align as u8, *pad]);
                            out.extend_from_slice(&(data.len() as u16).to_le_bytes());
                            out.extend_from_slice(data);
                        }
                        Element::Fill { data } => {
                            out.push(6);
                            out.extend_from_slice(&(data.len() as u16).to_le_bytes());
                            out.extend_from_slice(data);
                        }
                    }
                }
                out.push(f.pad);
                out.extend_from_slice(&f.tail.to_le_bytes());
            }
        }
    }

    /// Parse the record at the start of `b`: the record and its length. `adts`: the run's frames
    /// have ADTS headers; `sf`: sampling-frequency index of a headerless run.
    pub fn read(b: &[u8], adts: bool, sf: usize) -> Result<(Rec, usize), String> {
        let bad = || "aac: bad frame record (corrupt file)".to_string();
        let mut c = Cur { b, o: 0 };
        let mut sf = sf;
        let hdr = if adts {
            let hdr: [u8; 7] = c.take(7)?.try_into().unwrap();
            let crc = match c.u8()? { 0 => None, 1 => Some(c.take(2)?.try_into().unwrap()), _ => return Err(bad()) };
            let a = Adts { hdr, crc };
            if !adts_ok(&hdr) || adts_len(&hdr) != 0 || (hdr[1] & 1 == 0) != crc.is_some() { return Err(bad()); }
            sf = a.sf();
            Some(a)
        } else {
            None
        };
        if sf >= SWB_LONG.len() { return Err(bad()); }
        let body = match c.u8()? {
            0 => Body::Raw(c.u16()?),
            1 => {
                let n = c.u8()? as usize;
                if n > 64 { return Err(bad()); }
                let mut f = Frame::default();
                for _ in 0..n {
                    f.elements.push(match c.u8()? {
                        id @ (0 | 3) => Element::Single { lfe: id == 3, tag: c.u8()?, ics: read_ics_rec(&mut c, sf, None)? },
                        1 => {
                            let (tag, common) = (c.u8()?, c.u8()?);
                            if common > 1 { return Err(bad()); }
                            let (mut info, mut ms_present, mut ms_used) = (None, 0, vec![]);
                            if common == 1 {
                                let i = get_info(&mut c, sf)?;
                                ms_present = c.u8()?;
                                if ms_present > 2 { return Err(bad()); }
                                if ms_present == 1 {
                                    ms_used = c.take(i.groups().len() * i.max_sfb as usize)?.to_vec();
                                    if ms_used.iter().any(|&m| m > 1) { return Err(bad()); }
                                }
                                info = Some(i);
                            }
                            let a = read_ics_rec(&mut c, sf, info)?;
                            let d = read_ics_rec(&mut c, sf, info)?;
                            Element::Pair { tag, common: common == 1, ms_present, ms_used, ics: Box::new([a, d]) }
                        }
                        4 => {
                            let (tag, align, pad) = (c.u8()?, c.u8()?, c.u8()?);
                            let n = c.u16()? as usize;
                            if align > 1 || n > 510 { return Err(bad()); }
                            Element::Data { tag, align: align == 1, pad, data: c.take(n)?.to_vec() }
                        }
                        6 => {
                            let n = c.u16()? as usize;
                            if n > 269 { return Err(bad()); }
                            Element::Fill { data: c.take(n)?.to_vec() }
                        }
                        _ => return Err(bad()),
                    });
                }
                f.pad = c.u8()?;
                f.tail = c.u16()?;
                Body::Parsed(f)
            }
            _ => return Err(bad()),
        };
        Ok((Rec { adts: hdr, body }, c.o))
    }
}

fn write_ics_rec(out: &mut Vec<u8>, s: &Ics, sf: usize, own_info: bool) {
    out.push(s.gain);
    if own_info { put_info(out, &s.info); }
    out.extend_from_slice(&s.cb);
    let mut noise_pcm = true;
    let mut sfs = s.sf.iter();
    for &c in &s.cb {
        let cb = c & 15;
        if !has_sf(cb) { continue; }
        let v = sfs.next().copied().unwrap_or(0);
        if cb == NOISE_HCB && noise_pcm {
            noise_pcm = false;
            out.extend_from_slice(&v.to_le_bytes());
        } else {
            out.push(v as u8);
        }
    }
    match &s.pulse {
        None => out.push(0),
        Some(p) => {
            out.push(p.pulses.len() as u8);
            out.push(p.start);
            for &(o, a) in &p.pulses { out.extend_from_slice(&[o, a]); }
        }
    }
    match &s.tns {
        None => out.push(0),
        Some((n, bytes)) => {
            out.push(1);
            out.extend_from_slice(&n.to_le_bytes());
            out.extend_from_slice(bytes);
        }
    }
    // spectral values band by band: only book 11 has escapes
    let (off, max_sfb) = (s.info.offsets(sf), s.info.max_sfb as usize);
    let mut k = 0usize;
    for (g, &glen) in s.info.groups().iter().enumerate() {
        for sfb in 0..max_sfb {
            let cb = s.cb[g * max_sfb + sfb] & 15;
            if !has_spec(cb) { continue; }
            for _ in 0..band_lines(off, sfb, glen) {
                let v = s.spec.get(k).copied().unwrap_or(0);
                k += 1;
                let a = v.unsigned_abs();
                out.push(a.min(16) as u8 | ((v < 0) as u8) << 7);
                if cb == ESC_HCB && a >= 16 { out.extend_from_slice(&a.to_le_bytes()); }
            }
        }
    }
}

fn read_ics_rec(c: &mut Cur, sf: usize, common: Option<Info>) -> Result<Ics, String> {
    let bad = || "aac: bad channel record (corrupt file)".to_string();
    let mut s = Ics { gain: c.u8()?, ..Default::default() };
    s.info = match common { Some(i) => i, None => get_info(c, sf)? };
    let groups = s.info.groups();
    let (off, max_sfb) = (s.info.offsets(sf), s.info.max_sfb as usize);
    s.cb = c.take(groups.len() * max_sfb)?.to_vec();
    for g in 0..groups.len() {
        for sfb in 0..max_sfb {
            let x = s.cb[g * max_sfb + sfb];
            if x & 0x70 != 0 || !valid_cb(x & 15) || (sfb == 0 && x & SECTION_START == 0) { return Err(bad()); }
            if sfb > 0 && x & SECTION_START == 0 && x != s.cb[g * max_sfb + sfb - 1] & 15 { return Err(bad()); }
        }
    }
    let mut noise_pcm = true;
    for i in 0..s.cb.len() {
        let cb = s.cb[i] & 15;
        if !has_sf(cb) { continue; }
        if cb == NOISE_HCB && noise_pcm {
            noise_pcm = false;
            let v = c.u16()?;
            if v > 511 { return Err(bad()); }
            s.sf.push(v);
        } else {
            let v = c.u8()?;
            if v > 120 { return Err(bad()); }
            s.sf.push(v as u16);
        }
    }
    let n = c.u8()? as usize;
    if n > 4 || (n > 0 && s.info.short()) { return Err(bad()); }
    if n > 0 {
        let mut p = Pulse { start: c.u8()?, pulses: Vec::with_capacity(n) };
        for _ in 0..n { p.pulses.push((c.u8()?, c.u8()?)); }
        if p.start > 63 || p.pulses.iter().any(|&(o, a)| o > 31 || a > 15) { return Err(bad()); }
        s.pulse = Some(p);
    }
    match c.u8()? {
        0 => {}
        1 => {
            let n = c.u16()?;
            if n > 4000 { return Err(bad()); }
            s.tns = Some((n, c.take((n as usize).div_ceil(8))?.to_vec()));
        }
        _ => return Err(bad()),
    }
    for (g, &glen) in groups.iter().enumerate() {
        for sfb in 0..max_sfb {
            let cb = s.cb[g * max_sfb + sfb] & 15;
            if !has_spec(cb) { continue; }
            let lav = SHAPE[cb as usize - 1].2 as u16;
            for _ in 0..band_lines(off, sfb, glen) {
                let raw = c.u8()?;
                let mut a = (raw & 0x7F) as u16;
                if a > 16 || (a == 0 && raw != 0) || (cb != ESC_HCB && a > lav) { return Err(bad()); }
                if cb == ESC_HCB && a == 16 {
                    a = c.u16()?;
                    if !(16..=8191).contains(&a) { return Err(bad()); }
                }
                s.spec.push(if raw & 0x80 != 0 { -(a as i16) } else { a as i16 });
            }
        }
    }
    Ok(s)
}

// ------------------------------------------------------------------ ADTS
fn adts_ok(h: &[u8]) -> bool {
    h.len() >= 7 && h[0] == 0xFF && h[1] & 0xF6 == 0xF0 && ((h[2] >> 2) & 15) < 12 && h[6] & 3 == 0
}

/// The frame length an ADTS header states.
fn adts_len(h: &[u8]) -> usize { ((h[3] & 3) as usize) << 11 | (h[4] as usize) << 3 | (h[5] >> 5) as usize }

/// The bits that stay equal through a stream: version, profile, sampling frequency, channels.
fn adts_key(h: &[u8]) -> (u8, u8, u8) { (h[1] & 0xF8, h[2] & 0xFD, h[3] & 0xC0) }

/// The run of ADTS frames at the start of `b`: each (header length, frame length), and whether
/// `b` ended inside a frame.
fn adts_chain(b: &[u8]) -> (Vec<(usize, usize)>, bool) {
    let mut frames = vec![];
    let mut o = 0usize;
    loop {
        let h = &b[o..];
        if !adts_ok(h) || frames.first().is_some_and(|_| adts_key(h) != adts_key(b)) { return (frames, false); }
        let (hl, len) = (if h[1] & 1 == 0 { 9 } else { 7 }, adts_len(h));
        if len < hl { return (frames, false); }
        if o + len > b.len() { return (frames, true); }
        frames.push((hl, len));
        o += len;
    }
}

/// The extent of the ADTS stream starting at `b[i]`, by frame headers only.
pub fn extent(b: &[u8], i: usize) -> Result<(usize, usize), ExtentError> {
    let (frames, truncated) = adts_chain(&b[i..]);
    if truncated && (frames.len() >= MIN_FRAMES || b.len() - i < 4 * 8191) { return Err(ExtentError::Truncated); }
    if frames.len() < MIN_FRAMES { return Err(ExtentError::Invalid); }
    let len: usize = frames.iter().map(|f| f.1).sum();
    Ok((i + len, 5 * len))
}

// ------------------------------------------------------------------ split and rebuild
/// A run of frames taken apart: its length in the source, its frame records (with the offsets
/// where a leaf may end), the bytes kept raw, and how many frames it has / were parsed.
pub struct Split {
    pub len: usize,
    pub frames: Vec<u8>,
    pub cuts: Vec<usize>,
    pub rest: Vec<u8>,
    pub count: (usize, usize),
}

fn split_frames<'a>(payloads: impl Iterator<Item = (Option<Adts>, &'a [u8])>, sf: usize) -> Split {
    let mut s = Split { len: 0, frames: vec![], cuts: vec![], rest: vec![], count: (0, 0) };
    for (k, (adts, p)) in payloads.enumerate() {
        if k > 0 && k % LEAF_FRAMES == 0 { s.cuts.push(s.frames.len()); }
        let fsf = adts.map_or(sf, |a| a.sf());
        s.len += p.len() + adts.map_or(0, |a| a.len());
        s.count.0 += 1;
        // keep the parse only if it writes back to the same bytes
        let parsed = read_frame(p, fsf).filter(|f| write_frame(f, fsf).is_ok_and(|w| w[..] == p[..p.len() - f.tail as usize]));
        let body = match parsed {
            Some(f) => {
                s.count.1 += 1;
                s.rest.extend_from_slice(&p[p.len() - f.tail as usize..]);
                Body::Parsed(f)
            }
            None => {
                s.rest.extend_from_slice(p);
                Body::Raw(p.len() as u16)
            }
        };
        Rec { adts, body }.write(sf, &mut s.frames);
    }
    s
}

/// Take apart the ADTS stream at the start of `b`.
pub fn split_adts(b: &[u8]) -> Option<Split> {
    let (frames, _) = adts_chain(b);
    if frames.len() < MIN_FRAMES { return None; }
    let mut o = 0;
    let s = split_frames(frames.iter().map(|&(hl, len)| {
        let f = &b[o..o + len];
        o += len;
        let mut hdr: [u8; 7] = f[..7].try_into().unwrap();
        hdr[3] &= !3;
        hdr[4] = 0;
        hdr[5] &= 0x1F;
        (Some(Adts { hdr, crc: (hl == 9).then(|| [f[7], f[8]]) }), &f[hl..])
    }), 0);
    (s.count.1 >= 2 && 2 * s.count.1 >= s.count.0).then_some(s)
}

/// Take apart a run of headerless frames of the given sizes (an MP4 track's samples) at the
/// start of `b`, for sampling-frequency index `sf`.
pub fn split_run(b: &[u8], sizes: &[u32], sf: usize) -> Option<Split> {
    if sf >= SWB_LONG.len() || sizes.iter().any(|&n| n == 0 || n > u16::MAX as u32) || sizes.iter().map(|&n| n as usize).sum::<usize>() > b.len() { return None; }
    let mut o = 0;
    let s = split_frames(sizes.iter().map(|&n| {
        let f = &b[o..o + n as usize];
        o += n as usize;
        (None, f)
    }), sf);
    (s.count.1 >= 1 && 2 * s.count.1 >= s.count.0).then_some(s)
}

/// The original bytes of a run of frames from its frame records and raw bytes.
pub fn rebuild(frames: &[u8], rest: &[u8], adts: bool, sf: usize, out: &mut Vec<u8>) -> Result<(), String> {
    let bad = |what: &str| format!("aac: {what} (corrupt file)");
    let (mut o, mut r) = (0usize, 0usize);
    while o < frames.len() {
        let (rec, n) = Rec::read(&frames[o..], adts, sf)?;
        o += n;
        let fsf = rec.adts.map_or(sf, |a| a.sf());
        let (payload, tail) = match &rec.body {
            Body::Raw(len) => (vec![], *len as usize),
            Body::Parsed(f) => (write_frame(f, fsf)?, f.tail as usize),
        };
        let raw = rest.get(r..r + tail).ok_or_else(|| bad("raw bytes missing"))?;
        r += tail;
        if let Some(a) = &rec.adts {
            let len = a.len() + payload.len() + tail;
            if len > 8191 { return Err(bad("frame too long")); }
            let mut h = a.hdr;
            h[3] |= (len >> 11) as u8;
            h[4] = (len >> 3) as u8;
            h[5] |= ((len & 7) << 5) as u8;
            out.extend_from_slice(&h);
            if let Some(c) = a.crc { out.extend_from_slice(&c); }
        }
        out.extend_from_slice(&payload);
        out.extend_from_slice(raw);
    }
    if r != rest.len() { return Err(bad("raw bytes left over")); }
    Ok(())
}

/// Developer statistics: the bits the original stream spends per category in the frame records
/// `frames`: [headers and element framing, window layout and M/S, section data, scale factors,
/// spectral code words (with their signs and escapes), pulse and noise-shaping data, fill and
/// data payloads].
pub fn original_bits(frames: &[u8], adts: bool, sf: usize) -> Result<[u64; 7], String> {
    let mut t = [0u64; 7];
    let mut o = 0;
    let ics_bits = |t: &mut [u64; 7], s: &Ics, own: bool, fsf: usize| {
        t[1] += 8 + if own { if s.info.short() { 15 } else { 11 } } else { 0 };
        let (bits, esc) = if s.info.short() { (3u64, 7usize) } else { (5, 31) };
        let max_sfb = s.info.max_sfb as usize;
        let mut run = 0usize;
        for (i, &c) in s.cb.iter().enumerate() {
            let last = i + 1 == s.cb.len() || (i + 1) % max_sfb.max(1) == 0 || s.cb[i + 1] & SECTION_START != 0;
            run += 1;
            let _ = c;
            if last {
                t[2] += 4 + bits * (run / esc + 1) as u64;
                run = 0;
            }
        }
        let mut noise_pcm = true;
        let mut sfs = s.sf.iter();
        for &c in &s.cb {
            let cb = c & 15;
            if !has_sf(cb) { continue; }
            let v = sfs.next().copied().unwrap_or(0) as usize;
            if cb == NOISE_HCB && noise_pcm { noise_pcm = false; t[3] += 9; } else { t[3] += SCALEFACTOR[v.min(120)].1 as u64; }
        }
        t[5] += 3 + s.pulse.as_ref().map_or(0, |p| 8 + 9 * p.pulses.len() as u64) + s.tns.as_ref().map_or(0, |x| x.0 as u64);
        let off = s.info.offsets(fsf);
        let mut k = 0usize;
        for (g, &glen) in s.info.groups().iter().enumerate() {
            for sfb in 0..max_sfb {
                let cb = s.cb[g * max_sfb + sfb] & 15;
                if !has_spec(cb) { continue; }
                let (dim, signed, _) = SHAPE[cb as usize - 1];
                for _ in 0..band_lines(off, sfb, glen) / dim {
                    let v = &s.spec[k..k + dim];
                    k += dim;
                    let mut tuple = [0i16; 4];
                    tuple[..dim].copy_from_slice(v);
                    t[4] += tuple_index(cb, &tuple).map_or(0, |i| SPECTRAL[cb as usize - 1][i].1 as u64);
                    for &x in v {
                        let a = x.unsigned_abs() as u32;
                        if !signed && a != 0 { t[4] += 1; }
                        if cb == ESC_HCB && a >= 16 { t[4] += 2 * (31 - a.leading_zeros()) as u64 - 3; }
                    }
                }
            }
        }
    };
    while o < frames.len() {
        let (rec, n) = Rec::read(&frames[o..], adts, sf)?;
        o += n;
        let fsf = rec.adts.map_or(sf, |a| a.sf());
        t[0] += rec.adts.map_or(0, |a| 8 * a.len() as u64);
        let Body::Parsed(f) = &rec.body else { continue };
        t[0] += 3;
        for e in &f.elements {
            t[0] += 3;
            match e {
                Element::Single { ics, .. } => { t[0] += 4; ics_bits(&mut t, ics, true, fsf); }
                Element::Pair { common, ms_used, ics, .. } => {
                    t[0] += 5;
                    if *common { t[1] += 2 + ms_used.len() as u64 + if ics[0].info.short() { 15 } else { 11 }; }
                    ics_bits(&mut t, &ics[0], !*common, fsf);
                    ics_bits(&mut t, &ics[1], !*common, fsf);
                }
                Element::Data { data, .. } => t[6] += 13 + 8 * data.len() as u64,
                Element::Fill { data } => t[6] += 4 + 8 * data.len() as u64,
            }
        }
    }
    Ok(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ADTS: &[u8] = include_bytes!("../../tests/fixtures/tone_adts.aac");
    const M4A: &[u8] = include_bytes!("../../tests/fixtures/tone.m4a");

    fn rng(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }

    /// Every ADTS stream in `b` splits and rebuilds to exactly its bytes; returns (frames, parsed).
    fn check(b: &[u8]) -> (usize, usize) {
        let (mut i, mut n) = (0, (0, 0));
        while i < b.len() {
            match (b[i] == 0xFF).then(|| split_adts(&b[i..])).flatten() {
                Some(s) => {
                    let mut out = vec![];
                    rebuild(&s.frames, &s.rest, true, 0, &mut out).unwrap();
                    assert!(out == b[i..i + s.len], "rebuild differs at {i}");
                    n = (n.0 + s.count.0, n.1 + s.count.1);
                    i += s.len;
                }
                None => i += 1,
            }
        }
        n
    }

    #[test]
    fn adts_streams_and_track_runs_rebuild_exactly() {
        let (frames, parsed) = check(ADTS);
        assert!(frames >= 60 && parsed == frames, "{parsed}/{frames}");
        assert_eq!(extent(ADTS, 0).unwrap().0, ADTS.len());
        assert!(matches!(extent(&ADTS[..ADTS.len() - 9], 0), Err(ExtentError::Truncated)));
        // the same kind of frames without headers, as an MP4 track has them
        let idx = crate::recode::mp4::index(&mut std::io::Cursor::new(M4A), M4A.len() as u64).unwrap();
        assert_eq!(idx.runs.len(), 1);
        let run = &idx.runs[0];
        let crate::recode::mp4::Stream::Aac { sf } = run.what else { panic!("not an AAC track") };
        let s = split_run(&M4A[run.start as usize..], &run.sizes, sf as usize).unwrap();
        assert!(s.count.0 == run.sizes.len() && s.count.1 == s.count.0, "{:?}", s.count);
        let mut out = vec![];
        rebuild(&s.frames, &s.rest, false, sf as usize, &mut out).unwrap();
        assert!(out == M4A[run.start as usize..run.start as usize + s.len]);
        // with the wrong band tables most frames do not parse: the run is not taken
        assert!(split_run(&M4A[run.start as usize..], &run.sizes, 11).is_none_or(|s| s.count.1 < s.count.0));
    }

    /// Damaged streams never panic the parser, and what it accepts still rebuilds exactly.
    #[test]
    fn damaged_streams_stay_exact() {
        let mut s = 91u64;
        for _ in 0..300 {
            let mut b = ADTS.to_vec();
            for _ in 0..1 + rng(&mut s) % 6 {
                let at = (rng(&mut s) % b.len() as u64) as usize;
                match rng(&mut s) % 4 {
                    0 => b[at] ^= 1 << (rng(&mut s) % 8),
                    1 => b[at] = rng(&mut s) as u8,
                    2 => { b.remove(at); }
                    _ => b.truncate(at.max(8)),
                }
            }
            check(&b);
        }
        // sample sizes that are not the frames' (a wrong index): no panic, exact or refused
        let idx = crate::recode::mp4::index(&mut std::io::Cursor::new(M4A), M4A.len() as u64).unwrap();
        let run = &idx.runs[0];
        for _ in 0..200 {
            let mut sizes = run.sizes.clone();
            for _ in 0..1 + rng(&mut s) % 4 {
                let k = (rng(&mut s) % sizes.len() as u64) as usize;
                sizes[k] = (rng(&mut s) % 700) as u32;
            }
            let start = (run.start as usize + (rng(&mut s) % 3) as usize).min(M4A.len());
            if let Some(sp) = split_run(&M4A[start..], &sizes, 4) {
                let mut out = vec![];
                rebuild(&sp.frames, &sp.rest, false, 4, &mut out).unwrap();
                assert!(out == M4A[start..start + sp.len]);
            }
        }
    }

    /// Frame records that are not ours are refused, never followed blindly.
    #[test]
    fn bad_records_are_refused() {
        let sp = split_adts(ADTS).unwrap();
        let mut s = 6u64;
        for _ in 0..2000 {
            let mut f = sp.frames.clone();
            for _ in 0..1 + rng(&mut s) % 4 {
                let at = (rng(&mut s) % f.len() as u64) as usize;
                f[at] = rng(&mut s) as u8;
            }
            let mut out = vec![];
            let _ = rebuild(&f, &sp.rest, true, 0, &mut out);
            let mut out = vec![];
            let _ = rebuild(&f[..(rng(&mut s) % f.len() as u64) as usize], &sp.rest, rng(&mut s) % 2 == 0, (rng(&mut s) % 13) as usize, &mut out);
        }
    }
}
