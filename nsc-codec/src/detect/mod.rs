//! Content detection: split a byte stream into typed segments.
//!
//! [`Detector::next`] looks at the bytes from the current position (with at least [`LOOKAHEAD`]
//! bytes of lookahead unless the stream ends) and returns the next segment. A segment is either:
//! * a typed region found by parsing a recognised header (image raster, PCM samples, executable
//!   code, tar member, PNG/JPEG stream…), or
//! * a run of 4 KB windows of one class: text, binary, or high-entropy (already compressed).
//!
//! Decisions depend only on the bytes at and after the position, never on how much happens to be
//! buffered beyond the lookahead. So a file and the same bytes arriving through a pipe segment
//! identically. Parsers validate strictly and fall back to plain classification on anything odd.
use crate::nsc3::format::kind;
use std::collections::VecDeque;

pub const LOOKAHEAD: usize = 1 << 20;
const WIN: usize = 4096;
const MIN_RUN: usize = 4; // windows; shorter class changes are absorbed into the surrounding run
/// 4 KB windows above this order-0 entropy (milli-bits/byte) count as already-compressed data.
const HIGH_ENTROPY_MBITS: u32 = 7880;

/// Image raster formats (first byte of IMAGE params).
pub mod imgfmt {
    pub const GRAY8: u8 = 1;
    pub const RGB24: u8 = 2; // byte order in params[10]: 0 = RGB, 1 = BGR
    pub const RGBA32: u8 = 3;
    pub const GRAY16LE: u8 = 4;
    pub const GRAY16BE: u8 = 5;
    pub const RGB48BE: u8 = 6;
}

#[derive(Clone, Debug, PartialEq)]
pub struct Seg {
    pub kind: u8,
    pub len: u64,
    pub params: Vec<u8>,
    /// Split granularity when a block boundary falls inside (row stride, sample frame…).
    pub align: u32,
    /// Human-readable format label (for `nsc analyze`).
    pub label: &'static str,
}

impl Seg {
    fn new(kind: u8, len: u64, label: &'static str) -> Seg { Seg { kind, len, params: vec![], align: 1, label } }
}

/// Image params: fmt u8, width u32, stride u32, channels u8, order u8.
pub fn image_params(fmt: u8, width: u32, stride: u32, channels: u8, bgr: bool) -> Vec<u8> {
    let mut p = vec![fmt];
    p.extend_from_slice(&width.to_le_bytes());
    p.extend_from_slice(&stride.to_le_bytes());
    p.push(channels);
    p.push(bgr as u8);
    p
}

/// Audio params: channels u8, bits u8, flags u8 (bit0 big-endian, bit1 unsigned), frame bytes u8.
pub fn audio_params(channels: u8, bits: u8, big_endian: bool, unsigned: bool) -> Vec<u8> {
    let frame = channels as u32 * (bits as u32).div_ceil(8);
    vec![channels, bits, (big_endian as u8) | ((unsigned as u8) << 1), frame.min(255) as u8]
}

/// Exe params: arch u8 (1 x86, 2 x86-64, 3 arm64, 4 arm32), with [`EXE_RELOCATABLE`] set for
/// code whose branch operands are not filled in yet (object files): no branch transform there.
pub fn exe_params(arch: u8) -> Vec<u8> { vec![arch] }
pub const EXE_RELOCATABLE: u8 = 0x80;

// ------------------------------------------------------------------ little helpers
fn u16le(b: &[u8], o: usize) -> Option<u16> { b.get(o..o + 2).map(|s| u16::from_le_bytes([s[0], s[1]])) }
fn u32le(b: &[u8], o: usize) -> Option<u32> { b.get(o..o + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap())) }
fn u64le(b: &[u8], o: usize) -> Option<u64> { b.get(o..o + 8).map(|s| u64::from_le_bytes(s.try_into().unwrap())) }
fn u16be(b: &[u8], o: usize) -> Option<u16> { b.get(o..o + 2).map(|s| u16::from_be_bytes([s[0], s[1]])) }
fn u32be(b: &[u8], o: usize) -> Option<u32> { b.get(o..o + 4).map(|s| u32::from_be_bytes(s.try_into().unwrap())) }

// ------------------------------------------------------------------ format parsers
// Each returns the segments for a file starting at buf[0], in order, or None.

fn parse_bmp(b: &[u8]) -> Option<Vec<Seg>> {
    if b.len() < 54 || &b[0..2] != b"BM" { return None; }
    let fsize = u32le(b, 2)? as u64;
    let off = u32le(b, 10)? as u64;
    let dib = u32le(b, 14)?;
    if u32le(b, 6)? != 0 || !matches!(dib, 40 | 52 | 56 | 108 | 124) { return None; }
    let w = u32le(b, 18)? as i32;
    let h = u32le(b, 22)? as i32;
    let planes = u16le(b, 26)?;
    let bpp = u16le(b, 28)?;
    let comp = u32le(b, 30)?;
    if planes != 1 || !(1..=65536).contains(&w) || h == 0 || h.unsigned_abs() > 65536 { return None; }
    if !(comp == 0 || (comp == 3 && bpp == 32)) { return None; }
    if off < 14 + dib as u64 || off > 1 << 16 { return None; }
    let (fmt, ch) = match bpp { 8 => (imgfmt::GRAY8, 1u8), 24 => (imgfmt::RGB24, 3), 32 => (imgfmt::RGBA32, 4), _ => return None };
    if bpp == 8 {
        // only grayscale palettes make sense to model as intensities
        let pal = &b.get(14 + dib as usize..off as usize)?;
        if pal.len() < 1024 { return None; }
        for i in 0..256 {
            let e = &pal[i * 4..i * 4 + 3];
            if !(e[0] == e[1] && e[1] == e[2]) { return None; }
        }
    }
    let stride = ((w as u64 * bpp as u64 + 31) / 32) * 4;
    let data = stride * h.unsigned_abs() as u64;
    if fsize != 0 && fsize < off + data { return None; }
    Some(vec![
        Seg::new(kind::GENERIC, off, "bmp header"),
        Seg { kind: kind::IMAGE, len: data, params: image_params(fmt, w as u32, stride as u32, ch, true), align: stride as u32, label: "bmp raster" },
    ])
}

/// PNM/PAM header: returns (header length, width, height, maxval, depth).
fn parse_pnm_header(b: &[u8]) -> Option<(usize, u32, u32, u32, u32)> {
    if b.len() < 8 || b[0] != b'P' || !matches!(b[1], b'5' | b'6' | b'7') { return None; }
    if b[1] == b'7' {
        let end = b.windows(7).take(512).position(|w| w == b"ENDHDR\n")? + 7;
        let txt = std::str::from_utf8(&b[3..end]).ok()?;
        let (mut w, mut h, mut d, mut m) = (0u32, 0u32, 0u32, 0u32);
        for line in txt.lines() {
            let mut it = line.split_whitespace();
            match (it.next(), it.next()) {
                (Some("WIDTH"), Some(v)) => w = v.parse().ok()?,
                (Some("HEIGHT"), Some(v)) => h = v.parse().ok()?,
                (Some("DEPTH"), Some(v)) => d = v.parse().ok()?,
                (Some("MAXVAL"), Some(v)) => m = v.parse().ok()?,
                _ => {}
            }
        }
        return (w > 0 && h > 0 && (1..=4).contains(&d) && m > 0).then_some((end, w, h, m, d));
    }
    let depth = if b[1] == b'5' { 1 } else { 3 };
    let mut i = 2;
    let mut vals = [0u32; 3];
    for v in vals.iter_mut() {
        loop {
            while i < b.len() && b[i].is_ascii_whitespace() { i += 1; }
            if b.get(i) == Some(&b'#') { while i < b.len() && b[i] != b'\n' { i += 1; } } else { break; }
            if i > 1024 { return None; }
        }
        let s = i;
        while i < b.len() && b[i].is_ascii_digit() { i += 1; }
        if i == s || i - s > 6 { return None; }
        *v = std::str::from_utf8(&b[s..i]).ok()?.parse().ok()?;
    }
    if !b.get(i)?.is_ascii_whitespace() { return None; }
    Some((i + 1, vals[0], vals[1], vals[2], depth))
}

fn parse_pnm(b: &[u8]) -> Option<Vec<Seg>> {
    let (hl, w, h, maxval, depth) = parse_pnm_header(b)?;
    if !(1..=65536).contains(&w) || !(1..=65536).contains(&h) || !(1..=65535).contains(&maxval) { return None; }
    let bytes = if maxval > 255 { 2 } else { 1 };
    let fmt = match (depth, bytes) {
        (1, 1) => imgfmt::GRAY8, (3, 1) => imgfmt::RGB24, (4, 1) => imgfmt::RGBA32,
        (1, 2) => imgfmt::GRAY16BE, (3, 2) => imgfmt::RGB48BE, _ => return None,
    };
    let stride = w as u64 * depth as u64 * bytes;
    Some(vec![
        Seg::new(kind::GENERIC, hl as u64, "pnm header"),
        Seg { kind: kind::IMAGE, len: stride * h as u64, params: image_params(fmt, w, stride as u32, depth as u8, false), align: stride as u32, label: "pnm raster" },
    ])
}

/// Uncompressed TGA (types 2 and 3). TGA has no magic, so this is only tried at file starts.
fn parse_tga(b: &[u8], remaining: u64) -> Option<Vec<Seg>> {
    if b.len() < 18 { return None; }
    let (idlen, cmap, ty) = (b[0] as u64, b[1], b[2]);
    if cmap != 0 || !matches!(ty, 2 | 3) || b[3..8].iter().any(|&x| x != 0) { return None; }
    let w = u16le(b, 12)? as u32;
    let h = u16le(b, 14)? as u32;
    let depth = b[16];
    let desc = b[17];
    if w == 0 || h == 0 || desc & 0xC0 != 0 { return None; }
    let (fmt, ch) = match (ty, depth) { (3, 8) => (imgfmt::GRAY8, 1u8), (2, 24) => (imgfmt::RGB24, 3), (2, 32) => (imgfmt::RGBA32, 4), _ => return None };
    let stride = w as u64 * (depth as u64 / 8);
    let data = stride * h as u64;
    if 18 + idlen + data > remaining { return None; }
    Some(vec![
        Seg::new(kind::GENERIC, 18 + idlen, "tga header"),
        Seg { kind: kind::IMAGE, len: data, params: image_params(fmt, w, stride as u32, ch, true), align: stride as u32, label: "tga raster" },
    ])
}

/// Uncompressed, single-plane TIFF whose strips are contiguous and follow the IFD.
fn parse_tiff(b: &[u8]) -> Option<Vec<Seg>> {
    if b.len() < 16 { return None; }
    let le = match &b[0..4] { b"II*\0" => true, b"MM\0*" => false, _ => return None };
    let r16 = |o: usize| if le { u16le(b, o) } else { u16be(b, o) };
    let r32 = |o: usize| if le { u32le(b, o) } else { u32be(b, o) };
    let ifd = r32(4)? as usize;
    let n = r16(ifd)? as usize;
    if n == 0 || n > 64 { return None; }
    let (mut w, mut h, mut bps, mut comp, mut spp, mut planar) = (0u32, 0u32, 8u32, 1u32, 1u32, 1u32);
    let (mut strip_off, mut strip_cnt, mut nstrips, mut count_type) = (0u32, 0u32, 0u32, 3u16);
    for k in 0..n {
        let e = ifd + 2 + k * 12;
        let tag = r16(e)?;
        let ty = r16(e + 2)?;
        let cnt = r32(e + 4)?;
        let val = if ty == 3 && cnt == 1 { r16(e + 8)? as u32 } else { r32(e + 8)? };
        match tag {
            256 => w = val,
            257 => h = val,
            258 => bps = if cnt == 1 { val } else { r16(val as usize)? as u32 },
            259 => comp = val,
            273 => { strip_off = val; nstrips = cnt; }
            277 => spp = val,
            279 => { strip_cnt = val; count_type = ty; }
            284 => planar = val,
            _ => {}
        }
    }
    if comp != 1 || planar != 1 || w == 0 || h == 0 || w > 65536 || h > 65536 { return None; }
    let (fmt, ch) = match (spp, bps) {
        (1, 8) => (imgfmt::GRAY8, 1u8), (3, 8) => (imgfmt::RGB24, 3), (4, 8) => (imgfmt::RGBA32, 4),
        (1, 16) => (if le { imgfmt::GRAY16LE } else { imgfmt::GRAY16BE }, 1), _ => return None,
    };
    let stride = w as u64 * spp as u64 * (bps as u64 / 8);
    let data = stride * h as u64;
    // first strip offset: inline when one strip, else read from the offsets array
    let first = if nstrips == 1 { strip_off as u64 } else { r32(strip_off as usize)? as u64 };
    let _ = (strip_cnt, count_type);
    if first < 8 || first > 1 << 20 { return None; }
    // strips must be contiguous: check the offsets array if present
    if nstrips > 1 {
        let mut expect = first;
        let rows_per = h.div_ceil(nstrips) as u64;
        for s in 0..nstrips as usize {
            let o = r32(strip_off as usize + s * 4)? as u64;
            if o != expect { return None; }
            expect += (rows_per * stride).min(data - (o - first));
        }
    }
    Some(vec![
        Seg::new(kind::GENERIC, first, "tiff header"),
        Seg { kind: kind::IMAGE, len: data, params: image_params(fmt, w, stride as u32, ch, false), align: stride as u32, label: "tiff raster" },
    ])
}

fn parse_wav(b: &[u8]) -> Option<Vec<Seg>> {
    if b.len() < 44 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" { return None; }
    let mut o = 12;
    let (mut ch, mut bits, mut fmt_ok) = (0u16, 0u16, false);
    while o + 8 <= b.len().min(1 << 16) {
        let id = &b[o..o + 4];
        let len = u32le(b, o + 4)? as u64;
        if id == b"fmt " {
            let tag = u16le(b, o + 8)?;
            ch = u16le(b, o + 10)?;
            bits = u16le(b, o + 22)?;
            let pcm = tag == 1 || (tag == 0xFFFE && len >= 40 && u16le(b, o + 32)? == 1);
            fmt_ok = pcm && (1..=8).contains(&ch) && matches!(bits, 8 | 16 | 24);
        } else if id == b"data" {
            if !fmt_ok { return None; }
            let hdr = o as u64 + 8;
            return Some(vec![
                Seg::new(kind::GENERIC, hdr, "wav header"),
                Seg { kind: kind::AUDIO, len, params: audio_params(ch as u8, bits as u8, false, bits == 8),
                      align: ch as u32 * bits as u32 / 8, label: "wav pcm" },
            ]);
        }
        o += 8 + len as usize + (len as usize & 1);
    }
    None
}

fn parse_aiff(b: &[u8]) -> Option<Vec<Seg>> {
    if b.len() < 54 || &b[0..4] != b"FORM" || &b[8..12] != b"AIFF" { return None; }
    let mut o = 12;
    let (mut ch, mut bits) = (0u16, 0u16);
    while o + 8 <= b.len().min(1 << 16) {
        let id = &b[o..o + 4];
        let len = u32be(b, o + 4)? as u64;
        if id == b"COMM" { ch = u16be(b, o + 8)?; bits = u16be(b, o + 14)?; }
        if id == b"SSND" {
            if !(1..=8).contains(&ch) || !matches!(bits, 8 | 16 | 24) { return None; }
            let off = u32be(b, o + 8)? as u64;
            let hdr = o as u64 + 16 + off;
            return Some(vec![
                Seg::new(kind::GENERIC, hdr, "aiff header"),
                Seg { kind: kind::AUDIO, len: len.saturating_sub(8 + off), params: audio_params(ch as u8, bits as u8, true, false),
                      align: ch as u32 * bits as u32 / 8, label: "aiff pcm" },
            ]);
        }
        o += 8 + len as usize + (len as usize & 1);
    }
    None
}

fn parse_au(b: &[u8]) -> Option<Vec<Seg>> {
    if b.len() < 24 || &b[0..4] != b".snd" { return None; }
    let off = u32be(b, 4)? as u64;
    let size = u32be(b, 8)? as u64;
    let enc = u32be(b, 12)?;
    let ch = u32be(b, 20)?;
    let bits = match enc { 2 => 8, 3 => 16, 4 => 24, _ => return None };
    if !(24..=4096).contains(&off) || !(1..=8).contains(&ch) || size == 0xFFFF_FFFF { return None; }
    Some(vec![
        Seg::new(kind::GENERIC, off, "au header"),
        Seg { kind: kind::AUDIO, len: size, params: audio_params(ch as u8, bits, true, false), align: ch * bits as u32 / 8, label: "au pcm" },
    ])
}

/// Split a file into [header/data | code | rest] from a list of executable file ranges.
fn exe_layout(ranges: &mut Vec<(u64, u64)>, arch: u8, label: &'static str) -> Option<Vec<Seg>> {
    ranges.retain(|r| r.1 > 0);
    ranges.sort();
    let mut out = vec![];
    let mut pos = 0u64;
    for &(off, len) in ranges.iter() {
        if off < pos || len < 256 { continue; }
        if off > pos { out.push(Seg::new(kind::GENERIC, off - pos, "exe headers/data")); }
        out.push(Seg { kind: kind::EXE, len, params: exe_params(arch), align: 1, label });
        pos = off + len;
    }
    if out.iter().any(|s| s.kind == kind::EXE) { Some(out) } else { None }
}

fn parse_elf(b: &[u8]) -> Option<Vec<Seg>> {
    if b.len() < 64 || &b[0..4] != b"\x7fELF" || b[5] != 1 { return None; }
    let is64 = match b[4] { 1 => false, 2 => true, _ => return None };
    let mach = u16le(b, 18)?;
    let arch = match mach { 3 => 1, 62 => 2, 183 => 3, 40 => 4, _ => 0 };
    let (phoff, phentsize, phnum) = if is64 {
        (u64le(b, 32)? as usize, u16le(b, 54)? as usize, u16le(b, 56)? as usize)
    } else {
        (u32le(b, 28)? as usize, u16le(b, 42)? as usize, u16le(b, 44)? as usize)
    };
    let mut ranges = vec![];
    if phnum == 0 {
        // a relocatable object (as in a static library): its code is in the sections flagged
        // executable, found through the section table when that is within reach
        if u16le(b, 16)? != 1 { return None; }
        let (shoff, shentsize, shnum) = if is64 {
            (u64le(b, 40)? as usize, u16le(b, 58)? as usize, u16le(b, 60)? as usize)
        } else {
            (u32le(b, 32)? as usize, u16le(b, 46)? as usize, u16le(b, 48)? as usize)
        };
        if shnum == 0 || shnum > 4096 || shentsize < 40 { return None; }
        for k in 0..shnum {
            let e = shoff.checked_add(k * shentsize)?;
            let (ty, flags, off, size) = if is64 {
                (u32le(b, e + 4)?, u64le(b, e + 8)?, u64le(b, e + 24)?, u64le(b, e + 32)?)
            } else {
                (u32le(b, e + 4)?, u32le(b, e + 8)? as u64, u32le(b, e + 16)? as u64, u32le(b, e + 20)? as u64)
            };
            if ty == 1 && flags & 4 != 0 { ranges.push((off, size)); } // SHT_PROGBITS with SHF_EXECINSTR
        }
        return exe_layout(&mut ranges, arch | EXE_RELOCATABLE, "elf object code");
    }
    if phnum > 64 || phentsize < 32 { return None; }
    for k in 0..phnum {
        let e = phoff + k * phentsize;
        let ty = u32le(b, e)?;
        let (flags, off, filesz) = if is64 {
            (u32le(b, e + 4)?, u64le(b, e + 8)?, u64le(b, e + 32)?)
        } else {
            (u32le(b, e + 24)?, u32le(b, e + 4)? as u64, u32le(b, e + 16)? as u64)
        };
        if ty == 1 && flags & 1 != 0 { ranges.push((off, filesz)); } // PT_LOAD with PF_X
    }
    exe_layout(&mut ranges, arch, "elf code")
}

fn parse_pe(b: &[u8]) -> Option<Vec<Seg>> {
    if b.len() < 0x40 || &b[0..2] != b"MZ" { return None; }
    let pe = u32le(b, 0x3C)? as usize;
    if pe > 4096 || b.get(pe..pe + 4)? != b"PE\0\0" { return None; }
    let mach = u16le(b, pe + 4)?;
    let arch = match mach { 0x14c => 1, 0x8664 => 2, 0xAA64 => 3, 0x1c0 | 0x1c4 => 4, _ => 0 };
    let nsec = u16le(b, pe + 6)? as usize;
    let opt = u16le(b, pe + 20)? as usize;
    if nsec == 0 || nsec > 96 { return None; }
    let st = pe + 24 + opt;
    let mut ranges = vec![];
    for k in 0..nsec {
        let s = st + k * 40;
        let raw = u32le(b, s + 16)? as u64;
        let ptr = u32le(b, s + 20)? as u64;
        let ch = u32le(b, s + 36)?;
        if ch & 0x20 != 0 || ch & 0x2000_0000 != 0 { ranges.push((ptr, raw)); }
    }
    exe_layout(&mut ranges, arch, "pe code")
}

fn parse_macho(b: &[u8]) -> Option<Vec<Seg>> {
    if b.len() < 32 || u32le(b, 0)? != 0xFEED_FACF { return None; }
    let cpu = u32le(b, 4)?;
    let arch = match cpu { 0x0100_0007 => 2, 0x0100_000C => 3, _ => 0 };
    let ncmds = u32le(b, 16)? as usize;
    let mut o = 32;
    let mut ranges = vec![];
    for _ in 0..ncmds.min(256) {
        let cmd = u32le(b, o)?;
        let size = u32le(b, o + 4)? as usize;
        if size < 8 { return None; }
        if cmd == 0x19 {
            let nsects = u32le(b, o + 64)? as usize;
            for s in 0..nsects.min(64) {
                let sec = o + 72 + s * 80;
                let flags = u32le(b, sec + 64)?;
                if flags & 0x8000_0400 != 0 { ranges.push((u32le(b, sec + 48)? as u64, u64le(b, sec + 40)?)); }
            }
        }
        o += size;
    }
    exe_layout(&mut ranges, arch, "mach-o code")
}

/// PNG: walk chunks to IEND for the exact length.
fn parse_png(b: &[u8]) -> Option<Vec<Seg>> {
    if b.len() < 16 || &b[0..8] != b"\x89PNG\r\n\x1a\n" { return None; }
    let mut o = 8usize;
    loop {
        let len = u32be(b, o)? as usize;
        let ty = b.get(o + 4..o + 8)?;
        o = o.checked_add(12 + len)?;
        if ty == b"IEND" { return Some(vec![Seg::new(kind::FAST, o as u64, "png")]); }
        if o > b.len() { return None; }
    }
}

/// JPEG: parse marker segments and scan entropy-coded data to EOI.
fn parse_jpeg(b: &[u8]) -> Option<Vec<Seg>> {
    if b.len() < 4 || b[0] != 0xFF || b[1] != 0xD8 || b[2] != 0xFF { return None; }
    let mut o = 2usize;
    while o + 4 <= b.len() {
        if b[o] != 0xFF { return None; }
        let m = b[o + 1];
        if m == 0xD9 { return Some(vec![Seg::new(kind::FAST, o as u64 + 2, "jpeg")]); }
        if m == 0xFF { o += 1; continue; }
        if (0xD0..=0xD7).contains(&m) || m == 0x01 { o += 2; continue; }
        let len = u16be(b, o + 2)? as usize;
        o += 2 + len;
        if m == 0xDA {
            // entropy-coded data: until a marker that is not a stuffed 0x00 or a restart
            while o + 1 < b.len() {
                if b[o] == 0xFF && b[o + 1] != 0 && !(0xD0..=0xD7).contains(&b[o + 1]) { break; }
                o += 1;
            }
        }
    }
    None
}

/// Tar header at buf[0]: returns member content size if the header is valid.
fn tar_member_size(b: &[u8]) -> Option<u64> {
    if b.len() < 512 || &b[257..262] != b"ustar" { return None; }
    let field = &b[148..156];
    let stored = std::str::from_utf8(field).ok()?.trim_matches(|c: char| c == '\0' || c == ' ');
    let stored = u32::from_str_radix(stored, 8).ok()?;
    let sum: u32 = b[..512].iter().enumerate().map(|(i, &x)| if (148..156).contains(&i) { b' ' as u32 } else { x as u32 }).sum();
    if sum != stored { return None; }
    let sz = &b[124..136];
    if sz[0] & 0x80 != 0 {
        let mut v = 0u64;
        for &x in &sz[1..] { v = (v << 8) | x as u64; }
        return Some(v);
    }
    let s = std::str::from_utf8(sz).ok()?.trim_matches(|c: char| c == '\0' || c == ' ');
    if s.is_empty() { return Some(0); }
    u64::from_str_radix(s, 8).ok()
}

/// Label high-entropy data by signature (for `nsc analyze` and future recompression).
fn compressed_label(b: &[u8]) -> Option<&'static str> {
    let sig = |s: &[u8]| b.len() >= s.len() && &b[..s.len()] == s;
    if sig(b"PK\x03\x04") { return Some("zip"); }
    if sig(&[0x1f, 0x8b, 0x08]) { return Some("gzip"); }
    if sig(b"BZh") && b.len() > 3 && (b'1'..=b'9').contains(&b[3]) { return Some("bzip2"); }
    if sig(&[0xFD, b'7', b'z', b'X', b'Z', 0]) { return Some("xz"); }
    if sig(&[0x28, 0xB5, 0x2F, 0xFD]) { return Some("zstd"); }
    if sig(&[b'7', b'z', 0xBC, 0xAF, 0x27, 0x1C]) { return Some("7z"); }
    if sig(b"GIF87a") || sig(b"GIF89a") { return Some("gif"); }
    if sig(b"OggS") { return Some("ogg"); }
    if sig(b"fLaC") { return Some("flac"); }
    if sig(b"ID3") { return Some("mp3"); }
    if sig(&[0x1A, 0x45, 0xDF, 0xA3]) { return Some("matroska"); }
    if b.len() >= 12 && &b[4..8] == b"ftyp" { return Some("mp4"); }
    if sig(b"RIFF") && b.len() >= 12 && &b[8..12] == b"WEBP" { return Some("webp"); }
    if sig(b"%PDF") { return Some("pdf"); }
    None
}

/// First bytes that can start a recognised format (cheap pre-filter for the signature scan).
const fn starts() -> [bool; 256] {
    let mut t = [false; 256];
    let s = [b'B', b'P', b'R', b'F', b'.', 0x7F, b'M', b'I', 0x89, 0xFF, 0xCF];
    let mut i = 0;
    while i < s.len() { t[s[i] as usize] = true; i += 1; }
    t
}
static STARTS: [bool; 256] = starts();

/// Does a typed format start at buf[0]? (`file_start`: allow magic-less formats such as TGA.)
fn typed_at(b: &[u8], file_start: bool, remaining: u64) -> Option<Vec<Seg>> {
    let first = *b.first()?;
    if !STARTS[first as usize] && !file_start { return None; }
    match first {
        b'B' => parse_bmp(b),
        b'P' => parse_pnm(b),
        b'R' => parse_wav(b),
        b'F' => parse_aiff(b),
        b'.' => parse_au(b),
        0x7F => parse_elf(b),
        b'M' => parse_pe(b).or_else(|| parse_tiff(b)),
        b'I' => parse_tiff(b),
        0x89 => parse_png(b),
        0xFF => parse_jpeg(b),
        0xCF => parse_macho(b),
        _ => None,
    }
    .or_else(|| if file_start { parse_tga(b, remaining) } else { None })
}

// ------------------------------------------------------------------ window classification
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class { Text, Binary, High }

fn classify(w: &[u8]) -> Class {
    let mut cnt = [0u32; 256];
    for &x in w { cnt[x as usize] += 1; }
    let n = w.len() as u32;
    let ctrl: u32 = (0..32usize).filter(|&c| !matches!(c, 9 | 10 | 13)).map(|c| cnt[c]).sum::<u32>() + cnt[127];
    let high: u32 = cnt[128..].iter().sum();
    if ctrl * 200 <= n && high * 4 <= n { return Class::Text; }
    if n >= 1024 && crate::nsc3::block::order0_mbits(w) >= HIGH_ENTROPY_MBITS { return Class::High; }
    Class::Binary
}

fn class_seg(c: Class, len: u64, b: &[u8]) -> Seg {
    match c {
        Class::Text => Seg::new(kind::TEXT, len, "text"),
        Class::Binary => Seg::new(kind::GENERIC, len, "binary"),
        Class::High => Seg::new(kind::FAST, len, compressed_label(b).unwrap_or("high-entropy")),
    }
}

/// Model kind for a whole buffer (decompressed deflate content): text if most 4 KB windows are text,
/// otherwise generic.
pub fn kind_of(data: &[u8]) -> u8 {
    let (mut text, mut total) = (0usize, 0usize);
    for w in data.chunks(WIN).take(64) {
        if w.len() < 64 { continue; }
        total += 1;
        if classify(w) == Class::Text { text += 1; }
    }
    if total > 0 && text * 2 > total { kind::TEXT } else { kind::GENERIC }
}

// ------------------------------------------------------------------ the detector
pub struct Detector {
    pending: VecDeque<Seg>,
    /// A run cut only by the lookahead limit: its class and label carry over to the continuation.
    carry: Option<(Class, &'static str)>,
    /// Start and end of the current tar member's content, and where the next header is expected.
    member_start: Option<u64>,
    member_end: Option<u64>,
    tar_next: Option<u64>,
}

impl Detector {
    pub fn new() -> Self { Detector { pending: VecDeque::new(), carry: None, member_start: None, member_end: None, tar_next: None } }

    /// Next segment starting at absolute position `abs`; `b` holds the bytes from there (at least
    /// `LOOKAHEAD` unless `eof`). A segment's length may extend past `b` (typed regions).
    pub fn next(&mut self, b: &[u8], abs: u64, eof: bool) -> Seg {
        if let Some(s) = self.pending.pop_front() { return s; }
        // tar structure: header, member content, padding, next header
        if self.member_end == Some(abs) {
            self.member_end = None;
            self.member_start = None;
            if let Some(t) = self.tar_next.filter(|&t| t > abs) {
                return Seg::new(kind::GENERIC, t - abs, "tar padding");
            }
        }
        if (abs == 0 || self.tar_next == Some(abs)) && self.member_end.is_none() {
            if let Some(size) = tar_member_size(b) {
                self.member_start = Some(abs + 512);
                self.member_end = Some(abs + 512 + size);
                self.tar_next = Some(abs + 512 + size.div_ceil(512) * 512);
                return Seg::new(kind::GENERIC, 512, "tar header");
            } else if self.tar_next == Some(abs) {
                self.tar_next = None;
            }
        }
        let limit = self.member_end.map(|e| e - abs).unwrap_or(u64::MAX);
        let file_start = abs == 0 || self.member_start == Some(abs);
        let remaining = if eof { b.len() as u64 } else { u64::MAX };
        if let Some(segs) = typed_at(b, file_start, remaining.min(limit)) {
            self.carry = None;
            let mut left = limit;
            for mut s in segs {
                if left == 0 { break; }
                s.len = s.len.min(left);
                left -= s.len;
                if s.len > 0 { self.pending.push_back(s); }
            }
            if let Some(s) = self.pending.pop_front() { return s; }
        }
        self.run(b, limit, eof)
    }

    /// A run of same-class windows from b[0], stopping at a signature, the member limit, or the end
    /// of the lookahead.
    fn run(&mut self, b: &[u8], limit: u64, eof: bool) -> Seg {
        let avail = (b.len() as u64).min(limit) as usize;
        let scan_end = if eof || avail < b.len() { avail } else { avail.saturating_sub(LOOKAHEAD / 2).max(WIN.min(avail)) };
        if avail == 0 { return Seg::new(kind::GENERIC, limit.min(b.len() as u64).max(1), "binary"); }
        // nearest signature after position 0 (checked on every byte via a first-byte filter)
        let mut sig_at = scan_end;
        for i in 1..scan_end {
            if STARTS[b[i] as usize] && typed_at(&b[i..avail], false, (avail - i) as u64).is_some() { sig_at = i; break; }
        }
        let end = sig_at.max(1);
        let win = |k: usize| -> Option<Class> {
            let s = k * WIN;
            if s >= end { return None; }
            let e = (s + WIN).min(end);
            if e - s < 64 { None } else { Some(classify(&b[s..e])) }
        };
        let first = win(0).unwrap_or(Class::Binary);
        let mut k = 1;
        while let Some(c) = win(k) {
            if c != first {
                // absorb short excursions: a change must persist MIN_RUN windows to end the run
                let persists = (k..k + MIN_RUN).all(|j| win(j).map_or(true, |x| x == c)) && win(k + MIN_RUN - 1).is_some();
                if persists { break; }
            }
            k += 1;
        }
        let len = (k * WIN).min(end);
        let mut seg = class_seg(first, len as u64, b);
        if let Some((c, label)) = self.carry.take() {
            if c == first { seg.label = label; }
        }
        // cut only by the lookahead (no class change, signature, member end or end of stream)?
        if len == end && sig_at == scan_end && scan_end < avail {
            self.carry = Some((first, seg.label));
        }
        seg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all(data: &[u8]) -> Vec<Seg> {
        let mut d = Detector::new();
        let mut out = vec![];
        let mut pos = 0u64;
        while (pos as usize) < data.len() {
            let s = d.next(&data[pos as usize..], pos, true);
            let len = s.len.min(data.len() as u64 - pos).max(1);
            pos += len;
            out.push(Seg { len, ..s });
        }
        out
    }

    #[test]
    fn bmp_wav_text_random_are_typed() {
        // 24-bit BMP 4x2
        let mut bmp = b"BM".to_vec();
        let data_len = 2 * 12u32;
        bmp.extend_from_slice(&(54 + data_len).to_le_bytes());
        bmp.extend_from_slice(&[0, 0, 0, 0, 54, 0, 0, 0, 40, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 1, 0, 24, 0, 0, 0, 0, 0]);
        bmp.resize(54, 0);
        bmp.extend((0..data_len).map(|i| i as u8));
        let segs = all(&bmp);
        assert_eq!(segs[0].label, "bmp header");
        assert_eq!(segs[1].kind, kind::IMAGE);
        assert_eq!(segs[1].len, data_len as u64);

        let text = b"Hello world. This is plain text.\n".repeat(400);
        assert_eq!(all(&text)[0].kind, kind::TEXT);
        let mut s = 7u64;
        let rnd: Vec<u8> = (0..65536).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s as u8 }).collect();
        let segs = all(&rnd);
        assert!(segs.iter().all(|x| x.kind == kind::FAST), "{segs:?}");
    }

    #[test]
    fn elf_code_is_typed() {
        let code = vec![0x90u8; 600];
        // an executable: one loadable, executable program header
        let mut exe = vec![0u8; 0x200];
        exe[..8].copy_from_slice(b"\x7fELF\x02\x01\x01\0");
        exe[16] = 2;
        exe[18] = 62;
        exe[32] = 64; // e_phoff
        exe[54] = 56; // e_phentsize
        exe[56] = 1; // e_phnum
        exe[64] = 1; // PT_LOAD
        exe[68] = 5; // readable, executable
        exe[72..80].copy_from_slice(&0x200u64.to_le_bytes());
        exe[96..104].copy_from_slice(&600u64.to_le_bytes());
        exe.extend_from_slice(&code);
        exe.extend_from_slice(&[7u8; 300]);
        let segs = all(&exe);
        assert_eq!((segs[0].kind, segs[0].len), (kind::GENERIC, 0x200));
        assert_eq!((segs[1].kind, segs[1].len, &segs[1].params[..]), (kind::EXE, 600, &[2u8][..]));
        // an object file: no program headers; a section flagged executable. Its code is marked
        // relocatable
        let mut obj = vec![0u8; 192];
        obj[..8].copy_from_slice(b"\x7fELF\x02\x01\x01\0");
        obj[16] = 1;
        obj[18] = 62;
        obj[40] = 64; // e_shoff
        obj[58] = 64; // e_shentsize
        obj[60] = 2; // e_shnum
        obj[128 + 4] = 1; // SHT_PROGBITS
        obj[128 + 8] = 6; // allocated, executable
        obj[128 + 24..128 + 32].copy_from_slice(&192u64.to_le_bytes());
        obj[128 + 32..128 + 40].copy_from_slice(&600u64.to_le_bytes());
        obj.extend_from_slice(&code);
        let segs = all(&obj);
        assert_eq!((segs[1].kind, segs[1].len, &segs[1].params[..]), (kind::EXE, 600, &[2 | EXE_RELOCATABLE][..]), "{segs:?}");
        // without sections either, it is not an executable this parser knows
        obj[60] = 0;
        assert!(all(&obj).iter().all(|s| s.kind != kind::EXE));
    }

    #[test]
    fn random_bytes_rarely_parse_as_formats() {
        let mut s = 99u64;
        let rnd: Vec<u8> = (0..1 << 20).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 11) as u8 }).collect();
        let segs = all(&rnd);
        let typed = segs.iter().filter(|x| x.kind == kind::IMAGE || x.kind == kind::AUDIO || x.kind == kind::EXE).count();
        assert_eq!(typed, 0, "random data produced typed segments: {segs:?}");
    }
}
