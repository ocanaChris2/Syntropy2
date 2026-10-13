//! ISO base media files (MP4, M4A, MOV, 3GP): the index.
//!
//! The samples of a track have no headers of their own: where they are and how long each is, is
//! written in the `moov` box, which may sit after the media data. [`index`] reads it (the file
//! must be seekable) and lists, as offsets in the file,
//! * the runs of consecutive samples of tracks a recoder knows (AAC-LC audio), and
//! * the boxes that are not media data (`moov`, `ftyp`, …), which the planner always models
//!   instead of storing them with the media around them.
//!
//! The index is only a hint: a recoder still proves every run it takes apart, so a wrong or
//! hostile index can cost compression, never correctness.
use crate::bits::BitReader;
use std::io::{Read, Seek, SeekFrom};

/// What the samples of a run are.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Stream {
    /// AAC-LC raw data blocks; `sf`: sampling-frequency index.
    Aac { sf: u8 },
    /// H.264 access units as length-prefixed NAL units; `cfg`: index into [`Index::avc`].
    Avc { cfg: u16 },
    /// MPEG audio Layer III frames (a Matroska or AVI track's).
    Mp3,
    /// PCM samples, little-endian (an AVI track's): 8 bits unsigned, 16 or 24 signed.
    Pcm { channels: u8, bits: u8 },
    /// An elementary stream of a transport stream, by its PID: H.264 as a byte stream (NAL
    /// units behind start codes); AAC in ADTS frames.
    AnnexB { pid: u16 },
    Adts { pid: u16 },
}

/// An H.264 track's decoder configuration (`avcC`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Avc {
    /// Bytes of the length before each NAL unit (1, 2 or 4).
    pub nal_len: u8,
    /// Parameter set NAL units (with their header byte).
    pub sps: Vec<Vec<u8>>,
    pub pps: Vec<Vec<u8>>,
}

/// Consecutive samples of one track, back to back in the file from `start`.
#[derive(Clone, Debug, PartialEq)]
pub struct Run {
    pub start: u64,
    pub sizes: Vec<u32>,
    pub what: Stream,
}

impl Run {
    pub fn len(&self) -> u64 { self.sizes.iter().map(|&s| s as u64).sum() }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Index {
    /// Sample runs, by offset, not overlapping.
    pub runs: Vec<Run>,
    /// Byte ranges (start, end) of the boxes that are not media data.
    pub meta: Vec<(u64, u64)>,
    /// The configurations of the H.264 tracks.
    pub avc: Vec<Avc>,
}

/// Largest `moov` box read, and most samples indexed.
const MAX_MOOV: u64 = 256 << 20;
const MAX_SAMPLES: usize = 1 << 26;

fn be32(b: &[u8], o: usize) -> Option<u32> { Some(u32::from_be_bytes(b.get(o..o + 4)?.try_into().ok()?)) }
fn be64(b: &[u8], o: usize) -> Option<u64> { Some(u64::from_be_bytes(b.get(o..o + 8)?.try_into().ok()?)) }

/// The child boxes of `b`: (type, payload).
fn boxes(b: &[u8]) -> Vec<([u8; 4], &[u8])> {
    let mut out = vec![];
    let mut o = 0usize;
    while o + 8 <= b.len() {
        let size = be32(b, o).unwrap() as u64;
        let ty: [u8; 4] = b[o + 4..o + 8].try_into().unwrap();
        let (hdr, size) = match size {
            0 => (8, (b.len() - o) as u64),
            1 => match be64(b, o + 8) { Some(s) => (16, s), None => break },
            s => (8, s),
        };
        if size < hdr as u64 || size > (b.len() - o) as u64 { break; }
        out.push((ty, &b[o + hdr..o + size as usize]));
        o += size as usize;
    }
    out
}

fn child<'a>(b: &'a [u8], ty: &[u8; 4]) -> Option<&'a [u8]> { boxes(b).into_iter().find(|(t, _)| t == ty).map(|(_, p)| p) }

/// Does `b` (the first bytes of a file) start an ISO base media file?
pub fn looks_like(b: &[u8]) -> bool {
    b.len() >= 12 && matches!(&b[4..8], b"ftyp" | b"moov" | b"mdat" | b"free" | b"skip" | b"wide" | b"styp") && {
        let s = be32(b, 0).unwrap();
        s == 1 || s >= 8
    }
}

/// The sampling-frequency index of an AAC-LC `AudioSpecificConfig` with 1024-sample frames.
pub fn aac_config(asc: &[u8]) -> Option<u8> {
    let mut r = BitReader::new(asc, 0);
    let object_type = |r: &mut BitReader| -> u32 { let t = r.get(5); if t == 31 { 32 + r.get(6) } else { t } };
    let mut aot = object_type(&mut r);
    let sf = r.get(4);
    if sf >= 12 { return None; }
    r.skip(4); // channel configuration: the elements say what they are
    if aot == 5 || aot == 29 {
        // explicit SBR / PS signalling: the extension's rate, then the core's object type
        if r.get(4) == 15 { r.skip(24); }
        aot = object_type(&mut r);
    }
    if aot != 2 { return None; }
    // frameLengthFlag (960-sample frames) and dependsOnCoreCoder are not supported
    if r.get(1) != 0 || r.get(1) != 0 || r.pos() > asc.len() * 8 { return None; }
    Some(sf as u8)
}

/// A descriptor's (tag, payload) at the start of `b` and the bytes it takes.
fn descriptor(b: &[u8]) -> Option<(u8, &[u8], usize)> {
    let tag = *b.first()?;
    let (mut len, mut o) = (0usize, 1usize);
    for _ in 0..4 {
        let x = *b.get(o)?;
        o += 1;
        len = len << 7 | (x & 0x7F) as usize;
        if x & 0x80 == 0 { break; }
    }
    Some((tag, b.get(o..o + len)?, o + len))
}

/// The stream of an audio sample entry (`mp4a`), from the decoder configuration in its `esds`.
fn audio_entry(entry: &[u8]) -> Option<Stream> {
    // the esds box sits after the fixed fields, directly or inside a QuickTime `wave` box
    let at = entry.windows(4).position(|w| w == b"esds")?;
    let es = entry.get(at + 8..)?; // past the type and version/flags
    let (tag, es, _) = descriptor(es)?;
    if tag != 3 { return None; }
    let flags = *es.get(2)?;
    let mut o = 3usize;
    if flags & 0x80 != 0 { o += 2; }
    if flags & 0x40 != 0 { o += 1 + *es.get(o)? as usize; }
    if flags & 0x20 != 0 { o += 2; }
    let (tag, dc, _) = descriptor(es.get(o..)?)?;
    if tag != 4 { return None; }
    // MPEG-4 audio, or MPEG-2 AAC LC
    if !matches!(*dc.first()?, 0x40 | 0x67) { return None; }
    let (tag, asc, _) = descriptor(dc.get(13..)?)?;
    if tag != 5 { return None; }
    Some(Stream::Aac { sf: aac_config(asc)? })
}

/// The decoder configuration of a video sample entry (`avc1` / `avc3`), from its `avcC` box.
fn video_entry(entry: &[u8]) -> Option<Avc> {
    let at = entry.windows(4).position(|w| w == b"avcC")?;
    let size = be32(entry, at.checked_sub(4)?)? as usize;
    avc_config(entry.get(at + 4..(at - 4).checked_add(size)?)?)
}

/// An `AVCDecoderConfigurationRecord` (the payload of an `avcC` box; a Matroska track's codec
/// private data).
pub fn avc_config(c: &[u8]) -> Option<Avc> {
    if *c.first()? != 1 { return None; }
    let mut a = Avc { nal_len: (*c.get(4)? & 3) + 1, ..Default::default() };
    if a.nal_len == 3 { return None; }
    let mut o = 6;
    for k in 0..2 {
        let n = if k == 0 { *c.get(5)? & 31 } else { let n = *c.get(o)?; o += 1; n };
        for _ in 0..n {
            let len = u16::from_be_bytes(c.get(o..o + 2)?.try_into().ok()?) as usize;
            let nal = c.get(o + 2..o + 2 + len)?.to_vec();
            o += 2 + len;
            if k == 0 { a.sps.push(nal); } else { a.pps.push(nal); }
        }
    }
    Some(a)
}

/// A `trak` box's track id and what its samples are, if a recoder knows them. An H.264 track's
/// configuration is added to `avc`.
fn track_stream(trak: &[u8], avc: &mut Vec<Avc>) -> Option<(u32, Stream)> {
    let tkhd = child(trak, b"tkhd")?;
    let id = be32(tkhd, if *tkhd.first()? == 1 { 20 } else { 12 })?;
    let stsd = child(child(child(child(trak, b"mdia")?, b"minf")?, b"stbl")?, b"stsd")?;
    if be32(stsd, 4)? != 1 { return None; }
    let entry = stsd.get(8..)?;
    match entry.get(4..8)? {
        b"mp4a" => Some((id, audio_entry(entry)?)),
        b"avc1" | b"avc3" if avc.len() < 1 << 16 => {
            avc.push(video_entry(entry)?);
            Some((id, Stream::Avc { cfg: (avc.len() - 1) as u16 }))
        }
        _ => None,
    }
}

/// The sample runs a movie fragment (`moof` box at file offset `at`, payload `moof`) gives the
/// known tracks; `tracks`: id → (stream, default sample size).
fn fragment_runs(moof: &[u8], at: u64, tracks: &[(u32, Stream, u32)], file_len: u64, runs: &mut Vec<Run>) -> Option<()> {
    let mut next = at; // where a track fragment without an offset of its own continues
    for (ty, traf) in boxes(moof) {
        if &ty != b"traf" { continue; }
        let tfhd = child(traf, b"tfhd")?;
        let (flags, id) = (be32(tfhd, 0)? & 0xFF_FFFF, be32(tfhd, 4)?);
        let mut o = 8usize;
        let mut base = if flags & 0x1 != 0 { o += 8; be64(tfhd, 8)? } else if flags & 0x2_0000 != 0 { at } else { next };
        if flags & 0x2 != 0 { o += 4; }
        if flags & 0x8 != 0 { o += 4; }
        let known = tracks.iter().find(|t| t.0 == id);
        let default_size = if flags & 0x10 != 0 { be32(tfhd, o)? } else { known.map_or(0, |t| t.2) };
        for (ty, trun) in boxes(traf) {
            if &ty != b"trun" { continue; }
            let (tf, n) = (be32(trun, 0)? & 0xFF_FFFF, be32(trun, 4)? as usize);
            if n > MAX_SAMPLES { return None; }
            let mut o = 8usize;
            if tf & 0x1 != 0 {
                base = (if flags & 0x1 != 0 { be64(tfhd, 8)? } else { at }).checked_add_signed(be32(trun, o)? as i32 as i64)?;
                o += 4;
            }
            if tf & 0x4 != 0 { o += 4; }
            let per = 4 * ((tf >> 8 & 1) + (tf >> 9 & 1) + (tf >> 10 & 1) + (tf >> 11 & 1)) as usize;
            let size_at = 4 * (tf >> 8 & 1) as usize;
            let sizes: Vec<u32> = if tf & 0x200 != 0 { (0..n).map(|i| be32(trun, o + i * per + size_at)).collect::<Option<_>>()? } else { vec![default_size; n] };
            let len: u64 = sizes.iter().map(|&s| s as u64).sum();
            if let Some(&(_, what, _)) = known {
                if n > 0 && base.checked_add(len)? <= file_len {
                    match runs.last_mut() {
                        Some(r) if r.what == what && r.start + r.len() == base => r.sizes.extend_from_slice(&sizes),
                        _ => runs.push(Run { start: base, sizes, what }),
                    }
                }
            }
            base += len;
            next = base;
        }
    }
    Some(())
}

/// The sample runs of one `trak` box's own tables (none in a fragmented file), if a recoder knows
/// its samples.
fn track_runs(trak: &[u8], what: Stream, file_len: u64, runs: &mut Vec<Run>) -> Option<()> {
    let stbl = child(child(child(trak, b"mdia")?, b"minf")?, b"stbl")?;
    // sample sizes
    let sizes: Vec<u32> = if let Some(s) = child(stbl, b"stsz") {
        let (fixed, n) = (be32(s, 4)?, be32(s, 8)? as usize);
        if n > MAX_SAMPLES { return None; }
        if fixed != 0 { vec![fixed; n] } else { (0..n).map(|i| be32(s, 12 + 4 * i)).collect::<Option<_>>()? }
    } else {
        let s = child(stbl, b"stz2")?;
        let (field, n) = (*s.get(7)? as usize, be32(s, 8)? as usize);
        if n > MAX_SAMPLES || !matches!(field, 8 | 16) { return None; }
        (0..n).map(|i| if field == 8 { s.get(12 + i).map(|&x| x as u32) } else { s.get(12 + 2 * i..14 + 2 * i).map(|x| u16::from_be_bytes([x[0], x[1]]) as u32) }).collect::<Option<_>>()?
    };
    // chunk offsets and how many samples each chunk has
    let offsets: Vec<u64> = if let Some(s) = child(stbl, b"stco") {
        let n = be32(s, 4)? as usize;
        if n > MAX_SAMPLES { return None; }
        (0..n).map(|i| be32(s, 8 + 4 * i).map(|x| x as u64)).collect::<Option<_>>()?
    } else {
        let s = child(stbl, b"co64")?;
        let n = be32(s, 4)? as usize;
        if n > MAX_SAMPLES { return None; }
        (0..n).map(|i| be64(s, 8 + 8 * i)).collect::<Option<_>>()?
    };
    let stsc = child(stbl, b"stsc")?;
    let n = be32(stsc, 4)? as usize;
    if n > MAX_SAMPLES { return None; }
    let stsc: Vec<(u32, u32)> = (0..n).map(|i| Some((be32(stsc, 8 + 12 * i)?, be32(stsc, 12 + 12 * i)?))).collect::<Option<_>>()?;
    let (mut next, mut entry) = (0usize, 0usize);
    for (c, &start) in offsets.iter().enumerate() {
        while entry + 1 < stsc.len() && stsc[entry + 1].0 as usize <= c + 1 { entry += 1; }
        let per = stsc.get(entry)?.1 as usize;
        let chunk = sizes.get(next..next + per)?;
        next += per;
        let len: u64 = chunk.iter().map(|&s| s as u64).sum();
        if per == 0 || start.checked_add(len)? > file_len { continue; }
        // chunks that follow each other in the file make one run
        match runs.last_mut() {
            Some(r) if r.what == what && r.start + r.len() == start => r.sizes.extend_from_slice(chunk),
            _ => runs.push(Run { start, sizes: chunk.to_vec(), what }),
        }
    }
    Some(())
}

/// Index the ISO base media file `f` of `len` bytes (None if it is not one, or has no `moov`).
/// Leaves the read position unspecified.
pub fn index<F: Read + Seek>(f: &mut F, len: u64) -> Option<Index> {
    let mut idx = Index::default();
    let mut moov = None;
    let mut moofs: Vec<(u64, Vec<u8>)> = vec![];
    let mut moof_bytes = 0u64;
    let mut o = 0u64;
    let mut first = true;
    while o + 8 <= len {
        let mut h = [0u8; 16];
        f.seek(SeekFrom::Start(o)).ok()?;
        let got = crate::bench::read_full(f, &mut h).ok()?;
        if got < 8 { break; }
        if first && !looks_like(&h[..got]) { return None; }
        first = false;
        let ty: [u8; 4] = h[4..8].try_into().unwrap();
        let (hdr, size) = match be32(&h, 0)? as u64 {
            0 => (8, len - o),
            1 if got == 16 => (16, be64(&h, 8)?),
            1 => break,
            s => (8, s),
        };
        if size < hdr || o + size > len || !ty.iter().all(|c| c.is_ascii_graphic() || *c == b' ' || *c == 0xA9) { break; }
        if &ty == b"mdat" {
            idx.meta.push((o, o + hdr));
        } else {
            idx.meta.push((o, o + size));
            let wanted = (&ty == b"moov" && moov.is_none() && size <= MAX_MOOV) || (&ty == b"moof" && moof_bytes + size <= MAX_MOOV);
            if wanted {
                let mut b = vec![0u8; (size - hdr) as usize];
                f.seek(SeekFrom::Start(o + hdr)).ok()?;
                f.read_exact(&mut b).ok()?;
                if &ty == b"moov" { moov = Some(b); } else { moof_bytes += size; moofs.push((o, b)); }
            }
        }
        o += size;
    }
    let moov = moov?;
    // the tracks a recoder knows, with the default sample size of their fragments
    let mut tracks: Vec<(u32, Stream, u32)> = vec![];
    for (ty, trak) in boxes(&moov) {
        if &ty == b"trak" {
            let Some((id, what)) = track_stream(trak, &mut idx.avc) else { continue };
            let mut runs = vec![];
            if track_runs(trak, what, len, &mut runs).is_some() { idx.runs.append(&mut runs); }
            tracks.push((id, what, 0));
        }
    }
    if let Some(mvex) = child(&moov, b"mvex") {
        for (ty, trex) in boxes(mvex) {
            if &ty == b"trex" {
                if let (Some(id), Some(size)) = (be32(trex, 4), be32(trex, 16)) {
                    for t in tracks.iter_mut().filter(|t| t.0 == id) { t.2 = size; }
                }
            }
        }
    }
    for (at, moof) in &moofs {
        let mut runs = vec![];
        if fragment_runs(moof, *at, &tracks, len, &mut runs).is_some() { idx.runs.append(&mut runs); }
    }
    // by offset; a run that overlaps the one before it is dropped (the recoders would refuse it)
    idx.runs.sort_by_key(|r| r.start);
    let mut end = 0u64;
    idx.runs.retain(|r| {
        let keep = r.start >= end && !r.sizes.is_empty();
        if keep { end = r.start + r.len(); }
        keep
    });
    idx.meta.sort();
    Some(idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const M4A: &[u8] = include_bytes!("../../tests/fixtures/tone.m4a");
    const VIDEO: &[u8] = include_bytes!("../../tests/fixtures/tone_video.mp4");
    const FRAGMENTED: &[u8] = include_bytes!("../../tests/fixtures/tone_fragmented.mp4");

    fn rng(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }

    #[test]
    fn indexes_tracks_fragments_and_metadata() {
        for (name, b) in [("m4a", M4A), ("video", VIDEO), ("fragmented", FRAGMENTED)] {
            let idx = index(&mut Cursor::new(b), b.len() as u64).unwrap_or_else(|| panic!("{name}: no index"));
            let samples: usize = idx.runs.iter().filter(|r| matches!(r.what, Stream::Aac { sf: 4 })).map(|r| r.sizes.len()).sum();
            assert_eq!(samples, 66, "{name}: {samples} audio samples");
            let frames: usize = idx.runs.iter().filter(|r| matches!(r.what, Stream::Avc { .. })).map(|r| r.sizes.len()).sum();
            assert_eq!(frames, if name == "video" { 15 } else { 0 }, "{name}: {frames} video samples");
            assert_eq!(idx.avc.len(), (name == "video") as usize);
            let mut end = 0;
            for r in &idx.runs {
                assert!(r.start >= end && r.start + r.len() <= b.len() as u64, "{name}: run out of place");
                end = r.start + r.len();
            }
            // the moov box is metadata, the samples are not
            let moov = b.windows(4).position(|w| w == b"moov").unwrap() as u64 - 4;
            assert!(idx.meta.iter().any(|&(a, e)| a == moov && e > moov + 8), "{name}: moov not listed");
            assert!(idx.runs.iter().all(|r| idx.meta.iter().all(|&(a, e)| r.start >= e || r.start + r.len() <= a)), "{name}: samples inside metadata");
        }
        assert!(index(&mut Cursor::new(b"just some text, not a media file at all"), 39).is_none());
    }

    /// A damaged index is an index of something else or none at all: never a panic, and its runs
    /// stay inside the file.
    #[test]
    fn damaged_indexes_do_not_panic() {
        let mut s = 17u64;
        for round in 0..600 {
            let mut b = [M4A, VIDEO, FRAGMENTED][round % 3].to_vec();
            for _ in 0..1 + rng(&mut s) % 8 {
                let at = (rng(&mut s) % b.len() as u64) as usize;
                match rng(&mut s) % 3 {
                    0 => b[at] = rng(&mut s) as u8,
                    1 => b[at] ^= 1 << (rng(&mut s) % 8),
                    _ => b.truncate(at.max(16)),
                }
            }
            if let Some(idx) = index(&mut Cursor::new(&b), b.len() as u64) {
                let mut end = 0;
                for r in &idx.runs {
                    assert!(r.start >= end && r.start + r.len() <= b.len() as u64);
                    end = r.start + r.len();
                }
            }
        }
    }
}
