//! Matroska / WebM: the index.
//!
//! A Matroska file is a tree of EBML elements. Its media sits in `Cluster` elements as blocks,
//! each a frame (or a few laced together) of one track behind a small header; what the tracks
//! are is written in `Tracks`. [`index`] walks the tree (reading element headers only, so the
//! file must be seekable) and lists what [`mp4::index`] lists for an MP4: the frames of tracks a
//! recoder knows — H.264 (as in MP4: length-prefixed NAL units, configuration in the track's
//! codec private data), AAC-LC, MPEG audio Layer III — and the elements that are not media.
//!
//! As there, the index is only a hint: every stream taken apart is proven before it is used.
use super::mp4::{self, Avc, Index, Run, Stream};
use std::io::{Read, Seek, SeekFrom};

/// Most frames and most blocks indexed (a block is a run of its own: a header precedes each),
/// the largest `Tracks` element read, and most elements walked.
const MAX_SAMPLES: usize = 1 << 26;
const MAX_RUNS: usize = 1 << 22;
const MAX_TRACKS: u64 = 4 << 20;
const MAX_ELEMENTS: usize = 1 << 26;
/// Bytes read at a time, and the most looked at of a block (its header, and the sizes of the
/// frames laced into it).
const WINDOW: usize = 8 << 10;
const BLOCK_HEAD: u64 = 4 << 10;

const SEGMENT: u32 = 0x1853_8067;
const TRACKS: u32 = 0x1654_AE6B;
const CLUSTER: u32 = 0x1F43_B675;
const ATTACHMENTS: u32 = 0x1941_A469;
const TRACK_ENTRY: u32 = 0xAE;
const TRACK_NUMBER: u32 = 0xD7;
const CODEC_ID: u32 = 0x86;
const CODEC_PRIVATE: u32 = 0x63A2;
const CONTENT_ENCODINGS: u32 = 0x6D80;
const SIMPLE_BLOCK: u32 = 0xA3;
const BLOCK_GROUP: u32 = 0xA0;
const BLOCK: u32 = 0xA1;
/// The elements a segment holds directly (a cluster of unknown size ends at the next of them):
/// seek head, info, tracks, clusters, cues, attachments, chapters, tags.
const LEVEL1: [u32; 8] = [0x114D_9B74, 0x1549_A966, TRACKS, CLUSTER, 0x1C53_BB6B, ATTACHMENTS, 0x1043_A770, 0x1254_C367];

/// Does `b` (the first bytes of a file) start an EBML document?
pub fn looks_like(b: &[u8]) -> bool { b.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) }

/// A variable-length number at `b[0]`: its value and length. `keep`: with its length marker
/// (element IDs keep it).
fn vint(b: &[u8], keep: bool) -> Option<(u64, usize)> {
    let first = *b.first()?;
    let n = first.leading_zeros() as usize + 1;
    if n > 8 || b.len() < n { return None; }
    let mut v = if keep { first as u64 } else { (first as u64) & (0xFF >> n) };
    for &x in &b[1..n] { v = v << 8 | x as u64; }
    Some((v, n))
}

/// An element header at `b[0]`: its ID, payload size (None: unknown) and header length.
fn header(b: &[u8]) -> Option<(u32, Option<u64>, usize)> {
    let (id, n) = vint(b, true)?;
    if n > 4 { return None; }
    let (size, m) = vint(&b[n..], false)?;
    Some((id as u32, (size != (1u64 << (7 * m)) - 1).then_some(size), n + m))
}

/// The children of the element payload `b` held in memory: (ID, payload).
fn children(b: &[u8]) -> Vec<(u32, &[u8])> {
    let mut out = vec![];
    let mut o = 0usize;
    while let Some((id, Some(size), h)) = b.get(o..).and_then(header) {
        let Some(p) = b.get(o + h..(o + h).saturating_add(size as usize)) else { break };
        out.push((id, p));
        o += h + size as usize;
    }
    out
}

/// What the frames of the tracks in `tracks` (the payload of a `Tracks` element) are, by track
/// number, for the tracks a recoder knows.
fn tracks(b: &[u8], avc: &mut Vec<Avc>) -> Vec<(u64, Stream)> {
    let mut out = vec![];
    for (id, entry) in children(b) {
        if id != TRACK_ENTRY { continue; }
        let fields = children(entry);
        let get = |want: u32| fields.iter().find(|f| f.0 == want).map(|f| f.1);
        // compressed or encrypted frames (header stripping too) are not the codec's bytes
        if get(CONTENT_ENCODINGS).is_some() { continue; }
        let (Some(number), Some(codec)) = (get(TRACK_NUMBER), get(CODEC_ID)) else { continue };
        if number.is_empty() || number.len() > 8 { continue; }
        let number = number.iter().fold(0u64, |a, &x| a << 8 | x as u64);
        let codec = codec.split(|&c| c == 0).next().unwrap_or(codec);
        let what = match codec {
            b"V_MPEG4/ISO/AVC" if avc.len() < 1 << 16 => get(CODEC_PRIVATE).and_then(mp4::avc_config).map(|c| {
                avc.push(c);
                Stream::Avc { cfg: (avc.len() - 1) as u16 }
            }),
            b"A_AAC" => get(CODEC_PRIVATE).and_then(mp4::aac_config).map(|sf| Stream::Aac { sf }),
            b"A_MPEG/L3" => Some(Stream::Mp3),
            _ => None,
        };
        if let Some(w) = what { out.push((number, w)); }
    }
    out
}

/// The frames of the block whose payload of `size` bytes starts with `b` (the block is at file
/// offset `at`): the track number, and the file offset and sizes of its frames.
fn block(b: &[u8], at: u64, size: u64) -> Option<(u64, u64, Vec<u32>)> {
    let (track, n) = vint(b, false)?;
    let flags = *b.get(n + 2)?;
    let mut o = n + 3;
    let lacing = flags >> 1 & 3;
    if lacing == 0 {
        let len = size.checked_sub(o as u64).filter(|&l| l > 0 && l <= u32::MAX as u64)?;
        return Some((track, at + o as u64, vec![len as u32]));
    }
    let count = *b.get(o)? as usize + 1;
    o += 1;
    let mut sizes: Vec<u64> = Vec::with_capacity(count);
    match lacing {
        1 => {
            // Xiph: each size but the last as a run of 255s and a remainder
            for _ in 0..count - 1 {
                let mut len = 0u64;
                loop {
                    let x = *b.get(o)?;
                    o += 1;
                    len += x as u64;
                    if x != 255 { break; }
                }
                sizes.push(len);
            }
        }
        3 => {
            // EBML: the first size, then differences
            let mut last = 0i64;
            for k in 0..count - 1 {
                let (v, m) = vint(b.get(o..)?, false)?;
                o += m;
                last = if k == 0 { v as i64 } else { last + v as i64 - ((1i64 << (7 * m - 1)) - 1) };
                if last < 0 { return None; }
                sizes.push(last as u64);
            }
        }
        _ => {
            let each = size.checked_sub(o as u64)? / count as u64;
            sizes = vec![each; count - 1];
        }
    }
    let used: u64 = sizes.iter().sum::<u64>() + o as u64;
    sizes.push(size.checked_sub(used)?);
    if sizes.iter().any(|&l| l == 0 || l > u32::MAX as u64) { return None; }
    Some((track, at + o as u64, sizes.into_iter().map(|l| l as u32).collect()))
}

/// A window on the file: element headers are a few bytes each and often next to each other.
pub(super) struct Window<'a, F> {
    f: &'a mut F,
    len: u64,
    at: u64,
    buf: Vec<u8>,
}

impl<'a, F: Read + Seek> Window<'a, F> {
    pub(super) fn new(f: &'a mut F, len: u64) -> Self { Window { f, len, at: 0, buf: vec![] } }

    /// The `n` bytes at offset `o` (fewer where the file ends).
    pub(super) fn get(&mut self, o: u64, n: usize) -> Option<&[u8]> {
        let want = (n as u64).min(self.len.saturating_sub(o)) as usize;
        if o < self.at || o + want as u64 > self.at + self.buf.len() as u64 {
            self.f.seek(SeekFrom::Start(o)).ok()?;
            self.buf.resize(want.max(WINDOW), 0);
            let got = crate::bench::read_full(self.f, &mut self.buf).ok()?;
            self.buf.truncate(got);
            self.at = o;
        }
        let a = ((o - self.at) as usize).min(self.buf.len());
        Some(&self.buf[a..(a + want).min(self.buf.len())])
    }
}

/// Index the Matroska file `f` of `len` bytes (None if it is not one, or holds no track a
/// recoder knows). A file that is damaged or cut short is indexed up to there. Leaves the read
/// position unspecified.
pub fn index<F: Read + Seek>(f: &mut F, len: u64) -> Option<Index> {
    let mut w = Window { f, len, at: 0, buf: vec![] };
    if !looks_like(w.get(0, 4)?) { return None; }
    let mut idx = Index::default();
    let mut known: Vec<(u64, Stream)> = vec![];
    let (mut samples, mut elements) = (0usize, 0usize);
    // ranges that touch are one range
    let meta = |idx: &mut Index, a: u64, e: u64| match idx.meta.last_mut() {
        Some(m) if m.1 == a => m.1 = e,
        _ => idx.meta.push((a, e)),
    };
    // (offset, end, inside a cluster) of the elements being walked: the file, the segment, a
    // cluster, a block group
    let mut stack: Vec<(u64, u64, bool)> = vec![(0, len, false)];
    while let Some(&(o, end, in_cluster)) = stack.last() {
        if o + 2 > end {
            stack.pop();
            continue;
        }
        elements += 1;
        if elements > MAX_ELEMENTS { break; }
        let Some((id, size, h)) = w.get(o, 12).and_then(header) else { break };
        let body = o + h as u64;
        // an element of unknown size runs to the end of its parent; a cluster, to the next
        // element a segment holds
        if in_cluster && stack.len() == 3 && LEVEL1.contains(&id) {
            stack.pop();
            stack.last_mut().unwrap().0 = o;
            continue;
        }
        // (the elements that hold others may also end where their parent does: a file cut short)
        let whole = size.and_then(|s| body.checked_add(s)).filter(|&e| e <= end);
        let stop = whole.unwrap_or(end);
        stack.last_mut().unwrap().0 = stop;
        match (id, whole) {
            (SEGMENT, _) if stack.len() == 1 => {
                meta(&mut idx, o, body);
                stack.push((body, stop, false));
            }
            (CLUSTER, _) if stack.len() == 2 => stack.push((body, stop, true)),
            (BLOCK_GROUP, _) if in_cluster && stack.len() == 3 => stack.push((body, stop, true)),
            (SIMPLE_BLOCK | BLOCK, Some(_)) if in_cluster => {
                let size = stop - body;
                let Some(b) = w.get(body, size.min(BLOCK_HEAD) as usize) else { break };
                let Some((track, start, sizes)) = block(b, body, size) else { continue };
                let Some(&(_, what)) = known.iter().find(|t| t.0 == track) else { continue };
                samples += sizes.len();
                if samples > MAX_SAMPLES || idx.runs.len() >= MAX_RUNS { break; }
                idx.runs.push(Run { start, sizes, what });
            }
            (TRACKS, Some(_)) if stack.len() == 2 && stop - body <= MAX_TRACKS => {
                meta(&mut idx, o, stop);
                match w.get(body, (stop - body) as usize) {
                    Some(b) if b.len() as u64 == stop - body => known = tracks(b, &mut idx.avc),
                    _ => break,
                }
            }
            // everything else outside the clusters is not media (seek heads, tags, cues), but
            // for the files attached: fonts, pictures, whatever the detector makes of them
            (_, Some(_)) if stack.len() <= 2 && id != ATTACHMENTS => meta(&mut idx, o, stop),
            // a block or the tracks cut short: nothing after it can be relied on
            (SIMPLE_BLOCK | BLOCK | TRACKS, None) => break,
            _ => {}
        }
    }
    if idx.runs.is_empty() { return None; }
    idx.runs.sort_by_key(|r| r.start);
    let mut end = 0u64;
    idx.runs.retain(|r| {
        let keep = r.start >= end;
        if keep { end = r.start + r.len(); }
        keep
    });
    idx.meta.sort();
    Some(idx)
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use std::io::Cursor;

    /// H.264, AAC and MP3 tracks: the streams of `tone_video.mp4` and `tone_mono_lsf.mp3`.
    pub const FILE: &[u8] = include_bytes!("../../tests/fixtures/tone_video.mkv");
    const MP4: &[u8] = include_bytes!("../../tests/fixtures/tone_video.mp4");
    const MP3: &[u8] = include_bytes!("../../tests/fixtures/tone_stereo.mp3");
    const MP3_LSF: &[u8] = include_bytes!("../../tests/fixtures/tone_mono_lsf.mp3");

    fn rng(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }

    /// The frames of an MP3 file.
    pub fn mp3_frames(b: &[u8]) -> Vec<&[u8]> {
        let len = |o: usize| b.get(o..).and_then(super::super::mp3::header).map(|h| h.frame_len).filter(|l| o + l <= b.len());
        let mut o = (0..b.len()).find(|&o| len(o).is_some_and(|l| len(o + l).is_some())).unwrap();
        let mut out = vec![];
        while let Some(l) = len(o) {
            out.push(&b[o..o + l]);
            o += l;
        }
        out
    }

    /// `v` as a variable-length number of the fewest bytes (or `bytes`).
    fn number(v: u64, bytes: usize) -> Vec<u8> {
        let mut m = bytes.max(1);
        while v >= (1u64 << (7 * m)) - 1 { m += 1; }
        (v | 1u64 << (7 * m)).to_be_bytes()[8 - m..].to_vec()
    }

    /// An element; `open`: of unknown size.
    fn el(id: u32, payload: &[u8], open: bool) -> Vec<u8> {
        let mut out = id.to_be_bytes()[(id.leading_zeros() / 8) as usize..].to_vec();
        out.extend_from_slice(&if open { vec![0xFF] } else { number(payload.len() as u64, 1) });
        out.extend_from_slice(payload);
        out
    }

    /// A Matroska file of one MP3 track: its `frames` laced `per` to a block (`lacing` 0: not
    /// laced, one frame to a block; 1: Xiph; 2: fixed-size where the sizes allow it; 3: EBML),
    /// four blocks to a cluster. `open`: segment and clusters of unknown size; `groups`: block
    /// groups instead of simple blocks. With the offsets of the frames in the file.
    pub fn mp3_file(frames: &[&[u8]], lacing: u8, per: usize, open: bool, groups: bool) -> (Vec<u8>, Vec<u64>) {
        let per = if lacing == 0 { 1 } else { per };
        let head = el(0x1A45_DFA3, &el(0x4282, b"matroska", false), false);
        let entry = [el(TRACK_NUMBER, &[1], false), el(0x83, &[2], false), el(CODEC_ID, b"A_MPEG/L3", false)].concat();
        let mut body = [el(0x1549_A966, &el(0x2A_D7B1, &[0x0F, 0x42, 0x40], false), false), el(TRACKS, &el(TRACK_ENTRY, &entry, false), false)].concat();
        // frame offsets: in a block, in the cluster's payload, in the segment's payload
        let mut at: Vec<u64> = vec![];
        for (c, cluster) in frames.chunks(4 * per).enumerate() {
            let mut payload = el(0xE7, &[c as u8], false);
            for (k, laced) in cluster.chunks(per).enumerate() {
                let fixed = laced.iter().all(|f| f.len() == laced[0].len());
                let lacing = if lacing == 2 && !fixed { 1 } else { lacing };
                let mut b = vec![0x81, 0, k as u8, lacing << 1 | if groups { 0 } else { 0x80 }];
                if lacing != 0 { b.push((laced.len() - 1) as u8); }
                let mut last = 0i64;
                for (i, f) in laced[..laced.len() - 1].iter().enumerate() {
                    let n = f.len() as i64;
                    match lacing {
                        1 => {
                            b.extend(std::iter::repeat_n(255u8, f.len() / 255));
                            b.push((f.len() % 255) as u8);
                        }
                        3 if i == 0 => b.extend(number(n as u64, 1)),
                        3 => {
                            let mut m = 1;
                            while (n - last).abs() > (1i64 << (7 * m - 1)) - 1 { m += 1; }
                            b.extend(number((n - last + (1i64 << (7 * m - 1)) - 1) as u64, m));
                        }
                        _ => {}
                    }
                    last = n;
                }
                let mut o = b.len();
                for f in laced {
                    at.push(o as u64);
                    o += f.len();
                    b.extend_from_slice(f);
                }
                let e = if groups { el(BLOCK_GROUP, &[el(0x9B, &[24], false), el(BLOCK, &b, false)].concat(), false) } else { el(SIMPLE_BLOCK, &b, false) };
                let shift = (payload.len() + e.len() - b.len()) as u64;
                let n = at.len();
                for a in &mut at[n - laced.len()..] { *a += shift; }
                payload.extend_from_slice(&e);
            }
            let e = el(CLUSTER, &payload, open);
            let shift = (body.len() + e.len() - payload.len()) as u64;
            let from = c * 4 * per;
            for a in &mut at[from..] { *a += shift; }
            body.extend_from_slice(&e);
        }
        // tags after the clusters: where an open cluster ends
        body.extend_from_slice(&el(0x1254_C367, &el(0x7373, &el(0x67C8, &[el(0x45A3, b"TITLE", false), el(0x4487, b"tone", false)].concat(), false), false), false));
        let mut file = head;
        file.extend_from_slice(&SEGMENT.to_be_bytes());
        file.extend_from_slice(&if open { vec![0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF] } else { number(body.len() as u64, 8) });
        for a in &mut at { *a += file.len() as u64; }
        file.extend_from_slice(&body);
        (file, at)
    }

    fn check(name: &str, b: &[u8], idx: &Index) {
        let mut end = 0;
        for r in &idx.runs {
            assert!(r.start >= end && r.start + r.len() <= b.len() as u64, "{name}: run out of place");
            assert!(r.sizes.iter().all(|&l| l > 0), "{name}: empty frame");
            end = r.start + r.len();
        }
        for m in idx.meta.windows(2) { assert!(m[0].1 < m[1].0, "{name}: metadata ranges out of order"); }
        assert!(idx.meta.iter().all(|&(a, e)| a < e && e <= b.len() as u64), "{name}: metadata out of the file");
        assert!(idx.runs.iter().all(|r| idx.meta.iter().all(|&(a, e)| r.start >= e || r.start + r.len() <= a)), "{name}: frames inside metadata");
    }

    /// The bytes of the frames of one kind, in file order.
    fn stream(b: &[u8], idx: &Index, want: fn(&Stream) -> bool) -> (Vec<u8>, usize) {
        let mut out = vec![];
        let mut n = 0;
        for r in idx.runs.iter().filter(|r| want(&r.what)) {
            out.extend_from_slice(&b[r.start as usize..(r.start + r.len()) as usize]);
            n += r.sizes.len();
        }
        (out, n)
    }

    #[test]
    fn indexes_tracks_and_metadata() {
        let idx = index(&mut Cursor::new(FILE), FILE.len() as u64).expect("no index");
        check("file", FILE, &idx);
        // the same streams as in the MP4 file they were taken from, frame for frame
        let mp4 = mp4::index(&mut Cursor::new(MP4), MP4.len() as u64).unwrap();
        let kinds: [(fn(&Stream) -> bool, usize); 2] = [(|s| matches!(s, Stream::Avc { cfg: 0 }), 15), (|s| matches!(s, Stream::Aac { sf: 4 }), 66)];
        for (want, frames) in kinds {
            let (a, n) = stream(FILE, &idx, want);
            let (b, m) = stream(MP4, &mp4, want);
            assert_eq!((n, m), (frames, frames));
            assert!(a == b, "a track differs from the MP4 file's");
        }
        assert_eq!(idx.avc, mp4.avc);
        let (a, n) = stream(FILE, &idx, |s| *s == Stream::Mp3);
        assert_eq!(n, 48);
        // (but for the first frame of the file, which holds no audio: a table of contents)
        let frames = mp3_frames(MP3_LSF);
        assert_eq!(frames.len(), 49);
        assert!(a == frames[1..].concat(), "the MP3 track is not the MP3 file's frames");
        // the segment's header, and everything in it up to the first cluster
        let cluster = FILE.windows(4).position(|w| w == CLUSTER.to_be_bytes()).unwrap() as u64;
        assert!(idx.meta.iter().any(|&(a, e)| a < 100 && e == cluster), "metadata {:?}, first cluster at {cluster}", idx.meta);
        assert!(index(&mut Cursor::new(b"just some text, not a media file at all"), 39).is_none());
        assert!(index(&mut Cursor::new(MP4), MP4.len() as u64).is_none());
        // a file cut short is indexed up to there
        let cut = &FILE[..FILE.len() * 2 / 3];
        let part = index(&mut Cursor::new(cut), cut.len() as u64).expect("no index of a file cut short");
        check("cut short", cut, &part);
        let frames = |i: &Index| i.runs.iter().map(|r| r.sizes.len()).sum::<usize>();
        assert!(frames(&part) > 60 && frames(&part) < frames(&idx), "{} frames", frames(&part));
    }

    /// Blocks hold one frame or several laced together, in block groups or alone, in clusters
    /// (and a segment) that say their size or do not.
    #[test]
    fn finds_laced_frames() {
        for (name, mp3) in [("stereo", MP3), ("mono", MP3_LSF)] {
            let frames = mp3_frames(mp3);
            assert!(frames.len() >= 40, "{name}: {} frames", frames.len());
            for lacing in 0..4u8 {
                for per in [1usize, 2, 3, 8] {
                    for (open, groups) in [(false, false), (true, false), (false, true), (true, true)] {
                        let what = format!("{name}, lacing {lacing}, {per} frames a block, open {open}, groups {groups}");
                        let (file, at) = mp3_file(&frames, lacing, per, open, groups);
                        let idx = index(&mut Cursor::new(&file), file.len() as u64).unwrap_or_else(|| panic!("{what}: no index"));
                        check(&what, &file, &idx);
                        let mut found = vec![];
                        for r in &idx.runs {
                            assert_eq!(r.what, Stream::Mp3);
                            let mut o = r.start;
                            for &l in &r.sizes {
                                found.push((o, l as usize));
                                o += l as u64;
                            }
                        }
                        let expect: Vec<(u64, usize)> = at.iter().zip(&frames).map(|(&a, f)| (a, f.len())).collect();
                        assert!(found == expect, "{what}: {} frames of {}", found.len(), expect.len());
                        for (&a, f) in at.iter().zip(&frames) { assert!(&file[a as usize..a as usize + f.len()] == *f, "{what}: the test's own offsets"); }
                        // the tags after the last cluster are metadata
                        let tags = file.windows(4).rposition(|w| w == 0x1254_C367u32.to_be_bytes()).unwrap() as u64;
                        assert!(idx.meta.last() == Some(&(tags, file.len() as u64)), "{what}: metadata {:?}", idx.meta);
                    }
                }
            }
        }
    }

    #[test]
    fn block_headers() {
        // not laced: track 1, 5 bytes of frame
        assert_eq!(block(&[0x81, 0, 0, 0x80, 1, 2, 3, 4, 5], 100, 9), Some((1, 104, vec![5])));
        // a two-byte track number
        assert_eq!(block(&[0x41, 0x00, 0, 0, 0x80, 1, 2], 0, 7), Some((256, 5, vec![2])));
        // Xiph: 3 frames of 300, 2 and the rest
        assert_eq!(block(&[0x81, 0, 0, 0x02, 2, 255, 45, 2], 10, 8 + 300 + 2 + 7), Some((1, 18, vec![300, 2, 7])));
        // EBML: 3 frames of 800, 800 - 300 and the rest
        assert_eq!(block(&[0x81, 0, 0, 0x06, 2, 0x43, 0x20, 0x5E, 0xD3], 0, 9 + 800 + 500 + 1), Some((1, 9, vec![800, 500, 1])));
        // fixed-size: 4 frames of 6
        assert_eq!(block(&[0x81, 0, 0, 0x04, 3], 0, 5 + 24), Some((1, 5, vec![6, 6, 6, 6])));
        // sizes that do not fit the block, an empty frame, a header cut short
        assert_eq!(block(&[0x81, 0, 0, 0x02, 1, 200], 0, 6 + 150), None);
        assert_eq!(block(&[0x81, 0, 0, 0x02, 1, 150], 0, 6 + 150), None);
        assert_eq!(block(&[0x81, 0, 0, 0x06, 2, 0x82, 0x80], 0, 100), None);
        assert_eq!(block(&[0x81, 0, 0, 0x80], 0, 4), None);
        assert_eq!(block(&[0x81, 0, 0, 0x02, 200, 1, 1], 0, 4000), None);
        assert_eq!(block(&[0x81, 0], 0, 100), None);
        assert_eq!(block(&[], 0, 100), None);
    }

    /// A damaged file has an index of something else or none at all: never a panic, and its runs
    /// stay inside the file.
    #[test]
    fn damaged_files_do_not_panic() {
        let frames = mp3_frames(MP3);
        let files = [FILE.to_vec(), mp3_file(&frames, 3, 3, true, true).0, mp3_file(&frames, 1, 8, false, false).0];
        let mut s = 23u64;
        for round in 0..900 {
            let mut b = files[round % 3].clone();
            for _ in 0..1 + rng(&mut s) % 8 {
                let at = (rng(&mut s) % b.len() as u64) as usize;
                match rng(&mut s) % 3 {
                    0 => b[at] = rng(&mut s) as u8,
                    1 => b[at] ^= 1 << (rng(&mut s) % 8),
                    _ => b.truncate(at.max(16)),
                }
            }
            if let Some(idx) = index(&mut Cursor::new(&b), b.len() as u64) { check("damaged", &b, &idx); }
        }
        // nothing but tiny elements, and a file that is shorter than it says
        let tiny = [&[0x1A, 0x45, 0xDF, 0xA3, 0x80][..], &[0xEC, 0x80].repeat(50_000)].concat();
        assert!(index(&mut Cursor::new(&tiny), tiny.len() as u64).is_none());
        assert!(index(&mut Cursor::new(&FILE[..500]), FILE.len() as u64).is_none());
    }
}
