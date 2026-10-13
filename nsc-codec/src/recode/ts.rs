//! MPEG transport streams: where the video and the audio are, and H.264 as a byte stream.
//!
//! A transport stream is packets of 188 bytes, each a few bytes of header and a piece of one
//! elementary stream; which stream a packet belongs to (its PID) and what that stream is, two
//! small tables say (PAT, PMT). [`index`] reads the tables and gives each packet's payload —
//! past the packet header and, where a PES packet starts, its header — as a run of its stream:
//! gathered, the runs are the H.264 or audio stream as its encoder wrote it.
//!
//! H.264 in a transport stream is in byte-stream form (Annex B): NAL units behind start codes
//! instead of lengths. [`split`] turns it into the length-prefixed form the H.264 recoder
//! reads, and notes how many zero bytes each start code had; [`rebuild`] puts the start codes
//! back.
use super::mkv::Window;
use super::mp4::{Index, Run, Stream};
use std::io::{Read, Seek};

const PACKET: u64 = 188;
/// Most packets indexed.
const MAX_PACKETS: usize = 1 << 24;

fn put_var(o: &mut Vec<u8>, v: usize) { crate::nsc3::format::put_var(o, v as u64) }
fn get_var(b: &[u8], p: &mut usize) -> Option<usize> { super::h264::get_var(b, p) }

pub fn looks_like(b: &[u8]) -> bool { b.first() == Some(&0x47) }

/// The payload of the section a table packet starts: past the pointer field.
fn section(p: &[u8], off: usize) -> Option<&[u8]> {
    let at = off + 1 + *p.get(off)? as usize;
    p.get(at..)
}

/// The index of the transport stream `f` of `len` bytes: the payload of every packet of its
/// H.264 and audio streams. None unless it starts with packets and names such a stream.
pub fn index<F: Read + Seek>(f: &mut F, len: u64) -> Option<Index> {
    let mut w = Window::new(f, len);
    for k in 0..3 { if w.get(k * PACKET, 1)? != [0x47] { return None; } }
    let mut idx = Index::default();
    let mut pmt: Option<usize> = None;
    let mut streams: Vec<(usize, Stream)> = vec![];
    let (mut o, mut n) = (0u64, 0usize);
    while o + PACKET <= len && n < MAX_PACKETS {
        let p = w.get(o, PACKET as usize)?;
        if p.len() < PACKET as usize || p[0] != 0x47 { break; }
        n += 1;
        let pid = ((p[1] & 0x1F) as usize) << 8 | p[2] as usize;
        let (start, afc, scrambled) = (p[1] & 0x40 != 0, p[3] >> 4 & 3, p[3] >> 6 != 0);
        let mut off = 4 + if afc & 2 != 0 { 1 + p[4] as usize } else { 0 };
        if afc & 1 != 0 && off < PACKET as usize && !scrambled {
            if pid == 0 && start {
                // the program association table: the first program's map
                if let Some(s) = section(p, off).filter(|s| s.len() >= 12 && s[0] == 0) {
                    let end = (3 + (((s[1] & 0x0F) as usize) << 8 | s[2] as usize)).saturating_sub(4).min(s.len());
                    pmt = s.get(8..end).and_then(|e| e.chunks_exact(4).find(|e| e[0] | e[1] != 0)).map(|e| ((e[2] & 0x1F) as usize) << 8 | e[3] as usize).or(pmt);
                }
            } else if Some(pid) == pmt && start && streams.is_empty() {
                // the program map table: each stream's kind and PID
                if let Some(s) = section(p, off).filter(|s| s.len() >= 12 && s[0] == 2) {
                    let end = (3 + (((s[1] & 0x0F) as usize) << 8 | s[2] as usize)).saturating_sub(4).min(s.len());
                    let mut q = 12 + (((s[10] & 0x0F) as usize) << 8 | s[11] as usize);
                    while q + 5 <= end {
                        let es = ((s[q + 1] & 0x1F) as usize) << 8 | s[q + 2] as usize;
                        match s[q] {
                            0x1B => streams.push((es, Stream::AnnexB { pid: es as u16 })),
                            0x0F => streams.push((es, Stream::Adts { pid: es as u16 })),
                            0x03 | 0x04 => streams.push((es, Stream::Mp3)),
                            _ => {}
                        }
                        q += 5 + (((s[q + 3] & 0x0F) as usize) << 8 | s[q + 4] as usize);
                    }
                }
            } else if let Some(&(_, what)) = streams.iter().find(|s| s.0 == pid) {
                // where a PES packet starts, its header is not the stream's
                if start {
                    off = match p.get(off..off + 9) {
                        Some(h) if h[..3] == [0, 0, 1] => off + 9 + h[8] as usize,
                        _ => PACKET as usize,
                    };
                }
                if off < PACKET as usize { idx.runs.push(Run { start: o + off as u64, sizes: vec![(PACKET as usize - off) as u32], what }); }
            }
        }
        o += PACKET;
    }
    (!idx.runs.is_empty()).then_some(idx)
}

/// Does a file start as an H.264 byte stream does: a start code, then a parameter set, an
/// access unit delimiter or an SEI message?
pub fn looks_like_bytestream(b: &[u8]) -> bool {
    let n = if b.starts_with(&[0, 0, 0, 1]) { 4 } else if b.starts_with(&[0, 0, 1]) { 3 } else { return false };
    b.get(n).is_some_and(|x| x & 0x80 == 0 && matches!(x & 31, 6 | 7 | 9))
}

/// The index of the H.264 byte stream `f` of `len` bytes (a `.h264` file): one run of its NAL
/// units, each with its start code, so that a block takes whole units.
pub fn index_bytestream<F: Read + Seek>(f: &mut F, len: u64) -> Option<Index> {
    let mut w = Window::new(f, len);
    if !looks_like_bytestream(w.get(0, 5)?) { return None; }
    let mut sizes: Vec<u32> = vec![];
    // where the unit being walked began (with its start code's zeros), and the zeros just seen
    let (mut from, mut zeros, mut o) = (0u64, 0u64, 0u64);
    while o < len && sizes.len() < MAX_PACKETS {
        let chunk = w.get(o, 8 << 10)?;
        if chunk.is_empty() { break; }
        for (k, &x) in chunk.iter().enumerate() {
            let at = o + k as u64;
            if x == 1 && zeros >= 2 && at - zeros > from {
                let unit = at - zeros - from;
                if unit > u32::MAX as u64 { return None; }
                sizes.push(unit as u32);
                from = at - zeros;
            }
            zeros = if x == 0 { zeros + 1 } else { 0 };
        }
        o += chunk.len() as u64;
    }
    if o < len || len - from > u32::MAX as u64 { return None; }
    sizes.push((len - from) as u32);
    Some(Index { runs: vec![Run { start: 0, sizes, what: Stream::AnnexB { pid: 0 } }], ..Index::default() })
}

/// An H.264 byte stream as NAL units behind four-byte lengths.
pub struct Split {
    /// What stands before the first start code, and each start code's zero bytes.
    pub record: Vec<u8>,
    pub prefixed: Vec<u8>,
    /// Bytes of each unit with its length.
    pub sizes: Vec<u32>,
}

/// Take the byte stream `b` apart at its start codes; None if it has none.
pub fn split(b: &[u8]) -> Option<Split> {
    // each start code: (where its zeros begin, where the unit after it begins)
    let mut codes: Vec<(usize, usize)> = vec![];
    let mut i = 0usize;
    while i + 3 <= b.len() {
        if b[i + 2] > 1 {
            i += 3;
        } else if b[i] == 0 && b[i + 1] == 0 && b[i + 2] == 1 {
            let mut z = i;
            while z > codes.last().map_or(0, |c| c.1) && b[z - 1] == 0 { z -= 1; }
            codes.push((z, i + 3));
            i += 3;
        } else {
            i += 1;
        }
    }
    let first = codes.first()?.0;
    let mut s = Split { record: vec![], prefixed: Vec::with_capacity(b.len() + codes.len()), sizes: Vec::with_capacity(codes.len()) };
    put_var(&mut s.record, first);
    s.record.extend_from_slice(&b[..first]);
    for (k, &(z, at)) in codes.iter().enumerate() {
        let end = codes.get(k + 1).map_or(b.len(), |c| c.0);
        if end - at > u32::MAX as usize - 4 { return None; }
        put_var(&mut s.record, at - 1 - z);
        s.prefixed.extend_from_slice(&((end - at) as u32).to_be_bytes());
        s.prefixed.extend_from_slice(&b[at..end]);
        s.sizes.push((end - at + 4) as u32);
    }
    Some(s)
}

/// Append the byte stream [`split`] took apart.
pub fn rebuild(record: &[u8], prefixed: &[u8], out: &mut Vec<u8>) -> Result<(), String> {
    let bad = || "h264 byte stream: bad record (corrupt file)".to_string();
    let mut p = 0usize;
    let n = get_var(record, &mut p).ok_or_else(bad)?;
    out.extend_from_slice(record.get(p..p.checked_add(n).ok_or_else(bad)?).ok_or_else(bad)?);
    p += n;
    let mut q = 0usize;
    while q < prefixed.len() {
        let len = u32::from_be_bytes(prefixed.get(q..q + 4).ok_or_else(bad)?.try_into().unwrap()) as usize;
        let unit = prefixed.get(q + 4..(q + 4).checked_add(len).ok_or_else(bad)?).ok_or_else(bad)?;
        let zeros = get_var(record, &mut p).ok_or_else(bad)?;
        if zeros > 1 << 20 { return Err(bad()); }
        out.resize(out.len() + zeros, 0);
        out.push(1);
        out.extend_from_slice(unit);
        q += 4 + len;
    }
    if p != record.len() { return Err(bad()); }
    Ok(())
}

#[cfg(test)]
pub mod tests {
    use super::*;

    fn rng(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }

    /// A transport stream of an H.264 stream (PID 256) and an ADTS stream (PID 257), cut into
    /// packets in turn, with the tables first.
    pub fn file(video: &[u8], audio: &[u8]) -> Vec<u8> {
        let packet = |pid: usize, start: bool, head: &[u8], data: &[u8]| -> Vec<u8> {
            let room = 184 - head.len();
            let pad = room - data.len();
            let mut p = vec![0x47, (start as u8) << 6 | (pid >> 8) as u8, pid as u8, if pad > 0 { 0x30 } else { 0x10 }];
            if pad > 0 {
                p.push((pad - 1) as u8);
                if pad > 1 { p.push(0); p.resize(4 + pad, 0xFF); }
            }
            p.extend_from_slice(head);
            p.extend_from_slice(data);
            assert_eq!(p.len(), 188);
            p
        };
        let table = |pid: usize, body: &[u8]| { let mut d = vec![0u8]; d.extend_from_slice(body); d.resize(184, 0xFF); packet(pid, true, &[], &d) };
        let mut out = table(0, &[0, 0xB0, 13, 0, 1, 0xC1, 0, 0, 0, 1, 0xF0, 0x00, 1, 2, 3, 4]);
        out.extend(table(0x1000, &[2, 0xB0, 23, 0, 1, 0xC1, 0, 0, 0xE1, 0x00, 0xF0, 0, 0x1B, 0xE1, 0x00, 0xF0, 0, 0x0F, 0xE1, 0x01, 0xF0, 0, 1, 2, 3, 4]));
        let pes = [0, 0, 1, 0xE0, 0, 0, 0x80, 0x80, 5, 1, 2, 3, 4, 5];
        let (mut v, mut a, mut k) = (video, audio, 0usize);
        while !v.is_empty() || !a.is_empty() {
            let (src, pid) = if (k % 4 != 3 && !v.is_empty()) || a.is_empty() { (&mut v, 256) } else { (&mut a, 257) };
            let head: &[u8] = if k % 5 < 2 { &pes } else { &[] };
            let n = src.len().min(184 - head.len()).min(if k % 7 == 6 { 100 } else { 184 });
            out.extend(packet(pid, !head.is_empty(), head, &src[..n]));
            *src = &src[n..];
            k += 1;
        }
        out
    }

    #[test]
    fn streams_are_indexed_and_byte_streams_come_back() {
        let mut s = 9u64;
        let video: Vec<u8> = (0..40).flat_map(|k| {
            let mut n = vec![0u8; 2 + k % 3];
            n.push(1);
            n.extend((0..50 + rng(&mut s) % 900).map(|_| (rng(&mut s) % 251) as u8 + 2));
            if k % 6 == 5 { n.extend_from_slice(&[0, 0, 0]); }
            n
        }).collect();
        let audio: Vec<u8> = (0..6000).map(|_| rng(&mut s) as u8).collect();
        let f = file(&video, &audio);
        let idx = index(&mut std::io::Cursor::new(&f), f.len() as u64).unwrap();
        let of = |what: Stream| -> Vec<u8> { idx.runs.iter().filter(|r| r.what == what).flat_map(|r| f[r.start as usize..(r.start + r.len()) as usize].to_vec()).collect() };
        assert!(of(Stream::AnnexB { pid: 256 }) == video && of(Stream::Adts { pid: 257 }) == audio);
        assert!(index(&mut std::io::Cursor::new(&f[5..]), f.len() as u64 - 5).is_none());
        // the byte stream
        for b in [&video[..], &[&b"junk"[..], &video[..]].concat(), &[0, 0, 1], &[0, 0, 0, 1, 9, 0, 0, 1, 0, 0, 1, 7, 0]] {
            let sp = split(b).unwrap();
            assert_eq!(sp.sizes.iter().map(|&x| x as usize).sum::<usize>(), sp.prefixed.len());
            let mut out = vec![];
            rebuild(&sp.record, &sp.prefixed, &mut out).unwrap();
            assert!(out == b);
        }
        assert!(split(b"no start code here").is_none());
        // a byte stream as a file: a run of whole units
        let mut file = vec![0, 0, 0, 1, 0x67, 1, 2, 3];
        file.extend_from_slice(&video);
        let idx = index_bytestream(&mut std::io::Cursor::new(&file), file.len() as u64).unwrap();
        assert_eq!(idx.runs[0].len(), file.len() as u64);
        let mut at = 0usize;
        for &n in &idx.runs[0].sizes {
            assert!(file[at..].starts_with(&[0, 0, 1]) || file[at..].starts_with(&[0, 0, 0]), "unit at {at}");
            at += n as usize;
        }
        assert!(idx.runs[0].sizes.len() > 40 && index_bytestream(&mut std::io::Cursor::new(&video[1..]), 10).is_none());
        // damaged: no panic, and what is indexed lies in the file
        for _ in 0..2000 {
            let mut bad = f.clone();
            for _ in 0..3 {
                let i = (rng(&mut s) % bad.len() as u64) as usize;
                bad[i] ^= 1 << (rng(&mut s) % 8);
            }
            if let Some(idx) = index(&mut std::io::Cursor::new(&bad), bad.len() as u64) { assert!(idx.runs.iter().all(|r| r.start + r.len() <= bad.len() as u64)); }
            let sp = split(&video).unwrap();
            let (mut rec, mut pre) = (sp.record, sp.prefixed);
            let which = if rng(&mut s) % 2 == 0 { &mut rec } else { &mut pre };
            let i = (rng(&mut s) % which.len() as u64) as usize;
            which[i] ^= 1 << (rng(&mut s) % 8);
            let mut out = vec![];
            let _ = rebuild(&rec, &pre, &mut out);
            assert!(out.len() < video.len() * 2 + (2 << 20));
        }
    }
}
