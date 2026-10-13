//! AVI: where the audio of a file is.
//!
//! An AVI is RIFF chunks: a header list that describes each stream, then a `movi` list of the
//! streams' chunks interleaved — `00dc` a video frame, `01wb` a piece of audio — and an index.
//! [`index`] walks the chunk headers (never the frames) and gives the audio chunks of the
//! streams it knows as runs: MP3, whose frames run on across the chunks and are one stream
//! again once gathered, and PCM, which is samples for the audio model. Video frames need no
//! index here: a Motion-JPEG frame is a JPEG and is found as one.
use super::mkv::Window;
use super::mp4::{Index, Run, Stream};
use std::io::{Read, Seek};

/// Most chunks walked, streams described, and lists nested.
const MAX_CHUNKS: usize = 1 << 24;
const MAX_STREAMS: usize = 100;

pub fn looks_like(b: &[u8]) -> bool { b.len() >= 12 && &b[..4] == b"RIFF" && &b[8..12] == b"AVI " }

fn le32(b: &[u8], o: usize) -> Option<u64> { Some(u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?) as u64) }

/// The index of the AVI file `f` of `len` bytes. A file cut short gives what was walked.
pub fn index<F: Read + Seek>(f: &mut F, len: u64) -> Option<Index> {
    let mut w = Window::new(f, len);
    if !looks_like(w.get(0, 12)?) { return None; }
    let mut idx = Index::default();
    // what each stream's audio is, by its number
    let mut streams: Vec<Option<Stream>> = vec![];
    let mut audio = false;
    let mut chunks = 0usize;
    // lists being walked: where each ends
    let mut ends: Vec<u64> = vec![len];
    let mut o = 0u64;
    let mut meta_from = 0u64;
    let mut in_movi = 0usize;
    while o + 8 <= len && chunks < MAX_CHUNKS {
        while ends.len() > 1 && o >= *ends.last().unwrap() {
            ends.pop();
            if in_movi > ends.len() {
                (in_movi, meta_from) = (0, o);
            }
        }
        let Some(h) = w.get(o, 12).filter(|h| h.len() >= 8) else { break };
        let (id, size) = ([h[0], h[1], h[2], h[3]], le32(h, 4)?);
        let end = (o + 8 + size).min(*ends.last().unwrap());
        chunks += 1;
        if (&id == b"RIFF" || &id == b"LIST") && h.len() >= 12 && ends.len() < 8 {
            if &h[8..12] == b"movi" && in_movi == 0 {
                in_movi = ends.len() + 1;
                idx.meta.push((meta_from, o + 12));
            }
            ends.push(end);
            o += 12;
            continue;
        }
        if in_movi == 0 {
            match &id {
                b"strh" => {
                    audio = w.get(o + 8, 4)? == b"auds";
                    if streams.len() < MAX_STREAMS { streams.push(None); }
                }
                b"strf" if audio && size >= 16 => {
                    let s = w.get(o + 8, 16)?;
                    if s.len() < 16 { break; }
                    let (tag, channels, bits) = (u16::from_le_bytes([s[0], s[1]]), u16::from_le_bytes([s[2], s[3]]), u16::from_le_bytes([s[14], s[15]]));
                    let what = match tag {
                        1 if (1..=8).contains(&channels) && matches!(bits, 8 | 16 | 24) => Some(Stream::Pcm { channels: channels as u8, bits: bits as u8 }),
                        0x55 => Some(Stream::Mp3),
                        _ => None,
                    };
                    if let Some(last) = streams.last_mut() { *last = what; }
                }
                _ => {}
            }
        } else if id[2..] == *b"wb" && id[0].is_ascii_digit() && id[1].is_ascii_digit() && size > 0 && o + 8 + size <= len && size <= u32::MAX as u64 {
            // a piece of the audio of stream NN
            let n = ((id[0] - b'0') * 10 + (id[1] - b'0')) as usize;
            if let Some(Some(what)) = streams.get(n) { idx.runs.push(Run { start: o + 8, sizes: vec![size as u32], what: *what }); }
        }
        o = o + 8 + size + (size & 1);
    }
    if in_movi == 0 || meta_from > 0 { idx.meta.push((meta_from, o.min(len).max(meta_from))); }
    idx.meta.retain(|m| m.1 > m.0);
    (!idx.runs.is_empty()).then_some(idx)
}

#[cfg(test)]
pub mod tests {
    use super::*;

    fn chunk(id: &[u8; 4], data: &[u8]) -> Vec<u8> {
        let mut o = id.to_vec();
        o.extend_from_slice(&(data.len() as u32).to_le_bytes());
        o.extend_from_slice(data);
        if data.len() % 2 == 1 { o.push(0); }
        o
    }

    fn list(id: &[u8; 4], kind: &[u8; 4], parts: &[Vec<u8>]) -> Vec<u8> {
        let body: Vec<u8> = kind.iter().copied().chain(parts.iter().flatten().copied()).collect();
        let mut o = id.to_vec();
        o.extend_from_slice(&(body.len() as u32).to_le_bytes());
        o.extend(body);
        o
    }

    /// An AVI of a video stream and an audio stream of format `tag`, whose audio is `audio` cut
    /// into chunks of `piece` bytes between the video's frames.
    pub fn file(tag: u16, channels: u16, bits: u16, audio: &[u8], piece: usize) -> Vec<u8> {
        let mut strf = vec![0u8; 18];
        strf[..2].copy_from_slice(&tag.to_le_bytes());
        strf[2..4].copy_from_slice(&channels.to_le_bytes());
        strf[14..16].copy_from_slice(&bits.to_le_bytes());
        let strh = |kind: &[u8; 4]| { let mut h = kind.to_vec(); h.resize(56, 0); h };
        let hdrl = list(b"LIST", b"hdrl", &[
            chunk(b"avih", &[0; 56]),
            list(b"LIST", b"strl", &[chunk(b"strh", &strh(b"vids")), chunk(b"strf", &[0; 40])]),
            list(b"LIST", b"strl", &[chunk(b"strh", &strh(b"auds")), chunk(b"strf", &strf)]),
        ]);
        let mut movi = vec![];
        for (k, a) in audio.chunks(piece).enumerate() {
            movi.push(chunk(b"00dc", &vec![k as u8; 301 + k % 7]));
            movi.push(chunk(b"01wb", a));
        }
        list(b"RIFF", b"AVI ", &[hdrl, list(b"LIST", b"movi", &movi), chunk(b"idx1", &[7; 64])])
    }

    #[test]
    fn audio_chunks_are_indexed() {
        let audio: Vec<u8> = (0..10_000u32).map(|i| (i * 7 % 251) as u8).collect();
        for (tag, what) in [(1u16, Some(Stream::Pcm { channels: 2, bits: 16 })), (0x55, Some(Stream::Mp3)), (0x2000, None)] {
            let f = file(tag, 2, 16, &audio, 1000);
            let idx = index(&mut std::io::Cursor::new(&f), f.len() as u64);
            let Some(what) = what else { assert!(idx.is_none()); continue };
            let idx = idx.unwrap();
            assert!(idx.runs.iter().all(|r| r.what == what));
            let got: Vec<u8> = idx.runs.iter().flat_map(|r| f[r.start as usize..(r.start + r.len()) as usize].to_vec()).collect();
            assert!(got == audio, "tag {tag}: the runs are the audio");
            // the lists' headers and the index are metadata; the frames are not
            assert!(idx.meta.len() == 2 && idx.meta[0].0 == 0 && idx.meta[1].1 == f.len() as u64, "{:?}", idx.meta);
            assert!(idx.meta[0].1 <= idx.runs[0].start && idx.meta[1].0 >= idx.runs.last().unwrap().start);
            // cut short: what was walked
            let cut = &f[..f.len() * 2 / 3];
            let part = index(&mut std::io::Cursor::new(cut), cut.len() as u64).unwrap();
            assert!(!part.runs.is_empty() && part.runs.len() < idx.runs.len() && part.runs.iter().all(|r| r.start + r.len() <= cut.len() as u64));
        }
        assert!(index(&mut std::io::Cursor::new(b"RIFF\0\0\0\0WAVEfmt "), 16).is_none());
        // damaged files do not panic
        let f = file(1, 1, 8, &audio, 333);
        let mut s = 5u64;
        for _ in 0..2000 {
            let mut bad = f.clone();
            for _ in 0..3 {
                s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                let i = (s % bad.len() as u64) as usize;
                bad[i] ^= 1 << (s >> 20 & 7);
            }
            if let Some(idx) = index(&mut std::io::Cursor::new(&bad), bad.len() as u64) {
                assert!(idx.runs.iter().all(|r| r.start + r.len() <= bad.len() as u64));
            }
        }
    }
}
