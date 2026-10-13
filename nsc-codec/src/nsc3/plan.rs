//! Block planning: read the input stream, run the detector over it, and cut fixed-size blocks whose
//! segment tables describe the typed regions inside them.
//!
//! The detector always sees exactly `LOOKAHEAD` bytes (or everything left at the end of the stream),
//! and reads use full-buffer semantics. So the plan depends only on the input bytes and the block
//! size, never on how the input arrives (file or pipe). A typed segment cut by a block boundary is
//! cut at its alignment (an image row, an audio frame) and continues, with the same parameters,
//! in the next block.
use crate::bench::read_full;
use crate::detect::{Detector, Seg, LOOKAHEAD};
use std::io::Read;

#[derive(Clone, Debug, PartialEq)]
pub struct PSeg {
    pub kind: u8,
    pub len: u32,
    pub params: Vec<u8>,
    pub label: &'static str,
}

pub struct PlannedBlock {
    pub data: Vec<u8>,
    pub segs: Vec<PSeg>,
    /// Estimated decompressed size of the recompressible streams inside (memory planning).
    pub inner_est: usize,
    /// Sample runs inside the block that a container's index points out, by offset.
    pub hints: Vec<Hint>,
}

/// A run of headerless samples inside a block (offset `start`), known from the index of the
/// container they are in: what a recoder needs to take them apart.
#[derive(Clone, Debug, PartialEq)]
pub struct Hint {
    pub start: usize,
    pub sizes: Vec<u32>,
    /// What the samples are; H.264 configurations are numbered across the whole input.
    pub what: crate::recode::mp4::Stream,
    /// An H.264 run's decoder configuration.
    pub avc: Option<std::sync::Arc<crate::recode::mp4::Avc>>,
}

pub struct Planner<R: Read> {
    r: R,
    buf: Vec<u8>,
    head: usize,
    abs: u64,
    eof: bool,
    det: Detector,
    cur: Option<(Seg, u64)>,
    block: usize,
    /// When set, a block whose cut falls inside a recompressible stream (gzip, zip entry, PDF
    /// stream, PNG image data) is extended to the stream's end, up to this many bytes in total.
    pub extend: Option<usize>,
    /// Streams of the format recoders also keep a block from being cut inside them.
    pub recode: bool,
    /// Container indexes of the stream, as stream offsets: the sample runs still ahead (by
    /// offset), the position reached in the first of them (sample, its offset), and the byte
    /// ranges of container metadata.
    runs: std::collections::VecDeque<crate::recode::mp4::Run>,
    avc: Vec<std::sync::Arc<crate::recode::mp4::Avc>>,
    run_pos: (usize, u64),
    meta: Vec<(u64, u64)>,
    /// Every segment piece as planned: (absolute offset, kind, length, label) — for `nsc analyze`.
    pub log: Option<Vec<(u64, u8, u64, &'static str)>>,
}

impl<R: Read> Planner<R> {
    pub fn new(r: R, block: usize) -> Self {
        Planner { r, buf: Vec::new(), head: 0, abs: 0, eof: false, det: Detector::new(), cur: None, block: block.max(1), extend: None, recode: false,
                  runs: Default::default(), avc: vec![], run_pos: (0, 0), meta: vec![], log: None }
    }

    /// Register the index of a container file that starts at stream offset `at`. Indexes are
    /// added in stream order, before planning starts.
    pub fn add_index(&mut self, at: u64, idx: crate::recode::mp4::Index) {
        // H.264 configurations are renumbered across all the indexes
        let base = self.avc.len();
        self.avc.extend(idx.avc.into_iter().map(std::sync::Arc::new));
        for mut r in idx.runs {
            r.start += at;
            if let crate::recode::mp4::Stream::Avc { cfg } = &mut r.what {
                match u16::try_from(base + *cfg as usize) { Ok(n) => *cfg = n, Err(_) => continue }
            }
            if self.runs.back().is_none_or(|l| l.start + l.len() <= r.start) { self.runs.push_back(r); }
        }
        self.meta.extend(idx.meta.iter().map(|&(a, b)| (a + at, b + at)));
        self.run_pos = (0, self.runs.front().map_or(0, |r| r.start));
    }

    /// The sample runs inside the block `[b0, b1)`: whole samples only (one cut by the block's
    /// edge stays plain bytes).
    fn hints(&mut self, b0: u64, b1: u64) -> Vec<Hint> {
        let mut out = vec![];
        while let Some(r) = self.runs.front() {
            let (mut k, mut o) = self.run_pos;
            // skip samples that started before the block (cut by the previous block's edge)
            while k < r.sizes.len() && o < b0 { o += r.sizes[k] as u64; k += 1; }
            let first = k;
            let start = o;
            while k < r.sizes.len() && o + r.sizes[k] as u64 <= b1 { o += r.sizes[k] as u64; k += 1; }
            if k > first && start >= b0 {
                let avc = match r.what { crate::recode::mp4::Stream::Avc { cfg } => self.avc.get(cfg as usize).cloned(), _ => None };
                out.push(Hint { start: (start - b0) as usize, sizes: r.sizes[first..k].to_vec(), what: r.what, avc });
            }
            if k < r.sizes.len() {
                // the run goes on in the next block
                self.run_pos = (k, o);
                break;
            }
            self.runs.pop_front();
            self.run_pos = (0, self.runs.front().map_or(0, |r| r.start));
        }
        out
    }

    /// Ensure `need` unconsumed bytes are buffered (or the stream has ended).
    fn fill(&mut self, need: usize) -> Result<(), String> {
        let have = self.buf.len() - self.head;
        if have >= need || self.eof { return Ok(()); }
        if self.head > 0 {
            self.buf.drain(..self.head);
            self.head = 0;
        }
        while !self.eof && self.buf.len() < need {
            let want = (need - self.buf.len()).max(1 << 20);
            let old = self.buf.len();
            self.buf.resize(old + want, 0);
            let n = read_full(&mut self.r, &mut self.buf[old..]).map_err(|e| format!("read: {e}"))?;
            self.buf.truncate(old + n);
            if n < want { self.eof = true; }
        }
        Ok(())
    }

    /// Where to end a full block of `filled` bytes when a stream crosses its cut: the stream's
    /// end, if the extended block fits `cap` bytes and the stream budget. Also the estimated
    /// decompressed size of the streams in the block.
    fn stream_cut(&mut self, filled: usize, cap: usize) -> Result<(Option<usize>, usize), String> {
        use crate::nsc3::block::INNER_BUDGET;
        use crate::transform::deflate::{inflate::ExtentError, scan};
        let avail = self.buf.len() - self.head;
        if avail <= filled { return Ok((None, 0)); }
        let sv = scan::survey(&self.buf[self.head..self.head + avail.min(cap)], filled, INNER_BUDGET, self.recode);
        let fits = |end: usize, inner: usize| (end <= cap && sv.inner + inner <= INNER_BUDGET).then_some(end);
        match sv.cross {
            scan::Cross::None => Ok((None, sv.inner)),
            scan::Cross::Ends { end, inner } => Ok((fits(end, inner), sv.inner + inner)),
            scan::Cross::Open { start } => {
                // read further ahead (doubling) until the stream's end is buffered or the cap is hit
                loop {
                    let have = self.buf.len() - self.head;
                    if have >= cap || self.eof { return Ok((None, sv.inner)); }
                    self.fill((have * 2).min(cap))?;
                    let have = self.buf.len() - self.head;
                    match scan::stream_extent(&self.buf[self.head..self.head + have.min(cap)], start, INNER_BUDGET, self.recode) {
                        Ok((end, inner)) => return Ok((fits(end, inner), sv.inner + inner)),
                        Err(ExtentError::Truncated) => continue,
                        Err(_) => return Ok((None, sv.inner)),
                    }
                }
            }
        }
    }

    pub fn next_block(&mut self) -> Result<Option<PlannedBlock>, String> {
        let mut segs: Vec<PSeg> = Vec::new();
        let mut filled = 0usize;
        let (mut limit, mut checked, mut inner_est) = (self.block, false, 0usize);
        loop {
        while filled < limit {
            self.fill(filled + LOOKAHEAD)?;
            let avail = self.buf.len() - self.head - filled;
            if avail == 0 { break; }
            if self.cur.is_none() {
                let pos = self.head + filled;
                let window = avail.min(LOOKAHEAD);
                let at_end = self.eof && avail <= LOOKAHEAD;
                let s = self.det.next(&self.buf[pos..pos + window], self.abs + filled as u64, at_end);
                let len = s.len.max(1);
                self.cur = Some((s, len));
            }
            let (s, rem) = self.cur.as_mut().unwrap();
            let space = limit - filled;
            let mut take = (*rem).min(space as u64).min(avail as u64) as usize;
            if (take as u64) < *rem && take == space && s.align > 1 {
                let a = s.align as usize;
                if take >= a { take -= take % a; } else if filled > 0 { break; }
            }
            if take == 0 { break; }
            if let Some(log) = self.log.as_mut() { log.push((self.abs + filled as u64, s.kind, take as u64, s.label)); }
            match segs.last_mut() {
                Some(l) if l.kind == s.kind && l.params == s.params && l.label == s.label => l.len += take as u32,
                _ => segs.push(PSeg { kind: s.kind, len: take as u32, params: s.params.clone(), label: s.label }),
            }
            filled += take;
            *rem -= take as u64;
            if *rem == 0 { self.cur = None; }
        }
        if let (Some(cap), false) = (self.extend, checked) {
            checked = true;
            if filled == limit {
                let (cut, est) = self.stream_cut(filled, cap.max(limit))?;
                inner_est = est;
                if let Some(end) = cut { limit = end; continue; }
            }
        }
        break;
        }
        if filled == 0 { return Ok(None); }
        let data = self.buf[self.head..self.head + filled].to_vec();
        let (b0, b1) = (self.abs, self.abs + filled as u64);
        // container metadata is modelled, whatever the detector made of the bytes around it
        for &(m0, m1) in &self.meta {
            if m0 < b1 && m1 > b0 { retype(&mut segs, (m0.max(b0) - b0) as usize, (m1.min(b1) - b0) as usize, "container"); }
        }
        let hints = self.hints(b0, b1);
        self.head += filled;
        self.abs += filled as u64;
        Ok(Some(PlannedBlock { data, segs, inner_est, hints }))
    }
}

/// Make the bytes `[a, b)` of a block's segments generic data labelled `label`.
fn retype(segs: &mut Vec<PSeg>, a: usize, b: usize, label: &'static str) {
    let mut out: Vec<PSeg> = Vec::with_capacity(segs.len() + 2);
    let mut push = |s: PSeg| {
        if s.len == 0 { return; }
        match out.last_mut() {
            Some(l) if l.kind == s.kind && l.params == s.params && l.label == s.label => l.len += s.len,
            _ => out.push(s),
        }
    };
    let mut o = 0usize;
    for s in segs.drain(..) {
        let (s0, s1) = (o, o + s.len as usize);
        o = s1;
        let (x0, x1) = (a.clamp(s0, s1), b.clamp(s0, s1));
        if x0 >= x1 {
            push(s);
            continue;
        }
        // a typed segment (image rows, audio frames) is only cut where its alignment allows:
        // metadata never lies inside one, so anything else is left alone
        if s.kind != crate::nsc3::format::kind::GENERIC && s.kind != crate::nsc3::format::kind::FAST && s.kind != crate::nsc3::format::kind::TEXT {
            push(s);
            continue;
        }
        push(PSeg { len: (x0 - s0) as u32, ..s.clone() });
        push(PSeg { kind: crate::nsc3::format::kind::GENERIC, len: (x1 - x0) as u32, params: vec![], label });
        push(PSeg { len: (s1 - x1) as u32, ..s });
    }
    *segs = out;
}

/// Typed segments of an in-memory buffer as one unbounded block (decompressed stream content).
pub fn segments(data: &[u8]) -> Vec<PSeg> {
    let mut det = Detector::new();
    let mut segs: Vec<PSeg> = Vec::new();
    let mut pos = 0usize;
    while pos < data.len() {
        let avail = data.len() - pos;
        let window = avail.min(LOOKAHEAD);
        let s = det.next(&data[pos..pos + window], pos as u64, avail <= LOOKAHEAD);
        let take = (s.len.max(1)).min(avail as u64) as usize;
        match segs.last_mut() {
            Some(l) if l.kind == s.kind && l.params == s.params && l.label == s.label => l.len += take as u32,
            _ => segs.push(PSeg { kind: s.kind, len: take as u32, params: s.params, label: s.label }),
        }
        pos += take;
    }
    segs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_covers_input_and_respects_alignment() {
        // a PGM image (rows of 100 bytes) followed by text, cut into 1000-byte blocks
        let mut data = b"P5\n100 50\n255\n".to_vec();
        data.extend((0..5000u32).map(|i| (i % 251) as u8));
        data.extend(b"some text after the image.\n".repeat(200));
        let mut p = Planner::new(&data[..], 1000);
        let mut total = 0usize;
        let mut first_image_piece = None;
        while let Some(b) = p.next_block().unwrap() {
            assert_eq!(b.segs.iter().map(|s| s.len as usize).sum::<usize>(), b.data.len());
            for s in &b.segs {
                if s.kind == crate::nsc3::format::kind::IMAGE && first_image_piece.is_none() { first_image_piece = Some(s.len); }
            }
            total += b.data.len();
        }
        assert_eq!(total, data.len());
        assert_eq!(first_image_piece.unwrap() % 100, 0, "image cut off its row alignment");
    }

    /// A gzip member that crosses a block cut extends that block to the member's end (when
    /// extension is on), so the stream can be recompressed whole.
    #[test]
    fn block_extends_over_a_crossing_stream() {
        let text: Vec<u8> = (0..40_000).flat_map(|i| format!("{i} lorem ipsum {}\n", i % 97).into_bytes()).collect();
        let mut gz = vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3];
        gz.extend(crate::transform::deflate::deflate_like_zlib(&text, 6));
        gz.extend_from_slice(&[0; 8]);
        let mut data = vec![b'x'; 200_000];
        let start = data.len();
        data.extend_from_slice(&gz);
        data.extend(vec![b'y'; 100_000]);
        assert!(start + gz.len() > 256 * 1024, "test stream must cross the cut");
        let blocks = |extend: Option<usize>| {
            let mut p = Planner::new(&data[..], 256 * 1024);
            p.extend = extend;
            let mut v = vec![];
            while let Some(b) = p.next_block().unwrap() { v.push((b.data.len(), b.inner_est)); }
            v
        };
        let plain = blocks(None);
        assert_eq!(plain[0].0, 256 * 1024);
        let ext = blocks(Some(4 << 20));
        assert_eq!(ext[0].0, start + gz.len() - 8, "block 0 should end where the deflate stream ends");
        assert!(ext[0].1 >= text.len(), "inner estimate {} < {}", ext[0].1, text.len());
        assert_eq!(ext.iter().map(|b| b.0).sum::<usize>(), data.len());
    }

    #[test]
    fn same_plan_for_any_read_chunking() {
        struct Trickle<'a>(&'a [u8], usize);
        impl Read for Trickle<'_> {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                let n = self.1.min(out.len()).min(self.0.len());
                out[..n].copy_from_slice(&self.0[..n]);
                self.0 = &self.0[n..];
                Ok(n)
            }
        }
        let mut s = 5u64;
        let mut data = b"text ".repeat(100_000);
        data.extend((0..300_000).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s as u8 }));
        let plan = |r: &mut dyn Read| {
            let mut p = Planner::new(r, 256 * 1024);
            let mut v = vec![];
            while let Some(b) = p.next_block().unwrap() { v.push(b.segs); }
            v
        };
        let a = plan(&mut &data[..]);
        let b = plan(&mut Trickle(&data, 777));
        assert_eq!(a, b);
    }
}
