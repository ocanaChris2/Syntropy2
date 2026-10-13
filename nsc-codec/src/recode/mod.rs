//! Format recoders: take an already-compressed stream (an MP3, …) apart down to its entropy-coded
//! symbols, so a dedicated model codes the symbols instead of the bytes, and rebuild the exact
//! original bytes from them on decode.
//!
//! A recoder turns a stream it recognises into a subtree of segments ([`Node`]): leaves coded by a
//! model, regions planned like ordinary content (an ID3 tag carrying a cover JPEG, say), and
//! containers it rebuilds itself. [`find_at`] accepts a stream only after rebuilding it from that
//! subtree — with the decoder's own [`rebuild`] — and comparing, so a file from an unusual encoder
//! is left as bytes, never mis-rebuilt.
//!
//! Not every recoder ends in a model of its own: a container can also sort a format's bytes by
//! what they are and put content back together that the format had cut into pieces, so that it
//! is found and modelled as what it is ([`sqlite`]).
//!
//! Registering a format: its container kinds in [`is_container`], [`validate`] and [`rebuild`];
//! its stream finder in [`may_start`], [`find_at`] and [`extent`]; its leaf models in
//! `cm::models`.
pub mod aac;
pub mod aac_tables;
pub mod avi;
pub mod flac;
pub mod gif;
pub mod h264;
pub mod h264_tables;
pub mod h264_vlc_tables;
pub mod ljpeg;
pub mod mkv;
pub mod mp3;
pub mod mp3_tables;
pub mod mp4;
pub mod psd;
pub mod sqlite;
pub mod tiff;
pub mod ts;
pub mod type1;
pub mod zip;

use crate::nsc3::format::{kind, Segment};
use crate::transform::deflate::inflate::ExtentError;

/// A recoded stream's segment subtree (encoder side), in the order the decoder rebuilds it.
pub enum Node {
    /// Bytes coded by the model of segment kind `kind`.
    Leaf { kind: u8, params: Vec<u8>, bytes: Vec<u8> },
    /// Bytes planned like ordinary content: typed by the detector and scanned for streams.
    Region(Vec<u8>),
    /// `len` bytes rebuilt by the recoder of `kind` from its children.
    Container { kind: u8, len: usize, params: Vec<u8>, kids: Vec<Node> },
    /// A deflate stream with its content and record (an archive's entry).
    Deflate(crate::transform::deflate::scan::Deflate),
}

impl Node {
    /// Bytes this node stands for.
    pub fn len(&self) -> usize {
        match self {
            Node::Leaf { bytes, .. } => bytes.len(),
            Node::Region(b) => b.len(),
            Node::Container { len, .. } => *len,
            Node::Deflate(d) => d.len,
        }
    }

    /// Bytes the subtree adds to the block's work buffer (its leaves and regions).
    pub fn inner_len(&self) -> usize {
        match self {
            Node::Leaf { bytes, .. } => bytes.len(),
            Node::Region(b) => b.len(),
            Node::Container { kids, .. } => kids.iter().map(Node::inner_len).sum(),
            Node::Deflate(d) => d.inner.len() + d.side.len(),
        }
    }
}

/// A recognised stream at `start..start + len` of the scanned buffer and its subtree.
pub struct Recoded {
    pub start: usize,
    pub len: usize,
    pub root: Node,
}

impl Recoded {
    pub fn inner_len(&self) -> usize { self.root.inner_len() }
}

/// Is any format recoder built in? (Files use decoder level 2 only when one may be used.)
pub const ANY: bool = true;

/// Can a recoded stream start with byte `b`? A cheap filter before [`find_at`] and [`extent`].
pub fn may_start(b: u8) -> bool { matches!(b, 0xFF | b'S' | b'G' | b'I' | b'M' | b'%' | b'8') }

/// Rows read as records below which a database is left as bytes, unless rows gathered from its
/// overflow chains hold a stream: seven streams for a few rows cost more than they save.
const SQLITE_MIN_RECORDS: usize = 512;

/// The SQLite database at `b[i]`, whole in `b`, taken apart.
fn sqlite_at(b: &[u8], i: usize, budget: usize, log: &mut Option<&mut Vec<String>>) -> Option<Recoded> {
    let db = sqlite::header(&b[i..])?;
    let len = db.pages.checked_mul(db.page).filter(|&l| l <= b.len() - i && l <= budget)?;
    let sp = sqlite::split(&b[i..i + len], &db)?;
    let rows = &sp.streams[sqlite::ROWS];
    if sp.records < SQLITE_MIN_RECORDS && (rows.is_empty() || crate::transform::deflate::scan::find_hinted(rows, budget, false, true, &[]).is_empty()) {
        if let Some(l) = log.as_mut() { l.push(format!("@{i}: sqlite: {} records, no stream in its long rows: left as bytes", sp.records)); }
        return None;
    }
    let note = format!("{} pages of {} bytes, {} of cells, {} records, {} long rows", db.pages, db.page, sp.leaves, sp.records, sp.long);
    // what is content of its own kind is planned as content; the rest are leaves
    let kids = sp.streams.into_iter().enumerate().map(|(k, bytes)| match k {
        sqlite::TEXT => Node::Leaf { kind: kind::TEXT, params: vec![], bytes },
        sqlite::BLOBS | sqlite::ROWS if !bytes.is_empty() => Node::Region(bytes),
        _ => Node::Leaf { kind: kind::GENERIC, params: vec![], bytes },
    }).collect();
    let root = Node::Container { kind: kind::SQLITE, len, params: db.params(), kids };
    if root.inner_len() > budget { return None; }
    let r = proven(i, &b[i..i + len], root);
    if let Some(l) = log.as_mut() { l.push(format!("@{i}: sqlite {}: {note}", if r.is_some() { "OK" } else { "REBUILD MISMATCH" })); }
    r
}

/// A run of frames as a container: its frame records cut into leaves, then the raw bytes.
fn frames_container(k: u8, leaf: u8, params: Vec<u8>, len: usize, frames: &[u8], cuts: &[usize], rest: &[u8]) -> Node {
    let mut kids = Vec::with_capacity(cuts.len() + 2);
    let mut from = 0;
    for &to in cuts.iter().chain([&frames.len()]) {
        kids.push(Node::Leaf { kind: leaf, params: params.clone(), bytes: frames[from..to].to_vec() });
        from = to;
    }
    kids.push(Node::Leaf { kind: kind::GENERIC, params: vec![], bytes: rest.to_vec() });
    Node::Container { kind: k, len, params, kids }
}

/// The stream starting at `b[i]`, recoded and proven to rebuild exactly, if a recoder recognises it
/// and its leaves fit `budget` work bytes. `log` collects one line per candidate.
pub fn find_at(b: &[u8], i: usize, budget: usize, log: &mut Option<&mut Vec<String>>) -> Option<Recoded> {
    if b[i] == b'S' { return sqlite_at(b, i, budget, log); }
    if b[i] == b'G' {
        // a GIF: its record, and its images as frames of colour indices
        let s = gif::split(&b[i..])?;
        if s.records.len() + s.frames.len() > budget { return None; }
        let note = format!("@{i}: gif {} bytes, {} of {} images decoded to {} bytes, record {} bytes", s.len, s.decoded, s.found, s.frames.len(), s.records.len());
        let kids = vec![Node::Leaf { kind: kind::GENERIC, params: vec![], bytes: s.records }, Node::Leaf { kind: kind::PIX, params: vec![], bytes: s.frames }];
        let r = proven(i, &b[i..i + s.len], Node::Container { kind: kind::GIF, len: s.len, params: vec![], kids });
        if let Some(l) = log.as_mut() { l.push(format!("{note}: {}", if r.is_some() { "OK" } else { "REBUILD MISMATCH" })); }
        return r;
    }
    if b[i] == b'8' {
        // a Photoshop document: its record, the bytes between its planes, each layer's pixels
        let s = psd::split(&b[i..], budget)?;
        let pixels: usize = s.images.iter().map(|im| im.1.len()).sum();
        if s.records.len() + s.gaps.len() + pixels > budget { return None; }
        let (start, len) = (i + s.start, s.len);
        let note = format!("@{i}: psd: {} planes of {} layers, {len} bytes at {start} -> {pixels} of pixels, record {} bytes, {} between", s.planes, s.images.len(), s.records.len(), s.gaps.len());
        let gaps = if s.gaps.len() < 4096 { Node::Leaf { kind: kind::GENERIC, params: vec![], bytes: s.gaps } } else { Node::Region(s.gaps) };
        let mut kids = vec![Node::Leaf { kind: kind::GENERIC, params: vec![], bytes: s.records }, gaps];
        kids.extend(s.images.into_iter().map(|(params, bytes)| Node::Leaf { kind: kind::IMAGE, params, bytes }));
        let r = proven(start, &b[start..start + len], Node::Container { kind: kind::PSD, len, params: vec![], kids });
        if let Some(l) = log.as_mut() { l.push(format!("{note}: {}", if r.is_some() { "OK" } else { "REBUILD MISMATCH" })); }
        return r;
    }
    if b[i] == b'%' {
        // a Type 1 font: its enciphered part in the clear, as content like any other
        let (at, plain) = type1::split(&b[i..], i == 0)?;
        if plain.len() > budget { return None; }
        let (start, len) = (i + at, plain.len());
        let r = proven(start, &b[start..start + len], Node::Container { kind: kind::TYPE1, len, params: vec![], kids: vec![Node::Region(plain)] });
        if let Some(l) = log.as_mut() { l.push(format!("@{i}: type 1 font, {len} enciphered bytes at {start}: {}", if r.is_some() { "OK" } else { "REBUILD MISMATCH" })); }
        return r;
    }
    if b[i] == b'I' || b[i] == b'M' {
        // a TIFF: its record, the bytes between its strips, and each picture's pixels as an image
        let s = tiff::split(&b[i..], budget)?;
        let pixels: usize = s.images.iter().map(|im| im.1.len()).sum();
        if s.records.len() + s.gaps.len() + pixels > budget { return None; }
        let (start, len) = (i + s.start, s.len);
        let note = format!("@{i}: tiff: {} strips of {} pictures, {len} bytes at {start} -> {pixels} of pixels, record {} bytes, {} between the strips", s.chunks, s.images.len(), s.records.len(), s.gaps.len());
        // what lies between the strips is a few bytes as a rule; a directory or a preview picture
        // there is content like any other
        let gaps = if s.gaps.len() < 4096 { Node::Leaf { kind: kind::GENERIC, params: vec![], bytes: s.gaps } } else { Node::Region(s.gaps) };
        // (samples of a size the image model has no layout for are bytes)
        let mut images = s.images.into_iter().map(|(params, bytes)| Node::Leaf { kind: if params.is_empty() { kind::GENERIC } else { kind::IMAGE }, params, bytes });
        let root = if s.plain {
            // pixels as the file has them: an image, and nothing to rebuild
            images.next()?
        } else {
            let mut kids = vec![Node::Leaf { kind: kind::GENERIC, params: vec![], bytes: s.records }, gaps];
            kids.extend(images);
            Node::Container { kind: kind::STRIPS, len, params: vec![], kids }
        };
        let r = proven(start, &b[start..start + len], root);
        if let Some(l) = log.as_mut() { l.push(format!("{note}: {}", if r.is_some() { "OK" } else { "REBUILD MISMATCH" })); }
        return r;
    }
    // FLAC frames (an ADTS stream of MPEG-2 AAC starts with the same bits: tried below): their
    // records, and the samples for the audio model
    if let Some(s) = flac::split(&b[i..]).filter(|_| b[i + 1] & 0xFE == 0xF8) {
        if s.records.len() + s.pcm.len() > budget { return None; }
        let mut params = crate::detect::audio_params(s.channels, 8 * s.width, false, false);
        params.push(crate::cm::audio::setting(5, true, &s.pcm, &params));
        let note = format!("@{i}: flac {} frames, {} bytes -> {} of samples and {} of records", s.frames, s.len, s.pcm.len(), s.records.len());
        let kids = vec![Node::Leaf { kind: kind::GENERIC, params: vec![], bytes: s.records }, Node::Leaf { kind: kind::AUDIO, params, bytes: s.pcm }];
        let r = proven(i, &b[i..i + s.len], Node::Container { kind: kind::FLAC, len: s.len, params: vec![s.channels, s.width], kids });
        if let Some(l) = log.as_mut() { l.push(format!("{note}: {}", if r.is_some() { "OK" } else { "REBUILD MISMATCH" })); }
        return r;
    }
    let (what, len, count, root) = if let Some(s) = mp3::split(&b[i..]) {
        if s.frames.len() + s.rest.len() > budget { return None; }
        ("mp3", s.len, s.count, frames_container(kind::MP3, kind::MP3F, vec![], s.len, &s.frames, &s.cuts, &s.rest))
    } else if let Some(s) = aac::split_adts(&b[i..]) {
        if s.frames.len() + s.rest.len() > budget { return None; }
        ("aac", s.len, s.count, frames_container(kind::AAC, kind::AACF, vec![1, 0], s.len, &s.frames, &s.cuts, &s.rest))
    } else {
        return None;
    };
    let r = proven(i, &b[i..i + len], root);
    if let Some(l) = log.as_mut() {
        l.push(format!("@{i}: {what} {}: {} frames ({} parsed), {len} bytes", if r.is_some() { "OK" } else { "REBUILD MISMATCH" }, count.0, count.1));
    }
    r
}

/// The ZIP archive at `b[i]`, its entries' data taken as what they are and the checksums and
/// sizes its headers repeat derived from them.
pub fn zip_at(b: &[u8], i: usize, budget: usize, log: &mut Option<&mut Vec<String>>) -> Option<Recoded> {
    let s = zip::split(&b[i..], budget, true)?;
    let note = format!("@{i}: zip of {} entries, {} bytes, {} fields derived", s.data.len(), s.len, s.derived);
    let leaf = |bytes: Vec<u8>| Node::Leaf { kind: kind::GENERIC, params: vec![], bytes };
    let mut kids = vec![leaf(s.record), leaf(s.skeleton)];
    kids.extend(s.data.into_iter().map(|d| match d {
        zip::Data::Deflate(d) => Node::Deflate(d),
        zip::Data::Stored(bytes) if !bytes.is_empty() => Node::Region(bytes),
        zip::Data::Stored(bytes) | zip::Data::Raw(bytes) => leaf(bytes),
    }));
    let root = Node::Container { kind: kind::ZIP, len: s.len, params: vec![], kids };
    if root.inner_len() > budget { return None; }
    let r = proven(i, &b[i..i + s.len], root);
    if let Some(l) = log.as_mut() { l.push(format!("{note}: {}", if r.is_some() { "OK" } else { "REBUILD MISMATCH" })); }
    r
}

/// The lossless JPEG at `b[i]` (what a camera's raw file holds its sensor data in), decoded to
/// samples for the image model.
pub fn ljpeg_at(b: &[u8], i: usize, budget: usize, log: &mut Option<&mut Vec<String>>) -> Option<Recoded> {
    let s = ljpeg::split(&b[i..])?;
    if s.samples.len() + s.header.len() > budget { return None; }
    let bytes = 1 + s.wide as usize;
    let fmt = if s.wide { crate::detect::imgfmt::GRAY16LE } else { crate::detect::imgfmt::GRAY8 };
    let params = crate::detect::image_params(fmt, (s.row / s.components) as u32, (s.row * bytes) as u32, s.components as u8, false);
    let note = format!("@{i}: lossless jpeg {} bytes -> {} of samples ({} a row, {} components)", s.len, s.samples.len(), s.row, s.components);
    let kids = vec![Node::Leaf { kind: kind::GENERIC, params: vec![], bytes: s.header }, Node::Leaf { kind: kind::IMAGE, params, bytes: s.samples }];
    let r = proven(i, &b[i..i + s.len], Node::Container { kind: kind::LJPEG, len: s.len, params: vec![], kids });
    if let Some(l) = log.as_mut() { l.push(format!("{note}: {}", if r.is_some() { "OK" } else { "REBUILD MISMATCH" })); }
    r
}

/// Do the first bytes of a file say it is a container [`index`] reads?
pub fn indexable(head: &[u8]) -> bool { mp4::looks_like(head) || mkv::looks_like(head) || avi::looks_like(head) || ts::looks_like(head) || ts::looks_like_bytestream(head) }

/// The index of the container file `f` of `len` bytes (ISO base media or Matroska): where the
/// samples of the tracks a recoder knows are. Leaves the read position unspecified.
pub fn index<F: std::io::Read + std::io::Seek>(f: &mut F, len: u64) -> Option<mp4::Index> {
    let mut head = [0u8; 16];
    f.seek(std::io::SeekFrom::Start(0)).ok()?;
    let got = crate::bench::read_full(f, &mut head).ok()?;
    if mkv::looks_like(&head[..got]) { mkv::index(f, len) } else if avi::looks_like(&head[..got]) { avi::index(f, len) } else if ts::looks_like(&head[..got]) { ts::index(f, len) } else if ts::looks_like_bytestream(&head[..got]) { ts::index_bytestream(f, len) } else { mp4::index(f, len) }
}

/// [`index`] of the file at `path`.
pub fn index_path(path: &std::path::Path) -> Option<mp4::Index> {
    let mut f = std::fs::File::open(path).ok()?;
    let len = f.metadata().ok()?.len();
    index(&mut f, len)
}

/// One track's samples in a block, gathered from its runs: taken apart, if its recoder can.
fn track_node(stream: &[u8], sizes: &[u32], what: mp4::Stream, avc: Option<&mp4::Avc>) -> Option<(Node, String)> {
    match what {
        mp4::Stream::Aac { sf } => {
            let s = aac::split_run(stream, sizes, sf as usize)?;
            let note = format!("aac {} frames ({} parsed)", s.count.0, s.count.1);
            Some((frames_container(kind::AAC, kind::AACF, vec![0, sf], s.len, &s.frames, &s.cuts, &s.rest), note))
        }
        mp4::Stream::Mp3 => {
            // the frames back to back are an MP3 stream; all of it must be one
            let s = mp3::split(stream).filter(|s| s.len == stream.len())?;
            let note = format!("mp3 {} frames ({} parsed)", s.count.0, s.count.1);
            Some((frames_container(kind::MP3, kind::MP3F, vec![], s.len, &s.frames, &s.cuts, &s.rest), note))
        }
        mp4::Stream::AnnexB { .. } => {
            // the units behind lengths for the H.264 recoder, and the start codes' record
            let sp = ts::split(stream)?;
            let s = h264::split(&sp.prefixed, &sp.sizes, 4, &[])?;
            let note = format!("h264 byte stream, {} slices ({} recoded)", s.count.0, s.count.1);
            let inner = frames_container(kind::H264, kind::H264B, vec![4, 0], sp.prefixed.len(), &s.records, &s.cuts, &s.raw);
            let kids = vec![Node::Leaf { kind: kind::GENERIC, params: vec![], bytes: sp.record }, inner];
            Some((Node::Container { kind: kind::ANNEXB, len: stream.len(), params: vec![], kids }, note))
        }
        mp4::Stream::Adts { .. } => {
            // the block may begin inside a frame: from the first frame that starts a chain
            let from = (0..stream.len().min(8192)).find(|&o| stream[o] == 0xFF && stream.get(o + 1).is_some_and(|x| x & 0xF6 == 0xF0) && aac::split_adts(&stream[o..]).is_some())?;
            let s = aac::split_adts(&stream[from..])?;
            let note = format!("aac (adts) {} frames ({} parsed)", s.count.0, s.count.1);
            let leaf = |b: &[u8]| Node::Leaf { kind: kind::GENERIC, params: vec![], bytes: b.to_vec() };
            let kids = vec![leaf(&stream[..from]), frames_container(kind::AAC, kind::AACF, vec![1, 0], s.len, &s.frames, &s.cuts, &s.rest), leaf(&stream[from + s.len..])];
            Some((Node::Container { kind: kind::GROUP, len: stream.len(), params: vec![], kids }, note))
        }
        mp4::Stream::Pcm { channels, bits } => {
            // samples for the audio model, whole frames of them
            if stream.is_empty() || stream.len() % (channels as usize * bits as usize / 8) != 0 { return None; }
            let mut params = crate::detect::audio_params(channels, bits, false, bits == 8);
            params.push(crate::cm::audio::setting(5, true, stream, &params));
            Some((Node::Leaf { kind: kind::AUDIO, params, bytes: stream.to_vec() }, format!("pcm {channels} channels of {bits} bits")))
        }
        mp4::Stream::Avc { .. } => {
            let avc = avc?;
            let config: Vec<Vec<u8>> = avc.sps.iter().chain(&avc.pps).cloned().collect();
            let s = h264::split(stream, sizes, avc.nal_len as usize, &config)?;
            let note = format!("h264 {} slices ({} recoded)", s.count.0, s.count.1);
            Some((frames_container(kind::H264, kind::H264B, vec![avc.nal_len, 0], stream.len(), &s.records, &s.cuts, &s.raw), note))
        }
    }
}

/// The sample runs `hints` of the block `b` (from its containers' indexes; by offset, not
/// overlapping), recoded and proven to rebuild exactly: each track's runs together as one stream,
/// and around them a [`kind::MUX`] container that puts the runs — and the bytes between them —
/// back in file order.
pub fn from_hints(b: &[u8], hints: &[crate::nsc3::plan::Hint], budget: usize, log: &mut Option<&mut Vec<String>>) -> Option<Recoded> {
    // the tracks, in order of appearance, with their runs
    let mut tracks: Vec<(mp4::Stream, Vec<usize>)> = vec![];
    let mut end = 0usize;
    for (i, h) in hints.iter().enumerate() {
        let len: usize = h.sizes.iter().map(|&n| n as usize).sum();
        if h.start < end || len == 0 || h.start.checked_add(len)? > b.len() { return None; }
        end = h.start + len;
        match tracks.iter_mut().find(|t| t.0 == h.what) {
            Some(t) => t.1.push(i),
            None => tracks.push((h.what, vec![i])),
        }
    }
    // take each track apart; `owner[i]`: the stream (child of the container) run i belongs to
    let run_len = |i: usize| hints[i].sizes.iter().map(|&n| n as usize).sum::<usize>();
    let mut owner: Vec<Option<usize>> = vec![None; hints.len()];
    let mut streams: Vec<Node> = vec![];
    let mut bytes: Vec<Vec<u8>> = vec![];
    let mut left = budget;
    for (what, runs) in &tracks {
        let mut stream = Vec::with_capacity(runs.iter().map(|&i| run_len(i)).sum());
        let mut sizes = vec![];
        for &i in runs {
            stream.extend_from_slice(&b[hints[i].start..hints[i].start + run_len(i)]);
            sizes.extend_from_slice(&hints[i].sizes);
        }
        let got = track_node(&stream, &sizes, *what, hints[runs[0]].avc.as_deref()).filter(|(n, _)| n.inner_len() <= left);
        // a track is only as good as its rebuild
        let got = got.filter(|(n, _)| {
            let mut out = Vec::with_capacity(stream.len());
            n.len() == stream.len() && rebuild_node(n, &mut out).is_ok() && out == stream
        });
        if let Some(l) = log.as_mut() {
            l.push(format!("@{}: track of {} runs, {} bytes: {}", hints[runs[0]].start, runs.len(), stream.len(), got.as_ref().map_or("left as bytes", |g| &g.1)));
        }
        if let Some((node, _)) = got {
            left -= node.inner_len();
            for &i in runs { owner[i] = Some(streams.len()); }
            streams.push(node);
            bytes.push(stream);
        }
    }
    // the span from the first recoded run to the last, as (stream, length) pieces; what lies
    // between the runs is one more stream
    let first = owner.iter().position(|o| o.is_some())?;
    let last = owner.iter().rposition(|o| o.is_some())?;
    let (start, end) = (hints[first].start, hints[last].start + run_len(last));
    let gap_stream = streams.len();
    let mut gap = vec![];
    let mut pieces: Vec<(usize, usize)> = vec![];
    let mut push = |stream: usize, len: usize| {
        if len == 0 { return; }
        match pieces.last_mut() {
            Some(p) if p.0 == stream => p.1 += len,
            _ => pieces.push((stream, len)),
        }
    };
    let mut pos = start;
    for i in first..=last {
        let (h, len) = (&hints[i], run_len(i));
        let from = if owner[i].is_some() { h.start } else { h.start + len };
        gap.extend_from_slice(&b[pos..from]);
        push(gap_stream, from - pos);
        if let Some(st) = owner[i] { push(st, len); }
        pos = h.start + len;
    }
    drop(push);
    if pieces.len() == 1 {
        // one track's samples back to back: the stream itself
        return Some(Recoded { start, len: end - start, root: streams.pop()? });
    }
    let mut plan = Vec::with_capacity(pieces.len() * 4);
    for &(st, len) in &pieces {
        crate::nsc3::format::put_var(&mut plan, st as u64);
        crate::nsc3::format::put_var(&mut plan, len as u64);
    }
    if plan.len() + gap.len() > left { return None; }
    // the streams are proven: what is left to prove is the plan
    if !gap.is_empty() { bytes.push(gap.clone()); }
    let mut out = Vec::with_capacity(end - start);
    if unmux(&plan, &bytes, &mut out).is_err() || out != b[start..end] { return None; }
    let mut kids = vec![Node::Leaf { kind: kind::GENERIC, params: vec![], bytes: plan }];
    kids.append(&mut streams);
    if !gap.is_empty() { kids.push(Node::Region(gap)); }
    Some(Recoded { start, len: end - start, root: Node::Container { kind: kind::MUX, len: end - start, params: vec![], kids } })
}

/// Put the pieces of interleaved streams back in order: `plan` is (stream, length) pairs.
fn unmux(plan: &[u8], streams: &[Vec<u8>], out: &mut Vec<u8>) -> Result<(), String> {
    let bad = || "mux: bad plan (corrupt file)".to_string();
    let mut at = vec![0usize; streams.len()];
    let mut p = 0usize;
    while p < plan.len() {
        let st = h264::get_var(plan, &mut p).ok_or_else(bad)?;
        let len = h264::get_var(plan, &mut p).ok_or_else(bad)?;
        let (s, o) = (streams.get(st).ok_or_else(bad)?, at[st]);
        out.extend_from_slice(s.get(o..o.checked_add(len).ok_or_else(bad)?).ok_or_else(bad)?);
        at[st] = o + len;
    }
    if at.iter().zip(streams).any(|(&o, s)| o != s.len()) { return Err(bad()); }
    Ok(())
}

/// The structural extent of the stream starting at `b[i]`, without recoding it: (end offset,
/// estimated work bytes). `Truncated` when `b` ends inside it.
pub fn extent(b: &[u8], i: usize) -> Result<(usize, usize), ExtentError> {
    if b[i] == b'S' {
        // a database is taken apart only whole: its pages, as its header counts them
        let db = sqlite::header(&b[i..]).ok_or(if b.len() - i < 100 && b"SQLite format 3\0".starts_with(&b[i..(i + 16).min(b.len())]) { ExtentError::Truncated } else { ExtentError::Invalid })?;
        let len = db.pages.checked_mul(db.page).ok_or(ExtentError::Invalid)?;
        return if len > b.len() - i { Err(ExtentError::Truncated) } else { Ok((i + len, len)) };
    }
    if b[i] == b'G' { return gif::extent(b, i); }
    if b[i] == b'I' || b[i] == b'M' { return tiff::extent(b, i); }
    if b[i] == b'8' { return psd::extent(b, i); }
    if b[i] == b'%' { return type1::extent(&b[i..], i == 0).map(|(at, len)| (i + at + len, len)).ok_or(ExtentError::Invalid); }
    match flac::extent(b, i) {
        Err(ExtentError::Invalid) => {}
        r => return r,
    }
    match mp3::extent(b, i) {
        Err(ExtentError::Invalid) => aac::extent(b, i),
        r => r,
    }
}

/// Accept `root` for the stream `orig` only if it rebuilds exactly (what the decoder will do).
pub fn proven(start: usize, orig: &[u8], root: Node) -> Option<Recoded> {
    if root.len() != orig.len() { return None; }
    let mut out = Vec::with_capacity(orig.len());
    rebuild_node(&root, &mut out).ok()?;
    (out == orig).then_some(Recoded { start, len: orig.len(), root })
}

/// Encoder side: append the bytes `n` rebuilds to.
pub fn rebuild_node(n: &Node, out: &mut Vec<u8>) -> Result<(), String> {
    match n {
        Node::Leaf { bytes, .. } => out.extend_from_slice(bytes),
        Node::Region(b) => out.extend_from_slice(b),
        Node::Deflate(d) => {
            let parse = crate::transform::deflate::side::decode(&d.inner, &d.side)?;
            out.extend_from_slice(&crate::transform::deflate::writer::write(&parse, &d.inner)?);
        }
        Node::Container { kind, len, params, kids } => {
            let mut kb = Vec::with_capacity(kids.len());
            for k in kids {
                let mut v = Vec::with_capacity(k.len());
                rebuild_node(k, &mut v)?;
                kb.push(v);
            }
            let start = out.len();
            rebuild(*kind, params, &kb, out)?;
            if out.len() - start != *len { return Err(format!("{} container rebuilt to the wrong length", kind::name(*kind))); }
        }
    }
    Ok(())
}

/// Settings a recoder's segments take from the compression level, applied to the finished tree:
/// from level 5 up the H.264 records are coded by the model's full profile (their parameters'
/// second byte, after the child count in the container's); below, by the lean one, which is
/// a little faster and 0.7% larger.
pub fn tune(s: &mut Segment, level: u32) {
    let full = (level >= 5) as u8;
    match s.kind {
        // samples a recoder decoded: the audio model's filters as the level sets them
        kind::AUDIO if s.params.len() == 5 => s.params[4] = crate::cm::audio::at_level(s.params[4], level),
        kind::H264 if s.params.len() == 6 => s.params[5] = full,
        kind::H264B if s.params.len() == 2 => s.params[1] = full,
        _ => {}
    }
}

/// Is `k` a container kind rebuilt here?
pub fn is_container(k: u8) -> bool { matches!(k, kind::GROUP | kind::MP3 | kind::AAC | kind::MUX | kind::H264 | kind::SQLITE | kind::FLAC | kind::GIF | kind::STRIPS | kind::ANNEXB | kind::TYPE1 | kind::LJPEG | kind::PSD | kind::ZIP) }

/// Is `k` a container whose children are told apart by their place? A child of one that is
/// split for the lanes stays one child: a group of its pieces.
pub fn positional(k: u8) -> bool { matches!(k, kind::MP3 | kind::AAC | kind::MUX | kind::H264 | kind::SQLITE | kind::FLAC | kind::GIF | kind::STRIPS | kind::ANNEXB | kind::LJPEG | kind::PSD | kind::ZIP) }

/// Decoder side: does a container of kind `k` (params after the child count) have the children
/// its recoder expects?
pub fn validate(k: u8, params: &[u8], kids: &[&Segment]) -> bool {
    // a leaf, or the group a leaf split for the lanes became
    let plain = |s: &Segment| (!kind::container(s.kind) || s.kind == kind::GROUP) && s.xform == 0;
    match k {
        kind::GROUP => true,
        kind::MP3 | kind::AAC | kind::H264 => {
            // record leaves (with the container's parameters), then one leaf of raw bytes
            let leaf = match k { kind::MP3 => kind::MP3F, kind::AAC => kind::AACF, _ => kind::H264B };
            let Some((rest, frames)) = kids.split_last() else { return false };
            let params_ok = match k {
                kind::MP3 => params.is_empty(),
                kind::AAC => params.len() == 2 && params[0] <= 1 && params[1] < 12,
                _ => params.len() == 2 && matches!(params[0], 1 | 2 | 4) && params[1] <= 1,
            };
            params_ok && !frames.is_empty() && frames.iter().all(|s| s.kind == leaf && plain(s) && s.params == params)
                && plain(rest) && !matches!(rest.kind, kind::MP3F | kind::AACF | kind::H264B)
        }
        // the plan, then the streams
        kind::MUX => params.is_empty() && kids.len() >= 2 && plain(kids[0]) && !matches!(kids[0].kind, kind::MP3F | kind::AACF | kind::H264B),
        // the records, then the samples: an audio leaf of the container's channels and width
        kind::FLAC => {
            params.len() == 2 && kids.len() == 2 && plain(kids[0]) && !matches!(kids[0].kind, kind::MP3F | kind::AACF | kind::H264B)
                && (kids[1].kind == kind::GROUP || kids[1].kind == kind::AUDIO && kids[1].xform == 0 && kids[1].params.len() == 5 && kids[1].params[..2] == [params[0], 8 * params[1]])
        }
        // the record, then the frames
        kind::GIF => params.is_empty() && kids.len() == 2 && plain(kids[0]) && !matches!(kids[0].kind, kind::MP3F | kind::AACF | kind::H264B) && kids[1].kind == kind::PIX && kids[1].xform == 0,
        // the record, the bytes between the strips, then each picture's pixels
        kind::STRIPS | kind::PSD => params.is_empty() && kids.len() >= 3 && kids.iter().all(|s| plain(s)) && !matches!(kids[0].kind, kind::MP3F | kind::AACF | kind::H264B),
        kind::TYPE1 => params.is_empty() && kids.len() == 1 && plain(kids[0]),
        // the record, the skeleton, then the entries' data (streams, content, bytes)
        kind::ZIP => params.is_empty() && kids.len() >= 3 && plain(kids[0]) && plain(kids[1]) && !kids[..2].iter().any(|s| matches!(s.kind, kind::MP3F | kind::AACF | kind::H264B)),
        kind::LJPEG => params.is_empty() && kids.len() == 2 && plain(kids[0]) && plain(kids[1]) && !matches!(kids[0].kind, kind::MP3F | kind::AACF | kind::H264B),
        // the start codes' record, then the stream behind lengths
        kind::ANNEXB => params.is_empty() && kids.len() == 2 && plain(kids[0]) && !matches!(kids[0].kind, kind::MP3F | kind::AACF | kind::H264B),
        // the seven streams, the structure first
        kind::SQLITE => sqlite::Db::from_params(params).is_some() && kids.len() == sqlite::STREAMS && plain(kids[0]) && !matches!(kids[0].kind, kind::MP3F | kind::AACF | kind::H264B),
        _ => false,
    }
}

/// Decoder side: append the original bytes of a container of kind `k` (params after the child
/// count), given the rebuilt bytes of its children.
pub fn rebuild(k: u8, params: &[u8], kids: &[Vec<u8>], out: &mut Vec<u8>) -> Result<(), String> {
    match k {
        kind::GROUP => {
            for kb in kids { out.extend_from_slice(kb); }
            Ok(())
        }
        kind::MP3 | kind::AAC | kind::H264 if kids.len() >= 2 => {
            let (rest, frames) = kids.split_last().unwrap();
            let all;
            let frames: &[u8] = if frames.len() == 1 { &frames[0] } else { all = frames.concat(); &all };
            match k {
                kind::MP3 => mp3::rebuild(frames, rest, out),
                kind::AAC => aac::rebuild(frames, rest, params.first() == Some(&1), params.get(1).copied().unwrap_or(0) as usize, out),
                _ => h264::rebuild(frames, rest, params.first().copied().unwrap_or(0) as usize, out),
            }
        }
        kind::MUX if kids.len() >= 2 => unmux(&kids[0], &kids[1..], out),
        kind::GIF if kids.len() == 2 => gif::rebuild(&kids[0], &kids[1], out),
        kind::TYPE1 if kids.len() == 1 => { type1::rebuild(&kids[0], out); Ok(()) }
        kind::LJPEG if kids.len() == 2 => ljpeg::rebuild(&kids[0], &kids[1], out),
        kind::ANNEXB if kids.len() == 2 => ts::rebuild(&kids[0], &kids[1], out),
        kind::ZIP if kids.len() >= 3 => zip::rebuild(&kids[0], &kids[1], &kids[2..], out),
        kind::PSD if kids.len() >= 3 => psd::rebuild(&kids[0], &kids[1], &kids[2..], out),
        kind::STRIPS if kids.len() >= 3 => tiff::rebuild(&kids[0], &kids[1], &kids[2..], out),
        kind::FLAC if kids.len() == 2 && params.len() == 2 => flac::rebuild(&kids[0], &kids[1], params[0] as usize, params[1] as usize, out),
        kind::SQLITE if kids.len() == sqlite::STREAMS => {
            let (page, usable) = sqlite::Db::from_params(params).ok_or("sqlite: bad page size")?;
            sqlite::rebuild(page, usable, &std::array::from_fn(|i| &kids[i][..]), out)
        }
        _ => Err(format!("unsupported container kind {k}")),
    }
}
