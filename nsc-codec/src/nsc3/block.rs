//! Encode and decode one independent NSC3 block.
//!
//! The encoder picks a model kind and codes the block, then keeps the result only if it is smaller
//! than the raw bytes and (when verifying) decodes back to exactly the input. Otherwise the block
//! is stored raw. So a block can never expand by more than its header, and a model bug can never
//! produce a file that fails to decode.
use super::format::{self, kind, BlockHeader, Segment, MODE_CM, MODE_STORED};
use crate::cm::coder::{Decoder, Encoder};
use crate::cm::levels;
use crate::cm::models::{self, Models};
use crate::cm::predictor::{code_bytes, Predictor, ReadBuf, WriteBuf};
use crate::crc::crc32;
use crate::transform;

pub struct EncOpts {
    pub level: u32,
    pub verify: bool,
    /// When set, text segments are coded by this neural model instead of the CM text model.
    pub neural: Option<std::sync::Arc<NeuralCtx>>,
    /// Target bytes per lane: a block whose work buffer (its bytes after recompressed streams
    /// are expanded) is much larger is coded as several independent lanes.
    pub lane: usize,
    /// Most lanes of one block coded at once (never changes the output).
    pub lane_jobs: usize,
    /// Spare threads for parallel work inside a block (neural windows), shared by all blocks in
    /// flight (never changes the output).
    pub threads: std::sync::Arc<Threads>,
    /// Write only what nsc 0.3 (decoder level 1) reads: no format recoders.
    pub compat: bool,
}

impl EncOpts {
    pub fn new(level: u32, verify: bool, neural: Option<std::sync::Arc<NeuralCtx>>) -> Self {
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        EncOpts { level, verify, neural, lane: levels::block_size(level), lane_jobs: 1, threads: Threads::new(cores), compat: false }
    }

    /// The same options with at most `lane_jobs` lanes coded at once.
    pub fn with_lane_jobs(&self, lane_jobs: usize) -> Self {
        EncOpts { neural: self.neural.clone(), threads: self.threads.clone(), lane_jobs, ..*self }
    }

    /// Are the format recoders (MP3, …) used? From level 3 up, unless writing for nsc 0.3.
    pub fn specialists(&self) -> bool { specialists(self.level, self.compat) }

    /// How lanes are coded: (verified concurrently, model sets per lane, lanes at once). A lane's
    /// concurrent verifier needs a second model, so it runs when `lane_jobs` (the model sets that
    /// fit in memory, at most the thread count) allows two, and halves the lanes coded at once.
    pub fn lane_plan(&self) -> (bool, usize, usize) {
        let pipelined = self.verify && self.lane_jobs >= 2;
        let sets = if pipelined { 2 } else { 1 };
        (pipelined, sets, (self.lane_jobs.max(1) / sets).max(1))
    }
}

/// Are the format recoders used at `level`? (`compat`: writing for decoder level 1.)
pub fn specialists(level: u32, compat: bool) -> bool { crate::recode::ANY && !compat && level >= 3 }

/// A pool of thread slots shared by the blocks being coded. A block worker holds one slot while it
/// runs; work inside a block takes whatever slots are spare. So one block alone can use every core,
/// and many blocks in flight use one each, without oversubscribing the machine.
pub struct Threads(std::sync::atomic::AtomicIsize);

impl Threads {
    pub fn new(n: usize) -> std::sync::Arc<Threads> { std::sync::Arc::new(Threads(std::sync::atomic::AtomicIsize::new(n as isize))) }

    /// Hold `n` slots (a block worker; may overdraw) until the guard drops.
    pub fn hold(&self, n: usize) -> Slots<'_> {
        self.0.fetch_sub(n as isize, std::sync::atomic::Ordering::AcqRel);
        Slots(self, n)
    }

    /// Take up to `want` spare slots (possibly none) until the guard drops.
    pub fn take(&self, want: usize) -> Slots<'_> {
        use std::sync::atomic::Ordering;
        let mut cur = self.0.load(Ordering::Acquire);
        loop {
            let k = (want as isize).min(cur).max(0);
            if k == 0 { return Slots(self, 0); }
            match self.0.compare_exchange_weak(cur, cur - k, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return Slots(self, k as usize),
                Err(now) => cur = now,
            }
        }
    }
}

/// Slots taken from [`Threads`], returned on drop.
pub struct Slots<'a>(&'a Threads, usize);

impl Slots<'_> {
    pub fn count(&self) -> usize { self.1 }
}

impl Drop for Slots<'_> {
    fn drop(&mut self) { self.0.0.fetch_add(self.1 as isize, std::sync::atomic::Ordering::AcqRel); }
}

/// A loaded neural model bound to a file (header binding `binding`), shared read-only by workers.
pub struct NeuralCtx {
    pub id: String,
    pub fingerprint: u64,
    pub binding: u8,
    pub model: crate::artifacts::Model,
    pub o2: crate::features::O2Table,
    pub tok: crate::tokenizer::Tokenizer,
}

/// Text shorter than this stays with the CM text model (neural restarts cost a few bytes).
const NEURAL_MIN: usize = 2048;

fn neural_params(binding: u8, n_tok: u32, first: u32, n_bits: u32, len: u32) -> Vec<u8> {
    let mut p = vec![binding];
    for v in [n_tok, first, n_bits, len] { p.extend_from_slice(&v.to_le_bytes()); }
    p
}

fn parse_neural_params(p: &[u8]) -> Result<(u8, u32, u32, u32, u32), String> {
    if p.len() != 17 { return Err("bad neural segment parameters".into()); }
    let r = |o: usize| u32::from_le_bytes(p[o..o + 4].try_into().unwrap());
    Ok((p[0], r(1), r(5), r(9), r(13)))
}

/// Code one text segment with the neural model: native tokenizer, then the portable-math codec,
/// its windows evaluated on this thread plus any spare `threads`. None if the tokenizer is not an
/// exact bijection on these bytes (the caller keeps CM text).
fn neural_encode(nc: &NeuralCtx, bytes: &[u8], threads: &Threads) -> Option<(Vec<u8>, Vec<u8>)> {
    let toks = nc.tok.encode(bytes);
    if toks.len() < 2 || toks.iter().any(|&t| t >= nc.model.cfg.vocab_size) { return None; }
    if nc.tok.decode(&toks).ok()? != bytes { return None; }
    let extra = threads.take(toks.len().div_ceil(nc.model.cfg.seg_len) - 1);
    let (payload, nbits) = crate::codec::encode_tokens_m::<crate::forward::Pure>(&nc.model, &nc.o2, &toks, 1 + extra.count());
    Some((neural_params(nc.binding, toks.len() as u32, toks[0] as u32, nbits as u32, payload.len() as u32), payload))
}

pub struct Encoded {
    pub bytes: Vec<u8>,
    pub raw_len: usize,
    pub mode: u8,
    /// Set of segment kinds used (`kind::bit`).
    pub kinds: u128,
    pub mem: usize,
    pub verify_failed: bool,
}

/// log2(x)·65536 for x ≥ 1, integer-only (squaring method).
pub fn log2_fx(x: u64) -> u64 {
    debug_assert!(x > 0);
    let lz = 63 - x.leading_zeros() as u64;
    let mut y: u64 = if lz >= 32 { x >> (lz - 32) } else { x << (32 - lz) };
    let mut frac = 0u64;
    for i in (0..16).rev() {
        y = ((y as u128 * y as u128) >> 32) as u64;
        if y >= 2u64 << 32 {
            y >>= 1;
            frac |= 1 << i;
        }
    }
    (lz << 16) | frac
}

/// Order-0 entropy in thousandths of a bit per byte.
pub fn order0_mbits(data: &[u8]) -> u32 {
    if data.is_empty() { return 0; }
    let mut cnt = [0u64; 256];
    for &b in data { cnt[b as usize] += 1; }
    let n = data.len() as u64;
    let ln = log2_fx(n);
    let mut sum: u128 = 0;
    for &c in &cnt {
        if c > 0 { sum += c as u128 * (ln - log2_fx(c)) as u128; }
    }
    (sum * 1000 / 65536 / n as u128) as u32
}

/// Content-defined anchors (gear hash): keys of 32-byte windows at data-dependent positions, about
/// one per 256 bytes. Identical content yields identical anchors wherever it sits in the block.
fn anchors(data: &[u8]) -> Vec<(usize, u64)> {
    const fn gear() -> [u64; 256] {
        let mut t = [0u64; 256];
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut i = 0;
        while i < 256 {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            t[i] = x;
            i += 1;
        }
        t
    }
    static GEAR: [u64; 256] = gear();
    let mut out = vec![];
    let mut h = 0u64;
    for (i, &b) in data.iter().enumerate() {
        h = (h << 1).wrapping_add(GEAR[b as usize]);
        if h >> 56 == 0 && i + 32 <= data.len() {
            let mut k = 0xcbf2_9ce4_8422_2325u64;
            for &x in &data[i..i + 32] { k = (k ^ x as u64).wrapping_mul(0x100_0000_01b3); }
            out.push((i, k));
        }
    }
    out
}

/// Formats whose payload is already strongly compressed (modern codecs/archives): modelling them
/// gains essentially nothing, so the body of a large segment is stored raw unless its content
/// repeats or a probe finds it compressible after all.
const MEDIA: &[&str] = &["mp4", "matroska", "mp3", "ogg", "flac", "7z", "xz", "zstd", "bzip2", "webp"];
/// High-entropy segments at least this long get a quick compressibility probe; shorter ones are
/// simply modelled (their headers, tags and indexes are worth a few percent).
const PROBE_MIN: usize = 256 << 10;
const PROBE_LEN: usize = 64 << 10;

/// Does a small independent model fail to shrink `sample` by at least 0.5%? (encoder-side only)
fn incompressible(sample: &[u8]) -> bool {
    let cfg = levels::preset(1, levels::Kind::Generic, sample.len());
    let mut pred = Predictor::new(&cfg);
    pred.set_kind(kind::FAST);
    let mut enc = Encoder::new();
    code_bytes(&mut pred, &mut enc, &mut ReadBuf(sample), 0..sample.len());
    enc.finish().len() * 1000 >= sample.len() * 995
}

/// The planned segments with their tiers decided. High-entropy data is still modelled by default:
/// small gzip members, PDF streams and JPEG scans keep a few percent of exploitable structure. It
/// is stored raw only when it is long, does not repeat earlier content, and a quick probe finds
/// it incompressible — so gigabytes of video or ciphertext stay fast. Known media is probed in
/// its middle and loses only its body: the first bytes (container headers, tags) stay modelled
/// when they compress.
fn decide_tiers(raw: &[u8], segs: &[super::plan::PSeg]) -> Vec<super::plan::PSeg> {
    let mut out: Vec<super::plan::PSeg> = Vec::with_capacity(segs.len() + 1);
    if !segs.iter().any(|s| s.kind == kind::FAST) { return segs.to_vec(); }
    let anch = anchors(raw);
    let mut seen = std::collections::HashSet::new();
    let (mut off, mut ai) = (0usize, 0usize);
    for s in segs {
        let end = off + s.len as usize;
        let (mut total, mut dup) = (0usize, 0usize);
        while ai < anch.len() && anch[ai].0 < end {
            if s.kind == kind::FAST {
                total += 1;
                if seen.contains(&anch[ai].1) { dup += 1; }
            }
            seen.insert(anch[ai].1);
            ai += 1;
        }
        let repeats = total >= 4 && dup * 4 >= total;
        let len = end - off;
        let stored = |len: usize| super::plan::PSeg { kind: kind::STORED, len: len as u32, ..s.clone() };
        // Labelled deflate/JPEG streams (gzip, zip, png, pdf, jpeg…) keep being modelled: the
        // full model still finds a little structure in them.
        if s.kind != kind::FAST || repeats || len < PROBE_MIN {
            out.push(s.clone());
        } else if MEDIA.contains(&s.label) {
            let mid = off + len / 2 / 4096 * 4096;
            if !incompressible(&raw[mid..mid + PROBE_LEN]) {
                out.push(s.clone());
            } else if incompressible(&raw[off..off + PROBE_LEN]) {
                out.push(stored(len));
            } else {
                out.push(super::plan::PSeg { len: PROBE_LEN as u32, ..s.clone() });
                out.push(stored(len - PROBE_LEN));
            }
        } else if s.label == "high-entropy" && incompressible(&raw[off..off + PROBE_LEN]) {
            out.push(stored(len));
        } else {
            out.push(s.clone());
        }
        off = end;
    }
    out
}

// ------------------------------------------------------------------ the segment tree
//
// A block's segments form a tree, stored in pre-order. Leaves are coded by the model (or stored
// raw, or coded by the neural model) and, concatenated in table order, form the block's work
// buffer. Containers rebuild their bytes from their children: a deflate stream from its
// decompressed content plus its reconstruction record, PNG chunk framing around a zlib stream,
// PNG scanlines from filter types plus pixels. Container params start with the child count.

/// Deterministic cap on the decompressed bytes of the recompressed streams in one block.
pub const INNER_BUDGET: usize = 256 << 20;
/// Streams inside streams (a docx in a zip, a tar in a gzip...) are found down to this depth.
const MAX_SCAN_DEPTH: usize = 3;
/// Decoder bounds: tree depth, work buffer, and bytes rebuilt by containers.
const MAX_TREE_DEPTH: usize = 8;
const MAX_WORK: usize = 1 << 30;
const MAX_REBUILD: usize = 2 << 30;

use crate::transform::deflate::scan::{self, Found};
use crate::transform::png;

fn u32_at(p: &[u8], o: usize) -> Result<usize, String> {
    Ok(u32::from_le_bytes(p.get(o..o + 4).ok_or("bad container parameters")?.try_into().unwrap()) as usize)
}

/// IDAT chunk sizes as params: all chunks `size` bytes but a shorter (non-empty) last one, or an
/// explicit list.
fn chunk_layout(sizes: &[u32]) -> Vec<u8> {
    let (first, total) = (sizes[0], sizes.iter().map(|&s| s as u64).sum::<u64>());
    let n = sizes.len() as u64;
    let uniform = first > 0 && total.div_ceil(first as u64) == n && sizes[..sizes.len() - 1].iter().all(|&s| s == first);
    let mut p = vec![];
    if uniform {
        p.push(0);
        p.extend_from_slice(&first.to_le_bytes());
        p.extend_from_slice(&(total as u32).to_le_bytes());
    } else {
        p.push(1);
        p.extend_from_slice(&(sizes.len() as u32).to_le_bytes());
        for s in sizes { p.extend_from_slice(&s.to_le_bytes()); }
    }
    p
}

fn chunk_sizes(p: &[u8]) -> Result<Vec<usize>, String> {
    match p.first() {
        Some(0) => {
            let (size, total) = (u32_at(p, 1)?, u32_at(p, 5)?);
            if size == 0 || p.len() != 9 || total.div_ceil(size) > MAX_SEGMENTS_HINT { return Err("bad png chunk layout".into()); }
            let mut v = vec![size; total / size];
            if total % size != 0 { v.push(total % size); }
            Ok(v)
        }
        Some(1) => {
            let n = u32_at(p, 1)?;
            if p.len() != 5 + 4 * n { return Err("bad png chunk layout".into()); }
            (0..n).map(|i| u32_at(p, 5 + 4 * i)).collect()
        }
        _ => Err("bad png chunk layout".into()),
    }
}
const MAX_SEGMENTS_HINT: usize = 1 << 24;

fn rows_params(g: &png::Geom) -> Vec<u8> {
    let mut p = (g.rowbytes as u32).to_le_bytes().to_vec();
    p.push(g.bpp as u8);
    p.extend_from_slice(&(g.rows as u32).to_le_bytes());
    p
}

fn parse_rows(p: &[u8]) -> Result<png::Geom, String> {
    if p.len() != 9 { return Err("bad png rows parameters".into()); }
    let g = png::Geom { rowbytes: u32_at(p, 0)?, bpp: p[4] as usize, rows: u32_at(p, 5)? };
    if g.rowbytes == 0 || g.rows == 0 || !(1..=8).contains(&g.bpp) { return Err("bad png rows parameters".into()); }
    Ok(g)
}

/// A planned run of bytes with one model setting.
struct Piece {
    kind: u8,
    len: usize,
    params: Vec<u8>,
    xform: u8,
}

/// Model settings for planned segments: tier decisions, branch transforms, and adjacent segments
/// with equal settings merged (a transform must see the same span in encoder and decoder).
fn pieces(bytes: &[u8], segs: &[super::plan::PSeg], level: u32, level2: bool) -> Vec<Piece> {
    let segs = decide_tiers(bytes, segs);
    let mut out: Vec<Piece> = Vec::with_capacity(segs.len());
    let mut at = 0usize;
    for s in &segs {
        let k = s.kind;
        let xform = if k == kind::EXE { transform::bcj::for_arch(s.params.first().copied().unwrap_or(0), level2) } else { transform::bcj::NONE };
        let mut params = s.params.clone();
        if k == kind::AUDIO { params.push(crate::cm::audio::setting(level, level2, &bytes[at..at + s.len as usize], &s.params)); }
        match out.last_mut() {
            Some(l) if l.kind == k && l.params == params && l.xform == xform => l.len += s.len as usize,
            _ => out.push(Piece { kind: k, len: s.len as usize, params, xform }),
        }
        at += s.len as usize;
    }
    out
}

/// The segment tree under construction (pre-order) and the work buffer its leaves form.
struct Tree {
    segs: Vec<Segment>,
    work: Vec<u8>,
    budget: usize,
    /// Prove each stream's rebuild while scanning (when the block is not decoded back anyway).
    verify_streams: bool,
    level: u32,
    /// Format recoders enabled.
    recode: bool,
}

impl Tree {
    /// Append a leaf (none for empty bytes); returns the number of nodes added.
    fn leaf(&mut self, k: u8, params: Vec<u8>, xform: u8, bytes: &[u8]) -> u32 {
        if bytes.is_empty() { return 0; }
        let st = self.work.len();
        self.work.extend_from_slice(bytes);
        transform::bcj::forward(xform, &mut self.work[st..]);
        self.segs.push(Segment { kind: k, ver: 1, xform, raw_len: bytes.len() as u32, params });
        1
    }

    /// Append a leaf even when empty (a recoder's children are positional).
    fn leaf_always(&mut self, k: u8, params: Vec<u8>, bytes: &[u8]) -> u32 {
        self.work.extend_from_slice(bytes);
        self.segs.push(Segment { kind: k, ver: 1, xform: 0, raw_len: bytes.len() as u32, params });
        1
    }

    fn open(&mut self, k: u8, raw_len: usize) -> usize {
        self.segs.push(Segment { kind: k, ver: 1, xform: 0, raw_len: raw_len as u32, params: vec![] });
        self.segs.len() - 1
    }

    fn close(&mut self, at: usize, children: u32, extra: &[u8]) {
        let mut p = children.to_le_bytes().to_vec();
        p.extend_from_slice(extra);
        self.segs[at].params = p;
    }

    /// Emit `bytes` as the planned pieces with the recompressible streams `found` (ordered,
    /// non-overlapping) spliced in. Returns the number of nodes added at this level.
    fn region(&mut self, bytes: &[u8], pieces: Vec<Piece>, found: Vec<Found>, depth: usize) -> u32 {
        let mut n = 0;
        let mut found = found.into_iter().peekable();
        let (mut pos, mut covered) = (0usize, 0usize);
        for p in pieces {
            let end = pos + p.len;
            let mut a = pos.max(covered);
            while a < end {
                if found.peek().is_some_and(|f| f.start() < end) {
                    let f = found.next().unwrap();
                    debug_assert!(f.start() >= a);
                    n += self.leaf(p.kind, p.params.clone(), p.xform, &bytes[a..f.start()]);
                    a = f.start() + f.len();
                    covered = a;
                    n += self.stream(f, depth);
                } else {
                    n += self.leaf(p.kind, p.params.clone(), p.xform, &bytes[a..end]);
                    a = end;
                }
            }
            pos = end;
        }
        n
    }

    fn stream(&mut self, f: Found, depth: usize) -> u32 {
        match f {
            Found::Deflate(d) => self.deflate(d, None, depth),
            Found::Jpeg(j) => {
                let sp = j.split;
                let at = self.open(kind::JPEG, sp.len);
                let mut n = self.leaf(kind::GENERIC, vec![], 0, &sp.header);
                n += self.leaf(kind::JPEGC, crate::cm::jpeg::params(&sp.frame, &sp.quant), 0, &sp.coefs);
                n += self.leaf(kind::GENERIC, vec![], 0, &sp.pads);
                self.close(at, n, &[]);
                1
            }
            Found::Recode(r) => self.node(r.root, depth),
            Found::Png(p) => {
                let at = self.open(kind::PNG, p.len);
                let rows = png::Geom::new(p.info.width, p.info.height, p.info.depth, p.info.color, p.info.interlace)
                    .map(|g| (g, png::image_params(p.info.width, p.info.depth, p.info.color, &g)));
                let mut n = self.leaf(kind::GENERIC, vec![], 0, &p.zhead);
                n += self.deflate(p.body, rows, depth);
                n += self.leaf(kind::GENERIC, vec![], 0, &p.tail);
                self.close(at, n, &chunk_layout(&p.sizes));
                1
            }
        }
    }

    /// Emit a recoded stream's subtree; returns the number of nodes added at this level (1).
    fn node(&mut self, n: crate::recode::Node, depth: usize) -> u32 {
        use crate::recode::Node;
        match n {
            Node::Leaf { kind: k, params, bytes } => self.leaf_always(k, params, &bytes),
            Node::Deflate(d) => self.deflate(d, None, depth),
            Node::Region(bytes) => {
                let at = self.open(kind::GROUP, bytes.len());
                let pieces = pieces(&bytes, &super::plan::segments(&bytes), self.level, self.recode);
                let found = if depth + 1 < MAX_SCAN_DEPTH { scan::find(&bytes, self.budget, self.verify_streams, self.recode) } else { vec![] };
                self.budget -= found.iter().map(|f| f.inner_len()).sum::<usize>();
                let n = self.region(&bytes, pieces, found, depth + 1);
                self.close(at, n, &[]);
                1
            }
            Node::Container { kind: k, len, params, kids } => {
                let at = self.open(k, len);
                let mut n = 0;
                for kid in kids { n += self.node(kid, depth); }
                self.close(at, n, &params);
                1
            }
        }
    }

    fn deflate(&mut self, d: scan::Deflate, rows: Option<(png::Geom, Vec<u8>)>, depth: usize) -> u32 {
        let at = self.open(kind::DEFLATE, d.len);
        let mut n = 0;
        match rows.and_then(|(g, ip)| png::unfilter(&d.inner, &g).map(|tp| (g, ip, tp))) {
            Some((g, ip, (types, pix))) => {
                let r = self.open(kind::ROWS, d.inner.len());
                self.leaf(kind::GENERIC, vec![], 0, &types);
                self.leaf(kind::IMAGE, ip, 0, &pix);
                self.close(r, 2, &rows_params(&g));
                n += 1;
            }
            None => {
                let pieces = pieces(&d.inner, &super::plan::segments(&d.inner), self.level, self.recode);
                let found = if depth + 1 < MAX_SCAN_DEPTH { scan::find(&d.inner, self.budget, self.verify_streams, self.recode) } else { vec![] };
                self.budget -= found.iter().map(|f| f.inner_len()).sum::<usize>();
                n += self.region(&d.inner, pieces, found, depth + 1);
            }
        }
        debug_assert!(!d.side.is_empty());
        n += self.leaf(kind::GENERIC, vec![], 0, &d.side);
        self.close(at, n, &[]);
        1
    }
}

/// What a first coding of a block showed to cost more than it saves, as byte ranges of the block:
/// streams better left as the bytes they are, and bytes better stored than modelled.
#[derive(Default, Debug, PartialEq)]
struct Lost {
    left: Vec<(usize, usize)>,
    stored: Vec<(usize, usize)>,
}

impl Lost {
    fn is_empty(&self) -> bool { self.left.is_empty() && self.stored.is_empty() }
}

/// The planned segments `segs` with the byte ranges `ranges` (in order, apart) stored.
fn store_ranges(segs: &[super::plan::PSeg], ranges: &[(usize, usize)]) -> Vec<super::plan::PSeg> {
    let mut out: Vec<super::plan::PSeg> = Vec::with_capacity(segs.len() + 2 * ranges.len());
    let mut push = |s: super::plan::PSeg| if s.len > 0 { out.push(s) };
    let (mut o, mut r) = (0usize, 0usize);
    for s in segs {
        let (mut at, end) = (o, o + s.len as usize);
        while at < end {
            while r < ranges.len() && ranges[r].1 <= at { r += 1; }
            // up to the next range as it is; inside one, stored
            let (a, e) = ranges.get(r).map_or((end, end), |&(a, e)| (a.clamp(at, end), e.clamp(at, end)));
            push(super::plan::PSeg { len: (a - at) as u32, ..s.clone() });
            push(super::plan::PSeg { kind: kind::STORED, len: (e - a) as u32, params: vec![], label: s.label });
            at = e.max(a);
        }
        o = end;
    }
    out
}

/// Build a block's segment tree and work buffer. `verify_streams`: see [`Tree`]; `recode`: format
/// recoders enabled. `lost`: what is left as bytes, or stored, this time.
fn build(pb: &super::plan::PlannedBlock, level: u32, verify_streams: bool, recode: bool, lost: &Lost) -> Tree {
    let raw = &pb.data[..];
    let mut found = if level >= 2 { scan::find_hinted(raw, INNER_BUDGET, verify_streams, recode, &pb.hints) } else { vec![] };
    found.retain(|f| !lost.left.iter().any(|&(a, e)| f.start() < e && a < f.start() + f.len()));
    let budget = INNER_BUDGET - found.iter().map(|f| f.inner_len()).sum::<usize>();
    let mut t = Tree { segs: Vec::with_capacity(pb.segs.len()), work: Vec::with_capacity(raw.len()), budget, verify_streams, level, recode };
    // (a stream that is not worth taking apart is compressed data: stored, like what codes
    // longer than it is)
    let stored;
    let segs = if lost.is_empty() { &pb.segs } else {
        let mut ranges: Vec<(usize, usize)> = lost.left.iter().chain(&lost.stored).copied().collect();
        ranges.sort_unstable();
        stored = store_ranges(&pb.segs, &ranges);
        &stored
    };
    t.region(raw, pieces(raw, segs, level, recode), found, 0);
    if recode {
        for s in &mut t.segs { crate::recode::tune(s, level); }
    }
    t
}

/// A parsed tree node (decoder side): its segment index, work offset (leaves) and children.
struct Node {
    seg: usize,
    woff: usize,
    kids: Vec<Node>,
}

fn is_plain_leaf(s: &Segment) -> bool { !kind::container(s.kind) && s.xform == 0 }

/// Parse the subtree at `segs[*i]`, checking its structure; leaves get consecutive work offsets.
fn parse_node(segs: &[Segment], i: &mut usize, depth: usize, woff: &mut usize, rebuilt: &mut usize) -> Result<Node, String> {
    let idx = *i;
    let s = segs.get(idx).ok_or("segment tree truncated (corrupt file)")?;
    *i += 1;
    if !kind::known(s.kind) { return Err(format!("unsupported segment kind {} in this build", s.kind)); }
    if s.ver != 1 { return Err(format!("unsupported version {} of segment kind {} (needs a newer nsc)", s.ver, kind::name(s.kind))); }
    if !kind::container(s.kind) {
        if s.xform > transform::bcj::X86D || (s.xform != 0 && s.kind != kind::EXE) {
            return Err(format!("unsupported segment transform {}", s.xform));
        }
        let w = *woff;
        *woff += s.raw_len as usize;
        if *woff > MAX_WORK { return Err("block work buffer exceeds the decoder limit (corrupt file)".into()); }
        return Ok(Node { seg: idx, woff: w, kids: vec![] });
    }
    if depth >= MAX_TREE_DEPTH || s.xform != 0 { return Err("bad segment tree (corrupt file)".into()); }
    let n = u32_at(&s.params, 0)?;
    if n > segs.len() - *i { return Err("segment tree truncated (corrupt file)".into()); }
    let mut kids = Vec::with_capacity(n);
    for _ in 0..n { kids.push(parse_node(segs, i, depth + 1, woff, rebuilt)?); }
    let inner: usize = kids.iter().map(|k| segs[k.seg].raw_len as usize).sum();
    *rebuilt += inner + s.raw_len as usize;
    if *rebuilt > MAX_REBUILD { return Err("block rebuild size exceeds the decoder limit (corrupt file)".into()); }
    let ok = match s.kind {
        kind::DEFLATE => s.params.len() == 4 && kids.last().is_some_and(|k| is_plain_leaf(&segs[k.seg])),
        kind::PNG => {
            let sizes = chunk_sizes(&s.params[4..])?;
            sizes.iter().sum::<usize>() == inner && sizes.iter().map(|&x| 12 + x).sum::<usize>() == s.raw_len as usize
        }
        kind::JPEG => {
            // header, coefficients, and the encoder record when present
            (n == 2 || n == 3) && s.params.len() == 4 && is_plain_leaf(&segs[kids[0].seg]) && segs[kids[1].seg].kind == kind::JPEGC
                && kids.get(2).is_none_or(|k| is_plain_leaf(&segs[k.seg]))
        }
        kind::ROWS => {
            // filter types, then the pixels (one leaf, or several when split for lanes)
            let g = parse_rows(&s.params[4..])?;
            n >= 2 && kids.iter().all(|k| is_plain_leaf(&segs[k.seg]))
                && segs[kids[0].seg].raw_len as usize == g.rows
                && (inner - g.rows) as u64 == g.rows as u64 * g.rowbytes as u64
                && s.raw_len as u64 == g.rows as u64 * (g.rowbytes as u64 + 1)
        }
        k if crate::recode::is_container(k) => {
            let ks: Vec<&Segment> = kids.iter().map(|k| &segs[k.seg]).collect();
            crate::recode::validate(k, &s.params[4..], &ks)
        }
        _ => false,
    };
    if !ok { return Err(format!("bad {} segment (corrupt file)", kind::name(s.kind))); }
    Ok(Node { seg: idx, woff: 0, kids })
}

/// Parse a block's segment tree, checking its structure: the roots, and the work buffer size its
/// leaves need.
fn parse_tree(segs: &[Segment]) -> Result<(Vec<Node>, usize), String> {
    let (mut i, mut total_work, mut rebuilt) = (0usize, 0usize, 0usize);
    let mut roots = vec![];
    while i < segs.len() { roots.push(parse_node(segs, &mut i, 0, &mut total_work, &mut rebuilt)?); }
    Ok((roots, total_work))
}

/// The block's bytes rebuilt from its parsed tree and work buffer.
fn rebuild_all(roots: &[Node], segs: &[Segment], work: &[u8], raw_len: usize) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(raw_len);
    for r in roots { rebuild(r, segs, work, &mut out)?; }
    Ok(out)
}

/// Append the original bytes of `n` to `out`.
fn rebuild(n: &Node, segs: &[Segment], work: &[u8], out: &mut Vec<u8>) -> Result<(), String> {
    let s = &segs[n.seg];
    let start = out.len();
    let leaf_bytes = |k: &Node| { let l = segs[k.seg].raw_len as usize; &work[k.woff..k.woff + l] };
    match s.kind {
        kind::DEFLATE => {
            let (side, kids) = n.kids.split_last().unwrap();
            let mut inner = Vec::new();
            for k in kids { rebuild(k, segs, work, &mut inner)?; }
            let parse = transform::deflate::side::decode(&inner, leaf_bytes(side))?;
            out.extend_from_slice(&transform::deflate::writer::write(&parse, &inner)?);
        }
        kind::PNG => {
            let mut z = Vec::new();
            for k in &n.kids { rebuild(k, segs, work, &mut z)?; }
            let mut o = 0;
            for size in chunk_sizes(&s.params[4..])? {
                out.extend_from_slice(&(size as u32).to_be_bytes());
                let c0 = out.len();
                out.extend_from_slice(b"IDAT");
                out.extend_from_slice(&z[o..o + size]);
                let c = crc32(&out[c0..]);
                out.extend_from_slice(&c.to_be_bytes());
                o += size;
            }
        }
        kind::JPEG => {
            let side = n.kids.get(2).map(leaf_bytes).unwrap_or(&[]);
            out.extend_from_slice(&transform::jpeg::rebuild(leaf_bytes(&n.kids[0]), leaf_bytes(&n.kids[1]), side)?);
        }
        kind::ROWS => {
            // pixel leaves are consecutive in the work buffer
            let g = parse_rows(&s.params[4..])?;
            let p0 = n.kids[1].woff;
            png::refilter(leaf_bytes(&n.kids[0]), &work[p0..p0 + g.rows * g.rowbytes], &g, out)?;
        }
        k if crate::recode::is_container(k) => {
            let mut kb = Vec::with_capacity(n.kids.len());
            for kid in &n.kids {
                let mut v = Vec::with_capacity(segs[kid.seg].raw_len as usize);
                rebuild(kid, segs, work, &mut v)?;
                kb.push(v);
            }
            crate::recode::rebuild(k, &s.params[4..], &kb, out)?;
        }
        _ => {
            out.extend_from_slice(leaf_bytes(n));
            transform::bcj::inverse(s.xform, &mut out[start..]);
        }
    }
    if out.len() - start != s.raw_len as usize {
        return Err(format!("{} segment rebuilt to the wrong length (corrupt file)", kind::name(s.kind)));
    }
    Ok(())
}

/// Estimated memory to decode a block: model tables, work buffer, rebuilt bytes and the
/// temporaries of containers.
pub fn decode_mem(bh: &BlockHeader) -> usize {
    let (mut leaves, mut cont) = (0usize, 0usize);
    for s in &bh.segments {
        if kind::container(s.kind) { cont = cont.saturating_add(s.raw_len as usize); } else { leaves = leaves.saturating_add(s.raw_len as usize); }
    }
    let tables = bh.cfg.map(|c| c.memory()).unwrap_or(0) + models::dedicated_memory(|k| bh.segments.iter().any(|s| s.kind == k));
    tables.saturating_add(bh.raw_len as usize).saturating_add(leaves.min(MAX_WORK)).saturating_add(cont.min(MAX_REBUILD))
}

fn stored_block(raw: &[u8], raw_crc: u32, verify_failed: bool, mem: usize) -> Encoded {
    let raw_len = raw.len() as u32;
    let bh = BlockHeader {
        raw_len, comp_len: raw_len, raw_crc, mode: MODE_STORED, cfg: None, lanes: vec![],
        segments: vec![Segment { kind: kind::STORED, ver: 1, xform: 0, raw_len, params: vec![] }],
        packed: false,
    };
    let mut bytes = format::write_block_header(&bh);
    bytes.extend_from_slice(raw);
    Encoded { bytes, raw_len: raw.len(), mode: MODE_STORED, kinds: kind::bit(kind::STORED), mem, verify_failed }
}

/// For each top-level node of a pre-order table, call `f(top index, segment index)`.
fn top_of(segs: &[Segment], mut f: impl FnMut(usize, usize)) {
    let (mut stack, mut top) = (Vec::<usize>::new(), 0usize);
    for (i, s) in segs.iter().enumerate() {
        if stack.is_empty() { top = i; }
        if let Some(r) = stack.last_mut() { *r -= 1; }
        if kind::container(s.kind) { stack.push(u32_at(&s.params, 0).unwrap_or(0)); }
        f(top, i);
        while stack.last() == Some(&0) { stack.pop(); }
    }
}

/// Compressed bytes attributed to each top-level segment (kind, raw length, cost; approximate
/// within a few bytes: the coder's output position is sampled at leaf boundaries). Used by
/// `nsc analyze --exact`.
pub fn segment_costs(pb: &super::plan::PlannedBlock, level: u32) -> Vec<(u8, u32, u64)> {
    let t = build(pb, level, false, specialists(level, false), &Lost::default());
    let cfg = levels::preset(level, levels::Kind::Generic, t.work.len().max(1));
    let mut models = Models::new(&cfg);
    let mut enc = Encoder::new();
    let mut cost = vec![0u64; t.segs.len()];
    let mut woff = 0usize;
    for (i, s) in t.segs.iter().enumerate() {
        if kind::container(s.kind) { continue; }
        let len = s.raw_len as usize;
        cost[i] = if s.kind == kind::STORED { len as u64 } else {
            let before = enc.len();
            let _ = models.code(&mut enc, &mut ReadBuf(&t.work), s, woff);
            (enc.len() - before) as u64
        };
        woff += len;
    }
    let mut out: Vec<(u8, u32, u64)> = vec![];
    let mut last = usize::MAX;
    top_of(&t.segs, |top, i| {
        if top != last { out.push((t.segs[top].kind, t.segs[top].raw_len, 0)); last = top; }
        out.last_mut().unwrap().2 += cost[i];
    });
    out
}

/// Split byte alignment of a leaf: a lane cut inside it must keep image rows and audio frames whole.
fn leaf_align(s: &Segment) -> usize {
    match s.kind {
        kind::IMAGE if s.params.len() >= 9 => u32::from_le_bytes(s.params[5..9].try_into().unwrap()).max(1) as usize,
        kind::AUDIO if s.params.len() >= 4 => s.params[3].max(1) as usize,
        _ => 1,
    }
}

/// Split leaves longer than `max` into consecutive leaves (same settings; aligned cuts) so lanes
/// can be balanced, fixing the parents' child counts. Leaves with a branch transform stay whole
/// (the transform spans the leaf). A recoder's container, whose children are told apart by
/// their place, keeps its count: the pieces of a child go into a group in that place.
fn split_leaves(segs: Vec<Segment>, max: usize) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::with_capacity(segs.len());
    let mut stack: Vec<(usize, usize)> = vec![]; // (container index in `out`, children left)
    for s in segs {
        if let Some(top) = stack.last_mut() { top.1 -= 1; }
        let parent = stack.last().map(|t| t.0);
        if kind::container(s.kind) {
            let n = u32_at(&s.params, 0).unwrap_or(0);
            out.push(s);
            stack.push((out.len() - 1, n));
        } else {
            let (len, a) = (s.raw_len as usize, leaf_align(&s));
            if len > max && s.xform == 0 && models::splittable(s.kind) && a <= max / 2 {
                let piece = max / a * a;
                let k = len.div_ceil(piece);
                match parent {
                    Some(p) if crate::recode::positional(out[p].kind) => {
                        out.push(Segment { kind: kind::GROUP, ver: 1, xform: 0, raw_len: s.raw_len, params: (k as u32).to_le_bytes().to_vec() });
                    }
                    Some(p) => {
                        let c = u32_at(&out[p].params, 0).unwrap() + k - 1;
                        out[p].params[..4].copy_from_slice(&(c as u32).to_le_bytes());
                    }
                    None => {}
                }
                for i in 0..k {
                    out.push(Segment { raw_len: piece.min(len - i * piece) as u32, ..s.clone() });
                }
            } else {
                out.push(s);
            }
        }
        while stack.last().is_some_and(|t| t.1 == 0) { stack.pop(); }
    }
    out
}

/// Lanes for a block's leaves: one, unless the work buffer is well over `lane` bytes; then leaves
/// are split and grouped into lanes of at least `lane` bytes. Returns the (possibly split) table
/// and the number of leaves per lane.
fn plan_lanes(segs: Vec<Segment>, work_len: usize, lane: usize) -> (Vec<Segment>, Vec<usize>) {
    let lane = lane.max(1 << 20);
    // a leaf counts for more than its bytes when its model is slow and needs little history
    let weight = |s: &Segment| s.raw_len as usize * models::lane_weight(s.kind);
    let heavy: usize = segs.iter().filter(|s| !kind::container(s.kind)).map(|s| weight(s) - s.raw_len as usize).sum();
    if work_len + heavy <= lane + lane / 2 {
        let n = segs.iter().filter(|s| !kind::container(s.kind)).count();
        return (segs, vec![n]);
    }
    let segs = split_leaves(segs, lane / 2);
    let mut lanes: Vec<usize> = vec![];
    let (mut n, mut bytes) = (0usize, 0usize);
    for s in segs.iter().filter(|s| !kind::container(s.kind)) {
        n += 1;
        bytes += weight(s);
        if bytes >= lane { lanes.push(n); n = 0; bytes = 0; }
    }
    if n > 0 {
        // a short tail joins the previous lane
        match lanes.last_mut() { Some(l) if bytes < lane / 4 => *l += n, _ => lanes.push(n) }
    }
    (segs, lanes)
}

/// Run `f` on each lane index with at most `jobs` threads (inline for a single lane); results in
/// lane order.
fn run_lanes<T: Send>(n: usize, jobs: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    if n <= 1 || jobs <= 1 { return (0..n).map(f).collect(); }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out: std::sync::Mutex<Vec<Option<T>>> = std::sync::Mutex::new((0..n).map(|_| None).collect());
    std::thread::scope(|sc| {
        for _ in 0..jobs.min(n) {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= n { break; }
                let r = f(i);
                out.lock().unwrap()[i] = Some(r);
            });
        }
    });
    out.into_inner().unwrap().into_iter().map(|r| r.unwrap()).collect()
}

/// Encode a planned block: each lane's leaves go through one predictor (switched per leaf kind)
/// into that lane's coded stream; stored leaves are appended raw after the coded streams, neural
/// payloads after those.
///
/// With verification on and room for a second model per lane (`lane_jobs` ≥ 2), every lane is
/// decoded *while* it is being coded: a verifier thread runs the decoder's own lane loop
/// ([`decode_lane`]) on the coded bytes as they leave the encoder. The final check then decodes the
/// block with those lane results in place of decoding the lanes again ([`Known`]), so verification
/// costs little more wall time than the coding itself.
pub fn encode(pb: &super::plan::PlannedBlock, opts: &EncOpts) -> Encoded {
    match code_block(pb, opts) {
        Step::Done(e) => e,
        Step::Coded(c) => {
            if opts.verify && !verify_coded(&c, opts) { return stored_block(c.raw, c.bh.raw_crc, true, c.mem); }
            let mut bytes = format::write_block_header(&c.bh);
            bytes.extend_from_slice(&c.payload);
            Encoded { bytes, raw_len: c.raw.len(), mode: MODE_CM, kinds: c.kinds, mem: c.mem, verify_failed: false }
        }
    }
}

/// A block coded but not yet verified or serialised.
struct Coded<'a> {
    raw: &'a [u8],
    work: Vec<u8>,
    bh: BlockHeader,
    payload: Vec<u8>,
    /// The concurrent verifier's lanes (empty when the lanes were not verified concurrently).
    proofs: Vec<LaneProof>,
    /// Rebuilding the tree from `work` was shown (concurrently) to give `raw`.
    rebuilt_ok: bool,
    pipelined: bool,
    kinds: u128,
    mem: usize,
}

enum Step<'a> {
    Done(Encoded),
    Coded(Coded<'a>),
}

/// What of a coded block came to more than the bytes it stands for, as byte ranges of the
/// block: `cost` is each leaf's coded size. A stream taken apart is a promise that its parts
/// code smaller than it; where the promise fails — a record that says too much, content the
/// models make nothing of — the stream is better left alone. And bytes that code longer than
/// they are (the model's guesses cost a few percent on what is noise to it) are better stored.
fn losers(segments: &[Segment], leaves: &[(usize, usize, usize)], cost: &[u32]) -> Lost {
    let mut leaf_of = vec![usize::MAX; segments.len()];
    for (li, l) in leaves.iter().enumerate() { leaf_of[l.0] = li; }
    // per top-level node: what its leaves came to, and its segments (each a few bytes of header)
    let mut sum = vec![(0u64, 0u64); segments.len()];
    top_of(segments, |top, i| {
        sum[top].1 += 1;
        if leaf_of[i] != usize::MAX { sum[top].0 += cost[leaf_of[i]] as u64; }
    });
    let mut out = Lost::default();
    let mut at = 0usize;
    for (i, s) in segments.iter().enumerate() {
        if sum[i].1 == 0 { continue; }
        let range = (at, at + s.raw_len as usize);
        if kind::container(s.kind) {
            if sum[i].0 + 3 * sum[i].1 > s.raw_len as u64 { out.left.push(range); }
        } else if !matches!(s.kind, kind::STORED | kind::NEURAL) && sum[i].0 > s.raw_len as u64 + 16 {
            // (a piece of what was cut for the lanes joins the piece before it)
            match out.stored.last_mut() {
                Some(l) if l.1 == at => l.1 = range.1,
                _ => out.stored.push(range),
            }
        }
        at += s.raw_len as usize;
    }
    out
}

/// The coding half of [`encode`].
fn code_block<'a>(pb: &'a super::plan::PlannedBlock, opts: &EncOpts) -> Step<'a> {
    match code_block_leaving(pb, opts, &Lost::default()) {
        Ok(step) => step,
        Err(lost) => {
            if std::env::var_os("NSC_TRACE").is_some() {
                eprintln!("block of {} bytes: coding again, with {} stream(s) left as bytes {:?} and {} range(s) stored {:?}", pb.data.len(),
                    lost.left.len(), &lost.left[..lost.left.len().min(6)], lost.stored.len(), &lost.stored[..lost.stored.len().min(6)]);
            }
            code_block_leaving(pb, opts, &lost).unwrap_or_else(|_| Step::Done(stored_block(&pb.data, crc32(&pb.data), false, 0)))
        }
    }
}

/// [`code_block`] with what `lost` names left as bytes or stored. Err: what a first coding
/// shows should be (never when `lost` names something, or for a decoder of level 1: a writer
/// for it does as nsc 0.3 did).
fn code_block_leaving<'a>(pb: &'a super::plan::PlannedBlock, opts: &EncOpts, left: &Lost) -> Result<Step<'a>, Lost> {
    let raw = &pb.data[..];
    let raw_crc = crc32(raw);
    if raw.is_empty() { return Ok(Step::Done(stored_block(raw, raw_crc, false, 0))); }
    // with block verification on, the decode-back below proves every stream rebuild
    let t = { let _t = crate::prof::span(|| format!("block {} B: build tree", raw.len())); build(pb, opts.level, !opts.verify, opts.specialists(), left) };
    if t.segs.iter().all(|s| s.kind == kind::STORED) { return Ok(Step::Done(stored_block(raw, raw_crc, false, 0))); }
    let work = t.work;
    let (mut segments, lane_leaves) = plan_lanes(t.segs, work.len(), opts.lane);
    // leaves with their work offsets (the third field, payload offset, is unused when coding), and
    // each lane's leaf range and work range
    let leaves: Vec<(usize, usize, usize)> = {
        let mut w = 0;
        segments.iter().enumerate().filter(|(_, s)| !kind::container(s.kind))
            .map(|(i, s)| { let o = w; w += s.raw_len as usize; (i, o, 0) }).collect()
    };
    let mut lanes = Vec::with_capacity(lane_leaves.len());
    let mut l0 = 0;
    for &n in &lane_leaves {
        let w0 = leaves[l0].1;
        let w1 = leaves.get(l0 + n).map(|l| l.1).unwrap_or(work.len());
        lanes.push((l0..l0 + n, w0..w1));
        l0 += n;
    }
    // Neural leaves first: they carry no coded bits, and the lane coders and their verifiers must
    // know which leaves they are.
    let mut neural_payload: Vec<Option<Vec<u8>>> = vec![None; leaves.len()];
    if let Some(nc) = opts.neural.as_ref() {
        let _t = crate::prof::span(|| format!("block {} B: neural", raw.len()));
        for (li, &(si, woff, _)) in leaves.iter().enumerate() {
            let len = segments[si].raw_len as usize;
            if segments[si].kind != kind::TEXT || len < NEURAL_MIN { continue; }
            if let Some((np, payload)) = neural_encode(nc, &work[woff..woff + len], &opts.threads) {
                segments[si].kind = kind::NEURAL;
                segments[si].params = np;
                neural_payload[li] = Some(payload);
            }
        }
    }
    // The byte predictor's tables are sized by the largest lane. With the format recoders, only
    // by what the predictor itself models there: frame and slice records have models of their own
    // (a decoder of level 1 sized by everything, and files written for it still do).
    let lane_max = lanes.iter().map(|l| {
        if !opts.specialists() { return l.1.len(); }
        leaves[l.0.clone()].iter().filter(|x| models::bytewise(segments[x.0].kind)).map(|x| segments[x.0].raw_len as usize).sum()
    }).max().unwrap_or(1);
    let mut cfg = levels::preset(opts.level, levels::Kind::Generic, lane_max.max(1));
    // x86 code gets the instruction model's contexts (blocks without it do not pay for them)
    if opts.specialists() && segments.iter().any(|s| s.kind == kind::EXE && matches!(s.params.first().map(|a| a & !crate::detect::EXE_RELOCATABLE), Some(1 | 2))) {
        cfg.ctx_mask |= levels::x86_mask(opts.level);
    }
    // binary data that is not code gets the second binary model
    if opts.specialists() && segments.iter().any(|s| matches!(s.kind, kind::GENERIC | kind::FAST)) { cfg.ctx_mask |= levels::bin2_mask(opts.level); }
    // text that is numbers to a fair part (logs, tables) gets the number model
    if opts.specialists() {
        let (mut digits, mut bytes) = (0usize, 0usize);
        for &(si, woff, _) in leaves.iter().filter(|l| segments[l.0].kind == kind::TEXT) {
            let len = segments[si].raw_len as usize;
            digits += crate::cm::num::digits(&work[woff..woff + len]);
            bytes += len;
        }
        if crate::cm::num::wanted(digits, bytes) { cfg.ctx_mask |= levels::num_mask(opts.level); }
    }
    let (pipelined, _, conc) = opts.lane_plan();
    let jobs = conc.min(lanes.len());
    // recorded in the trailer as the decoder's memory hint: one lane's model and the buffers
    // (never how many lanes run at once, so the output does not depend on --jobs)
    let mem = cfg.memory() + models::dedicated_memory(|k| segments.iter().any(|s| s.kind == k)) + 2 * (work.len() + raw.len());
    let t_code = crate::prof::span(|| format!("block {} B: code {} lane(s), work {} B{}", raw.len(), lanes.len(), work.len(),
                                              if pipelined { " + concurrent verify" } else { "" }));
    let segs = &segments;
    let (coded, rebuilt_ok) = std::thread::scope(|sc| {
        // The segment tree is final now: while the lanes are coded, check that rebuilding it from
        // the work buffer gives the block's bytes (the containers' part of the decode-back).
        let check = pipelined.then(|| sc.spawn(|| {
            let _t = crate::prof::span(|| format!("block {} B: rebuild check (concurrent)", raw.len()));
            parse_tree(segs).and_then(|(roots, _)| rebuild_all(&roots, segs, &work, raw.len())).is_ok_and(|o| o == raw)
        }));
        let coded = run_lanes(lanes.len(), jobs, |li| code_lane(&work, segs, &leaves, lanes[li].clone(), &cfg, pipelined));
        (coded, check.is_some_and(|h| h.join().unwrap_or(false)))
    });
    drop(t_code);
    let coded: Vec<(Vec<u8>, Option<LaneProof>, Vec<u32>)> = match coded.into_iter().collect() {
        Ok(v) => v,
        Err(_) => return Ok(Step::Done(stored_block(raw, raw_crc, true, mem))),
    };
    let (mut ac, mut stored, mut neural) = (Vec::new(), Vec::new(), Vec::new());
    let mut lane_table = Vec::with_capacity(lanes.len());
    let mut proofs = Vec::with_capacity(lanes.len());
    let mut cost: Vec<u32> = Vec::with_capacity(leaves.len());
    for ((lr, _), (lac, proof, lane_cost)) in lanes.iter().zip(coded) {
        lane_table.push((lr.len() as u32, lac.len() as u32));
        ac.extend_from_slice(&lac);
        proofs.extend(proof);
        cost.extend(lane_cost);
    }
    for (li, &(si, woff, _)) in leaves.iter().enumerate() {
        match segments[si].kind {
            kind::STORED => {
                stored.extend_from_slice(&work[woff..woff + segments[si].raw_len as usize]);
                cost[li] = segments[si].raw_len;
            }
            kind::NEURAL => {
                neural.extend_from_slice(neural_payload[li].as_deref().unwrap_or_default());
                cost[li] = neural_payload[li].as_ref().map_or(0, |p| p.len() as u32);
            }
            _ => {}
        }
    }
    // developer diagnostics: what each leaf of a megabyte and more came to
    if std::env::var("NSC_TRACE").is_ok_and(|v| v == "2") {
        for (li, &(si, _, _)) in leaves.iter().enumerate() {
            if segments[si].raw_len >= 1 << 20 { eprintln!("leaf {li}: {} of {} bytes -> {}", kind::name(segments[si].kind), segments[si].raw_len, cost[li]); }
        }
    }
    // what cost more than it saved: the caller codes the block again without
    if left.is_empty() && opts.specialists() {
        let l = losers(&segments, &leaves, &cost);
        if !l.is_empty() { return Err(l); }
    }
    let used = segments.iter().fold(0u128, |u, s| u | kind::bit(s.kind));
    if ac.len() + stored.len() + neural.len() >= raw.len() { return Ok(Step::Done(stored_block(raw, raw_crc, false, mem))); }
    let bh = BlockHeader {
        raw_len: raw.len() as u32, comp_len: (ac.len() + stored.len() + neural.len()) as u32, raw_crc, mode: MODE_CM, cfg: Some(cfg),
        lanes: lane_table, segments, packed: opts.specialists(),
    };
    let mut payload = ac;
    payload.extend_from_slice(&stored);
    payload.extend_from_slice(&neural);
    Ok(Step::Coded(Coded { raw, work, bh, payload, proofs, rebuilt_ok, pipelined, kinds: used, mem }))
}

/// The verification half of [`encode`]: does the coded block decode to exactly its bytes? With
/// the concurrent verifier's results, only what it could not check is left to do here.
fn verify_coded(c: &Coded, opts: &EncOpts) -> bool {
    let _t = crate::prof::span(|| format!("block {} B: verify", c.raw.len()));
    let lanes = (c.proofs.len() == c.bh.lanes.len()).then_some(&c.proofs[..]);
    // a verifier that did not finish proves nothing: the block fails
    if c.pipelined && lanes.is_none() { return false; }
    let known = Known { work: &c.work, threads: &opts.threads, lanes, rebuilt: c.rebuilt_ok.then_some(c.raw) };
    decode_inner(&c.bh, &c.payload, opts.neural.as_deref(), opts.lane_jobs, Some(&known)).is_ok_and(|d| d == c.raw)
}

/// Code one lane's arithmetic-coded leaves into its stream. With `verify`, a verifier thread
/// decodes the lane at the same time, from the coded bytes as they leave the encoder.
fn code_lane(work: &[u8], segs: &[Segment], leaves: &[(usize, usize, usize)], (lr, wr): (std::ops::Range<usize>, std::ops::Range<usize>),
             cfg: &levels::ModelCfg, verify: bool) -> Result<(Vec<u8>, Option<LaneProof>, Vec<u32>), String> {
    std::thread::scope(|sc| {
        let (tx, verifier) = if verify {
            let (tx, rx) = std::sync::mpsc::channel();
            let (lr, wr) = (lr.clone(), wr.clone());
            let h = sc.spawn(move || {
                let mut dec = crate::cm::coder::DecoderOf::with_source(crate::cm::coder::Stream::new(rx));
                let mut bytes = vec![0u8; wr.len()];
                // stored and neural leaves get their original bytes (later context for the model);
                // the final check proves that the decoder rebuilds them identically
                let ok = decode_lane(&mut dec, cfg, segs, &leaves[lr.clone()], wr.start, &mut bytes, |_, woff, _, out| {
                    out.copy_from_slice(&work[woff..woff + out.len()]);
                    Ok(())
                }).is_ok();
                LaneProof { ac: dec.into_source().into_bytes(), leaves: lr, work: wr, cfg: *cfg, bytes, ok }
            });
            (Some(tx), Some(h))
        } else {
            (None, None)
        };
        let mut enc = tx.map_or_else(Encoder::new, Encoder::tapped);
        let mut cost = Vec::with_capacity(lr.len());
        let r = encode_lane(&mut enc, cfg, segs, &leaves[lr], wr.start, &work[wr], &mut cost);
        // finishing the encoder closes the verifier's stream
        let lac = enc.finish();
        let proof = verifier.and_then(|h| h.join().ok());
        r.map(|()| (lac, proof, cost))
    })
}

/// The encoder's side of [`decode_lane`]: code the arithmetic-coded leaves of one lane (whose work
/// slice `lw` starts at work offset `w0`); stored and neural leaves carry no coded bits. `cost`
/// gets the coded bytes of each leaf (to a byte or two: the coder's length where the leaf ends).
fn encode_lane(enc: &mut Encoder, cfg: &levels::ModelCfg, segs: &[Segment], leaves: &[(usize, usize, usize)], w0: usize, lw: &[u8], cost: &mut Vec<u32>) -> Result<(), String> {
    let mut models = Models::new(cfg);
    for &(si, woff, _) in leaves {
        let s = &segs[si];
        let before = enc.len();
        if !matches!(s.kind, kind::STORED | kind::NEURAL) { models.code(enc, &mut ReadBuf(lw), s, woff - w0)?; }
        cost.push((enc.len() - before) as u32);
    }
    Ok(())
}

/// One lane decoded by the encoder's concurrent verifier: the coded bytes it read, the leaf and
/// work ranges and model it decoded them with, and the lane's work bytes as decoded.
pub struct LaneProof {
    ac: Vec<u8>,
    leaves: std::ops::Range<usize>,
    work: std::ops::Range<usize>,
    cfg: levels::ModelCfg,
    bytes: Vec<u8>,
    ok: bool,
}

/// Decode the arithmetic-coded leaves of one lane (CM, JPEG coefficients, audio) from `dec` into
/// `lw`, the lane's slice of the work buffer (which starts at work offset `w0`). Stored and neural
/// leaves carry no coded bits: `other(segment, work offset, payload offset, bytes)` fills them,
/// in order, since later leaves are modelled on them. The decoder and the encoder's concurrent
/// verification both run exactly this.
fn decode_lane<C: crate::cm::coder::Coder>(dec: &mut C, cfg: &levels::ModelCfg, segs: &[Segment], leaves: &[(usize, usize, usize)], w0: usize,
                                          lw: &mut [u8], mut other: impl FnMut(&Segment, usize, usize, &mut [u8]) -> Result<(), String>) -> Result<(), String> {
    let mut models = Models::new(cfg);
    for &(si, woff, src) in leaves {
        let s = &segs[si];
        let (rel, len) = (woff - w0, s.raw_len as usize);
        match s.kind {
            kind::STORED | kind::NEURAL => other(s, woff, src, &mut lw[rel..rel + len])?,
            _ => models.code(dec, &mut WriteBuf(lw), s, rel)?,
        }
    }
    Ok(())
}

/// A neural leaf's parameters, checked against the bound model and its payload and output sizes:
/// (token count, first token, coded bits).
fn neural_leaf_params(nc: &NeuralCtx, params: &[u8], payload: &[u8], len: usize) -> Result<(usize, usize, usize), String> {
    let (b, n_tok, first, n_bits, plen) = parse_neural_params(params)?;
    let (n_tok, n_bits) = (n_tok as usize, n_bits as usize);
    if b != nc.binding { return Err("neural segment refers to an unknown model binding".into()); }
    if n_tok < 2 || n_tok > len || first as usize >= nc.model.cfg.vocab_size || plen as usize != payload.len() || n_bits > payload.len() * 8 {
        return Err("neural segment parameters out of range (corrupt file)".into());
    }
    Ok((n_tok, first as usize, n_bits))
}

/// Decode one neural leaf's payload to exactly `len` bytes.
fn neural_leaf(nc: &NeuralCtx, params: &[u8], payload: &[u8], len: usize) -> Result<Vec<u8>, String> {
    let (n_tok, first, n_bits) = neural_leaf_params(nc, params, payload, len)?;
    let toks = crate::codec::decode_tokens_m::<crate::forward::Pure>(&nc.model, &nc.o2, first, n_tok, payload.to_vec(), n_bits);
    let bytes = nc.tok.decode(&toks)?;
    if bytes.len() != len { return Err("neural segment decoded to the wrong length (corrupt file)".into()); }
    Ok(bytes)
}

/// Prove that [`neural_leaf`] decodes `payload` to exactly `orig`: the same parameter checks, then
/// the tokens it must produce (the tokenizer is deterministic) are checked against the payload
/// with the model's windows evaluated on this thread plus any spare `threads`
/// ([`crate::codec::verify_tokens_m`]) instead of decoded one token at a time.
fn verify_neural_leaf(nc: &NeuralCtx, params: &[u8], payload: &[u8], orig: &[u8], threads: &Threads) -> Result<(), String> {
    let (n_tok, first, n_bits) = neural_leaf_params(nc, params, payload, orig.len())?;
    let toks = nc.tok.encode(orig);
    if toks.len() != n_tok || toks[0] != first { return Err("neural segment parameters do not match its text".into()); }
    let extra = threads.take(n_tok.div_ceil(nc.model.cfg.seg_len) - 1);
    if !crate::codec::verify_tokens_m::<crate::forward::Pure>(&nc.model, &nc.o2, &toks, payload, n_bits, 1 + extra.count()) {
        return Err("neural segment does not decode back to its text".into());
    }
    if nc.tok.decode(&toks)? != orig { return Err("neural segment tokens do not decode to its text".into()); }
    Ok(())
}

/// What the encoder's decode-back knows in advance: the block's original work buffer — with it,
/// neural leaves are proven by [`verify_neural_leaf`] (parallel) instead of decoded sequentially —
/// and possibly the lanes its concurrent verifier decoded.
pub struct Known<'a> {
    pub work: &'a [u8],
    pub threads: &'a Threads,
    /// Lanes already decoded by the concurrent verifier (all of them), used instead of decoding
    /// them again once they are shown to match this block exactly.
    pub lanes: Option<&'a [LaneProof]>,
    /// The block's bytes, when rebuilding them from this block's segment tree and `work` is
    /// already known to give exactly them (checked while the lanes were coded).
    pub rebuilt: Option<&'a [u8]>,
}

/// Decode a block whose header was already parsed; checks the block CRC. `neural` is the model
/// bound by the file header (needed only for neural segments); up to `lane_jobs` lanes decode at once.
pub fn decode(bh: &BlockHeader, payload: &[u8], neural: Option<&NeuralCtx>, lane_jobs: usize) -> Result<Vec<u8>, String> {
    decode_inner(bh, payload, neural, lane_jobs, None)
}

/// [`decode`], optionally as the encoder's verification pass ([`Known`]).
fn decode_inner(bh: &BlockHeader, payload: &[u8], neural: Option<&NeuralCtx>, lane_jobs: usize, known: Option<&Known>) -> Result<Vec<u8>, String> {
    if payload.len() != bh.comp_len as usize { return Err("block payload truncated".into()); }
    let out = match bh.mode {
        MODE_STORED => payload.to_vec(),
        MODE_CM => {
            let cfg = bh.cfg.ok_or("CM block without model config")?;
            let segs = &bh.segments;
            let (roots, total_work) = parse_tree(segs)?;
            if roots.iter().map(|r| segs[r.seg].raw_len as u64).sum::<u64>() != bh.raw_len as u64 {
                return Err("segment lengths do not add up to the block length (corrupt file)".into());
            }
            // every leaf: (segment, work offset, offset of its bytes in the stored / neural region)
            let (ac, rest) = payload.split_at(bh.ac_len());
            let (mut w, mut so, mut no) = (0usize, 0usize, 0usize);
            let mut leaves = vec![];
            for (si, s) in segs.iter().enumerate().filter(|(_, s)| !kind::container(s.kind)) {
                let len = s.raw_len as usize;
                let src = match s.kind {
                    kind::STORED => { so += len; so - len }
                    kind::NEURAL => { let p = parse_neural_params(&s.params)?.4 as usize; no += p; no - p }
                    _ => 0,
                };
                leaves.push((si, w, src));
                w += len;
            }
            if so + no != rest.len() { return Err("block payload size does not match its segments (corrupt file)".into()); }
            let (stored, nregion) = rest.split_at(so);
            if bh.lanes.iter().map(|l| l.0 as usize).sum::<usize>() != leaves.len() {
                return Err("lane table does not match the segment table (corrupt file)".into());
            }
            // lanes own disjoint, consecutive slices of the work buffer
            let mut work = vec![0u8; total_work];
            let mut jobs_in: Vec<(usize, usize, &mut [u8], &[u8], std::ops::Range<usize>)> = Vec::with_capacity(bh.lanes.len());
            {
                let (mut rest_w, mut l0, mut a0) = (&mut work[..], 0usize, 0usize);
                for (li, &(n, alen)) in bh.lanes.iter().enumerate() {
                    let n = n as usize;
                    let w0 = leaves[l0].1;
                    let w1 = leaves.get(l0 + n).map(|l| l.1).unwrap_or(total_work);
                    let (lw, r) = std::mem::take(&mut rest_w).split_at_mut(w1 - w0);
                    rest_w = r;
                    jobs_in.push((li, w0, lw, &ac[a0..a0 + alen as usize], l0..l0 + n));
                    l0 += n;
                    a0 += alen as usize;
                }
            }
            // stored and neural leaves: copied from the payload, decoded (or, verifying, proven)
            let other = |s: &Segment, woff: usize, src: usize, out: &mut [u8]| -> Result<(), String> {
                let len = out.len();
                if s.kind == kind::STORED { out.copy_from_slice(&stored[src..src + len]); return Ok(()); }
                let nc = neural.ok_or("this file has neural segments but no neural model is available")?;
                let plen = parse_neural_params(&s.params)?.4 as usize;
                let np = &nregion[src..src + plen];
                match known {
                    Some(k) => {
                        let orig = k.work.get(woff..woff + len).ok_or("neural segment outside the work buffer")?;
                        verify_neural_leaf(nc, &s.params, np, orig, k.threads)?;
                        out.copy_from_slice(orig);
                    }
                    None => out.copy_from_slice(&neural_leaf(nc, &s.params, np, len)?),
                }
                Ok(())
            };
            let proofs = known.and_then(|k| k.lanes);
            // with the verifier's lanes there is nothing left to decode: no extra threads
            let jobs = if proofs.is_some() { 1 } else { lane_jobs.max(1).min(jobs_in.len()) };
            let queue = std::sync::Mutex::new(jobs_in.into_iter().rev().collect::<Vec<_>>());
            let results = run_lanes(bh.lanes.len(), jobs, |_| -> Result<(), String> {
                let (li, w0, lw, lac, lr) = queue.lock().unwrap().pop().unwrap();
                let Some(p) = proofs.map(|p| &p[li]) else {
                    return decode_lane(&mut Decoder::new(lac), &cfg, segs, &leaves[lr], w0, lw, other);
                };
                // the verifier decoded exactly this lane — same coded bytes, leaves, work range and
                // model — so its bytes are what decoding it here would give
                if !p.ok || p.ac != lac || p.leaves != lr || p.work != (w0..w0 + lw.len()) || p.cfg != cfg {
                    return Err("verification: lane does not match its concurrent decode".into());
                }
                lw.copy_from_slice(&p.bytes);
                // the verifier gave stored and neural leaves their original bytes: prove them
                for &(si, woff, src) in &leaves[lr] {
                    let s = &segs[si];
                    if matches!(s.kind, kind::STORED | kind::NEURAL) {
                        let rel = woff - w0;
                        other(s, woff, src, &mut lw[rel..rel + s.raw_len as usize])?;
                    }
                }
                Ok(())
            });
            for r in results { r?; }
            match known {
                // rebuilding this exact tree from this exact work buffer was already shown to give
                // the block's bytes
                Some(Known { rebuilt: Some(raw), work: w, .. }) if work[..] == w[..] => raw.to_vec(),
                _ => rebuild_all(&roots, segs, &work, bh.raw_len as usize)?,
            }
        }
        m => return Err(format!("unknown block mode {m}")),
    };
    if crc32(&out) != bh.raw_crc { return Err("block checksum mismatch: the file is corrupt".into()); }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_log2_and_entropy() {
        assert_eq!(log2_fx(1), 0);
        assert_eq!(log2_fx(2), 65536);
        assert_eq!(log2_fx(1024), 10 * 65536);
        let l3 = log2_fx(3) as f64 / 65536.0;
        assert!((l3 - 3f64.log2()).abs() < 1e-4);
        assert_eq!(order0_mbits(&[5u8; 1000]), 0);
        let all: Vec<u8> = (0..=255u8).cycle().take(256 * 40).collect();
        assert!((order0_mbits(&all) as i32 - 8000).abs() <= 1);
    }

    fn split_block(b: &[u8]) -> (BlockHeader, &[u8]) {
        let (bh, n) = format::read_block_header(&mut &b[4..]).unwrap();
        (bh, &b[n..])
    }

    fn plan1(data: &[u8]) -> super::super::plan::PlannedBlock {
        let mut p = super::super::plan::Planner::new(data, 1 << 24);
        p.next_block().unwrap().unwrap()
    }

    /// Neural segments round-trip and reject corruption. Needs the dev model artifacts (skips on a
    /// fresh checkout where the 285 MB files are absent).
    #[test]
    fn neural_segments_roundtrip_and_reject_corruption() {
        let (m, o2, t) = ("artifacts/model.nsc", "artifacts/o2.nsc", "../results/export/tokenizer.json");
        if [m, o2, t].iter().any(|p| std::fs::metadata(p).is_err()) { eprintln!("neural test: artifacts missing — skipping"); return; }
        let nc = std::sync::Arc::new(NeuralCtx {
            id: "test".into(), fingerprint: 1, binding: 0,
            model: crate::artifacts::Model::load(m), o2: crate::features::O2Table::load(o2),
            tok: crate::tokenizer::Tokenizer::load(std::path::Path::new(t)).unwrap(),
        });
        let text = b"The quick brown fox jumps over the lazy dog. Compression of natural language text. ".repeat(40);
        let mut pb = plan1(&text);
        for s in pb.segs.iter_mut() { s.kind = kind::TEXT; }
        let e = encode(&pb, &EncOpts::new(3, true, Some(nc.clone())));
        assert!(e.kinds >> kind::NEURAL & 1 == 1, "text did not go neural");
        // verified concurrently (lanes) and in parallel (neural windows): same bytes, not failed
        let ec = encode(&pb, &EncOpts::new(3, true, Some(nc.clone())).with_lane_jobs(2));
        assert!(!ec.verify_failed && ec.bytes == e.bytes, "concurrent verification changed the neural block");
        let (bh, payload) = split_block(&e.bytes);
        assert_eq!(decode(&bh, payload, Some(&nc), 1).unwrap(), text);
        assert!(decode(&bh, payload, None, 1).is_err(), "decoded neural segments without a model");
        let mut s = 3u64;
        for _ in 0..40 {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            let mut bad = payload.to_vec();
            let i = (s as usize) % bad.len();
            bad[i] ^= 1 << (s % 8);
            match decode(&bh, &bad, Some(&nc), 1) { Ok(d) => assert!(d == text, "silent corruption"), Err(_) => {} }
        }
    }

    fn adler32(d: &[u8]) -> u32 {
        let (mut a, mut b) = (1u32, 0u32);
        for &x in d { a = (a + x as u32) % 65521; b = (b + a) % 65521; }
        b << 16 | a
    }

    fn chunk(ty: &[u8], data: &[u8]) -> Vec<u8> {
        let mut c = (data.len() as u32).to_be_bytes().to_vec();
        c.extend_from_slice(ty);
        c.extend_from_slice(data);
        let crc = crc32(&c[4..]);
        c.extend_from_slice(&crc.to_be_bytes());
        c
    }

    /// A gzip member and a PNG (several IDAT chunks, mixed row filters) inside one block: both are
    /// recompressed (deflate, png, png-rows segments), rebuilt exactly, and corruption is caught.
    #[test]
    fn deflate_and_png_streams_roundtrip() {
        use crate::transform::deflate::deflate_like_zlib;
        let text: Vec<u8> = (0..800).flat_map(|i| format!("line {i}: the quick brown fox {} jumps\n", i * 7 % 13).into_bytes()).collect();
        let mut gz = vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3];
        gz.extend(deflate_like_zlib(&text, 6));
        gz.extend_from_slice(&crc32(&text).to_le_bytes());
        gz.extend_from_slice(&(text.len() as u32).to_le_bytes());
        // 40x30 RGB gradient image, filter type = row % 5
        let (w, h) = (40u32, 30u32);
        let g = png::Geom::new(w, h, 8, 2, 0).unwrap();
        let pix: Vec<u8> = (0..h).flat_map(|y| (0..w).flat_map(move |x| [(x * 6) as u8, (y * 8) as u8, ((x + y) * 3) as u8])).collect();
        let types: Vec<u8> = (0..h).map(|r| (r % 5) as u8).collect();
        let mut scan = vec![];
        png::refilter(&types, &pix, &g, &mut scan).unwrap();
        let mut z = vec![0x78, 0x9c];
        z.extend(deflate_like_zlib(&scan, 6));
        z.extend_from_slice(&adler32(&scan).to_be_bytes());
        let mut ihdr = w.to_be_bytes().to_vec();
        ihdr.extend_from_slice(&h.to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
        let mut pngf = b"\x89PNG\r\n\x1a\n".to_vec();
        pngf.extend(chunk(b"IHDR", &ihdr));
        for c in z.chunks(500) { pngf.extend(chunk(b"IDAT", c)); }
        pngf.extend(chunk(b"IEND", &[]));
        let mut data = b"header text before the streams\n".repeat(20);
        data.extend_from_slice(&gz);
        data.extend_from_slice(b"some bytes between");
        data.extend_from_slice(&pngf);
        data.extend_from_slice(&b"trailing text after\n".repeat(30));

        let e = encode(&plan1(&data), &EncOpts::new(5, true, None));
        assert_eq!(e.mode, MODE_CM);
        for k in [kind::DEFLATE, kind::PNG, kind::ROWS, kind::IMAGE] {
            assert!(e.kinds >> k & 1 == 1, "no {} segment (kinds {:#x})", kind::name(k), e.kinds);
        }
        let (bh, payload) = split_block(&e.bytes);
        assert_eq!(decode(&bh, payload, None, 1).unwrap(), data);
        assert!(e.bytes.len() * 4 < gz.len() + pngf.len(), "streams were not recompressed: {} bytes", e.bytes.len());
        let mut s = 11u64;
        for _ in 0..40 {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            let mut bad = payload.to_vec();
            let i = (s as usize) % bad.len();
            bad[i] ^= 1 << (s % 8);
            if let Ok(d) = decode(&bh, &bad, None, 1) { assert!(d == data, "silent corruption"); }
        }
        // a lying tree (child count past the table) is rejected, not followed
        let mut bh2 = bh.clone();
        let c = bh2.segments.iter().position(|x| x.kind == kind::DEFLATE).unwrap();
        bh2.segments[c].params = 9999u32.to_le_bytes().to_vec();
        assert!(decode(&bh2, payload, None, 1).is_err());
    }

    /// A gzip member around a deflate stream of `parse`.
    fn gzip_of(data: &[u8], parse: &crate::transform::deflate::Parse) -> Vec<u8> {
        use crate::transform::deflate::{writer, Parse};
        let mut gz = vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3];
        gz.extend((0..8).find_map(|pb| writer::write(&Parse { end_pad: (pb, 0), ..parse.clone() }, data).ok()).unwrap());
        gz.extend_from_slice(&crc32(data).to_le_bytes());
        gz.extend_from_slice(&(data.len() as u32).to_le_bytes());
        gz
    }

    /// A deflate stream that zlib did not write (another encoder's choices at every other token)
    /// is recompressed all the same, from a writer for decoders of level 2: its tokens are noted
    /// as choices among the matches that exist. A writer for level 1 leaves it as bytes.
    #[test]
    fn streams_of_other_encoders_are_recompressed() {
        use crate::transform::deflate::front::tests::{foreign, text};
        let content = text(60_000, 5);
        for style in [2u32, 3] {
            let gz = gzip_of(&content, &foreign(&content, style, 4000, 3000, 9));
            let mut data = b"before the stream ".to_vec();
            data.extend_from_slice(&gz);
            data.extend_from_slice(b" and after it");
            let mut sizes = vec![];
            for compat in [false, true] {
                let mut opts = EncOpts::new(5, true, None);
                opts.compat = compat;
                let e = encode(&plan1(&data), &opts);
                assert!(!e.verify_failed, "style {style}");
                assert_eq!(e.kinds & kind::bit(kind::DEFLATE) != 0, !compat, "style {style}: kinds {:#x}", e.kinds);
                let (bh, payload) = split_block(&e.bytes);
                assert!(decode(&bh, payload, None, 2).unwrap() == data, "style {style}: roundtrip");
                sizes.push(e.bytes.len());
            }
            assert!(sizes[0] * 100 < sizes[1] * 85, "style {style}: level 2 {} bytes, level 1 {} (the stream {})", sizes[0], sizes[1], gz.len());
        }
    }

    /// A stream may not cost a block more taken apart than it is: matches over bytes no model
    /// predicts, which deflate writes in a few bytes each, do not; and a stream whose parts do
    /// come to more than it is named, to be left as the bytes it is.
    #[test]
    fn streams_that_cost_more_apart_are_left() {
        use crate::transform::deflate::deflate_like_zlib;
        use crate::transform::deflate::front::tests::text;
        let mut s = 77u64;
        let noise: Vec<u8> = (0..20_000).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 24) as u8 }).collect();
        let content = noise.repeat(5);
        let mut gz = vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3];
        gz.extend(deflate_like_zlib(&content, 6));
        gz.extend_from_slice(&crc32(&content).to_le_bytes());
        gz.extend_from_slice(&(content.len() as u32).to_le_bytes());
        assert!(gz.len() < 24_000, "the stream should be a fifth of its content: {} bytes", gz.len());
        let words = text(30_000, 3);
        let mut data = words.clone();
        data.extend_from_slice(&gz);
        data.extend_from_slice(&words[..10_000]);
        let alone = encode(&plan1(&words), &EncOpts::new(5, true, None)).bytes.len();
        let e = encode(&plan1(&data), &EncOpts::new(5, true, None));
        assert_eq!(e.mode, MODE_CM);
        let (bh, payload) = split_block(&e.bytes);
        assert!(decode(&bh, payload, None, 2).unwrap() == data);
        assert!(e.bytes.len() < alone + gz.len() + 400, "{} bytes for a stream of {} and text that alone takes {alone}", e.bytes.len(), gz.len());
        // content that is stored when it is seen for itself (long, nothing a quick look shrinks: no
        // match in its first part, a fifth of the rest copies of bytes just before) and that
        // deflate did shrink: the stream is shorter than its parts, and stays as it is
        let mut content: Vec<u8> = (0..70_000).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 24) as u8 }).collect();
        while content.len() < 330_000 {
            content.extend((0..6000).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 24) as u8 }));
            let from = content.len() - 8000;
            content.extend_from_within(from..from + 1500);
        }
        let mut gz = vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3];
        gz.extend(deflate_like_zlib(&content, 6));
        gz.extend_from_slice(&crc32(&content).to_le_bytes());
        gz.extend_from_slice(&(content.len() as u32).to_le_bytes());
        assert!(gz.len() * 100 < content.len() * 90, "the stream should be well below its content: {} bytes of {}", gz.len(), content.len());
        let mut data = words.clone();
        data.extend_from_slice(&gz);
        data.extend_from_slice(&words[..10_000]);
        let e = encode(&plan1(&data), &EncOpts::new(5, true, None));
        let (bh, payload) = split_block(&e.bytes);
        assert!(decode(&bh, payload, None, 2).unwrap() == data);
        assert!(e.kinds & kind::bit(kind::DEFLATE) == 0, "the stream was taken apart all the same: kinds {:#x}", e.kinds);
        assert!(e.bytes.len() < alone + gz.len() + 1200, "{} bytes for a stream of {} and text that alone takes {alone}", e.bytes.len(), gz.len());
        // (a writer for decoders of level 1 does as nsc 0.3 did: the stream apart whatever it costs,
        // which here is the whole block stored)
        let mut compat = EncOpts::new(5, true, None);
        compat.compat = true;
        let old = encode(&plan1(&data), &compat);
        assert!(old.bytes.len() > e.bytes.len() + 20_000, "level 1 {} bytes, level 2 {}", old.bytes.len(), e.bytes.len());
        // which streams those are: the top-level containers whose leaves came to more than they stand for
        let seg = |k: u8, raw_len: u32, kids: Option<u32>| Segment { kind: k, ver: 1, xform: 0, raw_len, params: kids.map_or(vec![], |n| n.to_le_bytes().to_vec()) };
        let table = vec![
            seg(kind::TEXT, 100, None),
            seg(kind::DEFLATE, 1000, Some(2)), seg(kind::GENERIC, 3000, None), seg(kind::GENERIC, 50, None),
            seg(kind::GENERIC, 60, None),
            seg(kind::PNG, 500, Some(3)), seg(kind::GENERIC, 2, None),
            seg(kind::DEFLATE, 490, Some(2)), seg(kind::STORED, 600, None), seg(kind::GENERIC, 9, None), seg(kind::GENERIC, 8, None),
            seg(kind::JPEG, 700, Some(2)), seg(kind::GENERIC, 300, None), seg(kind::JPEGC, 2000, None),
        ];
        let leaves: Vec<(usize, usize, usize)> = table.iter().enumerate().filter(|(_, s)| !kind::container(s.kind)).map(|(i, _)| (i, 0, 0)).collect();
        // text 30; the first stream 990 + 20 for 1000 bytes; between 60; the picture 2 + (600 stored + 9) + 8 for 500; the JPEG 100 + 400 for 700
        let cost = [30u32, 990, 20, 60, 2, 600, 9, 8, 100, 400];
        assert_eq!(losers(&table, &leaves, &cost), Lost { left: vec![(100, 1100), (1160, 1660)], stored: vec![] });
        assert!(losers(&table, &leaves, &[30, 900, 20, 60, 2, 400, 9, 8, 100, 400]).is_empty());
        // and bytes that code longer than they are
        assert_eq!(losers(&table, &leaves, &[130, 900, 20, 60, 2, 400, 9, 8, 100, 400]), Lost { left: vec![], stored: vec![(0, 100)] });
        let segs = vec![super::super::plan::PSeg { kind: kind::TEXT, len: 300, params: vec![], label: "text" }, super::super::plan::PSeg { kind: kind::GENERIC, len: 200, params: vec![], label: "binary" }];
        let got: Vec<(u8, u32)> = store_ranges(&segs, &[(100, 150), (250, 320), (400, 500)]).iter().map(|s| (s.kind, s.len)).collect();
        assert_eq!(got, vec![(kind::TEXT, 100), (kind::STORED, 50), (kind::TEXT, 100), (kind::STORED, 50), (kind::STORED, 20), (kind::GENERIC, 80), (kind::STORED, 100)]);
    }

    /// A block whose work buffer is several lanes long: leaves are split (image rows kept whole),
    /// the output does not depend on how many lanes run at once, and it decodes exactly.
    #[test]
    fn lanes_split_and_decode_in_parallel() {
        let mut data = b"P5\n1000 1500\n255\n".to_vec();
        data.extend((0..1500u32).flat_map(|y| (0..1000u32).map(move |x| ((x / 7 + y / 5) & 0xff) as u8)));
        let mut s = 5u64;
        data.extend((0..1_200_000).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; b"abcdefgh  \n"[(s % 11) as usize] }));
        let pb = plan1(&data);
        let mut opts = EncOpts::new(1, true, None);
        opts.lane = 1 << 20;
        let one = encode(&pb, &opts);
        opts.lane_jobs = 3;
        let three = encode(&pb, &opts);
        assert!(one.bytes == three.bytes, "output depends on lane parallelism");
        let (bh, payload) = split_block(&three.bytes);
        assert!(bh.lanes.len() >= 2, "expected several lanes, got {}", bh.lanes.len());
        let img_leaves: Vec<&Segment> = bh.segments.iter().filter(|x| x.kind == kind::IMAGE).collect();
        assert!(img_leaves.len() > 1 && img_leaves.iter().all(|x| x.raw_len % 1000 == 0), "image split off its rows");
        assert_eq!(decode(&bh, payload, None, 1).unwrap(), data);
        assert_eq!(decode(&bh, payload, None, 4).unwrap(), data);
    }

    /// x86-64 code in an executable: for a decoder of level 2 it is transformed by the instruction
    /// decoder and coded with the instruction model; for one of level 1, as nsc 0.3 did. Both
    /// decode exactly, and the first is smaller.
    #[test]
    fn x86_code_gets_the_instruction_model() {
        use crate::cm::levels::ctx;
        let code = crate::x86::sample_code(1500);
        let mut data = vec![0u8; 0x200];
        data[..8].copy_from_slice(b"\x7fELF\x02\x01\x01\0");
        (data[16], data[18], data[32], data[54], data[56], data[64], data[68]) = (2, 62, 64, 56, 1, 1, 5);
        data[72..80].copy_from_slice(&0x200u64.to_le_bytes());
        data[96..104].copy_from_slice(&(code.len() as u64).to_le_bytes());
        data.extend_from_slice(&code);
        data.extend((0..3000u32).flat_map(|i| (i * 16).to_le_bytes()));
        let pb = plan1(&data);
        let mut sizes = vec![];
        for compat in [false, true] {
            let mut opts = EncOpts::new(5, true, None);
            opts.compat = compat;
            let e = encode(&pb, &opts);
            assert!(!e.verify_failed);
            let (bh, payload) = split_block(&e.bytes);
            let exe: Vec<&Segment> = bh.segments.iter().filter(|s| s.kind == kind::EXE).collect();
            assert_eq!(exe.len(), 1);
            assert_eq!((exe[0].raw_len as usize, exe[0].xform), (code.len(), if compat { transform::bcj::X86 } else { transform::bcj::X64 }));
            assert_eq!(bh.cfg.unwrap().ctx_mask >> ctx::X86 & 1 == 1, !compat);
            assert_eq!(decode(&bh, payload, None, 2).unwrap(), data);
            sizes.push(e.bytes.len());
        }
        assert!(sizes[0] * 100 < sizes[1] * 94, "level 2 {} bytes, level 1 {} bytes", sizes[0], sizes[1]);
    }

    /// SQLite databases are taken apart: a table's pages into structure, numbers and text; rows
    /// that overflow their pages gathered from their chains, so that a compressed document in
    /// one is found as a zlib stream and taken apart too. A database of a few pages of plain
    /// rows is left as bytes, and so is every database for a decoder of level 1. A zlib stream
    /// on its own is found as well.
    #[test]
    fn databases_are_taken_apart() {
        use crate::recode::sqlite::tests::{blob, database, orders, table, zlib};
        let docs: Vec<Vec<u8>> = (0..5).map(|i| zlib(&blob(9000 + 3000 * i, 20 + i as u64, true))).collect();
        let refs: Vec<&[u8]> = docs.iter().map(|d| &d[..]).collect();
        let documents = database(&refs, 1024, true);
        let text: Vec<Vec<u8>> = (0..5).map(|i| blob(5000 + 1000 * i, 30 + i as u64, true)).collect();
        let refs: Vec<&[u8]> = text.iter().map(|d| &d[..]).collect();
        let plain = database(&refs, 1024, true);
        let rows = table(&orders(2500), 4096, false);
        let used = table(&orders(2500), 1024, true);
        let mut alone = b"a document kept compressed, between other bytes: ".to_vec();
        alone.extend_from_slice(&docs[4]);
        alone.extend_from_slice(b" and what follows it");
        let mut sizes = vec![];
        for (name, data, compat, sqlite, deflate) in [("documents", &documents, false, true, true), ("documents, level 1", &documents, true, false, false),
            ("a table", &rows, false, true, false), ("a table, level 1", &rows, true, false, false), ("a table in use", &used, false, true, false),
            ("a few long rows of text", &plain, false, false, false), ("a zlib stream", &alone, false, false, true), ("a zlib stream, level 1", &alone, true, false, false)] {
            let mut opts = EncOpts::new(5, true, None);
            opts.compat = compat;
            let e = encode(&plan1(data), &opts);
            assert!(!e.verify_failed, "{name}");
            assert_eq!(e.kinds & kind::bit(kind::SQLITE) != 0, sqlite, "{name}: kinds {:#x}", e.kinds);
            assert_eq!(e.kinds & kind::bit(kind::DEFLATE) != 0, deflate, "{name}: kinds {:#x}", e.kinds);
            let (bh, payload) = split_block(&e.bytes);
            assert!(decode(&bh, payload, None, 2).unwrap() == *data, "{name}: roundtrip");
            sizes.push(e.bytes.len());
        }
        assert!(sizes[0] * 100 < sizes[1] * 60, "documents: level 2 {} bytes, level 1 {} bytes", sizes[0], sizes[1]);
        assert!(sizes[2] * 100 < sizes[3] * 85, "a table: level 2 {} bytes, level 1 {} bytes", sizes[2], sizes[3]);
        assert!(sizes[6] * 100 < sizes[7] * 60, "a zlib stream: level 2 {} bytes, level 1 {} bytes", sizes[6], sizes[7]);
    }

    /// FLAC frames are decoded to samples for the audio model and come back bit for bit — also
    /// when the samples are more than a lane takes: the container keeps its two children, the
    /// second a group of the leaf's pieces. Not for a decoder of level 1.
    #[test]
    fn flac_frames_become_samples() {
        use crate::recode::flac::tests::stream;
        let mut data = b"fLaC and what a file has before its frames ".to_vec();
        data.extend(stream(420, 2, 16));
        data.extend_from_slice(b"TAG after them");
        let mut sizes = vec![];
        for (name, lane, compat) in [("one lane", 32 << 20, false), ("lanes", 1 << 20, false), ("level 1", 32 << 20, true)] {
            let mut opts = EncOpts::new(5, true, None);
            (opts.lane, opts.compat) = (lane, compat);
            let e = encode(&plan1(&data), &opts);
            assert!(!e.verify_failed, "{name}");
            assert_eq!(e.kinds & kind::bit(kind::FLAC) != 0, !compat, "{name}: kinds {:#x}", e.kinds);
            let (bh, payload) = split_block(&e.bytes);
            assert_eq!(bh.lanes.len() > 1, lane == 1 << 20, "{name}: {} lanes", bh.lanes.len());
            if !compat {
                let flac = bh.segments.iter().position(|s| s.kind == kind::FLAC).unwrap();
                assert_eq!(u32_at(&bh.segments[flac].params, 0).unwrap(), 2, "{name}");
                assert_eq!(bh.segments[flac + 2].kind, if bh.lanes.len() > 1 { kind::GROUP } else { kind::AUDIO }, "{name}");
            }
            assert!(decode(&bh, payload, None, 2).unwrap() == data, "{name}: roundtrip");
            sizes.push(e.bytes.len());
        }
        assert!(sizes[0] * 100 < sizes[2] * 97 && sizes[1] * 100 < sizes[2] * 98, "level 2 {} and {} bytes, level 1 {}", sizes[0], sizes[1], sizes[2]);
    }

    /// A GIF's images are decoded to frames of colour indices for their own model, and the file
    /// comes back bit for bit, whatever its encoder did; not for a decoder of level 1.
    #[test]
    fn gif_images_become_frames() {
        use crate::recode::gif::tests::animation;
        let mut data = b"before the picture ".to_vec();
        data.extend(animation(8, 120, 90, 8).0);
        data.extend_from_slice(b" and after it");
        let mut sizes = vec![];
        for compat in [false, true] {
            let mut opts = EncOpts::new(5, true, None);
            opts.compat = compat;
            let e = encode(&plan1(&data), &opts);
            assert!(!e.verify_failed);
            assert_eq!((e.kinds & kind::bit(kind::GIF) != 0, e.kinds & kind::bit(kind::PIX) != 0), (!compat, !compat), "kinds {:#x}", e.kinds);
            let (bh, payload) = split_block(&e.bytes);
            assert!(decode(&bh, payload, None, 2).unwrap() == data, "roundtrip");
            sizes.push(e.bytes.len());
        }
        assert!(sizes[0] * 100 < sizes[1] * 80, "level 2 {} bytes, level 1 {}", sizes[0], sizes[1]);
    }

    /// A TIFF's compressed strips are decoded to pixels for the image model, whatever compressed
    /// them, and the file comes back bit for bit; not for a decoder of level 1.
    #[test]
    fn tiff_strips_become_pixels() {
        use crate::recode::tiff::tests::{file, Layout, RGB};
        let big = Layout { width: 200, height: 150, rows_per_strip: 16, ..RGB };
        for (what, layouts) in [
            ("lzw", vec![Layout { predictor: 2, ..big }]),
            ("packbits and zlib", vec![Layout { compression: 32773, ..big }, Layout { compression: 8, samples: 1, bits: 16, predictor: 2, ..big }]),
            ("tiles", vec![Layout { tile: 64, ..big }]),
        ] {
            let mut data = b"before the picture ".to_vec();
            data.extend(file(&layouts, what != "tiles", what == "lzw", 5).0);
            data.extend_from_slice(b" and after it");
            let mut sizes = vec![];
            for compat in [false, true] {
                let mut opts = EncOpts::new(5, true, None);
                opts.compat = compat;
                let e = encode(&plan1(&data), &opts);
                assert!(!e.verify_failed, "{what}");
                assert_eq!(e.kinds & kind::bit(kind::STRIPS) != 0, !compat, "{what}: kinds {:#x}", e.kinds);
                assert!(e.kinds & kind::bit(kind::IMAGE) != 0 || compat, "{what}: kinds {:#x}", e.kinds);
                let (bh, payload) = split_block(&e.bytes);
                assert!(decode(&bh, payload, None, 2).unwrap() == data, "{what}: roundtrip");
                sizes.push(e.bytes.len());
            }
            assert!(sizes[0] * 100 < sizes[1] * 95, "{what}: level 2 {} bytes, level 1 {}", sizes[0], sizes[1]);
        }
    }

    /// Text that is numbers to a fair part gets the number model, from a writer for decoders of
    /// level 2 only; prose does not.
    #[test]
    fn logs_get_the_number_model() {
        use crate::cm::levels::ctx;
        let log = crate::cm::tests::sample_log(2500);
        let prose = b"It was the best of times, it was the worst of times, it was the age of wisdom. ".repeat(1500);
        let mut sizes = vec![];
        for (data, compat, on) in [(&log, false, true), (&log, true, false), (&prose, false, false)] {
            let mut opts = EncOpts::new(5, true, None);
            opts.compat = compat;
            let e = encode(&plan1(data), &opts);
            assert!(!e.verify_failed);
            let (bh, payload) = split_block(&e.bytes);
            assert!(bh.segments.iter().all(|s| s.kind == kind::TEXT));
            assert_eq!(bh.cfg.unwrap().ctx_mask >> ctx::NUM & 1 == 1, on);
            assert!(decode(&bh, payload, None, 2).unwrap() == *data);
            sizes.push(e.bytes.len());
        }
        assert!(sizes[0] * 100 < sizes[1] * 92, "a log: level 2 {} bytes, level 1 {} bytes", sizes[0], sizes[1]);
    }

    /// Corpus JPEGs (baseline and progressive) inside one block with other data: recompressed as
    /// JPEG containers, rebuilt exactly, smaller than the originals. Skips without the corpus.
    #[test]
    fn jpegs_recompress_exactly() {
        let names = ["baseline_3_DownloadVS2010.jpg", "progressive_0_jbaer_wave.jpg"];
        let files: Vec<Vec<u8>> = names.iter().filter_map(|n| std::fs::read(format!("bench/corpus/jpeg/{n}")).ok()).collect();
        if files.len() != names.len() { eprintln!("jpeg block test: corpus missing — skipping"); return; }
        let mut data = b"some leading text\n".repeat(10);
        for f in &files { data.extend_from_slice(f); data.extend_from_slice(b"between"); }
        let e = encode(&plan1(&data), &EncOpts::new(3, true, None));
        assert!(e.kinds >> kind::JPEG & 1 == 1 && e.kinds >> kind::JPEGC & 1 == 1, "kinds {:#x}", e.kinds);
        let (bh, payload) = split_block(&e.bytes);
        assert_eq!(decode(&bh, payload, None, 2).unwrap(), data);
        let orig: usize = files.iter().map(|f| f.len()).sum();
        assert!(e.bytes.len() * 100 < orig * 82, "jpegs only shrank to {} of {orig}", e.bytes.len());
        let mut s = 21u64;
        for _ in 0..12 {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            let mut bad = payload.to_vec();
            let i = (s as usize) % bad.len();
            bad[i] ^= 1 << (s % 8);
            if let Ok(d) = decode(&bh, &bad, None, 2) { assert!(d == data, "silent corruption"); }
        }
    }

    /// MP3 streams inside one block with other data: taken apart as MP3 containers from level 3
    /// up (not below, not when writing for decoder level 1), rebuilt exactly, and corruption is
    /// caught.
    #[test]
    fn mp3_streams_recompress_exactly() {
        let files: [&[u8]; 2] = [include_bytes!("../../tests/fixtures/tone_stereo.mp3"), include_bytes!("../../tests/fixtures/tone_mono_lsf.mp3")];
        let mut data = b"some leading text\n".repeat(10);
        for f in files { data.extend_from_slice(f); data.extend_from_slice(b"between the streams"); }
        let pb = plan1(&data);
        let e = encode(&pb, &EncOpts::new(3, true, None));
        assert!(e.kinds & kind::bit(kind::MP3) != 0 && e.kinds & kind::bit(kind::MP3F) != 0, "kinds {:#x}", e.kinds);
        let (bh, payload) = split_block(&e.bytes);
        assert_eq!(decode(&bh, payload, None, 2).unwrap(), data);
        let orig: usize = files.iter().map(|f| f.len()).sum();
        assert!(e.bytes.len() * 100 < orig * 85, "mp3s only shrank to {} of {orig}", e.bytes.len());
        // concurrent verification gives the same block
        assert!(encode(&pb, &EncOpts::new(3, true, None).with_lane_jobs(4)).bytes == e.bytes);
        let mut s = 31u64;
        for _ in 0..40 {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            let mut bad = payload.to_vec();
            let i = (s as usize) % bad.len();
            bad[i] ^= 1 << (s % 8);
            if let Ok(d) = decode(&bh, &bad, None, 2) { assert!(d == data, "silent corruption"); }
        }
        // a lying tree: the frames leaf swapped for a generic one is refused
        let mut bh2 = bh.clone();
        let c = bh2.segments.iter().position(|x| x.kind == kind::MP3F).unwrap();
        bh2.segments[c].kind = kind::GENERIC;
        assert!(decode(&bh2, payload, None, 1).is_err());
        // below level 3, and for decoder level 1, no recoder is used
        let low = encode(&pb, &EncOpts::new(2, true, None));
        let mut compat = EncOpts::new(5, true, None);
        compat.compat = true;
        for e in [low, encode(&pb, &compat)] {
            assert!(e.kinds & (kind::bit(kind::MP3) | kind::bit(kind::MP3F) | kind::bit(kind::GROUP)) == 0, "kinds {:#x}", e.kinds);
            let (bh, payload) = split_block(&e.bytes);
            assert_eq!(decode(&bh, payload, None, 1).unwrap(), data);
        }
    }

    /// Lanes verified while they are coded give the same block as the decode-back done afterwards,
    /// and the final check accepts the verifier's lanes only when they match the block exactly.
    #[test]
    fn concurrent_verification_matches_and_rejects_bad_lanes() {
        let mut data = b"P5\n600 500\n255\n".to_vec();
        data.extend((0..500u32).flat_map(|y| (0..600u32).map(move |x| ((x / 5 + y / 3) & 0xff) as u8)));
        let mut s = 7u64;
        data.extend((0..1_400_000).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; b"the cat sat on  a mat\n"[(s % 22) as usize] }));
        let pb = plan1(&data);
        let mut seq = EncOpts::new(2, true, None);
        seq.lane = 1 << 20;
        let conc = seq.with_lane_jobs(4);
        assert!(conc.lane_plan().0 && !seq.lane_plan().0);
        let a = encode(&pb, &seq);
        let b = encode(&pb, &conc);
        assert!(!a.verify_failed && !b.verify_failed && a.bytes == b.bytes, "concurrent verification changed the block");
        let (bh, payload) = split_block(&b.bytes);
        assert!(bh.lanes.len() >= 2, "expected several lanes");
        assert_eq!(decode(&bh, payload, None, 2).unwrap(), data);

        let Step::Coded(mut c) = code_block(&pb, &conc) else { panic!("block was not coded") };
        assert!(c.proofs.len() == c.bh.lanes.len() && c.rebuilt_ok);
        assert!(verify_coded(&c, &conc));
        let i = c.proofs[1].bytes.len() / 2;
        c.proofs[1].bytes[i] ^= 4;
        assert!(!verify_coded(&c, &conc), "a wrong decoded byte passed verification");
        c.proofs[1].bytes[i] ^= 4;
        c.proofs[0].ac.push(0);
        assert!(!verify_coded(&c, &conc), "a lane decoded from other bytes passed verification");
        c.proofs[0].ac.pop();
        c.proofs[0].ok = false;
        assert!(!verify_coded(&c, &conc), "a failed lane decode passed verification");
        c.proofs[0].ok = true;
        let last = c.proofs.pop().unwrap();
        assert!(!verify_coded(&c, &conc), "a missing lane passed verification");
        c.proofs.push(last);
        assert!(verify_coded(&c, &conc));
    }

    #[test]
    fn block_roundtrip_and_stored_fallback() {
        let text = b"hello world, hello nsc. ".repeat(500);
        let e = encode(&plan1(&text), &EncOpts::new(3, true, None));
        assert_eq!(e.mode, MODE_CM);
        let (bh, payload) = split_block(&e.bytes);
        assert_eq!(decode(&bh, payload, None, 1).unwrap(), text);
        // random bytes cannot be compressed: stored, and still roundtrips
        let mut s = 1u64;
        let rnd: Vec<u8> = (0..4096).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s as u8 }).collect();
        let e = encode(&plan1(&rnd), &EncOpts::new(1, true, None));
        assert_eq!(e.mode, MODE_STORED);
    }
}
