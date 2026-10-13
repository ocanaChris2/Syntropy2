//! NSC3: the block-parallel, model-mixing container (stable format 1).
//!
//! `compress_path`/`decompress_path` handle files or stdin/stdout ("-"). They write to a `.partial`
//! file renamed into place only on success, so a failed run never leaves a truncated output. Worker
//! count is capped by available memory, which can only lower `--jobs` and never changes the output.
pub mod block;
pub mod format;
pub mod plan;
pub mod sched;
pub mod table;

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom};

pub use block::{EncOpts, NeuralCtx};
pub use sched::{Loader, Stats};

/// Block size in neural mode: small, so neural text (inherently sequential to decode) spreads over
/// many blocks that decode in parallel.
pub const NEURAL_BLOCK: usize = 128 << 10;

/// A loader for files that bind no model (and an error if one does).
#[cfg(test)]
pub fn no_model(b: &[(String, u64)]) -> Result<Option<std::sync::Arc<NeuralCtx>>, String> {
    if b.is_empty() { Ok(None) } else { Err(format!("this file needs neural model '{}'", b[0].0)) }
}

/// `MemAvailable` from /proc/meminfo, in bytes.
pub fn mem_available() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = s.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

/// Memory budget for in-flight blocks: the `mem_cap` byte budget if given, else 85% of
/// `MemAvailable` (unbounded if unknown).
pub fn mem_budget(mem_cap: Option<u64>) -> u64 {
    mem_cap.or_else(|| mem_available().map(|a| a / 100 * 85)).unwrap_or(u64::MAX)
}

/// Largest block the planner may grow to so a recompressible stream is not cut (levels that
/// recompress streams only).
pub fn extend_cap(level: u32, block: usize) -> Option<usize> {
    (level >= 2).then(|| (8 * block).min(format::MAX_BLOCK_RAW as usize))
}

/// Worker count: at most `requested`, at most `max_units` (blocks), and as many as fit in available
/// memory (85% of `MemAvailable`, or the `mem_cap` byte budget if given). Never below 1.
pub fn plan_jobs(requested: usize, max_units: usize, per_job: usize, mem_cap: Option<u64>) -> (usize, Option<String>) {
    let mut jobs = requested.max(1).min(max_units.max(1));
    let budget = mem_cap.or_else(|| mem_available().map(|a| a / 100 * 85));
    let mut warn = None;
    if let Some(b) = budget {
        let fit = (b / per_job.max(1) as u64) as usize;
        if fit == 0 {
            warn = Some(format!("needs ~{} MB per worker but only ~{} MB is available; continuing with 1 worker \
                                 (a lower --level uses less memory)", per_job >> 20, b >> 20));
            jobs = 1;
        } else {
            jobs = jobs.min(fit);
        }
    }
    (jobs, warn)
}

fn open_in(p: &str) -> Result<Box<dyn Read + Send>, String> {
    if p == "-" { return Ok(Box::new(BufReader::with_capacity(1 << 20, std::io::stdin()))); }
    let f = File::open(p).map_err(|e| format!("open {p}: {e}"))?;
    Ok(Box::new(BufReader::with_capacity(1 << 20, f)))
}

/// Open the input and, for a container file whose samples are found through an index (MP4, whose
/// index may sit at its end; Matroska, whose blocks are found by walking it), read that index
/// first. A pipe is spooled to a temporary file for this, so the output is the same
/// however the input arrives. With `index` off (no format recoders) this is [`open_in`].
fn open_indexed(p: &str, index: bool) -> Result<(Box<dyn Read + Send>, Vec<(u64, crate::recode::mp4::Index)>), String> {
    if !index { return Ok((open_in(p)?, vec![])); }
    let io = |e: std::io::Error| format!("read {p}: {e}");
    let mut f = if p == "-" {
        let mut stdin = std::io::stdin();
        let mut head = [0u8; 16];
        let got = crate::bench::read_full(&mut stdin, &mut head).map_err(io)?;
        let head = std::io::Cursor::new(head[..got].to_vec());
        if !crate::recode::indexable(head.get_ref()) {
            return Ok((Box::new(BufReader::with_capacity(1 << 20, head.chain(stdin))), vec![]));
        }
        let path = std::env::temp_dir().join(format!("nsc-spool-{}", std::process::id()));
        let mut spool = File::options().read(true).write(true).create_new(true).open(&path).map_err(|e| format!("spool {}: {e}", path.display()))?;
        // the file lives as long as this handle
        let _ = std::fs::remove_file(&path);
        std::io::copy(&mut head.chain(stdin), &mut spool).map_err(io)?;
        spool
    } else {
        File::open(p).map_err(|e| format!("open {p}: {e}"))?
    };
    let len = f.metadata().map_err(io)?.len();
    let idx = crate::recode::index(&mut f, len);
    f.seek(SeekFrom::Start(0)).map_err(io)?;
    Ok((Box::new(BufReader::with_capacity(1 << 20, f)), idx.map(|i| vec![(0, i)]).unwrap_or_default()))
}

/// Run `f` writing to `out` ("-" = stdout) through a `.partial` file renamed on success.
fn with_output<T>(out: &str, f: impl FnOnce(&mut dyn std::io::Write) -> Result<T, String>) -> Result<T, String> {
    if out == "-" {
        let stdout = std::io::stdout();
        let mut w = BufWriter::with_capacity(1 << 20, stdout.lock());
        return f(&mut w);
    }
    let partial = format!("{out}.partial");
    let r = (|| {
        let file = File::create(&partial).map_err(|e| format!("create {partial}: {e}"))?;
        let mut w = BufWriter::with_capacity(1 << 20, file);
        let v = f(&mut w)?;
        let file = w.into_inner().map_err(|e| format!("write {partial}: {e}"))?;
        file.sync_all().ok();
        Ok(v)
    })();
    match r {
        Ok(v) => { std::fs::rename(&partial, out).map_err(|e| format!("rename {partial} -> {out}: {e}"))?; Ok(v) }
        Err(e) => { let _ = std::fs::remove_file(&partial); Err(e) }
    }
}

/// Per-worker memory estimate for encoding blocks of `block` bytes at `level`.
pub fn encode_mem(level: u32, block: usize) -> usize {
    crate::cm::levels::preset(level, crate::cm::levels::Kind::Generic, block).memory() + 3 * block
}

pub fn compress_path(input: &str, output: &str, opts: &EncOpts, jobs: usize, block_size: usize, mem_cap: Option<u64>) -> Result<(Stats, Option<String>), String> {
    let size = if input == "-" { None } else { std::fs::metadata(input).ok().map(|m| m.len()) };
    let units = size.map(|s| (s as usize).div_ceil(block_size).max(1)).unwrap_or(usize::MAX);
    let per_job = encode_mem(opts.level, size.map(|s| (s as usize).min(block_size)).unwrap_or(block_size));
    let (jobs, warn) = plan_jobs(jobs, units, per_job, mem_cap);
    // lanes of one big block coded at once: as many model table sets as fit the memory budget
    let budget = mem_budget(mem_cap);
    let lane_tables = crate::cm::levels::preset(opts.level, crate::cm::levels::Kind::Generic, opts.lane + opts.lane / 2).memory() + 3 * opts.lane;
    let opts = &opts.with_lane_jobs(((budget / lane_tables as u64) as usize).clamp(1, opts.lane_jobs.max(1)));
    let (reader, indexes) = open_indexed(input, opts.specialists())?;
    let st = with_output(output, |w| sched::compress_meta(reader, w, opts, jobs, block_size, budget, sched::Meta { indexes, ..Default::default() }))?;
    Ok((st, warn))
}

/// Compress an arbitrary reader (whose total size is `size` when known) to `output` with header
/// flags and a file table (archives).
#[allow(clippy::too_many_arguments)]
pub fn compress_reader<R: Read>(reader: R, size: Option<u64>, output: &str, opts: &EncOpts, jobs: usize, block_size: usize, mem_cap: Option<u64>, meta: sched::Meta) -> Result<(Stats, Option<String>), String> {
    let units = size.map(|s| (s as usize).div_ceil(block_size).max(1)).unwrap_or(usize::MAX);
    let per_job = encode_mem(opts.level, size.map(|s| (s as usize).min(block_size)).unwrap_or(block_size));
    let (jobs, warn) = plan_jobs(jobs, units, per_job, mem_cap);
    let budget = mem_budget(mem_cap);
    let lane_tables = crate::cm::levels::preset(opts.level, crate::cm::levels::Kind::Generic, opts.lane + opts.lane / 2).memory() + 3 * opts.lane;
    let opts = &opts.with_lane_jobs(((budget / lane_tables as u64) as usize).clamp(1, opts.lane_jobs.max(1)));
    let st = with_output(output, |w| sched::compress_meta(reader, w, opts, jobs, block_size, budget, meta))?;
    Ok((st, warn))
}

/// Read the header and the trailer (from the end of the file) without touching the blocks.
pub fn read_ends(path: &str) -> Result<(format::Header, format::Trailer), String> {
    let mut f = File::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let (h, _) = sched::read_header(&mut f)?;
    let len = f.metadata().map_err(|e| e.to_string())?.len();
    if len < 12 { return Err("truncated NSC3 file".into()); }
    let mut tail = [0u8; 12];
    f.seek(SeekFrom::Start(len - 12)).map_err(|e| e.to_string())?;
    f.read_exact(&mut tail).map_err(|e| e.to_string())?;
    if &tail[4..] != format::TAIL_MAGIC { return Err("bad NSC3 trailer (truncated file?)".into()); }
    let tl = u32::from_le_bytes(tail[..4].try_into().unwrap()) as u64;
    if tl < 4 || tl + 12 > len { return Err("bad NSC3 trailer length".into()); }
    let mut t = vec![0u8; tl as usize + 12];
    f.seek(SeekFrom::Start(len - 12 - tl)).map_err(|e| e.to_string())?;
    f.read_exact(&mut t).map_err(|e| e.to_string())?;
    if &t[..4] != format::END_SYNC { return Err("bad NSC3 trailer".into()); }
    Ok((h, format::parse_trailer(&t[4..])?))
}

pub fn decompress_path(input: &str, output: &str, jobs: usize, mem_cap: Option<u64>, loader: Loader) -> Result<Stats, String> {
    // Memory per worker comes from the file itself (largest block config), when seekable.
    let per_job = if input != "-" { info(input).map(|i| i.mem_per_job as usize + 2 * i.max_block as usize).unwrap_or(256 << 20) } else { 512 << 20 };
    let (jobs, warn) = plan_jobs(jobs, usize::MAX, per_job, mem_cap);
    if let Some(w) = warn { eprintln!("note: {w}"); }
    let reader = open_in(input)?;
    with_output(output, |w| sched::decompress(reader, w, jobs, mem_budget(mem_cap), loader))
}

/// Summary of an NSC3 file, read without decoding any payload.
pub struct Info {
    pub orig_len: u64,
    pub file_len: u64,
    pub blocks: usize,
    pub stored_blocks: usize,
    pub kinds: u128,
    pub max_block: u32,
    pub mem_per_job: u64,
    pub levels_hash_bits: u8,
    pub block_list: Vec<(u64, u32, u32, u8, u8)>, // (raw offset, raw len, comp len, mode, first segment kind)
}

pub fn info(path: &str) -> Result<Info, String> {
    let f = File::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let file_len = f.metadata().map(|m| m.len()).unwrap_or(0);
    let mut r = BufReader::new(f);
    let (_h, hlen) = sched::read_header(&mut r)?;
    let mut pos = hlen as u64;
    let mut inf = Info { orig_len: 0, file_len, blocks: 0, stored_blocks: 0, kinds: 0, max_block: 0, mem_per_job: 0, levels_hash_bits: 0, block_list: vec![] };
    let mut raw_off = 0u64;
    loop {
        let mut sync = [0u8; 4];
        r.read_exact(&mut sync).map_err(|_| "truncated NSC3 file".to_string())?;
        if &sync[..4] == format::END_SYNC {
            r.seek(SeekFrom::Start(pos)).map_err(|e| e.to_string())?;
            match sched::read_unit(&mut r)? {
                sched::Unit::End(t) => { inf.orig_len = t.orig_len; inf.mem_per_job = t.mem_per_job; }
                _ => return Err("corrupt NSC3 trailer".into()),
            }
            break;
        }
        if &sync[..4] != format::BLOCK_SYNC { return Err("corrupt NSC3 stream (bad block marker)".into()); }
        let (bh, hl) = format::read_block_header(&mut r)?;
        inf.blocks += 1;
        if bh.mode == format::MODE_STORED { inf.stored_blocks += 1; }
        for s in &bh.segments { inf.kinds |= format::kind::bit(s.kind); }
        if let Some(c) = bh.cfg { inf.levels_hash_bits = inf.levels_hash_bits.max(c.hash_bits); }
        inf.max_block = inf.max_block.max(bh.raw_len);
        inf.block_list.push((raw_off, bh.raw_len, bh.comp_len, bh.mode, bh.segments[0].kind));
        raw_off += bh.raw_len as u64;
        pos += hl as u64 + bh.comp_len as u64;
        r.seek(SeekFrom::Start(pos)).map_err(|e| e.to_string())?;
    }
    Ok(inf)
}

pub fn kind_names(kinds: u128) -> String {
    let v: Vec<&str> = (0..128u8).filter(|k| kinds >> k & 1 == 1).map(format::kind::name).collect();
    if v.is_empty() { "-".into() } else { v.join(", ") }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }

    fn roundtrip(data: &[u8], level: u32, jobs: usize, block: usize) -> Vec<u8> {
        let mut enc = Vec::new();
        sched::compress(data, &mut enc, &EncOpts::new(level, true, None), jobs, block, u64::MAX).unwrap();
        let mut dec = Vec::new();
        sched::decompress(&enc[..], &mut dec, jobs, u64::MAX, &no_model).unwrap();
        assert!(dec == data, "roundtrip mismatch (len {}, level {level}, jobs {jobs}, block {block})", data.len());
        enc
    }

    #[test]
    fn stream_roundtrip_and_jobs_invariance() {
        let mut s = 3u64;
        let mut data = b"NSC3 test corpus: ".repeat(2000);
        data.extend((0..50_000).map(|_| (rng(&mut s) % 7) as u8));
        data.extend((0..30_000).map(|_| rng(&mut s) as u8));
        for (block, jobs) in [(1 << 20, 1), (7_777, 1), (7_777, 3), (7_777, 16)] {
            let a = roundtrip(&data, 2, jobs, block);
            let b = roundtrip(&data, 2, 1, block);
            assert!(a == b, "output depends on --jobs (block {block})");
        }
        roundtrip(&[], 5, 4, 1 << 20);
        roundtrip(&[42], 5, 4, 1 << 20);
    }

    /// MP4 files: the index (wherever the `moov` box sits) turns into per-block hints, the audio
    /// samples become AAC containers even when blocks cut the track into pieces, the output does
    /// not depend on the block workers, and everything decodes exactly.
    #[test]
    fn mp4_audio_is_recoded_through_the_index() {
        let no_model = |b: &[(String, u64)]| -> Result<Option<std::sync::Arc<NeuralCtx>>, String> { if b.is_empty() { Ok(None) } else { Err("neural".into()) } };
        let files: [(&str, &[u8]); 3] = [("m4a", include_bytes!("../../tests/fixtures/tone.m4a")), ("video", include_bytes!("../../tests/fixtures/tone_video.mp4")),
            ("fragmented", include_bytes!("../../tests/fixtures/tone_fragmented.mp4"))];
        for (name, data) in files {
            let run = |block: usize, jobs: usize, with_index: bool, level: u32| -> (Vec<u8>, sched::Stats) {
                let idx = crate::recode::mp4::index(&mut std::io::Cursor::new(data), data.len() as u64).unwrap();
                let meta = sched::Meta { indexes: if with_index { vec![(0, idx)] } else { vec![] }, ..Default::default() };
                let mut enc = vec![];
                let st = sched::compress_meta(data, &mut enc, &EncOpts::new(level, true, None), jobs, block, u64::MAX, meta).unwrap();
                (enc, st)
            };
            let (enc, st) = run(1 << 20, 1, true, 5);
            assert!(st.kinds & format::kind::bit(format::kind::AAC) != 0, "{name}: no aac container (kinds {:#x})", st.kinds);
            assert!(enc.len() * 100 < data.len() * 88, "{name}: {} -> {}", data.len(), enc.len());
            let mut dec = vec![];
            sched::decompress(&enc[..], &mut dec, 2, u64::MAX, &no_model).unwrap();
            assert!(dec == data, "{name}: roundtrip");
            // without the index the samples cannot be found; at level 2 no recoder runs
            assert!(run(1 << 20, 1, false, 5).1.kinds & format::kind::bit(format::kind::AAC) == 0);
            assert!(run(1 << 20, 1, true, 2).1.kinds & format::kind::bit(format::kind::AAC) == 0);
            // small blocks cut the track: whole samples are still recoded, in every block
            let (small, st) = run(4096, 1, true, 5);
            assert!(st.blocks >= 3 && st.kinds & format::kind::bit(format::kind::AAC) != 0, "{name}: {} blocks", st.blocks);
            assert!(small == run(4096, 3, true, 5).0, "{name}: output depends on --jobs");
            let mut dec = vec![];
            sched::decompress(&small[..], &mut dec, 2, u64::MAX, &no_model).unwrap();
            assert!(dec == data, "{name}: roundtrip with small blocks");
            assert!(small.len() * 100 < data.len() * 96, "{name}: small blocks {} -> {}", data.len(), small.len());
        }
        // two files in one stream (an archive): each index applies at its file's offset
        let (a, b): (&[u8], &[u8]) = (files[1].1, files[0].1);
        let mut both = a.to_vec();
        both.extend_from_slice(b"some text between the two files\n");
        let at = both.len() as u64;
        both.extend_from_slice(b);
        let idx = |d: &[u8]| crate::recode::mp4::index(&mut std::io::Cursor::new(d), d.len() as u64).unwrap();
        let meta = sched::Meta { indexes: vec![(0, idx(a)), (at, idx(b))], ..Default::default() };
        let mut enc = vec![];
        sched::compress_meta(&both[..], &mut enc, &EncOpts::new(5, true, None), 2, 8192, u64::MAX, meta).unwrap();
        let mut dec = vec![];
        sched::decompress(&enc[..], &mut dec, 2, u64::MAX, &no_model).unwrap();
        assert!(dec == both);
        assert!(enc.len() * 100 < both.len() * 92, "archive of two: {} -> {}", both.len(), enc.len());
    }

    /// H.264 video tracks are recoded through the index too: alone (one stream), or with the
    /// audio track around a plan that interleaves them again.
    #[test]
    fn mp4_video_is_recoded_through_the_index() {
        use format::kind::{bit, AAC, H264, H264B, MUX};
        let no_model = |b: &[(String, u64)]| -> Result<Option<std::sync::Arc<NeuralCtx>>, String> { if b.is_empty() { Ok(None) } else { Err("neural".into()) } };
        let files: [(&str, &[u8], bool); 4] = [("video", include_bytes!("../../tests/fixtures/video_cabac.mp4"), false),
            ("video and audio", include_bytes!("../../tests/fixtures/tone_video.mp4"), true),
            ("baseline video", include_bytes!("../../tests/fixtures/video_cavlc.mp4"), false),
            ("high profile CAVLC video", include_bytes!("../../tests/fixtures/video_cavlc_high.mp4"), false)];
        for (name, data, audio) in files {
            let run = |block: usize, jobs: usize, level: u32, compat: bool| -> (Vec<u8>, sched::Stats) {
                let idx = crate::recode::mp4::index(&mut std::io::Cursor::new(data), data.len() as u64).unwrap();
                let mut opts = EncOpts::new(level, true, None);
                opts.compat = compat;
                let mut enc = vec![];
                let st = sched::compress_meta(data, &mut enc, &opts, jobs, block, u64::MAX, sched::Meta { indexes: vec![(0, idx)], ..Default::default() }).unwrap();
                (enc, st)
            };
            let (enc, st) = run(1 << 20, 1, 5, false);
            assert!(st.kinds & bit(H264) != 0 && st.kinds & bit(H264B) != 0, "{name}: no h264 container (kinds {:#x})", st.kinds);
            assert_eq!(st.kinds & bit(MUX) != 0, audio, "{name}: kinds {:#x}", st.kinds);
            assert_eq!(st.kinds & bit(AAC) != 0, audio, "{name}: kinds {:#x}", st.kinds);
            assert!(enc.len() * 100 < data.len() * 93, "{name}: {} -> {}", data.len(), enc.len());
            let mut dec = vec![];
            sched::decompress(&enc[..], &mut dec, 2, u64::MAX, &no_model).unwrap();
            assert!(dec == data, "{name}: roundtrip");
            // the lean profile of levels 3 and 4 decodes too
            let (lean, st) = run(1 << 20, 1, 3, false);
            assert!(st.kinds & bit(H264B) != 0 && lean != enc, "{name}: level 3");
            let mut dec = vec![];
            sched::decompress(&lean[..], &mut dec, 2, u64::MAX, &no_model).unwrap();
            assert!(dec == data, "{name}: roundtrip at level 3");
            // not below level 3, nor for an older decoder
            assert!(run(1 << 20, 1, 2, false).1.kinds & bit(H264) == 0);
            assert!(run(1 << 20, 1, 5, true).1.kinds & bit(H264) == 0);
            // small blocks cut the track: each block's whole samples are a stream of their own
            let (small, st) = run(3000, 1, 5, false);
            assert!(st.blocks >= 3 && st.kinds & bit(H264) != 0, "{name}: {} blocks", st.blocks);
            assert!(small == run(3000, 3, 5, false).0, "{name}: output depends on --jobs");
            let mut dec = vec![];
            sched::decompress(&small[..], &mut dec, 2, u64::MAX, &no_model).unwrap();
            assert!(dec == data, "{name}: roundtrip with small blocks");
            // damaged files: an error, or (a flip in padding) the same bytes
            let mut s = 17u64;
            for _ in 0..120 {
                let mut bad = enc.clone();
                let i = (rng(&mut s) as usize) % bad.len();
                bad[i] ^= 1 << (rng(&mut s) % 8);
                let mut out = Vec::new();
                if sched::decompress(&bad[..], &mut out, 2, u64::MAX, &no_model).is_ok() { assert!(out == data, "{name}: silent corruption at byte {i}"); }
            }
        }
    }

    /// Matroska files: the walk of their clusters gives the hints an MP4 file's sample tables
    /// give, for H.264, AAC and MP3 tracks, a frame to a block or several laced together.
    #[test]
    fn matroska_tracks_are_recoded_through_the_index() {
        use format::kind::{bit, AAC, H264, H264B, MP3, MUX};
        use crate::recode::mkv::tests::{mp3_file, mp3_frames, FILE};
        let no_model = |b: &[(String, u64)]| -> Result<Option<std::sync::Arc<NeuralCtx>>, String> { if b.is_empty() { Ok(None) } else { Err("neural".into()) } };
        let frames = mp3_frames(include_bytes!("../../tests/fixtures/tone_stereo.mp3"));
        let laced = [mp3_file(&frames, 3, 4, true, true).0, mp3_file(&frames, 1, 3, false, false).0, mp3_file(&frames, 2, 2, true, false).0];
        let all = bit(H264) | bit(H264B) | bit(AAC) | bit(MP3) | bit(MUX);
        let files: [(&str, &[u8], u128, usize); 4] = [("three tracks", FILE, all, 80), ("mp3, EBML lacing", &laced[0], bit(MP3) | bit(MUX), 90),
            ("mp3, Xiph lacing", &laced[1], bit(MP3) | bit(MUX), 90), ("mp3, fixed lacing", &laced[2], bit(MP3) | bit(MUX), 90)];
        for (name, data, kinds, most) in files {
            let run = |block: usize, jobs: usize, level: u32, compat: bool| -> (Vec<u8>, sched::Stats) {
                let idx = crate::recode::index(&mut std::io::Cursor::new(data), data.len() as u64).unwrap_or_else(|| panic!("{name}: no index"));
                let mut opts = EncOpts::new(level, true, None);
                opts.compat = compat;
                let mut enc = vec![];
                let st = sched::compress_meta(data, &mut enc, &opts, jobs, block, u64::MAX, sched::Meta { indexes: vec![(0, idx)], ..Default::default() }).unwrap();
                (enc, st)
            };
            let (enc, st) = run(1 << 20, 1, 5, false);
            assert!(st.kinds & kinds == kinds, "{name}: kinds {:#x}, expected {kinds:#x}", st.kinds);
            assert!(enc.len() * 100 < data.len() * most, "{name}: {} -> {}", data.len(), enc.len());
            let mut dec = vec![];
            sched::decompress(&enc[..], &mut dec, 2, u64::MAX, &no_model).unwrap();
            assert!(dec == data, "{name}: roundtrip");
            // not below level 3, nor for an older decoder
            assert!(run(1 << 20, 1, 2, false).1.kinds & all == 0, "{name}: level 2");
            assert!(run(1 << 20, 1, 5, true).1.kinds & all == 0, "{name}: an older decoder");
            // small blocks cut the tracks: each block's whole frames are streams of their own
            let (small, st) = run(6000, 1, 5, false);
            assert!(st.blocks >= 3 && st.kinds & (bit(AAC) | bit(MP3)) != 0, "{name}: {} blocks, kinds {:#x}", st.blocks, st.kinds);
            assert!(small == run(6000, 3, 5, false).0, "{name}: output depends on --jobs");
            let mut dec = vec![];
            sched::decompress(&small[..], &mut dec, 2, u64::MAX, &no_model).unwrap();
            assert!(dec == data, "{name}: roundtrip with small blocks");
            // damaged files: an error, or (a flip in padding) the same bytes
            let mut s = 29u64;
            for _ in 0..80 {
                let mut bad = enc.clone();
                let i = (rng(&mut s) as usize) % bad.len();
                bad[i] ^= 1 << (rng(&mut s) % 8);
                let mut out = Vec::new();
                if sched::decompress(&bad[..], &mut out, 2, u64::MAX, &no_model).is_ok() { assert!(out == data, "{name}: silent corruption at byte {i}"); }
            }
        }
    }

    #[test]
    fn corruption_is_detected_never_panics() {
        let data = b"corruption sweep payload 0123456789 ".repeat(3000);
        let mut enc = Vec::new();
        sched::compress(&data[..], &mut enc, &EncOpts::new(1, true, None), 2, 16_384, u64::MAX).unwrap();
        let mut s = 9u64;
        // bit flips everywhere: must error or (if a flip lands in an unused padding bit) still match exactly
        for _ in 0..300 {
            let mut bad = enc.clone();
            let i = (rng(&mut s) as usize) % bad.len();
            bad[i] ^= 1 << (rng(&mut s) % 8);
            let mut out = Vec::new();
            match sched::decompress(&bad[..], &mut out, 2, u64::MAX, &no_model) {
                Ok(_) => assert!(out == data, "silent corruption at byte {i}"),
                Err(_) => {}
            }
        }
        // truncation at every 97th byte
        for cut in (0..enc.len()).step_by(97) {
            let mut out = Vec::new();
            assert!(sched::decompress(&enc[..cut], &mut out, 2, u64::MAX, &no_model).is_err(), "truncated at {cut} decoded without error");
        }
    }
}
