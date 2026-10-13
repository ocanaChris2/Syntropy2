//! Streaming, block-parallel compression and decompression.
//!
//! Both directions use one pipeline. The main thread reads blocks sequentially and hands them to a
//! pool of workers. It then writes the results strictly in input order, holding at most `jobs + 1`
//! blocks in flight (bounded memory). The output therefore never depends on the number of workers.
use super::block::{self, EncOpts, Encoded};
use super::format::{self, kind, BlockHeader, Header, Trailer, BLOCK_SYNC, END_SYNC};
use super::plan::{PlannedBlock, Planner};
use crate::bench::read_full;
use crate::crc::Crc32;
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

#[derive(Default, Debug)]
pub struct Stats {
    pub blocks: usize,
    pub raw_bytes: u64,
    pub out_bytes: u64,
    pub stored_blocks: usize,
    pub verify_failures: usize,
    pub mem_per_job: usize,
    pub kinds: u128,
}

fn wr<W: Write>(w: &mut W, b: &[u8]) -> Result<(), String> { w.write_all(b).map_err(|e| format!("write: {e}")) }

/// Worker pool: run `f` on every job. Returns (job sender, result receiver); dropping the sender
/// stops the workers once the queue drains.
fn pool<'s, J: Send + 's, R: Send + 's>(
    scope: &'s std::thread::Scope<'s, '_>, jobs: usize, f: impl Fn(J) -> R + Send + Sync + 's,
) -> (mpsc::SyncSender<(usize, J)>, mpsc::Receiver<(usize, R)>) {
    let (jtx, jrx) = mpsc::sync_channel::<(usize, J)>(jobs);
    let (rtx, rrx) = mpsc::channel::<(usize, R)>();
    let jrx = Arc::new(Mutex::new(jrx));
    let f = Arc::new(f);
    for _ in 0..jobs {
        let jrx = Arc::clone(&jrx);
        let rtx = rtx.clone();
        let f = Arc::clone(&f);
        scope.spawn(move || loop {
            let job = { jrx.lock().unwrap().recv() };
            match job {
                Ok((i, j)) => { if rtx.send((i, f(j))).is_err() { break; } }
                Err(_) => break,
            }
        });
    }
    (jtx, rrx)
}

/// Estimated memory to encode a planned block: model tables (one set per lane coded at once, two
/// when each lane is verified concurrently), raw and work buffers, and the verify copies.
fn encode_mem(opts: &EncOpts, pb: &PlannedBlock) -> usize {
    let work = pb.data.len() + pb.inner_est;
    let lane = opts.lane.max(1 << 20);
    let (lanes, lane_len) = if work <= lane + lane / 2 { (1, work) } else { (work.div_ceil(lane), lane + lane / 2) };
    let tables = crate::cm::levels::preset(opts.level, crate::cm::levels::Kind::Generic, lane_len.max(1)).memory();
    let (pipelined, sets, conc) = opts.lane_plan();
    lanes.min(conc) * sets * tables + 2 * pb.data.len() + 2 * work + if pipelined { work } else { 0 }
}

/// Stream-level metadata: header flags and the trailer's file table (archives).
#[derive(Default)]
pub struct Meta {
    pub flags: u32,
    pub file_table: Vec<u8>,
    /// Indexes of the container files in the stream (stream offset, index), in stream order.
    pub indexes: Vec<(u64, crate::recode::mp4::Index)>,
}

/// Compress `input` to `out`: plan blocks of `block_size` bytes (detector-typed segments), encode
/// them in parallel, write them in order. At most `jobs` blocks are in flight, and fewer when
/// their estimated memory would exceed `mem_budget` bytes (never fewer than one).
pub fn compress<R: Read, W: Write>(input: R, out: W, opts: &EncOpts, jobs: usize, block_size: usize, mem_budget: u64) -> Result<Stats, String> {
    compress_meta(input, out, opts, jobs, block_size, mem_budget, Meta::default())
}

/// [`compress`] with header flags and a file table.
pub fn compress_meta<R: Read, W: Write>(input: R, mut out: W, opts: &EncOpts, jobs: usize, block_size: usize, mem_budget: u64, meta: Meta) -> Result<Stats, String> {
    let bindings = opts.neural.as_ref().map(|nc| vec![(nc.id.clone(), nc.fingerprint)]).unwrap_or_default();
    let (features, min_decoder) = format::writer_features(opts.specialists());
    let header = format::write_header(&Header { flags: meta.flags, features, min_decoder, bindings });
    wr(&mut out, &header)?;
    let mut st = Stats { out_bytes: header.len() as u64, ..Default::default() };
    let mut index: Vec<(u64, u64)> = vec![];
    let mut crc = Crc32::new();
    // a block verified while it is coded runs two threads (coder and verifier): half the blocks
    let jobs = if opts.lane_plan().0 { jobs.max(1).div_ceil(2) } else { jobs.max(1) };
    let mut raw_written = 0u64;
    let mut planner = Planner::new(input, block_size);
    planner.extend = super::extend_cap(opts.level, block_size);
    planner.recode = opts.specialists();
    if opts.specialists() {
        for (at, idx) in meta.indexes { planner.add_index(at, idx); }
    }
    std::thread::scope(|s| -> Result<(), String> {
        let (jtx, rrx) = pool(s, jobs, |pb: PlannedBlock| { let _slot = opts.threads.hold(1); block::encode(&pb, opts) });
        let mut pending: BTreeMap<usize, Encoded> = BTreeMap::new();
        let (mut next_read, mut next_write, mut inflight, mut eof) = (0usize, 0usize, 0usize, false);
        let (mut held, mut mem_inflight, mut mem_of) = (None::<PlannedBlock>, 0u64, BTreeMap::<usize, u64>::new());
        loop {
            while !eof && inflight <= jobs {
                if held.is_none() {
                    match planner.next_block()? {
                        None => { eof = true; break; }
                        Some(pb) => held = Some(pb),
                    }
                }
                let est = encode_mem(opts, held.as_ref().unwrap()) as u64;
                if inflight > 0 && mem_inflight + est > mem_budget { break; }
                let pb = held.take().unwrap();
                crc.update(&pb.data);
                st.raw_bytes += pb.data.len() as u64;
                if jtx.send((next_read, pb)).is_err() { return Err("worker pool stopped".into()); }
                mem_of.insert(next_read, est);
                mem_inflight += est;
                next_read += 1;
                inflight += 1;
            }
            if inflight == 0 { break; }
            let (i, enc) = rrx.recv().map_err(|_| "worker pool stopped".to_string())?;
            inflight -= 1;
            mem_inflight -= mem_of.remove(&i).unwrap_or(0);
            pending.insert(i, enc);
            while let Some(e) = pending.remove(&next_write) {
                index.push((st.out_bytes, raw_written));
                wr(&mut out, &e.bytes)?;
                st.out_bytes += e.bytes.len() as u64;
                raw_written += e.raw_len as u64;
                st.blocks += 1;
                if e.mode == format::MODE_STORED { st.stored_blocks += 1; }
                if e.verify_failed { st.verify_failures += 1; }
                st.mem_per_job = st.mem_per_job.max(e.mem);
                st.kinds |= e.kinds;
                next_write += 1;
            }
        }
        drop(jtx);
        Ok(())
    })?;
    let trailer = format::write_trailer(&Trailer {
        orig_len: st.raw_bytes, file_crc: crc.finish(), mem_per_job: st.mem_per_job as u64, index, file_table: meta.file_table,
    });
    wr(&mut out, &trailer)?;
    st.out_bytes += trailer.len() as u64;
    out.flush().map_err(|e| format!("flush: {e}"))?;
    Ok(st)
}

fn read_exact<R: Read>(r: &mut R, n: usize, what: &str) -> Result<Vec<u8>, String> {
    let mut b = vec![0u8; n];
    let got = read_full(r, &mut b).map_err(|e| format!("read: {e}"))?;
    if got != n { return Err(format!("truncated NSC3 stream ({what})")); }
    Ok(b)
}

/// Read and validate the header from a stream; returns it and its length.
pub fn read_header<R: Read>(r: &mut R) -> Result<(Header, usize), String> {
    let mut h = read_exact(r, 20, "header")?;
    let n = u16::from_le_bytes([h[18], h[19]]) as usize;
    if n > 64 { return Err("too many model bindings".into()); }
    for _ in 0..n {
        let l = read_exact(r, 2, "header")?;
        let len = u16::from_le_bytes([l[0], l[1]]) as usize;
        if len > 256 { return Err("model id too long".into()); }
        h.extend_from_slice(&l);
        h.extend_from_slice(&read_exact(r, len + 8, "header")?);
    }
    h.extend_from_slice(&read_exact(r, 4, "header")?);
    format::parse_header(&h)
}

/// Read the next unit after the header: a block (header + payload) or the trailer.
pub enum Unit {
    Block(BlockHeader, Vec<u8>),
    End(Trailer),
}

pub fn read_unit<R: Read>(r: &mut R) -> Result<Unit, String> {
    let sync = read_exact(r, 4, "block sync")?;
    if sync == BLOCK_SYNC {
        let (bh, _) = format::read_block_header(r)?;
        let payload = read_exact(r, bh.comp_len as usize, "block payload")?;
        Ok(Unit::Block(bh, payload))
    } else if sync == END_SYNC {
        Ok(Unit::End(format::read_trailer(r)?))
    } else {
        Err("corrupt NSC3 stream (bad block marker)".into())
    }
}

/// Loads the neural model a file binds (by id + fingerprint); `Ok(None)` when it binds none.
pub type Loader<'a> = &'a dyn Fn(&[(String, u64)]) -> Result<Option<std::sync::Arc<block::NeuralCtx>>, String>;

/// Decompress an NSC3 stream from `input` to `out`, with at most `jobs` blocks in flight and fewer
/// when their estimated memory would exceed `mem_budget` bytes (never fewer than one).
pub fn decompress<R: Read, W: Write>(mut input: R, mut out: W, jobs: usize, mem_budget: u64, loader: Loader) -> Result<Stats, String> {
    let lane_jobs = jobs.max(1);
    let (hdr, hlen) = read_header(&mut input)?;
    let neural = if hdr.bindings.is_empty() { None } else { loader(&hdr.bindings)? };
    let mut st = Stats { out_bytes: hlen as u64, ..Default::default() };
    let mut crc = Crc32::new();
    let jobs = jobs.max(1);
    let mut trailer: Option<Trailer> = None;
    std::thread::scope(|s| -> Result<(), String> {
        let nref = neural.as_deref();
        let (jtx, rrx) = pool(s, jobs, move |(bh, payload): (BlockHeader, Vec<u8>)| {
            // lanes decoded at once: as many model table sets as fit the memory budget
            let tables = bh.cfg.map(|c| c.memory() as u64).unwrap_or(1).max(1);
            let lj = ((mem_budget / tables) as usize).clamp(1, lane_jobs);
            block::decode(&bh, &payload, nref, lj)
        });
        let mut pending: BTreeMap<usize, Result<Vec<u8>, String>> = BTreeMap::new();
        let (mut next_read, mut next_write, mut inflight, mut done) = (0usize, 0usize, 0usize, false);
        let (mut held, mut mem_inflight, mut mem_of) = (None::<Unit>, 0u64, BTreeMap::<usize, u64>::new());
        loop {
            while !done && inflight <= jobs {
                let unit = match held.take() { Some(u) => u, None => read_unit(&mut input)? };
                if let Unit::Block(bh, _) = &unit {
                    let est = block::decode_mem(bh) as u64;
                    if inflight > 0 && mem_inflight + est > mem_budget { held = Some(unit); break; }
                    mem_of.insert(next_read, est);
                    mem_inflight += est;
                }
                match unit {
                    Unit::Block(bh, payload) => {
                        st.out_bytes += payload.len() as u64;
                        if bh.mode == format::MODE_STORED { st.stored_blocks += 1; }
                        if let Some(c) = bh.cfg { st.mem_per_job = st.mem_per_job.max(c.memory()); }
                        for sgm in &bh.segments { st.kinds |= kind::bit(sgm.kind); }
                        if jtx.send((next_read, (bh, payload))).is_err() { return Err("worker pool stopped".into()); }
                        next_read += 1;
                        inflight += 1;
                    }
                    Unit::End(t) => { trailer = Some(t); done = true; }
                }
            }
            if inflight == 0 { break; }
            let (i, r) = rrx.recv().map_err(|_| "worker pool stopped".to_string())?;
            inflight -= 1;
            mem_inflight -= mem_of.remove(&i).unwrap_or(0);
            pending.insert(i, r);
            while let Some(r) = pending.remove(&next_write) {
                let data = r?;
                crc.update(&data);
                wr(&mut out, &data)?;
                st.raw_bytes += data.len() as u64;
                st.blocks += 1;
                next_write += 1;
            }
        }
        drop(jtx);
        Ok(())
    })?;
    let t = trailer.ok_or("NSC3 stream ended without a trailer")?;
    if t.orig_len != st.raw_bytes { return Err(format!("length mismatch: trailer says {} bytes, decoded {}", t.orig_len, st.raw_bytes)); }
    if t.file_crc != crc.finish() { return Err("whole-file checksum mismatch: the file is corrupt".into()); }
    if t.index.len() != st.blocks { return Err("block count mismatch with trailer index".into()); }
    let mut extra = [0u8; 1];
    if read_full(&mut input, &mut extra).map_err(|e| format!("read: {e}"))? != 0 {
        return Err("unexpected data after the NSC3 trailer".into());
    }
    out.flush().map_err(|e| format!("flush: {e}"))?;
    Ok(st)
}
