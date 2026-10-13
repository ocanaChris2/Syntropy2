//! Shared codec core: little-endian byte readers, probability quantizer, and the
//! `encode_tokens`/`decode_tokens` drivers used by both the CLI (`main.rs`) and the parity
//! harness (`selftest.rs`). Byte-exact by construction: the identical int8 forward + feature
//! model drive encode and decode, so their coder tables match bit-for-bit.
use crate::artifacts::Model;
use crate::features::{FeatureModel, O2Table};
use crate::forward;
use crate::range_coder::{RangeDecoder, RangeEncoder};
use std::fs;

pub const QUANT_BITS: u32 = 22;

// ---- little-endian readers over an in-memory buffer (o = running offset) ----
pub fn ru16(b: &[u8], o: &mut usize) -> u16 { let v = u16::from_le_bytes(b[*o..*o + 2].try_into().unwrap()); *o += 2; v }
pub fn ru32(b: &[u8], o: &mut usize) -> u32 { let v = u32::from_le_bytes(b[*o..*o + 4].try_into().unwrap()); *o += 4; v }
pub fn ru64(b: &[u8], o: &mut usize) -> u64 { let v = u64::from_le_bytes(b[*o..*o + 8].try_into().unwrap()); *o += 8; v }
pub fn ri32(b: &[u8], o: &mut usize) -> i32 { let v = i32::from_le_bytes(b[*o..*o + 4].try_into().unwrap()); *o += 4; v }
pub fn rf64(b: &[u8], o: &mut usize) -> f64 { let v = f64::from_le_bytes(b[*o..*o + 8].try_into().unwrap()); *o += 8; v }
pub fn rf32(b: &[u8], o: &mut usize) -> f32 { let v = f32::from_le_bytes(b[*o..*o + 4].try_into().unwrap()); *o += 4; v }

fn argmax(f: &[i64]) -> usize { let mut b = 0; for i in 1..f.len() { if f[i] > f[b] { b = i; } } b }

/// Port of nsc._quantize(p, total_bits): floor(p*total) clamped>=1, renormalized to `total`, cumsum.
pub fn quantize(p: &[f32], total_bits: u32) -> Vec<i64> {
    let v = p.len();
    let mut total: i64 = 1 << total_bits;
    while total < v as i64 { total <<= 1; }
    let mut f: Vec<i64> = p.iter().map(|&x| ((x as f64 * total as f64).floor() as i64).max(1)).collect();
    let diff = total - f.iter().sum::<i64>();
    if diff != 0 {
        let j = argmax(&f); f[j] = (f[j] + diff).max(1);
        let diff2 = total - f.iter().sum::<i64>();
        if diff2 != 0 { let j = argmax(&f); f[j] = (f[j] + diff2).max(1); }
    }
    let mut cum = vec![0i64; v + 1];
    for s in 0..v { cum[s + 1] = cum[s] + f[s]; }
    cum
}

// ---- .ids token-id files: magic "NIDS" | n u32 | n x token-id u32 (shared with tok.py) ----
pub fn read_ids(path: &str) -> Result<Vec<usize>, String> {
    let b = fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
    if b.len() < 8 || &b[0..4] != b"NIDS" { return Err(format!("bad .ids magic in {path}")); }
    let mut o = 4usize;
    let n = ru32(&b, &mut o) as usize;
    if b.len() < 8 + n * 4 { return Err(format!("truncated .ids in {path}")); }
    Ok((0..n).map(|_| ru32(&b, &mut o) as usize).collect())
}

pub fn write_ids(path: &str, ids: &[usize]) -> Result<(), String> {
    let mut out = Vec::with_capacity(8 + ids.len() * 4);
    out.extend_from_slice(b"NIDS");
    out.extend_from_slice(&(ids.len() as u32).to_le_bytes());
    for &t in ids { out.extend_from_slice(&(t as u32).to_le_bytes()); }
    fs::write(path, out).map_err(|e| format!("write {path}: {e}"))
}

/// Drive the int8 forward + range coder over a token stream. The neural window resets every
/// `seg_len` tokens (matching Python's `if i % seg == 0: reset_window()`); the feature model runs
/// continuously. Returns (payload, n_bits). The first token is stored raw by the caller; only
/// tokens[1..] are entropy-coded.
///
/// The expensive neural forward (~96% of the cost) is window-local — the [`forward::Decoder`] is
/// reset every `seg_len` tokens, so no window depends on any earlier window's neural state. Because
/// encoding also knows the whole token stream up front, every window's contribution to the range
/// coder (the `(cum[next], cum[next+1], total)` triple per position) can be computed independently
/// and in parallel with `jobs` worker threads, then the (cheap) range coding is replayed
/// sequentially. This is byte-exact: identical arithmetic in identical order into the coder. Pass
/// `jobs == 1` to force the plain single-threaded loop (e.g. when the caller already parallelises
/// across files, as `batch` does).
pub fn encode_tokens(model: &Model, o2: &O2Table, tokens: &[usize], jobs: usize) -> (Vec<u8>, usize) {
    encode_tokens_m::<forward::Libm>(model, o2, tokens, jobs)
}

/// [`encode_tokens`] with an explicit math backend (`forward::Pure` for the portable NSC3 kind).
pub fn encode_tokens_m<M: forward::Math>(model: &Model, o2: &O2Table, tokens: &[usize], jobs: usize) -> (Vec<u8>, usize) {
    let (v, seg) = (model.cfg.vocab_size, model.cfg.seg_len);
    let n = tokens.len();

    // Feature tuples are stateful across the whole stream but cheap; compute them sequentially once.
    let mut feat = FeatureModel::new(o2);
    let feats: Vec<(usize, f32, usize, f32)> = tokens.iter().map(|&t| feat.push(t)).collect();

    // Per-position range-coder input: (cum[next], cum[next+1], total). The final position is never
    // encoded (it has no successor) so its triple stays the harmless (0, 0, 0).
    let mut triples: Vec<(u64, u64, u64)> = vec![(0, 0, 0); n];
    let fill_window = |w: usize, out: &mut [(u64, u64, u64)]| {
        let start = w * seg;
        let mut dec = forward::DecoderM::<M>::new(model);
        for (off, slot) in out.iter_mut().enumerate() {
            let i = start + off;
            let (o2t, o2c, mt, mc) = feats[i];
            let probs = dec.step(tokens[i], o2t, o2c, mt, mc);
            if i + 1 < n {
                let cum = quantize(&probs, QUANT_BITS);
                let s = tokens[i + 1];
                *slot = (cum[s] as u64, cum[s + 1] as u64, cum[v] as u64);
            }
        }
    };
    par_windows(&mut triples, seg, jobs, fill_window);

    let mut enc = RangeEncoder::new();
    for i in 0..n.saturating_sub(1) {
        let (lo, hi, tot) = triples[i];
        enc.encode(lo, hi, tot);
    }
    enc.finish()
}

/// Run `f(w, chunk)` on each `seg`-sized chunk `w` of `out` with up to `jobs` threads (the chunks
/// are disjoint, so the result does not depend on `jobs`).
fn par_windows<T: Send>(out: &mut [T], seg: usize, jobs: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let jobs = jobs.max(1).min(out.len().div_ceil(seg.max(1)).max(1));
    if jobs <= 1 {
        for (w, chunk) in out.chunks_mut(seg).enumerate() { f(w, chunk); }
        return;
    }
    // Deal the disjoint per-window mutable slices round-robin into per-thread buckets, then run one
    // scoped thread per bucket. `chunks_mut(seg)` yields exactly one window per chunk.
    let mut buckets: Vec<Vec<(usize, &mut [T])>> = (0..jobs).map(|_| Vec::new()).collect();
    for (w, chunk) in out.chunks_mut(seg).enumerate() { buckets[w % jobs].push((w, chunk)); }
    std::thread::scope(|scope| {
        for bucket in buckets {
            let f = &f;
            scope.spawn(move || { for (w, chunk) in bucket { f(w, chunk); } });
        }
    });
}

/// Prove that [`decode_tokens_m`]`(model, o2, tokens[0], tokens.len(), payload, nbits)` returns
/// exactly `tokens`, without the sequential decode.
///
/// The decoder's model is a pure function of the prefix it has decoded. Given the true tokens, the
/// coder interval of every position can therefore be recomputed up front: each window (the decoder
/// resets its neural state every `seg_len` tokens) is evaluated independently, in parallel, with the
/// decoder's own loop — a fresh feature model fed the prefix, a fresh window decoder, the same
/// quantizer. The range decoder is then replayed over those intervals; [`RangeDecoder::check`]
/// confirms that each one holds the decoder's target, i.e. that `decode` returns the true token
/// there. By induction over positions, that is exactly what a full decode would establish.
pub fn verify_tokens_m<M: forward::Math>(model: &Model, o2: &O2Table, tokens: &[usize], payload: &[u8], nbits: usize, jobs: usize) -> bool {
    let (v, seg) = (model.cfg.vocab_size, model.cfg.seg_len);
    let n = tokens.len();
    if n == 0 || tokens.iter().any(|&t| t >= v) { return false; }
    let steps = n - 1; // positions the decoder codes: each predicts the token after it
    // the features the decoder computes: it pushes every token it has decoded but the last
    let mut feat = FeatureModel::new(o2);
    let feats: Vec<(usize, f32, usize, f32)> = tokens[..steps].iter().map(|&t| feat.push(t)).collect();
    let mut intervals: Vec<(u64, u64, u64)> = vec![(0, 0, 0); steps];
    par_windows(&mut intervals, seg, jobs, |w, out| {
        let mut dec = forward::DecoderM::<M>::new(model);
        for (off, slot) in out.iter_mut().enumerate() {
            let i = w * seg + off;
            let (o2t, o2c, mt, mc) = feats[i];
            let cum = quantize(&dec.step(tokens[i], o2t, o2c, mt, mc), QUANT_BITS);
            let s = tokens[i + 1];
            *slot = (cum[s] as u64, cum[s + 1] as u64, cum[v] as u64);
        }
    });
    let mut rdec = RangeDecoder::new(payload.to_vec(), nbits);
    intervals.iter().all(|&(lo, hi, tot)| rdec.check(lo, hi, tot))
}

/// Code length in bits the model spends on the windows `wins` (window index `w` covers the
/// predictions made at positions `w*seg_len ..`). Used to estimate, from a few windows, what the
/// whole stream would cost.
pub fn window_bits_m<M: forward::Math>(model: &Model, o2: &O2Table, tokens: &[usize], wins: &[usize], jobs: usize) -> f64 {
    let (v, seg) = (model.cfg.vocab_size, model.cfg.seg_len);
    let steps = tokens.len().saturating_sub(1);
    let last = wins.iter().map(|&w| ((w + 1) * seg).min(steps)).max().unwrap_or(0);
    let mut feat = FeatureModel::new(o2);
    let feats: Vec<(usize, f32, usize, f32)> = tokens[..last].iter().map(|&t| feat.push(t)).collect();
    let mut bits: Vec<f64> = vec![0.0; wins.len()];
    par_windows(&mut bits, 1, jobs, |k, out| {
        let w = wins[k];
        let mut dec = forward::DecoderM::<M>::new(model);
        for i in w * seg..((w + 1) * seg).min(steps) {
            let (o2t, o2c, mt, mc) = feats[i];
            let cum = quantize(&dec.step(tokens[i], o2t, o2c, mt, mc), QUANT_BITS);
            let s = tokens[i + 1];
            out[0] += ((cum[v] as f64) / ((cum[s + 1] - cum[s]) as f64)).log2();
        }
    });
    bits.iter().sum()
}

/// Inverse of `encode_tokens`: the same forward drives the decoder, so the coder tables match
/// bit-for-bit and every token is recovered. Byte-exact by construction.
pub fn decode_tokens(model: &Model, o2: &O2Table, first: usize, n: usize, payload: Vec<u8>, nbits: usize) -> Vec<usize> {
    decode_tokens_m::<forward::Libm>(model, o2, first, n, payload, nbits)
}

/// [`decode_tokens`] with an explicit math backend.
pub fn decode_tokens_m<M: forward::Math>(model: &Model, o2: &O2Table, first: usize, n: usize, payload: Vec<u8>, nbits: usize) -> Vec<usize> {
    let seg = model.cfg.seg_len;
    let mut feat = FeatureModel::new(o2);
    let mut dec = forward::DecoderM::<M>::new(model);
    let mut rdec = RangeDecoder::new(payload, nbits);
    let mut rec: Vec<usize> = Vec::with_capacity(n);
    rec.push(first);
    for i in 0..n.saturating_sub(1) {
        if i % seg == 0 { dec = forward::DecoderM::<M>::new(model); }
        let (o2t, o2c, mt, mc) = feat.push(rec[i]);
        let probs = dec.step(rec[i], o2t, o2c, mt, mc);
        let cum: Vec<u64> = quantize(&probs, QUANT_BITS).iter().map(|&x| x as u64).collect();
        rec.push(rdec.decode(&cum));
    }
    rec
}
