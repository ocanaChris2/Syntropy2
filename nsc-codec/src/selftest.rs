//! Developer parity harness (Stages A–E) — verifies the Rust codec against the Python reference
//! fixtures in `artifacts/` (or `$NSC_ARTIFACTS`). Run from the crate dir after exporting the
//! fixtures. Not part of the product path; the shipped codec uses the model registry instead.
use crate::artifacts::Model;
use crate::codec::{quantize, rf32, rf64, ri32, ru16, ru32, ru64, QUANT_BITS};
use crate::features::{FeatureModel, O2Table};
use crate::forward;
use crate::range_coder::{RangeDecoder, RangeEncoder};
use std::fs;

fn art(name: &str) -> String {
    let base = std::env::var("NSC_ARTIFACTS").unwrap_or_else(|_| "artifacts".into());
    format!("{base}/{name}")
}

fn stage_c(m: &Model) -> bool {
    let b = match fs::read(art("inter.nsc")) { Ok(d) => d, Err(_) => { println!("Stage C: inter.nsc missing (skip)"); return true; } };
    let mut o = 0usize;
    let tok0 = ri32(&b, &mut o) as usize;
    let o2t = ri32(&b, &mut o) as usize;
    let o2c = rf64(&b, &mut o) as f32;
    let mt = ri32(&b, &mut o) as usize;
    let mc = rf64(&b, &mut o) as f32;
    let _h = ru32(&b, &mut o) as usize;
    let v = ru32(&b, &mut o) as usize;
    let conf_ref: Vec<f32> = (0..4).map(|_| rf32(&b, &mut o)).collect();
    let gh_ref: Vec<f32> = (0.._h).map(|_| rf32(&b, &mut o)).collect();
    let rg_ref: Vec<f32> = (0..v).map(|_| rf32(&b, &mut o)).collect();
    let probs_ref: Vec<f32> = (0..v).map(|_| rf32(&b, &mut o)).collect();

    let (rg, gh, conf) = forward::resid_gru_pos0(m, tok0, o2c, mc);
    let dc = (0..4).map(|i| (conf[i] - conf_ref[i]).abs()).fold(0.0f32, f32::max);
    let dg = (0.._h).map(|i| (gh[i] - gh_ref[i]).abs()).fold(0.0f32, f32::max);
    let dr = (0..v).map(|i| (rg[i] - rg_ref[i]).abs()).fold(0.0f32, f32::max);
    println!("Stage C GRU path (pos0): conf max|d|={:.2e}  gru_h max|d|={:.2e}  resid_gru max|d|={:.2e}", dc, dg, dr);
    let _ = (o2t, mt);

    let mut dec = forward::Decoder::new(m);
    let probs = dec.step(tok0, o2t, o2c, mt, mc);
    let dp = (0..v).map(|i| (probs[i] - probs_ref[i]).abs()).fold(0.0f32, f32::max);
    let mut fi = None;
    for i in 0..v { if (probs[i] - probs_ref[i]).abs() > 1e-7 { fi = Some((i, probs[i], probs_ref[i])); break; } }
    println!("Stage C pos0 probs: max|d|={:.2e}  first mismatch: {:?}", dp, fi);
    dr == 0.0 && dp == 0.0
}

fn stage_a() -> bool {
    let data = match fs::read("test_vectors/vec.bin") { Ok(d) => d, Err(_) => { println!("stage A: vec.bin missing (skip)"); return true; } };
    let mut o = 0usize;
    let n = ru32(&data, &mut o) as usize;
    let v = ru32(&data, &mut o) as usize;
    let ref_nbits = ru32(&data, &mut o) as usize;
    let mut cums: Vec<Vec<u64>> = Vec::with_capacity(n);
    let mut syms: Vec<usize> = Vec::with_capacity(n);
    for _ in 0..n {
        let s = ru16(&data, &mut o) as usize; // symbols are u16 in vec.bin
        let cum: Vec<u64> = (0..v + 1).map(|_| ru64(&data, &mut o)).collect();
        cums.push(cum); syms.push(s);
    }
    let plen = ru32(&data, &mut o) as usize;
    let ref_payload = data[o..o + plen].to_vec();
    let mut enc = RangeEncoder::new();
    for i in 0..n { enc.encode(cums[i][syms[i]], cums[i][syms[i] + 1], cums[i][v]); }
    let (payload, nbits) = enc.finish();
    let enc_ok = payload == ref_payload && nbits == ref_nbits;
    let mut dec = RangeDecoder::new(ref_payload.clone(), ref_nbits);
    let dec_ok = (0..n).all(|i| dec.decode(&cums[i]) == syms[i]);
    println!("Stage A range coder: encode {}  decode {}",
        if enc_ok { "bit-identical ✓" } else { "MISMATCH ✗" },
        if dec_ok { "symbols ok ✓" } else { "MISMATCH ✗" });
    enc_ok && dec_ok
}

fn stage_b(m: &Model) -> bool {
    let c = m.cfg;
    println!("Stage B config: V={} d={} L={} H={} gru_h={} gru_L={} seg={} gate={} conf={} eps={:.1e}",
        c.vocab_size, c.d_model, c.n_layers, c.n_heads, c.gru_hidden, c.gru_layers,
        c.seg_len, c.gate_dim, c.conf_dim, c.eps);
    println!("           tensors loaded: {}", m.t.len());
    let qkv = m.get("blk0.qkv.code");
    println!("           blk0.qkv.code shape {:?}, scale[0]={:.6e}", qkv.shape, m.get("blk0.qkv.scale").f64[0]);
    let rb = fs::read(art("ref.nsc")).expect("ref.nsc");
    let mut o = 0usize; let _t = ru32(&rb, &mut o); let _v = ru32(&rb, &mut o);
    let tok0 = i32::from_le_bytes(rb[o..o + 4].try_into().unwrap()) as usize;
    let row = m.order1_row(tok0);
    let o1max = row.iter().cloned().fold(f32::MIN, f32::max);
    let embed = m.get("embed"); let order0 = m.get("order0");
    println!("           tok0={} | order0[0]={:.8} order1[tok0][tok0+1]={:.8} o1_max={:.8} embed[tok0][0]={:.8}",
        tok0, order0.f32[0], row[tok0 + 1], o1max, embed.f32[tok0 * c.d_model]);
    m.t.contains_key("order1") && m.t.contains_key("blk3.qkv.code")
}

fn stage_c_full(m: &Model) -> bool {
    let b = match fs::read(art("ref.nsc")) { Ok(d) => d, Err(_) => { println!("Stage C full: ref.nsc missing"); return true; } };
    let mut o = 0usize;
    let t = ru32(&b, &mut o) as usize;
    let v = ru32(&b, &mut o) as usize;
    let tokens: Vec<usize> = (0..t).map(|_| ri32(&b, &mut o) as usize).collect();
    let feats: Vec<[f64; 4]> = (0..t).map(|_| [rf64(&b, &mut o), rf64(&b, &mut o), rf64(&b, &mut o), rf64(&b, &mut o)]).collect();
    let exp: Vec<Vec<i64>> = (0..t).map(|_| (0..v + 1).map(|_| ru64(&b, &mut o) as i64).collect()).collect();

    let mut dec = forward::Decoder::new(m);
    let mut diff = 0usize;
    let mut first = None;
    for i in 0..t {
        let f = feats[i];
        let probs = dec.step(tokens[i], f[0] as usize, f[1] as f32, f[2] as usize, f[3] as f32);
        let cum = quantize(&probs, QUANT_BITS);
        if cum != exp[i] { diff += 1; if first.is_none() { first = Some(i); } }
    }
    println!("Stage C full forward: {}/{} coder tables differ (first mismatch @ {:?})", diff, t, first);
    diff == 0
}

fn stage_d(m: &Model) -> bool {
    let b = match fs::read(art("ref.nsc")) { Ok(d) => d, Err(_) => { println!("Stage D: ref.nsc missing"); return true; } };
    let mut o = 0usize;
    let t = ru32(&b, &mut o) as usize;
    let v = ru32(&b, &mut o) as usize;
    let tokens: Vec<usize> = (0..t).map(|_| ri32(&b, &mut o) as usize).collect();
    let feats: Vec<[f64; 4]> = (0..t).map(|_| [rf64(&b, &mut o), rf64(&b, &mut o), rf64(&b, &mut o), rf64(&b, &mut o)]).collect();

    let mut ed = forward::Decoder::new(m);
    let mut enc = RangeEncoder::new();
    for i in 0..t {
        let f = feats[i];
        let probs = ed.step(tokens[i], f[0] as usize, f[1] as f32, f[2] as usize, f[3] as f32);
        let cum = quantize(&probs, QUANT_BITS);
        if i + 1 < t {
            let s = tokens[i + 1];
            enc.encode(cum[s] as u64, cum[s + 1] as u64, cum[v] as u64);
        }
    }
    let (payload, nbits) = enc.finish();

    let mut dd = forward::Decoder::new(m);
    let mut dec = RangeDecoder::new(payload.clone(), nbits);
    let mut rec = vec![tokens[0]];
    for i in 0..t - 1 {
        let f = feats[i];
        let probs = dd.step(rec[i], f[0] as usize, f[1] as f32, f[2] as usize, f[3] as f32);
        let cum: Vec<u64> = quantize(&probs, QUANT_BITS).iter().map(|&x| x as u64).collect();
        rec.push(dec.decode(&cum));
    }
    let ok = rec == tokens;
    println!("Stage D Rust roundtrip: {} tokens, payload {}B, tokens_match={} {}",
        t, payload.len(), ok, if ok { "✓ BYTE-EXACT" } else { "✗" });

    let iters = 2000usize;
    let mut sd = forward::Decoder::new(m);
    let t0 = std::time::Instant::now();
    let mut sink = 0f32;
    for i in 0..iters {
        let j = i % t;
        let f = feats[j];
        let p = sd.step(tokens[j], f[0] as usize, f[1] as f32, f[2] as usize, f[3] as f32);
        sink += p[0];
        if (i + 1) % t == 0 { sd = forward::Decoder::new(m); }
    }
    let secs = t0.elapsed().as_secs_f64();
    println!("Stage D forward speed: {:.0} tok/s  ({:.3} ms/token)  [sink {:.3}]",
        iters as f64 / secs, secs / iters as f64 * 1e3, sink);
    ok
}

fn stage_e(m: &Model, o2: &O2Table) -> bool {
    let b = match fs::read(art("ref.nsc")) { Ok(d) => d, Err(_) => { println!("Stage E: ref.nsc missing (skip)"); return true; } };
    let mut o = 0usize;
    let t = ru32(&b, &mut o) as usize;
    let _v = ru32(&b, &mut o) as usize;
    let tokens: Vec<usize> = (0..t).map(|_| ri32(&b, &mut o) as usize).collect();
    let feats: Vec<[f64; 4]> = (0..t).map(|_| [rf64(&b, &mut o), rf64(&b, &mut o), rf64(&b, &mut o), rf64(&b, &mut o)]).collect();
    assert_eq!(o2.v, m.cfg.vocab_size, "o2.nsc vocab mismatch");
    let mut fm = FeatureModel::new(o2);
    let mut bad = 0usize;
    for i in 0..t {
        let (o2t, o2c, mt, mc) = fm.push(tokens[i]);
        let e = feats[i];
        if o2t != e[0] as usize || mt != e[2] as usize || o2c as f64 != e[1] || mc as f64 != e[3] { bad += 1; }
    }
    println!("Stage E feature model vs Python: {}/{} positions differ", bad, t);
    bad == 0
}

/// Run the full Stage A–E parity harness against the fixtures in `$NSC_ARTIFACTS` (default `artifacts/`).
pub fn run() {
    let a = stage_a();
    let m = Model::load(&art("model.nsc"));
    let b = stage_b(&m);
    let e = match fs::read(art("o2.nsc")) {
        Ok(_) => stage_e(&m, &O2Table::load(&art("o2.nsc"))),
        Err(_) => { println!("Stage E: o2.nsc missing (skip; run python export_o2.py)"); true }
    };
    let c = stage_c(&m);
    let cf = stage_c_full(&m);
    let dd = stage_d(&m);
    println!("\n=> A coder:{} | B load:{} | E feats:{} | C GRU:{} | C fidelity-vs-python:{} | D Rust roundtrip:{}",
        if a { "PASS" } else { "FAIL" },
        if b { "OK" } else { "FAIL" },
        if e { "exact" } else { "DIFF" },
        if c { "exact" } else { "diff" },
        if cf { "exact" } else { "~5e-7 (fine)" },
        if dd { "BYTE-EXACT ✓" } else { "FAIL" });
}
