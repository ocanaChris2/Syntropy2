//! int8 neural forward — bit-exact port of int8_codec.py. Integer dot products accumulate in
//! i32 (int8·int8 summed over ≤1024 ≤ 16.5M < 2^31, exact), which vectorizes well; dequant in
//! f64 then f32 + f32 bias (matches int8_codec's `.double().mul().float()+b`). round-half-to-EVEN
//! quant (matches torch.round). erf-GELU via system libm.
use crate::artifacts::Model;

#[link(name = "m")]
unsafe extern "C" {
    fn erff(x: f32) -> f32;
}

/// Transcendental functions used by the forward pass. [`Libm`] is the system math library: it is
/// fixed for legacy NSC2 files, whose coder tables were produced with it. [`Pure`] is
/// [`crate::dmath`]: deterministic on every IEEE machine, used by the portable NSC3 neural kind.
pub trait Math: 'static {
    const PURE: bool;
    fn exp(x: f32) -> f32;
    fn ln(x: f32) -> f32;
    fn tanh(x: f32) -> f32;
    fn erf(x: f32) -> f32;
    fn sin_cos(x: f32) -> (f32, f32);
}

pub struct Libm;
impl Math for Libm {
    const PURE: bool = false;
    #[inline] fn exp(x: f32) -> f32 { x.exp() }
    #[inline] fn ln(x: f32) -> f32 { x.ln() }
    #[inline] fn tanh(x: f32) -> f32 { x.tanh() }
    #[inline] fn erf(x: f32) -> f32 { unsafe { erff(x) } }
    #[inline] fn sin_cos(x: f32) -> (f32, f32) { x.sin_cos() }
}

pub struct Pure;
impl Math for Pure {
    const PURE: bool = true;
    #[inline] fn exp(x: f32) -> f32 { crate::dmath::exp(x) }
    #[inline] fn ln(x: f32) -> f32 { crate::dmath::ln(x) }
    #[inline] fn tanh(x: f32) -> f32 { crate::dmath::tanh(x) }
    #[inline] fn erf(x: f32) -> f32 { crate::dmath::erf(x) }
    #[inline] fn sin_cos(x: f32) -> (f32, f32) { crate::dmath::sin_cos(x) }
}

#[inline]
pub fn gelu_m<M: Math>(x: f32) -> f32 { 0.5 * x * (1.0 + M::erf(x * std::f32::consts::FRAC_1_SQRT_2)) }
#[inline]
pub fn sigmoid_m<M: Math>(x: f32) -> f32 { 1.0 / (1.0 + M::exp(-x)) }
#[inline]
pub fn sigmoid(x: f32) -> f32 { sigmoid_m::<Libm>(x) }

/// per-row int8 dynamic quant: scale = max|x|/127 (>=1e-30), codes = round_ties_even(x/scale).
pub fn qrow(x: &[f32]) -> (Vec<i8>, f32) {
    let amax = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let s = (amax / 127.0).max(1e-30);
    let q = x.iter().map(|&v| (v / s).round_ties_even().clamp(-128.0, 127.0) as i8).collect();
    (q, s)
}

/// integer dot product of two int8 slices (exact in i32). Bit-identical to the scalar sum:
/// int8·int8 ≤ 16129 and i32 accumulation is associative, so SIMD reordering changes no bit.
#[inline]
fn idot(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { idot_avx2(a, b) };
        }
    }
    idot_scalar(a, b)
}

#[inline]
fn idot_scalar(a: &[i8], b: &[i8]) -> i32 {
    let mut acc: i32 = 0;
    for i in 0..a.len() {
        acc += a[i] as i32 * b[i] as i32;
    }
    acc
}

/// AVX2 int8 dot: widen i8→i16 (cvtepi8_epi16) then madd_epi16 (pairwise i16·i16→i32, summed).
/// Accumulates in eight i32 lanes; over inp≤1024 a lane holds ≤128·16129 ≈ 2.06M, no overflow.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn idot_avx2(a: &[i8], b: &[i8]) -> i32 { unsafe {
    use std::arch::x86_64::*;
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc = _mm256_setzero_si256();
    let mut i = 0usize;
    // 32 int8 per iteration: two 16-wide widened madds.
    while i + 32 <= n {
        let a0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(pa.add(i) as *const __m128i));
        let b0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(pb.add(i) as *const __m128i));
        let a1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(pa.add(i + 16) as *const __m128i));
        let b1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(pb.add(i + 16) as *const __m128i));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a0, b0));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a1, b1));
        i += 32;
    }
    while i + 16 <= n {
        let a0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(pa.add(i) as *const __m128i));
        let b0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(pb.add(i) as *const __m128i));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a0, b0));
        i += 16;
    }
    // horizontal sum of the eight i32 lanes.
    let hi = _mm256_extracti128_si256(acc, 1);
    let lo = _mm256_castsi256_si128(acc);
    let s128 = _mm_add_epi32(lo, hi);
    let s64 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, 0b01_00_11_10));
    let s32 = _mm_add_epi32(s64, _mm_shuffle_epi32(s64, 0b00_00_00_01));
    let mut acc_s = _mm_cvtsi128_si32(s32);
    while i < n {
        acc_s += *pa.add(i) as i32 * *pb.add(i) as i32;
        i += 1;
    }
    acc_s
}}

/// int8 Linear: y[o] = f32( (Σ xq·code[o]) · (xs·scale[o]) ) + bias[o].
pub fn mm(x: &[f32], code: &[i8], scale: &[f64], bias: &[f32], out: usize, inp: usize) -> Vec<f32> {
    let (xq, xs) = qrow(x);
    let xs = xs as f64;
    let mut y = vec![0f32; out];
    for o in 0..out {
        let base = o * inp;
        let acc = idot(&xq, &code[base..base + inp]);
        y[o] = (acc as f64 * (xs * scale[o])) as f32 + bias[o];
    }
    y
}

pub fn conf_state(m: &Model, tok: usize, o2c: f32, mc: f32) -> [f32; 4] { conf_state_m::<Libm>(m, tok, o2c, mc) }

pub fn conf_state_m<M: Math>(m: &Model, tok: usize, o2c: f32, mc: f32) -> [f32; 4] {
    // precomputed per-token (depends only on the row); the pure-math table is built on first use
    let ent = if M::PURE { m.o1_ent_pure()[tok] } else { m.o1_ent()[tok] };
    [m.o1_max[tok], ent, o2c, mc]
}

/// FiLM: gb = film2(gelu(film0(conf))); (1+gamma)*hid + beta.
fn film(m: &Model, prefix: &str, hid: &[f32], conf: &[f32]) -> Vec<f32> { film_m::<Libm>(m, prefix, hid, conf) }

fn film_m<M: Math>(m: &Model, prefix: &str, hid: &[f32], conf: &[f32]) -> Vec<f32> {
    let (f0c, f0s, f0b, f0o, f0i) = m.lin(&format!("{prefix}.film0"));
    let t0: Vec<f32> = mm(conf, f0c, f0s, f0b, f0o, f0i).iter().map(|&v| gelu_m::<M>(v)).collect();
    let (f2c, f2s, f2b, f2o, f2i) = m.lin(&format!("{prefix}.film2"));
    let gb = mm(&t0, f2c, f2s, f2b, f2o, f2i);
    let n = hid.len();
    (0..n).map(|k| (1.0 + gb[k]) * hid[k] + gb[n + k]).collect()
}

/// GRU(2L)+FiLM+to_resid for a single fresh token (pos 0) — Stage-C verification.
pub fn resid_gru_pos0(m: &Model, tok: usize, o2c: f32, mc: f32) -> (Vec<f32>, Vec<f32>, [f32; 4]) {
    let cfg = m.cfg;
    let (d, h) = (cfg.d_model, cfg.gru_hidden);
    let e = m.get("embed").f32[tok * d..(tok + 1) * d].to_vec();
    let conf = conf_state(m, tok, o2c, mc);
    let mut hs: Vec<Vec<f32>> = (0..cfg.gru_layers).map(|_| vec![0f32; h]).collect();
    let mut inp = e;
    for l in 0..cfg.gru_layers {
        let (wc, ws, wb, wo, wi) = m.lin(&format!("gru.wih.l{l}"));
        let gi = mm(&inp, wc, ws, wb, wo, wi);
        let (hc, hsc, hb, ho, hi) = m.lin(&format!("gru.whh.l{l}"));
        let gg = mm(&hs[l], hc, hsc, hb, ho, hi);
        let mut hnew = vec![0f32; h];
        for k in 0..h {
            let r = sigmoid(gi[k] + gg[k]);
            let z = sigmoid(gi[h + k] + gg[h + k]);
            let n = Libm::tanh(gi[2 * h + k] + r * gg[2 * h + k]);
            hnew[k] = (1.0 - z) * n + z * hs[l][k];
        }
        hs[l] = hnew.clone();
        inp = hnew;
    }
    let gru_h = hs[cfg.gru_layers - 1].clone();
    let ghf = film(m, "local", &gru_h, &conf);
    let (trc, trs, trb, tro, tri) = m.lin("local.to_resid");
    (mm(&ghf, trc, trs, trb, tro, tri), ghf, conf)
}

pub fn layernorm(x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + 1e-5).sqrt();
    (0..x.len()).map(|i| (x[i] - mean) * inv * w[i] + b[i]).collect()
}

fn softmax_m<M: Math>(z: &[f32]) -> Vec<f32> {
    let mx = z.iter().cloned().fold(f32::MIN, f32::max);
    let ex: Vec<f32> = z.iter().map(|&v| M::exp(v - mx)).collect();
    let s: f32 = ex.iter().sum();
    ex.iter().map(|&v| v / s).collect()
}

fn apply_rope_m<M: Math>(x: &mut [f32], pos: usize, hd: usize) {
    let half = hd / 2;
    let mut out = vec![0f32; hd];
    for i in 0..half {
        let inv = M::exp((2 * i) as f32 * (-M::ln(10000f32) / hd as f32));
        let (s, c) = M::sin_cos(pos as f32 * inv);
        out[i] = x[i] * c - x[i + half] * s;
        out[i + half] = x[i + half] * c + x[i] * s;
    }
    x.copy_from_slice(&out);
}

/// Stateful int8 KV-cache decoder — reproduces int8_codec.Int8Decoder (T-pad softmax), generic
/// over the transcendental-math backend.
pub struct DecoderM<'a, M: Math> {
    m: &'a Model,
    pos: usize,
    gru_hs: Vec<Vec<f32>>,
    kcode: Vec<Vec<Vec<i8>>>, // [block][key] -> [h*hd]
    kscale: Vec<Vec<Vec<f32>>>, // [block][key] -> [h]
    vcode: Vec<Vec<Vec<i8>>>,
    vscale: Vec<Vec<Vec<f32>>>,
    _math: std::marker::PhantomData<M>,
}

/// The legacy (system libm) decoder used by NSC2 files.
pub type Decoder<'a> = DecoderM<'a, Libm>;

impl<'a, M: Math> DecoderM<'a, M> {
    pub fn new(m: &'a Model) -> Self {
        let nb = m.cfg.n_layers;
        DecoderM {
            m, pos: 0,
            gru_hs: (0..m.cfg.gru_layers).map(|_| vec![0f32; m.cfg.gru_hidden]).collect(),
            kcode: vec![vec![]; nb], kscale: vec![vec![]; nb],
            vcode: vec![vec![]; nb], vscale: vec![vec![]; nb],
            _math: std::marker::PhantomData,
        }
    }

    pub fn step(&mut self, tok: usize, o2t: usize, o2c: f32, mt: usize, mc: f32) -> Vec<f32> {
        let m = self.m;
        let cfg = m.cfg;
        let (d, h, hd, gh) = (cfg.d_model, cfg.n_heads, cfg.d_model / cfg.n_heads, cfg.gru_hidden);
        let pos = self.pos;
        let e = m.get("embed").f32[tok * d..(tok + 1) * d].to_vec();
        let conf = conf_state_m::<M>(m, tok, o2c, mc);
        let o1_max = conf[0];

        // GRU
        let mut inp = e.clone();
        for l in 0..cfg.gru_layers {
            let (wc, ws, wb, wo, wi) = m.lin(&format!("gru.wih.l{l}"));
            let gi = mm(&inp, wc, ws, wb, wo, wi);
            let (hc, hsc, hb, ho, hi) = m.lin(&format!("gru.whh.l{l}"));
            let gg = mm(&self.gru_hs[l], hc, hsc, hb, ho, hi);
            let mut hnew = vec![0f32; gh];
            for k in 0..gh {
                let r = sigmoid_m::<M>(gi[k] + gg[k]);
                let z = sigmoid_m::<M>(gi[gh + k] + gg[gh + k]);
                let n = M::tanh(gi[2 * gh + k] + r * gg[2 * gh + k]);
                hnew[k] = (1.0 - z) * n + z * self.gru_hs[l][k];
            }
            self.gru_hs[l] = hnew.clone();
            inp = hnew;
        }
        let ghf = film_m::<M>(m, "local", &self.gru_hs[cfg.gru_layers - 1].clone(), &conf);
        let resid_gru = { let (c, s, b, o, i) = m.lin("local.to_resid"); mm(&ghf, c, s, b, o, i) };

        // transformer
        let mut x = e.clone();
        for bi in 0..cfg.n_layers {
            let xn = layernorm(&x, &m.get(&format!("blk{bi}.ln1.w")).f32, &m.get(&format!("blk{bi}.ln1.b")).f32);
            let qkv = { let (c, s, b, o, i) = m.lin(&format!("blk{bi}.qkv")); mm(&xn, c, s, b, o, i) };
            let (mut q, mut k, v) = (qkv[0..d].to_vec(), qkv[d..2 * d].to_vec(), qkv[2 * d..3 * d].to_vec());
            let (mut kc, mut ks, mut vc, mut vs) = (vec![0i8; d], vec![0f32; h], vec![0i8; d], vec![0f32; h]);
            for hh in 0..h {
                apply_rope_m::<M>(&mut q[hh * hd..(hh + 1) * hd], pos, hd);
                apply_rope_m::<M>(&mut k[hh * hd..(hh + 1) * hd], pos, hd);
                let (c, sc) = qrow(&k[hh * hd..(hh + 1) * hd]); kc[hh * hd..(hh + 1) * hd].copy_from_slice(&c); ks[hh] = sc;
                let (c, sc) = qrow(&v[hh * hd..(hh + 1) * hd]); vc[hh * hd..(hh + 1) * hd].copy_from_slice(&c); vs[hh] = sc;
            }
            self.kcode[bi].push(kc); self.kscale[bi].push(ks);
            self.vcode[bi].push(vc); self.vscale[bi].push(vs);
            let np = pos + 1;
            let mut out = vec![0f32; d];
            let sqhd = (hd as f32).sqrt();
            let mut iacc = vec![0i32; hd];
            for hh in 0..h {
                let (qc, qs) = qrow(&q[hh * hd..(hh + 1) * hd]);
                let mut sc = vec![0f32; np];
                for j in 0..np {
                    let acc = idot(&qc, &self.kcode[bi][j][hh * hd..(hh + 1) * hd]);
                    sc[j] = (acc as f64 * (qs as f64 * self.kscale[bi][j][hh] as f64)) as f32 / sqhd;
                }
                let att = softmax_m::<M>(&sc);
                let bfold: Vec<f32> = (0..np).map(|j| att[j] * self.vscale[bi][j][hh]).collect();
                let (bc, bs) = qrow(&bfold);
                // out[c] = Σ_j bc[j]·vcode[j][c] (exact i32). Key-outer/channel-inner so the inner
                // reads a contiguous vcode row and autovectorizes; same summation order → byte-exact.
                iacc.iter_mut().for_each(|x| *x = 0);
                for j in 0..np {
                    let bcj = bc[j] as i32;
                    let vrow = &self.vcode[bi][j][hh * hd..hh * hd + hd];
                    for c in 0..hd {
                        iacc[c] += bcj * vrow[c] as i32;
                    }
                }
                for c in 0..hd {
                    out[hh * hd + c] = (iacc[c] as f64 * bs as f64) as f32;
                }
            }
            let proj = { let (c, s, b, o, i) = m.lin(&format!("blk{bi}.proj")); mm(&out, c, s, b, o, i) };
            for t in 0..d { x[t] += proj[t]; }
            let x2 = layernorm(&x, &m.get(&format!("blk{bi}.ln2.w")).f32, &m.get(&format!("blk{bi}.ln2.b")).f32);
            let m0: Vec<f32> = { let (c, s, b, o, i) = m.lin(&format!("blk{bi}.mlp0")); mm(&x2, c, s, b, o, i) }.iter().map(|&v| gelu_m::<M>(v)).collect();
            let m1 = { let (c, s, b, o, i) = m.lin(&format!("blk{bi}.mlp2")); mm(&m0, c, s, b, o, i) };
            for t in 0..d { x[t] += m1[t]; }
        }
        let xf_h = film_m::<M>(m, "xformer", &layernorm(&x, &m.get("xformer.lnf.w").f32, &m.get("xformer.lnf.b").f32), &conf);
        let resid_xf = { let (c, s, b, o, i) = m.lin("xformer.to_resid"); mm(&xf_h, c, s, b, o, i) };

        // gate
        let mut qin = ghf.clone();
        qin.extend_from_slice(&xf_h);
        let q = { let (c, s, b, o, i) = m.lin("gate.q_proj"); mm(&qin, c, s, b, o, i) };
        let ma = |z: &[f32]| z.iter().map(|v| v.abs()).sum::<f32>() / z.len() as f32;
        let ef = [0.5f32, o1_max, o2c, mc, ma(&ghf), ma(&xf_h)];
        let (fc, fs, fb, fo, _fi) = m.lin("gate.feat_proj");
        let ee = &m.get("gate.expert_embed").f32;
        let gd = cfg.gate_dim;
        let mut scores = [0f32; 6];
        for e6 in 0..6 {
            let kfe = mm(&[ef[e6]], fc, fs, fb, fo, 1);
            let mut acc = 0f32;
            for kk in 0..gd { acc += q[kk] * (ee[e6 * gd + kk] + kfe[kk]); }
            scores[e6] = acc / (gd as f32).sqrt();
        }
        let w = softmax_m::<M>(&scores[0..4]);
        let (g0, g1) = (sigmoid_m::<M>(scores[4]), sigmoid_m::<M>(scores[5]));

        // mixture -> probs. The two "peaked" experts are uniform except at one symbol, so their
        // values are selected inline instead of materialized; slice zips (no bounds checks) let the
        // loop vectorize. Same arithmetic, same order as the reference.
        let v = cfg.vocab_size;
        let p0 = &m.get("order0").f32[..v];
        let p1 = &m.order1_row(tok)[..v];
        let (c2, c_m) = (o2c.clamp(1e-4, 1.0 - 1e-4), mc.clamp(1e-4, 1.0 - 1e-4));
        let (b2, b_m) = ((1.0 - c2) / ((v - 1) as f32), (1.0 - c_m) / ((v - 1) as f32));
        let (t2, t_m) = (o2t.min(v - 1), mt.min(v - 1));
        let eps = cfg.eps;
        let mut z = vec![0f32; v];
        for (s, (((zs, (&a, &b)), &rg), &rx)) in z.iter_mut().zip(p0.iter().zip(p1)).zip(&resid_gru[..v]).zip(&resid_xf[..v]).enumerate() {
            let e2 = if s == t2 { c2 } else { b2 };
            let em = if s == t_m { c_m } else { b_m };
            let pcls = w[0] * a + w[1] * b + w[2] * e2 + w[3] * em;
            *zs = M::ln(pcls + eps) + g0 * rg + g1 * rx;
        }
        self.pos += 1;
        softmax_m::<M>(&z)
    }
}
