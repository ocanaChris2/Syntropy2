//! Loads the int8 artifacts exported by export_int8.py (model.nsc) plus the classical
//! order0/order1 tables. Dependency-free binary parsing.
use std::collections::HashMap;
use std::fs;

#[derive(Clone, Copy)]
pub struct Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub gru_hidden: usize,
    pub gru_layers: usize,
    pub seg_len: usize,
    pub gate_dim: usize,
    pub conf_dim: usize,
    pub eps: f32,
}

pub struct Tensor {
    pub shape: Vec<usize>,
    pub f32: Vec<f32>, // dequant/float tensors (embed, order0, scales-as-f32 unused, LN, bias)
    pub i8: Vec<i8>,   // int8 weight codes
    pub f64: Vec<f64>, // per-channel weight scales
}
impl Tensor {
    pub fn numel(&self) -> usize { self.shape.iter().product() }
}

pub struct Model {
    pub cfg: Config,
    pub t: HashMap<String, Tensor>,
    pub o1_max: Vec<f32>, // per-token max of order1 row (conf_state, precomputed)
    // per-token normalized entropy of the order1 row with system libm (legacy NSC2) and with
    // deterministic math (NSC3); each built on first use, since a run needs only one of them
    o1_ent_libm: std::sync::OnceLock<Vec<f32>>,
    o1_ent_pure: std::sync::OnceLock<Vec<f32>>,
}

fn ru16(b: &[u8], o: &mut usize) -> u16 { let v = u16::from_le_bytes(b[*o..*o + 2].try_into().unwrap()); *o += 2; v }
fn ru32(b: &[u8], o: &mut usize) -> u32 { let v = u32::from_le_bytes(b[*o..*o + 4].try_into().unwrap()); *o += 4; v }
fn rf32(b: &[u8], o: &mut usize) -> f32 { let v = f32::from_le_bytes(b[*o..*o + 4].try_into().unwrap()); *o += 4; v }

impl Model {
    pub fn load(path: &str) -> Model {
        let b = fs::read(path).expect("model.nsc missing (run export_int8.py)");
        let mut o = 0usize;
        assert_eq!(&b[0..4], b"NSC1", "bad magic");
        o = 4;
        let g: Vec<usize> = (0..9).map(|_| ru32(&b, &mut o) as usize).collect();
        let cfg = Config {
            vocab_size: g[0], d_model: g[1], n_layers: g[2], n_heads: g[3], gru_hidden: g[4],
            gru_layers: g[5], seg_len: g[6], gate_dim: g[7], conf_dim: g[8], eps: rf32(&b, &mut o),
        };
        let n = ru32(&b, &mut o) as usize;
        let mut t = HashMap::new();
        for _ in 0..n {
            let nl = ru16(&b, &mut o) as usize;
            let name = String::from_utf8(b[o..o + nl].to_vec()).unwrap(); o += nl;
            let dt = b[o]; let nd = b[o + 1] as usize; o += 2;
            let shape: Vec<usize> = (0..nd).map(|_| ru32(&b, &mut o) as usize).collect();
            let ne: usize = shape.iter().product();
            let mut ten = Tensor { shape, f32: vec![], i8: vec![], f64: vec![] };
            match dt {
                0 => { ten.f32 = (0..ne).map(|i| f32::from_le_bytes(b[o + i * 4..o + i * 4 + 4].try_into().unwrap())).collect(); o += ne * 4; }
                1 => { ten.i8 = (0..ne).map(|i| b[o + i] as i8).collect(); o += ne; }
                2 => { ten.f64 = (0..ne).map(|i| f64::from_le_bytes(b[o + i * 8..o + i * 8 + 8].try_into().unwrap())).collect(); o += ne * 8; }
                _ => panic!("bad dtype"),
            }
            t.insert(name, ten);
        }
        // precompute per-token order1 stats (conf_state depends only on the token)
        let v = cfg.vocab_size;
        let o1 = &t.get("order1").expect("order1").f32;
        let o1_max = (0..v).map(|i| o1[i * v..(i + 1) * v].iter().cloned().fold(f32::MIN, f32::max)).collect();
        Model { cfg, t, o1_max, o1_ent_libm: std::sync::OnceLock::new(), o1_ent_pure: std::sync::OnceLock::new() }
    }

    /// Per-token order-1 entropy computed with system libm (legacy NSC2 files).
    pub fn o1_ent(&self) -> &[f32] {
        self.o1_ent_libm.get_or_init(|| self.o1_entropy(|x| x.ln()))
    }

    /// Per-token order-1 entropy computed with [`crate::dmath`] (portable neural kind).
    pub fn o1_ent_pure(&self) -> &[f32] {
        self.o1_ent_pure.get_or_init(|| self.o1_entropy(crate::dmath::ln))
    }

    /// Normalized entropy of every order-1 row with logarithm `ln` (V² logarithms: rows are split
    /// across threads; each row's sum keeps its order, so the values do not depend on the split).
    fn o1_entropy(&self, ln: fn(f32) -> f32) -> Vec<f32> {
        let v = self.cfg.vocab_size;
        let o1 = &self.get("order1").f32;
        let logv = ln(v as f32);
        let eps = self.cfg.eps;
        let mut out = vec![0f32; v];
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).clamp(1, 16);
        let rows = v.div_ceil(threads).max(1);
        std::thread::scope(|sc| {
            for (c, chunk) in out.chunks_mut(rows).enumerate() {
                sc.spawn(move || {
                    for (k, e) in chunk.iter_mut().enumerate() {
                        let i = c * rows + k;
                        let mut ent = 0f32;
                        for &p in &o1[i * v..(i + 1) * v] { ent += p * ln(p + eps); }
                        *e = -ent / logv;
                    }
                });
            }
        });
        out
    }

    pub fn get(&self, name: &str) -> &Tensor { self.t.get(name).unwrap_or_else(|| panic!("no tensor {name}")) }

    /// order1 row p(next | prev=i) as an f32 slice of length V (stored f32 in model.nsc).
    pub fn order1_row(&self, i: usize) -> &[f32] {
        let v = self.cfg.vocab_size;
        &self.get("order1").f32[i * v..(i + 1) * v]
    }

    /// A quantized Linear: (int8 codes [out,in], per-channel scale [out], bias [out], out, in).
    pub fn lin(&self, name: &str) -> (&[i8], &[f64], &[f32], usize, usize) {
        let c = self.get(&format!("{name}.code"));
        let s = self.get(&format!("{name}.scale"));
        let b = self.get(&format!("{name}.bias"));
        (&c.i8, &s.f64, &b.f32, c.shape[0], c.shape[1])
    }
}
