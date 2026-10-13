//! Online classical feature model — a bit-exact port of `IncrementalFeatures.push`
//! (test_compression_integrity.py:266). The order-2 expert is a static lookup into the
//! exported FastO2 table (`o2.nsc`); the match expert is rebuilt online from the decoded
//! prefix. Because encoder and decoder both replay the SAME prefix through this model, they
//! compute identical `(o2_top, o2_conf, match_top, match_conf)` features — hence identical
//! coder tables and a byte-exact round-trip. This is what lets the codec process arbitrary
//! input without shipping the Python-side `ref.nsc` feature dump.
use std::collections::HashMap;
use std::fs;

/// FastO2 order-2 table: for a two-token context key `a*V + b`, the most likely next token
/// `top` and its confidence `conf = count(top)/total`. `ctx` is strictly ascending so the
/// lookup is a binary search (mirrors Python's `np.searchsorted` + exact-match check).
pub struct O2Table {
    pub v: usize,
    pub k: usize, // match order (order2_match_order)
    pub cap: f32, // match_len_cap
    pub ctx: Vec<i64>,
    pub top: Vec<i32>,
    pub conf: Vec<f32>,
}

fn ru32(b: &[u8], o: &mut usize) -> u32 { let v = u32::from_le_bytes(b[*o..*o + 4].try_into().unwrap()); *o += 4; v }
fn rf32(b: &[u8], o: &mut usize) -> f32 { let v = f32::from_le_bytes(b[*o..*o + 4].try_into().unwrap()); *o += 4; v }

impl O2Table {
    /// Parse `o2.nsc` (magic "NO21" | V u32 | K u32 | cap f32 | n u32 | ctx[i64] | top[i32] | conf[f32]).
    pub fn load(path: &str) -> O2Table {
        let b = fs::read(path).expect("o2.nsc missing (run: python export_o2.py)");
        assert_eq!(&b[0..4], b"NO21", "bad o2.nsc magic");
        let mut o = 4usize;
        let v = ru32(&b, &mut o) as usize;
        let k = ru32(&b, &mut o) as usize;
        let cap = rf32(&b, &mut o);
        let n = ru32(&b, &mut o) as usize;
        let ctx: Vec<i64> = (0..n).map(|i| i64::from_le_bytes(b[o + i * 8..o + i * 8 + 8].try_into().unwrap())).collect();
        o += n * 8;
        let top: Vec<i32> = (0..n).map(|i| i32::from_le_bytes(b[o + i * 4..o + i * 4 + 4].try_into().unwrap())).collect();
        o += n * 4;
        let conf: Vec<f32> = (0..n).map(|i| f32::from_le_bytes(b[o + i * 4..o + i * 4 + 4].try_into().unwrap())).collect();
        O2Table { v, k, cap, ctx, top, conf }
    }
}

/// Continuous online feature model — NEVER reset at window boundaries (matches Python's
/// `compute_features`, which runs over the whole token array before windowing).
pub struct FeatureModel<'a> {
    o2: &'a O2Table,
    v: usize,
    k: usize,
    cap: f64,
    tokens: Vec<usize>,
    pos: HashMap<u64, usize>, // exact packed K-gram key -> last index
    run: i64,
    prev_mtop: usize,
}

impl<'a> FeatureModel<'a> {
    pub fn new(o2: &'a O2Table) -> Self {
        FeatureModel {
            o2, v: o2.v, k: o2.k, cap: o2.cap as f64,
            tokens: Vec::new(), pos: HashMap::new(), run: 0, prev_mtop: 0,
        }
    }

    /// Append token `t`; return `(o2_top, o2_conf, match_top, match_conf)` — the features used
    /// to predict the NEXT token. Exact port of `IncrementalFeatures.push`.
    pub fn push(&mut self, t: usize) -> (usize, f32, usize, f32) {
        let i = self.tokens.len();
        self.tokens.push(t);
        let v = self.v;

        // order-2: context (tokens[i-1], t) predicts i+1 (defined for i >= 1)
        let (mut o2t, mut o2c) = (0usize, 0f32);
        if i >= 1 {
            let key = (self.tokens[i - 1] * v + t) as i64;
            if let Ok(idx) = self.o2.ctx.binary_search(&key) {
                o2t = self.o2.top[idx] as usize;
                o2c = self.o2.conf[idx];
            }
        }

        // match: exact packed K-gram ending at i -> last occurrence -> the token that followed it
        let (mut mt, mut mc) = (0usize, 0f32);
        if i >= self.k {
            let mut h: u64 = 0;
            for j in (i - self.k + 1)..=i {
                h = h * v as u64 + self.tokens[j] as u64;
            }
            if let Some(&prev) = self.pos.get(&h) {
                mt = self.tokens[prev + 1];
                self.run = if self.prev_mtop == t { self.run + 1 } else { 1 };
                // computed f64-then-f32 to match Python's `float(np.float32(min(run/cap, 0.99)))`
                mc = (self.run as f64 / self.cap).min(0.99) as f32;
            } else {
                self.run = 0;
            }
            self.pos.insert(h, i);
        }
        self.prev_mtop = mt;
        (o2t, o2c, mt, mc)
    }
}
