//! Model configuration (recorded in every block, so the decoder never infers anything from a level
//! number) and the encoder-side level presets that produce it.

/// Context identifiers of the generic model (bit positions in [`ModelCfg::ctx_mask`]).
pub mod ctx {
    pub const O1: u32 = 0;
    pub const O2: u32 = 1;
    pub const O3: u32 = 2;
    pub const O4: u32 = 3;
    pub const O5: u32 = 4;
    pub const O6: u32 = 5;
    pub const O7: u32 = 6;
    pub const O8: u32 = 7;
    pub const O12: u32 = 8;
    pub const O16: u32 = 9;
    pub const O24: u32 = 10;
    pub const WORD: u32 = 11;
    pub const WORD2: u32 = 12;
    pub const SP_B2: u32 = 13;
    pub const SP_B3: u32 = 14;
    pub const SP_B2B3: u32 = 15;
    pub const SP_B1B3: u32 = 16;
    pub const SP_B4B8: u32 = 17;
    pub const SP_NIB: u32 = 18;
    pub const REC: u32 = 19;
    pub const REC_C1: u32 = 20;
    pub const IND1: u32 = 21;
    pub const IND2: u32 = 22;
    pub const MATCH: u32 = 23;
    pub const COL: u32 = 24;
    pub const POS: u32 = 25;
    // text model (active in text segments only)
    pub const T_W3: u32 = 26;
    pub const T_WSKIP: u32 = 27;
    pub const T_W1C2: u32 = 28;
    pub const T_XML: u32 = 29;
    pub const T_FIELD: u32 = 30;
    pub const T_INDENT: u32 = 31;
    pub const T_NUM: u32 = 32;
    pub const T_BRACKET: u32 = 33;
    pub const T_SENT: u32 = 34;
    // image model (active in image segments only)
    pub const I_WN: u32 = 35;
    pub const I_MED: u32 = 36;
    pub const I_GRAD: u32 = 37;
    pub const I_AVG: u32 = 38;
    pub const I_NNE: u32 = 39;
    pub const I_WW: u32 = 40;
    pub const I_NN: u32 = 41;
    pub const I_PAETH: u32 = 42;
    pub const I_TEX: u32 = 43;
    pub const I_CC: u32 = 44;
    pub const I_CC2: u32 = 45;
    pub const I_WNW: u32 = 46;
    pub const I_ERR: u32 = 47;
    pub const I_BIAS: u32 = 48;
    pub const I_CCE: u32 = 49;
    pub const I_WNQ: u32 = 50;
    /// The x86 instruction model: a set of contexts of its own (`crate::x86`), active in x86 and
    /// x86-64 code segments only. A writer sets it for blocks that hold such code, and only for
    /// decoders of level 2.
    pub const X86: u32 = 51;
    /// The second binary model: records, columns and aligned words (a set of contexts of its own,
    /// in the predictor), active outside text, images and code. For decoders of level 2.
    pub const BIN2: u32 = 52;
    /// The number model of text: numbers by the number last in their place, and lines as
    /// templates (a set of contexts of its own, `crate::cm::num`), active in text only. For
    /// decoders of level 2.
    pub const NUM: u32 = 53;
    pub const COUNT: u32 = 54;
    /// Context numbers of the instruction model's, the binary model's and the number model's
    /// contexts inside the predictor (not mask bits).
    pub const X_FIRST: u32 = 1000;
    pub const B_FIRST: u32 = 2000;
    pub const N_FIRST: u32 = 3000;

    pub fn image_only(id: u32) -> bool { (I_WN..=I_WNQ).contains(&id) }
    pub fn x86_only(id: u32) -> bool { (X_FIRST..B_FIRST).contains(&id) }
    pub fn bin2_only(id: u32) -> bool { (B_FIRST..N_FIRST).contains(&id) }
    pub fn num_only(id: u32) -> bool { id >= N_FIRST }

    /// Contexts used only inside text segments (the text model, plus the line-column context,
    /// which helps text and tables but slightly hurts binary data).
    pub fn text_only(id: u32) -> bool { (T_W3..=T_SENT).contains(&id) || id == COL }
    /// Generic contexts that only help binary data; switched off inside text segments.
    pub fn binary_only(id: u32) -> bool { matches!(id, REC | REC_C1 | SP_B4B8 | SP_NIB | POS) }
}

/// Mixer selectors (bits of [`ModelCfg::sel_mask`]).
pub mod sel {
    pub const C0: u8 = 0;
    pub const MATCH: u8 = 1;
    pub const C1: u8 = 2;
    pub const HITS: u8 = 3;
    pub const C2: u8 = 4;
}

/// Model kinds. The discriminant is stored in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    Generic = 1,
    Fast = 2,
}

impl Kind {
    pub fn from_u8(v: u8) -> Option<Kind> {
        match v { 1 => Some(Kind::Generic), 2 => Some(Kind::Fast), _ => None }
    }
    pub fn name(self) -> &'static str {
        match self { Kind::Generic => "generic", Kind::Fast => "fast" }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelCfg {
    pub kind: Kind,
    pub hash_bits: u8,
    pub mm_bits: u8,
    pub ctx_mask: u64,
    pub sel_mask: u8,
    pub apm_mask: u8,
    pub lr: u8,
    pub lr2: u8,
}

pub const CFG_BYTES: usize = 16;

impl ModelCfg {
    pub fn to_bytes(&self) -> [u8; CFG_BYTES] {
        let m = self.ctx_mask.to_le_bytes();
        [self.kind as u8, self.hash_bits, self.mm_bits, m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7],
         self.sel_mask, self.apm_mask, self.lr, self.lr2, 0]
    }

    pub fn from_bytes(b: &[u8]) -> Result<ModelCfg, String> {
        if b.len() < CFG_BYTES { return Err("truncated model config".into()); }
        let kind = Kind::from_u8(b[0]).ok_or_else(|| format!("unknown model kind {}", b[0]))?;
        let cfg = ModelCfg {
            kind,
            hash_bits: b[1],
            mm_bits: b[2],
            ctx_mask: u64::from_le_bytes(b[3..11].try_into().unwrap()),
            sel_mask: b[11],
            apm_mask: b[12],
            lr: b[13],
            lr2: b[14],
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Bounds a decoder must enforce before allocating anything (corrupt-input safety).
    pub fn validate(&self) -> Result<(), String> {
        if !(10..=26).contains(&self.hash_bits) { return Err(format!("hash_bits {} out of range", self.hash_bits)); }
        if !(10..=24).contains(&self.mm_bits) { return Err(format!("mm_bits {} out of range", self.mm_bits)); }
        if self.ctx_mask >> ctx::COUNT != 0 { return Err("unknown context bits".into()); }
        if self.sel_mask >> 5 != 0 || self.sel_mask == 0 { return Err("bad selector mask".into()); }
        if self.apm_mask >> 3 != 0 { return Err("bad apm mask".into()); }
        if self.lr == 0 || self.lr > 64 || self.lr2 > 64 { return Err("bad learning rate".into()); }
        Ok(())
    }

    /// Approximate memory the predictor allocates for this configuration (bytes).
    pub fn memory(&self) -> usize {
        (16usize << self.hash_bits) + (4usize << self.mm_bits) + 12 * 1024 * 1024
    }
}

fn mask(ids: &[u32]) -> u64 { ids.iter().fold(0, |m, &i| m | (1u64 << i)) }

/// The configuration bit of the x86 instruction model; of the second binary model.
pub fn x86_mask(_level: u32) -> u64 { 1u64 << ctx::X86 }
pub fn bin2_mask(_level: u32) -> u64 { 1u64 << ctx::BIN2 }
/// The configuration bit of the number model of text.
pub fn num_mask(_level: u32) -> u64 { 1u64 << ctx::NUM }

fn ceil_log2(n: usize) -> u32 { if n <= 1 { 0 } else { usize::BITS - (n - 1).leading_zeros() } }

/// Fixed block size per level (bytes). Never depends on the input size or the machine.
///
/// Blocks (and lanes) are coded independently, so their size is the speed/ratio dial: every
/// boundary restarts the model, but smaller blocks keep more cores busy on one file. The fast
/// levels use small blocks so that even a few-MB file spreads over all cores (a few percent larger
/// output); from level 3 up, blocks are large and ratio comes first.
pub fn block_size(level: u32) -> usize {
    match level {
        0..=1 => 2 << 20,
        2 => 4 << 20,
        3..=4 => 16 << 20,
        5..=6 => 32 << 20,
        7..=8 => 64 << 20,
        _ => 128 << 20,
    }
}

/// Encoder preset for `level` (1..=9) and a block of `len` bytes.
pub fn preset(level: u32, kind: Kind, len: usize) -> ModelCfg {
    use ctx::*;
    let level = level.clamp(1, 9);
    let (cap, ids, sels, apms): (u32, Vec<u32>, Vec<u8>, u8) = match kind {
        Kind::Fast => (20, vec![O1, O2, O3, MATCH], vec![sel::C0, sel::MATCH], 0b001),
        // Contexts are added in order of measured value (level-5 ablation over text, an ELF
        // library, fixed-size records and a photo): the record model dominates, then sparse and
        // indirect contexts, then the order-n ladder. Long orders and WORD2 only pay off on large
        // or highly repetitive inputs, so they join at the strong levels.
        Kind::Generic => {
            // Text contexts by measured value (ablation on prose, code and CSV): column and bracket
            // nesting first, then delimited fields and word n-grams.
            let l1 = [O1, O2, O3, O4, O6, REC, SP_B2B3, MATCH, I_WN, I_MED, I_GRAD, I_ERR, I_CCE];
            let l2 = [WORD, SP_NIB, REC_C1, POS, COL, T_BRACKET, I_AVG, I_CC, I_TEX, I_BIAS];
            let l3 = [IND2, SP_B1B3, O5, T_FIELD, WORD2, I_NNE, I_PAETH, I_WNQ];
            let l4 = [SP_B2, SP_B3, SP_B4B8, T_W1C2, T_WSKIP, I_WW, I_NN, I_CC2];
            let l5 = [IND1, O8, T_XML, I_WNW];
            let l6 = [O7, T_INDENT];
            let l7 = [O12, O16, T_SENT];
            let l9 = [O24, T_W3, T_NUM];
            let tiers: [&[u32]; 8] = [&l1, &l2, &l3, &l4, &l5, &l6, &l7, &l9];
            let upto = match level { 1 => 1, 2 => 2, 3 => 3, 4 => 4, 5 => 5, 6 => 6, 7 | 8 => 7, _ => 8 };
            let ids: Vec<u32> = tiers[..upto].iter().flat_map(|t| t.iter().copied()).collect();
            let all_sels = vec![sel::C0, sel::MATCH, sel::C1, sel::HITS, sel::C2];
            match level {
                1 => (20, ids, vec![sel::C0, sel::MATCH, sel::C1], 0b001),
                2 => (21, ids, vec![sel::C0, sel::MATCH, sel::C1], 0b011),
                3 => (22, ids, vec![sel::C0, sel::MATCH, sel::C1, sel::HITS], 0b111),
                4 => (23, ids, all_sels, 0b111),
                5 | 6 => (24, ids, all_sels, 0b111),
                7 | 8 => (25, ids, all_sels, 0b111),
                _ => (26, ids, all_sels, 0b111),
            }
        }
    };
    let need = ceil_log2(len.max(1)) + 2;
    let hash_bits = need.clamp(16, cap) as u8;
    let mm_bits = ceil_log2(len.max(1)).clamp(16, 24) as u8;
    ModelCfg {
        kind,
        hash_bits,
        mm_bits,
        ctx_mask: mask(&ids),
        sel_mask: sels.iter().fold(0, |m, &s| m | (1 << s)),
        apm_mask: apms,
        lr: 6,
        lr2: 6,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cfg_roundtrip_and_validate() {
        for level in 1..=9 {
            for kind in [Kind::Generic, Kind::Fast] {
                let c = preset(level, kind, 1 << 20);
                let b = c.to_bytes();
                assert_eq!(ModelCfg::from_bytes(&b).unwrap(), c);
            }
        }
        let mut b = preset(5, Kind::Generic, 100).to_bytes();
        b[1] = 40;
        assert!(ModelCfg::from_bytes(&b).is_err());
    }
}
