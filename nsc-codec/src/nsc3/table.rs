//! The segment table of a block header, coded.
//!
//! A block that holds many streams — an archive of small files, a PDF of small objects — has a
//! segment for each stream and each of its parts, and the table that lists them runs to a few
//! percent of what the block compresses to. The entries are much alike, so a writer for decoders
//! of level 2 codes the table with the byte model at a fixed, small configuration: it comes out
//! five to nine times smaller. The configuration is part of the format.
use crate::cm::coder::{DecoderOf, Encoder};
use crate::cm::levels::{ctx, sel, Kind, ModelCfg};
use crate::cm::predictor::{code_bytes, Predictor, ReadBuf, WriteBuf};
use crate::nsc3::format::kind;

/// Tables below this size are written as they are: not worth setting a model up for.
pub const MIN: usize = 96;
/// Largest table that is coded (and that a decoder accepts coded).
pub const MAX: usize = 8 << 20;

fn cfg() -> ModelCfg {
    let ids = [ctx::O1, ctx::O2, ctx::O3, ctx::O4, ctx::O6, ctx::REC, ctx::SP_B2B3, ctx::MATCH];
    ModelCfg {
        kind: Kind::Generic,
        hash_bits: 16,
        mm_bits: 16,
        ctx_mask: ids.iter().fold(0, |m, &i| m | 1u64 << i),
        sel_mask: 1 << sel::C0 | 1 << sel::MATCH | 1 << sel::C1,
        apm_mask: 0b001,
        lr: 6,
        lr2: 6,
    }
}

/// The coded form of `table`.
pub fn pack(table: &[u8]) -> Vec<u8> {
    let mut pred = Predictor::new(&cfg());
    pred.set_segment(kind::GENERIC, &[], 0);
    let mut enc = Encoder::new();
    code_bytes(&mut pred, &mut enc, &mut ReadBuf(table), 0..table.len());
    enc.finish()
}

/// The table of `len` bytes that `packed` is the coded form of. Damaged input gives other
/// bytes, which the reader of the table then refuses.
pub fn unpack(packed: &[u8], len: usize) -> Result<Vec<u8>, String> {
    if len > MAX { return Err("segment table too large (corrupt file)".into()); }
    let mut pred = Predictor::new(&cfg());
    pred.set_segment(kind::GENERIC, &[], 0);
    let mut dec = DecoderOf::new(packed);
    let mut table = vec![0u8; len];
    code_bytes(&mut pred, &mut dec, &mut WriteBuf(&mut table), 0..len);
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_come_back_smaller() {
        // entries as an archive of small files has them
        let mut table = vec![];
        let mut s = 9u64;
        for _ in 0..800 {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            table.extend_from_slice(&[8, 0x10, 0x80 | (s % 128) as u8, 3 + (s >> 8) as u8 % 4, 4, 2, 0, 0, 0, 3, 0x10, 0x80 | (s >> 16) as u8, 9, 0, 1, 0x10, 10, 0, 1, 0x10, 0x53, 0]);
        }
        let packed = pack(&table);
        assert!(packed.len() * 4 < table.len(), "{} bytes from {}", packed.len(), table.len());
        assert!(unpack(&packed, table.len()).unwrap() == table);
        for t in [&b""[..], b"x", &[0u8; 300]] { assert!(unpack(&pack(t), t.len()).unwrap() == t); }
        assert!(unpack(&packed, MAX + 1).is_err());
        // damaged: other bytes, of the length asked for
        let mut bad = packed.clone();
        bad[packed.len() / 2] ^= 4;
        assert!(unpack(&bad, table.len()).unwrap() != table);
        assert_eq!(unpack(&packed[..10], table.len()).unwrap().len(), table.len());
    }
}
