//! Context-mixing engine: the integer, deterministic bit-prediction core used by the NSC3 format.
//!
//! Every byte that leaves this module is a pure function of (input bytes, [`levels::ModelCfg`]),
//! independent of thread count and SIMD level.
pub mod aac;
pub mod audio;
pub mod binary;
pub mod coder;
pub mod guess;
pub mod indexed;
pub mod num;
pub mod contextmap;
pub mod h264;
pub mod hash;
pub mod jpeg;
pub mod levels;
pub mod matcher;
pub mod mixer;
pub mod models;
pub mod mp3;
pub mod predictor;
pub mod squash;
pub mod state;
pub mod statemap;

use coder::{Decoder, Encoder};
use levels::ModelCfg;
use predictor::{code_bytes, Predictor, ReadBuf, WriteBuf};

/// Compress `data` as one independent unit with model configuration `cfg`, modelled as segment
/// kind `kind` (see `nsc3::format::kind`).
pub fn encode_block(data: &[u8], cfg: &ModelCfg, kind: u8) -> Vec<u8> {
    let mut pred = Predictor::new(cfg);
    pred.set_kind(kind);
    let mut enc = Encoder::new();
    code_bytes(&mut pred, &mut enc, &mut ReadBuf(data), 0..data.len());
    enc.finish()
}

/// Inverse of [`encode_block`]: rebuild `len` bytes from `payload`.
pub fn decode_block(payload: &[u8], len: usize, cfg: &ModelCfg, kind: u8) -> Vec<u8> {
    let mut out = vec![0u8; len];
    let mut pred = Predictor::new(cfg);
    pred.set_kind(kind);
    let mut dec = Decoder::new(payload);
    code_bytes(&mut pred, &mut dec, &mut WriteBuf(&mut out), 0..len);
    out
}

#[cfg(test)]
pub mod tests {
    use super::levels::{preset, Kind};
    use super::*;

    fn rng(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }

    fn samples() -> Vec<Vec<u8>> {
        let mut s = 42u64;
        let mut v: Vec<Vec<u8>> = vec![vec![], vec![7], vec![0, 255], vec![0u8; 5000]];
        v.push((0..20_000).map(|_| rng(&mut s) as u8).collect());
        v.push(b"the quick brown fox jumps over the lazy dog. ".repeat(300));
        // periodic binary records with noise
        v.push((0..30_000u32).map(|i| if i % 12 < 4 { (i / 12) as u8 } else { (rng(&mut s) % 3) as u8 }).collect());
        v
    }

    #[test]
    fn roundtrip_all_levels() {
        for data in samples() {
            for level in [1, 3, 5, 9] {
                for kind in [Kind::Generic, Kind::Fast] {
                    let cfg = preset(level, kind, data.len());
                    for k in [1u8, 3] {
                        let enc = encode_block(&data, &cfg, k);
                        let dec = decode_block(&enc, data.len(), &cfg, k);
                        assert!(dec == data, "roundtrip failed: level {level} {kind:?} seg kind {k} len {}", data.len());
                    }
                    let enc = encode_block(&data, &cfg, 1);
                    let dec = decode_block(&enc, data.len(), &cfg, 1);
                    assert!(dec == data, "roundtrip failed: level {level} {kind:?} len {}", data.len());
                }
            }
        }
    }

    /// The instruction model and the binary model's guessers: both directions agree, and they
    /// pay on what they are for.
    #[test]
    fn code_and_table_models_roundtrip() {
        use crate::nsc3::format::kind;
        let code = crate::x86::sample_code(400);
        let mut sizes = vec![];
        for mask in [0, levels::x86_mask(5)] {
            let mut cfg = preset(5, Kind::Generic, code.len());
            cfg.ctx_mask |= mask;
            let run = |c: &mut dyn FnMut(&mut Predictor)| {
                let mut p = Predictor::new(&cfg);
                p.set_segment(kind::EXE, &[2], 0);
                c(&mut p);
            };
            let (mut enc, mut dec) = (vec![], vec![0u8; code.len()]);
            run(&mut |p| {
                let mut e = Encoder::new();
                code_bytes(p, &mut e, &mut ReadBuf(&code), 0..code.len());
                enc = e.finish();
            });
            run(&mut |p| code_bytes(p, &mut Decoder::new(&enc), &mut WriteBuf(&mut dec), 0..code.len()));
            assert!(dec == code, "code did not decode (context mask {mask:#x})");
            sizes.push(enc.len());
        }
        assert!(sizes[1] * 100 < sizes[0] * 97, "the instruction model: {} -> {} bytes", sizes[0], sizes[1]);
        // 24-byte records: a counter, a constant, a value that steps by 8
        let table: Vec<u8> = (0..4000u64).flat_map(|i| [i * 8 + 0x4_1000, 8, 0x2_0000 + 24 * (i / 3)]).flat_map(|v| v.to_le_bytes()).collect();
        let mut sizes = vec![];
        for mask in [0, levels::bin2_mask(5)] {
            let mut cfg = preset(5, Kind::Generic, table.len());
            cfg.ctx_mask |= mask;
            let enc = encode_block(&table, &cfg, kind::GENERIC);
            assert!(decode_block(&enc, table.len(), &cfg, kind::GENERIC) == table, "table did not decode (context mask {mask:#x})");
            // text keeps decoding with the bit set (the guessers stay out of it)
            let text = b"the quick brown fox jumps over the lazy dog. ".repeat(200);
            let t = encode_block(&text, &cfg, kind::TEXT);
            assert!(decode_block(&t, text.len(), &cfg, kind::TEXT) == text);
            sizes.push(enc.len());
        }
        assert!(sizes[1] * 100 < sizes[0] * 80, "the guessers on a table of counters: {} -> {} bytes", sizes[0], sizes[1]);
    }

    /// A log of lines whose numbers go on from the line before: times that pass, a counter in
    /// steps, a size said twice.
    pub fn sample_log(lines: usize) -> Vec<u8> {
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut rng = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let (mut time, mut seq) = (1_700_000_000u64, 4096u64);
        let mut out = String::new();
        for i in 0..lines {
            time += rng() % 40;
            seq += 16;
            let size = 512 + rng() % 9000;
            match rng() % 3 {
                0 => out.push_str(&format!("{time} worker-{} INFO request {seq} done in {} ms, {size} bytes ({size} sent)\n", i % 4, 3 + rng() % 90)),
                1 => out.push_str(&format!("{time} worker-{} WARN queue {} of 64 at offset {:08}\n", i % 4, rng() % 64, seq * 512)),
                _ => out.push_str(&format!("{time} gc pause {}.{:03} s, heap {} -> {} MB\n", rng() % 3, rng() % 1000, 900 + rng() % 90, 400 + rng() % 50)),
            }
        }
        out.into_bytes()
    }

    /// The number model of text: both directions agree, it pays on a log, and it stays out of
    /// what is not text.
    #[test]
    fn number_model_roundtrips() {
        use crate::nsc3::format::kind;
        let log = sample_log(3000);
        assert!(num::wanted(num::digits(&log), log.len()));
        let mut sizes = vec![];
        for mask in [0, levels::num_mask(5)] {
            let mut cfg = preset(5, Kind::Generic, log.len());
            cfg.ctx_mask |= mask;
            let enc = encode_block(&log, &cfg, kind::TEXT);
            assert!(decode_block(&enc, log.len(), &cfg, kind::TEXT) == log, "log did not decode (context mask {mask:#x})");
            sizes.push(enc.len());
            // binary data and a text without a digit, with the bit set
            let table: Vec<u8> = (0..3000u32).flat_map(|i| (i * 12).to_le_bytes()).collect();
            let t = encode_block(&table, &cfg, kind::GENERIC);
            assert!(decode_block(&t, table.len(), &cfg, kind::GENERIC) == table);
            let text = b"the quick brown fox jumps over the lazy dog. ".repeat(200);
            let t = encode_block(&text, &cfg, kind::TEXT);
            assert!(decode_block(&t, text.len(), &cfg, kind::TEXT) == text);
        }
        assert!(sizes[1] * 100 < sizes[0] * 90, "the number model on a log: {} -> {} bytes", sizes[0], sizes[1]);
        assert!(!num::wanted(num::digits(b"no numbers but 1 here"), 21) && !num::wanted(0, 0));
    }

    #[test]
    fn compresses_redundancy() {
        let data = b"abcdefgh".repeat(10_000);
        let enc = encode_block(&data, &preset(5, Kind::Generic, data.len()), 1);
        assert!(enc.len() < 400, "80 KB of a repeated 8-byte pattern took {} bytes", enc.len());
        let zeros = vec![0u8; 1 << 20];
        let enc = encode_block(&zeros, &preset(1, Kind::Generic, zeros.len()), 1);
        assert!(enc.len() < 300, "1 MiB of zeros took {} bytes", enc.len());
    }
}
