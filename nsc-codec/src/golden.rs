//! Golden files: the format-1 contract. For each case, a deterministic input is compressed and
//! compared byte for byte with `tests/golden/<case>.nsc` (the encoder must not drift), and the
//! golden file must decode to the input (every later build keeps reading format 1). Run with
//! `UPDATE_GOLDEN=1` to (re)create the files after an intended, versioned format change.
#![cfg(test)]
use crate::crc::crc32;
use crate::nsc3::{self, sched, EncOpts};
use std::path::PathBuf;

fn rng(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }

fn text() -> Vec<u8> {
    let words = ["the", "model", "mixes", "contexts", "of", "every", "kind", "and", "codes", "bits", "with", "an", "arithmetic", "coder", "while", "blocks", "run", "in", "parallel"];
    let mut s = 1u64;
    let mut out = vec![];
    while out.len() < 20_000 {
        let n = 5 + rng(&mut s) % 12;
        for i in 0..n {
            let w = words[(rng(&mut s) % words.len() as u64) as usize];
            if i == 0 { out.extend(w[..1].to_uppercase().bytes()); out.extend(&w.as_bytes()[1..]); } else { out.extend(w.bytes()); }
            out.push(if i + 1 == n { b'.' } else { b' ' });
        }
        out.extend(if rng(&mut s) % 5 == 0 { &b"\n\n"[..] } else { &b" "[..] });
    }
    out
}

fn records() -> Vec<u8> {
    let mut s = 2u64;
    (0..800u32).flat_map(|i| {
        let mut r = vec![];
        r.extend_from_slice(&i.to_le_bytes());
        r.extend_from_slice(&((i * 37) % 1000).to_le_bytes());
        r.extend_from_slice(&(rng(&mut s) as u32 & 0xff).to_le_bytes());
        r.extend_from_slice(b"ITEM");
        r.extend_from_slice(&[(i % 7) as u8; 8]);
        r
    }).collect()
}

fn pgm() -> Vec<u8> {
    let mut s = 3u64;
    let mut out = b"P5\n96 64\n255\n".to_vec();
    for y in 0..64u32 { for x in 0..96u32 { out.push(((x * 2 + y * 3) as u64 + rng(&mut s) % 6) as u8); } }
    out
}

fn wav() -> Vec<u8> {
    let mut s = 4u64;
    let frames = 11_025u32;
    let mut d = vec![];
    for n in 0..frames {
        for c in 0..2u32 {
            let v = ((n as f64 * 0.05 * (c + 1) as f64).sin() * 9000.0) as i32 + (rng(&mut s) % 64) as i32 - 32;
            d.extend_from_slice(&(v as i16).to_le_bytes());
        }
    }
    let mut w = b"RIFF".to_vec();
    w.extend_from_slice(&(36 + d.len() as u32).to_le_bytes());
    w.extend_from_slice(b"WAVEfmt ");
    w.extend_from_slice(&16u32.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes());
    w.extend_from_slice(&2u16.to_le_bytes());
    w.extend_from_slice(&22050u32.to_le_bytes());
    w.extend_from_slice(&(22050u32 * 4).to_le_bytes());
    w.extend_from_slice(&4u16.to_le_bytes());
    w.extend_from_slice(&16u16.to_le_bytes());
    w.extend_from_slice(b"data");
    w.extend_from_slice(&(d.len() as u32).to_le_bytes());
    w.extend(d);
    w
}

fn gzip() -> Vec<u8> {
    let t = text();
    let mut g = vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3];
    g.extend(crate::transform::deflate::deflate_like_zlib(&t[..16_000], 6));
    g.extend_from_slice(&crc32(&t[..16_000]).to_le_bytes());
    g.extend_from_slice(&16_000u32.to_le_bytes());
    g
}

fn png() -> Vec<u8> {
    use crate::transform::png::{refilter, Geom};
    let (w, h) = (40u32, 30u32);
    let g = Geom::new(w, h, 8, 2, 0).unwrap();
    let pix: Vec<u8> = (0..h).flat_map(|y| (0..w).flat_map(move |x| [(x * 6) as u8, (y * 8) as u8, ((x + y) * 3) as u8])).collect();
    let types: Vec<u8> = (0..h).map(|r| (r % 5) as u8).collect();
    let mut scan = vec![];
    refilter(&types, &pix, &g, &mut scan).unwrap();
    let (mut a, mut b) = (1u32, 0u32);
    for &x in &scan { a = (a + x as u32) % 65521; b = (b + a) % 65521; }
    let mut z = vec![0x78, 0x9c];
    z.extend(crate::transform::deflate::deflate_like_zlib(&scan, 6));
    z.extend_from_slice(&(b << 16 | a).to_be_bytes());
    let chunk = |ty: &[u8], data: &[u8]| {
        let mut c = (data.len() as u32).to_be_bytes().to_vec();
        c.extend_from_slice(ty);
        c.extend_from_slice(data);
        let crc = crc32(&c[4..]);
        c.extend_from_slice(&crc.to_be_bytes());
        c
    };
    let mut ihdr = w.to_be_bytes().to_vec();
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    let mut f = b"\x89PNG\r\n\x1a\n".to_vec();
    f.extend(chunk(b"IHDR", &ihdr));
    for c in z.chunks(700) { f.extend(chunk(b"IDAT", c)); }
    f.extend(chunk(b"IEND", &[]));
    f
}

/// A small baseline grayscale JPEG built from synthetic coefficients with our own encoder.
fn jpeg() -> Vec<u8> {
    let (w, h) = (64u16, 48u16);
    let mut hd = vec![0xff, 0xd8];
    // quantization table 0 (all 4)
    hd.extend_from_slice(&[0xff, 0xdb, 0, 67, 0]);
    hd.extend_from_slice(&[4u8; 64]);
    // frame: 8-bit, h × w, one component (id 1, 1×1, table 0)
    hd.extend_from_slice(&[0xff, 0xc0, 0, 11, 8]);
    hd.extend_from_slice(&h.to_be_bytes());
    hd.extend_from_slice(&w.to_be_bytes());
    hd.extend_from_slice(&[1, 1, 0x11, 0]);
    // Huffman tables with every symbol the coefficients below need, all codes 4 bits long
    let dc: Vec<u8> = (0..12).collect();
    let ac: Vec<u8> = vec![0x00, 0xf0, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16];
    for (class, syms) in [(0u8, &dc), (0x10, &ac)] {
        let mut seg = vec![class];
        let mut counts = [0u8; 16];
        counts[3] = syms.len() as u8;
        seg.extend_from_slice(&counts);
        seg.extend_from_slice(syms);
        hd.extend_from_slice(&[0xff, 0xc4]);
        hd.extend_from_slice(&((seg.len() + 2) as u16).to_be_bytes());
        hd.extend(seg);
    }
    hd.extend_from_slice(&[0xff, 0xda, 0, 8, 1, 1, 0x00, 0, 63, 0]);
    hd.extend_from_slice(&[0xff, 0xd9]);
    // coefficients: nonzeros at the start of each block (runs of at most one zero), sizes ≤ 6
    let mut s = 5u64;
    let mut coefs = vec![];
    let blocks = (w as usize / 8) * (h as usize / 8);
    let mut dc_v = 0i32;
    for _ in 0..blocks {
        dc_v = (dc_v + (rng(&mut s) % 41) as i32 - 20).clamp(-500, 500);
        let mut z = vec![dc_v];
        let n = 1 + rng(&mut s) % 12;
        for i in 0..n {
            if i % 3 == 2 { z.push(0); }
            let m = 1 + (rng(&mut s) % (40 / (i + 1) + 1)) as i32;
            z.push(if rng(&mut s) & 1 == 0 { m } else { -m });
        }
        coefs.push(z.len() as u8);
        for v in z {
            let mut u = ((v << 1) ^ (v >> 31)) as u32;
            while u >= 0x80 { coefs.push(u as u8 | 0x80); u >>= 7; }
            coefs.push(u as u8);
        }
    }
    crate::transform::jpeg::rebuild(&hd, &coefs, &[]).unwrap()
}

/// A real MP3 stream (LAME, generated from a synthetic signal) between other data: MPEG-1 joint
/// stereo, then MPEG-2 mono.
fn mp3() -> Vec<u8> {
    let mut out = b"ID3 is not here, only text before the frames\n".to_vec();
    out.extend_from_slice(include_bytes!("../tests/fixtures/tone_stereo.mp3"));
    out.extend_from_slice(b"between the two streams");
    out.extend_from_slice(include_bytes!("../tests/fixtures/tone_mono_lsf.mp3"));
    out
}

/// An ADTS AAC stream between other data.
fn aac() -> Vec<u8> {
    let mut out = b"text before the stream\n".to_vec();
    out.extend_from_slice(include_bytes!("../tests/fixtures/tone_adts.aac"));
    out.extend_from_slice(b"and after it");
    out
}

/// An x86-64 executable: an ELF header, one segment of code, a table after it.
fn exe() -> Vec<u8> {
    let code = crate::x86::sample_code(600);
    let mut data = vec![0u8; 0x200];
    data[..8].copy_from_slice(b"\x7fELF\x02\x01\x01\0");
    (data[16], data[18], data[32], data[54], data[56], data[64], data[68]) = (2, 62, 64, 56, 1, 1, 5);
    data[72..80].copy_from_slice(&0x200u64.to_le_bytes());
    data[96..104].copy_from_slice(&(code.len() as u64).to_le_bytes());
    data.extend_from_slice(&code);
    data.extend((0..1500u32).flat_map(|i| (i * 16).to_le_bytes()));
    data
}

/// The cases of decoder level 2: those of level 1, the formats its recoders handle, and what its
/// models are for (machine code, a log full of numbers, a database of compressed documents,
/// FLAC frames, a GIF animation, a TIFF of three pictures, a gzip member of another encoder than zlib). The MP4 and Matroska files are
/// compressed with their index, as `nsc compress` does.
fn cases2() -> Vec<(&'static str, Vec<u8>)> {
    let mut c = cases();
    c.push(("mp3", mp3()));
    c.push(("aac", aac()));
    c.push(("m4a", include_bytes!("../tests/fixtures/tone.m4a").to_vec()));
    c.push(("mp4", include_bytes!("../tests/fixtures/tone_video.mp4").to_vec()));
    c.push(("mp4frag", include_bytes!("../tests/fixtures/tone_fragmented.mp4").to_vec()));
    c.push(("h264", include_bytes!("../tests/fixtures/video_cabac.mp4").to_vec()));
    c.push(("h264vlc", include_bytes!("../tests/fixtures/video_cavlc.mp4").to_vec()));
    c.push(("mkv", include_bytes!("../tests/fixtures/tone_video.mkv").to_vec()));
    c.push(("exe", exe()));
    c.push(("sqlite", {
        use crate::recode::sqlite::tests::{blob, database, orders, table, zlib};
        // a table in pages that have been in use, and after it a database of compressed documents
        let docs: Vec<Vec<u8>> = (0..4).map(|i| zlib(&blob(6000 + 2500 * i, 40 + i as u64, true))).collect();
        let refs: Vec<&[u8]> = docs.iter().map(|d| &d[..]).collect();
        [table(&orders(600), 1024, true), database(&refs, 512, true)].concat()
    }));
    c.push(("log", crate::cm::tests::sample_log(700)));
    c.push(("flac", [&b"fLaC, then the frames: "[..], &crate::recode::flac::tests::stream(14, 2, 16)].concat()));
    c.push(("gif", crate::recode::gif::tests::animation(6, 70, 50, 8).0));
    {
        // a gzip member whose deflate stream is not zlib's
        use crate::transform::deflate::front::tests::{foreign, text};
        use crate::transform::deflate::{writer, Parse};
        let content = text(30_000, 21);
        let parse = foreign(&content, 3, 2500, 3000, 4);
        let mut gz = vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3];
        gz.extend((0..8).find_map(|pb| writer::write(&Parse { end_pad: (pb, 0), ..parse.clone() }, &content).ok()).unwrap());
        gz.extend_from_slice(&crate::crc::crc32(&content).to_le_bytes());
        gz.extend_from_slice(&(content.len() as u32).to_le_bytes());
        c.push(("gzip-other", gz));
        c.push(("type1", crate::recode::type1::tests::font(150, 3)));
        c.push(("ljpeg", crate::recode::ljpeg::tests::stream(90, 60, 2, 12, 4, 7).0));
        c.push(("psd", crate::recode::psd::tests::document(150, 70, 8, 2, false, 3).0));
        c.push(("zip-archive", crate::recode::zip::tests::archive(9, 3)));
    }
    {
        use crate::recode::tiff::tests::{file, Layout, RGB};
        let l = Layout { width: 120, height: 90, rows_per_strip: 16, predictor: 2, ..RGB };
        c.push(("tiff", file(&[l, Layout { compression: 32773, predictor: 1, ..l }, Layout { compression: 8, samples: 1, bits: 16, ..l }], true, false, 3).0));
    }
    c
}

/// Compress as `nsc compress` does for a file: with its container index, if it has one.
fn compress(input: &[u8], opts: &EncOpts, jobs: usize) -> Vec<u8> {
    let indexes = crate::recode::index(&mut std::io::Cursor::new(input), input.len() as u64).map(|i| vec![(0, i)]).unwrap_or_default();
    let mut enc = vec![];
    sched::compress_meta(input, &mut enc, opts, jobs, 1 << 20, u64::MAX, sched::Meta { indexes, ..Default::default() }).unwrap();
    enc
}

/// The cases of decoder level 1 (nsc 0.3), as (name, input).
fn cases() -> Vec<(&'static str, Vec<u8>)> {
    let mut mixed = text();
    mixed.extend(jpeg());
    mixed.extend(gzip());
    mixed.extend(png());
    mixed.extend(records());
    vec![("text", text()), ("records", records()), ("image", pgm()), ("audio", wav()), ("gzip", gzip()), ("png", png()), ("jpeg", jpeg()), ("mixed", mixed)]
}

fn golden_dir() -> PathBuf { PathBuf::from("tests/golden") }

#[test]
fn golden_files_decode_and_reencode_identically() {
    let update = std::env::var_os("UPDATE_GOLDEN").is_some();
    let no_model = |b: &[(String, u64)]| -> Result<Option<std::sync::Arc<nsc3::NeuralCtx>>, String> {
        if b.is_empty() { Ok(None) } else { Err("neural".into()) }
    };
    for (name, input) in cases2() {
        for level in [1u32, 5] {
            let path = golden_dir().join(format!("{name}.L{level}.nsc"));
            let enc = compress(&input, &EncOpts::new(level, true, None), 1);
            let enc3 = compress(&input, &EncOpts::new(level, true, None), 3);
            assert!(enc == enc3, "{name} L{level}: output depends on --jobs");
            // lanes verified while they are coded (a second model per lane)
            let encc = compress(&input, &EncOpts::new(level, true, None).with_lane_jobs(4), 4);
            assert!(enc == encc, "{name} L{level}: output depends on concurrent verification");
            if update {
                std::fs::create_dir_all(golden_dir()).unwrap();
                std::fs::write(&path, &enc).unwrap();
                continue;
            }
            let Ok(gold) = std::fs::read(&path) else { panic!("missing golden file {} (run with UPDATE_GOLDEN=1 once)", path.display()) };
            let mut dec = vec![];
            sched::decompress(&gold[..], &mut dec, 2, u64::MAX, &no_model).unwrap_or_else(|e| panic!("{name} L{level}: golden file no longer decodes: {e}"));
            assert!(dec == input, "{name} L{level}: golden file decodes to different bytes");
            assert!(enc == gold, "{name} L{level}: the encoder's output drifted from the golden file ({} vs {} bytes)", enc.len(), gold.len());
        }
    }
}

/// Files written by nsc 0.3 (decoder level 1) keep decoding, and `--compat 1` still writes
/// exactly them.
#[test]
fn level1_files_decode_and_compat_writes_them() {
    let no_model = |b: &[(String, u64)]| -> Result<Option<std::sync::Arc<nsc3::NeuralCtx>>, String> {
        if b.is_empty() { Ok(None) } else { Err("neural".into()) }
    };
    for (name, input) in cases() {
        for level in [1u32, 5] {
            let gold = std::fs::read(golden_dir().join(format!("level1/{name}.L{level}.nsc"))).expect("missing level-1 golden file");
            let (h, _) = nsc3::format::parse_header(&gold).unwrap();
            assert_eq!(h.min_decoder, 1, "{name} L{level}: not a level-1 file");
            let mut dec = vec![];
            sched::decompress(&gold[..], &mut dec, 2, u64::MAX, &no_model).unwrap_or_else(|e| panic!("{name} L{level}: level-1 file no longer decodes: {e}"));
            assert!(dec == input, "{name} L{level}: level-1 file decodes to different bytes");
            let mut opts = EncOpts::new(level, true, None);
            opts.compat = true;
            let mut enc = vec![];
            sched::compress(&input[..], &mut enc, &opts, 2, 1 << 20, u64::MAX).unwrap();
            assert!(enc == gold, "{name} L{level}: --compat 1 no longer writes the level-1 file ({} vs {} bytes)", enc.len(), gold.len());
        }
    }
    // an MP3 written for level 1 uses none of the recoders' kinds
    let mut opts = EncOpts::new(5, true, None);
    opts.compat = true;
    let mut enc = vec![];
    let st = sched::compress(&mp3()[..], &mut enc, &opts, 2, 1 << 20, u64::MAX).unwrap();
    assert_eq!(nsc3::format::parse_header(&enc).unwrap().0.min_decoder, 1);
    assert_eq!(st.kinds & !(nsc3::format::kind::LEVEL1 as u128), 0, "level-1 file with kinds {:#x}", st.kinds);
}

#[test]
fn golden_archive() {
    use crate::archive;
    use std::os::unix::fs::PermissionsExt;
    let update = std::env::var_os("UPDATE_GOLDEN").is_some();
    let dir = std::env::temp_dir().join(format!("nsc-golden-archive-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let tree = dir.join("tree");
    std::fs::create_dir_all(tree.join("sub")).unwrap();
    let fixed = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_750_000_000);
    for (p, data) in [("a.txt", text()), ("sub/b.bin", records()), ("sub/c.txt", text())] {
        let f = tree.join(p);
        std::fs::write(&f, &data).unwrap();
        std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o644)).unwrap();
        std::fs::File::options().write(true).open(&f).unwrap().set_modified(fixed).unwrap();
    }
    for d in [tree.join("sub"), tree.clone()] {
        std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::File::open(&d).unwrap().set_modified(fixed).unwrap();
    }
    // the current archive, and the one nsc 0.3 wrote (`--compat 1` must still write exactly it)
    for (compat, file) in [(false, "tree.nsca"), (true, "level1/tree.nsca")] {
        let out = dir.join("g.nsca");
        let mut opts = EncOpts::new(5, true, None);
        opts.compat = compat;
        archive::pack(out.to_str().unwrap(), &[(tree.clone(), "tree".into())], &opts, 2, 1 << 20, None, true).unwrap();
        let bytes = std::fs::read(&out).unwrap();
        let path = golden_dir().join(file);
        if update && !compat {
            std::fs::create_dir_all(golden_dir()).unwrap();
            std::fs::write(&path, &bytes).unwrap();
        } else if !update {
            let gold = std::fs::read(&path).expect("missing golden archive (run with UPDATE_GOLDEN=1 once)");
            assert!(bytes == gold, "{file}: archive output drifted from the golden file");
            let dest = dir.join(if compat { "out1" } else { "out" });
            archive::unpack(path.to_str().unwrap(), dest.to_str().unwrap(), None, false, 2, None).unwrap();
            assert_eq!(std::fs::read(dest.join("tree/sub/c.txt")).unwrap(), text());
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
