use super::*;

/// Deflate streams found in real files: gzip members, zip entries (method 8), PNG image data.
fn corpus_streams() -> Vec<(String, Vec<u8>)> {
    let mut out = vec![];
    let dir = "bench/corpus/deflate";
    let Ok(rd) = std::fs::read_dir(dir) else { return out };
    let mut files: Vec<_> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    files.sort();
    for f in files {
        let Ok(b) = std::fs::read(&f) else { continue };
        let name = f.display().to_string();
        // gzip members anywhere (tar of .gz files included)
        let mut i = 0;
        while i + 18 < b.len() && out.len() < 400 {
            if b[i] == 0x1f && b[i + 1] == 0x8b && b[i + 2] == 8 {
                let flg = b[i + 3];
                let mut o = i + 10;
                if flg & 4 != 0 { o += 2 + u16::from_le_bytes([b[o], b[o + 1]]) as usize; }
                if flg & 8 != 0 { while o < b.len() && b[o] != 0 { o += 1; } o += 1; }
                if flg & 16 != 0 { while o < b.len() && b[o] != 0 { o += 1; } o += 1; }
                if flg & 2 != 0 { o += 2; }
                if o < b.len() { out.push((format!("{name}@{i} gzip"), b[o..].to_vec())); }
                i = o.max(i + 1);
            } else if &b[i..i + 4] == b"PK\x03\x04" && u16::from_le_bytes([b[i + 8], b[i + 9]]) == 8 {
                let n = u16::from_le_bytes([b[i + 26], b[i + 27]]) as usize;
                let e = u16::from_le_bytes([b[i + 28], b[i + 29]]) as usize;
                let o = i + 30 + n + e;
                if o < b.len() { out.push((format!("{name}@{i} zip"), b[o..].to_vec())); }
                i = o.max(i + 1);
            } else { i += 1; }
        }
        if b.starts_with(b"\x89PNG") {
            let mut idat = vec![];
            let mut o = 8;
            while o + 12 <= b.len() {
                let len = u32::from_be_bytes(b[o..o + 4].try_into().unwrap()) as usize;
                if &b[o + 4..o + 8] == b"IDAT" { idat.extend_from_slice(&b[o + 8..o + 8 + len]); }
                o += 12 + len;
            }
            if idat.len() > 2 { out.push((format!("{name} png"), idat[2..].to_vec())); }
        }
    }
    out
}

#[test]
fn rewrite_is_bit_exact_on_real_streams() {
    let streams = corpus_streams();
    if streams.is_empty() { eprintln!("deflate test: corpus missing — skipping"); return; }
    let (mut ok, mut bad, mut skipped) = (0, 0, 0);
    for (name, s) in &streams {
        match inflate::inflate(s, 1 << 28) {
            Ok((data, parse)) => {
                let back = writer::write(&parse, &data).expect("writer failed");
                if back == s[..parse.consumed] { ok += 1; } else { bad += 1; eprintln!("MISMATCH {name}"); }
            }
            Err(_) => skipped += 1,
        }
    }
    eprintln!("deflate rewrite: {ok} exact, {bad} mismatched, {skipped} not deflate");
    assert_eq!(bad, 0);
    assert!(ok > 10, "too few streams exercised ({ok})");
}

#[test]
fn stored_and_fixed_blocks() {
    // stored block "hi" (final), then the same through a fixed block written by hand via the writer
    let raw = [0x01u8, 0x02, 0x00, 0xfd, 0xff, b'h', b'i'];
    let (d, p) = inflate::inflate(&raw, 100).unwrap();
    assert_eq!(d, b"hi");
    assert_eq!(writer::write(&p, &d).unwrap(), raw);
    let p2 = Parse { blocks: vec![Block { last: true, kind: Kind::Fixed, toks: vec![Tok::Lit(b'a'), Tok::Match { len: 5, dist: 1, alt: false }] }], end_pad: (2, 0), consumed: 0 };
    let bytes = writer::write(&p2, b"aaaaaa").unwrap();
    let (d2, p3) = inflate::inflate(&bytes, 100).unwrap();
    assert_eq!(d2, b"aaaaaa");
    assert_eq!(p3.blocks, p2.blocks);
}

#[test]
fn zlib_emulation_predicts_real_streams() {
    let streams = corpus_streams();
    if streams.is_empty() { return; }
    let mut report = vec![];
    let pick = |tag: &'static str, k: usize| streams.iter().filter(move |(n, _)| n.ends_with(tag)).take(k);
    for (name, st) in pick(" png", 4).chain(pick(" gzip", 4)).chain(streams.iter().filter(|(n, _)| n.ends_with(" zip")).step_by(7).take(14)) {
        let Ok((data, parse)) = inflate::inflate(st, 1 << 28) else { continue };
        let toks: Vec<Tok> = parse.blocks.iter().flat_map(|b| b.toks.iter().copied()).collect();
        if toks.len() < 100 { continue; }
        let mut best = (0usize, zsim::Params { level: 6, filtered: false, wbits: 15, mem_level: 8 });
        for level in 1..=9 {
            for filtered in [false, true] {
                let p = zsim::Params { level, filtered, wbits: 15, mem_level: 8 };
                let a = zsim::agreement(&data, &toks, p, 20_000);
                if a > best.0 { best = (a, p); }
            }
        }
        let n = toks.len().min(20_000);
        report.push(format!("{:5.1}%  L{}{}  {}", 100.0 * best.0 as f64 / n as f64, best.1.level, if best.1.filtered { "f" } else { "" }, name));
    }
    for r in &report { eprintln!("{r}"); }
}

#[test]
fn zlib_trees_predict_real_headers() {
    let streams = corpus_streams();
    if streams.is_empty() { return; }
    let (mut dyn_blocks, mut predicted) = (0usize, 0usize);
    for (_, st) in streams.iter().take(400) {
        let Ok((_, parse)) = inflate::inflate(st, 1 << 28) else { continue };
        for b in &parse.blocks {
            if let Kind::Dynamic { .. } = b.kind {
                dyn_blocks += 1;
                if ztrees::zlib_header(&b.toks) == b.kind { predicted += 1; }
            }
        }
    }
    eprintln!("dynamic headers predicted: {predicted}/{dyn_blocks}");
    assert!(predicted * 10 >= dyn_blocks * 9, "zlib tree emulation too weak: {predicted}/{dyn_blocks}");
}

#[test]
fn side_info_roundtrip_and_size() {
    let streams = corpus_streams();
    if streams.is_empty() { return; }
    let (mut n, mut comp, mut side_total, mut raw_total) = (0usize, 0usize, 0usize, 0usize);
    for (name, st) in streams.iter().take(400) {
        let Ok((data, parse)) = inflate::inflate(st, 1 << 28) else { continue };
        let side = side::encode(&data, &parse, 15);
        let back = side::decode(&data, &side).unwrap_or_else(|e| panic!("{name}: {e}"));
        let bytes = writer::write(&back, &data).unwrap();
        assert!(bytes == st[..parse.consumed], "{name}: side-info reconstruction not bit-exact");
        n += 1;
        comp += parse.consumed;
        side_total += side.len();
        raw_total += data.len();
    }
    eprintln!("side info: {n} streams, {comp} compressed bytes -> {side_total} side bytes ({:.3}%), {raw_total} raw bytes",
              100.0 * side_total as f64 / comp.max(1) as f64);
}
