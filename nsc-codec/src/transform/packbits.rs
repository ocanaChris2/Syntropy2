//! PackBits as TIFF writes it: packets back to bytes, and the same packets again.
//!
//! A PackBits stream is a row of packets — a count and that many bytes as they are, or a count
//! and one byte to repeat — and its bytes can be packed in more ways than one: where a literal
//! ends, whether two equal bytes are worth a packet of their own. Nearly every TIFF comes from
//! one encoder (libtiff's), which packs each row of the picture by itself. [`decode`] reads the
//! packets and then runs that encoder over the rows they stand for; where a row comes out as the
//! stream has it, the record says so with a count, and where it does not, the record keeps the
//! packets' counts. From the bytes and the record [`encode`] writes the stream again, bit for
//! bit, whichever encoder wrote it.

fn put_var(o: &mut Vec<u8>, v: usize) { crate::nsc3::format::put_var(o, v as u64) }
fn get_var(b: &[u8], p: &mut usize) -> Option<usize> { crate::recode::h264::get_var(b, p) }

/// Whose way of packing a row is expected: libtiff's (TIFF files), or Photoshop's (PSD).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Style {
    Tiff,
    /// With the bytes of a row before the first edge of the document's tiles (a layer that
    /// does not start on one is packed up to the edge first).
    Adobe(usize),
}

/// Append `row` packed the way `style` packs a row.
pub fn pack(row: &[u8], style: Style, out: &mut Vec<u8>) {
    match style {
        Style::Tiff => usual(row, out),
        Style::Adobe(lead) => {
            let lead = lead.min(row.len());
            adobe(&row[..lead], out);
            adobe(&row[lead..], out);
        }
    }
}

/// Append `row` packed the way Photoshop packs a row: 128 bytes at a time, each piece by
/// itself; in a piece, three equal bytes and more are a repeat, and two are one too unless a
/// literal is open, which takes them in.
pub fn adobe(row: &[u8], out: &mut Vec<u8>) {
    for piece in row.chunks(128) {
        let mut lit: Option<usize> = None;
        let mut i = 0usize;
        while i < piece.len() {
            let n = piece[i..].iter().take_while(|&&x| x == piece[i]).count();
            if n >= 3 || (n == 2 && lit.is_none()) {
                out.extend_from_slice(&[(257 - n) as u8, piece[i]]);
                lit = None;
            } else {
                for &x in &piece[i..i + n] {
                    match lit {
                        Some(l) => out[l] += 1,
                        None => {
                            lit = Some(out.len());
                            out.push(0);
                        }
                    }
                    out.push(x);
                }
            }
            i += n;
        }
    }
}

/// Append `row` packed the way libtiff packs a row: runs of two and more as repeats, except a
/// run of two between literals, which joins them.
pub fn usual(row: &[u8], out: &mut Vec<u8>) {
    #[derive(PartialEq)]
    enum State { Base, Literal, Run, LiteralRun }
    let mut state = State::Base;
    // where the count of the literal being written is
    let mut last = 0usize;
    let mut i = 0usize;
    while i < row.len() {
        let b = row[i];
        let mut n = 1usize;
        i += 1;
        while i < row.len() && row[i] == b {
            n += 1;
            i += 1;
        }
        loop {
            match state {
                State::Base | State::Run if n > 1 => {
                    state = State::Run;
                    if n > 128 {
                        out.extend_from_slice(&[0x81, b]);
                        n -= 128;
                        continue;
                    }
                    out.extend_from_slice(&[(257 - n) as u8, b]);
                }
                State::Base | State::Run => {
                    last = out.len();
                    out.extend_from_slice(&[0, b]);
                    state = State::Literal;
                }
                State::Literal if n > 1 => {
                    state = State::LiteralRun;
                    if n > 128 {
                        out.extend_from_slice(&[0x81, b]);
                        n -= 128;
                        continue;
                    }
                    out.extend_from_slice(&[(257 - n) as u8, b]);
                }
                State::Literal => {
                    out[last] += 1;
                    if out[last] == 127 { state = State::Base; }
                    out.push(b);
                }
                State::LiteralRun => {
                    // a literal, two equal bytes, a literal: one literal
                    let at = out.len() - 2;
                    if n == 1 && out[at] == 0xFF && out[last] < 126 {
                        out[last] += 2;
                        state = if out[last] == 127 { State::Base } else { State::Literal };
                        out[at] = out[at + 1];
                    } else {
                        state = State::Run;
                    }
                    continue;
                }
            }
            break;
        }
    }
}

/// A stream read back.
pub struct Decoded {
    pub bytes: Vec<u8>,
    /// What [`encode`] needs beside the bytes.
    pub record: Vec<u8>,
    /// Bytes of the stream its packets take (what follows is not theirs).
    pub consumed: usize,
}

/// The packet at `data[p]`: (bytes it stands for, bytes it takes); None if it is cut short.
#[inline]
fn packet(data: &[u8], p: usize) -> Option<(usize, usize)> {
    let c = *data.get(p)? as usize;
    let (n, takes) = match c {
        0..=127 => (c + 1, c + 2),
        128 => (0, 1),
        _ => (257 - c, 2),
    };
    (p + takes <= data.len()).then_some((n, takes))
}

/// Read the PackBits stream at the start of `data` that stands for exactly `len` bytes, in rows
/// of `row` bytes: None if it does not.
pub fn decode(data: &[u8], row: usize, len: usize, style: Style) -> Option<Decoded> {
    if row == 0 { return None; }
    let mut bytes = Vec::with_capacity(len.min(data.len().saturating_mul(64)));
    let mut p = 0usize;
    while bytes.len() < len {
        let (n, takes) = packet(data, p)?;
        if bytes.len() + n > len { return None; }
        match data[p] {
            0..=127 => bytes.extend_from_slice(&data[p + 1..p + takes]),
            128 => {}
            _ => bytes.resize(bytes.len() + n, data[p + 1]),
        }
        p += takes;
    }
    let consumed = p;
    // the usual encoder over the rows: which of them come out as the stream has them?
    let mut record = vec![];
    let (mut scratch, mut counts) = (vec![], vec![]);
    let (mut at, mut pos) = (0usize, 0usize);
    let mut same = 0usize;
    while pos < len {
        let end = (pos + row).min(len);
        scratch.clear();
        pack(&bytes[pos..end], style, &mut scratch);
        if data[at..consumed].starts_with(&scratch) {
            same += 1;
            at += scratch.len();
            pos = end;
            continue;
        }
        // the stream's own packets, up to where a row ends with one
        counts.clear();
        loop {
            let (n, takes) = packet(data, at)?;
            counts.push(data[at]);
            (pos, at) = (pos + n, at + takes);
            if pos == len || pos % row == 0 && n > 0 { break; }
        }
        put_var(&mut record, same);
        put_var(&mut record, counts.len());
        record.extend_from_slice(&counts);
        same = 0;
    }
    if same > 0 { put_var(&mut record, same); }
    Some(Decoded { bytes, record, consumed })
}

/// Write the stream of `bytes` as `record` says ([`decode`]'s), appending to `out`.
pub fn encode(bytes: &[u8], record: &[u8], row: usize, style: Style, out: &mut Vec<u8>) -> Result<(), String> {
    let bad = || "packbits: bad record (corrupt file)".to_string();
    if row == 0 { return Err(bad()); }
    let (mut rp, mut pos) = (0usize, 0usize);
    while rp < record.len() {
        // rows the usual encoder packs, then packets with the counts given
        let same = get_var(record, &mut rp).ok_or_else(bad)?;
        for _ in 0..same {
            if pos >= bytes.len() || pos % row != 0 { return Err(bad()); }
            let end = (pos + row).min(bytes.len());
            pack(&bytes[pos..end], style, out);
            pos = end;
        }
        if rp == record.len() { break; }
        let n = get_var(record, &mut rp).ok_or_else(bad)?;
        let counts = record.get(rp..rp.checked_add(n).ok_or_else(bad)?).ok_or_else(bad)?;
        rp += n;
        for &c in counts {
            out.push(c);
            match c {
                0..=127 => {
                    let l = c as usize + 1;
                    out.extend_from_slice(bytes.get(pos..pos + l).ok_or_else(bad)?);
                    pos += l;
                }
                128 => {}
                _ => {
                    out.push(*bytes.get(pos).ok_or_else(bad)?);
                    pos += 257 - c as usize;
                    if pos > bytes.len() { return Err(bad()); }
                }
            }
        }
    }
    if pos != bytes.len() { return Err(bad()); }
    Ok(())
}

#[cfg(test)]
pub mod tests {
    use super::*;

    fn rng(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }

    /// Bytes with runs of every length among single ones, as a picture's rows have.
    pub fn picture(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let v = rng(&mut s) as u8;
            match rng(&mut s) % 8 {
                0 => out.extend(std::iter::repeat_n(v, 2)),
                1 => out.extend(std::iter::repeat_n(v, 3 + (rng(&mut s) % 6) as usize)),
                2 if rng(&mut s) % 9 == 0 => out.extend(std::iter::repeat_n(v, 100 + (rng(&mut s) % 400) as usize)),
                _ => out.extend((0..1 + rng(&mut s) % 5).map(|k| v.wrapping_add(k as u8 * 37 + 1))),
            }
        }
        out.truncate(n);
        out
    }

    /// A PackBits stream of `bytes` by a plain encoder: every run of `min` and more a repeat,
    /// rows of `row` bytes apart (0: the rows run on).
    pub fn plain(bytes: &[u8], row: usize, min: usize) -> Vec<u8> {
        let mut out = vec![];
        for r in bytes.chunks(if row == 0 { bytes.len().max(1) } else { row }) {
            let mut i = 0;
            let mut lit: Option<usize> = None;
            while i < r.len() {
                let n = r[i..].iter().take_while(|&&x| x == r[i]).count().min(128);
                if n >= min {
                    out.extend_from_slice(&[(257 - n) as u8, r[i]]);
                    i += n;
                    lit = None;
                } else {
                    for _ in 0..n {
                        match lit.filter(|&l| out[l] < 127) {
                            Some(l) => out[l] += 1,
                            None => {
                                lit = Some(out.len());
                                out.push(0);
                            }
                        }
                        out.push(r[i]);
                        i += 1;
                    }
                }
            }
        }
        out
    }

    fn unpack(data: &[u8]) -> Vec<u8> {
        let (mut out, mut p) = (vec![], 0usize);
        while let Some((n, takes)) = packet(data, p) {
            match data[p] {
                0..=127 => out.extend_from_slice(&data[p + 1..p + takes]),
                128 => {}
                _ => out.resize(out.len() + n, data[p + 1]),
            }
            p += takes;
        }
        assert_eq!(p, data.len());
        out
    }

    #[test]
    fn streams_come_back() {
        for (n, row) in [(0usize, 7usize), (1, 1), (2, 5), (300, 40), (5000, 129), (70_000, 1000), (70_000, 3)] {
            let bytes = picture(n, 3 + n as u64);
            // the usual encoder: its packets are the bytes, and the record a count of rows
            let mut stream = vec![];
            for r in bytes.chunks(row) { usual(r, &mut stream); }
            assert!(unpack(&stream) == bytes, "{n} bytes in rows of {row}: the usual encoder's packets");
            let streams = [stream, plain(&bytes, row, 2), plain(&bytes, row, 3), plain(&bytes, 0, 3)];
            for (k, stream) in streams.iter().enumerate() {
                let what = format!("{n} bytes in rows of {row}, encoder {k}");
                let mut with = stream.clone();
                with.extend_from_slice(b"\x80after");
                let d = decode(&with, row, n, Style::Tiff).unwrap_or_else(|| panic!("{what}: not decoded"));
                assert!(d.bytes == bytes && d.consumed == stream.len(), "{what}: bytes");
                let mut back = vec![];
                encode(&d.bytes, &d.record, row, Style::Tiff, &mut back).unwrap_or_else(|e| panic!("{what}: {e}"));
                assert!(back == *stream, "{what}: stream");
                if k == 0 { assert!(d.record.len() <= 3, "{what}: record of {} bytes", d.record.len()); }
                // not a byte more or less than asked for
                assert!(n == 0 || decode(stream, row, n + 1, Style::Tiff).is_none(), "{what}: more than it holds");
            }
        }
        // Photoshop's way: its own rows cost a count, and every packet is the bytes
        let bytes = picture(9000, 77);
        let mut stream = vec![];
        for r in bytes.chunks(300) { adobe(r, &mut stream); }
        assert!(unpack(&stream) == bytes);
        let d = decode(&stream, 300, 9000, Style::Adobe(0)).unwrap();
        let mut back = vec![];
        encode(&d.bytes, &d.record, 300, Style::Adobe(0), &mut back).unwrap();
        assert!(d.record.len() <= 2 && back == stream);
        // packets that stand for nothing are kept where they are
        let bytes = picture(900, 5);
        let mut stream = vec![0x80];
        for (k, r) in bytes.chunks(100).enumerate() {
            usual(r, &mut stream);
            if k % 3 == 1 { stream.push(0x80); }
        }
        let d = decode(&stream, 100, 900, Style::Tiff).unwrap();
        let mut back = vec![];
        encode(&d.bytes, &d.record, 100, Style::Tiff, &mut back).unwrap();
        assert!(d.bytes == bytes && back == stream[..d.consumed]);
    }

    /// Damaged streams and records: None or an error, or other bytes; never a panic, and
    /// never output without bound.
    #[test]
    fn damaged_streams_and_records_do_not_panic() {
        let mut s = 5u64;
        let bytes = picture(20_000, 21);
        for stream in [plain(&bytes, 250, 2), { let mut v = vec![]; for r in bytes.chunks(250) { usual(r, &mut v); } v }] {
            let d = decode(&stream, 250, bytes.len(), Style::Tiff).unwrap();
            for _ in 0..800 {
                let mut bad = stream.clone();
                for _ in 0..1 + rng(&mut s) % 4 {
                    let i = (rng(&mut s) % bad.len() as u64) as usize;
                    if rng(&mut s) % 5 == 0 { bad.truncate(i.max(1)); } else { bad[i] ^= 1 << (rng(&mut s) % 8); }
                }
                for len in [bytes.len(), bytes.len() / 2] {
                    if let Some(d) = decode(&bad, 250, len, Style::Tiff) {
                        let mut back = vec![];
                        encode(&d.bytes, &d.record, 250, Style::Tiff, &mut back).unwrap();
                        assert!(back == bad[..d.consumed], "a damaged stream that decodes must come back");
                    }
                }
                let mut rec = d.record.clone();
                if rec.is_empty() { continue; }
                let i = (rng(&mut s) % rec.len() as u64) as usize;
                if rng(&mut s) % 4 == 0 { rec.truncate(i); } else { rec[i] ^= 1 << (rng(&mut s) % 8); }
                let mut back = vec![];
                let _ = encode(&d.bytes, &rec, 250, Style::Tiff, &mut back);
                assert!(back.len() <= 3 * bytes.len() + rec.len() + 4096, "{} bytes from a damaged record", back.len());
            }
        }
    }
}
