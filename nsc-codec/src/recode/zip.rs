//! ZIP archives: what the entries' headers repeat, derived.
//!
//! Every entry of a ZIP carries the CRC-32 of its content and its two sizes in its local
//! header, again in the central directory at the end, and sometimes a third time in a data
//! descriptor: twelve bytes each time that no model predicts, and that follow from the
//! content. [`split`] takes an archive apart into its entries' data — deflate streams to be
//! recompressed, stored content to be modelled as what it is — and a skeleton: everything
//! else, with those fields blanked wherever they are what the content gives. [`rebuild`]
//! puts the data back between the headers and fills the fields in again, from the content.
use crate::crc::crc32;
use crate::transform::deflate::scan::Deflate;
use crate::transform::deflate::{inflate, scan};

const VERSION: u8 = 1;
/// Fewer entries than this are not worth a container; more than this are not walked.
const MIN_ENTRIES: usize = 2;
const MAX_ENTRIES: usize = 1 << 20;
/// A decoder inflates an entry to derive its fields: up to this much.
const MAX_INFLATE: usize = 1 << 30;

fn put_var(o: &mut Vec<u8>, v: usize) { crate::nsc3::format::put_var(o, v as u64) }
fn get_var(b: &[u8], p: &mut usize) -> Option<usize> { super::h264::get_var(b, p) }
fn le16(b: &[u8], o: usize) -> Option<usize> { Some(u16::from_le_bytes([*b.get(o)?, *b.get(o + 1)?]) as usize) }
fn le32(b: &[u8], o: usize) -> Option<usize> { Some(u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?) as usize) }

/// An entry's data, as it is taken.
pub enum Data {
    /// A deflate stream with its content and record.
    Deflate(Deflate),
    /// Stored content (to be planned like any content), or data left as it is.
    Stored(Vec<u8>),
    Raw(Vec<u8>),
}

/// An archive taken apart.
pub struct Split {
    pub len: usize,
    pub record: Vec<u8>,
    pub skeleton: Vec<u8>,
    pub data: Vec<Data>,
    /// Fields blanked (of four bytes each).
    pub derived: usize,
}

/// What an entry's content gives: its CRC-32, packed size, unpacked size.
fn facts_of(method: usize, data: &[u8]) -> Option<[u32; 3]> {
    match method {
        0 => Some([crc32(data), data.len() as u32, data.len() as u32]),
        8 => {
            let (inner, parse) = inflate::inflate(data, MAX_INFLATE).ok()?;
            (parse.consumed == data.len() && inner.len() <= u32::MAX as usize).then(|| [crc32(&inner), data.len() as u32, inner.len() as u32])
        }
        _ => None,
    }
}

/// Take the ZIP archive at the start of `b` apart, its streams' content within `budget` bytes.
/// `any`: deflate streams of other encoders than zlib are recompressed too.
pub fn split(b: &[u8], budget: usize, any: bool) -> Option<Split> {
    if !b.starts_with(b"PK\x03\x04") { return None; }
    let mut sp = Split { len: 0, record: vec![VERSION], skeleton: vec![], data: vec![], derived: 0 };
    // per entry: where its local header is, its method, and what its content gives
    let mut entries: Vec<(usize, usize, Option<[u32; 3]>)> = vec![];
    // where each entry's data goes in the skeleton; the fields to fill: (skeleton offset, which, entry)
    let mut places: Vec<usize> = vec![];
    let mut patches: Vec<(usize, u8, usize)> = vec![];
    let mut left = budget;
    let mut o = 0usize;
    // a field of the archive that is what the content gives is blanked, and noted
    let mut field = |sk: &mut Vec<u8>, patches: &mut Vec<(usize, u8, usize)>, value: usize, which: u8, e: usize, fact: Option<[u32; 3]>| {
        if fact.is_some_and(|f| f[which as usize] as usize == value) {
            patches.push((sk.len(), which, e));
            sk.extend_from_slice(&[0; 4]);
        } else {
            sk.extend_from_slice(&(value as u32).to_le_bytes());
        }
    };
    while b[o..].starts_with(b"PK\x03\x04") && entries.len() < MAX_ENTRIES {
        let (Some(flags), Some(method), Some(crc), Some(csize), Some(usize_), Some(n), Some(m)) = (le16(b, o + 6), le16(b, o + 8), le32(b, o + 14), le32(b, o + 18), le32(b, o + 22), le16(b, o + 26), le16(b, o + 28)) else { break };
        let at = o + 30 + n + m;
        if at > b.len() || flags & 1 != 0 { break; }
        // the data's extent: by the header, or (sizes left to a descriptor) by the stream itself
        let (data, parsed) = if flags & 8 != 0 && csize == 0 {
            if method != 8 { break; }
            let Ok((inner, parse)) = inflate::inflate(&b[at..], left.min(MAX_INFLATE)) else { break };
            (&b[at..at + parse.consumed], Some((inner, parse)))
        } else {
            let Some(d) = b.get(at..at + csize) else { break };
            (d, None)
        };
        let e = entries.len();
        let (fact, taken) = match method {
            8 => {
                let got = parsed.or_else(|| inflate::inflate(data, left.min(MAX_INFLATE)).ok().filter(|r| r.1.consumed == data.len()));
                match got {
                    Some((inner, parse)) => {
                        let fact = (inner.len() <= u32::MAX as usize).then(|| [crc32(&inner), data.len() as u32, inner.len() as u32]);
                        match scan::record(&inner, &parse, 15, any) {
                            Ok(side) if inner.len() >= 64 => {
                                left -= inner.len();
                                (fact, Data::Deflate(Deflate { start: at, len: data.len(), inner, side }))
                            }
                            _ => (fact, Data::Raw(data.to_vec())),
                        }
                    }
                    None => (None, Data::Raw(data.to_vec())),
                }
            }
            0 if data.len() <= left => {
                left -= data.len();
                (facts_of(0, data), Data::Stored(data.to_vec()))
            }
            _ => (None, Data::Raw(data.to_vec())),
        };
        // the header, its three fields blanked where the content gives them
        sp.skeleton.extend_from_slice(&b[o..o + 14]);
        field(&mut sp.skeleton, &mut patches, crc, 0, e, fact);
        field(&mut sp.skeleton, &mut patches, csize, 1, e, fact);
        field(&mut sp.skeleton, &mut patches, usize_, 2, e, fact);
        sp.skeleton.extend_from_slice(&b[o + 26..at]);
        places.push(sp.skeleton.len());
        sp.data.push(taken);
        entries.push((o, method, fact));
        o = at + data.len();
        // a descriptor after the data says the same again
        if flags & 8 != 0 {
            let sig = b[o..].starts_with(b"PK\x07\x08") as usize * 4;
            if let (Some(c), Some(cs), Some(us)) = (le32(b, o + sig), le32(b, o + sig + 4), le32(b, o + sig + 8)) {
                if fact.is_some_and(|f| f == [c as u32, cs as u32, us as u32]) {
                    sp.skeleton.extend_from_slice(&b[o..o + sig]);
                    for (k, v) in [c, cs, us].into_iter().enumerate() { field(&mut sp.skeleton, &mut patches, v, k as u8, e, fact); }
                    o += sig + 12;
                }
            }
        }
    }
    if entries.len() < MIN_ENTRIES { return None; }
    // the central directory: each entry's fields once more, found by its header's place
    while b[o..].starts_with(b"PK\x01\x02") {
        let (Some(crc), Some(csize), Some(usize_), Some(n), Some(m), Some(k), Some(local)) = (le32(b, o + 16), le32(b, o + 20), le32(b, o + 24), le16(b, o + 28), le16(b, o + 30), le16(b, o + 32), le32(b, o + 42)) else { break };
        let end = o + 46 + n + m + k;
        if end > b.len() { break; }
        let e = entries.binary_search_by_key(&local, |x| x.0).ok();
        let fact = e.and_then(|e| entries[e].2);
        sp.skeleton.extend_from_slice(&b[o..o + 16]);
        field(&mut sp.skeleton, &mut patches, crc, 0, e.unwrap_or(0), fact);
        field(&mut sp.skeleton, &mut patches, csize, 1, e.unwrap_or(0), fact);
        field(&mut sp.skeleton, &mut patches, usize_, 2, e.unwrap_or(0), fact);
        sp.skeleton.extend_from_slice(&b[o + 28..end]);
        o = end;
    }
    // and its end record, with its comment
    if b[o..].starts_with(b"PK\x05\x06") {
        if let Some(end) = le16(b, o + 20).map(|c| o + 22 + c).filter(|&e| e <= b.len()) {
            sp.skeleton.extend_from_slice(&b[o..end]);
            o = end;
        }
    }
    drop(field);
    sp.len = o;
    sp.derived = patches.len();
    put_var(&mut sp.record, entries.len());
    let mut last = 0usize;
    for (&p, x) in places.iter().zip(&entries) {
        put_var(&mut sp.record, p - last);
        sp.record.push(x.1.min(255) as u8);
        last = p;
    }
    put_var(&mut sp.record, patches.len());
    last = 0;
    for &(p, which, e) in &patches {
        put_var(&mut sp.record, p - last);
        sp.record.push(which);
        put_var(&mut sp.record, e);
        last = p;
    }
    Some(sp)
}

/// Append the archive [`split`] took apart: `data` are its entries' data as they were.
pub fn rebuild(record: &[u8], skeleton: &[u8], data: &[Vec<u8>], out: &mut Vec<u8>) -> Result<(), String> {
    let bad = || "zip: bad record (corrupt file)".to_string();
    let mut p = 1usize;
    if record.first() != Some(&VERSION) || get_var(record, &mut p) != Some(data.len()) { return Err(bad()); }
    let mut places = Vec::with_capacity(data.len());
    let mut at = 0usize;
    for _ in data {
        at = at.checked_add(get_var(record, &mut p).ok_or_else(bad)?).ok_or_else(bad)?;
        places.push((at, *record.get(p).ok_or_else(bad)? as usize));
        p += 1;
    }
    let n = get_var(record, &mut p).ok_or_else(bad)?;
    if n > skeleton.len() / 4 { return Err(bad()); }
    let mut patches = Vec::with_capacity(n);
    at = 0;
    for _ in 0..n {
        at = at.checked_add(get_var(record, &mut p).ok_or_else(bad)?).ok_or_else(bad)?;
        let which = *record.get(p).ok_or_else(bad)?;
        p += 1;
        let e = get_var(record, &mut p).ok_or_else(bad)?;
        if which > 2 || e >= data.len() || at + 4 > skeleton.len() { return Err(bad()); }
        patches.push((at, which as usize, e));
    }
    if p != record.len() { return Err(bad()); }
    // the fields, from the content of the entries that have one derived
    let mut facts: Vec<Option<[u32; 3]>> = vec![None; data.len()];
    for &(_, _, e) in &patches {
        if facts[e].is_none() { facts[e] = Some(facts_of(places[e].1, &data[e]).ok_or_else(bad)?); }
    }
    let mut sk = skeleton.to_vec();
    for &(at, which, e) in &patches { sk[at..at + 4].copy_from_slice(&facts[e].unwrap()[which].to_le_bytes()); }
    let mut from = 0usize;
    for (d, &(place, _)) in data.iter().zip(&places) {
        out.extend_from_slice(sk.get(from..place).ok_or_else(bad)?);
        out.extend_from_slice(d);
        from = place;
    }
    out.extend_from_slice(&sk[from..]);
    Ok(())
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::transform::deflate::deflate_like_zlib;

    /// An archive of `n` entries: deflated text, stored bytes, one with a descriptor, one whose
    /// header lies about its checksum; then the directory.
    pub fn archive(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        let mut rng = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let (mut out, mut dir) = (vec![], vec![]);
        for k in 0..n {
            let content: Vec<u8> = (0..200 + rng() % 3000).map(|i| b"the quick brown fox jumps over the lazy dog\n"[(i as usize * (1 + k % 3)) % 44]).collect();
            let (method, data) = if k % 4 == 3 { (0u16, content.clone()) } else { (8, deflate_like_zlib(&content, 6)) };
            let descriptor = k % 5 == 2 && method == 8;
            let crc = crc32(&content) ^ (k == 1) as u32;
            let name = format!("dir/file{k}.txt");
            let at = out.len() as u32;
            let fields = |zero: bool| { let mut f = vec![]; for v in [crc, data.len() as u32, content.len() as u32] { f.extend_from_slice(&if zero { 0u32 } else { v }.to_le_bytes()); } f };
            out.extend_from_slice(b"PK\x03\x04\x14\x00");
            out.extend_from_slice(&(if descriptor { 8u16 } else { 0 }).to_le_bytes());
            out.extend_from_slice(&method.to_le_bytes());
            out.extend_from_slice(&[1, 2, 3, 4]);
            out.extend(fields(descriptor));
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(&[0, 0]);
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&data);
            if descriptor { out.extend_from_slice(b"PK\x07\x08"); out.extend(fields(false)); }
            dir.extend_from_slice(b"PK\x01\x02\x14\x00\x14\x00");
            dir.extend_from_slice(&(if descriptor { 8u16 } else { 0 }).to_le_bytes());
            dir.extend_from_slice(&method.to_le_bytes());
            dir.extend_from_slice(&[1, 2, 3, 4]);
            dir.extend(fields(false));
            dir.extend_from_slice(&(name.len() as u16).to_le_bytes());
            dir.extend_from_slice(&[0; 12]);
            dir.extend_from_slice(&at.to_le_bytes());
            dir.extend_from_slice(name.as_bytes());
        }
        let (dir_at, dir_len) = (out.len() as u32, dir.len() as u32);
        out.extend(dir);
        out.extend_from_slice(b"PK\x05\x06\0\0\0\0");
        out.extend_from_slice(&(n as u16).to_le_bytes());
        out.extend_from_slice(&(n as u16).to_le_bytes());
        out.extend_from_slice(&dir_len.to_le_bytes());
        out.extend_from_slice(&dir_at.to_le_bytes());
        out.extend_from_slice(&[7, 0]);
        out.extend_from_slice(b"comment");
        out
    }

    fn back(sp: &Split) -> Vec<u8> {
        let data: Vec<Vec<u8>> = sp.data.iter().map(|d| match d {
            Data::Deflate(d) => {
                let parse = crate::transform::deflate::side::decode(&d.inner, &d.side).unwrap();
                crate::transform::deflate::writer::write(&parse, &d.inner).unwrap()
            }
            Data::Stored(b) | Data::Raw(b) => b.clone(),
        }).collect();
        let mut out = vec![];
        rebuild(&sp.record, &sp.skeleton, &data, &mut out).unwrap();
        out
    }

    #[test]
    fn archives_come_back_with_their_fields_derived() {
        for n in [2usize, 9, 40] {
            let mut f = archive(n, 7);
            let end = f.len();
            f.extend_from_slice(b"after the archive");
            let sp = split(&f, 1 << 30, true).unwrap_or_else(|| panic!("{n} entries: not split"));
            assert_eq!((sp.len, sp.data.len()), (end, n), "{n} entries");
            assert!(back(&sp) == f[..end], "{n} entries: rebuild");
            // every entry's fields twice (thrice with a descriptor) but the one with a wrong checksum's
            assert!(sp.derived >= 6 * n - 2, "{n} entries: {} fields derived", sp.derived);
        }
        assert!(split(&archive(1, 3), 1 << 30, true).is_none() && split(b"PK\x03\x04 not one", 1 << 30, true).is_none());
        // damaged archives and records
        let f = archive(12, 5);
        let sp = split(&f, 1 << 30, true).unwrap();
        let mut s = 9u64;
        let mut rng = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        for _ in 0..250 {
            let mut bad = f.clone();
            for _ in 0..1 + rng() % 3 {
                let i = (rng() % bad.len() as u64) as usize;
                if rng() % 6 == 0 { bad.truncate(i.max(8)); } else { bad[i] ^= 1 << (rng() % 8); }
            }
            if let Some(sp) = split(&bad, 1 << 24, true) { assert!(back(&sp) == bad[..sp.len], "a damaged archive that splits must come back"); }
            let (mut rec, mut sk) = (sp.record.clone(), sp.skeleton.clone());
            let which = if rng() % 2 == 0 { &mut rec } else { &mut sk };
            let i = (rng() % which.len() as u64) as usize;
            if rng() % 4 == 0 { which.truncate(i); } else { which[i] ^= 1 << (rng() % 8); }
            let data: Vec<Vec<u8>> = sp.data.iter().map(|d| match d { Data::Deflate(d) => f[d.start..d.start + d.len].to_vec(), Data::Stored(b) | Data::Raw(b) => b.clone() }).collect();
            let mut out = vec![];
            let _ = rebuild(&rec, &sk, &data, &mut out);
            assert!(out.len() < 4 * f.len() + (1 << 16));
        }
    }
}
