//! SQLite databases, taken apart by what their bytes are.
//!
//! A database is pages; a page of a table or an index is a list of cells growing down from its
//! end and, at its start, an array of pointers to them in key order. A cell is a row: its size,
//! its number, then a record — a header of one type code per column, and the columns' values
//! back to back. Read as bytes, such a page is hard: a third of it is pointers and sizes that
//! say again what the cells say, numbers of every width stand between words, and the row above
//! is at no fixed distance.
//!
//! [`split`] writes a database as seven streams:
//!
//! * the **structure**: for every page whether it is a page of cells, how its cells are laid
//!   out (which is which, in the order of their addresses, and the lengths of the unused bytes
//!   between them), every row's number as a difference, and its columns' types;
//! * the **numbers**: integer and real values, column by column within a page;
//! * the **text**: text values, each ended by a zero byte instead of carrying its length;
//! * the **blobs**: binary values (and what of a row could not be read as a record);
//! * the **long rows**: a row that does not fit its page continues in a chain of overflow
//!   pages, each a pointer and then content, so a picture or a compressed document kept in a
//!   table lies in page-sized pieces. Here each such row is whole, and is planned like any
//!   other content: a zlib stream or a PNG in it is found and taken apart;
//! * the **gaps**: the unused bytes of the pages of cells (and the file's header);
//! * the **other pages**, as they are: interior pages, free pages, what is left of the
//!   overflow pages.
//!
//! Cell pointers, cell and header sizes and the chains' pointers are not written: [`rebuild`]
//! derives them. Nothing is trusted: a page that does not come back exactly is kept as it is,
//! and the database is taken apart only if the whole of it comes back.

const MAGIC: &[u8; 16] = b"SQLite format 3\0";

/// The streams, in the order of the container's children.
pub const STRUCTURE: usize = 0;
pub const NUMBERS: usize = 1;
pub const TEXT: usize = 2;
pub const BLOBS: usize = 3;
pub const ROWS: usize = 4;
pub const GAPS: usize = 5;
pub const PAGES: usize = 6;
pub const STREAMS: usize = 7;

/// A database's geometry: bytes to a page, bytes of a page in use (the rest is reserved), pages.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Db {
    pub page: usize,
    pub usable: usize,
    pub pages: usize,
}

impl Db {
    /// The container's parameters: the page size as a power of two, the reserved bytes.
    pub fn params(&self) -> Vec<u8> { vec![self.page.trailing_zeros() as u8, (self.page - self.usable) as u8] }

    /// The geometry those parameters say (the pages are counted from the structure).
    pub fn from_params(p: &[u8]) -> Option<(usize, usize)> {
        let &[log, reserved] = p else { return None };
        if !(9..=16).contains(&log) { return None; }
        let page = 1usize << log;
        (page - reserved as usize >= 480).then_some((page, page - reserved as usize))
    }
}

fn be16(b: &[u8], o: usize) -> Option<usize> { Some(u16::from_be_bytes(b.get(o..o + 2)?.try_into().ok()?) as usize) }
fn be32(b: &[u8], o: usize) -> Option<usize> { Some(u32::from_be_bytes(b.get(o..o + 4)?.try_into().ok()?) as usize) }

/// The database starting at `b[0]`, by its header (None unless the header says how many pages
/// there are: every SQLite since 3.7.0 writes it).
pub fn header(b: &[u8]) -> Option<Db> {
    if b.len() < 100 || &b[..16] != MAGIC { return None; }
    let page = match be16(b, 16)? { 1 => 65536, p if p >= 512 && p.is_power_of_two() => p, _ => return None };
    let usable = page - b[20] as usize;
    // the payload fractions are fixed, and a page keeps at least 480 bytes
    if usable < 480 || b[21] != 64 || b[22] != 32 || b[23] != 32 { return None; }
    let pages = be32(b, 28)?;
    (pages > 0).then_some(Db { page, usable, pages })
}

// ------------------------------------------------------------------ numbers
/// SQLite's variable-length integer at `b[o]`: its value and length.
fn varint(b: &[u8], o: usize) -> Option<(u64, usize)> {
    let mut v = 0u64;
    for k in 0..9 {
        let x = *b.get(o + k)? as u64;
        if k == 8 { return Some((v << 8 | x, 9)); }
        v = v << 7 | (x & 0x7F);
        if x & 0x80 == 0 { return Some((v, k + 1)); }
    }
    None
}

fn varint_len(v: u64) -> usize { if v >> 56 != 0 { 9 } else { ((64 - (v | 1).leading_zeros() as usize) + 6) / 7 } }

fn put_varint(o: &mut Vec<u8>, v: u64) {
    let n = varint_len(v);
    if n == 9 {
        for k in (0..8).rev() { o.push((v >> (8 + 7 * k)) as u8 | 0x80); }
        o.push(v as u8);
        return;
    }
    for k in (0..n).rev() { o.push(((v >> (7 * k)) & 0x7F) as u8 | if k > 0 { 0x80 } else { 0 }); }
}

/// The structure stream's own integers: seven bits a byte, the low ones first.
fn put(o: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        o.push(v as u8 | 0x80);
        v >>= 7;
    }
    o.push(v as u8);
}

fn get(b: &[u8], p: &mut usize) -> Option<u64> {
    let mut v = 0u64;
    for k in 0..10 {
        let x = *b.get(*p)?;
        *p += 1;
        v |= ((x & 0x7F) as u64) << (7 * k);
        if x & 0x80 == 0 { return Some(v); }
    }
    None
}

fn zig(v: i64) -> u64 { (v << 1 ^ v >> 63) as u64 }
fn zag(v: u64) -> i64 { (v >> 1) as i64 ^ -((v & 1) as i64) }

/// Bytes of a value of serial type `t` (None: a type SQLite does not write).
fn serial_len(t: u64) -> Option<usize> {
    match t {
        0 | 8 | 9 => Some(0),
        1..=4 => Some(t as usize),
        5 => Some(6),
        6 | 7 => Some(8),
        10 | 11 => None,
        _ => usize::try_from((t - 12) / 2).ok(),
    }
}

/// The size of a record's header as SQLite computes it: the type codes, and the size itself.
fn header_len(types: &[u64]) -> usize {
    let n: usize = types.iter().map(|&t| varint_len(t)).sum();
    let v = varint_len(n as u64);
    if v < varint_len((n + v) as u64) { n + v + 1 } else { n + v }
}

/// The most of a payload of `size` bytes a cell keeps in its page (of `u` usable bytes), for a
/// page of a table or of an index; and whether the rest overflows.
fn local(size: u64, u: usize, index: bool) -> (usize, bool) {
    let most = if index { (u - 12) * 64 / 255 - 23 } else { u - 35 };
    if size <= most as u64 { return (size as usize, false); }
    let least = (u - 12) * 32 / 255 - 23;
    let k = least + ((size - least as u64) % (u as u64 - 4)) as usize;
    (if k <= most { k } else { least }, true)
}

// ------------------------------------------------------------------ taking apart
/// What a cell's payload is written as.
enum Form {
    /// The bytes of it that are in the page, as they are (and the pointer to its overflow).
    Raw,
    /// Whole, from the page and the chain of overflow pages after it.
    Long(Vec<u32>),
    /// A record: its columns' types, and where each value starts.
    Record(Vec<u64>, Vec<u32>),
}

struct Cell {
    /// Where it starts in its page, its length there, where its payload starts in the database.
    ptr: usize,
    len: usize,
    body: usize,
    size: u64,
    rowid: u64,
    /// Bytes of the payload in the page, and whether the rest overflows.
    local: usize,
    over: bool,
    form: Form,
}

struct Leaf {
    index: bool,
    freeblock: usize,
    start: usize,
    fragments: u8,
    cells: Vec<Cell>,
}

/// Page `p` of `db` as a page of cells, if it is one whose cells lie where a page's cells may.
fn leaf(db: &[u8], geo: &Db, p: usize) -> Option<Leaf> {
    let (at, u) = (p * geo.page, geo.usable);
    let h = at + if p == 0 { 100 } else { 0 };
    let index = match db[h] { 0x0D => false, 0x0A => true, _ => return None };
    let n = be16(db, h + 3)?;
    let first = h - at + 8 + 2 * n;
    if first > u { return None; }
    let mut cells = Vec::with_capacity(n);
    for c in 0..n {
        let ptr = be16(db, h + 8 + 2 * c)?;
        if ptr < first || ptr >= u { return None; }
        let (size, a) = varint(db, at + ptr)?;
        let mut body = at + ptr + a;
        let mut rowid = 0;
        if varint_len(size) != a { return None; }
        if !index {
            let (r, a) = varint(db, body)?;
            if varint_len(r) != a { return None; }
            (rowid, body) = (r, body + a);
        }
        let (local, over) = local(size, u, index);
        let len = body - (at + ptr) + local + 4 * over as usize;
        if ptr + len > u { return None; }
        cells.push(Cell { ptr, len, body, size, rowid, local, over, form: if over { Form::Raw } else { record(&db[body..body + local]) } });
    }
    // no cell in another
    let mut by_address: Vec<(usize, usize)> = cells.iter().map(|c| (c.ptr, c.len)).collect();
    by_address.sort_unstable();
    if by_address.windows(2).any(|w| w[0].0 + w[0].1 > w[1].0) { return None; }
    Some(Leaf { index, freeblock: be16(db, h + 1)?, start: be16(db, h + 5)?, fragments: db[h + 7], cells })
}

/// The payload `b` as a record, if it is one written the way SQLite writes them.
fn record(b: &[u8]) -> Form {
    let parse = || -> Option<Form> {
        let (hl, a) = varint(b, 0)?;
        let hl = usize::try_from(hl).ok().filter(|&h| h >= a && h <= b.len())?;
        let (mut types, mut starts) = (vec![], vec![]);
        let (mut o, mut at) = (a, hl);
        while o < hl {
            let (t, a) = varint(b, o)?;
            if varint_len(t) != a { return None; }
            starts.push(at as u32);
            at = at.checked_add(serial_len(t)?).filter(|&e| e <= b.len())?;
            types.push(t);
            o += a;
        }
        (o == hl && at == b.len() && header_len(&types) == hl).then_some(Form::Record(types, starts))
    };
    parse().unwrap_or(Form::Raw)
}

/// What [`split`] made of a database.
pub struct Split {
    pub streams: [Vec<u8>; STREAMS],
    /// Pages of cells taken apart, rows read as records, rows gathered from overflow chains.
    pub leaves: usize,
    pub records: usize,
    pub long: usize,
}

/// Take the database `db` (all its pages) apart; None unless [`rebuild`] gives it back exactly.
pub fn split(db: &[u8], geo: &Db) -> Option<Split> {
    if geo.pages.checked_mul(geo.page)? != db.len() { return None; }
    // pages that did not come back are kept as they are, and the rest is tried again
    let mut keep = vec![false; geo.pages];
    let mut out = Vec::with_capacity(db.len());
    for _ in 0..4 {
        let sp = split_keeping(db, geo, &keep);
        out.clear();
        let st: [&[u8]; STREAMS] = std::array::from_fn(|i| &sp.streams[i][..]);
        if rebuild(geo.page, geo.usable, &st, &mut out).is_err() || out.len() != db.len() { return None; }
        if out == db { return Some(sp); }
        let mut any = false;
        for p in 0..geo.pages {
            let r = p * geo.page..(p + 1) * geo.page;
            if out[r.clone()] != db[r] && !keep[p] { (keep[p], any) = (true, true); }
        }
        if !any { return None; }
    }
    None
}

fn split_keeping(db: &[u8], geo: &Db, keep: &[bool]) -> Split {
    let (page, u) = (geo.page, geo.usable);
    let mut leaves: Vec<Option<Leaf>> = (0..geo.pages).map(|p| if keep[p] { None } else { leaf(db, geo, p) }).collect();
    // the rows that overflow: whole if their chain is sound — pages of the file that are no
    // pages of cells, each in one chain, the last pointing nowhere. `claimed`: the content bytes
    // of an overflow page in a chain, plus one.
    let is_leaf: Vec<bool> = leaves.iter().map(|l| l.is_some()).collect();
    let mut claimed = vec![0u32; geo.pages];
    for l in leaves.iter_mut().flatten() {
        for c in l.cells.iter_mut().filter(|c| c.over) {
            let mut chain: Vec<u32> = vec![];
            let mut left = c.size - c.local as u64;
            let mut next = be32(db, c.body + c.local).unwrap_or(0);
            while left > 0 && next >= 2 && next <= geo.pages && !is_leaf[next - 1] && !keep[next - 1] && claimed[next - 1] == 0 {
                let take = left.min(u as u64 - 4);
                claimed[next - 1] = take as u32 + 1;
                chain.push(next as u32);
                left -= take;
                next = be32(db, (next - 1) * page).unwrap_or(0);
            }
            if left == 0 && next == 0 {
                c.form = Form::Long(chain);
            } else {
                for q in chain { claimed[q as usize - 1] = 0; }
            }
        }
    }
    let mut sp = Split { streams: Default::default(), leaves: 0, records: 0, long: 0 };
    let mut last_row = 0i64;
    for (p, l) in leaves.iter().enumerate() {
        let at = p * page;
        let Some(l) = l else {
            // not a page of cells: as it is, but for what a chain took of it
            sp.streams[STRUCTURE].push(0);
            let from = if claimed[p] != 0 { 4 + claimed[p] as usize - 1 } else { 0 };
            sp.streams[PAGES].extend_from_slice(&db[at + from..at + page]);
            continue;
        };
        sp.leaves += 1;
        let [s, numbers, text, blobs, rows, gaps, _] = &mut sp.streams;
        let n = l.cells.len();
        s.push(if l.index { 0x0A } else { 0x0D });
        put(s, n as u64);
        put(s, l.freeblock as u64);
        s.push(l.fragments);
        // where the cells' area starts: at the lowest cell, as a rule
        let lowest = l.cells.iter().map(|c| c.ptr).min().unwrap_or(u) & 0xFFFF;
        put(s, zig(l.start as i64 - lowest as i64));
        // the cells from the page's end down: as a rule, in the order of the pointers
        let mut down: Vec<usize> = (0..n).collect();
        down.sort_unstable_by_key(|&c| std::cmp::Reverse(l.cells[c].ptr));
        let mut rank = vec![0i64; n];
        for (r, &c) in down.iter().enumerate() { rank[c] = r as i64; }
        let mut before = -1;
        for &r in &rank {
            put(s, zig(r - before - 1));
            before = r;
        }
        // the unused bytes: after the pointers, between the cells, at the end
        if p == 0 { gaps.extend_from_slice(&db[..100]); }
        let mut pos = if p == 0 { 100 } else { 0 } + 8 + 2 * n;
        for &c in down.iter().rev() {
            let c = &l.cells[c];
            put(s, (c.ptr - pos) as u64);
            gaps.extend_from_slice(&db[at + pos..at + c.ptr]);
            pos = c.ptr + c.len;
        }
        put(s, (page - pos) as u64);
        gaps.extend_from_slice(&db[at + pos..at + page]);
        // the rows, in the order of the pointers
        for c in &l.cells {
            if !l.index {
                put(s, zig((c.rowid as i64).wrapping_sub(last_row).wrapping_sub(1)));
                last_row = c.rowid as i64;
            }
            let payload = &db[c.body..c.body + c.local];
            match &c.form {
                Form::Raw => {
                    put(s, 0);
                    put(s, c.size);
                    blobs.extend_from_slice(payload);
                    if c.over { s.extend_from_slice(&db[c.body + c.local..c.body + c.local + 4]); }
                }
                Form::Long(chain) => {
                    sp.long += 1;
                    put(s, 1);
                    put(s, c.size);
                    rows.extend_from_slice(payload);
                    let mut before = p as i64 + 1;
                    for &q in chain {
                        put(s, zig(q as i64 - before - 1));
                        before = q as i64;
                        let o = (q as usize - 1) * page + 4;
                        rows.extend_from_slice(&db[o..o + claimed[q as usize - 1] as usize - 1]);
                    }
                }
                Form::Record(types, starts) => {
                    sp.records += 1;
                    put(s, types.len() as u64 + 2);
                    for (&t, &o) in types.iter().zip(starts) {
                        let v = &payload[o as usize..o as usize + serial_len(t).unwrap_or(0)];
                        if t >= 13 && t % 2 == 1 && !v.contains(&0) {
                            // text: its end is in the text, not here
                            put(s, 13);
                            text.extend_from_slice(v);
                            text.push(0);
                        } else {
                            put(s, t);
                            if t >= 12 { blobs.extend_from_slice(v); }
                        }
                    }
                }
            }
        }
        // the numbers, column by column
        let columns = l.cells.iter().map(|c| if let Form::Record(t, _) = &c.form { t.len() } else { 0 }).max().unwrap_or(0);
        for k in 0..columns {
            for c in &l.cells {
                let Form::Record(types, starts) = &c.form else { continue };
                if let Some(&t @ 1..=7) = types.get(k) {
                    let o = c.body + starts[k] as usize;
                    numbers.extend_from_slice(&db[o..o + serial_len(t).unwrap_or(0)]);
                }
            }
        }
    }
    sp
}

// ------------------------------------------------------------------ putting together
/// Bytes of stream `b` from `*at` on, `n` of them.
fn take<'a>(b: &'a [u8], at: &mut usize, n: usize) -> Result<&'a [u8], String> {
    let r = b.get(*at..at.checked_add(n).ok_or_else(bad)?).ok_or_else(bad)?;
    *at += n;
    Ok(r)
}

fn bad() -> String { "sqlite: bad streams (corrupt file)".to_string() }

/// Append the database [`split`] took apart: `page` bytes to a page, `usable` of them in use,
/// from its seven streams.
pub fn rebuild(page: usize, usable: usize, st: &[&[u8]; STREAMS], out: &mut Vec<u8>) -> Result<(), String> {
    let (s, u) = (st[STRUCTURE], usable);
    let base = out.len();
    let mut at = [0usize; STREAMS];
    let num = |p: &mut usize| -> Result<usize, String> { get(s, p).and_then(|v| usize::try_from(v).ok()).ok_or_else(bad) };
    // every byte of a page is one of a stream's, but for the few a page and a cell derive
    let most = 40 * st.iter().map(|x| x.len()).sum::<usize>() + page;
    // pages that are no pages of cells; rows to gather, in the order they were written: (the
    // cell's payload in `out`, its bytes there, its size, its chain)
    let mut other: Vec<usize> = vec![];
    let mut long: Vec<(usize, usize, u64, Vec<usize>)> = vec![];
    let mut last_row = 0i64;
    let mut sp = 0usize;
    let mut pages = 0usize;
    // per page: the cells and where each was placed, the numbers still to fill in (column, cell,
    // where in it, bytes), and the long rows (cell, where in it, …)
    let mut cells: Vec<Vec<u8>> = vec![];
    let mut placed: Vec<usize> = vec![];
    let mut numbers: Vec<(usize, usize, usize, usize)> = vec![];
    let mut chains: Vec<(usize, usize, usize, u64, Vec<usize>)> = vec![];
    while sp < s.len() {
        let p = pages;
        pages += 1;
        let start = out.len();
        if start - base + page > most { return Err(bad()); }
        out.resize(start + page, 0);
        let ty = s[sp];
        sp += 1;
        if ty == 0 {
            other.push(p);
            // a page as it is has its bytes in a stream (but for the pointer of a chain)
            if other.len() * (page - 4) > st[PAGES].len() + st[ROWS].len() { return Err(bad()); }
            continue;
        }
        if ty != 0x0D && ty != 0x0A { return Err(bad()); }
        let index = ty == 0x0A;
        let h = if p == 0 { 100 } else { 0 };
        let n = num(&mut sp)?;
        if h + 8 + 2 * n > u { return Err(bad()); }
        let freeblock = num(&mut sp)?;
        let fragments = *s.get(sp).ok_or_else(bad)?;
        sp += 1;
        let start_delta = zag(get(s, &mut sp).ok_or_else(bad)?);
        // the cells from the page's end down
        let mut down = vec![usize::MAX; n];
        let mut before = -1i64;
        for c in 0..n {
            let r = before + 1 + zag(get(s, &mut sp).ok_or_else(bad)?);
            let slot = usize::try_from(r).ok().and_then(|r| down.get_mut(r)).filter(|d| **d == usize::MAX).ok_or_else(bad)?;
            *slot = c;
            before = r;
        }
        let mut gaps = Vec::with_capacity(n + 1);
        for _ in 0..=n { gaps.push(num(&mut sp)?); }
        cells.clear();
        numbers.clear();
        chains.clear();
        for c in 0..n {
            let mut cell = vec![];
            let rowid = if index { None } else {
                last_row = last_row.wrapping_add(1).wrapping_add(zag(get(s, &mut sp).ok_or_else(bad)?));
                Some(last_row as u64)
            };
            let form = get(s, &mut sp).ok_or_else(bad)?;
            if form <= 1 {
                let size = get(s, &mut sp).ok_or_else(bad)?;
                let (local, over) = local(size, u, index);
                put_varint(&mut cell, size);
                if let Some(r) = rowid { put_varint(&mut cell, r); }
                if form == 0 {
                    cell.extend_from_slice(take(st[BLOBS], &mut at[BLOBS], local)?);
                    if over { cell.extend_from_slice(take(s, &mut sp, 4)?); }
                } else {
                    // the row is filled in when every page is known
                    if !over { return Err(bad()); }
                    let count = (size - local as u64).div_ceil(u as u64 - 4);
                    if count > (s.len() - sp) as u64 { return Err(bad()); }
                    let mut chain = Vec::with_capacity(count as usize);
                    let mut before = p as i64 + 1;
                    for _ in 0..count {
                        before = before + 1 + zag(get(s, &mut sp).ok_or_else(bad)?);
                        chain.push(usize::try_from(before).ok().filter(|&q| q >= 2).ok_or_else(bad)?);
                    }
                    chains.push((c, cell.len(), local, size, chain));
                    let first = chains.last().unwrap().4[0];
                    cell.resize(cell.len() + local, 0);
                    cell.extend_from_slice(&u32::try_from(first).map_err(|_| bad())?.to_be_bytes());
                }
            } else {
                // a record: the types, then the values
                let columns = usize::try_from(form - 2).ok().filter(|&k| k <= s.len() - sp).ok_or_else(bad)?;
                let mut types = Vec::with_capacity(columns);
                let mut values = vec![];
                let first = numbers.len();
                for k in 0..columns {
                    let t = get(s, &mut sp).ok_or_else(bad)?;
                    if t == 13 {
                        let rest = &st[TEXT][at[TEXT]..];
                        let len = rest.iter().position(|&x| x == 0).ok_or_else(bad)?;
                        values.extend_from_slice(&rest[..len]);
                        at[TEXT] += len + 1;
                        types.push(13 + 2 * len as u64);
                        continue;
                    }
                    let len = serial_len(t).ok_or_else(bad)?;
                    match t {
                        1..=7 => {
                            numbers.push((k, c, values.len(), len));
                            values.resize(values.len() + len, 0);
                        }
                        0 | 8 | 9 => {}
                        _ => values.extend_from_slice(take(st[BLOBS], &mut at[BLOBS], len)?),
                    }
                    types.push(t);
                }
                let hl = header_len(&types);
                put_varint(&mut cell, (hl + values.len()) as u64);
                if let Some(r) = rowid { put_varint(&mut cell, r); }
                put_varint(&mut cell, hl as u64);
                for &t in &types { put_varint(&mut cell, t); }
                // where the values start in the cell, for the numbers
                let o = cell.len();
                for x in &mut numbers[first..] { x.2 += o; }
                cell.extend_from_slice(&values);
            }
            cells.push(cell);
        }
        numbers.sort_unstable();
        for &(_, c, o, len) in &numbers { cells[c][o..o + len].copy_from_slice(take(st[NUMBERS], &mut at[NUMBERS], len)?); }
        // the page: header, pointers, and the gaps and cells from the pointers' end on
        if p == 0 { out[start..start + 100].copy_from_slice(take(st[GAPS], &mut at[GAPS], 100)?); }
        let mut pos = h + 8 + 2 * n;
        let mut lowest = u;
        placed.clear();
        placed.resize(n, 0);
        for (k, &c) in down.iter().rev().enumerate() {
            let gap = take(st[GAPS], &mut at[GAPS], gaps[k])?;
            let cell = &cells[c];
            if pos + gap.len() + cell.len() > page { return Err(bad()); }
            out[start + pos..start + pos + gap.len()].copy_from_slice(gap);
            pos += gap.len();
            out[start + h + 8 + 2 * c..start + h + 10 + 2 * c].copy_from_slice(&(pos as u16).to_be_bytes());
            out[start + pos..start + pos + cell.len()].copy_from_slice(cell);
            lowest = lowest.min(pos);
            placed[c] = start + pos;
            pos += cell.len();
        }
        for (c, o, local, size, chain) in chains.drain(..) { long.push((placed[c] + o, local, size, chain)); }
        let gap = take(st[GAPS], &mut at[GAPS], gaps[n])?;
        if pos + gap.len() != page { return Err(bad()); }
        out[start + pos..start + page].copy_from_slice(gap);
        let head = &mut out[start + h..start + h + 8];
        head[0] = ty;
        head[1..3].copy_from_slice(&(freeblock as u16).to_be_bytes());
        head[3..5].copy_from_slice(&(n as u16).to_be_bytes());
        head[5..7].copy_from_slice(&(((lowest & 0xFFFF) as i64 + start_delta) as u16).to_be_bytes());
        head[7] = fragments;
    }
    // the long rows: the part in the page, then along the chain
    let mut claimed = vec![0usize; pages];
    let mut is_other = vec![false; pages];
    for &p in &other { is_other[p] = true; }
    for (o, local, size, chain) in &long {
        out[*o..*o + *local].copy_from_slice(take(st[ROWS], &mut at[ROWS], *local)?);
        let mut left = *size - *local as u64;
        for (k, &q) in chain.iter().enumerate() {
            if q > pages || !is_other[q - 1] || claimed[q - 1] != 0 { return Err(bad()); }
            let n = left.min(u as u64 - 4) as usize;
            claimed[q - 1] = n + 1;
            let o = base + (q - 1) * page;
            let next = chain.get(k + 1).copied().unwrap_or(0) as u32;
            out[o..o + 4].copy_from_slice(&next.to_be_bytes());
            out[o + 4..o + 4 + n].copy_from_slice(take(st[ROWS], &mut at[ROWS], n)?);
            left -= n as u64;
        }
        if left != 0 { return Err(bad()); }
    }
    // the other pages
    for &p in &other {
        let from = if claimed[p] != 0 { 4 + claimed[p] - 1 } else { 0 };
        let o = base + p * page;
        out[o + from..o + page].copy_from_slice(take(st[PAGES], &mut at[PAGES], page - from)?);
    }
    if (0..STREAMS).any(|i| i != STRUCTURE && at[i] != st[i].len()) { return Err(bad()); }
    Ok(())
}

#[cfg(test)]
pub mod tests {
    use super::*;

    fn rng(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }

    /// A database of one table whose rows are `blobs` (page 1: its header and an empty schema
    /// page; then leaf pages of one row each and their overflow pages). `shuffle`: the overflow
    /// pages of all rows are allocated in a scrambled order.
    pub fn database(blobs: &[&[u8]], page: usize, shuffle: bool) -> Vec<u8> {
        let u = page;
        // payload: a record header (2 columns: NULL for the row id, a blob) and the blob
        let payloads: Vec<Vec<u8>> = blobs.iter().map(|b| {
            let types = [0, 12 + 2 * b.len() as u64];
            let mut p = vec![];
            put_varint(&mut p, header_len(&types) as u64);
            for t in types { put_varint(&mut p, t); }
            p.extend_from_slice(b);
            p
        }).collect();
        let need: Vec<usize> = payloads.iter().map(|p| (p.len() - local(p.len() as u64, u, false).0).div_ceil(u - 4)).collect();
        // pages: 1 = header, then a leaf a row, then the overflow pages
        let leaves = payloads.len();
        let total = 1 + leaves + need.iter().sum::<usize>();
        let mut order: Vec<usize> = (2 + leaves..=total).collect();
        if shuffle {
            let mut s = 99u64;
            for i in (1..order.len()).rev() { order.swap(i, (rng(&mut s) % (i as u64 + 1)) as usize); }
        }
        let mut db = vec![0u8; total * page];
        db[..16].copy_from_slice(MAGIC);
        db[16..18].copy_from_slice(&(if page == 65536 { 1u16 } else { page as u16 }).to_be_bytes());
        (db[18], db[19], db[21], db[22], db[23]) = (1, 1, 64, 32, 32);
        db[28..32].copy_from_slice(&(total as u32).to_be_bytes());
        db[100] = 0x0D;
        db[105..107].copy_from_slice(&(page as u16).to_be_bytes());
        let mut free = order.into_iter();
        for (r, p) in payloads.iter().enumerate() {
            let at = (1 + r) * page;
            let l = local(p.len() as u64, u, false).0;
            let mut cell = vec![];
            put_varint(&mut cell, p.len() as u64);
            put_varint(&mut cell, r as u64 + 1);
            cell.extend_from_slice(&p[..l]);
            let chain: Vec<usize> = (0..need[r]).map(|_| free.next().unwrap()).collect();
            if l < p.len() { cell.extend_from_slice(&(chain[0] as u32).to_be_bytes()); }
            let ptr = page - cell.len();
            db[at] = 0x0D;
            db[at + 3..at + 5].copy_from_slice(&1u16.to_be_bytes());
            db[at + 5..at + 7].copy_from_slice(&(ptr as u16).to_be_bytes());
            db[at + 8..at + 10].copy_from_slice(&(ptr as u16).to_be_bytes());
            db[at + ptr..at + page].copy_from_slice(&cell);
            let mut done = l;
            for (k, &pg) in chain.iter().enumerate() {
                let o = (pg - 1) * page;
                let take = (p.len() - done).min(u - 4);
                db[o..o + 4].copy_from_slice(&(chain.get(k + 1).copied().unwrap_or(0) as u32).to_be_bytes());
                db[o + 4..o + 4 + take].copy_from_slice(&p[done..done + take]);
                done += take;
            }
        }
        db
    }

    /// A value of a row of [`table`].
    #[derive(Clone)]
    pub enum Value {
        Null,
        Int(i64),
        Real(f64),
        Text(Vec<u8>),
        Blob(Vec<u8>),
    }

    /// A database of one table (and an index on its first column): `rows` of values in pages of
    /// `page` bytes, filled as SQLite fills them (cells from the end down, pointers in key
    /// order). `churn`: the pages look used — cells in a scrambled order, a freed cell between
    /// them (a free block with stale bytes in it), fragments, bytes left in the unused area.
    pub fn table(rows: &[Vec<Value>], page: usize, churn: bool) -> Vec<u8> {
        let mut s = 0x9E37_79B9u64;
        let record = |row: &[Value]| -> Vec<u8> {
            let (mut types, mut values) = (vec![], vec![]);
            for v in row {
                match v {
                    Value::Null => types.push(0),
                    Value::Int(0) => types.push(8),
                    Value::Int(1) => types.push(9),
                    Value::Int(i) => {
                        let n = [1usize, 2, 3, 4, 6, 8].into_iter().find(|&n| n == 8 || (-(1i64 << (8 * n - 1))..1i64 << (8 * n - 1)).contains(i)).unwrap();
                        types.push([0, 1, 2, 3, 4, 0, 5, 0, 6][n]);
                        values.extend_from_slice(&i.to_be_bytes()[8 - n..]);
                    }
                    Value::Real(f) => {
                        types.push(7);
                        values.extend_from_slice(&f.to_be_bytes());
                    }
                    Value::Text(t) => {
                        types.push(13 + 2 * t.len() as u64);
                        values.extend_from_slice(t);
                    }
                    Value::Blob(b) => {
                        types.push(12 + 2 * b.len() as u64);
                        values.extend_from_slice(b);
                    }
                }
            }
            let mut r = vec![];
            put_varint(&mut r, header_len(&types) as u64);
            for t in types { put_varint(&mut r, t); }
            r.extend(values);
            r
        };
        // the cells of the table, and of the index: (first column, row id)
        let table: Vec<Vec<u8>> = rows.iter().enumerate().map(|(i, row)| {
            let rec = record(row);
            let mut c = vec![];
            put_varint(&mut c, rec.len() as u64);
            put_varint(&mut c, 1000 + 3 * i as u64);
            c.extend(rec);
            c
        }).collect();
        let index: Vec<Vec<u8>> = rows.iter().enumerate().map(|(i, row)| {
            let rec = record(&[row[0].clone(), Value::Int(1000 + 3 * i as i64)]);
            let mut c = vec![];
            put_varint(&mut c, rec.len() as u64);
            c.extend(rec);
            c
        }).collect();
        let mut pages: Vec<Vec<u8>> = vec![vec![0u8; page]];
        for (ty, cells) in [(0x0Du8, &table), (0x0A, &index)] {
            let mut k = 0;
            while k < cells.len() {
                let mut pg = vec![0u8; page];
                if churn { for x in pg.iter_mut() { *x = (rng(&mut s) % 7) as u8 * 16; } }
                let (mut end, mut n) = (page, 0usize);
                let mut ptrs = vec![];
                let mut freeblock = 0usize;
                while k < cells.len() && 8 + 2 * (n + 1) + cells[k].len() + if churn { 40 } else { 0 } <= end {
                    if churn && n % 5 == 3 {
                        // a freed cell: a free block of 12 bytes, and a fragment of two
                        end -= 12;
                        pg[end..end + 2].copy_from_slice(&(freeblock as u16).to_be_bytes());
                        pg[end + 2..end + 4].copy_from_slice(&12u16.to_be_bytes());
                        freeblock = end;
                        end -= 2;
                        pg[7] += 2;
                    }
                    end -= cells[k].len();
                    pg[end..end + cells[k].len()].copy_from_slice(&cells[k]);
                    ptrs.push(end);
                    (k, n) = (k + 1, n + 1);
                }
                assert!(n > 0, "a row too long for the page");
                if churn {
                    // the cells keep their places; the keys they hold are taken in another order
                    for i in (1..n).rev() { ptrs.swap(i, (rng(&mut s) % (i as u64 + 1)) as usize); }
                }
                pg[0] = ty;
                pg[1..3].copy_from_slice(&(freeblock as u16).to_be_bytes());
                pg[3..5].copy_from_slice(&(n as u16).to_be_bytes());
                pg[5..7].copy_from_slice(&(end as u16).to_be_bytes());
                for (i, &p) in ptrs.iter().enumerate() { pg[8 + 2 * i..10 + 2 * i].copy_from_slice(&(p as u16).to_be_bytes()); }
                pages.push(pg);
            }
        }
        let total = pages.len();
        let first = &mut pages[0];
        first[..16].copy_from_slice(MAGIC);
        first[16..18].copy_from_slice(&(if page == 65536 { 1u16 } else { page as u16 }).to_be_bytes());
        (first[18], first[19], first[21], first[22], first[23]) = (1, 1, 64, 32, 32);
        first[28..32].copy_from_slice(&(total as u32).to_be_bytes());
        first[100] = 0x0D;
        first[105..107].copy_from_slice(&(page as u16).to_be_bytes());
        pages.concat()
    }

    /// Rows of a table of orders: a name, a count, a price, a note or none, a flag, a stamp.
    pub fn orders(n: usize) -> Vec<Vec<Value>> {
        let mut s = 0xC0FFEEu64;
        let words = ["anchor", "bolt", "bracket", "clamp", "flange", "gasket", "hinge", "nut", "pin", "rivet", "screw", "washer"];
        (0..n).map(|i| {
            let name = format!("{} {} {}", words[(rng(&mut s) % 12) as usize], words[(rng(&mut s) % 12) as usize], rng(&mut s) % 500);
            vec![
                Value::Text(name.into_bytes()),
                Value::Int((rng(&mut s) % 300) as i64),
                Value::Real((rng(&mut s) % 100_000) as f64 / 100.0),
                if rng(&mut s) % 4 == 0 { Value::Null } else { Value::Text(format!("shelf {}", rng(&mut s) % 40).into_bytes()) },
                Value::Int((rng(&mut s) % 2) as i64),
                Value::Int(1_700_000_000 + 37 * i as i64),
                if i % 17 == 0 { Value::Blob((0..24).map(|_| rng(&mut s) as u8).collect()) } else { Value::Null },
            ]
        }).collect()
    }

    /// `data` as a document a database keeps compressed: its length, then a zlib stream.
    pub fn zlib(data: &[u8]) -> Vec<u8> {
        let (mut a, mut b) = (1u32, 0u32);
        for &x in data {
            a = (a + x as u32) % 65521;
            b = (b + a) % 65521;
        }
        let mut z = (data.len() as u32).to_be_bytes().to_vec();
        z.extend_from_slice(&[0x78, 0x9c]);
        z.extend(crate::transform::deflate::deflate_like_zlib(data, 6));
        z.extend_from_slice(&(b << 16 | a).to_be_bytes());
        z
    }

    /// Bytes no model shortens much, and bytes one does.
    pub fn blob(n: usize, seed: u64, text: bool) -> Vec<u8> {
        let mut s = seed;
        if text { (0..n).map(|i| b"the row that does not fit its page "[(i + (rng(&mut s) % 2) as usize) % 35]).collect() } else { (0..n).map(|_| rng(&mut s) as u8).collect() }
    }

    fn back(db: &[u8], sp: &Split) -> Vec<u8> {
        let geo = header(db).unwrap();
        let st: [&[u8]; STREAMS] = std::array::from_fn(|i| &sp.streams[i][..]);
        let mut out = vec![];
        rebuild(geo.page, geo.usable, &st, &mut out).unwrap();
        out
    }

    #[test]
    fn numbers_as_sqlite_writes_them() {
        for v in [0u64, 1, 127, 128, 16383, 16384, 1 << 28, (1 << 56) - 1, 1 << 56, u64::MAX] {
            let mut b = vec![];
            put_varint(&mut b, v);
            assert_eq!((varint(&b, 0), b.len()), (Some((v, b.len())), varint_len(v)), "{v}");
            let mut b = vec![];
            put(&mut b, v);
            assert_eq!(get(&b, &mut 0), Some(v));
        }
        for v in [0i64, 1, -1, 63, -64, i64::MAX, i64::MIN] { assert_eq!(zag(zig(v)), v); }
        // the header's size counts itself: 126 one-byte types need one byte for it, 127 need two
        assert_eq!((header_len(&[1; 126]), header_len(&[1; 127]), header_len(&[1; 128])), (127, 129, 130));
        assert_eq!((local(100, 4096, false), local(4061, 4096, false), local(4062, 4096, false)), ((100, false), (4061, false), (489, true)));
        assert!(header(b"SQLite format 3\0 but nothing more").is_none());
    }

    #[test]
    fn rows_that_overflow_are_gathered() {
        for page in [512usize, 1024, 4096, 65536] {
            for shuffle in [false, true] {
                let blobs = [blob(40, 1, true), blob(3 * page + 17, 2, false), blob(page - 100, 3, true), blob(page - 30, 4, false), blob(9 * page, 5, true), blob(page * 2 - 8, 6, false)];
                let refs: Vec<&[u8]> = blobs.iter().map(|b| &b[..]).collect();
                let db = database(&refs, page, shuffle);
                let geo = header(&db).unwrap();
                assert_eq!(geo, Db { page, usable: page, pages: db.len() / page });
                assert_eq!(Db::from_params(&geo.params()), Some((page, page)));
                let sp = split(&db, &geo).unwrap_or_else(|| panic!("page {page}, shuffle {shuffle}: not split"));
                assert!(back(&db, &sp) == db);
                // every row that overflows is whole and in order; the pages keep nothing of them
                let long: Vec<&Vec<u8>> = blobs.iter().filter(|b| b.len() + header_len(&[0, 12 + 2 * b.len() as u64]) > page - 35).collect();
                assert_eq!(long.len(), 4);
                assert_eq!((sp.long, sp.leaves), (long.len(), 1 + blobs.len()), "page {page}, shuffle {shuffle}");
                let mut o = 0;
                for b in long {
                    let at = sp.streams[ROWS][o..].windows(b.len()).position(|w| w == &b[..]).expect("a long row is not whole");
                    assert!(at < 12, "page {page}: {at} bytes before a long row");
                    o += at + b.len();
                }
                assert!(sp.streams[PAGES].len() < page, "page {page}: {} bytes of other pages", sp.streams[PAGES].len());
            }
        }
    }

    #[test]
    fn tables_are_taken_apart() {
        let rows = orders(700);
        for page in [512usize, 4096] {
            for churn in [false, true] {
                let db = table(&rows, page, churn);
                let geo = header(&db).unwrap();
                let sp = split(&db, &geo).unwrap_or_else(|| panic!("page {page}, churn {churn}: not split"));
                assert!(back(&db, &sp) == db, "page {page}, churn {churn}");
                assert_eq!((sp.leaves, sp.records), (geo.pages, 2 * rows.len()), "page {page}, churn {churn}");
                // the text is the names and notes, each ended; the numbers are all there
                let names: usize = rows.iter().map(|r| r.iter().filter(|v| matches!(v, Value::Text(_))).count()).sum();
                assert_eq!(sp.streams[TEXT].iter().filter(|&&x| x == 0).count(), names + rows.len());
                assert!(churn || sp.streams[TEXT].starts_with(match &rows[0][0] { Value::Text(t) => t, _ => unreachable!() }));
                assert!(sp.streams[NUMBERS].len() > rows.len() * 12 && sp.streams[PAGES].is_empty());
                // what is derived is not written: the streams together are shorter than the pages' content
                let all: usize = sp.streams.iter().map(|s| s.len()).sum();
                assert!(churn || all < db.len(), "page {page}: {all} bytes of streams for {} of database", db.len());
            }
        }
    }

    /// Damaged databases: no panic; what is split comes back exactly. And damaged streams: an
    /// error or other bytes, never a panic or more memory than they have bytes for.
    #[test]
    fn damaged_databases_and_streams_do_not_panic() {
        let rows = orders(120);
        let blobs = [blob(5000, 1, false), blob(300, 2, true), blob(2000, 3, false)];
        let refs: Vec<&[u8]> = blobs.iter().map(|b| &b[..]).collect();
        let files = [table(&rows, 512, true), database(&refs, 512, true), table(&rows, 1024, false)];
        let mut s = 7u64;
        for round in 0..1500 {
            let db = &files[round % 3];
            let mut bad = db.clone();
            for _ in 0..1 + rng(&mut s) % 6 {
                let i = (rng(&mut s) % bad.len() as u64) as usize;
                // the cells and pointers more often than the content
                let i = if rng(&mut s) % 2 == 0 { i / 512 * 512 + i % 24 } else { i };
                bad[i] = rng(&mut s) as u8;
            }
            let Some(geo) = header(&bad) else { continue };
            if geo.pages * geo.page != bad.len() { continue; }
            if let Some(sp) = split(&bad, &geo) { assert!(back(&bad, &sp) == bad, "round {round}"); }
        }
        for (k, db) in files.iter().enumerate() {
            let geo = header(db).unwrap();
            let sp = split(db, &geo).unwrap();
            for _ in 0..400 {
                let mut streams = sp.streams.clone();
                let which = (rng(&mut s) % STREAMS as u64) as usize;
                if streams[which].is_empty() { continue; }
                let i = (rng(&mut s) % streams[which].len() as u64) as usize;
                match rng(&mut s) % 3 {
                    0 => streams[which][i] = rng(&mut s) as u8,
                    1 => streams[which][i] ^= 1 << (rng(&mut s) % 8),
                    _ => streams[which].truncate(i),
                }
                let st: [&[u8]; STREAMS] = std::array::from_fn(|i| &streams[i][..]);
                let mut out = vec![];
                let _ = rebuild(geo.page, geo.usable, &st, &mut out);
                assert!(out.len() <= 4 * db.len() + (1 << 16), "file {k}: damaged streams gave {} bytes", out.len());
            }
        }
    }
}
