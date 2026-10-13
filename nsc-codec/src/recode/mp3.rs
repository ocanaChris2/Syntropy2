//! MP3 (MPEG-1, MPEG-2 and MPEG-2.5 audio Layer III) recoder.
//!
//! A run of consecutive frames is taken apart into
//! * the frames as parsed records ([`Frame`]): header, side information, and each granule's scale
//!   factors and Huffman-decoded spectral values — what the MP3 model codes instead of the bytes;
//! * the unclaimed bytes of the frames' main-data areas: ancillary data, padding, the Xing/LAME
//!   information frame, and the areas of frames left unparsed.
//!
//! The lengths the format stores but the data implies are not kept: a granule's `part2_3_length`
//! is the size of its re-encoded bits, and a frame's `main_data_begin` follows from where the
//! previous frame's data ended. A frame whose data cannot be parsed cleanly (its bit-reservoir
//! reaches before the run, an invalid table, an overrun) is kept raw, and [`super::proven`] accepts
//! the split only when [`rebuild`] reproduces the run exactly. The parser therefore only has to be
//! a bijection on what it accepts, not a conforming decoder.
use super::mp3_tables::{BOOKS, LINBITS, NONE, QUADS, SCF_PARTITIONS, SFB_LONG, TREE_OF};
use crate::bits::{BitReader, BitWriter, PrefixDec};
use crate::transform::deflate::inflate::ExtentError;
use std::sync::OnceLock;

/// Fewest consecutive frames taken as an MP3 stream.
const MIN_FRAMES: usize = 4;
/// Frame records per leaf: leaves are what lanes are made of, so a long stream codes in parallel.
const LEAF_FRAMES: usize = 512;
/// Spectral lines per granule.
pub const LINES: usize = 576;

// ------------------------------------------------------------------ frame header
/// What a frame header says about the frame's layout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hdr {
    /// MPEG-2 or MPEG-2.5: one granule per frame, 9-bit scalefac_compress.
    pub lsf: bool,
    /// A 16-bit CRC follows the header.
    pub protected: bool,
    pub mono: bool,
    /// Joint stereo with intensity stereo on: the second channel's scale factors use the
    /// intensity partitions (MPEG-2).
    pub intensity: bool,
    /// Joint stereo coded as mid/side.
    pub ms: bool,
    /// Row of the band tables (see `mp3_tables`).
    pub row: usize,
    pub frame_len: usize,
    pub side_len: usize,
    /// Version and sample-rate bits: equal for all frames of one stream.
    key: u8,
}

impl Hdr {
    pub fn channels(&self) -> usize { if self.mono { 1 } else { 2 } }
    pub fn granules(&self) -> usize { if self.lsf { 1 } else { 2 } }
    /// Bytes before the main-data area: header, CRC, side information.
    pub fn head_len(&self) -> usize { 4 + 2 * self.protected as usize + self.side_len }
    /// 8 kHz MPEG-2.5, whose window-switching regions are twice as wide.
    fn is_8k(&self) -> bool { self.row == 1 }
}

/// Parse a Layer III frame header (free-format streams are not supported).
pub fn header(b: &[u8]) -> Option<Hdr> {
    if b.len() < 4 || b[0] != 0xFF || b[1] & 0xE0 != 0xE0 { return None; }
    let ver = (b[1] >> 3) & 3; // 0: MPEG-2.5, 2: MPEG-2, 3: MPEG-1
    if ver == 1 || (b[1] >> 1) & 3 != 1 { return None; }
    let (br, sr) = ((b[2] >> 4) as usize, ((b[2] >> 2) & 3) as usize);
    if br == 0 || br == 15 || sr == 3 { return None; }
    let lsf = ver != 3;
    let kbps = if lsf { [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160][br] } else { [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320][br] };
    let hz = [44100usize, 48000, 32000][sr] >> (lsf as u32 + (ver == 0) as u32);
    let frame_len = (if lsf { 72_000 } else { 144_000 }) * kbps / hz + ((b[2] >> 1) & 1) as usize;
    let mode = b[3] >> 6;
    let mono = mode == 3;
    let side_len = match (lsf, mono) { (false, false) => 32, (false, true) => 17, (true, false) => 17, (true, true) => 9 };
    let row = match ver { 0 => (sr == 2) as usize, 2 => 2 + sr, _ => 5 + sr };
    let h = Hdr { lsf, protected: b[1] & 1 == 0, mono, intensity: mode == 1 && b[3] & 0x10 != 0, ms: mode == 1 && b[3] & 0x20 != 0, row, frame_len, side_len, key: ver << 2 | sr as u8 };
    (frame_len > h.head_len()).then_some(h)
}

/// The run of consecutive frames of one stream at the start of `b`: their headers, and whether
/// `b` ended inside a frame that would have continued the run. (A run that stops at a frame
/// boundary where `b` ends is complete: should the stream go on, it is simply cut between two
/// frames.)
fn chain(b: &[u8]) -> (Vec<Hdr>, bool) {
    let mut frames: Vec<Hdr> = vec![];
    let mut o = 0usize;
    loop {
        let Some(h) = header(&b[o..]) else { return (frames, false) };
        if frames.first().is_some_and(|f| f.key != h.key) { return (frames, false); }
        if o + h.frame_len > b.len() { return (frames, true); }
        o += h.frame_len;
        frames.push(h);
    }
}

/// The extent of the MP3 stream starting at `b[i]`, by frame headers only: (end, estimated work
/// bytes).
pub fn extent(b: &[u8], i: usize) -> Result<(usize, usize), ExtentError> {
    let (frames, truncated) = chain(&b[i..]);
    if truncated && (frames.len() >= MIN_FRAMES || b.len() - i < 3 * 1441) { return Err(ExtentError::Truncated); }
    if frames.len() < MIN_FRAMES { return Err(ExtentError::Invalid); }
    let len: usize = frames.iter().map(|h| h.frame_len).sum();
    Ok((i + len, 5 * len))
}

// ------------------------------------------------------------------ parsed frames
/// One granule of one channel: its side information (without the lengths the data implies), its
/// scale factors and its spectral values.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Granule {
    pub big_values: u16,
    pub global_gain: u8,
    pub scalefac_compress: u16,
    pub window_switching: bool,
    pub block_type: u8,
    pub mixed: bool,
    pub preflag: bool,
    pub scalefac_scale: bool,
    pub count1table: bool,
    pub table_select: [u8; 3],
    /// Window switching: the three subblock gains. Otherwise: region0_count, region1_count, 0.
    pub aux: [u8; 3],
    /// The transmitted scale factors (those with a nonzero width).
    pub scf: Vec<u8>,
    /// The `big_values` pairs, signed.
    pub pairs: Vec<[i16; 2]>,
    /// The count1 quadruples: nonzero flags `v << 3 | w << 2 | x << 1 | y`, signs in the high nibble.
    pub quads: Vec<u8>,
    /// Bits left in the granule after its last code word, kept as they are.
    pub stuff_bits: u16,
    pub stuff: Vec<u8>,
}

/// A parsed frame's side information and data.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Parsed {
    /// Unclaimed main-data bytes between the previous parsed frame's data and this frame's.
    pub gap: u16,
    pub private: u8,
    /// MPEG-1: per channel, which scale-factor bands granule 1 shares with granule 0.
    pub scfsi: [u8; 2],
    /// `[granule][channel]`.
    pub gr: [[Granule; 2]; 2],
    /// The bits between the end of the data and the next byte boundary.
    pub padbits: u8,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Body {
    /// Side information kept as bytes; the frame's data stays among the unclaimed bytes.
    Raw(Vec<u8>),
    Parsed(Box<Parsed>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub hdr: [u8; 4],
    /// The frame's CRC when it is not the one its header and side information give (protected
    /// frames only); None when it is recomputed.
    pub crc: Option<[u8; 2]>,
    pub body: Body,
}

/// The CRC-16 of a protected Layer III frame: over the header's last two bytes and the side
/// information.
pub fn frame_crc(hdr: &[u8; 4], side: &[u8]) -> [u8; 2] {
    let mut crc = 0xFFFFu16;
    for &b in hdr[2..].iter().chain(side) {
        crc ^= (b as u16) << 8;
        for _ in 0..8 { crc = if crc & 0x8000 != 0 { crc << 1 ^ 0x8005 } else { crc << 1 }; }
    }
    crc.to_be_bytes()
}

/// Block layout of a granule: 0 long, 1 mixed, 2 short.
pub fn layout(g: &Granule) -> usize { if g.block_type == 2 { if g.mixed { 1 } else { 2 } } else { 0 } }

/// A granule's transmitted scale factors in order: (slot in the layout's band list, bit width —
/// possibly zero). `second`: granule 1 of an MPEG-1 frame, where `scfsi` drops shared bands (their
/// slots are skipped).
pub fn scf_slots(h: &Hdr, g: &Granule, second: bool, ch: usize, scfsi: u8) -> Vec<(u8, u8)> {
    let layout = layout(g);
    let (part, slen): ([u8; 4], [u8; 4]) = if !h.lsf {
        const SLEN: [[u8; 2]; 16] = [[0, 0], [0, 1], [0, 2], [0, 3], [3, 0], [1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [3, 3], [4, 2], [4, 3]];
        let [a, b] = SLEN[(g.scalefac_compress & 15) as usize];
        (SCF_PARTITIONS[layout][0], [a, a, b, b])
    } else {
        let sf = g.scalefac_compress as u32;
        let (idx, s): (usize, [u32; 4]) = if h.intensity && ch == 1 {
            let sf = sf >> 1;
            if sf < 180 { (4, [sf / 36, sf % 36 / 6, sf % 6, 0]) }
            else if sf < 244 { let sf = sf - 180; (5, [sf / 16, sf / 4 % 4, sf % 4, 0]) }
            else { let sf = sf - 244; (6, [sf / 3, sf % 3, 0, 0]) }
        } else if sf < 400 { (1, [sf / 80, sf / 16 % 5, sf / 4 % 4, sf % 4]) }
        else if sf < 500 { let sf = sf - 400; (2, [sf / 20, sf / 4 % 5, sf % 4, 0]) }
        else { let sf = sf - 500; (3, [sf / 3, sf % 3, 0, 0]) };
        (SCF_PARTITIONS[layout][idx], [s[0] as u8, s[1] as u8, s[2] as u8, s[3] as u8])
    };
    let mut w = Vec::with_capacity(39);
    let mut slot = 0u8;
    for k in 0..4 {
        let shared = second && layout == 0 && scfsi & (8 >> k) != 0;
        for _ in 0..part[k] {
            if !shared { w.push((slot, slen[k])); }
            slot += 1;
        }
    }
    w
}

/// Ends (in lines, at most `2 * big_values`) of the three Huffman regions.
pub fn regions(h: &Hdr, g: &Granule) -> [usize; 3] {
    let bv2 = 2 * g.big_values as usize;
    if g.window_switching {
        let r0 = match (g.block_type == 2, h.lsf, h.is_8k()) {
            (true, _, false) | (false, false, _) => 36,
            (true, _, true) => 72,
            (false, true, false) => 54,
            (false, true, true) => 108,
        };
        return [r0.min(bv2), bv2, bv2];
    }
    let end = |bands: usize| -> usize { SFB_LONG[h.row][..bands.min(22)].iter().map(|&w| w as usize).sum() };
    let i0 = g.aux[0] as usize + 1;
    let i1 = i0 + g.aux[1] as usize + 1;
    [end(i0).min(bv2), end(i1).min(bv2), bv2]
}

/// The pair table of each of the `big_values` pairs' regions: (pairs in region, table).
fn region_tables(h: &Hdr, g: &Granule) -> [(usize, u8); 3] {
    let r = regions(h, g);
    [(r[0] / 2, g.table_select[0]), ((r[1] - r[0]) / 2, g.table_select[1]), ((r[2] - r[1]) / 2, g.table_select[2])]
}

/// The pair table in force at each pair of a granule.
#[derive(Clone, Copy)]
pub struct PairTables {
    ends: [usize; 3],
    tables: [u8; 3],
}

impl PairTables {
    pub fn new(h: &Hdr, g: &Granule) -> Self {
        PairTables { ends: regions(h, g), tables: [g.table_select[0] & 31, g.table_select[1] & 31, g.table_select[2] & 31] }
    }
    /// Table of pair `p`.
    #[inline]
    pub fn table(&self, p: usize) -> u8 {
        if 2 * p < self.ends[0] { self.tables[0] } else if 2 * p < self.ends[1] { self.tables[1] } else { self.tables[2] }
    }
    /// Escape bits of pair `p` (0 when its table has none).
    #[inline]
    pub fn linbits(&self, p: usize) -> u8 { LINBITS[self.table(p) as usize] }
}

struct Decoders {
    pairs: Vec<PrefixDec>,
    quad_a: PrefixDec,
}

fn decoders() -> &'static Decoders {
    static D: OnceLock<Decoders> = OnceLock::new();
    D.get_or_init(|| Decoders {
        pairs: BOOKS.iter().map(|b| PrefixDec::new(b.iter().enumerate().filter(|(_, c)| c.1 > 0).map(|(i, c)| (c.0, c.1, i as u32)))).collect(),
        quad_a: PrefixDec::new(QUADS[0].iter().enumerate().map(|(i, c)| (c.0, c.1, i as u32))),
    })
}

pub fn side_ok(h: &Hdr, g: &Granule) -> bool {
    if g.big_values as usize > LINES / 2 || (g.window_switching && g.block_type == 0) { return false; }
    region_tables(h, g).iter().all(|&(n, t)| n == 0 || t == 0 || TREE_OF[(t & 31) as usize] != NONE)
}

/// Read one granule's side information (after part2_3_length); returns it with its
/// part2_3_length.
fn read_side_granule(r: &mut BitReader, lsf: bool) -> (Granule, usize) {
    let p23 = r.get(12) as usize;
    let mut g = Granule { big_values: r.get(9) as u16, global_gain: r.get(8) as u8, scalefac_compress: r.get(if lsf { 9 } else { 4 }) as u16, ..Default::default() };
    g.window_switching = r.get(1) == 1;
    if g.window_switching {
        g.block_type = r.get(2) as u8;
        g.mixed = r.get(1) == 1;
        g.table_select = [r.get(5) as u8, r.get(5) as u8, 0];
        g.aux = [r.get(3) as u8, r.get(3) as u8, r.get(3) as u8];
    } else {
        g.table_select = [r.get(5) as u8, r.get(5) as u8, r.get(5) as u8];
        g.aux = [r.get(4) as u8, r.get(3) as u8, 0];
    }
    if !lsf { g.preflag = r.get(1) == 1; }
    g.scalefac_scale = r.get(1) == 1;
    g.count1table = r.get(1) == 1;
    (g, p23)
}

fn write_side_granule(w: &mut BitWriter, g: &Granule, p23: usize, lsf: bool) {
    w.put(p23 as u32, 12);
    w.put(g.big_values as u32, 9);
    w.put(g.global_gain as u32, 8);
    w.put(g.scalefac_compress as u32, if lsf { 9 } else { 4 });
    w.put(g.window_switching as u32, 1);
    if g.window_switching {
        w.put(g.block_type as u32, 2);
        w.put(g.mixed as u32, 1);
        w.put(g.table_select[0] as u32, 5);
        w.put(g.table_select[1] as u32, 5);
        for a in g.aux { w.put(a as u32, 3); }
    } else {
        for t in g.table_select { w.put(t as u32, 5); }
        w.put(g.aux[0] as u32, 4);
        w.put(g.aux[1] as u32, 3);
    }
    if !lsf { w.put(g.preflag as u32, 1); }
    w.put(g.scalefac_scale as u32, 1);
    w.put(g.count1table as u32, 1);
}

/// Parse a granule's `p23` bits at the reader into `g` (whose side information is set). False if
/// they are not scale factors, `big_values` pairs and count1 quadruples within those bits.
fn read_granule(r: &mut BitReader, h: &Hdr, g: &mut Granule, p23: usize, second: bool, ch: usize, scfsi: u8) -> bool {
    let end = r.pos() + p23;
    for (_, w) in scf_slots(h, g, second, ch, scfsi) {
        if w > 0 { g.scf.push(r.get(w as u32) as u8); }
    }
    if r.pos() > end { return false; }
    let d = decoders();
    g.pairs.reserve(g.big_values as usize);
    for (n, t) in region_tables(h, g) {
        if n == 0 { continue; }
        let t = (t & 31) as usize;
        if t == 0 {
            g.pairs.extend(std::iter::repeat_n([0, 0], n));
            continue;
        }
        let (dec, lb) = (&d.pairs[TREE_OF[t] as usize], LINBITS[t] as u32);
        for _ in 0..n {
            let Some((sym, len)) = dec.peek(r) else { return false };
            r.skip(len);
            let mut v = [(sym >> 4) as i16, (sym & 15) as i16];
            for x in v.iter_mut() {
                if *x == 15 && lb > 0 { *x += r.get(lb) as i16; }
                if *x != 0 && r.get(1) == 1 { *x = -*x; }
            }
            g.pairs.push(v);
        }
        if r.pos() > end { return false; }
    }
    let mut line = 2 * g.big_values as usize;
    while line <= LINES - 4 && r.pos() < end {
        let save = r.pos();
        let flags = if g.count1table { 15 - r.get(4) } else {
            let Some((f, len)) = d.quad_a.peek(r) else { break };
            r.skip(len);
            f
        };
        let mut signs = 0;
        for k in [8, 4, 2, 1] {
            if flags & k != 0 && r.get(1) == 1 { signs |= k; }
        }
        if r.pos() > end {
            r.seek(save);
            break;
        }
        g.quads.push((flags | signs << 4) as u8);
        line += 4;
    }
    let left = end - r.pos();
    g.stuff_bits = left as u16;
    g.stuff = (0..left.div_ceil(8)).map(|i| r.get((left - 8 * i).min(8) as u32) as u8).collect();
    true
}

/// Append a granule's bits (scale factors, pairs, quadruples, stuffing). Errors on values no
/// code book has (a corrupt leaf).
fn write_granule(w: &mut BitWriter, h: &Hdr, g: &Granule, second: bool, ch: usize, scfsi: u8) -> Result<(), String> {
    let bad = || "mp3: values outside the code books (corrupt file)".to_string();
    let mut scf = g.scf.iter();
    for (_, width) in scf_slots(h, g, second, ch, scfsi) {
        if width > 0 {
            let v = *scf.next().ok_or_else(bad)? as u32;
            if v >> width != 0 { return Err(bad()); }
            w.put(v, width as u32);
        }
    }
    let mut pairs = g.pairs.iter();
    for (n, t) in region_tables(h, g) {
        let t = (t & 31) as usize;
        for _ in 0..n {
            let v = pairs.next().ok_or_else(bad)?;
            if t == 0 {
                if *v != [0, 0] { return Err(bad()); }
                continue;
            }
            if TREE_OF[t] == NONE { return Err(bad()); }
            let lb = LINBITS[t] as u32;
            let a = [v[0].unsigned_abs() as u32, v[1].unsigned_abs() as u32];
            let (code, len) = BOOKS[TREE_OF[t] as usize][(a[0].min(15) * 16 + a[1].min(15)) as usize];
            if len == 0 { return Err(bad()); }
            w.put(code, len as u32);
            for k in 0..2 {
                if a[k] >= 15 {
                    if lb == 0 && a[k] > 15 || a[k] - 15 >= 1 << lb { return Err(bad()); }
                    w.put(a[k] - 15, lb);
                }
                if a[k] != 0 { w.put((v[k] < 0) as u32, 1); }
            }
        }
    }
    for &q in &g.quads {
        let (flags, signs) = ((q & 15) as u32, (q >> 4) as u32);
        if signs & !flags != 0 { return Err(bad()); }
        let (code, len) = QUADS[g.count1table as usize][flags as usize];
        w.put(code, len as u32);
        for k in [8, 4, 2, 1] {
            if flags & k != 0 { w.put((signs & k != 0) as u32, 1); }
        }
    }
    let left = g.stuff_bits as usize;
    if g.stuff.len() != left.div_ceil(8) { return Err(bad()); }
    for (i, &s) in g.stuff.iter().enumerate() {
        let n = (left - 8 * i).min(8) as u32;
        if n < 8 && s as u32 >> n != 0 { return Err(bad()); }
        w.put(s as u32, n);
    }
    Ok(())
}

// ------------------------------------------------------------------ the frame records (leaf bytes)
/// Record tag bits: the frame is parsed (else its side information is raw); its CRC bytes
/// follow (else the CRC is recomputed or absent).
const TAG_PARSED: u8 = 1;
const TAG_CRC: u8 = 2;

impl Frame {
    /// Append the frame's record.
    pub fn write(&self, h: &Hdr, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.hdr);
        let crc_tag = if self.crc.is_some() { TAG_CRC } else { 0 };
        match &self.body {
            Body::Raw(side) => {
                out.push(crc_tag);
                out.extend_from_slice(&self.crc.unwrap_or_default()[..2 * self.crc.is_some() as usize]);
                out.extend_from_slice(side);
            }
            Body::Parsed(p) => {
                out.push(TAG_PARSED | crc_tag);
                out.extend_from_slice(&self.crc.unwrap_or_default()[..2 * self.crc.is_some() as usize]);
                out.extend_from_slice(&p.gap.to_le_bytes());
                out.push(p.private);
                if !h.lsf { out.push(p.scfsi[0] | p.scfsi[1] << 4); }
                for gr in 0..h.granules() {
                    for ch in 0..h.channels() {
                        let g = &p.gr[gr][ch];
                        out.extend_from_slice(&g.big_values.to_le_bytes());
                        out.push(g.global_gain);
                        out.extend_from_slice(&g.scalefac_compress.to_le_bytes());
                        out.push(g.window_switching as u8 | g.block_type << 1 | (g.mixed as u8) << 3 | (g.preflag as u8) << 4
                            | (g.scalefac_scale as u8) << 5 | (g.count1table as u8) << 6);
                        out.extend_from_slice(&g.table_select);
                        out.extend_from_slice(&g.aux);
                        out.extend_from_slice(&g.scf);
                        let pt = PairTables::new(h, g);
                        for (i, v) in g.pairs.iter().enumerate() {
                            let a = [v[0].unsigned_abs(), v[1].unsigned_abs()];
                            out.push(a[0].min(15) as u8 | ((v[0] < 0) as u8) << 7);
                            out.push(a[1].min(15) as u8 | ((v[1] < 0) as u8) << 7);
                            if pt.linbits(i) > 0 {
                                for x in a {
                                    if x >= 15 { out.extend_from_slice(&(x - 15).to_le_bytes()); }
                                }
                            }
                        }
                        out.push(g.quads.len() as u8);
                        out.extend_from_slice(&g.quads);
                        out.extend_from_slice(&g.stuff_bits.to_le_bytes());
                        out.extend_from_slice(&g.stuff);
                    }
                }
                out.push(p.padbits);
            }
        }
    }

    /// Parse the record at the start of `b`: the frame and the record's length.
    pub fn read(b: &[u8]) -> Result<(Frame, Hdr, usize), String> {
        let bad = || "mp3: bad frame record (corrupt file)".to_string();
        let mut c = Cur { b, o: 0 };
        let hdr: [u8; 4] = c.take(4)?.try_into().unwrap();
        let h = header(&hdr).ok_or_else(bad)?;
        let tag = c.u8()?;
        if tag > (TAG_PARSED | TAG_CRC) || (tag & TAG_CRC != 0 && !h.protected) { return Err(bad()); }
        let crc: Option<[u8; 2]> = if tag & TAG_CRC != 0 { Some(c.take(2)?.try_into().unwrap()) } else { None };
        let body = match tag & TAG_PARSED {
            0 => Body::Raw(c.take(h.side_len)?.to_vec()),
            _ => {
                let mut p = Box::new(Parsed { gap: c.u16()?, private: c.u8()?, ..Default::default() });
                if !h.lsf {
                    let s = c.u8()?;
                    p.scfsi = [s & 15, s >> 4];
                }
                for gr in 0..h.granules() {
                    for ch in 0..h.channels() {
                        let mut g = Granule { big_values: c.u16()?, global_gain: c.u8()?, scalefac_compress: c.u16()?, ..Default::default() };
                        let f = c.u8()?;
                        g.window_switching = f & 1 != 0;
                        g.block_type = (f >> 1) & 3;
                        g.mixed = f & 8 != 0;
                        g.preflag = f & 16 != 0;
                        g.scalefac_scale = f & 32 != 0;
                        g.count1table = f & 64 != 0;
                        g.table_select = c.take(3)?.try_into().unwrap();
                        g.aux = c.take(3)?.try_into().unwrap();
                        if f & 128 != 0 || g.scalefac_compress >= if h.lsf { 512 } else { 16 } || !side_ok(&h, &g) { return Err(bad()); }
                        let n = scf_slots(&h, &g, gr == 1, ch, p.scfsi[ch]).iter().filter(|s| s.1 > 0).count();
                        g.scf = c.take(n)?.to_vec();
                        let pt = PairTables::new(&h, &g);
                        g.pairs.reserve(g.big_values as usize);
                        for i in 0..g.big_values as usize {
                            let raw = [c.u8()?, c.u8()?];
                            let mut v = [0i16; 2];
                            for k in 0..2 {
                                let mut a = (raw[k] & 0x7F) as u16;
                                if a > 15 || (a == 0 && raw[k] != 0) { return Err(bad()); }
                                if a == 15 && pt.linbits(i) > 0 {
                                    a += c.u16()?;
                                    if a > 15 + 8191 { return Err(bad()); }
                                }
                                v[k] = if raw[k] & 0x80 != 0 { -(a as i16) } else { a as i16 };
                            }
                            g.pairs.push(v);
                        }
                        let nq = c.u8()? as usize;
                        if 2 * g.big_values as usize + 4 * nq > LINES { return Err(bad()); }
                        g.quads = c.take(nq)?.to_vec();
                        g.stuff_bits = c.u16()?;
                        if g.stuff_bits > 4095 { return Err(bad()); }
                        g.stuff = c.take((g.stuff_bits as usize).div_ceil(8))?.to_vec();
                        p.gr[gr][ch] = g;
                    }
                }
                p.padbits = c.u8()?;
                Body::Parsed(p)
            }
        };
        Ok((Frame { hdr, crc, body }, h, c.o))
    }
}

struct Cur<'a> {
    b: &'a [u8],
    o: usize,
}

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let s = self.b.get(self.o..self.o + n).ok_or("mp3: truncated frame record (corrupt file)")?;
        self.o += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> { Ok(self.take(1)?[0]) }
    fn u16(&mut self) -> Result<u16, String> { Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap())) }
}

// ------------------------------------------------------------------ split and rebuild
/// A run of frames taken apart: its length in the source, its frame records, and the unclaimed
/// main-data bytes.
pub struct Split {
    pub len: usize,
    pub frames: Vec<u8>,
    /// Offsets in `frames` where a leaf may end: every `LEAF_FRAMES` records.
    pub cuts: Vec<usize>,
    pub rest: Vec<u8>,
    /// Frames in the run, and how many of them were parsed.
    pub count: (usize, usize),
}

/// Take apart the MP3 stream at the start of `b` (None unless it starts with a run of frames).
pub fn split(b: &[u8]) -> Option<Split> {
    let (hdrs, _) = chain(b);
    if hdrs.len() < MIN_FRAMES { return None; }
    // the main-data areas of all frames, as one stream
    let len: usize = hdrs.iter().map(|h| h.frame_len).sum();
    let mut m = Vec::with_capacity(len);
    let mut area = Vec::with_capacity(hdrs.len());
    let mut o = 0;
    for h in &hdrs {
        area.push(m.len());
        m.extend_from_slice(&b[o + h.head_len()..o + h.frame_len]);
        o += h.frame_len;
    }
    let (mut frames, mut rest) = (Vec::with_capacity(4 * len), Vec::new());
    let (mut prev_end, mut parsed) = (0usize, 0usize);
    let mut cuts = vec![];
    o = 0;
    for (k, h) in hdrs.iter().enumerate() {
        if k > 0 && k % LEAF_FRAMES == 0 { cuts.push(frames.len()); }
        let f = &b[o..o + h.frame_len];
        let side = &f[h.head_len() - h.side_len..h.head_len()];
        let body = parse_frame(h, side, &m, area[k], prev_end);
        let hdr: [u8; 4] = f[..4].try_into().unwrap();
        let crc = (h.protected && [f[4], f[5]] != frame_crc(&hdr, side)).then(|| [f[4], f[5]]);
        if let Some((p, start, end)) = body {
            rest.extend_from_slice(&m[prev_end..start]);
            prev_end = end;
            parsed += 1;
            Frame { hdr, crc, body: Body::Parsed(p) }.write(h, &mut frames);
        } else {
            Frame { hdr, crc, body: Body::Raw(side.to_vec()) }.write(h, &mut frames);
        }
        o += h.frame_len;
    }
    rest.extend_from_slice(&m[prev_end..]);
    // Frame headers alone are weak evidence (tables in binaries can look like a few of them): it
    // is an MP3 stream, and worth a container, only if at least half its frames parse.
    if parsed < 2 || 2 * parsed < hdrs.len() { return None; }
    Some(Split { len, frames, cuts, rest, count: (hdrs.len(), parsed) })
}

/// Parse one frame's side information and main data (in the area stream `m`, the frame's own
/// area starting at `area`; nothing before `prev_end` is available): the parsed body and the byte
/// range of `m` its data occupies.
fn parse_frame(h: &Hdr, side: &[u8], m: &[u8], area: usize, prev_end: usize) -> Option<(Box<Parsed>, usize, usize)> {
    let mut r = BitReader::new(side, 0);
    let begin = r.get(if h.lsf { 8 } else { 9 }) as usize;
    let mut p = Box::new(Parsed { private: r.get(match (h.lsf, h.mono) { (false, true) => 5, (false, false) => 3, (true, true) => 1, (true, false) => 2 }) as u8, ..Default::default() });
    if !h.lsf {
        for ch in 0..h.channels() { p.scfsi[ch] = r.get(4) as u8; }
    }
    let mut p23 = [[0usize; 2]; 2];
    for gr in 0..h.granules() {
        for ch in 0..h.channels() {
            let (g, n) = read_side_granule(&mut r, h.lsf);
            if !side_ok(h, &g) { return None; }
            p.gr[gr][ch] = g;
            p23[gr][ch] = n;
        }
    }
    let start = area.checked_sub(begin)?;
    if start < prev_end || start - prev_end > u16::MAX as usize { return None; }
    p.gap = (start - prev_end) as u16;
    let mut r = BitReader::new(m, start * 8);
    for gr in 0..h.granules() {
        for ch in 0..h.channels() {
            if r.pos() + p23[gr][ch] > m.len() * 8 { return None; }
            let scfsi = p.scfsi[ch];
            if !read_granule(&mut r, h, &mut p.gr[gr][ch], p23[gr][ch], gr == 1, ch, scfsi) { return None; }
        }
    }
    let end = r.pos().div_ceil(8);
    if r.pos() % 8 != 0 { p.padbits = m[end - 1] & (0xFF >> (r.pos() % 8)); }
    Some((p, start, end))
}

/// The original bytes of a run of frames from its frame records and unclaimed bytes.
pub fn rebuild(frames: &[u8], rest: &[u8], out: &mut Vec<u8>) -> Result<(), String> {
    let bad = |what: &str| format!("mp3: {what} (corrupt file)");
    // pass 1: the records, and every frame's place in the area stream
    let mut recs: Vec<(Frame, Hdr, usize)> = vec![];
    let (mut o, mut m_len) = (0usize, 0usize);
    while o < frames.len() {
        let (f, h, n) = Frame::read(&frames[o..])?;
        o += n;
        recs.push((f, h, m_len));
        m_len += h.frame_len - h.head_len();
    }
    // pass 2: the area stream: each parsed frame's data at the place its gap gives, the
    // unclaimed bytes around them
    let mut m: Vec<u8> = Vec::with_capacity(m_len);
    let mut rest = rest.iter().copied();
    let mut sides: Vec<Vec<u8>> = Vec::with_capacity(recs.len());
    for (f, h, area) in &recs {
        let p = match &f.body {
            Body::Raw(side) => { sides.push(side.clone()); continue; }
            Body::Parsed(p) => p,
        };
        for _ in 0..p.gap { m.push(rest.next().ok_or_else(|| bad("unclaimed bytes missing"))?); }
        let start = m.len();
        let begin = area.checked_sub(start).filter(|&b| b < if h.lsf { 256 } else { 512 }).ok_or_else(|| bad("frame data out of place"))?;
        let mut w = BitWriter::new();
        let mut side = BitWriter::new();
        side.put(begin as u32, if h.lsf { 8 } else { 9 });
        side.put(p.private as u32, match (h.lsf, h.mono) { (false, true) => 5, (false, false) => 3, (true, true) => 1, (true, false) => 2 });
        if !h.lsf {
            for ch in 0..h.channels() { side.put(p.scfsi[ch] as u32, 4); }
        }
        for gr in 0..h.granules() {
            for ch in 0..h.channels() {
                let before = w.len();
                write_granule(&mut w, h, &p.gr[gr][ch], gr == 1, ch, p.scfsi[ch])?;
                let p23 = w.len() - before;
                if p23 > 4095 { return Err(bad("granule too long")); }
                write_side_granule(&mut side, &p.gr[gr][ch], p23, h.lsf);
            }
        }
        let pad = (8 - w.len() % 8) % 8;
        if p.padbits as u32 >> pad != 0 { return Err(bad("bad padding bits")); }
        w.put(p.padbits as u32, pad as u32);
        m.extend_from_slice(&w.into_bytes());
        let side = side.into_bytes();
        if side.len() != h.side_len { return Err(bad("side information size")); }
        sides.push(side);
    }
    m.extend(rest);
    if m.len() != m_len { return Err(bad("main data size mismatch")); }
    // pass 3: the frames
    for ((f, h, area), side) in recs.iter().zip(&sides) {
        out.extend_from_slice(&f.hdr);
        if h.protected { out.extend_from_slice(&f.crc.unwrap_or_else(|| frame_crc(&f.hdr, side))); }
        out.extend_from_slice(side);
        out.extend_from_slice(&m[*area..area + h.frame_len - h.head_len()]);
    }
    Ok(())
}

/// Developer statistics: the bits the original stream spends per category in the frame records
/// `frames`: [header + CRC, side information, scale factors, pair code words, escape bits, signs,
/// count1 code words, stuffing].
pub fn original_bits(frames: &[u8]) -> Result<[u64; 8], String> {
    let mut t = [0u64; 8];
    let mut o = 0;
    while o < frames.len() {
        let (f, h, n) = Frame::read(&frames[o..])?;
        o += n;
        t[0] += 32 + 16 * h.protected as u64;
        let _ = &f.crc;
        t[1] += 8 * h.side_len as u64;
        let Body::Parsed(p) = &f.body else { continue };
        for gr in 0..h.granules() {
            for ch in 0..h.channels() {
                let g = &p.gr[gr][ch];
                t[2] += scf_slots(&h, g, gr == 1, ch, p.scfsi[ch]).iter().map(|s| s.1 as u64).sum::<u64>();
                let pt = PairTables::new(&h, g);
                for (i, v) in g.pairs.iter().enumerate() {
                    let tb = pt.table(i) as usize;
                    if tb == 0 { continue; }
                    let a = [v[0].unsigned_abs().min(15) as usize, v[1].unsigned_abs().min(15) as usize];
                    t[3] += BOOKS[TREE_OF[tb] as usize][a[0] * 16 + a[1]].1 as u64;
                    for k in 0..2 {
                        if a[k] == 15 { t[4] += LINBITS[tb] as u64; }
                        if v[k] != 0 { t[5] += 1; }
                    }
                }
                for &q in &g.quads {
                    t[6] += QUADS[g.count1table as usize][(q & 15) as usize].1 as u64;
                    t[5] += (q & 15).count_ones() as u64;
                }
                t[7] += g.stuff_bits as u64;
            }
        }
    }
    Ok(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    const STEREO: &[u8] = include_bytes!("../../tests/fixtures/tone_stereo.mp3");
    const MONO_LSF: &[u8] = include_bytes!("../../tests/fixtures/tone_mono_lsf.mp3");

    fn rng(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }

    /// Every run of frames in `b` splits and rebuilds to exactly its bytes; returns (frames, parsed).
    fn check(b: &[u8]) -> (usize, usize) {
        let (mut i, mut n) = (0, (0, 0));
        while i < b.len() {
            match (b[i] == 0xFF).then(|| split(&b[i..])).flatten() {
                Some(s) => {
                    let mut out = vec![];
                    // a split that does not rebuild is refused by `recode::proven`; here it must
                    // never happen for what the parser accepted
                    rebuild(&s.frames, &s.rest, &mut out).unwrap();
                    assert!(out == b[i..i + s.len], "rebuild differs at {i}");
                    n = (n.0 + s.count.0, n.1 + s.count.1);
                    i += s.len;
                }
                None => i += 1,
            }
        }
        n
    }

    #[test]
    fn real_streams_rebuild_exactly() {
        // MPEG-1 joint stereo and MPEG-2 mono from LAME: every frame parses
        let (frames, parsed) = check(STEREO);
        assert!(frames >= 60 && parsed == frames, "{parsed}/{frames}");
        let (frames, parsed) = check(MONO_LSF);
        assert!(frames >= 40 && parsed == frames, "{parsed}/{frames}");
        // a stream cut in the middle: the first frames' data reaches before the cut, so they stay raw
        let cut = &STEREO[STEREO.len() / 3..];
        let (frames, parsed) = check(cut);
        assert!(frames > 10 && parsed * 2 >= frames, "{parsed}/{frames}");
        // frame headers without frame data are not taken as a stream
        let mut fake = vec![];
        for _ in 0..40 { fake.extend_from_slice(&[0xFF, 0xFB, 0x14, 0xC4]); fake.extend(std::iter::repeat_n(0xA7u8, 92)); }
        assert!(split(&fake).is_none(), "a table of header-like bytes was taken as an MP3 stream");
    }

    #[test]
    fn extent_follows_the_frame_headers() {
        let start = STEREO.iter().position(|&b| b == 0xFF).unwrap();
        let (end, _) = extent(STEREO, start).unwrap_or_else(|_| panic!("no stream at {start}"));
        assert_eq!(end, STEREO.len());
        assert!(matches!(extent(&STEREO[..STEREO.len() - 50], start), Err(ExtentError::Truncated)));
        assert!(matches!(extent(b"\xff\xfb\x90\x00 not an mp3 at all, just some text that goes on for a while", 0), Err(_)));
    }

    /// Damaged streams never panic the parser, and what it accepts still rebuilds exactly.
    #[test]
    fn damaged_streams_stay_exact() {
        let mut s = 77u64;
        for round in 0..300 {
            let mut b = if round % 2 == 0 { STEREO.to_vec() } else { MONO_LSF.to_vec() };
            for _ in 0..1 + rng(&mut s) % 6 {
                let at = (rng(&mut s) % b.len() as u64) as usize;
                match rng(&mut s) % 4 {
                    0 => b[at] ^= 1 << (rng(&mut s) % 8),
                    1 => b[at] = rng(&mut s) as u8,
                    2 => { b.remove(at); }
                    _ => b.truncate(at.max(8)),
                }
            }
            check(&b);
        }
    }

    /// Frame records that are not ours are refused, never followed blindly.
    #[test]
    fn bad_records_are_refused() {
        let sp = split(&STEREO[STEREO.iter().position(|&b| b == 0xFF).unwrap()..]).unwrap();
        let mut s = 5u64;
        for _ in 0..2000 {
            let mut f = sp.frames.clone();
            for _ in 0..1 + rng(&mut s) % 4 {
                let at = (rng(&mut s) % f.len() as u64) as usize;
                f[at] = rng(&mut s) as u8;
            }
            let mut out = vec![];
            let _ = rebuild(&f, &sp.rest, &mut out); // any result, no panic
            let mut out = vec![];
            let _ = rebuild(&f[..(rng(&mut s) % f.len() as u64) as usize], &sp.rest, &mut out);
        }
    }
}
