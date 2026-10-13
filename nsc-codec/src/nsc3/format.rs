//! NSC3 byte layout: a fixed header, self-describing blocks, and a trailer.
//!
//! ```text
//! header : "NSC3" fmt:u16 rev:u16 flags:u32 min_decoder:u16 features:u32
//!          n_bind:u16 × (id_len:u16 id fingerprint:u64)  hdr_crc:u32
//! block  : "BLK\0" hdr_len:v | raw_len:v comp_len:v raw_crc:u32 mode:u8
//!          [cfg:16 n_lanes:v × (leaves:v ac_len:v) if mode = CM]
//!          n_seg:v × (kind:u8 ver<<4|xform:u8 raw_len:v params_len:v params) | hdr_crc:u32
//!          | payload[comp_len] = lane coded streams, stored leaves, neural payloads
//! trailer: "END\0" orig_len:v file_crc:u32 mem_per_job:v n_blocks:v
//!          n_blocks × (Δfile_off:v Δraw_off:v) file_table_len:v file_table
//!          trailer_len:u32 "NSC3END\0"
//! (v = LEB128 varint; hdr_crc covers the block header from its sync.)
//! ```
//!
//! The segment table is a tree in pre-order. Leaves are coded by the model, in table order, into
//! one work buffer. Container segments (deflate stream, PNG chunk framing, PNG row filters) rebuild
//! their `raw_len` bytes from their children; their params start with the child count (u32).
//! The leaves are coded in lanes: consecutive runs of leaves, each with its own model and coded
//! stream, so a large block codes and decodes in parallel.
//!
//! Nothing in the header depends on the data, so a stream can be written front to back
//! (stdin → stdout). Each block carries everything needed to decode it, which allows sequential
//! decoding without the trailer. The trailer index allows random access. All integers are
//! little-endian. The parser bounds every length before it allocates.
use crate::cm::levels::{ModelCfg, CFG_BYTES};
use crate::crc::crc32;

pub const MAGIC: &[u8; 4] = b"NSC3";
/// Development builds wrote format 0 ("experimental", one revision at a time); release files are
/// format 1, which every later build must keep decoding.
pub const FMT_EXPERIMENTAL: u16 = 0;
pub const FORMAT: u16 = 1;
/// Decoder feature level of this build: a file whose `min_decoder` is higher needs a newer nsc.
/// New model kinds or kind versions are added under a higher level; existing ones never change.
/// Level 2 adds the format recoders (group, mp3, …).
pub const DECODER: u16 = 2;
/// Writer revision (informational, recorded in the header).
pub const REV: u16 = 15;
pub const BLOCK_SYNC: &[u8; 4] = b"BLK\0";
pub const END_SYNC: &[u8; 4] = b"END\0";
pub const TAIL_MAGIC: &[u8; 8] = b"NSC3END\0";

/// Largest block a decoder accepts. The encoder's largest level uses 128 MiB.
pub const MAX_BLOCK_RAW: u32 = 256 << 20;
pub const MAX_SEGMENTS: u32 = 1 << 20;
/// Largest block header a decoder accepts.
pub const MAX_BLOCK_HDR: usize = 32 << 20;

/// Segment kinds (stored in the file; also the bit index in `features`).
pub mod kind {
    pub const STORED: u8 = 0;
    pub const GENERIC: u8 = 1;
    pub const FAST: u8 = 2;
    pub const TEXT: u8 = 3;
    pub const IMAGE: u8 = 4;
    pub const AUDIO: u8 = 5;
    pub const EXE: u8 = 6;
    /// Text coded by the neural model (portable math); params carry the token stream layout.
    pub const NEURAL: u8 = 7;
    /// Container: a deflate stream, rebuilt from its decompressed content (the children but the
    /// last) and its reconstruction record (the last child, a leaf).
    pub const DEFLATE: u8 = 8;
    /// Container: PNG IDAT chunk framing around a zlib stream (the concatenated children).
    pub const PNG: u8 = 9;
    /// Container: PNG scanlines, rebuilt by re-applying each row's filter (children: filter types,
    /// unfiltered pixels).
    pub const ROWS: u8 = 10;
    /// Container: a JPEG, rebuilt by re-encoding its coefficients (children: header bytes,
    /// coefficients, and the encoder record when not empty).
    pub const JPEG: u8 = 11;
    /// JPEG DCT coefficients (compact serialization), coded by the coefficient model.
    pub const JPEGC: u8 = 12;
    /// Container: the concatenation of its children (the parts of a recoded stream that are
    /// planned like ordinary content, such as an MP3's ID3 tag with its cover picture).
    pub const GROUP: u8 = 13;
    /// Container: a run of MP3 frames, rebuilt from its frame records (the first child) and the
    /// unclaimed bytes of the frames' data areas (the second).
    pub const MP3: u8 = 14;
    /// MP3 frame records (`recode::mp3::Frame`), coded by the MP3 model.
    pub const MP3F: u8 = 15;
    /// Container: a run of AAC frames (ADTS, or headerless as in an MP4 track), rebuilt from its
    /// frame records (all children but the last) and the bytes kept raw (the last).
    pub const AAC: u8 = 16;
    /// AAC frame records (`recode::aac::Rec`), coded by the AAC model.
    pub const AACF: u8 = 17;
    /// Container: runs of several streams, interleaved (the samples of an MP4's tracks, with
    /// whatever lies between them). Children: the plan — (stream, length) pairs in file order —
    /// then one child per stream.
    pub const MUX: u8 = 18;
    /// Container: an H.264 stream of length-prefixed NAL units, rebuilt from its records (all
    /// children but the last) and the bytes kept raw (the last).
    pub const H264: u8 = 19;
    /// H.264 stream records (`recode::h264`): parameter sets, and CABAC slices as their bins,
    /// coded by the H.264 model.
    pub const H264B: u8 = 20;
    /// Container: a SQLite database taken apart (`recode::sqlite`). Parameters: the page size
    /// as a power of two, the reserved bytes of a page. Children: the structure, the numbers,
    /// the text, the blobs, the long rows, the gaps, the other pages.
    pub const SQLITE: u8 = 21;
    /// Container: a run of FLAC frames decoded to samples (`recode::flac`). Parameters: the
    /// channels, the bytes of a sample. Children: the frames' records, the samples (an audio
    /// leaf).
    pub const FLAC: u8 = 22;
    /// Container: a GIF with its images decoded (`recode::gif`). Children: the record, then the
    /// images as a stream of frames.
    pub const GIF: u8 = 23;
    /// Leaf: frames of colour indices, coded by `cm::indexed`.
    pub const PIX: u8 = 24;
    /// Container: the compressed strips of a file's pictures, decoded (`recode::tiff`).
    /// Children: the record, the bytes between the strips, then each picture's pixels (an image
    /// leaf).
    pub const STRIPS: u8 = 25;
    /// Container: an H.264 byte stream (`recode::ts`). Children: the start codes' record, the
    /// stream with lengths instead (an H.264 container, or bytes).
    pub const ANNEXB: u8 = 26;
    /// Container: the enciphered part of a Type 1 font (`recode::type1`). Its one child: the
    /// same bytes in the clear.
    pub const TYPE1: u8 = 27;
    /// Container: a lossless JPEG decoded to its samples (`recode::ljpeg`). Children: the
    /// stream's header (and its last byte's padding), the samples (an image leaf).
    pub const LJPEG: u8 = 28;
    /// Container: the packed planes of a Photoshop document, unpacked (`recode::psd`).
    /// Children: the record, the bytes between the planes, then each layer's pixels.
    pub const PSD: u8 = 29;
    /// Container: a ZIP archive with the fields its headers repeat derived (`recode::zip`).
    /// Children: the record, the archive less its entries' data, then each entry's data.
    pub const ZIP: u8 = 30;
    /// The kinds of decoder level 1 (nsc 0.3): all a file written with `--compat 1` contains.
    pub const LEVEL1: u32 = (1 << STORED) | (1 << GENERIC) | (1 << FAST) | (1 << TEXT) | (1 << IMAGE) | (1 << AUDIO) | (1 << EXE)
        | (1 << NEURAL) | (1 << DEFLATE) | (1 << PNG) | (1 << ROWS) | (1 << JPEG) | (1 << JPEGC);
    /// Kinds this build can decode. Kinds from 32 up have no `features` bit: the decoder level
    /// that introduced them gates them.
    pub const KNOWN: u32 = LEVEL1 | (1 << GROUP) | (1 << MP3) | (1 << MP3F) | (1 << AAC) | (1 << AACF) | (1 << MUX) | (1 << H264) | (1 << H264B) | (1 << SQLITE) | (1 << FLAC) | (1 << GIF) | (1 << PIX) | (1 << STRIPS) | (1 << ANNEXB) | (1 << TYPE1) | (1 << LJPEG) | (1 << PSD) | (1 << ZIP);
    pub fn known(k: u8) -> bool { k < 32 && KNOWN >> k & 1 == 1 }
    pub fn container(k: u8) -> bool { matches!(k, DEFLATE | PNG | ROWS | JPEG | GROUP | MP3 | AAC | MUX | H264 | SQLITE | FLAC | GIF | STRIPS | ANNEXB | TYPE1 | LJPEG | PSD | ZIP) }
    pub fn name(k: u8) -> &'static str {
        match k {
            STORED => "stored", GENERIC => "generic", FAST => "fast", TEXT => "text", IMAGE => "image", AUDIO => "audio", EXE => "exe",
            NEURAL => "neural", DEFLATE => "deflate", PNG => "png", ROWS => "png-rows", JPEG => "jpeg", JPEGC => "jpeg-coefs",
            GROUP => "group", MP3 => "mp3", MP3F => "mp3-frames", AAC => "aac", AACF => "aac-frames", MUX => "mux", H264 => "h264",
            H264B => "h264-records", SQLITE => "sqlite", FLAC => "flac", GIF => "gif", PIX => "gif-frames", STRIPS => "strips", ANNEXB => "h264-bytestream", TYPE1 => "type1-font", LJPEG => "lossless-jpeg", PSD => "psd", ZIP => "zip", _ => "?",
        }
    }
    /// Bit of kind `k` in a set of kinds (kinds past the set's width are dropped).
    pub fn bit(k: u8) -> u128 { 1u128.checked_shl(k as u32).unwrap_or(0) }
}

/// What a writer may put in a file: the header's `features` (kinds it may use) and the decoder
/// level they need. `specialists`: the format recoders and their kinds are enabled; without them
/// the file is readable by every format-1 decoder (nsc 0.3).
pub fn writer_features(specialists: bool) -> (u32, u16) {
    if specialists { (kind::KNOWN, DECODER) } else { (kind::LEVEL1, 1) }
}

pub const MODE_STORED: u8 = 0;
pub const MODE_CM: u8 = 1;
/// In the file only: a CM block whose segment table is coded (`table`), for decoders of level 2.
const MODE_CM_PACKED: u8 = 2;

pub struct Header {
    pub flags: u32,
    pub features: u32,
    /// Lowest decoder level that can read the file.
    pub min_decoder: u16,
    pub bindings: Vec<(String, u64)>,
}

#[derive(Clone, Debug)]
pub struct Segment {
    pub kind: u8,
    pub ver: u8,
    pub xform: u8,
    pub raw_len: u32,
    pub params: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct BlockHeader {
    pub raw_len: u32,
    pub comp_len: u32,
    pub raw_crc: u32,
    pub mode: u8,
    pub cfg: Option<ModelCfg>,
    /// Lanes in order: (number of leaves, coded-stream bytes). Empty for stored blocks.
    pub lanes: Vec<(u32, u32)>,
    pub segments: Vec<Segment>,
    /// The segment table may be written coded (a writer for decoders of level 2); as read: it was.
    pub packed: bool,
}

impl BlockHeader {
    /// Bytes of all lanes' coded streams (the start of the payload).
    pub fn ac_len(&self) -> usize { self.lanes.iter().map(|l| l.1 as usize).sum() }
}

pub struct Trailer {
    pub orig_len: u64,
    pub file_crc: u32,
    pub mem_per_job: u64,
    pub index: Vec<(u64, u64)>,
    pub file_table: Vec<u8>,
}

// ------------------------------------------------------------------ writing
pub fn write_header(h: &Header) -> Vec<u8> {
    let mut b = Vec::with_capacity(64);
    b.extend_from_slice(MAGIC);
    b.extend_from_slice(&FORMAT.to_le_bytes());
    b.extend_from_slice(&REV.to_le_bytes());
    b.extend_from_slice(&h.flags.to_le_bytes());
    b.extend_from_slice(&h.min_decoder.to_le_bytes());
    b.extend_from_slice(&h.features.to_le_bytes());
    b.extend_from_slice(&(h.bindings.len() as u16).to_le_bytes());
    for (id, fp) in &h.bindings {
        b.extend_from_slice(&(id.len() as u16).to_le_bytes());
        b.extend_from_slice(id.as_bytes());
        b.extend_from_slice(&fp.to_le_bytes());
    }
    let c = crc32(&b);
    b.extend_from_slice(&c.to_le_bytes());
    b
}

pub fn put_var(o: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 { o.push(v as u8 | 0x80); v >>= 7; }
    o.push(v as u8);
}

/// Serialise a block header (sync through hdr_crc). The payload follows it.
pub fn write_block_header(bh: &BlockHeader) -> Vec<u8> {
    let mut body = Vec::with_capacity(64);
    put_var(&mut body, bh.raw_len as u64);
    put_var(&mut body, bh.comp_len as u64);
    body.extend_from_slice(&bh.raw_crc.to_le_bytes());
    let mut table = Vec::with_capacity(8 * bh.segments.len() + 4);
    put_var(&mut table, bh.segments.len() as u64);
    for s in &bh.segments {
        debug_assert!(s.ver < 16 && s.xform < 16);
        table.extend_from_slice(&[s.kind, s.ver << 4 | s.xform]);
        put_var(&mut table, s.raw_len as u64);
        put_var(&mut table, s.params.len() as u64);
        table.extend_from_slice(&s.params);
    }
    // the table coded, where that is allowed and shorter
    let coded = if bh.packed && bh.mode == MODE_CM && (super::table::MIN..=super::table::MAX).contains(&table.len()) {
        let mut c = vec![];
        put_var(&mut c, table.len() as u64);
        c.extend(super::table::pack(&table));
        (c.len() < table.len()).then_some(c)
    } else {
        None
    };
    body.push(if coded.is_some() { MODE_CM_PACKED } else { bh.mode });
    if bh.mode == MODE_CM {
        body.extend_from_slice(&bh.cfg.expect("CM block needs a model config").to_bytes());
        put_var(&mut body, bh.lanes.len() as u64);
        for &(leaves, ac) in &bh.lanes {
            put_var(&mut body, leaves as u64);
            put_var(&mut body, ac as u64);
        }
    }
    body.extend_from_slice(coded.as_deref().unwrap_or(&table));
    let mut b = Vec::with_capacity(body.len() + 12);
    b.extend_from_slice(BLOCK_SYNC);
    put_var(&mut b, (body.len() + 4) as u64);
    b.extend_from_slice(&body);
    let c = crc32(&b);
    b.extend_from_slice(&c.to_le_bytes());
    b
}

pub fn write_trailer(t: &Trailer) -> Vec<u8> {
    let mut b = Vec::with_capacity(64 + t.index.len() * 6);
    b.extend_from_slice(END_SYNC);
    put_var(&mut b, t.orig_len);
    b.extend_from_slice(&t.file_crc.to_le_bytes());
    put_var(&mut b, t.mem_per_job);
    put_var(&mut b, t.index.len() as u64);
    let (mut pf, mut pr) = (0u64, 0u64);
    for &(fo, ro) in &t.index {
        put_var(&mut b, fo - pf);
        put_var(&mut b, ro - pr);
        (pf, pr) = (fo, ro);
    }
    put_var(&mut b, t.file_table.len() as u64);
    b.extend_from_slice(&t.file_table);
    let len = b.len() as u32;
    b.extend_from_slice(&len.to_le_bytes());
    b.extend_from_slice(TAIL_MAGIC);
    b
}

// ------------------------------------------------------------------ reading
/// Cursor over a byte slice whose reads fail instead of panicking.
pub struct Cur<'a> {
    pub b: &'a [u8],
    pub o: usize,
}

impl<'a> Cur<'a> {
    pub fn new(b: &'a [u8]) -> Self { Cur { b, o: 0 } }
    pub fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.b.len() - self.o < n { return Err("truncated NSC3 data".into()); }
        let s = &self.b[self.o..self.o + n];
        self.o += n;
        Ok(s)
    }
    pub fn u8(&mut self) -> Result<u8, String> { Ok(self.take(1)?[0]) }
    pub fn u16(&mut self) -> Result<u16, String> { Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap())) }
    pub fn u32(&mut self) -> Result<u32, String> { Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap())) }
    pub fn u64(&mut self) -> Result<u64, String> { Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap())) }
    pub fn var(&mut self) -> Result<u64, String> {
        let mut v = 0u64;
        for sh in (0..64).step_by(7) {
            let b = self.u8()?;
            v |= ((b & 0x7f) as u64) << sh;
            if b & 0x80 == 0 { return Ok(v); }
        }
        Err("bad number in NSC3 data".into())
    }
    fn var32(&mut self) -> Result<u32, String> { u32::try_from(self.var()?).map_err(|_| "number out of range in NSC3 data".to_string()) }
}

/// Read a varint from a stream, appending its bytes to `raw`.
fn read_var<R: std::io::Read>(r: &mut R, raw: &mut Vec<u8>) -> Result<u64, String> {
    let mut v = 0u64;
    for sh in (0..64).step_by(7) {
        let mut b = [0u8; 1];
        r.read_exact(&mut b).map_err(|_| "truncated NSC3 stream".to_string())?;
        raw.push(b[0]);
        v |= ((b[0] & 0x7f) as u64) << sh;
        if b[0] & 0x80 == 0 { return Ok(v); }
    }
    Err("bad number in NSC3 data".into())
}

/// Read the rest of a block header from `r` (its sync was already read and matched). Returns the
/// header and its total length, sync included.
pub fn read_block_header<R: std::io::Read>(r: &mut R) -> Result<(BlockHeader, usize), String> {
    let mut raw = BLOCK_SYNC.to_vec();
    let hl = read_var(r, &mut raw)? as usize;
    if !(4..=MAX_BLOCK_HDR).contains(&hl) { return Err("bad block header length (corrupt file)".into()); }
    let start = raw.len();
    raw.resize(start + hl, 0);
    r.read_exact(&mut raw[start..]).map_err(|_| "truncated NSC3 stream (block header)".to_string())?;
    let n = raw.len();
    Ok((parse_block_header(&raw)?, n))
}

/// Read the rest of a trailer from `r` (its sync was already read and matched).
pub fn read_trailer<R: std::io::Read>(r: &mut R) -> Result<Trailer, String> {
    let mut raw = vec![];
    let mut take = |r: &mut R, raw: &mut Vec<u8>, n: usize| -> Result<(), String> {
        let s = raw.len();
        raw.resize(s + n, 0);
        r.read_exact(&mut raw[s..]).map_err(|_| "truncated NSC3 stream (trailer)".to_string())
    };
    read_var(r, &mut raw)?;
    take(r, &mut raw, 4)?;
    read_var(r, &mut raw)?;
    let n = read_var(r, &mut raw)?;
    if n > 1 << 28 { return Err("trailer index too large".into()); }
    for _ in 0..2 * n { read_var(r, &mut raw)?; }
    let ft = read_var(r, &mut raw)? as usize;
    if ft > 1 << 30 { return Err("trailer file table too large".into()); }
    take(r, &mut raw, ft + 12)?;
    parse_trailer(&raw)
}

/// Bytes needed to know the header's full length, given its first 20 bytes; then the header
/// itself. Returns (header, total header length).
pub fn parse_header(b: &[u8]) -> Result<(Header, usize), String> {
    let mut c = Cur::new(b);
    if c.take(4)? != MAGIC { return Err("not an NSC3 file".into()); }
    let fmt = c.u16()?;
    let rev = c.u16()?;
    let flags = c.u32()?;
    let min_decoder = c.u16()?;
    let features = c.u32()?;
    if fmt == FMT_EXPERIMENTAL {
        return Err(format!(
            "this file was written by a development build of nsc (experimental NSC3 revision {rev}); \
             release builds read NSC3 format {FORMAT}. Decompress it with the build that wrote it."));
    }
    if fmt != FORMAT || min_decoder > DECODER {
        return Err(format!("this file needs a newer nsc (NSC3 format {fmt}, decoder level {min_decoder}; this build reads format {FORMAT}, level {DECODER})"));
    }
    if features & !kind::KNOWN != 0 {
        return Err(format!("file needs model kinds this nsc build lacks (feature bits {:#x})", features & !kind::KNOWN));
    }
    let n = c.u16()? as usize;
    if n > 64 { return Err("too many model bindings".into()); }
    let mut bindings = vec![];
    for _ in 0..n {
        let l = c.u16()? as usize;
        if l > 256 { return Err("model id too long".into()); }
        let id = String::from_utf8(c.take(l)?.to_vec()).map_err(|_| "bad model id".to_string())?;
        bindings.push((id, c.u64()?));
    }
    let end = c.o;
    let stored = c.u32()?;
    if crc32(&b[..end]) != stored { return Err("NSC3 header checksum mismatch (corrupt file)".into()); }
    Ok((Header { flags, features, min_decoder, bindings }, c.o))
}

/// Parse a whole block header (sync through hdr_crc). The segment tree's structure is checked by
/// the block decoder.
pub fn parse_block_header(raw: &[u8]) -> Result<BlockHeader, String> {
    if raw.len() < 9 || &raw[..4] != BLOCK_SYNC { return Err("truncated block header".into()); }
    let (whole, crc_bytes) = raw.split_at(raw.len() - 4);
    if crc32(whole) != u32::from_le_bytes(crc_bytes.try_into().unwrap()) {
        return Err("block header checksum mismatch (corrupt file)".into());
    }
    let mut c = Cur::new(whole);
    c.take(4)?;
    let hl = c.var()? as usize;
    if hl != raw.len() - c.o { return Err("block header length mismatch (corrupt file)".into()); }
    let data = whole;
    let raw_len = c.var32()?;
    let comp_len = c.var32()?;
    let raw_crc = c.u32()?;
    let mode = c.u8()?;
    if raw_len > MAX_BLOCK_RAW { return Err(format!("block of {raw_len} bytes exceeds the {MAX_BLOCK_RAW}-byte limit")); }
    let packed = mode == MODE_CM_PACKED;
    let mode = if packed { MODE_CM } else { mode };
    let (cfg, lanes) = match mode {
        MODE_STORED => {
            if comp_len != raw_len { return Err("stored block length mismatch".into()); }
            (None, vec![])
        }
        MODE_CM => {
            let cfg = ModelCfg::from_bytes(c.take(CFG_BYTES)?)?;
            let n = c.var()? as usize;
            if n == 0 || n > (data.len() - c.o) / 2 { return Err("bad lane count".into()); }
            let mut lanes = Vec::with_capacity(n);
            let mut ac = 0u64;
            for _ in 0..n {
                let (leaves, len) = (c.var32()?, c.var32()?);
                if leaves == 0 { return Err("empty lane".into()); }
                ac += len as u64;
                lanes.push((leaves, len));
            }
            if ac > comp_len as u64 { return Err("coded stream longer than block payload".into()); }
            if comp_len > raw_len.saturating_add(1 << 16) { return Err("block payload implausibly large".into()); }
            (Some(cfg), lanes)
        }
        m => return Err(format!("unknown block mode {m}")),
    };
    // the segment table: as it is, or coded
    let table;
    let (mut c, data) = if packed {
        let len = c.var()? as usize;
        table = super::table::unpack(&whole[c.o..], len)?;
        (Cur::new(&table), &table[..])
    } else {
        (c, data)
    };
    let n = c.var()?;
    if n == 0 || n > MAX_SEGMENTS as u64 || n as usize > data.len() / 4 { return Err("bad segment count".into()); }
    let mut segments = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let kind = c.u8()?;
        let vx = c.u8()?;
        let (ver, xform) = (vx >> 4, vx & 15);
        let len = c.var32()?;
        let pl = c.var()? as usize;
        if pl > u16::MAX as usize { return Err("segment parameters too long".into()); }
        let params = c.take(pl)?.to_vec();
        if kind >= 32 { return Err(format!("bad segment kind {kind} (corrupt file)")); }
        segments.push(Segment { kind, ver, xform, raw_len: len, params });
    }
    if c.o != data.len() { return Err("block header has trailing bytes".into()); }
    Ok(BlockHeader { raw_len, comp_len, raw_crc, mode, cfg, lanes, segments, packed })
}

/// Parse the trailer body (everything after the "END\0" sync, through the tail magic).
pub fn parse_trailer(b: &[u8]) -> Result<Trailer, String> {
    let mut c = Cur::new(b);
    let orig_len = c.var()?;
    let file_crc = c.u32()?;
    let mem_per_job = c.var()?;
    let n = c.var()? as usize;
    if n > b.len() / 2 { return Err("trailer index truncated".into()); }
    let mut index = Vec::with_capacity(n);
    let (mut fo, mut ro) = (0u64, 0u64);
    for _ in 0..n {
        fo = fo.checked_add(c.var()?).ok_or("bad trailer index")?;
        ro = ro.checked_add(c.var()?).ok_or("bad trailer index")?;
        index.push((fo, ro));
    }
    let ft = c.var()? as usize;
    let file_table = c.take(ft)?.to_vec();
    let len = c.u32()? as usize;
    if c.take(8)? != TAIL_MAGIC || len != c.o - 8 { return Err("bad NSC3 trailer".into()); }
    if c.o != b.len() { return Err("data after the NSC3 trailer".into()); }
    Ok(Trailer { orig_len, file_crc, mem_per_job, index, file_table })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cm::levels::{preset, Kind};

    #[test]
    fn header_block_trailer_roundtrip() {
        let h = write_header(&Header { flags: 0, features: 0b111, min_decoder: 1, bindings: vec![("base-v1".into(), 42)] });
        let (hh, n) = parse_header(&h).unwrap();
        assert_eq!(n, h.len());
        assert_eq!(hh.bindings, vec![("base-v1".to_string(), 42)]);
        let mut bad = h.clone();
        bad[9] ^= 1;
        assert!(parse_header(&bad).is_err());

        let bh = BlockHeader {
            raw_len: 1000, comp_len: 400, raw_crc: 7, mode: MODE_CM,
            cfg: Some(preset(5, Kind::Generic, 1000)), lanes: vec![(1, 400)],
            segments: vec![Segment { kind: kind::GENERIC, ver: 1, xform: 0, raw_len: 1000, params: vec![] }],
            packed: false,
        };
        let b = write_block_header(&bh);
        let (got, n) = read_block_header(&mut &b[4..]).unwrap();
        assert_eq!(n, b.len());
        assert_eq!((got.raw_len, got.comp_len, got.mode, got.ac_len()), (1000, 400, MODE_CM, 400));
        let mut bad = b.clone();
        bad[10] ^= 0x10;
        assert!(parse_block_header(&bad).is_err());

        let t = write_trailer(&Trailer { orig_len: 1000, file_crc: 9, mem_per_job: 1 << 20, index: vec![(30, 0), (900, 500)], file_table: vec![5] });
        let tt = parse_trailer(&t[4..]).unwrap();
        assert_eq!((tt.orig_len, tt.file_crc, tt.index.clone(), tt.file_table.clone()), (1000, 9, vec![(30, 0), (900, 500)], vec![5]));
        assert_eq!(read_trailer(&mut &t[4..]).unwrap().index, vec![(30, 0), (900, 500)]);
    }
}
