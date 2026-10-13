//! H.264 / AVC: parsing CABAC-coded slices down to their binary decisions ("bins").
//!
//! CABAC is an adaptive binary arithmetic coder: every syntax element is turned into bins, and
//! each bin is coded with one of 460 adaptive contexts chosen from the element and its already
//! decoded neighbourhood. To code a slice better than CABAC did, its bins have to be recovered,
//! which takes the full macroblock syntax — but no picture reconstruction: contexts depend on
//! syntax elements only.
//!
//! [`walk`] is that syntax, written once over a source of bins ([`Bins`]): the arithmetic decoder
//! when a slice is taken apart, the stored bins when it is modelled or rebuilt. Everything it
//! does not support (interlaced coding, 4:4:4, high bit depths, slice groups, CAVLC) is refused
//! up front, and a slice is only accepted when re-encoding its bins reproduces its bytes.
use super::h264_tables::{CONTEXTS, INIT, LAST_8X8, NEXT_LPS, RANGE_LPS, SIG_8X8};
use crate::bits::{BitReader, BitWriter};

#[path = "h264_cavlc.rs"]
pub mod cavlc;

// ------------------------------------------------------------------ NAL units and Exp-Golomb
/// The RBSP of a NAL unit payload: emulation prevention bytes removed.
pub fn unescape(nal: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(nal.len());
    let mut zeros = 0;
    for &b in nal {
        if zeros >= 2 && b == 3 {
            zeros = 0;
            continue;
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
        out.push(b);
    }
    out
}

/// The NAL unit payload of an RBSP: emulation prevention bytes inserted as the standard requires.
pub fn escape(rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len() + rbsp.len() / 64 + 2);
    let mut zeros = 0;
    for &b in rbsp {
        if zeros >= 2 && b <= 3 {
            out.push(3);
            zeros = 0;
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
        out.push(b);
    }
    // an RBSP ending in zero bytes (cabac_zero_words) gets a final 0x03; so does any two zeros
    // left at the end, which makes [`unescape`] this function's inverse for all input
    if zeros >= 2 { out.push(3); }
    out
}

fn ue(r: &mut BitReader) -> Option<u32> {
    let mut zeros = 0;
    while r.get(1) == 0 {
        zeros += 1;
        if zeros > 31 { return None; }
    }
    Some(((1u64 << zeros) - 1 + r.get(zeros) as u64) as u32)
}

fn se(r: &mut BitReader) -> Option<i32> {
    let k = ue(r)? as i64;
    Some(if k & 1 == 1 { (k + 1) / 2 } else { -(k / 2) } as i32)
}

// ------------------------------------------------------------------ parameter sets
/// Largest picture handled, in macroblocks (the largest frame size of level 6.2: 8K video).
pub const MAX_MBS: usize = 139_264;

/// What the parser needs of a sequence parameter set.
#[derive(Clone, Debug, Default)]
pub struct Sps {
    pub id: u32,
    pub chroma_format: u32,
    pub log2_max_frame_num: u32,
    pub poc_type: u32,
    pub log2_max_poc_lsb: u32,
    pub delta_pic_order_always_zero: bool,
    pub frame_mbs_only: bool,
    pub direct_8x8_inference: bool,
    pub width_mbs: usize,
    pub height_mbs: usize,
    /// Bits of a sample above 8 (0 to 6; luma and chroma alike).
    pub depth: u32,
    /// Everything [`walk`] handles: progressive 8-bit 4:2:0 or monochrome.
    pub supported: bool,
    /// Why not, for diagnostics.
    pub why: &'static str,
}

#[derive(Clone, Debug, Default)]
pub struct Pps {
    pub id: u32,
    pub sps: u32,
    pub cabac: bool,
    pub pic_order_present: bool,
    pub num_ref_idx: [u32; 2],
    pub weighted_pred: bool,
    pub weighted_bipred: u32,
    pub pic_init_qp: i32,
    pub deblocking_control: bool,
    pub redundant_pic_cnt: bool,
    pub transform_8x8: bool,
    pub supported: bool,
    /// Slice groups (flexible macroblock ordering) in use.
    pub fmo: bool,
}

fn skip_scaling_list(r: &mut BitReader, size: usize) -> Option<()> {
    let (mut last, mut next) = (8i32, 8i32);
    for _ in 0..size {
        if next != 0 { next = (last + se(r)? + 256) % 256; }
        if next != 0 { last = next; }
    }
    Some(())
}

/// Parse a sequence parameter set (the NAL payload after its header byte).
pub fn parse_sps(nal: &[u8]) -> Option<Sps> {
    let b = unescape(nal);
    let mut r = BitReader::new(&b, 0);
    let profile = r.get(8);
    r.skip(16);
    let mut s = Sps { id: ue(&mut r)?, chroma_format: 1, ..Default::default() };
    let mut plain = true;
    if matches!(profile, 100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135) {
        s.chroma_format = ue(&mut r)?;
        if s.chroma_format == 3 && r.get(1) == 1 { plain = false; } // separate colour planes
        let (bl, bc) = (ue(&mut r)?, ue(&mut r)?);
        // deeper samples change the range of the quantiser and nothing of the syntax (but a
        // PCM macroblock's samples: such a slice stays as bytes)
        if bl != bc || bl > 6 { plain = false; }
        s.depth = bl;
        if r.get(1) == 1 { plain = false; } // lossless transform bypass
        if r.get(1) == 1 {
            for i in 0..if s.chroma_format != 3 { 8 } else { 12 } {
                if r.get(1) == 1 { skip_scaling_list(&mut r, if i < 6 { 16 } else { 64 })?; }
            }
        }
    }
    s.log2_max_frame_num = ue(&mut r)? + 4;
    s.poc_type = ue(&mut r)?;
    if s.poc_type == 0 {
        s.log2_max_poc_lsb = ue(&mut r)? + 4;
    } else if s.poc_type == 1 {
        s.delta_pic_order_always_zero = r.get(1) == 1;
        se(&mut r)?;
        se(&mut r)?;
        for _ in 0..ue(&mut r)?.min(256) { se(&mut r)?; }
    }
    ue(&mut r)?; // max_num_ref_frames
    r.skip(1);
    s.width_mbs = ue(&mut r)? as usize + 1;
    let map_units = ue(&mut r)? as usize + 1;
    s.frame_mbs_only = r.get(1) == 1;
    if !s.frame_mbs_only { r.skip(1); }
    s.height_mbs = map_units * if s.frame_mbs_only { 1 } else { 2 };
    s.direct_8x8_inference = r.get(1) == 1;
    if r.pos() > b.len() * 8 || s.log2_max_frame_num > 16 || s.log2_max_poc_lsb > 16 || s.width_mbs > 1024 || s.height_mbs > 1024 { return None; }
    s.why = if !plain { "high bit depth, separate planes or lossless" } else if !s.frame_mbs_only { "interlaced" } else if s.chroma_format > 1 { "4:2:2 or 4:4:4" }
        else if s.id >= 32 || s.width_mbs * s.height_mbs > MAX_MBS { "out of range" } else { "" };
    s.supported = s.why.is_empty();
    Some(s)
}

/// Is there more data in the RBSP `b` at bit `pos` than its trailing bits?
fn more_rbsp_data(b: &[u8], pos: usize) -> bool {
    let Some(last) = b.iter().rposition(|&x| x != 0) else { return false };
    pos < last * 8 + 7 - b[last].trailing_zeros() as usize
}

/// Parse a picture parameter set (the NAL payload after its header byte).
pub fn parse_pps(nal: &[u8]) -> Option<Pps> {
    let b = unescape(nal);
    let mut r = BitReader::new(&b, 0);
    let mut p = Pps { id: ue(&mut r)?, sps: ue(&mut r)?, ..Default::default() };
    p.cabac = r.get(1) == 1;
    p.pic_order_present = r.get(1) == 1;
    let slice_groups = ue(&mut r)? + 1;
    if slice_groups > 1 {
        // flexible macroblock ordering: unsupported
        p.fmo = true;
        return Some(p);
    }
    p.num_ref_idx = [ue(&mut r)? + 1, ue(&mut r)? + 1];
    p.weighted_pred = r.get(1) == 1;
    p.weighted_bipred = r.get(2);
    p.pic_init_qp = 26 + se(&mut r)?;
    se(&mut r)?; // pic_init_qs
    se(&mut r)?; // chroma_qp_index_offset
    p.deblocking_control = r.get(1) == 1;
    r.skip(1); // constrained_intra_pred
    p.redundant_pic_cnt = r.get(1) == 1;
    if more_rbsp_data(&b, r.pos()) { p.transform_8x8 = r.get(1) == 1; }
    if r.pos() > b.len() * 8 || p.num_ref_idx[0] > 32 || p.num_ref_idx[1] > 32 || p.id >= 256 { return None; }
    p.supported = true;
    Some(p)
}

// ------------------------------------------------------------------ slice header
pub const SLICE_P: u8 = 0;
pub const SLICE_B: u8 = 1;
pub const SLICE_I: u8 = 2;

/// What the parser needs of a slice header, and where the slice data starts.
#[derive(Clone, Debug, Default)]
pub struct SliceHeader {
    pub first_mb: usize,
    pub slice_type: u8,
    pub pps: u32,
    pub num_ref_idx: [u32; 2],
    pub cabac_init_idc: u32,
    pub qp: i32,
    /// Bit offset of the slice data in the RBSP (a CABAC slice's: after the alignment bits).
    pub data: usize,
}

/// Parse the header of a slice NAL unit (`nal_type` 1 or 5, `ref_idc` from its header byte) whose
/// RBSP is `b` (or starts with `b`: the header alone is enough). None for what [`walk`] does not
/// handle.
pub fn parse_slice_header(b: &[u8], nal_type: u8, ref_idc: u8, sps_of: &dyn Fn(u32) -> Option<(Sps, Pps)>) -> Option<(SliceHeader, Sps, Pps)> {
    let mut r = BitReader::new(b, 0);
    let mut h = SliceHeader { first_mb: ue(&mut r)? as usize, ..Default::default() };
    let st = ue(&mut r)?;
    if st > 9 || st % 5 > 2 { return None; } // SP and SI slices
    h.slice_type = (st % 5) as u8;
    h.pps = ue(&mut r)?;
    let (sps, pps) = sps_of(h.pps)?;
    if !sps.supported || !pps.supported { return None; }
    r.skip(sps.log2_max_frame_num);
    let idr = nal_type == 5;
    if idr { ue(&mut r)?; }
    if sps.poc_type == 0 {
        r.skip(sps.log2_max_poc_lsb);
        if pps.pic_order_present { se(&mut r)?; }
    } else if sps.poc_type == 1 && !sps.delta_pic_order_always_zero {
        se(&mut r)?;
        if pps.pic_order_present { se(&mut r)?; }
    }
    if pps.redundant_pic_cnt { ue(&mut r)?; }
    if h.slice_type == SLICE_B { r.skip(1); }
    h.num_ref_idx = pps.num_ref_idx;
    if h.slice_type != SLICE_I {
        if r.get(1) == 1 {
            h.num_ref_idx[0] = ue(&mut r)? + 1;
            if h.slice_type == SLICE_B { h.num_ref_idx[1] = ue(&mut r)? + 1; }
        }
        if h.num_ref_idx[0] > 32 || h.num_ref_idx[1] > 32 { return None; }
        // reference picture list modification
        for _ in 0..if h.slice_type == SLICE_B { 2 } else { 1 } {
            if r.get(1) == 1 {
                loop {
                    let idc = ue(&mut r)?;
                    if idc == 3 { break; }
                    if idc > 3 { return None; }
                    ue(&mut r)?;
                    if r.pos() > b.len() * 8 { return None; }
                }
            }
        }
        if (pps.weighted_pred && h.slice_type == SLICE_P) || (pps.weighted_bipred == 1 && h.slice_type == SLICE_B) {
            ue(&mut r)?;
            if sps.chroma_format != 0 { ue(&mut r)?; }
            for list in 0..if h.slice_type == SLICE_B { 2 } else { 1 } {
                for _ in 0..h.num_ref_idx[list] {
                    if r.get(1) == 1 { se(&mut r)?; se(&mut r)?; }
                    if sps.chroma_format != 0 && r.get(1) == 1 {
                        for _ in 0..4 { se(&mut r)?; }
                    }
                }
            }
        }
    }
    if ref_idc != 0 {
        if idr {
            r.skip(2);
        } else if r.get(1) == 1 {
            loop {
                let op = ue(&mut r)?;
                if op == 0 { break; }
                if op > 6 { return None; }
                if matches!(op, 1 | 3) { ue(&mut r)?; }
                if op == 2 { ue(&mut r)?; }
                if matches!(op, 3 | 6) { ue(&mut r)?; }
                if op == 4 { ue(&mut r)?; }
                if r.pos() > b.len() * 8 { return None; }
            }
        }
    }
    if pps.cabac && h.slice_type != SLICE_I {
        h.cabac_init_idc = ue(&mut r)?;
        if h.cabac_init_idc > 2 { return None; }
    }
    h.qp = pps.pic_init_qp + se(&mut r)?;
    if !(-6 * sps.depth as i32..=51).contains(&h.qp) { return None; }
    if pps.deblocking_control && ue(&mut r)? != 1 {
        se(&mut r)?;
        se(&mut r)?;
    }
    // cabac_alignment_one_bit (CAVLC slice data follows the header bit by bit)
    while pps.cabac && r.pos() % 8 != 0 {
        if r.get(1) != 1 { return None; }
    }
    if r.pos() > b.len() * 8 || h.first_mb >= sps.width_mbs * sps.height_mbs { return None; }
    h.data = r.pos();
    Some((h, sps, pps))
}

// ------------------------------------------------------------------ the arithmetic coder
/// The adaptive state of one context: probability state index (0..=63) and more probable symbol.
#[derive(Clone, Copy, Default)]
struct Ctx {
    state: u8,
    mps: u8,
}

fn init_contexts(slice_type: u8, cabac_init_idc: u32, qp: i32) -> [Ctx; CONTEXTS] {
    let set = if slice_type == SLICE_I { 0 } else { 1 + cabac_init_idc as usize };
    let mut c = [Ctx::default(); CONTEXTS];
    for (i, x) in c.iter_mut().enumerate() {
        let (m, n) = INIT[i][set];
        let pre = (((m as i32 * qp.clamp(0, 51)) >> 4) + n as i32).clamp(1, 126);
        *x = if pre <= 63 { Ctx { state: (63 - pre) as u8, mps: 0 } } else { Ctx { state: (pre - 64) as u8, mps: 1 } };
    }
    c
}

/// The CABAC decoding engine (9.3.3.2) over an RBSP.
pub struct CabacDecoder<'a> {
    r: BitReader<'a>,
    range: u32,
    offset: u32,
    ctx: [Ctx; CONTEXTS],
}

impl<'a> CabacDecoder<'a> {
    fn new(b: &'a [u8], pos: usize, ctx: [Ctx; CONTEXTS]) -> Self {
        let mut r = BitReader::new(b, pos);
        let offset = r.get(9);
        CabacDecoder { r, range: 510, offset, ctx }
    }

    #[inline]
    fn decision(&mut self, i: usize) -> u32 {
        let c = &mut self.ctx[i];
        let lps = RANGE_LPS[c.state as usize][((self.range >> 6) & 3) as usize] as u32;
        self.range -= lps;
        let bin;
        if self.offset >= self.range {
            bin = (c.mps ^ 1) as u32;
            self.offset -= self.range;
            self.range = lps;
            if c.state == 0 { c.mps ^= 1; }
            c.state = NEXT_LPS[c.state as usize];
        } else {
            bin = c.mps as u32;
            c.state = (c.state + 1).min(62);
        }
        while self.range < 256 {
            self.range <<= 1;
            self.offset = self.offset << 1 | self.r.get(1);
        }
        bin
    }

    #[inline]
    fn bypass(&mut self) -> u32 {
        self.offset = self.offset << 1 | self.r.get(1);
        if self.offset >= self.range {
            self.offset -= self.range;
            1
        } else {
            0
        }
    }

    #[inline]
    fn terminate(&mut self) -> u32 {
        self.range -= 2;
        if self.offset >= self.range { return 1; }
        while self.range < 256 {
            self.range <<= 1;
            self.offset = self.offset << 1 | self.r.get(1);
        }
        0
    }

    /// The bit position in the RBSP after a terminating bin of 1: the engine has read exactly the
    /// bits the encoder's flush wrote, the last of which (always 1) is the RBSP stop bit.
    fn end(&self) -> usize { self.r.pos() }
}

/// The CABAC encoding engine (9.3.4.2), writing bits.
pub struct CabacEncoder {
    low: u32,
    range: u32,
    outstanding: u32,
    first: bool,
    ctx: [Ctx; CONTEXTS],
    out: BitWriter,
}

impl CabacEncoder {
    fn new(ctx: [Ctx; CONTEXTS], out: BitWriter) -> Self { CabacEncoder { low: 0, range: 510, outstanding: 0, first: true, ctx, out } }

    #[inline]
    fn put(&mut self, b: u32) {
        if self.first { self.first = false; } else { self.out.put(b, 1); }
        while self.outstanding > 0 {
            self.out.put(1 - b, 1);
            self.outstanding -= 1;
        }
    }

    #[inline]
    fn renorm(&mut self) {
        while self.range < 256 {
            if self.low < 256 {
                self.put(0);
            } else if self.low >= 512 {
                self.low -= 512;
                self.put(1);
            } else {
                self.low -= 256;
                self.outstanding += 1;
            }
            self.range <<= 1;
            self.low <<= 1;
        }
    }

    #[inline]
    fn decision(&mut self, i: usize, bin: u32) {
        let c = &mut self.ctx[i];
        let lps = RANGE_LPS[c.state as usize][((self.range >> 6) & 3) as usize] as u32;
        self.range -= lps;
        if bin != c.mps as u32 {
            self.low += self.range;
            self.range = lps;
            if c.state == 0 { c.mps ^= 1; }
            c.state = NEXT_LPS[c.state as usize];
        } else {
            c.state = (c.state + 1).min(62);
        }
        self.renorm();
    }

    #[inline]
    fn bypass(&mut self, bin: u32) {
        self.low <<= 1;
        if bin != 0 { self.low += self.range; }
        if self.low >= 1024 {
            self.put(1);
            self.low -= 1024;
        } else if self.low < 512 {
            self.put(0);
        } else {
            self.low -= 512;
            self.outstanding += 1;
        }
    }

    /// Code a terminating bin; a 1 flushes the engine (its last bit, always 1, included).
    fn terminate(&mut self, bin: u32) {
        self.range -= 2;
        if bin == 0 {
            self.renorm();
            return;
        }
        self.low += self.range;
        self.range = 2;
        self.renorm();
        self.put((self.low >> 9) & 1);
        self.out.put(((self.low >> 7) & 3) | 1, 2);
    }
}

// ------------------------------------------------------------------ the bins of a slice
/// Syntax elements, as [`Info::se`].
pub mod se {
    pub const SKIP: u8 = 0;
    pub const MB_TYPE: u8 = 1;
    pub const SUB_TYPE: u8 = 2;
    pub const REF: u8 = 3;
    /// Motion vector difference: its context-coded prefix, bypass suffix, and sign.
    pub const MVD: u8 = 4;
    pub const MVD_SUF: u8 = 5;
    pub const MVD_SIGN: u8 = 6;
    pub const QP: u8 = 7;
    pub const IPRED_FLAG: u8 = 8;
    pub const IPRED_REM: u8 = 9;
    pub const CHROMA_PRED: u8 = 10;
    pub const CBP_LUMA: u8 = 11;
    pub const CBP_CHROMA: u8 = 12;
    pub const T8X8: u8 = 13;
    pub const CBF: u8 = 14;
    pub const SIG: u8 = 15;
    pub const LAST: u8 = 16;
    /// Coefficient level: its context-coded prefix, bypass suffix, and sign.
    pub const LEVEL: u8 = 17;
    pub const LEVEL_SUF: u8 = 18;
    pub const SIGN: u8 = 19;
    pub const END: u8 = 20;
    pub const COUNT: usize = 21;
    pub const NAMES: [&str; COUNT] = ["mb skip", "mb type", "sub mb type", "ref idx", "mvd", "mvd suffix", "mvd sign", "qp delta", "intra pred flag",
        "intra pred mode", "chroma pred", "cbp luma", "cbp chroma", "transform 8x8", "coded block flag", "significance", "last", "level", "level suffix",
        "sign", "end of slice"];
}

/// What [`walk`] tells about the bin it asks for, beyond the standard's context: the syntax
/// element and values from its surroundings a model can condition on. Their meaning depends on
/// the element (see where each is set).
#[derive(Clone, Copy, Default)]
pub struct Info {
    /// Macroblock address.
    pub addr: u32,
    pub se: u8,
    pub a: u16,
    pub b: u16,
    pub c: u16,
    pub d: u16,
    /// Signed extras. For a motion vector difference: the values that would give the partition
    /// the vector of its neighbour A, B or C, or the one at its place in the last P picture
    /// ([`NO_MV`]: none).
    pub x: [i16; 4],
}

/// No motion vector.
pub const NO_MV: i16 = i16::MIN;

/// Where [`walk`] gets its bins from, and what it tells about each: the context (or bypass /
/// terminate) the standard codes it with, and its [`Info`].
pub trait Bins {
    fn decision(&mut self, ctx: usize, info: &Info) -> u32;
    fn bypass(&mut self, info: &Info) -> u32;
    fn terminate(&mut self, info: &Info) -> u32;
    /// `n` bytes of PCM samples follow (after a terminating bin of 1).
    fn pcm(&mut self, n: usize) -> Result<(), String>;
    /// Does it want each macroblock's syntax elements ([`Bins::macroblock`])?
    fn collect(&self) -> bool { false }
    /// The macroblock at `addr` has been decoded (called when [`Bins::collect`] says so).
    fn macroblock(&mut self, _addr: usize, _mb: &mut MbSyn) -> Result<(), String> { Ok(()) }
}

/// An intra prediction mode taken as predicted (in [`MbSyn::ipred`]).
pub const PREDICTED: u8 = 8;
/// Per B macroblock type with two partitions (types 4..=21 in pairs: 16x8, then 8x16): the
/// lists each partition is predicted from (bit 0: list 0, bit 1: list 1).
const B_PRED: [[u8; 2]; 9] = [[1, 1], [2, 2], [1, 2], [2, 1], [1, 3], [2, 3], [3, 1], [3, 2], [3, 3]];
/// Per B sub-macroblock type: the lists used (0: direct) and the partition shape (0 8x8, 1 8x4,
/// 2 4x8, 3 4x4).
const B_SUB: [(u8, u8); 13] = [(0, 0), (1, 0), (2, 0), (3, 0), (1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (1, 3), (2, 3), (3, 3)];

/// A macroblock's syntax elements, as [`walk`] decodes them from bins: what the other entropy
/// coding (CAVLC) of the same macroblock is written from, and read into.
#[derive(Clone, Default)]
pub struct MbSyn {
    pub skip: bool,
    /// mb_type in its slice type's numbering (an intra type counts from 5 in a P slice and from
    /// 23 in a B slice).
    pub mb_type: u8,
    /// A P_8x8ref0 macroblock (CAVLC only): type 3 of a P slice whose reference indexes are all
    /// zero and not written.
    pub ref0: bool,
    pub t8x8: bool,
    /// Per intra 4x4 (or 8x8) block in coding order: the remaining mode, or [`PREDICTED`].
    pub ipred: [u8; 16],
    pub chroma_pred: u8,
    pub sub: [u8; 4],
    /// Reference indexes and motion vector differences in the order they are coded.
    pub refs: [u8; 8],
    pub nref: u8,
    pub mvd: [[i32; 2]; 32],
    pub nmvd: u8,
    /// Coded block pattern: luma bits 0..4, chroma in bits 4..6.
    pub cbp: u8,
    pub dqp: i32,
    /// Coefficients in scan order: the luma DC block of an intra 16x16 macroblock; the 4x4 luma
    /// blocks (raster; an intra 16x16 block's from index 1; an 8x8 block's dealt out to its
    /// four in turn); chroma DC per plane; chroma AC per plane and block (from index 1).
    pub dc: [i32; 16],
    pub luma: [[i32; 16]; 16],
    pub cdc: [[i32; 4]; 2],
    pub cac: [[[i32; 16]; 4]; 2],
    /// The slice ends with this macroblock.
    pub end: bool,
}

/// Macroblock classes (for models): skipped; intra 4x4 / 8x8 / 16x16 / PCM; P 16x16 / 16x8 / 8x16 /
/// 8x8; B direct / 16x16 / two partitions / 8x8.
pub mod class {
    pub const SKIP: u8 = 0;
    pub const I4: u8 = 1;
    pub const I8: u8 = 2;
    pub const I16: u8 = 3;
    pub const PCM: u8 = 4;
    pub const P16: u8 = 5;
    pub const P16X8: u8 = 6;
    pub const P8X16: u8 = 7;
    pub const P8: u8 = 8;
    pub const B_DIRECT: u8 = 9;
    pub const B16: u8 = 10;
    pub const B2: u8 = 11;
    pub const B8: u8 = 12;
}

/// What the contexts of later macroblocks, and the hints to models, need of a decoded macroblock.
#[derive(Clone, Copy, Default)]
struct Mb {
    skip: bool,
    intra: bool,
    /// Intra 16x16 or PCM (for the I-slice mb_type context).
    i16_or_pcm: bool,
    pcm: bool,
    /// Intra 4x4 or 8x8: has prediction modes of its own.
    nxn: bool,
    /// B_Skip or B_Direct_16x16.
    direct: bool,
    t8x8: bool,
    class: u8,
    chroma_pred: u8,
    /// Coded block pattern: luma bits 0..4, chroma in bits 4..6.
    cbp: u8,
    /// coded_block_flag of the DC blocks: bit 0 luma (intra 16x16), bit 1 Cb, bit 2 Cr.
    cbf_dc: u8,
    /// coded_block_flag per 4x4 block: bits 0..16 luma (raster), 16..20 Cb, 20..24 Cr.
    cbf: u32,
    /// Coefficients per 4x4 block (same order), capped at 16.
    nz: [u8; 24],
    /// Intra prediction mode per 4x4 luma block (raster), when `nxn`.
    ipred: [u8; 16],
    /// The quantiser in force after the macroblock, and the change it coded.
    qp: u8,
    dqp: i8,
    /// Per list, per 8x8 block: reference index above zero; per 8x8 block: predicted direct.
    ref_pos: [u8; 2],
    direct8: u8,
    /// Per list, per 8x8 block: the reference index (0 when none is coded).
    refs: [[u8; 4]; 2],
    /// Per list, per 4x4 block (raster), per component: |mvd| capped at 70, and the signs of both
    /// components (0 zero, 1 positive, 2 negative; x in bits 0..2, y in bits 2..4).
    mvd: [[[u8; 2]; 16]; 2],
    mvs: [[u8; 16]; 2],
    /// Sum of the motion vector difference magnitudes coded in the macroblock.
    mvsum: u16,
    /// P slices: predicted from list 0 (a reference index per 8x8 block in `refs`), and its
    /// motion vector per 4x4 block. Only models use these (hints), so they need not be exact.
    inter: bool,
    mv: [[i16; 2]; 16],
}

/// Raster position (x + 4y) of the 4x4 luma blocks in coding order, and the inverse.
const BLK: [usize; 16] = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];
const ZIDX: [usize; 16] = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];

/// Working memory of [`walk`], reusable from slice to slice. It also remembers the motion of the
/// last P picture (hints for models), so models that code a stream keep one for all its slices.
#[derive(Default)]
pub struct Scratch {
    mbs: Vec<Mb>,
    /// Per 4x4 block of the last P picture and of the one being decoded: its motion vector, each
    /// component stored as `v ^ 0x8000` (so zeroed memory means none).
    prev_mv: Vec<[u16; 2]>,
    cur_mv: Vec<[u16; 2]>,
    dims: (usize, usize),
    /// CAVLC: coefficients per 4x4 block of each macroblock.
    tc: Vec<[u8; 24]>,
    syn: Box<MbSyn>,
}

struct Walker<'a, B: Bins> {
    bins: &'a mut B,
    h: &'a SliceHeader,
    sps: &'a Sps,
    pps: &'a Pps,
    /// The macroblocks of the picture; only those of this slice already decoded are read.
    mbs: &'a mut [Mb],
    prev_mv: &'a [[u16; 2]],
    cur_mv: &'a mut [[u16; 2]],
    /// Address of the macroblock being decoded, and its record so far.
    addr: usize,
    cur: Mb,
    last_dqp: bool,
    /// The quantiser in force.
    qp: i32,
    info: Info,
    /// The macroblock's syntax elements, complete when the source collects them.
    collect: bool,
    syn: &'a mut MbSyn,
}

impl<B: Bins> Walker<'_, B> {
    #[inline]
    fn d(&mut self, ctx: usize) -> u32 { self.bins.decision(ctx, &self.info) }
    #[inline]
    fn by(&mut self) -> u32 { self.bins.bypass(&self.info) }
    #[inline]
    fn set(&mut self, se: u8, a: usize, b: usize, c: usize, d: usize) {
        self.info.se = se;
        self.info.a = a as u16;
        self.info.b = b as u16;
        self.info.c = c as u16;
        self.info.d = d as u16;
    }

    /// The left / upper neighbour macroblock, if it is in this slice.
    fn left(&self) -> Option<&Mb> {
        (self.addr % self.sps.width_mbs != 0 && self.addr > self.h.first_mb).then(|| &self.mbs[self.addr - 1])
    }
    fn top(&self) -> Option<&Mb> {
        (self.addr >= self.h.first_mb + self.sps.width_mbs).then(|| &self.mbs[self.addr - self.sps.width_mbs])
    }

    /// The macroblock holding the 4x4 luma block left of / above block (x, y), and that block's
    /// raster position in it; None outside the slice.
    fn blk_a(&self, x: usize, y: usize) -> Option<(&Mb, usize)> {
        if x > 0 { Some((&self.cur, x - 1 + 4 * y)) } else { self.left().map(|m| (m, 3 + 4 * y)) }
    }
    fn blk_b(&self, x: usize, y: usize) -> Option<(&Mb, usize)> {
        if y > 0 { Some((&self.cur, x + 4 * (y - 1))) } else { self.top().map(|m| (m, x + 12)) }
    }

    fn mb_type_intra(&mut self, base: usize, i_slice: bool) -> u32 {
        // hints: the bin's index in the intra type
        self.set(se::MB_TYPE, 16, 0, 0, 0);
        let mut state = base;
        if i_slice {
            let inc = self.left().is_some_and(|m| m.i16_or_pcm) as usize + self.top().is_some_and(|m| m.i16_or_pcm) as usize;
            if self.d(base + inc) == 0 { return 0; }
            state += 2;
        } else if self.d(base) == 0 {
            return 0;
        }
        self.info.a = 17;
        if self.bins.terminate(&self.info) == 1 { return 25; }
        let k = i_slice as usize;
        self.info.a = 18;
        let mut t = 1 + 12 * self.d(state + 1);
        self.info.a = 19;
        if self.d(state + 2) == 1 {
            self.info.a = 20;
            t += 4 + 4 * self.d(state + 2 + k);
        }
        self.info.a = 21;
        t += 2 * self.d(state + 3 + k);
        self.info.a = 22;
        t += self.d(state + 3 + 2 * k);
        t
    }

    /// Reference index of a partition whose top-left 8x8 block is `b8`, for `list`.
    fn ref_idx(&mut self, list: usize, b8: usize) -> Result<u32, String> {
        let (x, y) = (b8 % 2, b8 / 2);
        let pos = |m: &Mb, k: usize| (m.ref_pos[list] >> k & 1 == 1 && m.direct8 >> k & 1 == 0) as usize;
        let a = if x > 0 { pos(&self.cur, b8 - 1) } else { self.left().map_or(0, |m| pos(m, b8 + 1)) };
        let b = if y > 0 { pos(&self.cur, b8 - 2) } else { self.top().map_or(0, |m| pos(m, b8 + 2)) };
        // hints: the neighbours' reference indexes themselves
        let ra = if x > 0 { self.cur.refs[list][b8 - 1] } else { self.left().map_or(0, |m| m.refs[list][b8 + 1]) };
        let rb = if y > 0 { self.cur.refs[list][b8 - 2] } else { self.top().map_or(0, |m| m.refs[list][b8 + 2]) };
        self.set(se::REF, list << 2 | b8, 0, (ra.min(15) as usize) << 4 | rb.min(15) as usize, self.cur.class as usize);
        let mut ctx = a + 2 * b;
        let mut v = 0u32;
        while self.d(54 + ctx) == 1 {
            v += 1;
            self.info.b = v.min(15) as u16;
            ctx = (ctx >> 2) + 4;
            if v >= 32 { return Err("h264: reference index out of range".into()); }
        }
        if let Some(r) = self.syn.refs.get_mut(self.syn.nref as usize) {
            *r = v as u8;
            self.syn.nref += 1;
        }
        Ok(v)
    }

    fn fill_refs(&mut self, list: usize, mask: u8, v: u32) {
        for k in 0..4 {
            if mask >> k & 1 == 1 { self.cur.refs[list][k] = v.min(255) as u8; }
        }
        if v > 0 { self.cur.ref_pos[list] |= mask; }
    }

    /// The list-0 motion at the 4x4 block (x, y) of the macroblock, or across its edges (-1 and 4):
    /// None when that block's macroblock is outside the slice or not decoded yet; else its
    /// reference index (-1: none, an intra macroblock) and vector.
    fn motion_at(&self, x: isize, y: isize) -> Option<(i32, [i16; 2])> {
        let w = self.sps.width_mbs;
        let (mb, bx, by): (&Mb, usize, usize) = if y < 0 {
            if x < 0 {
                if self.addr % w == 0 || self.addr < self.h.first_mb + w + 1 { return None; }
                (&self.mbs[self.addr - w - 1], 3, 3)
            } else if x > 3 {
                if self.addr % w == w - 1 || self.addr + 1 < self.h.first_mb + w { return None; }
                (&self.mbs[self.addr + 1 - w], (x - 4) as usize, 3)
            } else {
                (self.top()?, x as usize, 3)
            }
        } else if x < 0 {
            (self.left()?, 3, y as usize)
        } else if x > 3 {
            return None;
        } else {
            (&self.cur, x as usize, y as usize)
        };
        if !mb.inter { return Some((-1, [0, 0])); }
        Some((mb.refs[0][bx / 2 + 2 * (by / 2)] as i32, mb.mv[bx + 4 * by]))
    }

    /// The predicted list-0 motion vector of the partition `w` blocks wide at (x, y) with reference
    /// index `r` (8.4.1.3) — `shape`: 1 the upper of two 16x8 partitions, 2 the lower, 3 the left
    /// of two 8x16, 4 the right, 0 any other — and the vectors of its neighbours A, B and C where
    /// they use the same reference.
    fn predict(&self, x: usize, y: usize, w: usize, r: i32, shape: u8) -> ([i16; 2], [[i16; 2]; 3]) {
        let (xi, yi) = (x as isize, y as isize);
        let a = self.motion_at(xi - 1, yi);
        let mut b = self.motion_at(xi, yi - 1);
        // C is above right, unless it comes later in decoding order: then D, above left
        let cx = xi + w as isize;
        let later = cx <= 3 && y >= 1 && ZIDX[cx as usize + 4 * (y - 1)] > ZIDX[x + 4 * y];
        let mut c = if later { None } else { self.motion_at(cx, yi - 1) };
        if c.is_none() { c = self.motion_at(xi - 1, yi - 1); }
        let cands = [a, b, c].map(|n| match n { Some((rr, mv)) if rr == r => mv, _ => [NO_MV; 2] });
        if let Some((rr, mv)) = match shape { 1 => b, 2 | 3 => a, 4 => c, _ => None } {
            if rr == r { return (mv, cands); }
        }
        if b.is_none() && c.is_none() && a.is_some() { (b, c) = (a, a); }
        let n = [a, b, c].map(|n| n.unwrap_or((-1, [0, 0])));
        let mut same = n.iter().filter(|m| m.0 == r);
        let mv = match (same.next(), same.next()) {
            (Some(m), None) => m.1,
            _ => {
                let med = |k: usize| n[0].1[k].max(n[1].1[k]).min(n[0].1[k].min(n[1].1[k]).max(n[2].1[k]));
                [med(0), med(1)]
            }
        };
        (mv, cands)
    }

    /// The motion of a skipped macroblock of a P slice.
    fn skip_motion(&mut self) {
        let still = |n: Option<(i32, [i16; 2])>| n.is_none_or(|(r, mv)| r == 0 && mv == [0, 0]);
        let mv = if still(self.motion_at(-1, 0)) || still(self.motion_at(0, -1)) { [0, 0] } else { self.predict(0, 0, 4, 0, 0).0 };
        self.cur.inter = true;
        self.cur.mv = [mv; 16];
    }

    /// One motion vector difference component (`comp` 0 x, 1 y) of the partition of size class
    /// `size` whose top-left 4x4 block is (x, y): its magnitude and whether it is negative.
    fn mvd(&mut self, list: usize, x: usize, y: usize, comp: usize, size: usize) -> Result<(u32, bool), String> {
        let side = |n: Option<(&Mb, usize)>| n.map_or((0u32, 0usize), |(m, k)| (m.mvd[list][k][comp] as u32, (m.mvs[list][k] >> (2 * comp) & 3) as usize));
        let ((a, sa), (b, sb)) = (side(self.blk_a(x, y)), side(self.blk_b(x, y)));
        let sum = a + b;
        let base = if comp == 0 { 40 } else { 47 };
        // hints: component, list, partition size and macroblock class; the neighbours' magnitudes
        // and signs
        self.set(se::MVD, comp | list << 1 | size << 2 | (self.cur.class as usize) << 5, a as usize | sa << 8, b as usize | sb << 8, 0);
        if self.d(base + (sum > 2) as usize + (sum > 32) as usize) == 0 { return Ok((0, false)); }
        let (mut v, mut ctx) = (1u32, base + 3);
        loop {
            self.info.d = v as u16;
            if v >= 9 || self.d(ctx) == 0 { break; }
            if v < 4 { ctx += 1; }
            v += 1;
        }
        if v >= 9 {
            self.info.se = se::MVD_SUF;
            let mut k = 3;
            loop {
                self.info.d = k as u16;
                if self.by() == 0 { break; }
                v += 1 << k;
                k += 1;
                if k > 24 { return Err("h264: motion vector difference out of range".into()); }
            }
            while k > 0 {
                k -= 1;
                self.info.d = 32 + k as u16;
                v += self.by() << k;
            }
        }
        self.info.se = se::MVD_SIGN;
        self.info.d = v.min(255) as u16;
        let neg = self.by();
        Ok((v, neg == 1))
    }

    /// Motion data of a partition: `w` x `hgt` 4x4 blocks at (x, y); `shape`: see [`Self::predict`].
    fn part_mvd(&mut self, list: usize, x: usize, y: usize, w: usize, hgt: usize, shape: u8) -> Result<(), String> {
        let size = match (w, hgt) { (4, 4) => 0, (4, 2) => 1, (2, 4) => 2, (2, 2) => 3, (2, 1) => 4, (1, 2) => 5, _ => 6 };
        // P slices: the predicted vector, and what difference each candidate vector would be
        let known = self.h.slice_type == SLICE_P;
        let (pred, cands) = if known { self.predict(x, y, w, self.cur.refs[0][x / 2 + 2 * (y / 2)] as i32, shape) } else { ([0, 0], [[NO_MV; 2]; 3]) };
        let before = if known { self.prev_mv[self.addr * 16 + x + 4 * y].map(|v| (v ^ 0x8000) as i16) } else { [NO_MV; 2] };
        let mut got = [(0u32, false); 2];
        for comp in 0..2 {
            let diff = |v: i16| if v == NO_MV { NO_MV } else { v.wrapping_sub(pred[comp]).max(NO_MV + 1) };
            self.info.x = [diff(cands[0][comp]), diff(cands[1][comp]), diff(cands[2][comp]), diff(before[comp])];
            got[comp] = self.mvd(list, x, y, comp, size)?;
        }
        if let Some(d) = self.syn.mvd.get_mut(self.syn.nmvd as usize) {
            *d = got.map(|(v, neg)| if neg { -(v.min(i32::MAX as u32) as i32) } else { v.min(i32::MAX as u32) as i32 });
            self.syn.nmvd += 1;
        }
        let capped = got.map(|(v, _)| v.min(70) as u8);
        let signs = got.map(|(v, neg)| if v == 0 { 0 } else { 1 + neg as u8 });
        let mv = [0, 1].map(|k| {
            let d = if got[k].1 { -(got[k].0 as i64) } else { got[k].0 as i64 };
            (pred[k] as i64 + d).clamp(NO_MV as i64 + 1, i16::MAX as i64) as i16
        });
        for yy in y..y + hgt {
            for xx in x..x + w {
                self.cur.mvd[list][xx + 4 * yy] = capped;
                self.cur.mvs[list][xx + 4 * yy] = signs[0] | signs[1] << 2;
                self.cur.mv[xx + 4 * yy] = mv;
            }
        }
        self.cur.mvsum += capped[0] as u16 + capped[1] as u16;
        Ok(())
    }

    /// coded_block_flag context increment from the flags of blocks A and B (None: outside the
    /// slice, which counts as coded for an intra macroblock).
    fn cbf_inc(&self, a: Option<bool>, b: Option<bool>) -> usize {
        let un = self.cur.intra;
        a.unwrap_or(un) as usize + 2 * b.unwrap_or(un) as usize
    }

    /// One residual block of `cat` (0 intra16 DC, 1 intra16 AC, 2 luma 4x4, 3 chroma DC, 4 chroma
    /// AC, 5 luma 8x8) with `max` coefficients; `cbf_ctx`: its coded_block_flag context, None
    /// when the flag is not coded. Hints: `blk` identifies the block, `nb` is the number of
    /// coefficients of the blocks left of and above it. Returns the number of coefficients.
    fn residual(&mut self, cat: usize, max: usize, cbf_ctx: Option<usize>, blk: usize, nb: usize) -> Result<usize, String> {
        let cls = self.cur.class as usize;
        if let Some(c) = cbf_ctx {
            self.set(se::CBF, cat << 8 | blk, nb, cls, self.cur.cbp as usize);
            if self.d(c) == 0 { return Ok(0); }
        }
        const SIG: [usize; 6] = [105, 105 + 15, 105 + 29, 105 + 44, 105 + 47, 402];
        const LAST: [usize; 6] = [166, 166 + 15, 166 + 29, 166 + 44, 166 + 47, 417];
        const ABS: [usize; 6] = [227, 227 + 10, 227 + 20, 227 + 30, 227 + 39, 426];
        // the significance map: which positions hold a coefficient
        let mut pos = [0u8; 64];
        let mut coefs = 0usize;
        let mut i = 0;
        while i < max - 1 {
            let (s, l) = if cat == 5 { (SIG_8X8[i] as usize, LAST_8X8[i] as usize) } else if cat == 3 { (i.min(2), i.min(2)) } else { (i, i) };
            self.set(se::SIG, cat << 8 | i, coefs, nb, cls | blk << 4);
            if self.d(SIG[cat] + s) == 1 {
                pos[coefs] = i as u8;
                coefs += 1;
                self.info.se = se::LAST;
                self.info.b = coefs as u16;
                if self.d(LAST[cat] + l) == 1 { break; }
            }
            i += 1;
        }
        if i == max - 1 {
            // the last position is significant when none before ended the map
            pos[coefs] = i as u8;
            coefs += 1;
        }
        // the levels, last coefficient first
        let mut node = 0usize;
        let mut big = 0usize; // levels above one so far, and the sum of the levels
        for k in 0..coefs {
            const ONE: [usize; 8] = [1, 2, 3, 4, 0, 0, 0, 0];
            const GT1: [usize; 8] = [5, 5, 5, 5, 6, 7, 8, 9];
            const NEXT: [[usize; 8]; 2] = [[1, 2, 3, 3, 4, 5, 6, 7], [4, 4, 4, 4, 5, 6, 7, 7]];
            let p = pos[coefs - 1 - k] as usize;
            self.set(se::LEVEL, cat << 8 | p, k.min(15) | big.min(15) << 4, nb | cls << 8, coefs);
            let mut abs = 1usize;
            if self.d(ABS[cat] + ONE[node]) == 0 {
                node = NEXT[0][node];
            } else {
                let ctx = ABS[cat] + GT1[node].min(if cat == 3 { 8 } else { 9 });
                node = NEXT[1][node];
                big += 1;
                abs = 2;
                loop {
                    self.info.d = (coefs | abs.min(15) << 8) as u16;
                    if abs >= 15 || self.d(ctx) == 0 { break; }
                    abs += 1;
                }
                if abs >= 15 {
                    self.info.se = se::LEVEL_SUF;
                    let mut j = 0;
                    loop {
                        self.info.d = j as u16;
                        if self.by() == 0 { break; }
                        abs += 1 << j;
                        j += 1;
                        if j > 30 { return Err("h264: coefficient out of range".into()); }
                    }
                    for b in (0..j).rev() {
                        self.info.d = 32 + b as u16;
                        abs += (self.by() as usize) << b;
                    }
                }
            }
            // hints: the magnitude just coded
            self.set(se::SIGN, cat << 8 | p, k.min(15) | abs.min(255) << 4, nb | cls << 8, coefs);
            let neg = self.by() == 1;
            if self.collect {
                let v = abs.min(i32::MAX as usize) as i32;
                let v = if neg { -v } else { v };
                match cat {
                    0 => self.syn.dc[p] = v,
                    1 => self.syn.luma[blk][p + 1] = v,
                    2 => self.syn.luma[blk][p] = v,
                    3 => self.syn.cdc[blk][p] = v,
                    4 => self.syn.cac[blk >> 2][blk & 3][p + 1] = v,
                    _ => self.syn.luma[BLK[blk * 4 + (p & 3)]][p >> 2] = v,
                }
            }
        }
        Ok(coefs)
    }

    /// The residual of a macroblock: `i16` intra 16x16; coded block pattern in `self.cur.cbp`.
    fn residuals(&mut self, i16: bool) -> Result<(), String> {
        let cbp = self.cur.cbp;
        let count = |n: Option<(&Mb, usize)>| n.map_or(0, |(m, k)| m.nz[k] as usize);
        if i16 {
            let inc = self.cbf_inc(self.left().map(|m| m.cbf_dc & 1 != 0), self.top().map(|m| m.cbf_dc & 1 != 0));
            if self.residual(0, 16, Some(85 + inc), 0, 0)? > 0 { self.cur.cbf_dc |= 1; }
        }
        for b8 in 0..4 {
            if cbp >> b8 & 1 == 0 { continue; }
            if self.cur.t8x8 {
                // an 8x8 block has no flag of its own: its four 4x4 positions count as coded
                let p0 = BLK[b8 * 4];
                let nb = count(self.blk_a(p0 % 4, p0 / 4)) + count(self.blk_b(p0 % 4, p0 / 4));
                let n = self.residual(5, 64, None, b8, nb)?;
                for k in 0..4 {
                    self.cur.cbf |= 1 << BLK[b8 * 4 + k];
                    self.cur.nz[BLK[b8 * 4 + k]] = n.div_ceil(4).min(16) as u8;
                }
                continue;
            }
            for k in 0..4 {
                let p = BLK[b8 * 4 + k];
                let (x, y) = (p % 4, p / 4);
                let flag = |n: Option<(&Mb, usize)>| n.map(|(m, k)| m.cbf >> k & 1 == 1);
                let inc = self.cbf_inc(flag(self.blk_a(x, y)), flag(self.blk_b(x, y)));
                let nb = count(self.blk_a(x, y)) + count(self.blk_b(x, y));
                let (cat, max) = if i16 { (1, 15) } else { (2, 16) };
                let n = self.residual(cat, max, Some(85 + 4 * cat + inc), p, nb)?;
                if n > 0 {
                    self.cur.cbf |= 1 << p;
                    self.cur.nz[p] = n.min(16) as u8;
                }
            }
        }
        if self.sps.chroma_format == 0 { return Ok(()); }
        if cbp >> 4 != 0 {
            for c in 0..2 {
                let bit = 2 << c;
                let inc = self.cbf_inc(self.left().map(|m| m.cbf_dc & bit != 0), self.top().map(|m| m.cbf_dc & bit != 0));
                if self.residual(3, 4, Some(85 + 12 + inc), c, 0)? > 0 { self.cur.cbf_dc |= bit; }
            }
        }
        if cbp >> 4 == 2 {
            for c in 0..2 {
                for k in 0..4 {
                    let (x, y, base) = (k % 2, k / 2, 16 + 4 * c);
                    let (l, t) = (self.left().copied(), self.top().copied());
                    let a = if x > 0 { Some(self.cur.cbf >> (base + k - 1) & 1 == 1) } else { l.map(|m| m.cbf >> (base + k + 1) & 1 == 1) };
                    let b = if y > 0 { Some(self.cur.cbf >> (base + k - 2) & 1 == 1) } else { t.map(|m| m.cbf >> (base + k + 2) & 1 == 1) };
                    let na = if x > 0 { self.cur.nz[base + k - 1] } else { l.map_or(0, |m| m.nz[base + k + 1]) } as usize;
                    let nbv = if y > 0 { self.cur.nz[base + k - 2] } else { t.map_or(0, |m| m.nz[base + k + 2]) } as usize;
                    let inc = self.cbf_inc(a, b);
                    let n = self.residual(4, 15, Some(85 + 16 + inc), c << 2 | k, na + nbv)?;
                    if n > 0 {
                        self.cur.cbf |= 1 << (base + k);
                        self.cur.nz[base + k] = n.min(16) as u8;
                    }
                }
            }
        }
        Ok(())
    }

    /// The prediction modes of an intra 4x4 (16 blocks) or 8x8 (4 blocks) macroblock.
    fn intra_pred(&mut self, t8: bool) {
        // a neighbour's mode as the prediction uses it: 9 when outside the slice, DC (2) for a
        // macroblock without modes of its own
        let mode = |n: Option<(&Mb, usize)>| n.map_or(9usize, |(m, k)| if m.nxn { m.ipred[k] as usize } else { 2 });
        for k in 0..if t8 { 4 } else { 16 } {
            let p = if t8 { BLK[k * 4] } else { BLK[k] };
            let (x, y) = (p % 4, p / 4);
            let (ma, mb) = (mode(self.blk_a(x, y)), mode(self.blk_b(x, y)));
            let pred = if ma == 9 || mb == 9 { 2 } else { ma.min(mb) };
            self.set(se::IPRED_FLAG, k | (t8 as usize) << 4, ma, mb, pred);
            let m = if self.d(68) == 1 {
                self.syn.ipred[k] = PREDICTED;
                pred
            } else {
                self.info.se = se::IPRED_REM;
                let mut rem = 0usize;
                for bit in 0..3 {
                    self.info.a = (k | (t8 as usize) << 4 | bit << 5 | rem << 7) as u16;
                    rem |= (self.d(69) as usize) << bit;
                }
                self.syn.ipred[k] = rem as u8;
                if rem < pred { rem } else { rem + 1 }
            };
            let span = if t8 { 2 } else { 1 };
            for yy in y..y + span {
                for xx in x..x + span { self.cur.ipred[xx + 4 * yy] = m as u8; }
            }
        }
    }

    fn chroma_pred(&mut self) {
        if self.sps.chroma_format == 0 { return; }
        let cond = |m: Option<&Mb>| m.is_some_and(|m| m.intra && !m.pcm && m.chroma_pred != 0) as usize;
        let hint = |m: Option<&Mb>| m.map_or(0, |m| if m.intra && !m.pcm { 1 + m.chroma_pred as usize } else { 5 });
        let inc = cond(self.left()) + cond(self.top());
        self.set(se::CHROMA_PRED, 0, hint(self.left()), hint(self.top()), self.cur.class as usize);
        let mut v = 0;
        if self.d(64 + inc) == 1 {
            v = 1;
            self.info.a = 1;
            if self.d(64 + 3) == 1 {
                self.info.a = 2;
                v = 2 + self.d(64 + 3) as u8;
            }
        }
        self.cur.chroma_pred = v;
        self.syn.chroma_pred = v;
    }

    fn cbp(&mut self) {
        let luma = |m: Option<&Mb>, bit: usize| -> usize {
            // a block outside the slice, or in a PCM macroblock, counts as coded
            m.map_or(0, |m| (m.cbp >> bit & 1 == 0 && !m.pcm) as usize)
        };
        let (l, t) = (self.left().copied(), self.top().copied());
        let cls = self.cur.class as usize;
        let mut cbp = 0u8;
        for b8 in 0..4usize {
            let a = if b8 % 2 == 1 { (cbp >> (b8 - 1) & 1 == 0) as usize } else { luma(l.as_ref(), b8 + 1) };
            let b = if b8 >= 2 { (cbp >> (b8 - 2) & 1 == 0) as usize } else { luma(t.as_ref(), b8 + 2) };
            self.set(se::CBP_LUMA, b8, cls, cbp as usize, 0);
            cbp |= (self.d(73 + a + 2 * b) as u8) << b8;
        }
        if self.sps.chroma_format != 0 {
            let chroma = |m: Option<&Mb>| m.map_or(0, |m| if m.pcm { 2 } else { m.cbp >> 4 });
            let (a, b) = (chroma(l.as_ref()), chroma(t.as_ref()));
            self.set(se::CBP_CHROMA, 0, cls, cbp as usize, 0);
            if self.d(77 + (a > 0) as usize + 2 * (b > 0) as usize) == 1 {
                self.info.a = 1;
                cbp |= (1 + self.d(77 + 4 + (a == 2) as usize + 2 * (b == 2) as usize) as u8) << 4;
            }
        }
        self.cur.cbp = cbp;
        self.syn.cbp = cbp;
    }

    fn t8x8_flag(&mut self) {
        let inc = self.left().is_some_and(|m| m.t8x8) as usize + self.top().is_some_and(|m| m.t8x8) as usize;
        self.set(se::T8X8, 0, self.cur.class as usize, self.cur.cbp as usize, 0);
        self.cur.t8x8 = self.d(399 + inc) == 1;
        self.syn.t8x8 = self.cur.t8x8;
    }

    fn qp_delta(&mut self) -> Result<(), String> {
        // hints: how the quantiser above differs from the one in force (33 + difference, 0 when
        // there is no macroblock above), the previous change, and the quantiser in force
        let above = self.top().map_or(0, |m| (33 + (m.qp as i32 - self.qp).clamp(-32, 32)) as usize);
        let prev = if self.addr > self.h.first_mb { (64 + self.mbs[self.addr - 1].dqp as i32) as usize } else { 0 };
        self.set(se::QP, 0, above, prev, self.qp as usize | (self.cur.class as usize) << 8);
        let mut v = 0;
        if self.d(60 + self.last_dqp as usize) == 1 {
            v = 1;
            let mut ctx = 2;
            loop {
                self.info.a = v.min(15) as u16;
                if self.d(60 + ctx) == 0 { break; }
                ctx = 3;
                v += 1;
                if v > 104 { return Err("h264: quantiser change out of range".into()); }
            }
        }
        self.last_dqp = v != 0;
        let dqp = if v & 1 == 1 { (v + 1) / 2 } else { -(v / 2) };
        self.cur.dqp = dqp.clamp(-64, 63) as i8;
        self.syn.dqp = dqp;
        self.qp = (self.qp + dqp + 52).rem_euclid(52);
        Ok(())
    }

    /// The macroblock layer of an intra macroblock of type `t` (0 NxN, 1..=24 16x16, 25 PCM).
    fn intra_mb(&mut self, t: u32) -> Result<(), String> {
        self.cur.intra = true;
        if t == 25 {
            self.cur.pcm = true;
            self.cur.class = class::PCM;
            self.cur.i16_or_pcm = true;
            self.cur.cbp = 0x2F;
            self.cur.cbf = 0xFF_FFFF;
            self.cur.cbf_dc = 7;
            self.cur.nz = [16; 24];
            self.last_dqp = false;
            if self.sps.depth > 0 { return Err("h264: PCM samples of more than 8 bits".into()); }
            return self.bins.pcm(if self.sps.chroma_format == 0 { 256 } else { 384 });
        }
        if t == 0 {
            self.cur.nxn = true;
            self.cur.class = class::I4;
            if self.pps.transform_8x8 { self.t8x8_flag(); }
            if self.cur.t8x8 { self.cur.class = class::I8; }
            self.intra_pred(self.cur.t8x8);
            self.chroma_pred();
            self.cbp();
            if self.cur.cbp != 0 {
                self.qp_delta()?;
                self.residuals(false)?;
            } else {
                self.last_dqp = false;
            }
        } else {
            self.cur.i16_or_pcm = true;
            self.cur.class = class::I16;
            self.chroma_pred();
            let k = t - 1;
            self.cur.cbp = if k >= 12 { 15 } else { 0 } | ((k / 4 % 3) as u8) << 4;
            self.qp_delta()?;
            self.residuals(true)?;
        }
        Ok(())
    }

    /// Coded block pattern, transform size, quantiser change and residual of an inter macroblock;
    /// `may_8x8`: its partitions allow the 8x8 transform.
    fn inter_tail(&mut self, may_8x8: bool) -> Result<(), String> {
        self.cbp();
        if self.cur.cbp & 15 != 0 && self.pps.transform_8x8 && may_8x8 { self.t8x8_flag(); }
        if self.cur.cbp != 0 {
            self.qp_delta()?;
            self.residuals(false)?;
        } else {
            self.last_dqp = false;
        }
        Ok(())
    }

    fn p_mb(&mut self) -> Result<(), String> {
        self.set(se::MB_TYPE, 0, 0, 0, 0);
        if self.d(14) == 1 {
            let t = self.mb_type_intra(17, false);
            self.syn.mb_type = 5 + t as u8;
            return self.intra_mb(t);
        }
        self.info.a = 1;
        let t = if self.d(15) == 0 {
            self.info.a = 2;
            3 * self.d(16)
        } else {
            self.info.a = 3;
            2 - self.d(17)
        };
        let multi = self.h.num_ref_idx[0] > 1;
        self.cur.inter = true;
        self.syn.mb_type = t as u8;
        match t {
            0 => {
                self.cur.class = class::P16;
                if multi {
                    let r = self.ref_idx(0, 0)?;
                    self.fill_refs(0, 15, r);
                }
                self.part_mvd(0, 0, 0, 4, 4, 0)?;
            }
            1 | 2 => {
                // two 16x8 (1) or 8x16 (2) partitions: their top-left 8x8 blocks, and the blocks they cover
                self.cur.class = if t == 1 { class::P16X8 } else { class::P8X16 };
                let parts: [(usize, u8); 2] = if t == 1 { [(0, 0b0011), (2, 0b1100)] } else { [(0, 0b0101), (1, 0b1010)] };
                for (b8, mask) in parts {
                    if multi {
                        let r = self.ref_idx(0, b8)?;
                        self.fill_refs(0, mask, r);
                    }
                }
                for (k, (b8, _)) in parts.into_iter().enumerate() {
                    let (x, y) = (2 * (b8 % 2), 2 * (b8 / 2));
                    if t == 1 { self.part_mvd(0, x, y, 4, 2, 1 + k as u8)?; } else { self.part_mvd(0, x, y, 2, 4, 3 + k as u8)?; }
                }
            }
            _ => {
                self.cur.class = class::P8;
                let mut sub = [0u32; 4];
                for (k, s) in sub.iter_mut().enumerate() {
                    self.set(se::SUB_TYPE, k, 0, 0, 0);
                    *s = if self.d(21) == 1 { 0 } else {
                        self.info.b = 1;
                        if self.d(22) == 0 { 1 } else {
                            self.info.b = 2;
                            if self.d(23) == 1 { 2 } else { 3 }
                        }
                    };
                }
                for (k, &s) in sub.iter().enumerate() { self.syn.sub[k] = s as u8; }
                for b8 in 0..4 {
                    if multi {
                        let r = self.ref_idx(0, b8)?;
                        self.fill_refs(0, 1 << b8, r);
                    }
                }
                for (b8, &s) in sub.iter().enumerate() {
                    let (x, y) = (2 * (b8 % 2), 2 * (b8 / 2));
                    match s {
                        0 => self.part_mvd(0, x, y, 2, 2, 0)?,
                        1 => { self.part_mvd(0, x, y, 2, 1, 0)?; self.part_mvd(0, x, y + 1, 2, 1, 0)?; }
                        2 => { self.part_mvd(0, x, y, 1, 2, 0)?; self.part_mvd(0, x + 1, y, 1, 2, 0)?; }
                        _ => for k in 0..4 { self.part_mvd(0, x + k % 2, y + k / 2, 1, 1, 0)?; },
                    }
                }
                return self.inter_tail(sub.iter().all(|&s| s == 0));
            }
        }
        self.inter_tail(true)
    }

    fn b_mb(&mut self) -> Result<(), String> {
        let cond = |m: Option<&Mb>| m.is_some_and(|m| !m.direct) as usize;
        let inc = cond(self.left()) + cond(self.top());
        self.set(se::MB_TYPE, 0, 0, 0, 0);
        let t = if self.d(27 + inc) == 0 {
            0
        } else {
            self.info.a = 1;
            if self.d(27 + 3) == 0 {
                self.info.a = 2;
                1 + self.d(27 + 5)
            } else {
                let mut bits = 0;
                for k in 0..4 {
                    self.info.a = 3 + k as u16;
                    self.info.b = bits as u16;
                    bits = bits << 1 | self.d(if k == 0 { 27 + 4 } else { 27 + 5 });
                }
                match bits {
                    0..=7 => bits + 3,
                    13 => {
                        let t = self.mb_type_intra(32, false);
                        self.syn.mb_type = 23 + t as u8;
                        return self.intra_mb(t);
                    }
                    14 => 11,
                    15 => 22,
                    _ => {
                        self.info.a = 7;
                        self.info.b = bits as u16;
                        (bits << 1 | self.d(27 + 5)) - 4
                    }
                }
            }
        };
        let multi = [self.h.num_ref_idx[0] > 1, self.h.num_ref_idx[1] > 1];
        self.syn.mb_type = t as u8;
        match t {
            0 => {
                self.cur.class = class::B_DIRECT;
                self.cur.direct = true;
                self.cur.direct8 = 15;
                return self.inter_tail(self.sps.direct_8x8_inference);
            }
            1..=3 => {
                // 16x16: list 0, list 1 or both
                self.cur.class = class::B16;
                let uses = [t != 2, t != 1];
                for list in 0..2 {
                    if uses[list] && multi[list] {
                        let r = self.ref_idx(list, 0)?;
                        self.fill_refs(list, 15, r);
                    }
                }
                for list in 0..2 {
                    if uses[list] { self.part_mvd(list, 0, 0, 4, 4, 0)?; }
                }
            }
            4..=21 => {
                // two partitions, each predicted from list 0 (bit 0), list 1 (bit 1) or both
                self.cur.class = class::B2;
                let pred = B_PRED[(t as usize - 4) / 2];
                let wide = t % 2 == 0; // 16x8
                let parts: [(usize, u8); 2] = if wide { [(0, 0b0011), (2, 0b1100)] } else { [(0, 0b0101), (1, 0b1010)] };
                for list in 0..2 {
                    for (k, (b8, mask)) in parts.into_iter().enumerate() {
                        if pred[k] >> list & 1 == 1 && multi[list] {
                            let r = self.ref_idx(list, b8)?;
                            self.fill_refs(list, mask, r);
                        }
                    }
                }
                for list in 0..2 {
                    for (k, (b8, _)) in parts.into_iter().enumerate() {
                        if pred[k] >> list & 1 == 0 { continue; }
                        let (x, y) = (2 * (b8 % 2), 2 * (b8 / 2));
                        if wide { self.part_mvd(list, x, y, 4, 2, 0)?; } else { self.part_mvd(list, x, y, 2, 4, 0)?; }
                    }
                }
            }
            _ => {
                self.cur.class = class::B8;
                let mut sub = [0u32; 4];
                for (k, s) in sub.iter_mut().enumerate() {
                    self.set(se::SUB_TYPE, k, 0, 0, 1);
                    *s = if self.d(36) == 0 {
                        0
                    } else {
                        self.info.b = 1;
                        if self.d(37) == 0 {
                            self.info.b = 2;
                            1 + self.d(39)
                        } else {
                            self.info.b = 3;
                            let mut v = 3;
                            if self.d(38) == 1 {
                                self.info.b = 4;
                                if self.d(39) == 1 {
                                    self.info.b = 5;
                                    11 + self.d(39)
                                } else {
                                    v += 4;
                                    self.info.b = 6;
                                    v += 2 * self.d(39);
                                    self.info.b = 7 + (v as u16 & 2);
                                    v + self.d(39)
                                }
                            } else {
                                self.info.b = 10;
                                v += 2 * self.d(39);
                                self.info.b = 11 + (v as u16 & 2);
                                v + self.d(39)
                            }
                        }
                    };
                }
                const SUB: [(u8, u8); 13] = B_SUB;
                for (k, &s) in sub.iter().enumerate() { self.syn.sub[k] = s as u8; }
                for (b8, &s) in sub.iter().enumerate() {
                    if s == 0 { self.cur.direct8 |= 1 << b8; }
                }
                for list in 0..2 {
                    for (b8, &s) in sub.iter().enumerate() {
                        if SUB[s as usize].0 >> list & 1 == 1 && multi[list] {
                            let r = self.ref_idx(list, b8)?;
                            self.fill_refs(list, 1 << b8, r);
                        }
                    }
                }
                for list in 0..2 {
                    for (b8, &s) in sub.iter().enumerate() {
                        let (lists, shape) = SUB[s as usize];
                        if lists >> list & 1 == 0 { continue; }
                        let (x, y) = (2 * (b8 % 2), 2 * (b8 / 2));
                        match shape {
                            0 => self.part_mvd(list, x, y, 2, 2, 0)?,
                            1 => { self.part_mvd(list, x, y, 2, 1, 0)?; self.part_mvd(list, x, y + 1, 2, 1, 0)?; }
                            2 => { self.part_mvd(list, x, y, 1, 2, 0)?; self.part_mvd(list, x + 1, y, 1, 2, 0)?; }
                            _ => for k in 0..4 { self.part_mvd(list, x + k % 2, y + k / 2, 1, 1, 0)?; },
                        }
                    }
                }
                // the 8x8 transform needs every sub-macroblock whole (a direct one: inferred as 8x8)
                let whole = sub.iter().all(|&s| if s == 0 { self.sps.direct_8x8_inference } else { SUB[s as usize].1 == 0 });
                return self.inter_tail(whole);
            }
        }
        self.inter_tail(true)
    }

    /// The slice data: macroblocks until the end-of-slice bin. Returns how many were decoded.
    fn slice(&mut self) -> Result<usize, String> {
        let total = self.sps.width_mbs * self.sps.height_mbs;
        let mut n = 0;
        loop {
            if self.addr >= total { return Err("h264: slice runs past the picture".into()); }
            self.cur = Mb::default();
            self.info.addr = self.addr as u32;
            if self.collect { *self.syn = MbSyn::default(); } else { (self.syn.nref, self.syn.nmvd) = (0, 0); }
            let mut skipped = false;
            if self.h.slice_type != SLICE_I {
                let cond = |m: Option<&Mb>| m.is_some_and(|m| !m.skip) as usize;
                let inc = cond(self.left()) + cond(self.top());
                self.set(se::SKIP, 0, 0, 0, 0);
                skipped = self.d(if self.h.slice_type == SLICE_P { 11 } else { 24 } + inc) == 1;
            }
            if skipped {
                self.syn.skip = true;
                self.cur.skip = true;
                self.cur.class = class::SKIP;
                self.cur.direct = self.h.slice_type == SLICE_B;
                self.cur.direct8 = if self.cur.direct { 15 } else { 0 };
                if self.h.slice_type == SLICE_P { self.skip_motion(); }
                self.last_dqp = false;
            } else {
                match self.h.slice_type {
                    SLICE_I => {
                        let t = self.mb_type_intra(3, true);
                        self.syn.mb_type = t as u8;
                        self.intra_mb(t)?;
                    }
                    SLICE_P => self.p_mb()?,
                    _ => self.b_mb()?,
                }
            }
            self.cur.qp = self.qp as u8;
            self.mbs[self.addr] = self.cur;
            if self.cur.inter {
                for (k, mv) in self.cur.mv.iter().enumerate() { self.cur_mv[self.addr * 16 + k] = mv.map(|v| v as u16 ^ 0x8000); }
            }
            // hints for the end-of-slice bin: a summary of the macroblock, for models to remember
            let coefs: usize = self.cur.nz.iter().map(|&x| x as usize).sum();
            self.set(se::END, self.cur.class as usize | (self.cur.t8x8 as usize) << 4 | (self.cur.intra as usize) << 5 | (self.cur.mvsum.min(255) as usize) << 8,
                self.cur.cbp as usize | ((self.cur.dqp as i32 + 64) as usize) << 6, self.qp as usize, coefs.min(1023));
            self.addr += 1;
            n += 1;
            let end = self.bins.terminate(&self.info) == 1;
            if self.collect {
                self.syn.end = end;
                self.bins.macroblock(self.addr - 1, self.syn)?;
            }
            if end { return Ok(n); }
        }
    }
}

/// Walk the slice data of a slice with header `h`, taking its bins from `bins`. Returns the
/// number of macroblocks.
pub fn walk<B: Bins>(bins: &mut B, h: &SliceHeader, sps: &Sps, pps: &Pps, scratch: &mut Scratch) -> Result<usize, String> {
    let total = sps.width_mbs * sps.height_mbs;
    if !sps.supported || !pps.supported || total > MAX_MBS || h.first_mb >= total { return Err("h264: unsupported slice".into()); }
    if scratch.mbs.len() < total { scratch.mbs.resize(total, Mb::default()); }
    // the motion memory: of the right size, and at a new P picture this one becomes the last
    if scratch.dims != (sps.width_mbs, sps.height_mbs) {
        scratch.dims = (sps.width_mbs, sps.height_mbs);
        scratch.prev_mv = vec![[0; 2]; total * 16];
        scratch.cur_mv = vec![[0; 2]; total * 16];
    } else if h.slice_type == SLICE_P && h.first_mb == 0 {
        std::mem::swap(&mut scratch.prev_mv, &mut scratch.cur_mv);
        scratch.cur_mv.fill([0; 2]);
    }
    let collect = bins.collect();
    let mut w = Walker {
        bins, h, sps, pps, mbs: &mut scratch.mbs[..total], prev_mv: &scratch.prev_mv, cur_mv: &mut scratch.cur_mv, addr: h.first_mb, cur: Mb::default(),
        last_dqp: false, qp: h.qp, info: Info::default(), collect, syn: &mut *scratch.syn,
    };
    w.slice()
}

// ------------------------------------------------------------------ taking a slice apart
/// The context of a bin coded without one: bypass, and the terminating bins.
pub const BYPASS: u16 = 1000;
pub const TERMINATE: u16 = 1001;

/// Decodes a slice with the arithmetic decoder, packing its bins and showing each to `seen`.
///
/// Packed bins: one bit per bin in coding order, most significant bit first. A PCM macroblock's
/// samples are inline: after its terminating bin the bits are padded to a byte, then come the
/// slice's own padding bits before the samples (one byte) and the samples. Zero bits pad the end
/// to a byte.
struct Recorder<'a> {
    dec: CabacDecoder<'a>,
    rbsp: &'a [u8],
    seen: &'a mut dyn FnMut(u16, u8, &Info),
    out: BitWriter,
    count: usize,
    overrun: bool,
}

impl Bins for Recorder<'_> {
    fn decision(&mut self, ctx: usize, info: &Info) -> u32 {
        let b = self.dec.decision(ctx);
        self.out.put(b, 1);
        self.count += 1;
        (self.seen)(ctx as u16, b as u8, info);
        b
    }
    fn bypass(&mut self, info: &Info) -> u32 {
        let b = self.dec.bypass();
        self.out.put(b, 1);
        self.count += 1;
        (self.seen)(BYPASS, b as u8, info);
        b
    }
    fn terminate(&mut self, info: &Info) -> u32 {
        let b = self.dec.terminate();
        self.out.put(b, 1);
        self.count += 1;
        (self.seen)(TERMINATE, b as u8, info);
        b
    }
    fn pcm(&mut self, n: usize) -> Result<(), String> {
        // after the flush: alignment, then the samples as bytes; then the engine restarts
        let mut pos = self.dec.end();
        let pad = (8 - pos % 8) % 8;
        let mut r = BitReader::new(self.rbsp, pos);
        let padding = r.get(pad as u32) as u8;
        pos += pad;
        let from = pos / 8;
        let Some(bytes) = self.rbsp.get(from..from + n) else {
            self.overrun = true;
            return Err("h264: PCM samples past the slice".into());
        };
        self.out.align();
        self.out.extend(&[padding]);
        self.out.extend(bytes);
        let ctx = self.dec.ctx;
        self.dec = CabacDecoder::new(self.rbsp, (from + n) * 8, ctx);
        Ok(())
    }
}

/// Replays packed bins into the arithmetic encoder.
struct Replayer<'a> {
    enc: CabacEncoder,
    r: BitReader<'a>,
    /// Bits of packed bins available, and whether the walk ran out of them.
    limit: usize,
    short: bool,
}

impl Bins for Replayer<'_> {
    fn decision(&mut self, ctx: usize, _: &Info) -> u32 {
        let b = self.r.get(1);
        self.enc.decision(ctx, b);
        b
    }
    fn bypass(&mut self, _: &Info) -> u32 {
        let b = self.r.get(1);
        self.enc.bypass(b);
        b
    }
    fn terminate(&mut self, _: &Info) -> u32 {
        // past the end of the bins (a damaged record), end the slice at once
        let b = if self.r.pos() >= self.limit { self.short = true; 1 } else { self.r.get(1) };
        self.enc.terminate(b);
        b
    }
    fn pcm(&mut self, n: usize) -> Result<(), String> {
        self.r.seek(self.r.pos().div_ceil(8) * 8);
        if self.r.pos() + 8 * (n + 1) > self.limit { return Err("h264: PCM bytes missing".into()); }
        let padding = self.r.get(8);
        let bits = (8 - self.enc.out.len() % 8) % 8;
        if padding >> bits != 0 { return Err("h264: bad PCM alignment bits".into()); }
        self.enc.out.put(padding, bits as u32);
        for _ in 0..n {
            let b = self.r.get(8);
            self.enc.out.put(b, 8);
        }
        let (ctx, out) = (self.enc.ctx, std::mem::take(&mut self.enc.out));
        self.enc = CabacEncoder::new(ctx, out);
        Ok(())
    }
}

/// A slice taken apart.
pub struct SliceBins {
    /// The bins, packed (see [`Recorder`]).
    pub bins: Vec<u8>,
    /// How many bins.
    pub count: usize,
    pub macroblocks: usize,
    /// The bits between the end of the slice data (its stop bit) and the byte boundary: zero by
    /// the standard, but x264 marks them.
    pub align: u8,
    /// Offset in the RBSP of the first byte after the slice data: only cabac_zero_words follow.
    pub data_end: usize,
    /// A CAVLC slice's P_8x8ref0 macroblocks (see [`cavlc::Slice`]).
    pub ref0: Vec<u32>,
}

/// Decode the bins of the slice whose RBSP is `rbsp` (header already parsed).
pub fn slice_bins(rbsp: &[u8], h: &SliceHeader, sps: &Sps, pps: &Pps, scratch: &mut Scratch) -> Result<SliceBins, String> {
    slice_bins_seen(rbsp, h, sps, pps, scratch, &mut |_, _, _| {})
}

/// [`slice_bins`], showing every bin (context, value, hints) to `seen` as it is decoded.
pub fn slice_bins_seen(rbsp: &[u8], h: &SliceHeader, sps: &Sps, pps: &Pps, scratch: &mut Scratch, seen: &mut dyn FnMut(u16, u8, &Info)) -> Result<SliceBins, String> {
    if !pps.cabac {
        // a CAVLC slice: its syntax elements, binarized (`seen` sees nothing)
        let s = cavlc::to_bins(rbsp, h, sps, pps, scratch).ok_or("h264: CAVLC slice not taken")?;
        return Ok(SliceBins { bins: s.bins, count: 0, macroblocks: s.macroblocks, align: 0, data_end: rbsp.len(), ref0: s.ref0 });
    }
    let ctx = init_contexts(h.slice_type, h.cabac_init_idc, h.qp);
    let mut rec = Recorder { dec: CabacDecoder::new(rbsp, h.data, ctx), rbsp, seen, out: BitWriter::new(), count: 0, overrun: false };
    let macroblocks = walk(&mut rec, h, sps, pps, scratch)?;
    // the last bit the engine read is the RBSP stop bit; alignment bits fill the byte
    let end = rec.dec.end();
    if end > rbsp.len() * 8 || rec.overrun { return Err("h264: slice data runs past the NAL unit".into()); }
    let align = if end % 8 != 0 { rbsp[end / 8] & (0xFF >> (end % 8)) } else { 0 };
    Ok(SliceBins { bins: rec.out.into_bytes(), count: rec.count, macroblocks, align, data_end: end.div_ceil(8), ref0: vec![] })
}

/// Re-encode a slice's packed bins (`bins` starts with them): the bytes of its slice data (from
/// the header's `data` offset) through the stop bit and the `align` bits after it, and how many
/// bytes of `bins` the slice took.
pub fn encode_bins(h: &SliceHeader, bins: &[u8], align: u8, sps: &Sps, pps: &Pps, scratch: &mut Scratch) -> Result<(Vec<u8>, usize), String> {
    let ctx = init_contexts(h.slice_type, h.cabac_init_idc, h.qp);
    let mut rep = Replayer { enc: CabacEncoder::new(ctx, BitWriter::new()), r: BitReader::new(bins, 0), limit: bins.len() * 8, short: false };
    walk(&mut rep, h, sps, pps, scratch)?;
    if rep.short || rep.r.pos() > rep.limit { return Err("h264: bins end inside a slice".into()); }
    let used = rep.r.pos().div_ceil(8);
    let mut out = rep.enc.out;
    let pad = (8 - out.len() % 8) % 8;
    if align as u32 >> pad != 0 { return Err("h264: bad alignment bits".into()); }
    out.put(align as u32, pad as u32);
    Ok((out.into_bytes(), used))
}

/// What CABAC itself spends on bins, by the standard's probability model (developer statistics).
pub struct StdCost {
    ctx: [Ctx; CONTEXTS],
}

impl StdCost {
    pub fn new(h: &SliceHeader) -> Self { StdCost { ctx: init_contexts(h.slice_type, h.cabac_init_idc, h.qp) } }

    /// The cost in bits of bin `b` coded with context `c` (or [`BYPASS`] / [`TERMINATE`]).
    pub fn bin(&mut self, c: u16, b: u8) -> f64 {
        match c {
            BYPASS => 1.0,
            TERMINATE => if b == 1 { 7.0 } else { 0.012 },
            _ => {
                let x = &mut self.ctx[c as usize];
                // the LPS probability of state s is 0.5 · α^s with α = (0.01875 / 0.5)^(1/63)
                let p_lps = 0.5 * (0.0375f64).powf(x.state as f64 / 63.0);
                if b == x.mps {
                    x.state = (x.state + 1).min(62);
                    -(1.0 - p_lps).log2()
                } else {
                    if x.state == 0 { x.mps ^= 1; }
                    x.state = NEXT_LPS[x.state as usize];
                    -p_lps.log2()
                }
            }
        }
    }
}

// ------------------------------------------------------------------ streams as records
//
// A stream of NAL units, each after a big-endian length (the samples of an MP4 track), becomes
// records, in stream order:
//
//   0 RAW     n:varint                  n bytes kept as they are (in the container's raw leaf)
//   1 SLICE   nal:u8 hlen:u8 header[hlen] zeros:varint align:u8 [n:varint n × Δaddr:varint] bins…
//             a slice: its NAL header byte, the RBSP bytes the slice header is in (the bits of
//             the last one that are slice data: zero), the zero bytes after the slice data
//             (cabac_zero_words), the bits after its stop bit; of a CAVLC slice, the macroblocks
//             of type P_8x8ref0 (see `cavlc::Slice`); and its packed bins (see `Recorder`)
//   2 CONFIG  n:varint nal[n]           a parameter set the slices refer to, known from outside
//                                       the stream (or from before the leaf): not written out
//   3 PARAM   n:varint nal[n]           a parameter set NAL unit in the stream
//
// Records are cut into leaves, and every leaf starts with the parameter sets known at that
// point, so each is modelled on its own.
pub const REC_RAW: u8 = 0;
pub const REC_SLICE: u8 = 1;
pub const REC_CONFIG: u8 = 2;
pub const REC_PARAM: u8 = 3;

/// Largest parameter set NAL unit taken as a record, and most zero bytes after a slice's data.
pub const MAX_PARAM: usize = 4096;
pub const MAX_ZEROS: usize = 1 << 16;
/// Records of about this many bytes form one leaf.
const LEAF_BYTES: usize = 512 << 10;

/// The parameter sets in force at a point of a stream, by id.
#[derive(Clone, Default)]
pub struct Params {
    sps: Vec<Option<(Vec<u8>, Sps)>>,
    pps: Vec<Option<(Vec<u8>, Pps)>>,
}

impl Params {
    /// Take in a parameter set NAL unit (header byte and payload). False if it is not one that
    /// parses.
    pub fn add(&mut self, nal: &[u8]) -> bool {
        let Some((&hdr, body)) = nal.split_first() else { return false };
        if nal.len() > MAX_PARAM || hdr & 0x80 != 0 { return false; }
        match hdr & 31 {
            7 => {
                let Some(x) = parse_sps(body) else { return false };
                let id = x.id as usize;
                if id >= 32 { return false; }
                if self.sps.len() <= id { self.sps.resize(id + 1, None); }
                self.sps[id] = Some((nal.to_vec(), x));
                true
            }
            8 => {
                let Some(x) = parse_pps(body) else { return false };
                let id = x.id as usize;
                if id >= 256 { return false; }
                if self.pps.len() <= id { self.pps.resize(id + 1, None); }
                self.pps[id] = Some((nal.to_vec(), x));
                true
            }
            _ => false,
        }
    }

    /// The parameter sets a slice naming picture parameter set `id` uses.
    pub fn of(&self, id: u32) -> Option<(Sps, Pps)> {
        let p = &self.pps.get(id as usize)?.as_ref()?.1;
        let s = &self.sps.get(p.sps as usize)?.as_ref()?.1;
        Some((s.clone(), p.clone()))
    }

    /// The NAL units of all sets, sequence parameter sets first.
    pub fn nals(&self) -> impl Iterator<Item = &[u8]> {
        self.sps.iter().flatten().map(|x| &x.0[..]).chain(self.pps.iter().flatten().map(|x| &x.0[..]))
    }
}

/// Read a varint at `b[*p]`.
pub fn get_var(b: &[u8], p: &mut usize) -> Option<usize> {
    let mut v = 0u64;
    for k in 0..5 {
        let x = *b.get(*p)?;
        *p += 1;
        v |= ((x & 0x7F) as u64) << (7 * k);
        if x & 0x80 == 0 { return (k == 0 || x != 0).then_some(v as usize); }
    }
    None
}

fn put_var(o: &mut Vec<u8>, v: usize) { crate::nsc3::format::put_var(o, v as u64) }

/// The header of a slice record's NAL unit, parsed with the parameter sets in force. The header
/// must end in the last byte of `header`, and the bits after it there be zero.
pub fn record_header(nal: u8, header: &[u8], ps: &Params) -> Option<(SliceHeader, Sps, Pps)> {
    if nal & 0x80 != 0 || !matches!(nal & 31, 1 | 5) { return None; }
    let r = parse_slice_header(header, nal & 31, (nal >> 5) & 3, &|id| ps.of(id))?;
    let data = r.0.data;
    (data.div_ceil(8) == header.len() && (data % 8 == 0 || header[data / 8] & (0xFF >> (data % 8)) == 0)).then_some(r)
}

/// The RBSP of a slice from its record's parts: the header bytes (see [`record_header`]), its
/// bins (`bins` starts with them), and for a CABAC slice the bits after its stop bit and the
/// zero bytes after its data; for a CAVLC slice the P_8x8ref0 macroblocks. Also returns how many
/// bytes of `bins` the slice took.
#[allow(clippy::too_many_arguments)]
pub fn slice_rbsp(h: &SliceHeader, header: &[u8], bins: &[u8], align: u8, zeros: usize, ref0: &[u32], sps: &Sps, pps: &Pps, scratch: &mut Scratch)
    -> Result<(Vec<u8>, usize), String> {
    if pps.cabac {
        if !ref0.is_empty() { return Err("h264: bad record (corrupt file)".into()); }
        let (data, used) = encode_bins(h, bins, align, sps, pps, scratch)?;
        let mut rbsp = Vec::with_capacity(header.len() + data.len() + zeros);
        rbsp.extend_from_slice(header);
        rbsp.extend_from_slice(&data);
        rbsp.resize(rbsp.len() + zeros, 0);
        return Ok((rbsp, used));
    }
    if align != 0 || zeros != 0 { return Err("h264: bad record (corrupt file)".into()); }
    // the slice data follows the header's last bit
    let mut w = BitWriter::new();
    w.extend(&header[..h.data / 8]);
    if h.data % 8 != 0 { w.put((header[h.data / 8] >> (8 - h.data % 8)) as u32, (h.data % 8) as u32); }
    let used = cavlc::from_bins(h, bins, ref0, sps, pps, scratch, &mut w)?;
    Ok((w.into_bytes(), used))
}

/// The record of the slice NAL unit `nal`, if it is a slice [`walk`] handles and re-encoding
/// its bins gives its bytes back.
fn slice_record(nal: &[u8], ps: &Params, scratch: &mut Scratch, out: &mut Vec<u8>) -> Option<()> {
    let rbsp = unescape(&nal[1..]);
    if escape(&rbsp) != nal[1..] { return None; }
    let (h, _, _) = parse_slice_header(&rbsp, nal[0] & 31, (nal[0] >> 5) & 3, &|id| ps.of(id))?;
    let hlen = h.data.div_ceil(8);
    if hlen == 0 || hlen > 255 || hlen > rbsp.len() { return None; }
    // what the decoder will do: parse the header from its own bytes, slice data bits zeroed
    let mut header = rbsp[..hlen].to_vec();
    if h.data % 8 != 0 { header[hlen - 1] &= !(0xFF >> (h.data % 8)); }
    let (h, sps, pps) = record_header(nal[0], &header, ps)?;
    let sb = slice_bins(&rbsp, &h, &sps, &pps, scratch).ok()?;
    let zeros = rbsp.len().checked_sub(sb.data_end)?;
    if zeros > MAX_ZEROS || rbsp[sb.data_end..].iter().any(|&x| x != 0) { return None; }
    let (again, used) = slice_rbsp(&h, &header, &sb.bins, sb.align, zeros, &sb.ref0, &sps, &pps, scratch).ok()?;
    if again != rbsp || used != sb.bins.len() { return None; }
    out.extend_from_slice(&[REC_SLICE, nal[0], hlen as u8]);
    out.extend_from_slice(&header);
    put_var(out, zeros);
    out.push(sb.align);
    if !pps.cabac {
        put_var(out, sb.ref0.len());
        let mut last = 0;
        for &a in &sb.ref0 {
            put_var(out, (a - last) as usize);
            last = a;
        }
    }
    out.extend_from_slice(&sb.bins);
    Some(())
}

/// The P_8x8ref0 macroblocks of a CAVLC slice record at `b[*p]`: a count, then the differences
/// of their (rising) addresses.
pub fn record_ref0(b: &[u8], p: &mut usize) -> Option<Vec<u32>> {
    let n = get_var(b, p).filter(|&n| n <= MAX_MBS)?;
    let mut out = Vec::with_capacity(n);
    let mut last = 0usize;
    for k in 0..n {
        last = last.checked_add(get_var(b, p).filter(|&d| d > 0 || k == 0)?).filter(|&a| a < MAX_MBS)?;
        out.push(last as u32);
    }
    Some(out)
}

/// A stream taken apart.
pub struct Split {
    /// The records, the offsets where they are cut into leaves, and the bytes kept raw.
    pub records: Vec<u8>,
    pub cuts: Vec<usize>,
    pub raw: Vec<u8>,
    /// Slices seen, and recoded.
    pub count: (usize, usize),
}

/// Take apart the stream `b`: samples of `sizes` bytes back to back, each a sequence of NAL units
/// after `nal_len`-byte lengths; `config`: the parameter sets known from the container. None when
/// no slice can be recoded.
pub fn split(b: &[u8], sizes: &[u32], nal_len: usize, config: &[Vec<u8>]) -> Option<Split> { split_leaves(b, sizes, nal_len, config, LEAF_BYTES) }

/// [`split`] with leaves of about `leaf_bytes` bytes.
fn split_leaves(b: &[u8], sizes: &[u32], nal_len: usize, config: &[Vec<u8>], leaf_bytes: usize) -> Option<Split> {
    if !matches!(nal_len, 1 | 2 | 4) { return None; }
    let mut ps = Params::default();
    for n in config { ps.add(n); }
    let mut s = Split { records: Vec::with_capacity(b.len() + b.len() / 2), cuts: vec![], raw: vec![], count: (0, 0) };
    let mut scratch = Scratch::default();
    let (mut pending, mut leaf) = (0usize, 0usize);
    fn flush(s: &mut Split, pending: &mut usize) {
        if *pending > 0 {
            s.records.push(REC_RAW);
            put_var(&mut s.records, *pending);
            *pending = 0;
        }
    }
    fn configs(s: &mut Split, ps: &Params) {
        for n in ps.nals() {
            s.records.push(REC_CONFIG);
            put_var(&mut s.records, n.len());
            s.records.extend_from_slice(n);
        }
    }
    configs(&mut s, &ps);
    let mut o = 0usize;
    for &size in sizes {
        let end = o.checked_add(size as usize).filter(|&e| e <= b.len())?;
        let mut p = o;
        while p < end {
            // a NAL unit that does not fit its sample: the rest of the sample stays as it is
            let n = if p + nal_len <= end { b[p..p + nal_len].iter().fold(0usize, |a, &x| a << 8 | x as usize) } else { 0 };
            if n == 0 || n > end - p - nal_len {
                s.raw.extend_from_slice(&b[p..end]);
                pending += end - p;
                break;
            }
            let nal = &b[p + nal_len..p + nal_len + n];
            let mut taken = false;
            match nal[0] & 31 {
                7 | 8 => {
                    if ps.add(nal) {
                        flush(&mut s, &mut pending);
                        s.records.push(REC_PARAM);
                        put_var(&mut s.records, n);
                        s.records.extend_from_slice(nal);
                        taken = true;
                    }
                }
                1 | 5 if nal[0] & 0x80 == 0 => {
                    s.count.0 += 1;
                    flush(&mut s, &mut pending);
                    if s.records.len() - leaf >= leaf_bytes {
                        s.cuts.push(s.records.len());
                        leaf = s.records.len();
                        configs(&mut s, &ps);
                    }
                    let at = s.records.len();
                    if slice_record(nal, &ps, &mut scratch, &mut s.records).is_some() {
                        s.count.1 += 1;
                        taken = true;
                    } else {
                        s.records.truncate(at);
                    }
                }
                _ => {}
            }
            if !taken {
                s.raw.extend_from_slice(&b[p..p + nal_len + n]);
                pending += nal_len + n;
            }
            p += nal_len + n;
        }
        o = end;
    }
    flush(&mut s, &mut pending);
    (s.count.1 > 0).then_some(s)
}

/// Rebuild a stream from its records and raw bytes.
pub fn rebuild(records: &[u8], raw: &[u8], nal_len: usize, out: &mut Vec<u8>) -> Result<(), String> {
    let bad = || "h264: bad record (corrupt file)".to_string();
    if !matches!(nal_len, 1 | 2 | 4) { return Err(bad()); }
    let put_len = |out: &mut Vec<u8>, n: usize| -> Result<(), String> {
        if nal_len < 8 && n >> (8 * nal_len) != 0 { return Err(bad()); }
        out.extend_from_slice(&(n as u64).to_be_bytes()[8 - nal_len..]);
        Ok(())
    };
    let mut ps = Params::default();
    let mut scratch = Scratch::default();
    let (mut p, mut rp) = (0usize, 0usize);
    while p < records.len() {
        let tag = records[p];
        p += 1;
        match tag {
            REC_RAW => {
                let n = get_var(records, &mut p).ok_or_else(bad)?;
                out.extend_from_slice(raw.get(rp..rp.checked_add(n).ok_or_else(bad)?).ok_or_else(bad)?);
                rp += n;
            }
            REC_CONFIG | REC_PARAM => {
                let n = get_var(records, &mut p).ok_or_else(bad)?;
                let nal = records.get(p..p.checked_add(n).ok_or_else(bad)?).ok_or_else(bad)?;
                p += n;
                if !ps.add(nal) { return Err(bad()); }
                if tag == REC_PARAM {
                    put_len(out, n)?;
                    out.extend_from_slice(nal);
                }
            }
            REC_SLICE => {
                let (&nal, &hlen) = (records.get(p).ok_or_else(bad)?, records.get(p + 1).ok_or_else(bad)?);
                p += 2;
                let header = records.get(p..p + hlen as usize).ok_or_else(bad)?;
                p += hlen as usize;
                let zeros = get_var(records, &mut p).filter(|&z| z <= MAX_ZEROS).ok_or_else(bad)?;
                let align = *records.get(p).ok_or_else(bad)?;
                p += 1;
                let (h, sps, pps) = record_header(nal, header, &ps).ok_or_else(bad)?;
                let ref0 = if pps.cabac { vec![] } else { record_ref0(records, &mut p).ok_or_else(bad)? };
                let (rbsp, used) = slice_rbsp(&h, header, records.get(p..).ok_or_else(bad)?, align, zeros, &ref0, &sps, &pps, &mut scratch)?;
                p += used;
                let body = escape(&rbsp);
                put_len(out, 1 + body.len())?;
                out.push(nal);
                out.extend_from_slice(&body);
            }
            _ => return Err(bad()),
        }
    }
    if rp != raw.len() { return Err(bad()); }
    Ok(())
}

#[cfg(test)]
pub mod tests {
    use super::*;

    fn rng(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }

    /// The video track of an MP4: its samples back to back, their sizes, the NAL length size and
    /// the configuration's parameter sets.
    pub fn track(mp4: &[u8]) -> (Vec<u8>, Vec<u32>, usize, Vec<Vec<u8>>) {
        let idx = crate::recode::mp4::index(&mut std::io::Cursor::new(mp4), mp4.len() as u64).unwrap();
        let (mut stream, mut sizes) = (vec![], vec![]);
        for r in idx.runs.iter().filter(|r| matches!(r.what, crate::recode::mp4::Stream::Avc { .. })) {
            stream.extend_from_slice(&mp4[r.start as usize..(r.start + r.len()) as usize]);
            sizes.extend_from_slice(&r.sizes);
        }
        let avc = &idx.avc[0];
        (stream, sizes, avc.nal_len as usize, avc.sps.iter().chain(&avc.pps).cloned().collect())
    }

    /// x264 streams: High profile with CABAC, B slices and the 8x8 transform; a short clip with an audio
    /// track beside it; Baseline (CAVLC, two slices a picture); High profile coded with CAVLC.
    pub const FIXTURES: [&[u8]; 4] = [include_bytes!("../../tests/fixtures/video_cabac.mp4"), include_bytes!("../../tests/fixtures/tone_video.mp4"),
        include_bytes!("../../tests/fixtures/video_cavlc.mp4"), include_bytes!("../../tests/fixtures/video_cavlc_high.mp4")];

    #[test]
    fn escaping_inverts() {
        let mut s = 7u64;
        for _ in 0..300 {
            // few distinct values, so runs of zeros before 0..3 are common
            let rbsp: Vec<u8> = (0..rng(&mut s) % 60).map(|_| (rng(&mut s) % 5) as u8).collect();
            let nal = escape(&rbsp);
            assert_eq!(unescape(&nal), rbsp);
            assert!(!nal.windows(3).any(|w| w[0] == 0 && w[1] == 0 && w[2] < 3), "start code emulated: {nal:02x?}");
        }
    }

    #[test]
    fn streams_split_and_rebuild_exactly() {
        for (k, mp4) in FIXTURES.iter().enumerate() {
            let (stream, sizes, nal_len, config) = track(mp4);
            // every slice of these x264 streams is recoded; several leaves rebuild like one
            for leaf in [LEAF_BYTES, 1500] {
                let sp = split_leaves(&stream, &sizes, nal_len, &config, leaf).unwrap();
                assert!(sp.count.0 >= 10 && sp.count.1 == sp.count.0, "fixture {k}: {:?} slices recoded", sp.count);
                if k != 1 { assert_eq!(leaf == 1500, sp.cuts.len() >= 3, "fixture {k}: {} cuts", sp.cuts.len()); }
                let mut out = vec![];
                rebuild(&sp.records, &sp.raw, nal_len, &mut out).unwrap();
                assert!(out == stream, "fixture {k}: rebuild differs");
                // each leaf stands alone: it starts with the parameter sets
                for (a, b) in std::iter::once(&0).chain(&sp.cuts).zip(sp.cuts.iter().chain([&sp.records.len()])) {
                    assert_eq!(sp.records[*a], REC_CONFIG);
                    assert!(a < b);
                }
            }
            // without the configuration no slice can be parsed
            assert!(split(&stream, &sizes, nal_len, &[]).is_none());
            // a wrong length size, or sizes that do not fit, are refused or rebuild exactly
            assert!(split(&stream, &sizes, 3, &config).is_none());
            if let Some(sp) = split(&stream, &sizes, 2, &config) {
                let mut out = vec![];
                rebuild(&sp.records, &sp.raw, 2, &mut out).unwrap();
                assert!(out == stream);
            }
        }
    }

    #[test]
    fn damaged_streams_rebuild_exactly_or_stay_bytes() {
        for fixture in [0, 2, 3] { damaged_stream(fixture); }
    }

    fn damaged_stream(fixture: usize) {
        let (stream, sizes, nal_len, config) = track(FIXTURES[fixture]);
        let mut s = 11u64;
        let mut recoded = 0;
        for round in 0..150 {
            let mut bad = stream.clone();
            for _ in 0..1 + round % 4 {
                let at = (rng(&mut s) % bad.len() as u64) as usize;
                match rng(&mut s) % 3 {
                    0 => bad[at] ^= 1 << (rng(&mut s) % 8),
                    1 => bad[at] = 0,
                    _ => bad[at] = rng(&mut s) as u8,
                }
            }
            // whatever is still taken apart must come back exactly
            if let Some(sp) = split(&bad, &sizes, nal_len, &config) {
                let mut out = vec![];
                rebuild(&sp.records, &sp.raw, nal_len, &mut out).unwrap();
                assert!(out == bad, "fixture {fixture}, round {round}: a damaged stream was not rebuilt exactly");
                recoded += sp.count.1;
            }
        }
        assert!(recoded > 800, "fixture {fixture}: {recoded} slices recoded in damaged streams");
    }

    #[test]
    fn damaged_records_never_panic() {
        for fixture in [0, 2, 3] { damaged_records(fixture); }
    }

    fn damaged_records(fixture: usize) {
        let (stream, sizes, nal_len, config) = track(FIXTURES[fixture]);
        let sp = split(&stream, &sizes, nal_len, &config).unwrap();
        let mut s = 5u64;
        let mut errors = 0;
        for round in 0..300 {
            let mut bad = sp.records.clone();
            let at = (rng(&mut s) % bad.len() as u64) as usize;
            match round % 4 {
                0 => bad[at] ^= 1 << (rng(&mut s) % 8),
                1 => bad.truncate(at),
                2 => bad[at] = rng(&mut s) as u8,
                _ => { bad.insert(at, rng(&mut s) as u8); }
            }
            let mut out = vec![];
            if rebuild(&bad, &sp.raw, nal_len, &mut out).is_err() { errors += 1; }
        }
        assert!(errors > 50, "fixture {fixture}: only {errors} of 300 damaged record sets were refused");
        // raw bytes that do not match the records
        let mut out = vec![];
        assert!(rebuild(&sp.records, &sp.raw[..sp.raw.len() - 1], nal_len, &mut out).is_err());
        let mut more = sp.raw.clone();
        more.push(0);
        assert!(rebuild(&sp.records, &more, nal_len, &mut vec![]).is_err());
        assert!(rebuild(&sp.records, &sp.raw, 3, &mut vec![]).is_err());
    }

    #[test]
    fn parameter_sets_bound_what_is_parsed() {
        // a picture larger than the largest level allows is not supported
        let (_, _, _, config) = track(FIXTURES[0]);
        let mut ps = Params::default();
        for n in &config { assert!(ps.add(n)); }
        let (sps, pps) = ps.of(0).unwrap();
        assert!(sps.supported && pps.supported && sps.width_mbs == 10 && sps.height_mbs == 6);
        assert!(!ps.add(&[0x67]) && !ps.add(&[]) && !ps.add(&[0x65, 1, 2, 3]));
        assert!(ps.of(1).is_none());
        // random bytes as parameter sets and slice headers: parsed or refused, never a panic
        let mut s = 3u64;
        for _ in 0..3000 {
            let n: Vec<u8> = (0..1 + rng(&mut s) % 40).map(|_| rng(&mut s) as u8).collect();
            let _ = (parse_sps(&n), parse_pps(&n));
            let _ = record_header(0x65, &n, &ps);
            let _ = record_header(0x41, &n, &ps);
        }
    }
}

impl Params {
    /// Why the slice whose RBSP is `rbsp` is not one [`walk`] handles (developer diagnostics).
    pub fn why_not(&self, rbsp: &[u8]) -> &'static str {
        let mut r = BitReader::new(rbsp, 0);
        let (Some(_), Some(st), Some(id)) = (ue(&mut r), ue(&mut r), ue(&mut r)) else { return "bad slice header" };
        if st > 9 || st % 5 > 2 { return "SP or SI slice"; }
        let Some(p) = self.pps.get(id as usize).and_then(|x| x.as_ref()) else { return "no picture parameter set" };
        let Some(s) = self.sps.get(p.1.sps as usize).and_then(|x| x.as_ref()) else { return "no sequence parameter set" };
        if !s.1.supported { return s.1.why; }
        if p.1.fmo { return "slice groups"; }
        "slice header"
    }
}
