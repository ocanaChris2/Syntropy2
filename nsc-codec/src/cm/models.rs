//! The models of one lane, by leaf kind: the generic byte predictor and the dedicated models
//! (JPEG coefficients, PCM audio, …). Each is created when its first leaf arrives and keeps
//! learning across the lane's leaves of its kind. The encoder, the decoder and the encoder's
//! concurrent verifier all code leaves through [`Models::code`], so they agree by construction.
use super::aac::AacModel;
use super::audio::AudioModel;
use super::coder::Coder;
use super::h264::H264Model;
use super::indexed::IndexedModel;
use super::jpeg::JpegModel;
use super::levels::ModelCfg;
use super::mp3::Mp3Model;
use super::predictor::{code_bytes, ByteBuf, Predictor};
use crate::nsc3::format::{kind, Segment};

pub struct Models<'a> {
    cfg: &'a ModelCfg,
    pred: Option<Predictor>,
    jpeg: Option<JpegModel>,
    audio: Option<AudioModel>,
    mp3: Option<Box<Mp3Model>>,
    aac: Option<Box<AacModel>>,
    h264: Option<Box<H264Model>>,
    indexed: Option<Box<IndexedModel>>,
}

impl<'a> Models<'a> {
    pub fn new(cfg: &'a ModelCfg) -> Self { Models { cfg, pred: None, jpeg: None, audio: None, mp3: None, aac: None, h264: None, indexed: None } }

    /// Code (encode, or decode when `buf` is the decoder's) leaf `s`, which occupies
    /// `buf[rel..rel + s.raw_len]`. Stored and neural leaves carry no coded bits: callers handle
    /// them.
    pub fn code<C: Coder, B: ByteBuf>(&mut self, c: &mut C, buf: &mut B, s: &Segment, rel: usize) -> Result<(), String> {
        let range = rel..rel + s.raw_len as usize;
        match s.kind {
            kind::JPEGC => self.jpeg.get_or_insert_with(JpegModel::new).code_leaf(c, buf, range, &s.params),
            kind::AUDIO => self.audio.get_or_insert_with(AudioModel::new).code_leaf(c, buf, range, &s.params),
            kind::MP3F => self.mp3.get_or_insert_with(|| Box::new(Mp3Model::new())).code_leaf(c, buf, range, &s.params),
            kind::AACF => self.aac.get_or_insert_with(|| Box::new(AacModel::new())).code_leaf(c, buf, range, &s.params),
            kind::PIX => self.indexed.get_or_insert_with(|| Box::new(IndexedModel::new(s.raw_len as usize))).code_leaf(c, buf, range, &s.params),
            kind::H264B => {
                // the leaf's profile picks the model (one per lane: a writer uses one profile)
                let full = s.params.get(1) == Some(&1);
                if self.h264.as_ref().is_some_and(|m| m.full() != full) { self.h264 = None; }
                self.h264.get_or_insert_with(|| Box::new(H264Model::new(full))).code_leaf(c, buf, range, &s.params)
            }
            k => {
                let pred = self.pred.get_or_insert_with(|| Predictor::new(self.cfg));
                pred.set_segment(k, &s.params, rel);
                code_bytes(pred, c, buf, range);
                Ok(())
            }
        }
    }
}

/// How many times its size a leaf of kind `k` counts when a block's leaves are grouped into
/// lanes. The MP3 model is slow per byte and learns within a few hundred frames, so its leaves
/// form short lanes: a large MP3 codes on many cores at once. The same goes for AAC, and for
/// H.264 records (a model started afresh costs about 4 KB there, so lanes of a few megabytes
/// lose 0.1-0.2%).
pub fn lane_weight(k: u8) -> usize { if matches!(k, kind::MP3F | kind::AACF | kind::H264B) { 8 } else { 1 } }

/// Is a leaf of kind `k` coded by the byte predictor (whose tables are sized by the bytes it
/// models)? The others have models of their own, or none.
pub fn bytewise(k: u8) -> bool { !matches!(k, kind::STORED | kind::NEURAL | kind::JPEGC | kind::AUDIO | kind::MP3F | kind::AACF | kind::H264B | kind::PIX) }

/// Memory of the dedicated models a block with the kinds `has` needs per lane, beyond the byte
/// predictor's tables.
pub fn dedicated_memory(has: impl Fn(u8) -> bool) -> usize {
    let mut m = 0;
    if has(kind::MP3F) { m += 12 << 20; }
    if has(kind::AACF) { m += 12 << 20; }
    if has(kind::H264B) { m += 48 << 20; }
    if has(kind::PIX) { m += 120 << 20; }
    m
}

/// Can a lane boundary split a leaf of kind `k` (at its alignment)? Dedicated models code
/// structured leaves that must stay whole.
pub fn splittable(k: u8) -> bool { !matches!(k, kind::JPEGC | kind::MP3F | kind::AACF | kind::H264B | kind::PIX) }
