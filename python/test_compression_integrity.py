#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  COMPRESSION INTEGRITY & SECURITY TEST  (standalone, training-free)
# =============================================================================
#  Proves, end to end, that the exported neuro-symbolic compressor is LOSSLESS
#  and that its container format cannot fail silently:
#
#    1. Calibrates the model if necessary (temperature T fitted on the val
#       split, cached per model hash, baked into the archive header).
#    2. COMPRESSES a slice of enwik8 with the real model: a streaming stepper
#       (GRU state + transformer KV cache, window reset every seg_len tokens)
#       drives a 32-bit range coder over the model's per-token distributions.
#    3. DECOMPRESSES it back using ONLY the archive + exported artifacts —
#       classical features are rebuilt incrementally from decoded tokens, the
#       decoder never sees the original data — and verifies byte-for-byte
#       identity (SHA-256 + full memcmp).
#    4. SECURITY suite: bit flips, truncation, header tampering, wrong-model
#       decode and random-garbage payloads must all be DETECTED (rejected or
#       flagged), never returned as silently-wrong output. Optional HMAC
#       authentication via the NSC_INTEGRITY_KEY environment variable.
#
#  The encoder and decoder share the same stepper code, so their probability
#  tables are bit-identical by construction; a batched-vs-streaming parity
#  diagnostic reports how far the fast batched path drifts (informational,
#  relevant for the Rust port).
#
#  Usage:
#      python test_compression_integrity.py                 # 64 KiB of the test split
#      python test_compression_integrity.py --bytes 262144  # bigger slice
#      python test_compression_integrity.py --bytes 0       # whole 5 MB test split (slow)
#      python test_compression_integrity.py --quick         # 8 KiB smoke (~30 s)
#      python test_compression_integrity.py --mode selftest # coder-only gates, no model
#      python test_compression_integrity.py --mode compress --bytes 65536
#      python test_compression_integrity.py --mode decompress --archive results/integrity/....syz
#
#  Exit code 0 = every gate passed; 1 = at least one gate failed.
# =============================================================================
from __future__ import annotations

import os
import sys
import json
import time
import math
import gzip
import lzma
import hmac
import hashlib
import argparse
import platform
import subprocess
from typing import Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (this file is in python/)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import torch

import neuro_symbolic_compressor as nsc
from neuro_symbolic_compressor import Config, _quantize, _RangeCoder, peaked_distribution
from benchmark_enterprise import (locate_paths, load_checkpoint_meta, build_model,
                                  FastO2, compute_features, encode_bytes_chunked,
                                  writable_dir)

MAGIC = b"SYNZIT01"
FORMAT_NAME = "syntropy-integrity-v1"
QUANT_BITS = 22  # table total 2^22: floor mass 8192/2^22 ~ 0.2%; safe, coder needs total <= 2^30
HMAC_ENV = "NSC_INTEGRITY_KEY"


# =============================================================================
#  Errors: every rejection path is typed so the security suite can assert that
#  tampering produces the RIGHT kind of controlled failure.
# =============================================================================
class IntegrityError(Exception):
    kind = "integrity"


class FileCorrupt(IntegrityError):
    kind = "file-corrupt"


class HeaderInvalid(IntegrityError):
    kind = "header-invalid"


class ArtifactMismatch(IntegrityError):
    kind = "artifact-mismatch"


class AuthFailure(IntegrityError):
    kind = "auth-failure"


class DecodedMismatch(IntegrityError):
    kind = "decoded-mismatch"


# =============================================================================
#  Streaming range coder — same Witten-Neal-Cleary math as nsc._RangeCoder but
#  restructured for symbol-at-a-time use (the decoder cannot know all
#  cumulative tables upfront: each one depends on previously decoded tokens).
#  coder_selftests() proves bit-for-bit equivalence with the reference.
# =============================================================================
class RangeEncoder:
    PREC = _RangeCoder.PREC
    MASK = _RangeCoder.MASK
    HALF = _RangeCoder.HALF
    QUARTER = _RangeCoder.QUARTER
    THREEQ = _RangeCoder.THREEQ

    def __init__(self):
        self.low, self.high, self.pending = 0, self.MASK, 0
        self.bits = bytearray()          # one byte per bit; packed at finish()

    def _emit(self, bit: int) -> None:
        self.bits.append(bit)
        if self.pending:
            self.bits.extend([1 - bit] * self.pending)
            self.pending = 0

    def encode(self, cl: int, ch: int, total: int) -> None:
        rng = self.high - self.low + 1
        self.high = self.low + (rng * ch) // total - 1
        self.low = self.low + (rng * cl) // total
        while True:
            if self.high < self.HALF:
                self._emit(0)
            elif self.low >= self.HALF:
                self._emit(1)
                self.low -= self.HALF
                self.high -= self.HALF
            elif self.low >= self.QUARTER and self.high < self.THREEQ:
                self.pending += 1
                self.low -= self.QUARTER
                self.high -= self.QUARTER
            else:
                break
            self.low = (self.low << 1) & self.MASK
            self.high = ((self.high << 1) | 1) & self.MASK

    def finish(self) -> Tuple[bytes, int]:
        self.pending += 1
        self._emit(0 if self.low < self.QUARTER else 1)
        n_bits = len(self.bits)
        payload = np.packbits(np.frombuffer(bytes(self.bits), dtype=np.uint8)).tobytes()
        return payload, n_bits


class RangeDecoder:
    PREC = _RangeCoder.PREC
    MASK = _RangeCoder.MASK
    HALF = _RangeCoder.HALF
    QUARTER = _RangeCoder.QUARTER
    THREEQ = _RangeCoder.THREEQ

    def __init__(self, payload: bytes, n_bits: int):
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))[:n_bits]
        self._bits = bits
        self._i = 0
        self.low, self.high, self.value = 0, self.MASK, 0
        for _ in range(self.PREC):
            self.value = ((self.value << 1) | self._next()) & self.MASK

    def _next(self) -> int:
        # Past the end the reference decoder reads zero padding; an unlimited
        # zero tail keeps hostile/garbage streams bounded by n_tokens alone.
        i = self._i
        self._i = i + 1
        return int(self._bits[i]) if i < len(self._bits) else 0

    def decode(self, cum: np.ndarray) -> int:
        total = int(cum[-1])
        rng = self.high - self.low + 1
        scaled = ((self.value - self.low + 1) * total - 1) // rng
        s = int(np.searchsorted(cum, scaled, side="right") - 1)
        s = max(0, min(len(cum) - 2, s))
        cl, ch = int(cum[s]), int(cum[s + 1])
        self.high = self.low + (rng * ch) // total - 1
        self.low = self.low + (rng * cl) // total
        while True:
            if self.high < self.HALF:
                pass
            elif self.low >= self.HALF:
                self.low -= self.HALF
                self.high -= self.HALF
                self.value -= self.HALF
            elif self.low >= self.QUARTER and self.high < self.THREEQ:
                self.low -= self.QUARTER
                self.high -= self.QUARTER
                self.value -= self.QUARTER
            else:
                break
            self.low = (self.low << 1) & self.MASK
            self.high = ((self.high << 1) | 1) & self.MASK
            self.value = ((self.value << 1) | self._next()) & self.MASK
        return s


def coder_selftests() -> List[Tuple[str, bool, str]]:
    """Coder-level gates: reference self-test, streaming==reference bit parity,
    and adversarial distributions (near-deterministic + full-vocab tables)."""
    out = []
    out.append(("coder.reference-selftest", nsc._selftest_coder(), "nsc._selftest_coder()"))

    rng = np.random.RandomState(7)
    V, n = 512, 400
    cums, syms = [], []
    for _ in range(n):
        p = rng.dirichlet(np.ones(V) * 0.05)
        cums.append(_quantize(p, total_bits=QUANT_BITS))
        syms.append(int(rng.randint(V)))
    triples = [(int(cums[t][syms[t]]), int(cums[t][syms[t] + 1]), int(cums[t][-1]))
               for t in range(n)]
    ref_bits = _RangeCoder().encode(triples)
    enc = RangeEncoder()
    for cl, ch, tot in triples:
        enc.encode(cl, ch, tot)
    payload, n_bits = enc.finish()
    same = (n_bits == len(ref_bits) and
            list(np.unpackbits(np.frombuffer(payload, dtype=np.uint8))[:n_bits]) == ref_bits)
    out.append(("coder.streaming==reference", same,
                f"{n_bits} bits, byte-level BPE-scale vocab"))

    dec = RangeDecoder(payload, n_bits)
    decoded = [dec.decode(c) for c in cums]
    out.append(("coder.streaming-roundtrip", decoded == syms, f"{n} symbols"))

    # adversarial: probability mass ~1 on one symbol, everything else at the
    # quantizer floor, full 8192-symbol tables (the real operating regime)
    V = 8192
    cums, syms = [], []
    for t in range(200):
        p = np.full(V, 1e-9, dtype=np.float64)
        p[t % V] = 1.0
        p /= p.sum()
        cums.append(_quantize(p, total_bits=QUANT_BITS))
        syms.append((t * 37) % V)          # often NOT the peak: worst case cost
    enc = RangeEncoder()
    for t in range(200):
        enc.encode(int(cums[t][syms[t]]), int(cums[t][syms[t] + 1]), int(cums[t][-1]))
    payload, n_bits = enc.finish()
    dec = RangeDecoder(payload, n_bits)
    ok = all(dec.decode(cums[t]) == syms[t] for t in range(200))
    out.append(("coder.adversarial-peaked", ok, "peaked V=8192 tables, off-peak symbols"))
    return out


# =============================================================================
#  Incremental classical features — exact replica of benchmark_enterprise.
#  compute_features(), but token-at-a-time so the DECODER can rebuild the
#  model inputs from nothing but its own decoded output. Uses exact packed
#  K-gram keys (deterministic across processes, unlike training's hash()).
# =============================================================================
class IncrementalFeatures:
    def __init__(self, o2: FastO2, K: int, cap: float):
        self.o2, self.K, self.cap, self.V = o2, K, float(cap), o2.V
        self.tokens: List[int] = []
        self.pos: Dict[int, int] = {}
        self.run = 0
        self.prev_mtop = 0

    def push(self, t: int) -> Tuple[int, float, int, float]:
        """Append token t; return (o2_top, o2_conf, m_top, m_conf) for its
        position — the features the model consumes to predict the NEXT token."""
        i = len(self.tokens)
        self.tokens.append(t)
        o2t, o2c = 0, 0.0
        if i >= 1:
            key = self.tokens[i - 1] * self.V + t
            idx = int(np.searchsorted(self.o2.ctx, key))
            idx = min(max(idx, 0), len(self.o2.ctx) - 1)
            if int(self.o2.ctx[idx]) == key:
                o2t, o2c = int(self.o2.top[idx]), float(self.o2.conf[idx])
        mt, mc = 0, 0.0
        if i >= self.K:
            h = 0
            for j in range(i - self.K + 1, i + 1):
                h = h * self.V + self.tokens[j]
            prev = self.pos.get(h)
            if prev is not None:               # prev+1 <= i always: stays causal
                mt = self.tokens[prev + 1]
                self.run = self.run + 1 if self.prev_mtop == t else 1
                mc = float(np.float32(min(self.run / self.cap, 0.99)))
            else:
                self.run = 0
            self.pos[h] = i
        self.prev_mtop = mt
        return o2t, o2c, mt, mc


# =============================================================================
#  Streaming model stepper — one token at a time, mirroring HybridLM.forward()
#  exactly (same modules, same op order). State: GRU hidden + per-layer KV
#  cache + RoPE position, all reset at every seg_len window boundary, matching
#  how training/eval batches present independent windows.
# =============================================================================
class HybridStepper:
    def __init__(self, model: nsc.HybridLM, cfg: Config, T: float):
        model.eval()
        self.m, self.cfg = model, cfg
        self.V = cfg.vocab_size
        self.T = float(T)
        self.logV = math.log(cfg.vocab_size)
        self.reset_window()

    def reset_window(self) -> None:
        self.hx = None
        self.kvs: List[Optional[tuple]] = [None] * self.cfg.n_layers
        self.pos = 0

    @torch.no_grad()
    def step(self, tok: int, o2t: int, o2c: float, mt: int, mc: float) -> np.ndarray:
        m, V = self.m, self.V
        x = torch.tensor([[tok]], dtype=torch.long)
        o2c_t = torch.tensor([[o2c]], dtype=torch.float32)
        mc_t = torch.tensor([[mc]], dtype=torch.float32)

        p0 = m.order0.to(torch.float32)[None, None]
        p1 = m.order1.to(torch.float32)[x]
        p2 = peaked_distribution(torch.tensor([[o2t]]), o2c_t, V)
        pm = peaked_distribution(torch.tensor([[mt]]), mc_t, V)
        o1_max = p1.max(dim=-1).values
        o1_ent = -(p1 * (p1 + m.eps).log()).sum(-1) / self.logV
        conf = torch.stack([o1_max, o1_ent, o2c_t, mc_t], dim=-1)

        xe = m.embed(x)
        gh, self.hx = m.local.gru(xe, self.hx)
        gh = m.local.film(gh, conf)
        resid_gru = m.local.to_resid(gh)

        if m.xformer is not None:
            h = xe
            for i, blk in enumerate(m.xformer.blocks):
                h, kv = blk(h, self.kvs[i], self.pos)
                self.kvs[i] = kv
            xh = m.xformer.lnf(h)
            xh = m.xformer.film(xh, conf)
            resid_xf = m.xformer.to_resid(xh)
            self.pos += 1
        else:                                    # Track B: recurrence-only
            xh = torch.zeros_like(xe)
            resid_xf = torch.zeros_like(resid_gru)

        ef = torch.stack([torch.full_like(o1_max, 0.5), o1_max, o2c_t, mc_t,
                          gh.abs().mean(-1), xh.abs().mean(-1)], dim=-1)
        w, g, _ = m.gate(gh, xh, ef)
        p_cls = (w[..., 0:1] * p0 + w[..., 1:2] * p1 +
                 w[..., 2:3] * p2 + w[..., 3:4] * pm)
        z = (p_cls + m.eps).log() + g[..., 0:1] * resid_gru + g[..., 1:2] * resid_xf
        if self.T != 1.0:
            z = z / self.T
        return torch.softmax(z[0, 0], dim=-1).numpy()


# =============================================================================
#  Artifact bundle: model + tokenizer + order-2 table, each content-hashed so
#  archives are cryptographically bound to the exact predictor that wrote them
#  (decoding with the wrong model would otherwise emit silent garbage).
# =============================================================================
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Artifacts:
    def __init__(self, export_dir=None, data_path=None, quiet=False):
        paths = locate_paths(export_dir, data_path)
        self.paths = paths
        if not quiet:
            print(f"[artifacts] checkpoint: {paths['best_ckpt']}")
            print(f"[artifacts] tokenizer : {paths['tokenizer']}")
            print(f"[artifacts] enwik8    : {paths['data']}")
        # weights_only=True → no pickle code execution from the checkpoint file
        state, cfg_dict, meta = load_checkpoint_meta(paths["best_ckpt"])
        self.model, self.cfg = build_model(state, cfg_dict)
        del state
        self.meta = meta
        self.model_sha = sha256_file(paths["best_ckpt"])

        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(paths["tokenizer"])
        self.tokenizer_sha = sha256_file(paths["tokenizer"])
        if self.tok.get_vocab_size() != self.cfg.vocab_size:
            raise ArtifactMismatch(
                f"tokenizer vocab {self.tok.get_vocab_size()} != cfg {self.cfg.vocab_size}")

        self.data_len = os.path.getsize(paths["data"])
        cache_dir = writable_dir(os.path.join(BASE_DIR, "results", "cache"))
        self.cache_dir = cache_dir
        self.cache_key = f"v{self.cfg.vocab_size}_n{self.data_len}"
        if nsc.corpus_layout(paths["data"]):     # mixed corpus: bind caches to the tokenizer
            self.cache_key += f"_t{self.tokenizer_sha[:12]}"
        o2_path = os.path.join(cache_dir, f"o2_{self.cache_key}.npz")
        if os.path.exists(o2_path):
            self.o2 = FastO2.load(o2_path, self.cfg.vocab_size)
        else:
            print("[artifacts] order-2 cache missing — rebuilding from the train split "
                  "(one-time, several minutes)")
            train_tokens = self._split_tokens("train")
            self.o2 = FastO2(self.cfg.vocab_size).fit(train_tokens)
            self.o2.save(o2_path)
        # content hash (file hashes of .npz are timestamp-dependent)
        self.o2_sha = sha256_bytes(self.o2.ctx.tobytes() + self.o2.top.tobytes()
                                   + self.o2.conf.tobytes())
        if not quiet:
            print(f"[artifacts] model step {meta['step']} val_bpb {meta['val_bpb']:.4f} | "
                  f"o2 contexts {len(self.o2.ctx):,}")

    def split_range(self, split: str) -> Tuple[int, int]:
        return nsc.split_ranges(self.paths["data"], self.cfg)[split]

    def read_slice(self, offset: int, length: int) -> bytes:
        with open(self.paths["data"], "rb") as f:
            f.seek(offset)
            return f.read(length)

    def _split_tokens(self, split: str) -> np.ndarray:
        p = os.path.join(self.cache_dir, f"tokens_{split}_{self.cache_key}.npy")
        if os.path.exists(p):
            return np.load(p)
        a, b = self.split_range(split)
        doms = nsc.layout_domains(self.paths["data"])
        if doms:    # mixed corpus: same per-domain token stream as the trainer
            toks, _ = nsc.encode_domains(self.tok, self.read_slice(a, b - a), doms[split])
        else:
            toks = encode_bytes_chunked(self.tok, self.read_slice(a, b - a))
        np.save(p, toks)
        return toks

    def val_eval_set(self):
        toks = self._split_tokens("val")
        p = os.path.join(self.cache_dir, f"feats_val_{self.cache_key}.npz")
        if os.path.exists(p):
            z = np.load(p)
            feats = {k: z[k] for k in z.files}
        else:
            feats = compute_features(toks, self.o2, self.cfg.order2_match_order,
                                     float(self.cfg.match_len_cap))
            np.savez_compressed(p, **feats)
        return toks, feats

    def header_artifacts(self) -> Dict:
        return {"model_sha256": self.model_sha, "tokenizer_sha256": self.tokenizer_sha,
                "o2_sha256": self.o2_sha, "model_step": self.meta["step"]}


# =============================================================================
#  Calibration ("if necessary"): fit temperature T on the VAL split (never on
#  the data being compressed), cache per model hash. |T-1| <= 0.005 ⇒ T = 1.
# =============================================================================
@torch.no_grad()
def measure_temperature(art: Artifacts, batches=8, batch_size=4, pos_per_seq=128,
                        seed=1234) -> Dict:
    toks, feats = art.val_eval_set()
    from dataclasses import asdict as _asdict
    ecfg = Config(**{**_asdict(art.cfg), "batch_size": batch_size})
    loader = nsc.make_loader(toks, feats, ecfg, shuffle=False)
    art.model.eval()
    rng = np.random.RandomState(seed)
    zs, ys = [], []
    for bi, batch in enumerate(loader):
        if bi >= batches:
            break
        out = art.model(batch, stage=2)
        z = out["z_final"].float()
        y = batch["target_ids"]
        B, L, V = z.shape
        idx = rng.choice(L, size=min(pos_per_seq, L), replace=False)
        zs.append(z[:, idx].reshape(-1, V).cpu())
        ys.append(y[:, idx].reshape(-1).cpu())
    z, y = torch.cat(zs), torch.cat(ys)
    art.model.eval()

    def nll(T):
        return torch.nn.functional.cross_entropy(z / T, y).item()

    lo, hi, gr = 0.5, 2.0, (math.sqrt(5) - 1) / 2
    a, b = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fa, fb = nll(a), nll(b)
    for _ in range(40):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - gr * (hi - lo)
            fa = nll(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + gr * (hi - lo)
            fb = nll(b)
        if hi - lo < 1e-3:
            break
    T = 0.5 * (lo + hi)
    return {"T_opt": float(T), "nll_before": float(nll(1.0)), "nll_after": float(nll(T)),
            "ece_before": float(nsc._ece(z, y)), "ece_after": float(nsc._ece(z / T, y)),
            "positions": int(len(y))}


def calibrate(art: Artifacts, args) -> Dict:
    if args.temperature is not None:
        return {"T": float(args.temperature), "source": "cli"}
    if args.no_calibration:
        return {"T": 1.0, "source": "disabled"}
    cache = os.path.join(args.out, "calibration.json")
    if os.path.exists(cache) and not args.recalibrate:
        d = json.load(open(cache))
        if d.get("model_sha256") == art.model_sha:
            d["source"] = "cache"
            return d
    print("[calibrate] measuring temperature on the val split ...")
    t0 = time.time()
    m = measure_temperature(art)
    T = m["T_opt"] if abs(m["T_opt"] - 1.0) > 0.005 else 1.0
    d = {"T": T, "source": "measured", "model_sha256": art.model_sha,
         "seconds": round(time.time() - t0, 1), **m}
    os.makedirs(args.out, exist_ok=True)
    json.dump(d, open(cache, "w"), indent=2)
    needed = "APPLIED" if T != 1.0 else "not needed (|T-1| <= 0.005)"
    print(f"[calibrate] T*={m['T_opt']:.4f}  nll {m['nll_before']:.4f}->{m['nll_after']:.4f}  "
          f"ECE {m['ece_before']:.4f}->{m['ece_after']:.4f}  ({d['seconds']}s) — {needed}")
    return d


# =============================================================================
#  Container format
#    MAGIC | u32 header_len | header JSON | payload | sha256(file so far)
#  The header binds: original length + SHA-256, payload SHA-256, artifact
#  hashes, exact-hex temperature, token counts. Optional HMAC-SHA256 over
#  (canonical header sans hmac) + payload when NSC_INTEGRITY_KEY is set —
#  plain hashes detect corruption; only the keyed MAC detects substitution.
# =============================================================================
def _canonical(header: Dict) -> bytes:
    h = {k: v for k, v in header.items() if k != "hmac_sha256"}
    return json.dumps(h, sort_keys=True, separators=(",", ":")).encode()


def write_container(path: str, header: Dict, payload: bytes) -> bytes:
    key = os.environ.get(HMAC_ENV)
    if key:
        header["hmac_sha256"] = hmac.new(key.encode(), _canonical(header) + payload,
                                         hashlib.sha256).hexdigest()
    hj = json.dumps(header, sort_keys=True).encode()
    body = MAGIC + len(hj).to_bytes(4, "little") + hj + payload
    blob = body + hashlib.sha256(body).digest()
    with open(path, "wb") as f:
        f.write(blob)
    return blob


def parse_container(blob: bytes, verify=True) -> Tuple[Dict, bytes]:
    if len(blob) < len(MAGIC) + 4 + 32 or blob[:len(MAGIC)] != MAGIC:
        raise FileCorrupt("bad magic or truncated file")
    if verify and hashlib.sha256(blob[:-32]).digest() != blob[-32:]:
        raise FileCorrupt("file digest mismatch (corrupted or tampered)")
    hlen = int.from_bytes(blob[len(MAGIC):len(MAGIC) + 4], "little")
    if hlen <= 0 or hlen > 1 << 20 or len(MAGIC) + 4 + hlen + 32 > len(blob):
        raise HeaderInvalid("header length out of bounds")
    try:
        header = json.loads(blob[len(MAGIC) + 4:len(MAGIC) + 4 + hlen])
    except Exception as e:
        raise HeaderInvalid(f"header not valid JSON: {e}")
    payload = blob[len(MAGIC) + 4 + hlen:-32]
    if verify:
        _validate_header(header, payload)
    return header, payload


def _validate_header(h: Dict, payload: bytes) -> None:
    try:
        if h["format"] != FORMAT_NAME or h["quant_bits"] != QUANT_BITS:
            raise HeaderInvalid("unknown format/version")
        n_tok, first, olen = int(h["n_tokens"]), int(h["first_token"]), int(h["original_len"])
        V, seg = int(h["vocab_size"]), int(h["seg_len"])
        nbits = int(h["n_payload_bits"])
        T = float.fromhex(h["temperature_hex"])
        for name in ("original_sha256", "payload_sha256"):
            if len(h[name]) != 64:
                raise HeaderInvalid(f"{name} malformed")
    except IntegrityError:
        raise
    except Exception as e:
        raise HeaderInvalid(f"missing/malformed header field: {e}")
    # bounds: byte-level BPE emits >= 1 byte per token, so n_tokens can never
    # exceed original_len — rejects headers built to force unbounded decodes
    if not (1 <= n_tok <= olen <= (1 << 31)):
        raise HeaderInvalid(f"token/length bounds violated (n_tokens={n_tok}, len={olen})")
    if not (0 <= first < V) or seg <= 0 or not (0.1 <= T <= 10.0):
        raise HeaderInvalid("first_token/seg_len/temperature out of range")
    if nbits > 8 * len(payload):
        raise HeaderInvalid("n_payload_bits exceeds payload")
    if h["payload_sha256"] != sha256_bytes(payload):
        raise FileCorrupt("payload digest mismatch")
    key = os.environ.get(HMAC_ENV)
    if key:
        # key configured => signatures are mandatory, so an attacker cannot
        # bypass authentication by simply stripping the hmac field
        if not h.get("hmac_sha256"):
            raise AuthFailure("archive is unsigned but an integrity key is configured")
        want = hmac.new(key.encode(), _canonical(h) + payload,
                        hashlib.sha256).hexdigest()
        if not hmac.compare_digest(want, h["hmac_sha256"]):
            raise AuthFailure("HMAC verification failed (wrong key or forged archive)")
    elif h.get("hmac_sha256"):
        print(f"[warn] archive is HMAC-signed but {HMAC_ENV} is not set — "
              "authenticity NOT verified")


# =============================================================================
#  Compress / decompress
# =============================================================================
def _progress(tag, i, n, t0, quiet, every=2048):
    if quiet or (i % every and i != n - 1):
        return
    dt = time.time() - t0
    rate = (i + 1) / max(dt, 1e-9)
    eta = (n - i - 1) / max(rate, 1e-9)
    sys.stderr.write(f"\r  [{tag}] {i + 1:,}/{n:,} tok  {rate:,.0f} tok/s  ETA {eta:,.0f}s ")
    if i == n - 1:
        sys.stderr.write("\n")
    sys.stderr.flush()


def compress_bytes(art: Artifacts, data: bytes, T: float, out_path: str,
                   quiet=False, collect_window0=False) -> Dict:
    tokens = nsc.encode_with(art.tok, data)
    n = len(tokens)
    if n < 2:
        raise ValueError("slice too small to code (need >= 2 tokens)")
    # tokenizer must invert exactly, otherwise the archive is corrupt by design
    if nsc.decode_with(art.tok, tokens.tolist()) != data:
        raise DecodedMismatch("tokenizer roundtrip failed on this input")

    inc = IncrementalFeatures(art.o2, art.cfg.order2_match_order, art.cfg.match_len_cap)
    stepper = HybridStepper(art.model, art.cfg, T)
    enc = RangeEncoder()
    seg = art.cfg.seg_len
    feats_log = np.zeros((n, 4), dtype=np.float64)   # for the feature-parity gate
    win0_z: List[np.ndarray] = []
    xent_bits = 0.0
    feat = inc.push(int(tokens[0]))
    feats_log[0] = feat
    t0 = time.time()
    for i in range(n - 1):
        if i % seg == 0:
            stepper.reset_window()
        probs = stepper.step(int(tokens[i]), *feat)
        if collect_window0 and i < seg:
            win0_z.append(probs.copy())
        cum = _quantize(probs, total_bits=QUANT_BITS)
        s = int(tokens[i + 1])
        enc.encode(int(cum[s]), int(cum[s + 1]), int(cum[-1]))
        xent_bits += -math.log2(max(float(probs[s]), 1e-12))
        feat = inc.push(s)
        feats_log[i + 1] = feat
        _progress("encode", i, n - 1, t0, quiet)
    payload, n_bits = enc.finish()
    enc_secs = time.time() - t0

    header = {
        "format": FORMAT_NAME, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_tokens": n, "first_token": int(tokens[0]),
        "seg_len": seg, "vocab_size": art.cfg.vocab_size, "quant_bits": QUANT_BITS,
        "temperature_hex": float(T).hex(), "n_payload_bits": n_bits,
        "original_len": len(data), "original_sha256": sha256_bytes(data),
        "payload_sha256": sha256_bytes(payload),
        "artifacts": art.header_artifacts(),
        "env": {"torch": torch.__version__, "numpy": np.__version__,
                "threads": torch.get_num_threads(), "platform": platform.platform()},
    }
    blob = write_container(out_path, header, payload)
    stats = {
        "archive": out_path, "n_tokens": n, "original_len": len(data),
        "payload_bytes": len(payload), "container_bytes": len(blob),
        "bpb_payload": 8.0 * len(payload) / len(data),
        "bpb_container": 8.0 * len(blob) / len(data),
        "bpb_model_xent": xent_bits / len(data),
        "encode_secs": round(enc_secs, 1), "tok_per_s": round((n - 1) / max(enc_secs, 1e-9)),
        "tokens": tokens, "feats_log": feats_log, "win0_z": win0_z,
    }
    return stats


def decompress_archive(art: Artifacts, archive: str, quiet=False,
                       check_artifacts=True) -> Tuple[bytes, Dict]:
    with open(archive, "rb") as f:
        blob = f.read()
    header, payload = parse_container(blob)
    if check_artifacts:
        ha = header.get("artifacts", {})
        ours = art.header_artifacts()
        for k in ("model_sha256", "tokenizer_sha256", "o2_sha256"):
            if ha.get(k) != ours[k]:
                raise ArtifactMismatch(
                    f"archive was written with a different {k.split('_')[0]} "
                    f"({ha.get(k, '?')[:12]}… != {ours[k][:12]}…) — decoding would "
                    f"produce silent garbage; pass --ignore-artifact-hash to force")
    if int(header["vocab_size"]) != art.cfg.vocab_size or \
            int(header["seg_len"]) != art.cfg.seg_len:
        raise ArtifactMismatch("archive vocab/seg_len do not match the loaded config")
    T = float.fromhex(header["temperature_hex"])
    n, seg = int(header["n_tokens"]), int(header["seg_len"])
    # decode with the encoder's thread count: identical BLAS reduction order is
    # part of the format's bit-determinism contract on a given machine
    hdr_threads = int(header.get("env", {}).get("threads", torch.get_num_threads()))
    torch.set_num_threads(min(max(hdr_threads, 1), 64))

    inc = IncrementalFeatures(art.o2, art.cfg.order2_match_order, art.cfg.match_len_cap)
    stepper = HybridStepper(art.model, art.cfg, T)
    dec = RangeDecoder(payload, int(header["n_payload_bits"]))
    toks = [int(header["first_token"])]
    feat = inc.push(toks[0])
    t0 = time.time()
    for i in range(n - 1):
        if i % seg == 0:
            stepper.reset_window()
        probs = stepper.step(toks[i], *feat)
        cum = _quantize(probs, total_bits=QUANT_BITS)
        s = dec.decode(cum)
        toks.append(s)
        feat = inc.push(s)
        _progress("decode", i, n - 1, t0, quiet)
    dec_secs = time.time() - t0

    try:
        data = nsc.decode_with(art.tok, toks)
    except (UnicodeEncodeError, ValueError, OverflowError) as e:
        raise DecodedMismatch(
            f"decoded token stream does not detokenize to bytes — data corrupted ({e})")
    stats = {"decode_secs": round(dec_secs, 1),
             "tok_per_s": round((n - 1) / max(dec_secs, 1e-9)),
             "header": header}
    if len(data) != int(header["original_len"]) or \
            sha256_bytes(data) != header["original_sha256"]:
        raise DecodedMismatch(
            f"decoded output does not match the archived checksum "
            f"(len {len(data)} vs {header['original_len']}) — data corrupted")
    return data, stats


# =============================================================================
#  Diagnostics & security suite
# =============================================================================
def feature_parity_gate(art: Artifacts, tokens: np.ndarray, feats_log: np.ndarray):
    ref = compute_features(tokens, art.o2, art.cfg.order2_match_order,
                           float(art.cfg.match_len_cap))
    got_top = feats_log[:, 0].astype(np.int64)
    got_o2c = feats_log[:, 1].astype(np.float32)
    got_mt = feats_log[:, 2].astype(np.int64)
    got_mc = feats_log[:, 3].astype(np.float32)
    ok = (np.array_equal(got_top, ref["order2_top"]) and
          np.array_equal(got_o2c, ref["order2_conf"]) and
          np.array_equal(got_mt, ref["match_top"]) and
          np.array_equal(got_mc, ref["match_conf"]))
    detail = f"{len(tokens):,} positions, incremental == vectorized"
    if not ok:
        for name, a, b in (("order2_top", got_top, ref["order2_top"]),
                           ("order2_conf", got_o2c, ref["order2_conf"]),
                           ("match_top", got_mt, ref["match_top"]),
                           ("match_conf", got_mc, ref["match_conf"])):
            bad = np.nonzero(a != b)[0]
            if len(bad):
                detail = f"first divergence: {name}[{bad[0]}] {a[bad[0]]} != {b[bad[0]]}"
                break
    return ok, detail


@torch.no_grad()
def parity_diagnostic(art: Artifacts, tokens: np.ndarray, win0_z: List[np.ndarray],
                      T: float) -> Dict:
    """Batched (training/eval path) vs streaming distributions on window 0.
    Informational: OUR archive uses the streaming path on both sides, but the
    Rust plan (batched encode + streaming decode) needs this gap to be zero."""
    seg = art.cfg.seg_len
    L = min(len(win0_z), seg, len(tokens) - 1)
    if L < 2:
        return {"skipped": True}
    feats = compute_features(tokens[:L], art.o2, art.cfg.order2_match_order,
                             float(art.cfg.match_len_cap))
    batch = {
        "input_ids": torch.from_numpy(tokens[:L].astype(np.int64))[None],
        "target_ids": torch.from_numpy(tokens[1:L + 1].astype(np.int64))[None],
        "order2_top": torch.from_numpy(feats["order2_top"][:L])[None],
        "order2_conf": torch.from_numpy(feats["order2_conf"][:L])[None],
        "match_top": torch.from_numpy(feats["match_top"][:L])[None],
        "match_conf": torch.from_numpy(feats["match_conf"][:L])[None],
    }
    art.model.eval()
    z = art.model(batch, stage=2)["z_final"][0, :L].float() / T
    pb = torch.softmax(z, dim=-1).numpy()
    ps = np.stack(win0_z[:L])
    max_dp = float(np.abs(pb - ps).max())
    diff_tables = sum(
        not np.array_equal(_quantize(pb[i], QUANT_BITS), _quantize(ps[i], QUANT_BITS))
        for i in range(L))
    return {"positions": L, "max_abs_prob_delta": max_dp,
            "quantized_tables_differing": int(diff_tables)}


def _rebuild_tampered(blob: bytes, mutate_header=None, mutate_payload=None) -> bytes:
    """Attacker model: modify fields AND recompute the outer digests they can
    recompute (file digest + payload sha). Detection must then come from the
    inner bindings (original sha / artifact hashes / HMAC)."""
    header, payload = parse_container(blob, verify=False)
    if mutate_payload:
        payload = mutate_payload(payload)
        header["payload_sha256"] = sha256_bytes(payload)
    if mutate_header:
        mutate_header(header)
    header.pop("hmac_sha256", None)          # attacker has no key
    hj = json.dumps(header, sort_keys=True).encode()
    body = MAGIC + len(hj).to_bytes(4, "little") + hj + payload
    return body + hashlib.sha256(body).digest()


def security_suite(art: Artifacts, tiny: bytes, T: float, outdir: str,
                   extra_args: Optional[List[str]] = None) -> List[Tuple[str, bool, str]]:
    gates = []
    tiny_path = os.path.join(outdir, "security_probe.syz")
    st = compress_bytes(art, tiny, T, tiny_path, quiet=True)
    blob = open(tiny_path, "rb").read()
    n_tok = st["n_tokens"]

    def expect(name, mutated_blob, kinds, detail):
        p = os.path.join(outdir, "tampered.syz")
        with open(p, "wb") as f:
            f.write(mutated_blob)
        try:
            decompress_archive(art, p, quiet=True)
            gates.append((name, False, "tampering was NOT detected"))
        except IntegrityError as e:
            gates.append((name, e.kind in kinds, f"{detail} -> rejected [{e.kind}]"))
        except Exception as e:  # any crash is a controlled-failure violation
            gates.append((name, False, f"uncontrolled error: {type(e).__name__}: {e}"))
        finally:
            os.path.exists(p) and os.remove(p)

    # 1. random bit flip in the payload region
    mid = len(MAGIC) + 4 + len(blob) // 2
    flipped = bytearray(blob)
    flipped[mid] ^= 0x10
    expect("security.bit-flip", bytes(flipped), {"file-corrupt"}, "1 bit flipped")

    # 2. truncation
    expect("security.truncation", blob[:len(blob) // 2], {"file-corrupt", "header-invalid"},
           "file cut in half")

    # 3. header tamper with digests fixed up: n_tokens stretched — the decode
    #    must run bounded and be caught by the original-sha binding
    t3 = _rebuild_tampered(blob, mutate_header=lambda h: h.update(
        n_tokens=min(int(h["n_tokens"]) + 7, int(h["original_len"]))))
    expect("security.header-tamper", t3, {"decoded-mismatch", "auth-failure"},
           "n_tokens+7, digests recomputed")

    # 4. wrong-model guard: decoding with mismatched artifact hashes must refuse
    t4 = _rebuild_tampered(blob, mutate_header=lambda h: h["artifacts"].update(
        model_sha256="0" * 64))
    expect("security.wrong-model-refusal", t4, {"artifact-mismatch", "auth-failure"},
           "model hash forged")

    # 5. garbage payload with valid digests: decode must terminate (bounded by
    #    n_tokens) and be detected — never returned as data
    rnd = np.random.RandomState(3).randint(0, 256, size=st["payload_bytes"],
                                           dtype=np.uint8).tobytes()
    t5 = _rebuild_tampered(blob, mutate_payload=lambda p: rnd)
    expect("security.garbage-payload", t5, {"decoded-mismatch", "auth-failure"},
           f"{st['payload_bytes']}B random payload, bounded {n_tok}-token decode")

    # 6. fresh-process decode: proves the archive + artifacts alone reproduce
    #    the bytes (cross-process determinism; no hidden in-process state)
    cmd = [sys.executable, os.path.abspath(__file__), "--mode", "decompress",
           "--archive", tiny_path, "--out", outdir, "--no-progress"]
    if extra_args:
        cmd += extra_args
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                       cwd=BASE_DIR, env={**os.environ})
    ok = r.returncode == 0
    tail = (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr) else ""
    gates.append(("security.subprocess-decode", ok,
                  "fresh python process, decode+sha verify" if ok else tail))

    if os.environ.get(HMAC_ENV):
        header, _ = parse_container(blob, verify=False)
        gates.append(("security.hmac-present", "hmac_sha256" in header,
                      "archives are HMAC-signed when key is set"))
        # forged signature: attacker without the key cannot re-sign
        forged = dict(header)
        forged["hmac_sha256"] = "0" * 64
        hj = json.dumps(forged, sort_keys=True).encode()
        _, payload = parse_container(blob, verify=False)
        body = MAGIC + len(hj).to_bytes(4, "little") + hj + payload
        expect("security.hmac-forged", body + hashlib.sha256(body).digest(),
               {"auth-failure"}, "signature replaced without key")
    return gates


# =============================================================================
#  Main
# =============================================================================
def _fmt_gates(gates) -> str:
    w = max(len(g[0]) for g in gates)
    lines = []
    for name, ok, detail in gates:
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {name:<{w}}  {detail}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["roundtrip", "compress", "decompress", "selftest"],
                    default="roundtrip")
    ap.add_argument("--bytes", type=int, default=65536,
                    help="slice size; 0 = the whole chosen split (default 65536)")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test",
                    help="which held-out region of enwik8 to compress (default test)")
    ap.add_argument("--offset", type=int, default=None,
                    help="absolute byte offset into enwik8 (overrides --split)")
    ap.add_argument("--input", type=str, default=None,
                    help="compress this file instead of an enwik8 slice")
    ap.add_argument("--archive", type=str, default=None, help="archive for --mode decompress")
    ap.add_argument("--out", type=str, default=os.path.join(BASE_DIR, "results", "integrity"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--no-calibration", action="store_true")
    ap.add_argument("--recalibrate", action="store_true")
    ap.add_argument("--no-security", action="store_true")
    ap.add_argument("--no-parity", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--quick", action="store_true", help="8 KiB smoke run")
    ap.add_argument("--ignore-artifact-hash", action="store_true")
    ap.add_argument("--export-dir", type=str, default=None)
    ap.add_argument("--data", type=str, default=None)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, args.threads))
    if args.quick:
        args.bytes = 8192
        args.no_parity = True
    os.makedirs(args.out, exist_ok=True)
    gates: List[Tuple[str, bool, str]] = []
    report: Dict = {"mode": args.mode, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "argv": sys.argv[1:]}

    # ---- gate 0: entropy coder ------------------------------------------------
    gates += coder_selftests()
    if args.mode == "selftest":
        print("\n=== coder self-tests ===\n" + _fmt_gates(gates))
        ok = all(g[1] for g in gates)
        print(f"\nSELFTEST: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1

    art = Artifacts(args.export_dir, args.data, quiet=False)
    report["artifacts"] = art.header_artifacts()
    report["checkpoint"] = {k: art.meta[k] for k in ("step", "val_bpb", "path")}

    # ---- decompress-only mode ---------------------------------------------------
    if args.mode == "decompress":
        if not args.archive:
            ap.error("--mode decompress requires --archive")
        try:
            data, st = decompress_archive(art, args.archive, quiet=args.no_progress,
                                          check_artifacts=not args.ignore_artifact_hash)
        except IntegrityError as e:
            print(f"DECOMPRESS: REJECTED [{e.kind}] {e}")
            return 1
        outp = args.archive + ".decoded"
        with open(outp, "wb") as f:
            f.write(data)
        print(f"DECOMPRESS: OK — {len(data):,} bytes, sha256 verified against header, "
              f"{st['tok_per_s']:,} tok/s -> {outp}")
        return 0

    # ---- choose the input slice -------------------------------------------------
    if args.input:
        with open(args.input, "rb") as f:
            data = f.read() if args.bytes == 0 else f.read(args.bytes)
        src = args.input
    else:
        a, b = art.split_range(args.split)
        off = args.offset if args.offset is not None else a
        length = (b - off) if args.bytes == 0 else min(args.bytes, art.data_len - off)
        data = art.read_slice(off, length)
        src = f"enwik8[{off}:{off + length}] ({args.split} split)"
    print(f"[input] {src}: {len(data):,} bytes, sha256 {sha256_bytes(data)[:16]}…")

    # ---- calibration ("if necessary") -------------------------------------------
    cal = calibrate(art, args)
    T = float(cal["T"])
    report["calibration"] = cal

    # ---- compress -----------------------------------------------------------------
    name = f"{os.path.basename(src).split('[')[0]}_{len(data)}B.syz" if args.input \
        else f"enwik8_{args.split}_{len(data)}B.syz"
    archive = os.path.join(args.out, name)
    print(f"[compress] {len(data):,} bytes -> {archive}  (T={T:g}, "
          f"threads={torch.get_num_threads()})")
    st = compress_bytes(art, data, T, archive, quiet=args.no_progress,
                        collect_window0=not args.no_parity)
    gz = 8.0 * len(gzip.compress(data, 9)) / len(data)
    xz = 8.0 * len(lzma.compress(data, preset=9)) / len(data)
    print(f"[compress] {st['container_bytes']:,} B container  "
          f"payload {st['bpb_payload']:.4f} bpb (model xent {st['bpb_model_xent']:.4f}, "
          f"quantization +{st['bpb_payload'] - st['bpb_model_xent']:.4f})  |  "
          f"gzip-9 {gz:.4f}  xz-9 {xz:.4f}  |  {st['tok_per_s']:,} tok/s")
    report["compress"] = {k: v for k, v in st.items()
                          if k not in ("tokens", "feats_log", "win0_z")}
    report["compress"].update({"gzip9_bpb": gz, "xz9_bpb": xz, "temperature": T})
    gates.append(("compress.quantization-overhead",
                  st["bpb_payload"] - st["bpb_model_xent"] < 0.02,
                  f"+{st['bpb_payload'] - st['bpb_model_xent']:.5f} bpb over model entropy"))

    # ---- feature decodability gate --------------------------------------------------
    ok, detail = feature_parity_gate(art, st["tokens"], st["feats_log"])
    gates.append(("features.incremental==vectorized", ok, detail))

    # ---- batched vs streaming parity diagnostic (informational) ---------------------
    if not args.no_parity:
        par = parity_diagnostic(art, st["tokens"], st["win0_z"], T)
        report["parity_batched_vs_streaming"] = par
        if not par.get("skipped"):
            print(f"[parity] batched-vs-streaming on window 0: max|Δp|="
                  f"{par['max_abs_prob_delta']:.2e}, "
                  f"{par['quantized_tables_differing']}/{par['positions']} quantized "
                  f"tables differ (Rust batched-encode plan must close this gap)")

    if args.mode == "compress":
        print("\n" + _fmt_gates(gates))
        ok = all(g[1] for g in gates)
        report["gates"] = [{"name": n2, "pass": bool(p), "detail": d} for n2, p, d in gates]
        report["overall_pass"] = ok
        json.dump(report, open(os.path.join(args.out, "integrity_report.json"), "w"),
                  indent=2, default=str)
        return 0 if ok else 1

    # ---- decompress + byte-identity verdict -------------------------------------------
    print(f"[decompress] {archive} (decoder sees only the archive + artifacts)")
    try:
        decoded, dst = decompress_archive(art, archive, quiet=args.no_progress,
                                          check_artifacts=not args.ignore_artifact_hash)
        identical = decoded == data           # full memcmp, not just the sha
        gates.append(("roundtrip.byte-identical", identical,
                      f"{len(data):,} bytes, sha256 + memcmp"
                      if identical else "sha matched header but bytes differ from source"))
        report["decompress"] = {"decode_secs": dst["decode_secs"],
                                "tok_per_s": dst["tok_per_s"]}
        print(f"[decompress] OK — {dst['tok_per_s']:,} tok/s")
    except IntegrityError as e:
        gates.append(("roundtrip.byte-identical", False, f"[{e.kind}] {e}"))

    # ---- security suite ------------------------------------------------------------
    if not args.no_security:
        tiny_len = min(4096, len(data))
        print(f"[security] tamper suite on a {tiny_len}-byte probe archive ...")
        extra = []
        if args.export_dir:
            extra += ["--export-dir", args.export_dir]
        if args.data:
            extra += ["--data", args.data]
        gates += security_suite(art, data[:tiny_len], T, args.out, extra)

    # ---- verdict ---------------------------------------------------------------------
    print("\n=== integrity & security gates ===\n" + _fmt_gates(gates))
    ok = all(g[1] for g in gates)
    report["gates"] = [{"name": n, "pass": bool(p), "detail": d} for n, p, d in gates]
    report["overall_pass"] = ok
    rp = os.path.join(args.out, "integrity_report.json")
    json.dump(report, open(rp, "w"), indent=2, default=str)
    print(f"\nOVERALL: {'PASS — compression is lossless and tamper-evident' if ok else 'FAIL'}"
          f"\nreport: {rp}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
