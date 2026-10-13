# -*- coding: utf-8 -*-
# =============================================================================
#  ENTERPRISE BENCHMARK & STOPPING-POLICY HARNESS  (standalone, training-free)
# =============================================================================
#  Professional evaluation of an exported neuro-symbolic compressor checkpoint
#  (results/export/best.pt + tokenizer.json) WITHOUT re-running training.
#
#  Test suites (industry-standard compressor + ML metrics):
#    A. Compression quality   : bits/byte (raw + temperature-calibrated),
#                               perplexity, top-1/top-5 accuracy, projected
#                               compressed size, ratio vs gzip/bzip2/xz(/zstd).
#    B. Correctness gates     : arithmetic-coder self-test, byte-exact
#                               encode->decode roundtrip on REAL model
#                               distributions, tokenizer roundtrip, streaming
#                               reference-decoder parity.  Hard pass/fail.
#    C. Calibration           : optimal temperature T*, NLL before/after, ECE.
#    D. Performance           : throughput (tok/s, MB/s), latency p50/p95,
#                               parameter count, artifact sizes, peak memory.
#    E. Robustness (stress)   : repetitive / random / structured-text inputs.
#    F. Stopping policy       : overtraining check (latest vs best checkpoint),
#                               marginal-gain extrapolation to the end of the
#                               curriculum, and a CONTINUE / STOP verdict.
#
#  Usage:
#      python benchmark_enterprise.py                  # evaluate results/export
#      python benchmark_enterprise.py --full           # no batch caps (slow)
#      python benchmark_enterprise.py --train --budget 80   # train first, then bench
#
#  SKIP_TRAINING=True (default) evaluates the exported previous-session result
#  found under results/.  Set it to False (or pass --train) to run a training
#  session first via run_budget() and benchmark its freshest checkpoint.
#
#  Exit code is non-zero if any hard correctness gate fails, so this file can
#  run as a CI job.
# =============================================================================
from __future__ import annotations

import os
import sys
import json
import time
import math
import gzip
import hashlib
import bz2
import lzma
import argparse
import platform
import resource
from dataclasses import asdict

import numpy as np
import torch

import neuro_symbolic_compressor as nsc
from neuro_symbolic_compressor import (
    Config, HybridLM, make_loader, move_batch, count_params,
    _xla_safe_load, _selftest_coder, _quantize, _RangeCoder,
    encode_with, decode_with, reference_decoder_check,
)

# =============================================================================
#  CONFIG — edit here (every value can also be overridden on the CLI)
# =============================================================================
SKIP_TRAINING = True          # True: benchmark results/export only. False: train first.

# ---- stopping policy (Suite F): when is the model "done"? ------------------
# GOAL_TEST_BPB is the statistically-projected BEST achievable calibrated test
# bits/byte (== bits/char on enwik8) for this architecture at curriculum end
# (step 160k).  Ensemble of decay models (linear-rate-decay, power laws
# alpha=0.2..1.0, cosine-LR-integral) fitted to the recorded checkpoints
# (val 1.3100@67.3k -> 1.3071@72k, cal-test 1.2693@72k, 2026-07-06) projects a
# final calibrated test bpb of 1.2437 +/- 0.0035; 1.245 is the ensemble median.
# Training continues across sessions until this value is CONFIRMED, then
# run_budget() auto-stops (keep in sync with
# neuro_symbolic_compressor.GOAL_TEST_BPB).
GOAL_TEST_BPB   = 1.245   # absolute goal: stop sessions once calibrated test bpb <= this
PLATEAU_ABS_BPB = 0.003   # stop when a session (~30k steps) gains less val bpb than this...
PLATEAU_REL     = 0.0025  # ...or less than 0.25% relative (whichever is larger)
OVERFIT_TOL     = 0.005   # latest-vs-best val bpb regression (abs) that flags overtraining

# ---- evaluation budget (CPU-friendly defaults; --full removes the caps) ----
EVAL_BATCH_SIZE   = 4     # sequences per eval batch (smaller = less RAM)
TEST_MAX_BATCHES  = 48    # test-split batches  (48 * 4 * 1024 ≈ 196k tokens ≈ 0.6 MB)
VAL_MAX_BATCHES   = 24    # val-split batches for calibration + overtraining check
CALIB_BATCHES     = 8     # batches whose logits feed temperature search / ECE
CALIB_POS_PER_SEQ = 128   # random positions sampled per sequence (bounds logit RAM)
STRESS_MAX_BATCHES = 4    # batches per stress case
REF_DECODER_N     = 96    # tokens for the sequential reference-decoder parity check

# ---- training mode (only used when SKIP_TRAINING=False / --train) ----------
TRAIN_BUDGET_H  = None    # total cross-session budget (None -> run_budget prompts/loads)
SESSION_CAP_H   = 11.5

# Notebook-safe: __file__ is undefined when these sections run as ipynb cells.
BASE_DIR = (os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (this file is in python/)
            if "__file__" in globals() else os.getcwd())


# =============================================================================
#  Artifact / data discovery
# =============================================================================
def locate_paths(export_dir=None, data_path=None):
    """Resolve exported-checkpoint dir + enwik8 path. Prefers results/ (previous
    session results), then the live export/ and data/ dirs, then any attached
    Kaggle input dataset."""
    import glob as _glob
    exp_candidates = ([export_dir] if export_dir else []) + [
        os.path.join(BASE_DIR, "results", "export"),
        os.path.join(BASE_DIR, "export"),
    ] + sorted(_glob.glob("/kaggle/input/*/export")) \
      + sorted(_glob.glob("/kaggle/input/*"))
    exp = next((d for d in exp_candidates
                if d and os.path.exists(os.path.join(d, "tokenizer.json"))), None)
    if exp is None:
        raise FileNotFoundError(f"no export dir with tokenizer.json in {exp_candidates}")

    data_candidates = ([data_path] if data_path else []) + [
        os.path.join(BASE_DIR, "results", "data", "enwik8"),
        os.path.join(BASE_DIR, "data", "enwik8"),
    ] + sorted(_glob.glob("/kaggle/input/*/enwik8")) \
      + sorted(_glob.glob("/kaggle/input/*/data/enwik8"))
    data = next((p for p in data_candidates
                 if p and os.path.exists(p) and os.path.getsize(p) > 90_000_000), None)
    if data is None:
        # SKIP_TRAINING runs never reach the trainer's downloader, so fetch here.
        print(f"[data] enwik8 not found in {data_candidates} — downloading...")
        data = nsc.download_enwik8(writable_dir(os.path.join(BASE_DIR, "data")))

    best = os.path.join(exp, "best.pt")
    if not os.path.exists(best):
        best = nsc._find_pt_file(exp)
    lasts = sorted(
        (p for p in (os.path.join(exp, f) for f in os.listdir(exp))
         if p.endswith(".last")), key=os.path.getmtime)
    return dict(export_dir=exp, data=data, best_ckpt=best,
                last_ckpt=lasts[-1] if lasts else None,
                tokenizer=os.path.join(exp, "tokenizer.json"))


def writable_dir(preferred: str) -> str:
    """Return `preferred` if it can be written to, else ./benchmark_out.
    Kaggle input datasets are read-only, so reports/caches fall back to the
    working directory there."""
    try:
        os.makedirs(preferred, exist_ok=True)
        if os.access(preferred, os.W_OK):
            return preferred
    except OSError:
        pass
    fb = os.path.join(os.getcwd(), "benchmark_out")
    os.makedirs(fb, exist_ok=True)
    return fb


def load_checkpoint_meta(path):
    """Load a checkpoint, returning (state_dict, meta). Handles both the
    weights-only best.pt format and the full-resume .last format."""
    try:
        ck = _xla_safe_load(path, map_location="cpu", weights_only=True)
    except Exception:
        ck = _xla_safe_load(path, map_location="cpu", weights_only=False)
    meta = {"path": path, "size_mb": os.path.getsize(path) / 2**20,
            "step": int(ck.get("step", -1)),
            "val_bpb": float(ck.get("val_bpb", ck.get("best", float("nan")))),
            "stage": ck.get("stage"), "patience": ck.get("patience"),
            "reason": ck.get("reason")}
    return ck["model"], ck["cfg"], meta


class _UniformClassical:
    """Placeholder for HybridLM construction; the real order0/order1 tables are
    buffers inside the checkpoint state_dict and overwrite these on load."""

    def __init__(self, V):
        self.order0 = np.full(V, 1.0 / V, dtype=np.float32)
        self.order1 = np.full((V, V), 1.0 / V, dtype=np.float32)


def build_model(state_dict, cfg_dict):
    cfg = Config(**cfg_dict)
    model = HybridLM(cfg, _UniformClassical(cfg.vocab_size))
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg


# =============================================================================
#  Fast classical features (vectorized rebuild of ClassicalStats.features)
# =============================================================================
#  The training loop fed the model order-2 and match-model (top, conf) features
#  computed by ClassicalStats.  Rebuilding that dict-of-dicts on this machine
#  would need several GB of RAM, so this is a numpy re-implementation with the
#  same semantics: order-2 lookups are exact (tie-break may differ on equal
#  counts); the match model uses exact K-gram integer keys instead of a 31-bit
#  truncated hash (strictly fewer collisions than training, effect ~0).
# =============================================================================
class FastO2:
    def __init__(self, V):
        self.V = V
        self.ctx = self.top = self.conf = None

    def fit(self, tokens: np.ndarray):
        V = self.V
        t = tokens.astype(np.int64)
        codes = (t[:-2] * V + t[1:-1]) * V + t[2:]      # (ctx a*V+b) * V + next
        uniq, counts = np.unique(codes, return_counts=True)
        ctx, nxt = uniq // V, uniq % V
        starts = np.flatnonzero(np.r_[True, ctx[1:] != ctx[:-1]])
        totals = np.add.reduceat(counts, starts)
        order = np.lexsort((nxt, -counts, ctx))          # ctx asc, count desc
        ctx_s, nxt_s, cnt_s = ctx[order], nxt[order], counts[order]
        s2 = np.flatnonzero(np.r_[True, ctx_s[1:] != ctx_s[:-1]])
        self.ctx = ctx_s[s2]
        self.top = nxt_s[s2].astype(np.int64)
        self.conf = (cnt_s[s2] / totals).astype(np.float32)
        return self

    def save(self, path):
        np.savez_compressed(path, ctx=self.ctx, top=self.top, conf=self.conf)

    @classmethod
    def load(cls, path, V):
        z = np.load(path)
        o = cls(V)
        o.ctx, o.top, o.conf = z["ctx"], z["top"], z["conf"]
        return o

    def lookup(self, keys: np.ndarray):
        """Vectorized (top, conf) for an array of a*V+b context keys."""
        idx = np.searchsorted(self.ctx, keys)
        idx = np.clip(idx, 0, len(self.ctx) - 1)
        hit = self.ctx[idx] == keys
        top = np.where(hit, self.top[idx], 0)
        conf = np.where(hit, self.conf[idx], 0.0).astype(np.float32)
        return top, conf


def compute_features(tokens: np.ndarray, o2: FastO2, K: int, cap: float):
    """Aligned per-position features, mirroring ClassicalStats.features()."""
    V = o2.V
    t = tokens.astype(np.int64)
    n = len(t)
    o2_top = np.zeros(n, dtype=np.int64)
    o2_conf = np.zeros(n, dtype=np.float32)
    if n >= 2:  # order-2: context (t[i-1], t[i]) predicts i+1, defined for i>=1
        keys = t[:-1] * V + t[1:]
        top, conf = o2.lookup(keys)
        o2_top[1:], o2_conf[1:] = top, conf

    m_top = np.zeros(n, dtype=np.int64)
    m_conf = np.zeros(n, dtype=np.float32)
    if n > K:  # match model: exact K-gram key ending at i, defined for i>=K
        kg = np.zeros(n - K + 1, dtype=np.int64)   # packed exact K-gram keys
        for j in range(K):
            kg = kg * V + t[j:n - K + 1 + j]
        pos: dict = {}
        run = 0
        for i in range(K, n):
            h = int(kg[i - K + 1])
            prev = pos.get(h)
            if prev is not None and prev + 1 < n:
                m_top[i] = int(t[prev + 1])
                run = run + 1 if m_top[i - 1] == t[i] else 1
                m_conf[i] = min(run / cap, 0.99)
            else:
                run = 0
            pos[h] = i
    return dict(order2_top=o2_top, order2_conf=o2_conf,
                match_top=m_top, match_conf=m_conf)


def encode_bytes_chunked(tok, data: bytes, chunk: int = 1 << 22) -> np.ndarray:
    """Chunked BPE encode (bounded RAM). Chunks split on newline boundaries so
    byte-level BPE tokenization matches the unchunked result."""
    if len(data) <= chunk:
        return encode_with(tok, data)
    parts, i = [], 0
    while i < len(data):
        j = min(len(data), i + chunk)
        if j < len(data):
            k = data.rfind(b"\n", i, j)
            if k > i:
                j = k + 1
        parts.append(encode_with(tok, data[i:j]))
        i = j
    return np.concatenate(parts)


def prepare_eval_data(tok, cfg: Config, data_path: str, cache_dir: str):
    """Split enwik8 exactly like training (train/val/test fractions), rebuild
    the order-2 table from the TRAIN split, and compute cached features for the
    val/test splits only (train features are not needed for evaluation)."""
    os.makedirs(cache_dir, exist_ok=True)
    with open(data_path, "rb") as f:
        data = f.read()
    n = len(data)
    rng = nsc.split_ranges(data_path, cfg)   # sidecar layout for a mixed corpus
    byts = {k: data[a:b] for k, (a, b) in rng.items()}
    del data

    key = f"v{cfg.vocab_size}_n{n}"
    if nsc.corpus_layout(data_path):         # mixed corpus: bind caches to the tokenizer
        key += "_t" + hashlib.sha256(tok.to_str().encode()).hexdigest()[:12]
    toks = {}
    for split in ("train", "val", "test"):
        p = os.path.join(cache_dir, f"tokens_{split}_{key}.npy")
        if os.path.exists(p):
            toks[split] = np.load(p)
            print(f"  [cache] tokens {split}: HIT ({len(toks[split]):,})")
        else:
            t0 = time.time()
            doms = nsc.layout_domains(data_path)
            if doms:    # mixed corpus: same per-domain token stream as the trainer
                toks[split], _ = nsc.encode_domains(tok, byts[split], doms[split])
            else:
                toks[split] = encode_bytes_chunked(tok, byts[split])
            np.save(p, toks[split])
            print(f"  [cache] tokens {split}: encoded {len(toks[split]):,} "
                  f"({time.time()-t0:.0f}s)")

    o2_path = os.path.join(cache_dir, f"o2_{key}.npz")
    if os.path.exists(o2_path):
        o2 = FastO2.load(o2_path, cfg.vocab_size)
        print(f"  [cache] order-2 table: HIT ({len(o2.ctx):,} contexts)")
    else:
        t0 = time.time()
        o2 = FastO2(cfg.vocab_size).fit(toks["train"])
        o2.save(o2_path)
        print(f"  [cache] order-2 table: fit on train "
              f"({len(o2.ctx):,} contexts, {time.time()-t0:.0f}s)")

    feats = {}
    for split in ("val", "test"):
        p = os.path.join(cache_dir, f"feats_{split}_{key}.npz")
        if os.path.exists(p):
            z = np.load(p)
            feats[split] = {k: z[k] for k in z.files}
            print(f"  [cache] features {split}: HIT")
        else:
            t0 = time.time()
            feats[split] = compute_features(toks[split], o2,
                                            cfg.order2_match_order,
                                            float(cfg.match_len_cap))
            np.savez_compressed(p, **feats[split])
            print(f"  [cache] features {split}: computed ({time.time()-t0:.0f}s)")
    return byts, toks, feats, o2


def eval_loader(tokens, feats, cfg: Config, batch_size: int):
    ecfg = Config(**{**asdict(cfg), "batch_size": batch_size})
    return make_loader(tokens, feats, ecfg, shuffle=False)


# =============================================================================
#  Suite A + D core: model evaluation with quality/latency metrics
# =============================================================================
@torch.no_grad()
def eval_metrics(model, loader, tokens_total, bytes_total,
                 max_batches=None, temperature=1.0, device=None):
    """Cross-entropy -> bits/byte plus top-k accuracy and latency percentiles."""
    device = device or nsc.DEVICE
    model.eval()
    nats, ntok, top1, top5 = 0.0, 0, 0, 0
    lat = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = move_batch(batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        out = model(batch, stage=2)
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat.append(time.time() - t0)
        z = out["z_final"] / temperature
        y = batch["target_ids"]
        ce = torch.nn.functional.cross_entropy(
            z.reshape(-1, z.shape[-1]), y.reshape(-1), reduction="sum")
        nats += ce.item()
        ntok += y.numel()
        tk = z.topk(5, dim=-1).indices
        top1 += (tk[..., 0] == y).sum().item()
        top5 += (tk == y.unsqueeze(-1)).any(-1).sum().item()
    npt = nats / max(1, ntok)
    ratio = tokens_total / max(1, bytes_total)
    lat_s = sorted(lat)
    seq_tokens = ntok // max(1, len(lat))
    return dict(
        nats_per_token=npt,
        perplexity=math.exp(min(50.0, npt)),
        bits_per_byte=npt * math.log2(math.e) * ratio,
        top1_acc=top1 / max(1, ntok),
        top5_acc=top5 / max(1, ntok),
        tokens_evaluated=ntok,
        coverage_frac=ntok / max(1, tokens_total),
        latency_ms_p50=1e3 * lat_s[len(lat_s) // 2] if lat_s else float("nan"),
        latency_ms_p95=1e3 * lat_s[int(len(lat_s) * 0.95)] if lat_s else float("nan"),
        throughput_tok_s=ntok / max(1e-9, sum(lat)),
        batch_tokens=seq_tokens,
    )


def baseline_bpb(data: bytes):
    """Classical-compressor baselines (stdlib + optional zstd/brotli)."""
    out = {
        "gzip-9": len(gzip.compress(data, 9)) * 8.0 / len(data),
        "bzip2-9": len(bz2.compress(data, 9)) * 8.0 / len(data),
        "xz-9": len(lzma.compress(data, preset=9)) * 8.0 / len(data),
    }
    try:
        import zstandard
        out["zstd-19"] = (len(zstandard.ZstdCompressor(level=19).compress(data))
                          * 8.0 / len(data))
    except ImportError:
        pass
    try:
        import brotli
        out["brotli-11"] = len(brotli.compress(data, quality=11)) * 8.0 / len(data)
    except ImportError:
        pass
    return out


# =============================================================================
#  Suite B: correctness gates
# =============================================================================
@torch.no_grad()
def arithmetic_roundtrip(model, tokens, feats, cfg, batch_size):
    """Byte-exact range-coder roundtrip on REAL causal model distributions."""
    loader = eval_loader(tokens, feats, cfg, batch_size)
    batch = move_batch(next(iter(loader)), nsc.DEVICE)
    out = model(batch, stage=2)
    z = out["z_final"][0].float().cpu()
    tgt = batch["target_ids"][0].cpu().numpy()
    probs = torch.softmax(z, -1).numpy()
    cum = [_quantize(probs[t]) for t in range(len(tgt))]
    triples = [(int(cum[t][tgt[t]]), int(cum[t][tgt[t] + 1]), int(cum[t][-1]))
               for t in range(len(tgt))]
    rc = _RangeCoder()
    bits = rc.encode(triples)
    dec = rc.decode(bits, cum, len(tgt))
    return bool(np.array_equal(np.asarray(dec), tgt)), len(bits) / max(1, len(tgt))


def tokenizer_roundtrip(tok, data: bytes) -> bool:
    ids = encode_with(tok, data)
    return decode_with(tok, ids.tolist()) == data


def suite_correctness(model, tok, toks, feats, cfg, batch_size, ref_n):
    print("\n[B] correctness gates ...")
    coder_ok = _selftest_coder(0) and _selftest_coder(7)
    print(f"  arithmetic-coder self-test         : {'PASS' if coder_ok else 'FAIL'}")
    rt_ok, rt_bpt = arithmetic_roundtrip(model, toks["test"], feats["test"], cfg,
                                         batch_size)
    print(f"  model-distribution roundtrip       : {'PASS' if rt_ok else 'FAIL'} "
          f"({rt_bpt:.3f} bits/token achieved)")
    with open(PATHS["data"], "rb") as f:
        tk_ok = tokenizer_roundtrip(tok, f.read(1 << 20))
    print(f"  tokenizer byte-exact roundtrip 1MB : {'PASS' if tk_ok else 'FAIL'}")
    try:
        ref_bpt = reference_decoder_check(model, toks["test"], feats["test"], cfg,
                                          n=min(ref_n, len(toks["test"]) - 2))
        ref_ok = math.isfinite(ref_bpt)
    except Exception as e:
        ref_bpt, ref_ok = float("nan"), False
        print(f"  [warn] reference decoder raised: {e}")
    print(f"  streaming reference decoder        : "
          f"{'PASS' if ref_ok else 'FAIL'} ({ref_bpt:.3f} bits/token)")
    return dict(coder_selftest=coder_ok, roundtrip_lossless=rt_ok,
                roundtrip_bits_per_token=rt_bpt, tokenizer_roundtrip=tk_ok,
                reference_decoder_ok=ref_ok, reference_bits_per_token=ref_bpt,
                lossless=bool(coder_ok and rt_ok and tk_ok))


# =============================================================================
#  Suite C: calibration (RAM-bounded logit sampling)
# =============================================================================
@torch.no_grad()
def suite_calibration(model, loader, max_batches, pos_per_seq, seed=0):
    print("\n[C] calibration ...")
    rng = np.random.RandomState(seed)
    zs, ys = [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = move_batch(batch, nsc.DEVICE)
        out = model(batch, stage=2)
        z = out["z_final"].float().cpu()          # [B,T,V]
        y = batch["target_ids"].cpu()
        B, T, V = z.shape
        idx = rng.choice(T, size=min(pos_per_seq, T), replace=False)
        zs.append(z[:, idx].reshape(-1, V))
        ys.append(y[:, idx].reshape(-1))
    z, y = torch.cat(zs), torch.cat(ys)

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
    res = dict(T_opt=float(T), nll_before=float(nll(1.0)), nll_after=float(nll(T)),
               ece_before=float(nsc._ece(z, y)), ece_after=float(nsc._ece(z / T, y)),
               samples=int(len(y)))
    print(f"  T* = {res['T_opt']:.4f}   NLL {res['nll_before']:.4f} -> "
          f"{res['nll_after']:.4f}   ECE {res['ece_before']:.4f} -> "
          f"{res['ece_after']:.4f}  ({res['samples']:,} sampled positions)")
    return res


# =============================================================================
#  Suite E: robustness / stress
# =============================================================================
def suite_stress(model, tok, o2, cfg, batch_size, max_batches):
    print("\n[E] robustness stress cases ...")
    cases = {
        "repetitive": b"ABCD" * (1 << 18),                                # 1 MB
        "random": bytes(np.random.RandomState(0)
                        .randint(0, 256, 1 << 17, dtype=np.uint8)),      # 128 KB
        "structured_text": b"the quick brown fox jumps over the lazy dog " * 8192,
    }
    rows = {}
    for name, data in cases.items():
        t = encode_bytes_chunked(tok, data)
        if len(t) < cfg.seg_len + 2:
            rows[name] = {"skipped": f"too short ({len(t)} tokens)"}
            print(f"  {name:<16s} skipped (only {len(t)} tokens)")
            continue
        f = compute_features(t, o2, cfg.order2_match_order, float(cfg.match_len_cap))
        loader = eval_loader(t, f, cfg, batch_size)
        m = eval_metrics(model, loader, len(t), len(data), max_batches=max_batches)
        rows[name] = {"model_bpb": round(m["bits_per_byte"], 4),
                      "gzip_bpb": round(nsc.gzip_bpb(data), 4),
                      "xz_bpb": round(nsc.xz_bpb(data), 4)}
        print(f"  {name:<16s} model {m['bits_per_byte']:7.4f}  "
              f"gzip {rows[name]['gzip_bpb']:7.4f}  xz {rows[name]['xz_bpb']:7.4f}  bpb")
    return rows


# =============================================================================
#  Suite F: stopping policy — goal, plateau, overtraining
# =============================================================================
def estimate_final_bpb(step1, bpb1, step2, bpb2, total_steps):
    """Extrapolate the val bpb at curriculum end assuming the improvement rate
    decays linearly to zero at total_steps (conservative cosine-LR heuristic)."""
    if not (math.isfinite(bpb1) and math.isfinite(bpb2)) or step2 <= step1:
        return None
    rate = (bpb1 - bpb2) / (step2 - step1)              # bpb gained per step now
    mid = 0.5 * (step1 + step2)
    if total_steps <= mid or rate <= 0:
        return bpb2
    r0 = rate / (1.0 - mid / total_steps)
    remaining = r0 * (total_steps - step2) ** 2 / (2.0 * total_steps)
    return bpb2 - remaining


def suite_stopping(best_meta, last_meta, val_best, val_last, test_bpb_cal,
                   cfg, goal_bpb):
    print("\n[F] stopping policy ...")
    total_steps = cfg.stage1_steps + cfg.stage2_steps
    completion = best_meta["step"] / max(1, total_steps)

    # overtraining: do the newest weights do WORSE on val than the best export?
    newest, oldest = ((best_meta, last_meta)
                      if best_meta["step"] >= last_meta.get("step", -1)
                      else (last_meta, best_meta))
    val_newest = val_best if newest is best_meta else val_last
    val_prev = val_last if newest is best_meta else val_best
    overfit = (val_last is not None and val_best is not None
               and newest["step"] > oldest.get("step", -1)
               and val_newest > val_prev + OVERFIT_TOL)

    # marginal gain measured between the two checkpoints on record
    s1, b1 = oldest.get("step", -1), oldest.get("val_bpb", float("nan"))
    s2, b2 = newest["step"], newest.get("val_bpb", float("nan"))
    gain = (b1 - b2) if (math.isfinite(b1) and math.isfinite(b2)) else None
    # normalise to a ~1-session stride (30k steps @ ~3k steps/h, 10h cap) so the
    # plateau rule is independent of how far apart the two exports happen to be
    gain_30k = (gain * 30_000.0 / (s2 - s1)) if (gain is not None and s2 > s1) else gain
    proj = estimate_final_bpb(s1, b1, s2, b2, total_steps)

    # calibrated-test projections: test tracks val minus a measured offset
    off = (best_meta["val_bpb"] - test_bpb_cal
           if (test_bpb_cal is not None
               and math.isfinite(best_meta.get("val_bpb", float("nan")))) else None)
    proj_test = (proj - off) if (proj is not None and off is not None) else None
    goal_eta_step = None            # projected step at which the goal is reached
    if (off is not None and math.isfinite(b1) and math.isfinite(b2) and s2 > s1
            and total_steps > 0.5 * (s1 + s2) and b1 > b2):
        rate = (b1 - b2) / (s2 - s1)
        r0 = rate / (1.0 - 0.5 * (s1 + s2) / total_steps)
        tgt_val = goal_bpb + off
        for s in range(int(s2), int(total_steps) + 1, 250):
            if b2 - r0 * ((s - s2) - (s * s - s2 * s2) / (2.0 * total_steps)) <= tgt_val:
                goal_eta_step = s
                break

    plateau_thresh = max(PLATEAU_ABS_BPB, PLATEAU_REL * (b2 if math.isfinite(b2) else 1.3))
    goal_met = test_bpb_cal is not None and test_bpb_cal <= goal_bpb

    if overfit:
        verdict, why = "STOP (overtraining)", (
            f"newest weights (step {newest['step']}) are worse on val than the "
            f"best export by more than {OVERFIT_TOL} bpb — ship best.pt.")
    elif goal_met:
        verdict, why = "STOP (goal reached)", (
            f"calibrated test bpb {test_bpb_cal:.4f} <= goal {goal_bpb}.")
    elif gain_30k is not None and gain_30k < plateau_thresh and completion > 0.5:
        verdict, why = "STOP (plateau)", (
            f"recent gain {gain_30k:+.4f} bpb/session < plateau threshold "
            f"{plateau_thresh:.4f}.")
    else:
        eta_steps = total_steps - newest["step"]
        eta_txt = ("" if goal_eta_step else
                   " The goal is likely decided at curriculum end.")
        verdict, why = "CONTINUE", (
            f"still improving (recent gain "
            f"{gain_30k if gain_30k is None else round(gain_30k, 4)} bpb per "
            f"~30k-step session); {eta_steps:,} curriculum steps remain "
            f"({completion:.0%} complete). Projected final val bpb ≈ {proj:.4f}"
            + (f", calibrated test ≈ {proj_test:.4f}" if proj_test else "")
            + (f"; goal {goal_bpb} projected reachable near step "
               f"{goal_eta_step:,}." if goal_eta_step else "." + eta_txt)
            if proj else "still improving.")

    res = dict(verdict=verdict, reason=why,
               goal_test_bpb=goal_bpb, goal_met=bool(goal_met),
               curriculum_step=newest["step"], curriculum_total=total_steps,
               curriculum_completion=round(completion, 4),
               recent_gain_bpb=gain,
               recent_gain_bpb_per_30k=None if gain_30k is None else round(gain_30k, 4),
               plateau_threshold_bpb=round(plateau_thresh, 4),
               projected_final_val_bpb=None if proj is None else round(proj, 4),
               projected_final_test_bpb_cal=None if proj_test is None else round(proj_test, 4),
               goal_projected_step=goal_eta_step,
               overtraining_detected=bool(overfit),
               measured_val_bpb_best=val_best, measured_val_bpb_last=val_last,
               policy=dict(GOAL_TEST_BPB=goal_bpb, PLATEAU_ABS_BPB=PLATEAU_ABS_BPB,
                           PLATEAU_REL=PLATEAU_REL, OVERFIT_TOL=OVERFIT_TOL,
                           early_stop_patience_in_trainer=cfg.early_stop_patience))
    print(f"  checkpoint step {newest['step']:,}/{total_steps:,} "
          f"({completion:.0%} of curriculum)")
    if gain is not None:
        print(f"  recent val gain {gain:+.4f} bpb over "
              f"{newest['step']-oldest.get('step',0):,} steps "
              f"≈ {gain_30k:+.4f} bpb/session "
              f"(plateau threshold {plateau_thresh:.4f})")
    if proj is not None:
        print(f"  projected final val bpb ≈ {proj:.4f}"
              + (f"   calibrated test ≈ {proj_test:.4f}" if proj_test else "")
              + f" (goal: {goal_bpb})")
    if goal_eta_step is not None and not goal_met:
        print(f"  goal {goal_bpb} projected reachable near step {goal_eta_step:,} "
              f"(~{max(0, goal_eta_step - s2):,} steps away)")
    print(f"  VERDICT: {verdict} — {why}")
    return res


# =============================================================================
#  Suite G: training state — actual LR / loss + tuning recommendations
# =============================================================================
def lr_at(cfg: Config, step: int) -> float:
    """Reconstruct the trainer's scheduled learning rate at a given step."""
    if step < cfg.stage1_steps:
        return nsc.cosine_lr(step, cfg.stage1_steps, cfg.lr_stage1, cfg.warmup_steps)
    return nsc.cosine_lr(step - cfg.stage1_steps, cfg.stage2_steps,
                         cfg.lr_stage2, cfg.warmup_steps)


def suite_training_state(cfg, best_meta, calib, comp, stress, m_test, m_val,
                         goal_bpb):
    print("\n[G] training state & tuning ...")
    step = best_meta["step"]
    total = cfg.stage1_steps + cfg.stage2_steps
    lr_now = lr_at(cfg, step)
    stage = 1 if step < cfg.stage1_steps else 2
    log2e = math.log2(math.e)
    state = dict(
        step=step, total_steps=total, stage=stage,
        lr_now=lr_now, lr_stage1=cfg.lr_stage1, lr_stage2=cfg.lr_stage2,
        warmup_steps=cfg.warmup_steps,
        val_loss_nats=m_val["nats_per_token"],
        val_loss_bits=m_val["nats_per_token"] * log2e,
        val_bpb=m_val["bits_per_byte"],
        test_loss_nats=m_test["nats_per_token"],
        test_loss_bits=m_test["nats_per_token"] * log2e,
        test_bpb=m_test["bits_per_byte"],
        stored_val_bpb=best_meta.get("val_bpb"),
    )
    print(f"  step {step:,}/{total:,} (stage {stage})   scheduled LR now = {lr_now:.3e}")
    print(f"  val  loss {state['val_loss_nats']:.4f} nats/tok "
          f"({state['val_loss_bits']:.4f} bits/tok, {state['val_bpb']:.4f} bpb)")
    print(f"  test loss {state['test_loss_nats']:.4f} nats/tok "
          f"({state['test_loss_bits']:.4f} bits/tok, {state['test_bpb']:.4f} bpb)")

    recs = []
    if step < total:
        eta_h = (total - step) / 3000.0     # T4 ≈ 3000 optimizer steps/h (base tier)
        recs.append(
            f"Finish the curriculum first — {total - step:,} steps remain "
            f"(~{eta_h:.0f} h on a Kaggle T4 at ≈3,000 steps/h, i.e. "
            f"~{max(1, round(eta_h / 11.5))} more sessions). The cosine schedule "
            f"continues automatically on resume; the LR is already at "
            f"{lr_now:.2e} and anneals to ~0 at step {total:,}, which is itself "
            f"the main protection against late-stage overfitting.")
    gain_cal = comp["test_bpb"] - comp["test_bpb_cal"]
    if calib["T_opt"] > 1.05:
        recs.append(
            f"Logits are overconfident (T* = {calib['T_opt']:.3f}). Keep baking "
            f"calibration_T into the export — it is worth {gain_cal:+.4f} bpb "
            f"for free. If a fresh run is ever started, raising dropout "
            f"{cfg.dropout} → 0.15 or weight_decay {cfg.weight_decay} → 0.02 "
            f"should reduce this overconfidence at the source.")
    rnd = stress.get("random", {})
    if isinstance(rnd.get("model_bpb"), (int, float)) and rnd["model_bpb"] > 8.5:
        recs.append(
            f"Incompressible input EXPANDS badly (random bytes: "
            f"{rnd['model_bpb']:.1f} bpb vs 8.0 raw). Add a raw-block escape to "
            f"the Rust coder: when the running model cost exceeds ~8 bpb over a "
            f"block, emit literal bytes. Caps the worst case at ≈8.03 bpb "
            f"without touching the model.")
    rep = stress.get("repetitive", {})
    if (isinstance(rep.get("model_bpb"), (int, float))
            and rep["model_bpb"] > 20 * max(1e-9, rep.get("xz_bpb", 0))):
        recs.append(
            f"Highly repetitive data: classical LZ crushes it "
            f"(xz {rep.get('xz_bpb')} vs model {rep.get('model_bpb')} bpb). "
            f"The match expert saturates at conf 0.99 with a smoothed tail; a "
            f"cheap LZ pre-pass (or letting match_conf reach ~0.999) would "
            f"close most of that gap.")
    if step >= total:
        recs.append(
            "Curriculum complete. If more budget exists, do NOT keep training "
            "this architecture past the schedule — escalate one ladder row "
            "(d=384, L=5, ~23 M params, rec. 185k steps ≈ 60 h) and reuse the "
            "cached data; more steps on a finished cosine buy almost nothing.")
    recs.append(
        f"Stopping is AUTOMATIC: run_budget() ends the session chain the "
        f"moment a phase confirms calibrated test bpb ≤ {goal_bpb} (the "
        f"statistically projected best-achievable value for this "
        f"architecture), or at curriculum end (step {total:,}), or on a val "
        f"plateau (< {PLATEAU_ABS_BPB} bpb per ~30k-step session; trainer "
        f"patience {cfg.early_stop_patience} evals) — whichever happens "
        f"first. Sessions started after the stop exit immediately.")
    state["recommendations"] = recs
    for i, r in enumerate(recs, 1):
        print(f"  ({i}) {r}")
    return state


# =============================================================================
#  Charts (matplotlib; light-surface palette, saved as PNG next to the report)
# =============================================================================
PAL = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
           grid="#e1e0d9", axis="#c3c2b7",
           s1="#2a78d6", s2="#1baf7a", s3="#eda100", good="#0ca30c")


def _new_ax(figsize):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor(PAL["surface"])
    ax.set_facecolor(PAL["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PAL["axis"])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=PAL["muted"], labelsize=8.5, length=0)
    return fig, ax


def _finish(fig, ax, title, subtitle, path):
    import matplotlib.pyplot as plt
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="semibold",
                 color=PAL["ink"], pad=16)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=8.5,
                color=PAL["ink2"], va="bottom")
    fig.tight_layout()
    fig.savefig(path, facecolor=PAL["surface"], bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart] {path}")


def chart_sizes(report, path):
    """Before/after: the 5 MB test corpus vs every compressor's output size."""
    a = report["compression"]
    orig_mb = a["test_bytes"] / 2**20
    rows = [("original (enwik8 test)", orig_mb, PAL["grid"])]
    rows += [(f"hybrid model (ours)", a["test_bpb_cal"] * a["test_bytes"] / 8 / 2**20,
              PAL["s1"])]
    for name, bpb in sorted(a["baselines"].items(), key=lambda kv: kv[1]):
        rows.append((name, bpb * a["test_bytes"] / 8 / 2**20, PAL["axis"]))
    fig, ax = _new_ax((7.4, 0.62 * len(rows) + 1.2))
    y = np.arange(len(rows))[::-1]
    for yi, (name, mb, color) in zip(y, rows):
        ax.barh(yi, mb, height=0.58, color=color, zorder=3)
        ratio = "" if mb >= orig_mb else f"   ({orig_mb / mb:.1f}x smaller)"
        ax.text(mb + orig_mb * 0.012, yi, f"{mb:.2f} MB{ratio}", va="center",
                fontsize=8.5, color=PAL["ink2"])
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9, color=PAL["ink"])
    ax.set_xlim(0, orig_mb * 1.30)
    ax.set_xlabel("size (MB)", fontsize=8.5, color=PAL["muted"])
    ax.xaxis.grid(True, color=PAL["grid"], linewidth=0.8, zorder=0)
    _finish(fig, ax, "File size before and after compression",
            "projected from measured bits/byte on the held-out enwik8 test split",
            path)


def chart_bpb_cases(report, path):
    """Dot plot (log x): bits/byte by content type for model vs gzip vs xz."""
    a = report["compression"]
    cases = [("enwik8 test", a["test_bpb_cal"], a["baselines"].get("gzip-9"),
              a["baselines"].get("xz-9"))]
    for name, row in report["stress"].items():
        if "model_bpb" in row:
            cases.append((name.replace("_", " "), row["model_bpb"],
                          row["gzip_bpb"], row["xz_bpb"]))
    series = [("hybrid (ours)", 1, PAL["s1"]), ("gzip -9", 2, PAL["s2"]),
              ("xz -9", 3, PAL["s3"])]
    fig, ax = _new_ax((7.4, 0.75 * len(cases) + 1.5))
    y = np.arange(len(cases))[::-1]
    for label, col, color in series:
        xs = [max(1e-3, c[col]) for c in cases]
        ax.scatter(xs, y, s=64, color=color, zorder=4, label=label,
                   edgecolors=PAL["surface"], linewidths=2)
    ax.set_xscale("log")
    ax.set_yticks(y, [c[0] for c in cases], fontsize=9, color=PAL["ink"])
    ax.axvline(8.0, color=PAL["axis"], linewidth=1.0, zorder=1)
    ax.text(8.0, y.max() + 0.55, " 8 bpb = raw bytes", fontsize=7.5,
            color=PAL["muted"], va="bottom")
    ax.set_xlabel("bits per byte — lower is better (log scale)",
                  fontsize=8.5, color=PAL["muted"])
    ax.xaxis.grid(True, color=PAL["grid"], linewidth=0.8, zorder=0)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5,
              labelcolor=PAL["ink2"])
    _finish(fig, ax, "Compression by content type",
            "points left of the 8 bpb line compress; right of it expand", path)


def chart_lr(report, cfg, path):
    """The trainer's LR schedule with the checkpoint's position marked."""
    total = cfg.stage1_steps + cfg.stage2_steps
    steps = np.linspace(0, total, 800)
    lrs = np.array([lr_at(cfg, int(s)) for s in steps])
    now = report["checkpoint"]["step"]
    lr_now = lr_at(cfg, now)
    fig, ax = _new_ax((7.4, 3.4))
    ax.plot(steps, lrs * 1e4, color=PAL["s1"], linewidth=2, zorder=3,
            solid_joinstyle="round")
    ax.axvline(cfg.stage1_steps, color=PAL["grid"], linewidth=1.0, zorder=1)
    ax.text(cfg.stage1_steps, ax.get_ylim()[1] * 0.97, " stage 1 → 2",
            fontsize=7.5, color=PAL["muted"], va="top")
    ax.scatter([now], [lr_now * 1e4], s=72, color=PAL["s1"], zorder=5,
               edgecolors=PAL["surface"], linewidths=2)
    ax.annotate(f"now: {lr_now:.2e} @ step {now:,}",
                (now, lr_now * 1e4), xytext=(8, 10),
                textcoords="offset points", fontsize=8.5, color=PAL["ink2"])
    ax.set_xlabel("optimizer step", fontsize=8.5, color=PAL["muted"])
    ax.set_ylabel("learning rate (x 1e-4)", fontsize=8.5, color=PAL["muted"])
    ax.yaxis.grid(True, color=PAL["grid"], linewidth=0.8, zorder=0)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v/1000:.0f}k")
    _finish(fig, ax, "Learning-rate schedule (two-stage cosine)",
            f"stage 1: {cfg.lr_stage1:.0e} pretrain - stage 2: "
            f"{cfg.lr_stage2:.0e} residual fine-tune - anneals to ~0 at "
            f"step {total // 1000}k", path)


def chart_trajectory(report, cfg, goal_bpb, path):
    """Observed val bpb + decay-model projection to curriculum end + goal."""
    total = cfg.stage1_steps + cfg.stage2_steps
    pts = []
    for key in ("latest_checkpoint", "checkpoint"):
        m = report.get(key) or {}
        if m.get("step", -1) > 0 and math.isfinite(m.get("val_bpb", float("nan"))):
            pts.append((m["step"], m["val_bpb"]))
    pts = sorted(set(pts))
    fig, ax = _new_ax((7.4, 3.6))
    if len(pts) >= 2:
        (s1, b1), (s2, b2) = pts[0], pts[-1]
        rate = (b1 - b2) / max(1, s2 - s1)
        mid = 0.5 * (s1 + s2)
        r0 = rate / max(1e-9, 1.0 - mid / total)
        proj_s = np.linspace(s2, total, 200)
        proj_b = b2 - r0 * ((proj_s - s2) - (proj_s**2 - s2**2) / (2 * total))
        ax.plot(proj_s, proj_b, color=PAL["s1"], linewidth=2, linestyle=(0, (4, 3)),
                zorder=3)
        ax.annotate(f"projected ≈ {proj_b[-1]:.4f}", (total, proj_b[-1]),
                    xytext=(-8, -14), textcoords="offset points", ha="right",
                    fontsize=8.5, color=PAL["ink2"])
    xs, ys = zip(*pts) if pts else ((), ())
    ax.plot(xs, ys, color=PAL["s1"], linewidth=2, zorder=4)
    ax.scatter(xs, ys, s=64, color=PAL["s1"], zorder=5,
               edgecolors=PAL["surface"], linewidths=2)
    for s, b in pts:
        ax.annotate(f"{b:.4f}", (s, b), xytext=(0, 9), ha="center",
                    textcoords="offset points", fontsize=8, color=PAL["ink2"])
    ax.axhline(goal_bpb, color=PAL["good"], linewidth=1.4, zorder=2)
    ax.text(0.01, goal_bpb, f"goal: {goal_bpb} bpb — stop here",
            transform=ax.get_yaxis_transform(), fontsize=8, color="#006300",
            va="bottom")
    ax.set_xlim(min(xs) - 4000 if xs else 0, total + 6000)
    ax.set_xlabel("optimizer step", fontsize=8.5, color=PAL["muted"])
    ax.set_ylabel("validation bits/byte", fontsize=8.5, color=PAL["muted"])
    ax.yaxis.grid(True, color=PAL["grid"], linewidth=0.8, zorder=0)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v/1000:.0f}k")
    _finish(fig, ax, "Validation bits/byte — observed, projected, and the goal",
            "solid: checkpoints on record - dashed: linear-decay projection to "
            "curriculum end", path)


def make_charts(report, cfg, goal_bpb, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("  [charts] matplotlib not installed — skipping charts")
        return {}
    cdir = os.path.join(outdir, "charts")
    os.makedirs(cdir, exist_ok=True)
    print("\n[charts] rendering ...")
    jobs = {"sizes": lambda p: chart_sizes(report, p),
            "bpb_cases": lambda p: chart_bpb_cases(report, p),
            "lr_schedule": lambda p: chart_lr(report, cfg, p),
            "val_trajectory": lambda p: chart_trajectory(report, cfg, goal_bpb, p)}
    out = {}
    for name, fn in jobs.items():
        path = os.path.join(cdir, f"{name}.png")
        try:
            fn(path)
            out[name] = os.path.join("charts", f"{name}.png")
        except Exception as e:
            print(f"  [charts] {name} failed (non-fatal): {e}")
    return out


# =============================================================================
#  Report
# =============================================================================
def write_report(report, outdir):
    jpath = os.path.join(outdir, "benchmark_report.json")
    with open(jpath, "w") as f:
        json.dump(report, f, indent=2, default=float)

    a, b_, c = report["compression"], report["correctness"], report["calibration"]
    d, f_, env = report["performance"], report["stopping"], report["environment"]
    g, charts = report.get("training_state", {}), report.get("charts", {})
    test_mb = a.get("test_bytes", 0) / 2**20
    lines = [
        "# Enterprise benchmark report", "",
        f"*Generated {report['generated']} — checkpoint "
        f"`{os.path.basename(report['checkpoint']['path'])}` step "
        f"{report['checkpoint']['step']:,} — device {env['device']}*", "",
        "## A. Compression quality (test split)", "",
        "| metric | value |", "|---|---|",
        f"| bits/byte (raw) | {a['test_bpb']:.4f} |",
        f"| bits/byte (calibrated T*={c['T_opt']:.3f}) | {a['test_bpb_cal']:.4f} |",
        f"| perplexity | {a['perplexity']:.2f} |",
        f"| top-1 / top-5 accuracy | {a['top1_acc']:.2%} / {a['top5_acc']:.2%} |",
        f"| projected size of 100 MB enwik8 | {a['projected_mb_per_100mb']:.2f} MB |",
        f"| eval coverage | {a['coverage_frac']:.1%} of test split |", "",
        "### Before / after (5 MB held-out corpus)", "",
        "| compressor | size after | ratio | bits/byte |", "|---|---|---|---|",
        f"| — original | {test_mb:.2f} MB | 1.00x | 8.0000 |",
        f"| **hybrid model (ours)** | "
        f"**{a['test_bpb_cal'] * a.get('test_bytes', 0) / 8 / 2**20:.2f} MB** | "
        f"**{8.0 / a['test_bpb_cal']:.2f}x** | **{a['test_bpb_cal']:.4f}** |"]
    for name, bpb in sorted(a["baselines"].items(), key=lambda kv: kv[1]):
        lines.append(f"| {name} | {bpb * a.get('test_bytes', 0) / 8 / 2**20:.2f} MB "
                     f"| {8.0 / bpb:.2f}x | {bpb:.4f} |")
    if charts.get("sizes"):
        lines += ["", f"![before/after sizes]({charts['sizes']})"]
    lines += [
        "", "## B. Correctness gates", "",
        f"- arithmetic-coder self-test: **{'PASS' if b_['coder_selftest'] else 'FAIL'}**",
        f"- model-distribution lossless roundtrip: "
        f"**{'PASS' if b_['roundtrip_lossless'] else 'FAIL'}** "
        f"({b_['roundtrip_bits_per_token']:.3f} bits/token)",
        f"- tokenizer byte-exact roundtrip: "
        f"**{'PASS' if b_['tokenizer_roundtrip'] else 'FAIL'}**",
        f"- streaming reference decoder: "
        f"**{'PASS' if b_['reference_decoder_ok'] else 'FAIL'}**", "",
        "## C. Calibration", "",
        f"- optimal temperature T* = {c['T_opt']:.4f}",
        f"- NLL {c['nll_before']:.4f} → {c['nll_after']:.4f}",
        f"- ECE {c['ece_before']:.4f} → {c['ece_after']:.4f}", "",
        "## D. Performance", "",
        f"- parameters: {d['params_m']:.2f} M  (checkpoint "
        f"{report['checkpoint']['size_mb']:.0f} MB on disk)",
        f"- throughput: {d['throughput_tok_s']:.0f} tok/s ≈ "
        f"{d['throughput_mb_s']:.3f} MB/s  ({env['device']}, batched upper bound)",
        f"- latency per batch p50/p95: {d['latency_ms_p50']:.0f} / "
        f"{d['latency_ms_p95']:.0f} ms",
        f"- peak RSS: {d['peak_rss_gb']:.2f} GB", "",
        "## E. Robustness", "",
        "| case | model bpb | gzip bpb | xz bpb |", "|---|---|---|---|"]
    for name, row in report["stress"].items():
        if "skipped" in row:
            lines.append(f"| {name} | skipped | | |")
        else:
            lines.append(f"| {name} | {row['model_bpb']} | {row['gzip_bpb']} | "
                         f"{row['xz_bpb']} |")
    if charts.get("bpb_cases"):
        lines += ["", f"![bpb by content type]({charts['bpb_cases']})"]
    lines += [
        "", "## F. Stopping policy", "",
        f"**{f_['verdict']}** — {f_['reason']}", "",
        f"- curriculum: step {f_['curriculum_step']:,} / "
        f"{f_['curriculum_total']:,} ({f_['curriculum_completion']:.0%})",
        f"- goal (calibrated test bpb): {f_['goal_test_bpb']} — "
        f"{'met' if f_['goal_met'] else 'not met'}",
        f"- projected final val bpb: {f_['projected_final_val_bpb']}",
        f"- projected final calibrated test bpb: "
        f"{f_.get('projected_final_test_bpb_cal')}",
        f"- goal projected reachable near step: "
        f"{f_.get('goal_projected_step') or '— (curriculum end decides)'}",
        f"- overtraining detected: {f_['overtraining_detected']}"]
    if charts.get("val_trajectory"):
        lines += ["", f"![validation trajectory]({charts['val_trajectory']})"]
    if g:
        lines += [
            "", "## G. Training state & tuning", "",
            "| quantity | value |", "|---|---|",
            f"| optimizer step | {g['step']:,} / {g['total_steps']:,} "
            f"(stage {g['stage']}) |",
            f"| scheduled LR now | {g['lr_now']:.3e} |",
            f"| LR plan | stage 1: {g['lr_stage1']:.0e} → stage 2: "
            f"{g['lr_stage2']:.0e}, warmup {g['warmup_steps']:,}, cosine → 0 |",
            f"| val loss | {g['val_loss_nats']:.4f} nats/tok = "
            f"{g['val_loss_bits']:.4f} bits/tok ({g['val_bpb']:.4f} bpb) |",
            f"| test loss | {g['test_loss_nats']:.4f} nats/tok = "
            f"{g['test_loss_bits']:.4f} bits/tok ({g['test_bpb']:.4f} bpb) |",
            f"| stored best val bpb (full split) | {g['stored_val_bpb']:.4f} |"]
        if charts.get("lr_schedule"):
            lines += ["", f"![LR schedule]({charts['lr_schedule']})"]
        lines += ["", "### Recommendations", ""]
        lines += [f"{i}. {r}" for i, r in enumerate(g.get("recommendations", []), 1)]
    lines += ["", f"**Overall: {'PASS' if report['gates_passed'] else 'FAIL'}**", ""]
    mpath = os.path.join(outdir, "benchmark_report.md")
    with open(mpath, "w") as f:
        f.write("\n".join(lines))
    return jpath, mpath


# =============================================================================
#  Main
# =============================================================================
PATHS = {}


def main(argv=None):
    """Run the benchmark. `argv=None` parses sys.argv (CLI); pass a list (e.g.
    [] or ["--full"]) when calling from a notebook cell — unknown kernel args
    are ignored."""
    global PATHS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", action="store_true",
                    help="run a training session first (overrides SKIP_TRAINING=True)")
    ap.add_argument("--budget", type=float, default=TRAIN_BUDGET_H,
                    help="total cross-session budget in hours (training mode)")
    ap.add_argument("--session-cap", type=float, default=SESSION_CAP_H)
    ap.add_argument("--export-dir", default=None,
                    help="dir with best.pt + tokenizer.json (default: results/export)")
    ap.add_argument("--data", default=None, help="path to enwik8")
    ap.add_argument("--full", action="store_true",
                    help="evaluate the FULL val/test splits (slow on CPU)")
    ap.add_argument("--max-batches", type=int, default=TEST_MAX_BATCHES)
    ap.add_argument("--goal-bpb", type=float, default=GOAL_TEST_BPB)
    ap.add_argument("--skip-last-ckpt", action="store_true",
                    help="skip evaluating the .last checkpoint (halves RAM/time)")
    args, _unknown = ap.parse_known_args(argv)

    skip_training = SKIP_TRAINING and not args.train
    if not skip_training:
        print("[mode] TRAINING ENABLED — running a session via run_budget() first")
        nsc.run_budget(total_budget_h=args.budget, session_cap_h=args.session_cap)
        runs = os.path.join(nsc.WORKDIR, "runs")
        cands = sorted((os.path.join(r, "best.pt")
                        for r in (os.path.join(runs, d) for d in os.listdir(runs))
                        if os.path.exists(os.path.join(r, "best.pt"))),
                       key=os.path.getmtime)
        if cands:
            args.export_dir = os.path.dirname(cands[-1])
    else:
        print("[mode] SKIP_TRAINING=True — evaluating previously exported results only")

    PATHS = locate_paths(args.export_dir, args.data)
    print(f"[paths] export={PATHS['export_dir']}")
    print(f"[paths] ckpt  ={PATHS['best_ckpt']}  last={PATHS['last_ckpt']}")
    print(f"[paths] data  ={PATHS['data']}")

    test_cap = None if args.full else args.max_batches
    val_cap = None if args.full else VAL_MAX_BATCHES

    # ---- load model + tokenizer + data ----
    state, cfg_dict, best_meta = load_checkpoint_meta(PATHS["best_ckpt"])
    model, cfg = build_model(state, cfg_dict)
    del state
    print(f"[model] tier={cfg.tier} d={cfg.d_model} L={cfg.n_layers} "
          f"vocab={cfg.vocab_size} params={count_params(model)/1e6:.2f}M "
          f"step={best_meta['step']:,} stored val_bpb={best_meta['val_bpb']:.4f}")

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(PATHS["tokenizer"])
    assert tok.get_vocab_size() == cfg.vocab_size, "tokenizer/config vocab mismatch"

    cache_dir = writable_dir(os.path.join(os.path.dirname(PATHS["export_dir"]),
                                          "cache"))
    print("\n[data] preparing splits + classical features (cached) ...")
    byts, toks, feats, o2 = prepare_eval_data(tok, cfg, PATHS["data"], cache_dir)

    va_loader = eval_loader(toks["val"], feats["val"], cfg, EVAL_BATCH_SIZE)
    te_loader = eval_loader(toks["test"], feats["test"], cfg, EVAL_BATCH_SIZE)

    # ---- Suite A: compression quality ----
    print(f"\n[A] compression quality on test split "
          f"({'FULL' if test_cap is None else f'{test_cap} batches'}) ...")
    m_test = eval_metrics(model, te_loader, len(toks["test"]), len(byts["test"]),
                          max_batches=test_cap)
    print(f"  raw test bpb {m_test['bits_per_byte']:.4f}   ppl "
          f"{m_test['perplexity']:.2f}   top1 {m_test['top1_acc']:.2%}   "
          f"top5 {m_test['top5_acc']:.2%}   ({m_test['tokens_evaluated']:,} tokens, "
          f"{m_test['coverage_frac']:.1%} of split)")
    print("  classical baselines on the same bytes ...")
    baselines = baseline_bpb(byts["test"])
    for k, v in baselines.items():
        print(f"    {k:<10s} {v:.4f} bpb")

    # ---- Suite C: calibration, then calibrated test bpb ----
    calib = suite_calibration(model, va_loader, CALIB_BATCHES, CALIB_POS_PER_SEQ)
    T = calib["T_opt"]
    m_test_cal = eval_metrics(model, te_loader, len(toks["test"]), len(byts["test"]),
                              max_batches=test_cap, temperature=T)
    print(f"  calibrated test bpb {m_test_cal['bits_per_byte']:.4f} (T*={T:.3f})")

    comp = dict(test_bpb=m_test["bits_per_byte"],
                test_bpb_cal=m_test_cal["bits_per_byte"],
                perplexity=m_test["perplexity"], top1_acc=m_test["top1_acc"],
                top5_acc=m_test["top5_acc"], coverage_frac=m_test["coverage_frac"],
                tokens_evaluated=m_test["tokens_evaluated"],
                test_bytes=len(byts["test"]),
                projected_mb_per_100mb=m_test_cal["bits_per_byte"] * 100 / 8,
                baselines=baselines)

    # ---- Suite B: correctness ----
    corr = suite_correctness(model, tok, toks, feats, cfg, EVAL_BATCH_SIZE,
                             REF_DECODER_N)

    # ---- Suite D: performance ----
    perf = dict(params_m=count_params(model) / 1e6,
                throughput_tok_s=m_test["throughput_tok_s"],
                throughput_mb_s=(m_test["throughput_tok_s"]
                                 * len(byts["test"]) / len(toks["test"]) / 2**20),
                latency_ms_p50=m_test["latency_ms_p50"],
                latency_ms_p95=m_test["latency_ms_p95"],
                peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
    print(f"\n[D] performance: {perf['throughput_tok_s']:.0f} tok/s "
          f"({perf['throughput_mb_s']:.3f} MB/s), batch latency p50 "
          f"{perf['latency_ms_p50']:.0f} ms, peak RSS {perf['peak_rss_gb']:.2f} GB")

    # ---- Suite E: stress ----
    stress = suite_stress(model, tok, o2, cfg, EVAL_BATCH_SIZE, STRESS_MAX_BATCHES)

    # ---- Suite F: stopping policy (compare best vs latest checkpoint) ----
    m_val = eval_metrics(model, va_loader, len(toks["val"]), len(byts["val"]),
                         max_batches=val_cap)
    val_best = m_val["bits_per_byte"]
    last_meta, val_last = {}, None
    if PATHS["last_ckpt"] and not args.skip_last_ckpt:
        del model
        state, cfg2, last_meta = load_checkpoint_meta(PATHS["last_ckpt"])
        model2, _ = build_model(state, cfg2)
        del state
        val_last = eval_metrics(model2, va_loader, len(toks["val"]),
                                len(byts["val"]), max_batches=val_cap)["bits_per_byte"]
        del model2
        print(f"\n  measured val bpb — best.pt(step {best_meta['step']:,}): "
              f"{val_best:.4f}   {os.path.basename(last_meta['path'])}"
              f"(step {last_meta['step']:,}): {val_last:.4f}")
    stop = suite_stopping(best_meta, last_meta or {"step": -1}, val_best, val_last,
                          comp["test_bpb_cal"], cfg, args.goal_bpb)

    # ---- Suite G: training state (actual LR / loss) + tuning advice ----
    tstate = suite_training_state(cfg, best_meta, calib, comp, stress,
                                  m_test, m_val, args.goal_bpb)

    # ---- gates + report ----
    gates = dict(lossless=corr["lossless"],
                 reference_decoder=corr["reference_decoder_ok"],
                 beats_gzip=comp["test_bpb_cal"] < baselines["gzip-9"],
                 calibration_finite=math.isfinite(calib["T_opt"]))
    passed = all(gates.values())
    report = dict(
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        environment=dict(python=platform.python_version(),
                         torch=torch.__version__,
                         device=str(nsc.DEVICE), host=platform.node()),
        checkpoint=best_meta, latest_checkpoint=last_meta,
        config={k: cfg_dict[k] for k in ("tier", "vocab_size", "d_model", "n_layers",
                                         "n_heads", "gru_hidden", "seg_len",
                                         "stage1_steps", "stage2_steps")},
        compression=comp, correctness=corr, calibration=calib, performance=perf,
        stress=stress, stopping=stop, training_state=tstate,
        gates=gates, gates_passed=passed)
    outdir = writable_dir(os.path.dirname(PATHS["export_dir"]))
    report["charts"] = make_charts(report, cfg, args.goal_bpb, outdir)
    jpath, mpath = write_report(report, outdir)

    print("\n" + "=" * 78)
    print(f"  OVERALL: {'PASS' if passed else 'FAIL'}   "
          f"test bpb {comp['test_bpb_cal']:.4f} (calibrated)   "
          f"gzip {baselines['gzip-9']:.3f}   xz {baselines['xz-9']:.3f}")
    print(f"  STOPPING: {stop['verdict']}")
    print(f"  report: {jpath}\n          {mpath}")
    print("=" * 78)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
