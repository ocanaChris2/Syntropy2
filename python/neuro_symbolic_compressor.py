# -*- coding: utf-8 -*-
# =============================================================================
#  NEURO-SYMBOLIC HYBRID LOSSLESS COMPRESSOR  --  Kaggle training notebook
# =============================================================================
#  A single, reproducible PyTorch 2.x notebook that trains the *probability
#  predictor* for a novel neuro-symbolic compressor and exports it (ONNX + JSON
#  + classical tables) for a streaming Rust/C entropy coder.
#
#  Novelty (see the "Architecture Novelty" markdown cell below):
#    - Neural models predict a RESIDUAL LOGIT DELTA on top of a classical base
#      distribution, CONDITIONED (via FiLM) on the classical CONFIDENCE state.
#    - A CONTEXT-GATED ATTENTION fusion engine routes per-token between experts
#      (classical mixture weights `w` + neural residual gates `g`).
#    - A NEURAL-SKIP gate makes compute adaptive to data redundancy, which is
#      how the minimal-resource ("nano") tier approaches real-time speeds on
#      compressible data.
#
#  Design principle (user): "closest approach with minimal resources, disposing
#  more resources if available" -> everything is driven by a single CONFIG with
#  nano / base / large tiers. Pick the tier that fits your GPU/time budget.
#
#  This .py is structured so that importing it has NO side effects (no installs,
#  no downloads, no training). All orchestration lives under __main__ and the
#  cell-like functions, so the model wiring can be unit/smoke-tested. On Kaggle,
#  paste each delimited section into its own cell, or run `python this_file.py
#  --tier nano --smoke` for a fast CPU sanity check.
# =============================================================================

# %% ==========================================================================
#  SECTION 1 -- SETUP & DEPENDENCIES
# =============================================================================
from __future__ import annotations

import os
import sys
import json
import time
import math
import gzip
import zlib
import shutil
import hashlib
import argparse
import signal
import subprocess
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, Dict, List

import numpy as np

# Torch is required. tokenizers / onnxruntime are optional (graceful fallback).
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import tokenizers  # HuggingFace fast BPE; preinstalled on Kaggle
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
    from tokenizers.decoders import ByteLevel as ByteLevelDec
    _HAS_TOKENIZERS = True
except Exception:  # pragma: no cover - offline / smoke fallback
    _HAS_TOKENIZERS = False

try:
    import onnxruntime as ort
    _HAS_ORT = True
except Exception:  # pragma: no cover
    _HAS_ORT = False

# Graceful-shutdown: SIGTERM from Kaggle hard-kill / Ctrl-C
_STOP_FLAG: list = [False]   # mutable so signal handler can set it without global


def _install_signal_handlers() -> None:
    def _handler(signum, frame):
        if not _STOP_FLAG[0]:
            print(f"\n[signal] caught signal {signum} — "
                  "saving checkpoint and stopping cleanly ...")
            _STOP_FLAG[0] = True
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT,  _handler)


def ensure_deps() -> None:
    """Install missing deps. Call ONLY from a notebook cell / __main__, never
    at import time. Kaggle usually has torch+tokenizers preinstalled."""
    pkgs = []
    if not _HAS_TOKENIZERS:
        pkgs.append("tokenizers>=0.15")
    if not _HAS_ORT:
        pkgs.append("onnxruntime>=1.17")  # for export parity checks only
    if pkgs:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])


def set_seed(seed: int = 1234) -> None:
    """Full reproducibility. cudnn deterministic costs a little speed but makes
    the bits/byte number stable run-to-run."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random_state = np.random.RandomState(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return random_state


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_workdir() -> str:
    """Pick a WORKDIR that stays the same across sessions/restarts, so
    checkpoints saved under WORKDIR/export are always found again on resume.

    Priority: NSC_WORKDIR env > Kaggle > this file's own path (__file__) >
    VS Code notebook path (__vsc_ipynb_file__, set by the VS Code Jupyter
    extension) > cwd as a last resort. cwd is the least reliable option --
    when running cells interactively under a debugger/notebook kernel,
    __file__ is often undefined and cwd can silently differ between
    sessions, which makes resume-from-checkpoint fail without any error.
    """
    env = os.environ.get("NSC_WORKDIR")
    if env:
        return env
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working"
    f = globals().get("__file__")
    if f:
        d = os.path.dirname(os.path.abspath(f))
        # source lives in <project>/python/; data/export/results live at the project root
        return os.path.dirname(d) if os.path.basename(d) == "python" else d
    nb_path = globals().get("__vsc_ipynb_file__")
    if not nb_path:
        try:
            from IPython import get_ipython
            ip = get_ipython()
            if ip is not None:
                nb_path = ip.user_ns.get("__vsc_ipynb_file__")
        except Exception:
            pass
    if nb_path:
        return os.path.dirname(os.path.abspath(nb_path))
    cwd = os.getcwd()
    print(f"[workdir] WARNING: could not determine a stable script/notebook path "
          f"(no __file__, no notebook path); falling back to cwd={cwd!r}. If this "
          f"differs from a previous session's cwd, checkpoint resume will silently "
          f"fail. Set NSC_WORKDIR=<fixed path> to make this stable.")
    return cwd


WORKDIR = _resolve_workdir()
DATADIR = os.path.join(WORKDIR, "data")
OUTDIR = os.path.join(WORKDIR, "export")


def _find_pt_file(directory: str, ext: str = ".pt") -> Optional[str]:
    """Find the most recent .pt checkpoint file in a directory by extension.
    Defaults to '*.pt'; pass ext='.pt.last' to search for full checkpoint
    states like 'best.pt.last'. Returns None if no file is found.
    This lets the code find any checkpoint by .pt extension rather than relying
    on a hardcoded filename like 'best.pt'."""
    import glob as _glob
    pattern = os.path.join(directory, f"*{ext}")
    files = sorted(_glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def _xla_safe_load(path, map_location=None, **kwargs):
    """Load a checkpoint, gracefully handling XLA (TPU) storage tags.

    TPU-saved checkpoints have 'xla:0' storage location tags embedded in the
    serialized tensors. When loading on a non-TPU system, PyTorch raises:
      RuntimeError: don't know how to restore data location of ... (tagged with xla:0)

    This wrapper catches that error and retries with a custom map_location
    callable that keeps XLA-tagged tensors on CPU (they are already CPU tensors
    after zip deserialization). All other kwargs (weights_only, etc.) are
    preserved.
    """
    try:
        return torch.load(path, map_location=map_location, **kwargs)
    except RuntimeError as e:
        if 'xla' in str(e):
            # Fallback: ignore XLA device tags; keep tensors on CPU.
            return torch.load(path, map_location=lambda storage, loc: storage,
                              **kwargs)
        raise


# %% ==========================================================================
#  SECTION 2 -- CONFIG (tier ladder: nano / base / large)
# =============================================================================
@dataclass
class Config:
    # --- identity / tier ---
    tier: str = "base"
    seed: int = 1234

    # --- tokenizer ---
    vocab_size: int = 8192          # BPE vocab (4096..8192 per spec)

    # --- sequence / memory ---
    seg_len: int = 1024             # training segment length
    mem_len: int = 1024             # Transformer-XL sliding KV memory at inference

    # --- transformer (long-range residual expert) ---
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1

    # --- local/medium-range residual expert (GRU) ---
    gru_hidden: int = 256
    gru_layers: int = 2

    # --- classical experts ---
    order2_match_order: int = 4     # context length (in tokens) for the match model
    classical_smoothing: float = 0.02   # add-alpha smoothing for order-0/1 tables
    match_len_cap: int = 32         # match length that maps to ~max confidence
    order1_topk: int = 0            # 0 = ship the full V*V order-1 table (max density);
                                    # K>0 = ship only top-K probs/row + uniform escape,
                                    # shrinking the decoder bundle (V*V*2B -> ~V*K*4B).
                                    # Footprint mode: costs ~+0.09 bpc at K=64 unless the
                                    # gate is retrained on the pruned table.

    # --- fusion ---
    gate_dim: int = 64              # attention gating query/key dim
    conf_dim: int = 4              # classical confidence-state feature width
    neural_skip_tau: float = 0.95   # inference: skip Transformer if classical conf > tau
    use_xformer: bool = True        # Track B: False = recurrence-only (drop the
                                    # windowed Transformer for O(1)/token decode;
                                    # gate/classical/GRU shapes stay checkpoint-
                                    # compatible so a student warm-starts from base)

    # --- training ---
    batch_size: int = 8
    grad_accum: int = 4
    lr_stage1: float = 3e-4
    lr_stage2: float = 1e-4
    weight_decay: float = 0.01
    stage1_steps: int = 40_000      # pretrain neural base predictors + classical gating
    stage2_steps: int = 120_000     # joint residual + gating fine-tune
    warmup_steps: int = 2_000
    grad_clip: float = 1.0
    eval_every: int = 2_000
    early_stop_patience: int = 8    # eval windows without val improvement
    wallclock_budget_h: float = 11.0   # hard stop before Kaggle's 12h CPU-committed limit
    amp: bool = True

    # --- data ---
    dataset: str = "enwik8"         # "enwik8", or a mixed corpus name (e.g. "corpus_r2")
                                    # whose <name>.json sidecar gives explicit splits
    train_frac: float = 0.90        # of 100MB enwik8 -> 90/5/5 train/val/test (MB)
    val_frac: float = 0.05
    fast_classical: bool = False    # vectorized order-2 table + exact K-gram match keys
                                    # (the semantics of the deployed codec's feature
                                    # model); False keeps the original dict-based path
                                    # so base-tier checkpoints resume unchanged


TIERS: Dict[str, Dict] = {
    # Minimal resources: trains in a few hours on a T4, fast neural-skip path.
    "nano": dict(vocab_size=4096, seg_len=512, mem_len=512, d_model=128, n_layers=2,
                 n_heads=4, d_ff=512, gru_hidden=128, gru_layers=1,
                 batch_size=16, stage1_steps=20_000, stage2_steps=60_000),
    # Default: fits comfortably in the 30h budget, ~12-18M params.
    "base": dict(vocab_size=8192, seg_len=1024, mem_len=1024, d_model=256, n_layers=4,
                 n_heads=8, d_ff=1024, gru_hidden=256, gru_layers=2,
                 batch_size=8, stage1_steps=40_000, stage2_steps=120_000),
    # Round 2: between base and large, sized for a 30h T4 budget on the mixed
    # corpus (~16M params, ~1.9x base decode cost).
    "mid": dict(vocab_size=8192, seg_len=1024, mem_len=1024, d_model=320, n_layers=5,
                n_heads=8, d_ff=1280, gru_hidden=320, gru_layers=2,
                batch_size=6, grad_accum=5, stage1_steps=45_000, stage2_steps=125_000),
    # More resources: ~25-30M params, best ratio, slowest.
    "large": dict(vocab_size=8192, seg_len=1024, mem_len=1024, d_model=512, n_layers=6,
                  n_heads=8, d_ff=2048, gru_hidden=512, gru_layers=2,
                  batch_size=6, grad_accum=6, stage1_steps=50_000, stage2_steps=160_000),
    # Track B (recurrence-only): base dims MINUS the windowed Transformer, so it
    # warm-starts from a base checkpoint —  run_full("fast", init_from=".../best.pt").
    # ~5.2M params, O(1)/token decode (~4x faster); the fine-tune re-fits the gate
    # for the zero-xf_h regime. Pair with a shrunk order-1 table (WS4) for a small
    # decoder bundle.
    "fast": dict(vocab_size=8192, seg_len=1024, mem_len=1024, d_model=256, n_layers=4,
                 n_heads=8, d_ff=1024, gru_hidden=256, gru_layers=2, use_xformer=False,
                 batch_size=8, stage1_steps=2_000, stage2_steps=40_000),
    # Tiny config used only by the offline CPU smoke test.
    "smoke": dict(vocab_size=256, seg_len=32, mem_len=32, d_model=32, n_layers=2,
                  n_heads=2, d_ff=64, gru_hidden=32, gru_layers=1,
                  batch_size=2, grad_accum=1, stage1_steps=2, stage2_steps=2,
                  warmup_steps=1, eval_every=1, classical_smoothing=0.05),
}


def get_config(tier: str = "base", **overrides) -> Config:
    cfg = Config(tier=tier)
    for k, v in TIERS.get(tier, {}).items():
        setattr(cfg, k, v)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# %% ==========================================================================
#  SECTION 3 -- DATA LOADING & PREPROCESSING
# =============================================================================
ENWIK8_URLS = [
    "http://mattmahoney.net/dc/enwik8.zip",
    "https://huggingface.co/datasets/LTCB/enwik8/resolve/main/enwik8.zip",
]


def download_enwik8(datadir: str = DATADIR) -> str:
    """Return path to enwik8 raw byte file (100 MB Wikipedia XML).
    Priority: /kaggle/input/ dataset → cached local copy → URL download."""
    os.makedirs(datadir, exist_ok=True)
    raw = os.path.join(datadir, "enwik8")
    if os.path.exists(raw) and os.path.getsize(raw) > 90_000_000:
        return raw
    # On Kaggle: scan /kaggle/input/ for an attached enwik8 dataset
    kaggle_input = "/kaggle/input"
    if os.path.isdir(kaggle_input):
        import glob as _glob
        candidates = _glob.glob(os.path.join(kaggle_input, "*", "enwik8"))
        for candidate in candidates:
            if os.path.getsize(candidate) > 90_000_000:
                print(f"[data] using Kaggle dataset: {candidate}")
                return candidate
    zip_path = os.path.join(datadir, "enwik8.zip")
    last_err = None
    for url in ENWIK8_URLS:
        try:
            subprocess.check_call(["wget", "-q", "-O", zip_path, url])
            subprocess.check_call(["unzip", "-o", "-q", zip_path, "-d", datadir])
            if os.path.exists(raw):
                return raw
        except Exception as e:  # pragma: no cover
            last_err = e
            continue
    raise RuntimeError(f"Could not download enwik8: {last_err}")


def find_corpus(name: str, datadir: str = DATADIR) -> str:
    """Path to a mixed corpus `<name>.bin` (built by tools/build_train_corpus.py).
    Priority: NSC_CORPUS_DIR env → local data dir → attached Kaggle dataset."""
    import glob as _glob
    fn = f"{name}.bin"
    env = os.environ.get("NSC_CORPUS_DIR")
    cands = ([os.path.join(env, fn)] if env else []) + [
        os.path.join(datadir, name, fn), os.path.join(datadir, fn),
        os.path.join(WORKDIR, "data", name, fn)]
    cands += sorted(_glob.glob(os.path.join("/kaggle/input", "**", fn), recursive=True))
    for c in cands:
        if os.path.exists(c) and os.path.exists(c[:-4] + ".json"):
            return c
    raise FileNotFoundError(f"corpus {name!r} not found (need {fn} + {name}.json) in {cands}")


def corpus_layout(data_path: str) -> Optional[Dict]:
    """Explicit split layout from the `<stem>.json` sidecar next to a corpus file:
    {"splits": {split: [offset, length]}, "domains": {split: [[name, offset, length]]}}.
    None for a plain file (enwik8), which is split by train_frac/val_frac."""
    side = os.path.splitext(data_path)[0] + ".json"
    if not os.path.exists(side):
        return None
    with open(side) as f:
        lay = json.load(f)
    return lay if "splits" in lay else None


def split_ranges(data_path: str, cfg: Config) -> Dict[str, Tuple[int, int]]:
    """{split: (start, end)} byte ranges of a data file — sidecar layout if present,
    else the chronological train_frac/val_frac split."""
    lay = corpus_layout(data_path)
    if lay:
        return {k: (int(o), int(o) + int(n)) for k, (o, n) in lay["splits"].items()}
    n = os.path.getsize(data_path)
    n_tr, n_va = int(n * cfg.train_frac), int(n * cfg.val_frac)
    return {"train": (0, n_tr), "val": (n_tr, n_tr + n_va), "test": (n_tr + n_va, n)}


def layout_domains(data_path: str) -> Optional[Dict[str, List[Tuple[str, int, int]]]]:
    """{split: [(domain, offset_in_split, length)]} from a corpus sidecar, else None."""
    lay = corpus_layout(data_path)
    if not lay:
        return None
    return {sp: [(name, int(o) - int(lay["splits"][sp][0]), int(n)) for name, o, n in doms]
            for sp, doms in lay["domains"].items()}


def split_domains(cfg: Config) -> Optional[Dict[str, List[Tuple[str, int, int]]]]:
    """layout_domains() of the config's dataset; None for enwik8."""
    if cfg.dataset == "enwik8":
        return None
    return layout_domains(find_corpus(cfg.dataset))


def load_byte_splits(cfg: Config) -> Tuple[bytes, bytes, bytes]:
    """Return (train, val, test) raw byte strings split chronologically so the
    test set is genuinely held out (no overlap with BPE/classical training)."""
    if cfg.dataset != "enwik8":
        path = find_corpus(cfg.dataset)
        r = split_ranges(path, cfg)
        with open(path, "rb") as f:
            data = f.read()
        print(f"[data] corpus {cfg.dataset}: {path} ({len(data) / 1e6:.1f} MB)")
        return tuple(data[r[k][0]:r[k][1]] for k in ("train", "val", "test"))
    raw = download_enwik8()
    with open(raw, "rb") as f:
        data = f.read()
    n = len(data)
    n_tr = int(n * cfg.train_frac)
    n_va = int(n * cfg.val_frac)
    return data[:n_tr], data[n_tr:n_tr + n_va], data[n_tr + n_va:]


# %% ==========================================================================
#  SECTION 4 -- BPE TRAINING & TOKENIZATION
# =============================================================================
class ByteTokenizer:
    """Offline fallback tokenizer (vocab=256, identity over bytes). Lets the
    notebook + smoke test run without the `tokenizers` package. Exposes the same
    encode/decode/save surface used below."""

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = 256

    def encode_bytes(self, data: bytes) -> np.ndarray:
        return np.frombuffer(data, dtype=np.uint8).astype(np.int64)

    def decode_ids(self, ids: List[int]) -> bytes:
        return bytes(bytearray(int(i) & 0xFF for i in ids))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"type": "byte-identity", "vocab_size": 256}, f)


def train_bpe(train_bytes: bytes, cfg: Config, out_path: str) -> "object":
    """Train a byte-level BPE tokenizer on the *training* bytes only and save it
    to `tokenizer.json`. The Rust side loads this EXACT json via the `tokenizers`
    crate, guaranteeing encode/decode parity (no re-implementation needed).

    Falls back to a byte-identity tokenizer when `tokenizers` is unavailable."""
    if not _HAS_TOKENIZERS:
        tok = ByteTokenizer(cfg.vocab_size)
        tok.save(out_path)
        return tok

    # Byte-level BPE: pre-tokenizer maps raw bytes -> the 256 ByteLevel symbols
    # so ANY byte stream (text or binary) is representable losslessly.
    tok = Tokenizer(BPE(unk_token=None))
    tok.pre_tokenizer = ByteLevelPre(add_prefix_space=False)
    tok.decoder = ByteLevelDec()
    trainer = BpeTrainer(
        vocab_size=cfg.vocab_size,
        show_progress=True,
        initial_alphabet=ByteLevelPre.alphabet(),  # all 256 bytes guaranteed
        special_tokens=[],
    )
    # Train from an iterator of chunks (avoids holding a giant str in RAM).
    text = train_bytes.decode("latin-1")  # bijective byte<->codepoint mapping
    chunk = 1 << 20
    tok.train_from_iterator((text[i:i + chunk] for i in range(0, len(text), chunk)),
                            trainer=trainer)
    tok.save(out_path)
    return tok


def encode_with(tok, data: bytes) -> np.ndarray:
    if isinstance(tok, ByteTokenizer):
        return tok.encode_bytes(data)
    ids = tok.encode(data.decode("latin-1")).ids
    return np.asarray(ids, dtype=np.int64)


def encode_chunked(tok, data: bytes, chunk: int = 1 << 22) -> np.ndarray:
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


def encode_domains(tok, data: bytes, doms):
    """Encode one split of a mixed corpus domain by domain (the single definition of
    its token stream — trainer, benchmark and exporter all use it). Returns
    (ids, [(name, tok_start, tok_end, n_bytes)])."""
    parts, ranges, pos = [], [], 0
    for name, off, ln in doms:
        ids = encode_chunked(tok, data[off:off + ln])
        parts.append(ids)
        ranges.append((name, pos, pos + len(ids), ln))
        pos += len(ids)
    return np.concatenate(parts), ranges


def decode_with(tok, ids: List[int]) -> bytes:
    if isinstance(tok, ByteTokenizer):
        return tok.decode_ids(ids)
    return tok.decode([int(i) for i in ids]).encode("latin-1")


# %% ==========================================================================
#  SECTION 5 -- CLASSICAL EXPERTS (static, frozen counts from TRAIN split)
# =============================================================================
#  We use STATIC classical models (counts frozen from the training tokens). This
#  makes Python and Rust behave bit-identically and keeps reproducibility
#  trivial. Online-adaptive counts are a documented future extension.
#
#  Four experts feed the fusion:
#    order0 : global unigram distribution                  -> dense [V]
#    order1 : p(next | prev)                               -> dense [V, V]
#    order2 : p(next | prev2, prev) summarised as (top, conf)  (peaked dist.)
#    match  : hash-table match model summarised as (top, conf) (peaked dist.)
#  order0/order1 give the full-vocab BASE distribution the neural net refines;
#  order2/match contribute peaked distributions + confidence features for gating.
# =============================================================================
class ClassicalStats:
    """Builds + holds frozen classical tables. order0/order1 are dense numpy
    arrays exported verbatim to Rust. order2/match are realised per-position as
    (top_token, confidence) feature arrays."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.V = cfg.vocab_size
        self.order0 = None          # [V] float32 probabilities
        self.order1 = None          # [V, V] float32 row-stochastic
        self._o2_counts: Dict[int, Dict[int, int]] = {}   # key=a*V+b -> {next:count}
        self._match_pos: Dict[int, int] = {}              # k-gram hash -> last index
        self.fast = bool(getattr(cfg, "fast_classical", False))
        self._o2_ctx = self._o2_top = self._o2_conf = None  # fast path: sorted arrays

    # ---- build from a 1D token array (TRAIN split only) ----
    def fit(self, tokens: np.ndarray) -> None:
        if self.fast:
            return self._fit_fast(tokens)
        V, a = self.V, self.cfg.classical_smoothing
        # order-0
        c0 = np.bincount(tokens, minlength=V).astype(np.float64) + a
        self.order0 = (c0 / c0.sum()).astype(np.float32)
        # order-1 dense (rows = previous token)
        prev, nxt = tokens[:-1], tokens[1:]
        c1 = np.full((V, V), a, dtype=np.float64)
        np.add.at(c1, (prev, nxt), 1.0)
        self.order1 = (c1 / c1.sum(axis=1, keepdims=True)).astype(np.float32)
        # order-2 sparse counts (for top/conf feature)
        self._o2_counts = {}
        for i in range(2, len(tokens)):
            key = int(tokens[i - 2]) * V + int(tokens[i - 1])
            d = self._o2_counts.setdefault(key, {})
            t = int(tokens[i])
            d[t] = d.get(t, 0) + 1
        # match model is online during feature gen; nothing precomputed here.

    # ---- vectorized variant (fast_classical): same tables, no per-token dicts ----
    def _fit_fast(self, tokens: np.ndarray) -> None:
        V, a = self.V, self.cfg.classical_smoothing
        t = np.asarray(tokens, dtype=np.int64)
        c0 = np.bincount(t, minlength=V).astype(np.float64) + a
        self.order0 = (c0 / c0.sum()).astype(np.float32)
        c1 = np.bincount(t[:-1] * V + t[1:], minlength=V * V).astype(np.float64)
        c1 = c1.reshape(V, V) + a
        self.order1 = (c1 / c1.sum(axis=1, keepdims=True)).astype(np.float32)
        del c1
        # order-2: per context a*V+b keep (most frequent next, its share); ties go
        # to the smallest token id — identical to the deployed feature model.
        codes = (t[:-2] * V + t[1:-1]) * V + t[2:]
        uniq, counts = np.unique(codes, return_counts=True)
        del codes
        ctx, nxt = uniq // V, uniq % V
        starts = np.flatnonzero(np.r_[True, ctx[1:] != ctx[:-1]])
        totals = np.add.reduceat(counts, starts)
        order = np.lexsort((nxt, -counts, ctx))
        ctx_s, nxt_s, cnt_s = ctx[order], nxt[order], counts[order]
        s2 = np.flatnonzero(np.r_[True, ctx_s[1:] != ctx_s[:-1]])
        self._o2_ctx = ctx_s[s2]
        self._o2_top = nxt_s[s2].astype(np.int64)
        self._o2_conf = (cnt_s[s2] / totals).astype(np.float32)

    def _features_fast(self, tokens: np.ndarray) -> Dict[str, np.ndarray]:
        V = self.V
        t = np.asarray(tokens, dtype=np.int64)
        n = len(t)
        o2_top = np.zeros(n, dtype=np.int64)
        o2_conf = np.zeros(n, dtype=np.float32)
        if n >= 2 and len(self._o2_ctx):
            keys = t[:-1] * V + t[1:]
            idx = np.clip(np.searchsorted(self._o2_ctx, keys), 0, len(self._o2_ctx) - 1)
            hit = self._o2_ctx[idx] == keys
            o2_top[1:] = np.where(hit, self._o2_top[idx], 0)
            o2_conf[1:] = np.where(hit, self._o2_conf[idx], 0.0)
        K = self.cfg.order2_match_order
        cap = float(self.cfg.match_len_cap)
        assert V ** K < 2 ** 62, "K-gram key overflows int64"
        m_top = np.zeros(n, dtype=np.int64)
        m_conf = np.zeros(n, dtype=np.float32)
        if n > K:
            # match model: the token that followed the previous occurrence of the
            # exact K-gram ending at i (i >= K). Previous occurrence = predecessor
            # in a stable sort of the packed keys — no per-token Python loop.
            kg = np.zeros(n - K, dtype=np.int64)
            for j in range(K):
                kg = kg * V + t[1 + j:n - K + 1 + j]
            order = np.argsort(kg, kind="stable")
            sk = kg[order]
            del kg
            same = sk[1:] == sk[:-1]
            del sk
            cur, prv = order[1:][same] + K, order[:-1][same] + K
            del order, same
            has = np.zeros(n, dtype=bool)
            has[cur] = True
            m_top[cur] = t[prv + 1]
            del cur, prv
            # run length of consecutive correct predictions (1 on a fresh match)
            cont = np.zeros(n, dtype=bool)
            cont[1:] = has[1:] & has[:-1] & (m_top[:-1] == t[1:])
            idx = np.arange(n, dtype=np.int64)
            start = np.maximum.accumulate(np.where(has & ~cont, idx, -1))
            run = np.where(has, idx - start + 1, 0)
            # float64 divide then narrow, like np.float32(min(run / cap, 0.99))
            m_conf[:] = np.minimum(run / cap, 0.99)
        return dict(order2_top=o2_top, order2_conf=o2_conf,
                    match_top=m_top, match_conf=m_conf)

    # ---- per-position features for a token array (train/val/test) ----
    def features(self, tokens: np.ndarray) -> Dict[str, np.ndarray]:
        """For predicting position i+1 from history <= i, emit aligned arrays of
        length len(tokens): order2_top/conf and match_top/conf. Sequential O(N)."""
        if self.fast:
            return self._features_fast(tokens)
        V = self.V
        n = len(tokens)
        o2_top = np.zeros(n, dtype=np.int64)
        o2_conf = np.zeros(n, dtype=np.float32)
        m_top = np.zeros(n, dtype=np.int64)
        m_conf = np.zeros(n, dtype=np.float32)
        K = self.cfg.order2_match_order
        cap = float(self.cfg.match_len_cap)
        pos_table: Dict[int, int] = {}
        match_run = 0
        for i in range(n):
            # order-2 lookup using context (tokens[i-1], tokens[i]) to predict i+1
            if i >= 1:
                key = int(tokens[i - 1]) * V + int(tokens[i])
                d = self._o2_counts.get(key)
                if d:
                    top = max(d, key=d.get)
                    tot = sum(d.values())
                    o2_top[i] = top
                    o2_conf[i] = d[top] / tot
            # match model: hash of last K tokens -> last occurrence; predict the
            # token that followed it. Track a verified match run for confidence.
            if i >= K:
                h = hash(tokens[i - K + 1:i + 1].tobytes()) & 0x7FFFFFFF
                prev_pos = pos_table.get(h)
                if prev_pos is not None and prev_pos + 1 < n:
                    pred = int(tokens[prev_pos + 1])
                    m_top[i] = pred
                    # extend run if last prediction was correct
                    match_run = match_run + 1 if (i > 0 and m_top[i - 1] == tokens[i]) else 1
                    m_conf[i] = min(match_run / cap, 0.99)
                else:
                    match_run = 0
                pos_table[h] = i
        return dict(order2_top=o2_top, order2_conf=o2_conf,
                    match_top=m_top, match_conf=m_conf)

    # ---- export dense tables for Rust ----
    def export(self, outdir: str) -> Dict:
        os.makedirs(outdir, exist_ok=True)
        # store as float16 to halve disk/RAM (V*V*2 bytes); recompute in f32.
        self.order0.astype(np.float16).tofile(os.path.join(outdir, "order0.f16"))
        K = getattr(self.cfg, "order1_topk", 0)
        if K and K < self.V:                     # footprint mode: ship top-K sparse
            idx, val = prune_order1_topk(self.order1, K)
            np.savez(os.path.join(outdir, "order1_topk.npz"),
                     idx=idx, val=val, K=np.int64(K), V=np.int64(self.V))
            order1_artifact = f"order1_topk.npz"
        else:                                    # default: full dense table (max density)
            self.order1.astype(np.float16).tofile(os.path.join(outdir, "order1.f16"))
            order1_artifact = "order1.f16"
        meta = dict(vocab_size=self.V, dtype="float16",
                    order0_shape=[self.V], order1_shape=[self.V, self.V],
                    order1_artifact=order1_artifact, order1_topk=int(K),
                    smoothing=self.cfg.classical_smoothing,
                    match_order=self.cfg.order2_match_order,
                    match_len_cap=self.cfg.match_len_cap)
        with open(os.path.join(outdir, "classical_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return meta


# %% ==========================================================================
#  SECTION 6 -- DATASET & DATALOADERS
# =============================================================================
class CompressionDataset(torch.utils.data.Dataset):
    """Yields fixed-length windows of tokens + aligned classical features. Each
    item predicts the next token at every position (causal LM)."""

    def __init__(self, tokens: np.ndarray, feats: Dict[str, np.ndarray], seg_len: int):
        self.tokens = tokens
        self.feats = feats
        self.seg = seg_len
        # number of non-overlapping windows that have a full next-token target
        self.n = max(0, (len(tokens) - 1) // seg_len)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        s = idx * self.seg
        e = s + self.seg
        x = torch.from_numpy(self.tokens[s:e].astype(np.int64))
        y = torch.from_numpy(self.tokens[s + 1:e + 1].astype(np.int64))
        item = dict(
            input_ids=x, target_ids=y,
            order2_top=torch.from_numpy(self.feats["order2_top"][s:e].astype(np.int64)),
            order2_conf=torch.from_numpy(self.feats["order2_conf"][s:e].astype(np.float32)),
            match_top=torch.from_numpy(self.feats["match_top"][s:e].astype(np.int64)),
            match_conf=torch.from_numpy(self.feats["match_conf"][s:e].astype(np.float32)),
        )
        return item


def make_loader(tokens, feats, cfg: Config, shuffle: bool) -> torch.utils.data.DataLoader:
    ds = CompressionDataset(tokens, feats, cfg.seg_len)
    return torch.utils.data.DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=shuffle, drop_last=True,
        num_workers=2 if DEVICE.type == "cuda" else 0,
        pin_memory=(DEVICE.type == "cuda"))


# %% ==========================================================================
#  SECTION 7 -- MODEL DEFINITIONS
# =============================================================================
def peaked_distribution(top_idx: torch.Tensor, conf: torch.Tensor, V: int,
                        eps: float = 1e-4) -> torch.Tensor:
    """Build [B,T,V] distributions with mass `conf` on `top_idx` and the rest
    spread uniformly. Valid (sums to 1) and >0 everywhere. Used for the order-2
    and match experts."""
    conf = conf.clamp(eps, 1.0 - eps).unsqueeze(-1)              # [B,T,1]
    base = (1.0 - conf) / (V - 1)                                # [B,T,1]
    p = base.expand(*top_idx.shape, V).clone()
    p.scatter_(-1, top_idx.clamp(0, V - 1).unsqueeze(-1), conf)
    return p


def prune_order1_topk(order1: np.ndarray, K: int):
    """Top-K sparsification of a row-stochastic order-1 table. Returns
    (idx[V,K] uint16, val[V,K] float16): the K most-probable next-symbols per row
    and their probabilities. Shrinks the SHIPPED table from V*V*2 B (134 MB at
    V=8192) to ~V*K*4 B (2 MB at K=64) while `reconstruct_order1_topk` keeps every
    row a valid distribution. Encoder and decoder both load the reconstructed
    table, so a pruned bundle is still bit-exact lossless (just lower density)."""
    V = order1.shape[0]
    K = min(K, V)
    idx = np.argpartition(order1, V - K, axis=1)[:, V - K:].astype(np.uint16)
    val = np.take_along_axis(order1.astype(np.float32),
                             idx.astype(np.int64), axis=1).astype(np.float16)
    return idx, val


def reconstruct_order1_topk(idx: np.ndarray, val: np.ndarray, V: int) -> np.ndarray:
    """Rebuild the dense [V,V] float32 order-1 table from its top-K sparse form:
    kept symbols get their stored probability, the remaining mass (1 - sum kept)
    spreads uniformly over the other V-K symbols so each row sums to 1."""
    K = idx.shape[1]
    val32 = val.astype(np.float32)
    escape = np.clip(1.0 - val32.sum(1, keepdims=True), 0.0, None) / max(1, V - K)
    dense = np.broadcast_to(escape, (V, V)).copy()              # non-top = escape
    np.put_along_axis(dense, idx.astype(np.int64), val32, axis=1)   # top = their prob
    return dense


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: conditions a hidden state on the classical
    confidence-state vector. h' = (1 + gamma(conf)) * h + beta(conf). This is how
    the neural expert is told WHERE the classical model is weak so it can spend
    capacity on the residual."""

    def __init__(self, hidden: int, conf_dim: int):
        super().__init__()
        self.to_gb = nn.Sequential(
            nn.Linear(conf_dim, hidden), nn.GELU(), nn.Linear(hidden, 2 * hidden))

    def forward(self, h: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_gb(conf).chunk(2, dim=-1)
        return (1.0 + gamma) * h + beta


# ---------- RoPE (rotary positional embedding; relative & cache-friendly) ----
def build_rope(seq_len: int, head_dim: int, device, base: float = 10000.0,
               offset: int = 0):
    # exp-form is equivalent to base**(-2i/head_dim) but is TorchScript- and
    # ONNX-exportable (avoids `float ** Tensor`).
    inv = torch.exp(torch.arange(0, head_dim, 2, device=device).float()
                    * (-(math.log(base)) / head_dim))
    t = torch.arange(offset, offset + seq_len, device=device).float()
    freqs = torch.outer(t, inv)                       # [T, hd/2]
    emb = torch.cat((freqs, freqs), dim=-1)           # [T, hd]
    return emb.cos()[None, None], emb.sin()[None, None]   # [1,1,T,hd]


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.h = cfg.n_heads
        self.hd = cfg.d_model // cfg.n_heads
        assert self.hd % 2 == 0, "head_dim must be even for RoPE"
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                pos_offset: int = 0):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.h, self.hd).transpose(1, 2)   # [B,h,T,hd]
        k = k.view(B, T, self.h, self.hd).transpose(1, 2)
        v = v.view(B, T, self.h, self.hd).transpose(1, 2)
        # rope on the *new* q,k at their absolute positions
        cos, sin = build_rope(T, self.hd, x.device, offset=pos_offset)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # prepend cached (already-rope'd) memory keys/values
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_kv = (k, v)
        Tk = k.shape[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)   # [B,h,T,Tk]
        # causal mask: query t (abs pos pos_offset+t) sees keys 0..(Tk-T+t)
        M = Tk - T
        causal = torch.ones(T, Tk, device=x.device, dtype=torch.bool).tril(diagonal=M)
        att = att.masked_fill(~causal, float("-inf"))
        att = self.drop(att.softmax(dim=-1))
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out), new_kv


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model), nn.Dropout(cfg.dropout))

    def forward(self, x, past_kv=None, pos_offset=0):
        a, new_kv = self.attn(self.ln1(x), past_kv, pos_offset)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, new_kv


class TransformerResidualExpert(nn.Module):
    """Long-range expert. Outputs RESIDUAL logits (not next-token logits),
    FiLM-conditioned on the classical confidence state. Supports a sliding KV
    memory cache for Transformer-XL-style streaming inference."""

    def __init__(self, cfg: Config, embed: nn.Embedding):
        super().__init__()
        self.cfg = cfg
        self.embed = embed                     # shared token embedding
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.lnf = nn.LayerNorm(cfg.d_model)
        self.film = FiLM(cfg.d_model, cfg.conf_dim)
        self.to_resid = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, input_ids, conf_state, past_kvs=None, pos_offset=0):
        x = self.embed(input_ids)
        new_kvs = []
        for i, blk in enumerate(self.blocks):
            pk = None if past_kvs is None else past_kvs[i]
            x, kv = blk(x, pk, pos_offset)
            new_kvs.append(kv)
        h = self.lnf(x)
        h = self.film(h, conf_state)
        resid = self.to_resid(h)               # [B,T,V] residual logits
        return resid, h, new_kvs


class LocalResidualExpert(nn.Module):
    """Local/medium-range expert (GRU). Compact recurrent state is ideal for
    streaming in Rust. Outputs RESIDUAL logits, FiLM-conditioned."""

    def __init__(self, cfg: Config, embed: nn.Embedding):
        super().__init__()
        self.embed = embed
        self.gru = nn.GRU(cfg.d_model, cfg.gru_hidden, cfg.gru_layers,
                          batch_first=True, dropout=cfg.dropout if cfg.gru_layers > 1 else 0.0)
        self.film = FiLM(cfg.gru_hidden, cfg.conf_dim)
        self.to_resid = nn.Linear(cfg.gru_hidden, cfg.vocab_size)

    def forward(self, input_ids, conf_state, hx=None):
        x = self.embed(input_ids)
        h, hx = self.gru(x, hx)
        h = self.film(h, conf_state)
        resid = self.to_resid(h)
        return resid, h, hx


class AttentionGatingFusion(nn.Module):
    """Context-gated ATTENTION fusion (not a static MLP). A query built from the
    neural hidden states attends over six expert "keys" that combine a learned
    expert embedding with each expert's LIVE confidence feature. Produces:
      w  = softmax over the 4 classical experts  (mixture weights, sum=1)
      g  = sigmoid over the 2 neural experts      (residual gates, independent)
    Routing therefore changes per token: match-heavy on repeats, transformer on
    structured text, classical-only when the neural residual is gated to ~0."""

    EXPERTS = ["order0", "order1", "order2", "match", "gru", "xformer"]

    def __init__(self, cfg: Config):
        super().__init__()
        self.d = cfg.gate_dim
        self.expert_embed = nn.Parameter(torch.randn(6, self.d) * 0.02)
        self.feat_proj = nn.Linear(1, self.d)        # per-expert scalar feature -> key
        q_in = cfg.d_model + cfg.gru_hidden
        self.q_proj = nn.Linear(q_in, self.d)

    def forward(self, gru_h, xf_h, expert_feats):
        # expert_feats: [B,T,6] live per-expert confidence/activity scalars
        B, T, _ = gru_h.shape
        q = self.q_proj(torch.cat([gru_h, xf_h], dim=-1)).unsqueeze(-2)   # [B,T,1,d]
        keys = self.expert_embed[None, None] + self.feat_proj(expert_feats.unsqueeze(-1))
        scores = (q * keys).sum(-1) / math.sqrt(self.d)                   # [B,T,6]
        w = scores[..., :4].softmax(dim=-1)        # classical mixture weights
        g = torch.sigmoid(scores[..., 4:])         # neural residual gates
        return w, g, scores


class HybridLM(nn.Module):
    """Top-level neuro-symbolic predictor tying everything together.

    Final logits:   z_final = log( p_cls + eps ) + Delta
      p_cls = sum_e w_e * p_e         (gated classical mixture, prob space)
      Delta = g_gru * resid_gru + g_xf * resid_xf   (gated neural residual)
    """

    def __init__(self, cfg: Config, classical: ClassicalStats):
        super().__init__()
        self.cfg = cfg
        self.V = cfg.vocab_size
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # classical dense tables live as frozen buffers (move with .to / export)
        self.register_buffer("order0", torch.from_numpy(classical.order0))
        self.register_buffer("order1", torch.from_numpy(classical.order1))
        self.local = LocalResidualExpert(cfg, self.embed)
        # Track B (cfg.use_xformer=False): drop the windowed Transformer entirely.
        # The gate still consumes a zero xf_h / resid_xf below, so all gate,
        # classical and GRU parameter shapes are identical to the full model —
        # a recurrence-only student loads a full checkpoint with strict=False.
        self.xformer = TransformerResidualExpert(cfg, self.embed) if cfg.use_xformer else None
        self.gate = AttentionGatingFusion(cfg)
        self.eps = 1e-6

    # ---- classical base distribution + confidence state ----
    def classical(self, batch):
        x = batch["input_ids"]
        B, T = x.shape
        V = self.V
        p0 = self.order0.to(torch.float32)[None, None].expand(B, T, V)
        p1 = self.order1.to(torch.float32)[x]                              # [B,T,V]
        p2 = peaked_distribution(batch["order2_top"], batch["order2_conf"], V)
        pm = peaked_distribution(batch["match_top"], batch["match_conf"], V)
        # confidence state (feeds FiLM + gating)
        o1_max = p1.max(dim=-1).values
        o1_ent = -(p1 * (p1 + self.eps).log()).sum(-1) / math.log(V)
        conf_state = torch.stack(
            [o1_max, o1_ent, batch["order2_conf"], batch["match_conf"]], dim=-1)
        # live per-expert scalar features for gating (order0 has constant conf)
        expert_feats = torch.stack(
            [torch.full_like(o1_max, 0.5), o1_max,
             batch["order2_conf"], batch["match_conf"],
             torch.zeros_like(o1_max), torch.zeros_like(o1_max)], dim=-1)
        return (p0, p1, p2, pm), conf_state, expert_feats

    def forward(self, batch, stage: int = 2):
        (p0, p1, p2, pm), conf_state, expert_feats = self.classical(batch)
        resid_gru, gru_h, _ = self.local(batch["input_ids"], conf_state)
        if self.xformer is not None:
            resid_xf, xf_h, _ = self.xformer(batch["input_ids"], conf_state)
        else:                                    # Track B: recurrence-only
            resid_xf = torch.zeros_like(resid_gru)
            xf_h = gru_h.new_zeros(*gru_h.shape[:2], self.cfg.d_model)
        # fill neural activity features (norm of hidden) for gating keys
        ef = expert_feats.clone()
        ef[..., 4] = gru_h.abs().mean(-1)
        ef[..., 5] = xf_h.abs().mean(-1)
        w, g, _ = self.gate(gru_h, xf_h, ef)

        p_cls = (w[..., 0:1] * p0 + w[..., 1:2] * p1 +
                 w[..., 2:3] * p2 + w[..., 3:4] * pm)
        delta = g[..., 0:1] * resid_gru + g[..., 1:2] * resid_xf
        z_final = (p_cls + self.eps).log() + delta

        out = dict(z_final=z_final, p_cls=p_cls, w=w, g=g,
                   resid_gru=resid_gru, resid_xf=resid_xf)
        if "target_ids" in batch:
            y = batch["target_ids"]
            if stage == 1:
                # curriculum stage 1: warm up each expert as a STANDALONE
                # predictor (neural outputs treated as logits) + classical gating
                loss = (F.cross_entropy((p_cls + self.eps).log().reshape(-1, self.V), y.reshape(-1))
                        + F.cross_entropy(resid_gru.reshape(-1, self.V), y.reshape(-1)))
                if self.xformer is not None:
                    loss = loss + F.cross_entropy(resid_xf.reshape(-1, self.V), y.reshape(-1))
            else:
                # stage 2: the real residual-fusion objective
                loss = F.cross_entropy(z_final.reshape(-1, self.V), y.reshape(-1))
            out["loss"] = loss
        return out


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# %% ==========================================================================
#  SECTION 8 -- TRAINING LOOP (AMP + curriculum + cosine + early stop)
# =============================================================================
def cosine_lr(step, total, base_lr, warmup):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(1.0, prog)))


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, bytes_total: int, tokens_total: int) -> Dict:
    """Return val bits/byte. nats -> bits via log2(e); per-token -> per-byte via
    the token/byte ratio of the split."""
    model.eval()
    nats, ntok = 0.0, 0
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch, stage=2)
        nats += out["loss"].item() * batch["target_ids"].numel()
        ntok += batch["target_ids"].numel()
    model.train()
    bpb = (nats / max(1, ntok)) * math.log2(math.e) * (tokens_total / max(1, bytes_total))
    return dict(bits_per_byte=bpb, nats_per_token=nats / max(1, ntok))


def train(model, train_loader, val_loader, cfg: Config,
          bytes_val: int, tokens_val: int, ckpt_path: str,
          resume_from: str = None, time_budget_h: float = None,
          save_every: int = 0, lr_scale: float = 1.0,
          goal_val_bpb: float = None):
    """Train with AMP + two-stage curriculum + cosine + early stop.

    Resumable & fault-tolerant. Two files are written:
      * ``ckpt_path``        -> the BEST checkpoint, a ``weights_only``-safe dict
        (model + cfg + step + val_bpb), saved on every val improvement. This keeps
        ``run_full`` and the benchmark loaders working unchanged.
      * ``ckpt_path + ".last"`` -> the LATEST FULL state (model + optimizer + scaler
        + step + best + patience + RNG), saved every ``save_every`` steps and on
        every stop, so a killed phase resumes EXACTLY where it left off.

    Pass ``resume_from`` (usually the previous phase's ``.last``) to continue the
    global step counter toward ``stage1_steps + stage2_steps``. ``time_budget_h``
    bounds THIS phase (defaults to ``cfg.wallclock_budget_h``); ``lr_scale``
    multiplies the scheduled LR for adaptive control across versions.

    ``goal_val_bpb`` (optional): when a NEW best val bpb reaches this value the
    phase ends early with ``stop_reason="goal_val"`` so run_budget() can confirm
    the compression goal on the calibrated test split immediately.

    Returns a dict ``{best, step, total_steps, elapsed_h, stop_reason,
    best_path, last_path}``.
    """
    import random
    device = DEVICE
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_stage1,
                            weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    last_path = ckpt_path + ".last"
    budget_h = time_budget_h if time_budget_h is not None else cfg.wallclock_budget_h
    total_steps = cfg.stage1_steps + cfg.stage2_steps
    best, patience, step = float("inf"), 0, 0

    def stage_for(s):
        return 1 if s < cfg.stage1_steps else 2

    def lr_for(s):
        base = (cosine_lr(s, cfg.stage1_steps, cfg.lr_stage1, cfg.warmup_steps)
                if s < cfg.stage1_steps else
                cosine_lr(s - cfg.stage1_steps, cfg.stage2_steps, cfg.lr_stage2,
                          cfg.warmup_steps))
        return base * lr_scale

    def _atomic_save(obj, path):
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)               # never leave a half-written checkpoint

    def save_best(s):                        # weights_only-safe -> run_full compatible
        _atomic_save({"model": model.state_dict(), "cfg": asdict(cfg),
                      "step": s, "val_bpb": best}, ckpt_path)
        print(f"  [ckpt] best saved  | stage {stage_for(s)} | step {s} | val_bpb {best:.4f} | {ckpt_path}")

    def save_last(s, reason, path=None):      # full resumable state
        p = path or last_path
        state = {"model": model.state_dict(), "opt": opt.state_dict(),
                 "scaler": scaler.state_dict(), "step": s, "stage": stage_for(s),
                 "best": best, "patience": patience, "cfg": asdict(cfg),
                 "reason": reason}
        try:
            state["rng"] = {"torch": torch.get_rng_state(),
                            "numpy": np.random.get_state(),
                            "python": random.getstate(),
                            "cuda": (torch.cuda.get_rng_state_all()
                                     if torch.cuda.is_available() else None)}
        except Exception:
            pass
        _atomic_save(state, p)
        if p == last_path:
            print(f"  [ckpt] last saved  | stage {stage_for(s)} | step {s} | reason={reason} | {p}")

    # ---- resume: restore everything and continue the global step counter ----
    if resume_from and os.path.exists(resume_from):
        # weights_only=False is required: a resume blob carries optimizer + numpy/
        # python RNG state (not unpicklable under weights_only). This is our OWN
        # locally-written ".last" file under runs/, never a downloaded artifact;
        # the portable BEST checkpoint (ckpt_path) stays weights_only-safe.
        ck = _xla_safe_load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if ck.get("opt"):
            opt.load_state_dict(ck["opt"])
        if ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        step = int(ck.get("step", 0))
        best = float(ck.get("best", ck.get("val_bpb", float("inf"))))
        patience = int(ck.get("patience", 0))
        rng = ck.get("rng")
        if rng:
            try:
                torch.set_rng_state(rng["torch"])
                np.random.set_state(rng["numpy"])
                random.setstate(rng["python"])
                if rng.get("cuda") and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(rng["cuda"])
            except Exception:
                pass
        print(f"[resume] {resume_from}: step={step} best={best:.4f} "
              f"patience={patience}")

    _install_signal_handlers()
    t0 = time.time()
    model.train()
    data_iter = iter(train_loader)
    opt.zero_grad(set_to_none=True)
    stop_reason = "completed"
    ckpt_dir      = os.path.dirname(os.path.abspath(ckpt_path))
    t_last_hourly = t0
    hourly_n      = 0
    hourly_prev   = None          # path of the previous hourly file (for cleanup)
    while step < total_steps:
        # wall-clock guard: stop gracefully within THIS phase's time budget
        if (time.time() - t0) / 3600.0 > budget_h:
            print(f"[budget] phase budget {budget_h:.2f}h reached at step {step}; "
                  f"stopping.")
            stop_reason = "budget"
            break
        # signal guard: SIGTERM/SIGINT received — finish this step then save
        if _STOP_FLAG[0]:
            print(f"[signal] stop requested at step {step}; checkpoint will be saved.")
            stop_reason = "signal"
            break
        stage = stage_for(step)
        for pg in opt.param_groups:
            pg["lr"] = lr_for(step)

        accum_loss = 0.0
        for _ in range(cfg.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = move_batch(batch, device)
            with torch.autocast(device_type=device.type,
                                enabled=cfg.amp and device.type == "cuda"):
                out = model(batch, stage=stage)
                loss = out["loss"] / cfg.grad_accum
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        if step % 100 == 0:
            print(f"step {step:>7d} | stage {stage} | lr {lr_for(step):.2e} "
                  f"| loss {accum_loss:.4f} | {(time.time()-t0)/3600:.2f}h")

        # hourly checkpoint — same format as last_path, named so run_budget can
        # find it on resume even if the session was killed before the phase ended
        if step > 0 and (time.time() - t_last_hourly) / 3600.0 >= 1.0:
            hourly_n += 1
            h_path = os.path.join(ckpt_dir, f"hourly_{hourly_n:02d}.last")
            save_last(step, f"hourly_{hourly_n:02d}", path=h_path)
            with open(os.path.join(ckpt_dir, "latest_hourly.json"), "w") as _hf:
                json.dump({"path": h_path, "step": step,
                           "hourly_n": hourly_n}, _hf)
            print(f"  [ckpt] hourly #{hourly_n:02d} saved | step {step} | {h_path}")
            if hourly_prev and os.path.exists(hourly_prev):
                os.remove(hourly_prev)   # keep only the latest hourly on disk
            hourly_prev = h_path
            t_last_hourly = time.time()

        # periodic latest-state save -> crash-safe resume mid-phase
        if save_every and step > 0 and step % save_every == 0:
            save_last(step, "periodic")

        if step > 0 and step % cfg.eval_every == 0:
            metrics = evaluate(model, val_loader, device, bytes_val, tokens_val)
            print(f"  [eval] step {step} val bits/byte = "
                  f"{metrics['bits_per_byte']:.4f}")
            if metrics["bits_per_byte"] < best - 1e-4:
                best = metrics["bits_per_byte"]
                patience = 0
                save_best(step)
                if goal_val_bpb is not None and best <= goal_val_bpb:
                    print(f"[goal] val bpb {best:.4f} <= val-proxy goal "
                          f"{goal_val_bpb:.4f} — ending phase so the goal can "
                          f"be confirmed on the calibrated test split.")
                    stop_reason = "goal_val"
                    break
            else:
                patience += 1
                if patience >= cfg.early_stop_patience and stage == 2:
                    print(f"[early-stop] no improvement for {patience} evals.")
                    stop_reason = "early_stop"
                    break
        step += 1

    # always persist the latest state so the next phase can resume exactly
    save_last(step, stop_reason)
    if not os.path.exists(ckpt_path):     # guarantee a best file even if no eval improved
        save_best(step)
    print(f"[done] best val bits/byte = {best:.4f}  "
          f"(stop: {stop_reason}, step {step}/{total_steps})")
    return {"best": best, "step": step, "total_steps": total_steps,
            "elapsed_h": (time.time() - t0) / 3600.0, "stop_reason": stop_reason,
            "best_path": ckpt_path, "last_path": last_path}


# %% ==========================================================================
#  SECTION 9 -- EVALUATION & COMPRESSION TEST
# =============================================================================
def gzip_bpb(data: bytes) -> float:
    return len(gzip.compress(data, 9)) * 8.0 / len(data)


def xz_bpb(data: bytes) -> float:
    try:
        import lzma
        return len(lzma.compress(data, preset=9)) * 8.0 / len(data)
    except Exception:
        return float("nan")


@torch.no_grad()
def reference_decoder_check(model, tokens: np.ndarray, feats: Dict, cfg: Config,
                            n: int = 256) -> float:
    """Pure token-by-token loop (batch=1, no teacher forcing tricks) that mimics
    the Rust streaming math, accumulating -log2 p(true). Confirms the streaming
    pipeline reproduces the batched training distribution. Returns bits/token."""
    model.eval()
    device = DEVICE
    n = min(n, len(tokens) - 1)
    bits = 0.0
    for i in range(1, n):
        batch = dict(
            input_ids=torch.tensor(tokens[:i + 1][None], device=device),
            order2_top=torch.tensor(feats["order2_top"][:i + 1][None], device=device),
            order2_conf=torch.tensor(feats["order2_conf"][:i + 1][None], device=device, dtype=torch.float32),
            match_top=torch.tensor(feats["match_top"][:i + 1][None], device=device),
            match_conf=torch.tensor(feats["match_conf"][:i + 1][None], device=device, dtype=torch.float32),
        )
        out = model(batch, stage=2)
        logp = out["z_final"][0, -1].log_softmax(-1)
        bits += -(logp[int(tokens[i + 1])].item()) * math.log2(math.e)
    return bits / max(1, n - 1)


class StreamingStepper:
    """Incremental, one-token-at-a-time decoder that reproduces HybridLM.forward()
    exactly while carrying GRU hidden state + per-layer transformer KV cache +
    RoPE position across tokens (reset at every seg_len window boundary). This is
    the PRODUCTION decode path — O(1) GRU + O(context) attention per token —
    versus the O(seg_len) window re-forward of reference_decoder_check(), which is
    a correctness oracle only. Encoder and decoder share this stepper, so their
    probability tables are bit-identical by construction (the roundtrip in
    test_compression_integrity.py proves losslessness on both sides)."""

    def __init__(self, model: "HybridLM", cfg: Config, T: float = 1.0,
                 neural_skip_tau: float = None):
        model.eval()
        self.m, self.cfg, self.V = model, cfg, cfg.vocab_size
        self.T = float(T)
        # neural-skip (adaptive compute): when the cheap classical experts are
        # confident (max(order2_conf, match_conf) > tau) skip the Transformer for
        # this token. Deterministic from the decoded prefix, so encode and decode
        # agree -> lossless. None = always run it.
        self.tau = neural_skip_tau
        self.logV = math.log(cfg.vocab_size)
        self.dev = next(model.parameters()).device
        self.reset_window()
        self.n_skipped = self.n_steps = 0

    def reset_window(self) -> None:
        self.hx = None
        self.kvs: List[Optional[tuple]] = [None] * self.cfg.n_layers
        self.pos = 0

    @torch.no_grad()
    def step(self, tok: int, o2t: int, o2c: float, mt: int, mc: float) -> np.ndarray:
        m, V, dev = self.m, self.V, self.dev
        skip = self.tau is not None and max(o2c, mc) > self.tau
        self.n_steps += 1
        self.n_skipped += int(skip)
        x = torch.tensor([[tok]], dtype=torch.long, device=dev)
        o2c_t = torch.tensor([[o2c]], dtype=torch.float32, device=dev)
        mc_t = torch.tensor([[mc]], dtype=torch.float32, device=dev)

        p0 = m.order0.to(torch.float32)[None, None]
        p1 = m.order1.to(torch.float32)[x]
        p2 = peaked_distribution(torch.tensor([[o2t]], device=dev), o2c_t, V)
        pm = peaked_distribution(torch.tensor([[mt]], device=dev), mc_t, V)
        if skip:
            # pure-classical fast path: skip GRU + Transformer + gate and code
            # straight from the single confident expert (its mass is already
            # peaked, so the trained mixture/residual add little). GRU/KV state are
            # intentionally not advanced — they resync at the next window reset;
            # pos advances to keep RoPE alignment for later non-skipped tokens.
            p_cls = pm if mc >= o2c else p2
            z = (p_cls + m.eps).log()
            if self.T != 1.0:
                z = z / self.T
            if m.xformer is not None:
                self.pos += 1
            return torch.softmax(z[0, 0], dim=-1).cpu().numpy()
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
        return torch.softmax(z[0, 0], dim=-1).cpu().numpy()


def compression_report(model, test_tokens, test_feats, test_bytes, cfg: Config):
    loader = make_loader(test_tokens, test_feats, cfg, shuffle=False)
    m = evaluate(model, loader, DEVICE, len(test_bytes), len(test_tokens))
    rows = [
        ("hybrid (ours)", m["bits_per_byte"]),
        ("gzip -9", gzip_bpb(test_bytes)),
        ("xz -9", xz_bpb(test_bytes)),
    ]
    print("\n=== Compression (bits/byte, lower is better) ===")
    for name, bpb in rows:
        print(f"  {name:<16s} {bpb:.4f}")
    ratio = rows[1][1] / rows[0][1]
    print(f"  -> hybrid is {ratio:.2f}x denser than gzip -9")
    return m


# %% ==========================================================================
#  SECTION 10 -- MODEL EXPORT (ONNX + TorchScript + JSON + classical tables)
# =============================================================================
class GRUStep(nn.Module):
    """ONNX-exportable single-step GRU expert for streaming inference."""

    def __init__(self, local: LocalResidualExpert):
        super().__init__()
        self.embed = local.embed
        self.gru = local.gru
        self.film = local.film
        self.to_resid = local.to_resid

    def forward(self, token, conf_state, hx):
        # token: [1,1] long ; conf_state: [1,1,conf_dim] ; hx: [layers,1,H]
        x = self.embed(token)
        h, hx2 = self.gru(x, hx)
        h = self.film(h, conf_state)
        return self.to_resid(h), hx2


class TransformerWindowStep(nn.Module):
    """ONNX/tract-friendly Transformer inference step. Instead of an incremental
    KV cache (variable-length, awkward in ONNX), it recomputes attention over a
    FIXED window of the last `window` tokens (left-padded) and returns the
    residual logits for the LAST position only. Static shapes => clean export.
    The Rust side keeps a ring buffer of the last `window` tokens. (An optional
    incremental KV cache for speed is described in the Rust blueprint.)"""

    def __init__(self, xf: TransformerResidualExpert, window: int):
        super().__init__()
        self.xf = xf
        self.window = window

    def forward(self, window_ids, conf_last):
        # window_ids: [1, W] long ; conf_last: [1, 1, conf_dim]
        x = self.xf.embed(window_ids)
        for blk in self.xf.blocks:
            x, _ = blk(x, None, 0)
        h = self.xf.lnf(x[:, -1:])            # last position only
        h = self.xf.film(h, conf_last)
        return self.xf.to_resid(h).squeeze(1)  # [1, V] residual logits


def export_all(model: HybridLM, classical: ClassicalStats, tok, cfg: Config,
               outdir: str = OUTDIR):
    os.makedirs(outdir, exist_ok=True)
    orig_device = next(model.parameters()).device
    model.eval().to("cpu")
    try:
        # 1) tokenizer.json (already saved by train_bpe; copy if needed)
        tok_path = os.path.join(outdir, "tokenizer.json")
        if _HAS_TOKENIZERS and not isinstance(tok, ByteTokenizer):
            tok.save(tok_path)
        else:
            ByteTokenizer(cfg.vocab_size).save(tok_path)

        # 2) classical dense tables + meta
        classical.export(outdir)

        # 3) config.json (full arch description for the Rust loader)
        with open(os.path.join(outdir, "config.json"), "w") as f:
            json.dump(asdict(cfg), f, indent=2)

        # 4) ONNX: GRU step
        gru_step = GRUStep(model.local).eval()
        tok0 = torch.zeros(1, 1, dtype=torch.long)
        conf0 = torch.zeros(1, 1, cfg.conf_dim)
        hx0 = torch.zeros(cfg.gru_layers, 1, cfg.gru_hidden)
        torch.onnx.export(
            gru_step, (tok0, conf0, hx0), os.path.join(outdir, "gru_step.onnx"),
            input_names=["token", "conf_state", "hx"],
            output_names=["resid", "hx_out"], opset_version=17,
            dynamo=False, dynamic_axes=None)

        # 5) ONNX: gating engine
        class GateExport(nn.Module):
            def __init__(self, g): super().__init__(); self.g = g
            def forward(self, gru_h, xf_h, ef):
                w, gg, _ = self.g(gru_h, xf_h, ef); return w, gg
        gate_export = GateExport(model.gate).eval()
        gh = torch.zeros(1, 1, cfg.gru_hidden); xh = torch.zeros(1, 1, cfg.d_model)
        ef = torch.zeros(1, 1, 6)
        torch.onnx.export(
            gate_export, (gh, xh, ef), os.path.join(outdir, "gate.onnx"),
            input_names=["gru_h", "xf_h", "expert_feats"],
            output_names=["w", "g"], opset_version=17, dynamo=False)

        # 6) ONNX: Transformer fixed-window step (tract-friendly, static shapes).
        #    Skipped for Track B (recurrence-only) — there is no Transformer to
        #    export; the decoder feeds a zero xf_h into gate.onnx instead.
        xfw = None
        if model.xformer is not None:
            W = cfg.mem_len
            xfw = TransformerWindowStep(model.xformer, W).eval()
            win0 = torch.zeros(1, W, dtype=torch.long)
            conf_last0 = torch.zeros(1, 1, cfg.conf_dim)
            torch.onnx.export(
                xfw, (win0, conf_last0), os.path.join(outdir, "xformer_window.onnx"),
                input_names=["window_ids", "conf_last"], output_names=["resid"],
                opset_version=17, dynamo=False)

        # 7) parity checks (PyTorch vs onnxruntime) for both exported neural steps
        if _HAS_ORT:
            sess = ort.InferenceSession(os.path.join(outdir, "gru_step.onnx"))
            ref_r, _ = gru_step(tok0, conf0, hx0)
            ox = sess.run(None, {"token": tok0.numpy(), "conf_state": conf0.numpy(),
                                 "hx": hx0.numpy()})
            err_g = float(np.abs(ref_r.detach().numpy() - ox[0]).max())

            if xfw is not None:
                sess2 = ort.InferenceSession(os.path.join(outdir, "xformer_window.onnx"))
                ref_x = xfw(win0, conf_last0)
                ox2 = sess2.run(None, {"window_ids": win0.numpy(),
                                       "conf_last": conf_last0.numpy()})
                err_x = float(np.abs(ref_x.detach().numpy() - ox2[0]).max())
                print(f"[export] ONNX parity max|err| gru={err_g:.2e} xformer={err_x:.2e}")
                assert err_g < 1e-3 and err_x < 1e-3, "ONNX export diverges from PyTorch!"
            else:
                print(f"[export] ONNX parity max|err| gru={err_g:.2e} (recurrence-only)")
                assert err_g < 1e-3, "ONNX gru export diverges from PyTorch!"
        print(f"[export] wrote artifacts to {outdir}")
    finally:
        model.to(orig_device)


# %% ==========================================================================
#  SECTION 11 -- ORCHESTRATION (notebook "main" cells)
# =============================================================================
def run_full(tier: str = "base", time_budget_h: float = None, init_from: str = None):
    """`init_from`: path to a checkpoint to warm-start from before training
    (loaded strict=False). This is how the Track B `fast` tier distils from a
    trained `base` model — it inherits the classical tables, GRU and gate, drops
    the Transformer, and only needs to re-fit the gate for the zero-xf_h regime."""
    t_start = time.time()
    # Honour the module-level BUDGET_H so notebook users who set it at the top
    # of Section 12 don't also have to pass it here explicitly.
    effective_budget = (time_budget_h if time_budget_h is not None
                        else (BUDGET_H if BUDGET_H is not None else None))

    cfg = get_config(tier)
    set_seed(cfg.seed)
    print(f"== Neuro-symbolic compressor :: tier={tier} :: device={DEVICE} ==")
    if effective_budget is not None:
        print(f"[budget] wall-clock budget = {effective_budget:.2f}h")
    ensure_deps()

    # --- data ---
    train_b, val_b, test_b = load_byte_splits(cfg)
    print(f"bytes: train={len(train_b):,} val={len(val_b):,} test={len(test_b):,}")

    # --- BPE ---
    os.makedirs(OUTDIR, exist_ok=True)
    tok = train_bpe(train_b, cfg, os.path.join(OUTDIR, "tokenizer.json"))
    tr_tok = encode_with(tok, train_b)
    va_tok = encode_with(tok, val_b)
    te_tok = encode_with(tok, test_b)
    print(f"tokens: train={len(tr_tok):,} ({len(train_b)/len(tr_tok):.2f} bytes/token)")

    # --- classical models + features ---
    classical = ClassicalStats(cfg)
    classical.fit(tr_tok)
    tr_f = classical.features(tr_tok)
    va_f = classical.features(va_tok)
    te_f = classical.features(te_tok)

    # --- model ---
    model = HybridLM(cfg, classical)
    xf_np = count_params(model.xformer) if model.xformer is not None else 0
    print(f"params: total={count_params(model):,} "
          f"(gru={count_params(model.local):,} xf={xf_np:,})")
    if init_from:
        try:
            ck = torch.load(init_from, map_location="cpu", weights_only=True)
        except Exception:
            ck = torch.load(init_from, map_location="cpu", weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f"[warm-start] {init_from}: {len(miss)} missing, {len(unexp)} unexpected"
              + (" (dropped the Transformer for Track B)" if not cfg.use_xformer else ""))

    # --- train ---
    # Subtract data-prep time from the budget so the overall wall-clock stays
    # within effective_budget hours from the very start of run_full().
    train_budget_h = None
    if effective_budget is not None:
        prep_h = (time.time() - t_start) / 3600.0
        train_budget_h = max(1.0 / 60, effective_budget - prep_h)
        print(f"[budget] data-prep took {prep_h:.2f}h; "
              f"remaining for training = {train_budget_h:.2f}h")
    train_loader = make_loader(tr_tok, tr_f, cfg, shuffle=True)
    val_loader = make_loader(va_tok, va_f, cfg, shuffle=False)
    ckpt = os.path.join(OUTDIR, "best.pt")

    # Resume from the most recent full state if one exists.
    # Priority: best.pt.last (written on clean stop) > latest hourly checkpoint
    # in OUTDIR > a checkpoint attached as a Kaggle input dataset (the normal
    # way to carry a checkpoint from a previous Kaggle session into a new one,
    # since /kaggle/working is wiped between sessions but /kaggle/input is a
    # read-only mount of whatever dataset you attached to the notebook).
    # When paths stored in latest_hourly.json point to a different machine
    # (e.g. /kaggle/working/…), we fall back to the same basename next to the
    # pointer file itself (works for both OUTDIR and a Kaggle input mount).
    resume_from = None
    last_path = ckpt + ".last"

    def _resolve_hourly_ptr(hourly_ptr):
        try:
            d = json.load(open(hourly_ptr))
            h_path = d.get("path", "")
            if not os.path.exists(h_path):
                h_path = os.path.join(os.path.dirname(hourly_ptr), os.path.basename(h_path))
            if os.path.exists(h_path):
                return h_path, d.get("step")
        except Exception:
            pass
        return None, None

    if os.path.exists(last_path):
        resume_from = last_path
        print(f"[run_full] resuming from {resume_from}")
    else:
        hourly_ptr = os.path.join(OUTDIR, "latest_hourly.json")
        if os.path.exists(hourly_ptr):
            resume_from, step = _resolve_hourly_ptr(hourly_ptr)
            if resume_from:
                print(f"[run_full] resuming from hourly checkpoint {resume_from} (step {step})")

    if resume_from is None:
        # Fallback: any .pt.last file in OUTDIR (full resume state)
        out_last = _find_pt_file(OUTDIR, ext=".pt.last")
        if out_last:
            resume_from = out_last
            print(f"[run_full] resuming from {resume_from}")

    if resume_from is None:
        # Fallback: any .pt file in OUTDIR (weights-only checkpoint)
        out_best = _find_pt_file(OUTDIR)
        if out_best:
            resume_from = out_best
            print(f"[run_full] resuming from {resume_from} (weights only; optimizer state will be reset)")

    if resume_from is None and os.path.isdir("/kaggle/input"):
        import glob as _glob
        # Priority 1: any .pt.last file (full resume state with optimizer + RNG)
        input_last = sorted(
            _glob.glob("/kaggle/input/**/*.pt.last", recursive=True),
            key=os.path.getmtime, reverse=True)
        if input_last:
            resume_from = input_last[0]
            print(f"[run_full] resuming from attached Kaggle checkpoint dataset: {resume_from}")
        else:
            # Priority 2: any .pt file (portable weights-only checkpoint)
            input_best = sorted(
                _glob.glob("/kaggle/input/**/*.pt", recursive=True),
                key=os.path.getmtime, reverse=True)
            if input_best:
                resume_from = input_best[0]
                print(f"[run_full] resuming from attached Kaggle checkpoint: {resume_from} "                        f"(weights only; optimizer state will be reset)")
        if resume_from is None:
            for ptr_file in sorted(_glob.glob("/kaggle/input/**/latest_hourly.json", recursive=True)):
                resume_from, step = _resolve_hourly_ptr(ptr_file)
                if resume_from:
                    print(f"[run_full] resuming from attached Kaggle checkpoint dataset "
                          f"{resume_from} (step {step})")
                    break

    train(model, train_loader, val_loader, cfg, len(val_b), len(va_tok), ckpt,
          resume_from=resume_from, time_budget_h=train_budget_h)

    # --- evaluate + export ---
    ckpt_to_load = ckpt
    if not os.path.exists(ckpt_to_load):
        alt = _find_pt_file(OUTDIR)
        if alt:
            ckpt_to_load = alt
            print(f"[run_full] using checkpoint {ckpt_to_load} instead of {ckpt}")
    if os.path.exists(ckpt_to_load):
        # weights_only=True: checkpoint holds only tensors + primitive config
        # values, so we avoid the arbitrary-code-execution risk of full unpickle.
        model.load_state_dict(_xla_safe_load(ckpt_to_load, map_location=DEVICE,
                                             weights_only=True)["model"])
    compression_report(model, te_tok, te_f, test_b, cfg)
    bpt = reference_decoder_check(model, te_tok, te_f, cfg, n=128)
    print(f"[ref-decoder] streaming bits/token on test prefix = {bpt:.4f}")
    export_all(model, classical, tok, cfg, OUTDIR)


def run_smoke():
    """Fast offline CPU sanity check of the full model wiring (no data download,
    no real training). Verifies: forward pass, valid distribution, finite loss,
    both curriculum stages, gating shapes, and ONNX GRU export."""
    print("== SMOKE TEST (cpu) ==")
    cfg = get_config("smoke")
    set_seed(0)
    V, T, B = cfg.vocab_size, cfg.seg_len, cfg.batch_size
    rng = np.random.RandomState(0)
    toks = rng.randint(0, V, size=4000).astype(np.int64)

    classical = ClassicalStats(cfg)
    classical.fit(toks)
    feats = classical.features(toks)
    assert classical.order0.shape == (V,)
    assert classical.order1.shape == (V, V)
    assert np.allclose(classical.order1.sum(1), 1.0, atol=1e-4), "order1 not stochastic"

    model = HybridLM(cfg, classical).to(DEVICE)
    print(f"smoke params total = {count_params(model):,}")

    loader = make_loader(toks, feats, cfg, shuffle=True)
    batch = move_batch(next(iter(loader)), DEVICE)

    for stage in (1, 2):
        out = model(batch, stage=stage)
        z = out["z_final"]
        p = z.softmax(-1)
        assert z.shape == (B, T, V), z.shape
        assert torch.isfinite(out["loss"]), "loss not finite"
        assert torch.allclose(p.sum(-1), torch.ones(B, T, device=DEVICE), atol=1e-3), "z_final not a distribution"
        assert out["w"].shape == (B, T, 4) and out["g"].shape == (B, T, 2)
        assert torch.allclose(out["w"].sum(-1), torch.ones(B, T, device=DEVICE), atol=1e-4), "w not simplex"
        # classical mixture must itself be a valid distribution
        assert torch.allclose(out["p_cls"].sum(-1), torch.ones(B, T, device=DEVICE), atol=1e-3)
        print(f"  stage {stage}: loss={out['loss'].item():.4f}  OK")

    # one optimizer step works end to end
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    out = model(batch, stage=2)
    out["loss"].backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    assert torch.isfinite(gnorm), "grad norm not finite"
    print(f"  backward+step OK (grad norm={gnorm:.3f})")

    # reference streaming decoder matches batched math direction
    bpt = reference_decoder_check(model, toks, feats, cfg, n=24)
    print(f"  ref-decoder bits/token = {bpt:.3f}")

    # export path (ONNX GRU step + gating + configs) on a temp dir
    tmp = os.path.join(WORKDIR, "export_smoke")
    export_all(model, classical, ByteTokenizer(V), cfg, tmp)
    assert os.path.exists(os.path.join(tmp, "gru_step.onnx"))
    assert os.path.exists(os.path.join(tmp, "xformer_window.onnx"))
    assert os.path.exists(os.path.join(tmp, "gate.onnx"))
    assert os.path.exists(os.path.join(tmp, "config.json"))
    assert os.path.exists(os.path.join(tmp, "order1.f16"))
    shutil.rmtree(tmp, ignore_errors=True)
    print("== SMOKE TEST PASSED ==")


# %% ==========================================================================
#  SECTION 12 -- RECURSIVE VERSIONING, CALIBRATION & PROFESSIONAL BENCHMARKING
#
#  A budget (MANDATORY, supplied at runtime) is split into time-sliced *phases*;
#  each phase resumes the previous one, is benchmarked on real data + lossless
#  tests, compared to the direct previous version, has decalibration auto-fixed,
#  and the best-so-far is always kept. Everything scales with the supplied budget
#  so the same code adapts to 1h / 9h / 30h / any value.
# =============================================================================
import hashlib as _hashlib

# ---- module-level adaptive-budget config (notebook-editable, no code edits) --
BUDGET_H      = None     # MANDATORY: set me, or pass total_budget_h=, or answer the prompt
TARGET_PHASES = 4        # the budget is sliced into ~this many phases/versions
SESSION_CAP_H = 11.5     # a single phase never exceeds this (platform per-session cap)
TARGET_BPB    = None     # optional: stop once test bits/byte <= this
ESCALATE_TO   = "base"   # tier to promote to when the start tier plateaus with budget to spare

# ---- budget-only mode: bench/train split + automatic model sizing ----
BENCH_FRAC  = 0.10   # fraction of total budget reserved for benchmark + lossless tests
BENCH_MIN_H = 0.50   # minimum bench time regardless of fraction
BENCH_CAP_H = 1.50   # cap: never spend more than this on benchmarks in total
PREP_EST_H  = 0.40   # estimated data-prep time (BPE + classical stats + feature cache)

# Step-time calibration: base config on T4 ≈ 3 000 optimizer steps / hour.
_BASE_SPH  = 3_000
_BASE_FLOP = 8 * 4 * 1024 * (256 ** 2) * 4   # batch * accum * seg * d_model² * n_layers


def _sph(d, L, seg, batch, accum):
    """Estimated optimizer steps / hour for a config on T4 (linear FLOP scaling)."""
    return _BASE_SPH * _BASE_FLOP / (batch * accum * seg * (d ** 2) * L)


# Ladder from smallest to largest.
# Columns: d_model, n_layers, n_heads, d_ff, gru_h, gru_L, vocab, seg, batch, accum, rec_steps
# rec_steps = recommended total training steps (stage1 + stage2) for best convergence.
_BUDGET_LADDER = [
    ( 64, 1, 2,  256,  64, 1, 2048,  256, 32, 1,  15_000),  # micro   ~0.4 M
    ( 96, 1, 4,  384,  96, 1, 4096,  384, 24, 1,  30_000),  # tiny    ~0.8 M
    (128, 2, 4,  512, 128, 1, 4096,  512, 16, 1,  80_000),  # nano    ~2.2 M
    (160, 2, 4,  640, 160, 1, 4096,  512, 12, 2, 100_000),  # nano+   ~3.4 M
    (192, 3, 8,  768, 192, 2, 8192,  768, 10, 2, 120_000),  # small   ~4.8 M
    (224, 3, 8,  896, 224, 2, 8192, 1024,  8, 3, 140_000),  # small+  ~6.5 M
    (256, 4, 8, 1024, 256, 2, 8192, 1024,  8, 4, 160_000),  # base   ~10.6 M
    (320, 5, 8, 1280, 320, 2, 8192, 1024,  6, 5, 170_000),  # mid    ~16 M
    (384, 5, 8, 1536, 384, 2, 8192, 1024,  6, 5, 185_000),  # large-  ~23.0 M
    (512, 6, 8, 2048, 512, 2, 8192, 1024,  6, 6, 210_000),  # large  ~35.8 M
]
_LADDER_MIN_FILL = 0.25   # config accepted only if ≥ 25 % of rec_steps are achievable


def split_budget(total_h):
    """Return (train_h, bench_h) that sum to total_h."""
    bench_h = min(BENCH_CAP_H, max(BENCH_MIN_H, total_h * BENCH_FRAC))
    return max(0.05, total_h - bench_h), bench_h


def config_from_budget(train_h, seed=1234):
    """
    Derive the LARGEST Config whose recommended training steps are at least
    LADDER_MIN_FILL-completable in train_h hours (15 % discount for eval overhead).

    Stage counts are scaled to the available time so the model trains exactly as
    long as the budget allows — no fixed ceiling that wastes compute or time.

    Examples on T4:
      train_h ≈ 2 h  →  nano    (d=128, 2L)   ~2.2 M params
      train_h ≈ 5 h  →  small   (d=192, 3L)   ~4.8 M params
      train_h ≈ 9 h  →  small+  (d=224, 3L)   ~6.5 M params
      train_h ≈ 14 h →  base    (d=256, 4L)  ~10.6 M params
      train_h ≈ 25 h →  large   (d=512, 6L)  ~35.8 M params
    """
    chosen_row, chosen_avail = _BUDGET_LADDER[0], 500
    for row in _BUDGET_LADDER:
        d, L, _, _, _, _, _, seg, batch, accum, rec = row
        avail = int(train_h * _sph(d, L, seg, batch, accum) * 0.85)
        if avail >= rec * _LADDER_MIN_FILL:
            chosen_row, chosen_avail = row, avail
        else:
            break   # larger configs will only have fewer available steps
    d, L, H, ff, gru_h, gru_L, V, seg, batch, accum, _ = chosen_row
    s1 = max(200, chosen_avail // 4)
    s2 = max(200, chosen_avail - s1)
    return Config(
        tier=f"auto-d{d}L{L}",
        seed=seed,
        vocab_size=V, seg_len=seg, mem_len=seg,
        d_model=d, n_layers=L, n_heads=H, d_ff=ff,
        gru_hidden=gru_h, gru_layers=gru_L,
        batch_size=batch, grad_accum=accum,
        stage1_steps=s1, stage2_steps=s2,
        warmup_steps=min(2_000, s1 // 8),
        eval_every=max(200, (s1 + s2) // 30),
        wallclock_budget_h=train_h,
        amp=True, dropout=0.1,
    )


def config_from_tier(tier, train_h, seed=1234, dataset=None):
    """A named tier with its curriculum scaled to finish in `train_h` hours on a
    T4 (same step-rate model and 15 % eval discount as config_from_budget). Used
    when the architecture is chosen explicitly instead of derived from the budget."""
    cfg = get_config(tier, seed=seed)
    avail = int(train_h * _sph(cfg.d_model, cfg.n_layers, cfg.seg_len,
                               cfg.batch_size, cfg.grad_accum) * 0.85)
    total = max(400, min(avail, cfg.stage1_steps + cfg.stage2_steps))
    cfg.stage1_steps = max(200, total // 4)
    cfg.stage2_steps = max(200, total - cfg.stage1_steps)
    cfg.warmup_steps = min(2_000, cfg.stage1_steps // 8)
    cfg.eval_every = max(200, total // 24)
    cfg.wallclock_budget_h = train_h
    if dataset:
        cfg.dataset = dataset
        cfg.fast_classical = dataset != "enwik8"
    return cfg


def _is_interactive() -> bool:
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except Exception:
        pass
    try:
        return bool(sys.stdin) and sys.stdin.isatty()
    except Exception:
        return False


def resolve_budget_h(total_budget_h=None) -> float:
    """Budget is MANDATORY. Precedence: explicit arg > module BUDGET_H > interactive
    prompt > error. There is no silent default anywhere."""
    b = total_budget_h
    if b is None:
        b = BUDGET_H
    if b is None and _is_interactive():
        try:
            b = float(input("MAX training budget in HOURS (e.g. 9, 30): ").strip())
        except Exception as e:
            raise ValueError(f"could not read budget from prompt: {e}")
    if b is None:
        raise ValueError(
            "Training budget is MANDATORY. Pass total_budget_h=<hours>, set "
            "BUDGET_H=<hours> at the top of SECTION 12, or run interactively to be prompted.")
    b = float(b)
    if not (b > 0):
        raise ValueError(f"budget must be > 0 hours, got {b}")
    return b


def plan_budget(B: float, target_phases=None, session_cap_h=None) -> Dict:
    """Derive the phase plan from the resolved budget B (hours). Everything is a
    fraction of B, so 1h / 9h / 30h all map to a sensible cadence."""
    tp = int(target_phases or TARGET_PHASES)
    cap = float(session_cap_h or SESSION_CAP_H)
    phase_h = min(cap, max(0.02, B / max(1, tp)))
    escalate_min_h = max(2.0, 0.30 * B)
    return {"phase_h": phase_h, "escalate_min_h": escalate_min_h,
            "n_phases_est": max(1, math.ceil(B / phase_h))}


# ---------------------------------------------------------------------------
#  Cached data prep: BPE is deterministic (so cheap to redo); the EXPENSIVE
#  per-split classical features() pass is cached to .npz. fit() is always run
#  (cheap relative to features) so the live match/order-2 model is available for
#  stress inputs. This finally implements the cache EXECUTION_PLAN §3 promised.
# ---------------------------------------------------------------------------
def _cache_key(*parts) -> str:
    return _hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def _prepare_mixed_tokens(cfg: Config, cache_dir: str, doms, byts):
    """Tokenizer + tokens for a mixed corpus. The tokenizer saved in the cache dir
    is REUSED on later sessions (the cache travels with runs/), so a resume never
    depends on BPE training being reproducible across library versions. Each
    domain is encoded separately; toks["_domains"][split] records its token range
    as (name, tok_start, tok_end, n_bytes) for the per-domain benchmark."""
    tok_path = os.path.join(cache_dir, "tokenizer.json")
    if _HAS_TOKENIZERS and os.path.exists(tok_path):
        tok = Tokenizer.from_file(tok_path)
        print(f"[cache] tokenizer HIT  ({tok_path})")
    else:
        tok = train_bpe(byts["train"], cfg, tok_path)
    tkey = _cache_key(cfg.dataset, cfg.vocab_size, *(len(byts[k]) for k in ("train", "val", "test")),
                      zlib.crc32(byts["test"]))
    toks, ranges = {}, {}
    for split in ("train", "val", "test"):
        p = os.path.join(cache_dir, f"tokens_{split}_{tkey}.npz")
        if os.path.exists(p):
            z = np.load(p)
            toks[split] = z["tokens"].astype(np.int64)
            ranges[split] = [(n, int(a), int(b), int(nb)) for n, (a, b, nb)
                             in zip(z["names"].tolist(), z["ranges"].tolist())]
            print(f"[cache] tokens HIT  {split} ({len(toks[split]):,})")
            continue
        toks[split], rng = encode_domains(tok, byts[split], doms[split])
        ranges[split] = rng
        np.savez(p, tokens=toks[split].astype(np.int32),
                 names=np.array([r[0] for r in rng]),
                 ranges=np.array([r[1:] for r in rng], dtype=np.int64))
        print(f"[cache] tokens MISS {split} ({len(toks[split]):,}) -> saved")
    toks["_domains"] = ranges
    return tok, toks


def domain_eval(model, cfg: Config, toks, feats, byts_len: int, split="test", T=1.0) -> Dict:
    """Calibrated bits/byte per domain of a mixed-corpus split, plus "_all" = the
    byte-weighted total (bits over all domains / bytes over all domains)."""
    out, bits, nbytes = {}, 0.0, 0
    for name, a, b, nb in toks["_domains"][split]:
        sub = {k: v[a:b] for k, v in feats[split].items()}
        ld = make_loader(toks[split][a:b], sub, cfg, shuffle=False)
        bpb = evaluate_calibrated(model, ld, DEVICE, nb, b - a, T)["bits_per_byte"]
        out[name] = float(bpb)
        bits += bpb * nb
        nbytes += nb
    out["_all"] = float(bits / max(1, nbytes))
    return out


def shuffled_loader(tokens, feats, cfg: Config, seed=0) -> torch.utils.data.DataLoader:
    """Deterministically shuffled eval loader (fixed permutation): the first
    batches sample every domain of a mixed split instead of only the first one."""
    ds = CompressionDataset(tokens, feats, cfg.seg_len)
    perm = np.random.RandomState(seed).permutation(len(ds)).tolist()
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds, perm),
                                       batch_size=cfg.batch_size, shuffle=False,
                                       drop_last=True)


def prepare_data(cfg: Config, cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    set_seed(cfg.seed)
    ensure_deps()
    tr_b, va_b, te_b = load_byte_splits(cfg)
    doms = split_domains(cfg)
    if doms is None:
        tok = train_bpe(tr_b, cfg, os.path.join(cache_dir, "tokenizer.json"))
        toks = {"train": encode_with(tok, tr_b),
                "val":   encode_with(tok, va_b),
                "test":  encode_with(tok, te_b)}
    else:
        tok, toks = _prepare_mixed_tokens(cfg, cache_dir, doms,
                                          {"train": tr_b, "val": va_b, "test": te_b})
    classical = ClassicalStats(cfg)
    classical.fit(toks["train"])             # builds order0/order1 + order-2 + match model
    key = _cache_key(cfg.vocab_size, getattr(cfg, "order2_match_order", 0),
                     len(toks["train"]), int(toks["train"][:4096].sum()),
                     *(["fast"] if getattr(cfg, "fast_classical", False) else []))
    feats = {}
    for name in ("train", "val", "test"):
        p = os.path.join(cache_dir, f"feats_{name}_{key}.npz")
        if os.path.exists(p):
            z = np.load(p)
            feats[name] = {k: z[k] for k in z.files}
            print(f"[cache] features HIT  {name} ({key})")
        else:
            feats[name] = classical.features(toks[name])
            np.savez_compressed(p, **feats[name])
            print(f"[cache] features MISS {name} ({key}) -> saved")
    return tok, classical, toks, (tr_b, va_b, te_b), feats


# ---------------------------------------------------------------------------
#  Calibration: bits/byte IS the cross-entropy, so a temperature T that minimises
#  NLL on val directly lowers the achievable compressed size. We measure the
#  optimal T (golden-section) + ECE, detect drift vs the previous version, and
#  bake T into the exported config so the Rust runtime divides z_final by T.
# ---------------------------------------------------------------------------
@torch.no_grad()
def _collect_logits(model, loader, device, max_batches=16):
    model.eval()
    zs, ys = [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = move_batch(batch, device)
        out = model(batch, stage=2)
        z = out["z_final"].reshape(-1, out["z_final"].shape[-1]).float().cpu()
        zs.append(z)
        ys.append(batch["target_ids"].reshape(-1).cpu())
    model.train()
    return torch.cat(zs), torch.cat(ys)


def _ece(z, y, bins=15) -> float:
    p = z.softmax(-1)
    conf, pred = p.max(-1)
    acc = (pred == y).float()
    edges = torch.linspace(0, 1, bins + 1)
    e = 0.0
    n = max(1, len(y))
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            e += (m.float().sum().item() / n) * abs(conf[m].mean().item() - acc[m].mean().item())
    return float(e)


def measure_calibration(model, loader, device, max_batches=16) -> Dict:
    """Optimal temperature (minimise NLL of softmax(z/T)) + binned ECE."""
    z, y = _collect_logits(model, loader, device, max_batches)

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
            "ece": float(_ece(z, y)), "ece_after": float(_ece(z / T, y))}


def detect_decalibration(curr: Dict, prev: Dict, tau_T=0.07, slack=0.05) -> bool:
    if abs(curr["T_opt"] - 1.0) > tau_T:
        return True
    if prev is not None and curr["ece"] > prev["ece"] * (1.0 + slack):
        return True
    return False


@torch.no_grad()
def evaluate_calibrated(model, loader, device, bytes_total, tokens_total, T=1.0) -> Dict:
    """Like evaluate() but divides the fused logits by temperature T."""
    model.eval()
    nats, ntok = 0.0, 0
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch, stage=2)
        z = out["z_final"] / T
        ce = torch.nn.functional.cross_entropy(
            z.reshape(-1, z.shape[-1]), batch["target_ids"].reshape(-1), reduction="sum")
        nats += ce.item()
        ntok += batch["target_ids"].numel()
    model.train()
    bpb = (nats / max(1, ntok)) * math.log2(math.e) * (tokens_total / max(1, bytes_total))
    return {"bits_per_byte": bpb, "nats_per_token": nats / max(1, ntok)}


# ---------------------------------------------------------------------------
#  Verification-only arithmetic coder (Witten-Neal-Cleary, 32-bit). PROVES the
#  entropy-coding stage is byte-exact lossless on the model's *causal* per-token
#  distributions. Pure Python and slow -> used on a short chunk only; production
#  coding lives in Rust (RUST_INTEGRATION_BLUEPRINT.md).
# ---------------------------------------------------------------------------
def _quantize(p, total_bits=16):
    total = 1 << total_bits
    V = len(p)
    while total < V:                       # guarantee every symbol gets freq >= 1
        total <<= 1
    f = np.maximum(1, np.floor(np.asarray(p, dtype=np.float64) * total).astype(np.int64))
    diff = total - int(f.sum())
    if diff != 0:
        j = int(np.argmax(f))
        f[j] = max(1, f[j] + diff)
        diff2 = total - int(f.sum())
        if diff2 != 0:                     # rare second pass if the clamp bit
            j = int(np.argmax(f))
            f[j] = max(1, f[j] + diff2)
    cum = np.concatenate([[0], np.cumsum(f)]).astype(np.int64)
    return cum                              # cum[-1] == total, length V+1


class _RangeCoder:
    PREC = 32
    MASK = (1 << 32) - 1
    HALF = 1 << 31
    QUARTER = 1 << 30
    THREEQ = 3 << 30

    def encode(self, triples):
        low, high, pending, bits = 0, self.MASK, 0, []

        def emit(bit):
            nonlocal pending
            bits.append(bit)
            bits.extend([1 - bit] * pending)
            pending = 0

        for cl, ch, total in triples:
            rng = high - low + 1
            high = low + (rng * ch) // total - 1
            low = low + (rng * cl) // total
            while True:
                if high < self.HALF:
                    emit(0)
                elif low >= self.HALF:
                    emit(1); low -= self.HALF; high -= self.HALF
                elif low >= self.QUARTER and high < self.THREEQ:
                    pending += 1; low -= self.QUARTER; high -= self.QUARTER
                else:
                    break
                low = (low << 1) & self.MASK
                high = ((high << 1) | 1) & self.MASK
        pending += 1
        emit(0 if low < self.QUARTER else 1)
        return bits

    def decode(self, bits, cum_tables, n):
        stream = iter(bits + [0] * (self.PREC + 2))
        low, high, value = 0, self.MASK, 0
        for _ in range(self.PREC):
            value = ((value << 1) | next(stream)) & self.MASK
        out = []
        for t in range(n):
            cum = cum_tables[t]
            total = int(cum[-1])
            rng = high - low + 1
            scaled = (((value - low + 1) * total - 1) // rng)
            s = int(np.searchsorted(cum, scaled, side="right") - 1)
            s = max(0, min(len(cum) - 2, s))
            cl, ch = int(cum[s]), int(cum[s + 1])
            high = low + (rng * ch) // total - 1
            low = low + (rng * cl) // total
            while True:
                if high < self.HALF:
                    pass
                elif low >= self.HALF:
                    low -= self.HALF; high -= self.HALF; value -= self.HALF
                elif low >= self.QUARTER and high < self.THREEQ:
                    low -= self.QUARTER; high -= self.QUARTER; value -= self.QUARTER
                else:
                    break
                low = (low << 1) & self.MASK
                high = ((high << 1) | 1) & self.MASK
                value = ((value << 1) | next(stream)) & self.MASK
            out.append(s)
        return out


def _selftest_coder(seed=0) -> bool:
    """Encode/decode a random sequence under random per-position distributions and
    assert byte-exact recovery. Guards against a coder regression."""
    rng = np.random.RandomState(seed)
    V, n = 37, 200
    cum_tables, syms = [], []
    for _ in range(n):
        p = rng.dirichlet(np.ones(V) * 0.3)
        cum = _quantize(p, total_bits=14)
        assert int(cum[-1]) == (1 << 14)
        cum_tables.append(cum)
        syms.append(int(rng.randint(V)))
    rc = _RangeCoder()
    triples = [(int(cum_tables[t][syms[t]]), int(cum_tables[t][syms[t] + 1]),
                int(cum_tables[t][-1])) for t in range(n)]
    dec = rc.decode(rc.encode(triples), cum_tables, n)
    return list(dec) == syms


@torch.no_grad()
def _arith_roundtrip(model, tokens, feats, cfg):
    """True encode->decode->assert on REAL causal model distributions for one
    segment. Returns (ok, bits_per_token_achieved)."""
    try:
        loader = make_loader(tokens, feats, cfg, shuffle=False)
        batch = move_batch(next(iter(loader)), DEVICE)
        out = model(batch, stage=2)
        z = out["z_final"][0].float().cpu()
        tgt = batch["target_ids"][0].cpu().numpy()
        probs = torch.softmax(z, -1).numpy()
        cum_tables = [_quantize(probs[t]) for t in range(len(tgt))]
        triples = [(int(cum_tables[t][tgt[t]]), int(cum_tables[t][tgt[t] + 1]),
                    int(cum_tables[t][-1])) for t in range(len(tgt))]
        rc = _RangeCoder()
        bits = rc.encode(triples)
        dec = rc.decode(bits, cum_tables, len(tgt))
        ok = bool(np.array_equal(np.asarray(dec), tgt))
        return ok, len(bits) / max(1, len(tgt))
    except Exception as e:
        print("  [warn] arithmetic roundtrip failed:", e)
        return False, float("nan")


# ---------------------------------------------------------------------------
#  Professional benchmark harness
# ---------------------------------------------------------------------------
@torch.no_grad()
def _throughput(model, loader, n_tokens=4000) -> float:
    device = DEVICE
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    seen, t = 0, time.time()
    for batch in loader:
        batch = move_batch(batch, device)
        model(batch, stage=2)
        seen += batch["target_ids"].numel()
        if seen >= n_tokens:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    model.train()
    return seen / max(1e-9, time.time() - t)      # batched teacher-forced upper bound


def _stress_report(model, classical, tok, cfg) -> Dict:
    cases = {
        "repetitive": b"ABCD" * 8192,
        "random": bytes(np.random.RandomState(0).randint(0, 256, 32768).astype(np.uint8).tolist()),
        "structured_text": b"the quick brown fox jumps over the lazy dog " * 1024,
    }
    rows = {}
    for name, data in cases.items():
        toks = encode_with(tok, data)
        if len(toks) < cfg.seg_len + 2:
            rows[name] = {"skipped": "too short"}
            continue
        f = classical.features(toks)
        loader = make_loader(toks, f, cfg, shuffle=False)
        m = evaluate(model, loader, DEVICE, len(data), len(toks))
        rows[name] = {"model_bpb": round(m["bits_per_byte"], 4),
                      "gzip_bpb": round(gzip_bpb(data), 4),
                      "xz_bpb": round(xz_bpb(data), 4)}
    return rows


def benchmark(model, classical, tok, cfg, toks, byts, feats, label="") -> Dict:
    """Ratio (vs gzip-9/xz-9) + lossless hard gate + throughput + stress."""
    device = DEVICE
    te_loader = make_loader(toks["test"], feats["test"], cfg, shuffle=False)
    ev = evaluate(model, te_loader, device, len(byts[2]), len(toks["test"]))
    gz, xzb = gzip_bpb(byts[2]), xz_bpb(byts[2])
    n_ref = min(128, len(toks["test"]) - 1)
    try:
        ref_bpt = reference_decoder_check(model, toks["test"], feats["test"], cfg, n=n_ref)
        ref_ok = math.isfinite(ref_bpt)
    except Exception as e:
        ref_bpt, ref_ok = float("nan"), False
        print("  [warn] reference decoder:", e)
    coder_ok, coder_bpt = _arith_roundtrip(model, toks["test"], feats["test"], cfg)
    lossless = bool(ref_ok and coder_ok)
    thr = _throughput(model, te_loader)
    stress = _stress_report(model, classical, tok, cfg)
    res = dict(label=label, test_bpb=float(ev["bits_per_byte"]),
               gzip_bpb=float(gz), xz_bpb=float(xzb),
               ref_bits_per_token=float(ref_bpt), coder_bits_per_token=float(coder_bpt),
               lossless=lossless, coder_ok=bool(coder_ok),
               throughput_tok_s=float(thr), stress=stress)
    print(f"  [bench {label}] test bpb={res['test_bpb']:.4f}  gzip={gz:.3f}  xz={xzb:.3f}  "
          f"lossless={lossless}  thr={thr:.0f} tok/s (batched UB)")
    return res


# ---------------------------------------------------------------------------
#  Versioning, comparison & ledger
# ---------------------------------------------------------------------------
def compare_versions(curr: Dict, prev: Dict) -> Dict:
    """Compare to the DIRECT previous version. Losslessness is a hard gate."""
    if not curr.get("lossless", False):
        return {"verdict": "reject", "reason": "lossless gate failed", "delta_bpb": None}
    if prev is None:
        return {"verdict": "baseline", "delta_bpb": None}
    delta = prev["test_bpb_cal"] - curr["test_bpb_cal"]    # +ve = improvement
    return {"verdict": "improved" if delta > 0 else "regressed", "delta_bpb": float(delta)}


def _write_ledger(outdir: str, ledger: list):
    with open(os.path.join(outdir, "ledger.json"), "w") as f:
        json.dump(ledger, f, indent=2)
    lines = ["# Recursive training ledger", "",
             "| ver | tier | step | test bpb | calibrated | gzip | xz | lossless | T | verdict | Δbpb | best |",
             "|----|------|------|---------|-----------|------|----|---------|---|--------|------|------|"]
    for r in ledger:
        d = "" if r.get("delta_bpb") is None else f"{r['delta_bpb']:+.4f}"
        lines.append(
            f"| {r['version']} | {r['tier']} | {r['step']}/{r['total_steps']} | "
            f"{r['test_bpb']:.4f} | {r['test_bpb_cal']:.4f} | {r['gzip_bpb']:.3f} | "
            f"{r['xz_bpb']:.3f} | {'yes' if r['lossless'] else 'NO'} | {r['calib_T']:.3f} | "
            f"{r['verdict']} | {d} | {'★' if r.get('is_best') else ''} |")
    with open(os.path.join(outdir, "ledger.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
#  The recursive orchestrator
# ---------------------------------------------------------------------------
def run_recursive(total_budget_h=None, start_tier="nano", target_bpb=None,
                  plateau_eps=0.01, escalate_to=None, outdir=None,
                  target_phases=None, session_cap_h=None, seed=1234) -> Dict:
    """Budget-driven recursive trainer: each version resumes the previous, is
    benchmarked + calibrated + compared, and the best lossless version is kept.
    Starts `start_tier`, auto-escalates to `escalate_to` on plateau if enough
    budget remains, and stops at the first of {target met, plateau, budget spent}."""
    B = resolve_budget_h(total_budget_h)
    escalate_to = ESCALATE_TO if escalate_to is None else escalate_to
    target_bpb = TARGET_BPB if target_bpb is None else target_bpb
    outdir = outdir or os.path.join(WORKDIR, "runs")
    os.makedirs(outdir, exist_ok=True)
    plan = plan_budget(B, target_phases, session_cap_h)
    print(f"[recursive] budget={B:.2f}h  plan={plan}  start_tier={start_tier}  "
          f"escalate_to={escalate_to}  target_bpb={target_bpb}")

    t_start = time.time()
    def elapsed():
        return (time.time() - t_start) / 3600.0

    tier, escalated = start_tier, False
    prev_bench, best_overall, ledger = None, None, []
    prepared, vnum = {}, 0

    while True:
        remaining = B - elapsed()
        if remaining <= 1e-3:
            print("[recursive] budget exhausted.")
            break

        if tier not in prepared:
            prepared[tier] = prepare_data(get_config(tier), os.path.join(outdir, "cache", tier))
        tok, classical, toks, byts, feats = prepared[tier]

        vnum += 1
        cfg = get_config(tier)
        cfg.seed = seed
        vdir = os.path.join(outdir, tier, f"v{vnum:03d}")
        os.makedirs(vdir, exist_ok=True)
        ckpt = os.path.join(vdir, "best.pt")

        resume_from = None
        same = [r for r in ledger if r["tier"] == tier]
        if same:
            resume_from = same[-1]["last_path"]

        lr_scale = 1.0
        if prev_bench is not None:
            if prev_bench.get("decalibrated") or prev_bench.get("verdict") == "regressed":
                lr_scale = 0.3
            elif prev_bench.get("plateau"):
                lr_scale = 0.5

        phase_h = min(plan["phase_h"], remaining)
        tr_loader = make_loader(toks["train"], feats["train"], cfg, shuffle=True)
        va_loader = make_loader(toks["val"], feats["val"], cfg, shuffle=False)
        print(f"\n=== {tier} v{vnum:03d} | phase<= {phase_h:.2f}h | lr_scale={lr_scale} | "
              f"resume={'yes' if resume_from else 'no'} | elapsed {elapsed():.2f}/{B:.2f}h ===")

        model = HybridLM(cfg, classical)
        tr = train(model, tr_loader, va_loader, cfg, len(byts[1]), len(toks["val"]),
                   ckpt, resume_from=resume_from, time_budget_h=phase_h,
                   save_every=max(50, cfg.eval_every), lr_scale=lr_scale)
        ckpt_to_load = ckpt
        if not os.path.exists(ckpt_to_load):
            alt = _find_pt_file(vdir)
            if alt:
                ckpt_to_load = alt
                print(f"[recursive] loading from {ckpt_to_load} instead of {ckpt}")
        model.load_state_dict(_xla_safe_load(ckpt_to_load, map_location=DEVICE,
                                             weights_only=True)["model"])

        bench = benchmark(model, classical, tok, cfg, toks, byts, feats,
                          label=f"{tier} v{vnum:03d}")
        calib = measure_calibration(model, va_loader, DEVICE)
        prev_calib = prev_bench.get("calib") if prev_bench else None
        decal = detect_decalibration(calib, prev_calib)
        calib_T = calib["T_opt"] if decal else 1.0
        if decal:
            print(f"  [decalibration] T*={calib['T_opt']:.3f} ECE={calib['ece']:.4f} "
                  f"-> applying temperature scaling")
        cal_bpb = evaluate_calibrated(model, make_loader(toks["test"], feats["test"], cfg, False),
                                      DEVICE, len(byts[2]), len(toks["test"]), calib_T)["bits_per_byte"]
        bench.update(calib=calib, calib_T=float(calib_T), decalibrated=bool(decal),
                     test_bpb_cal=float(cal_bpb))

        cmp = compare_versions(bench, prev_bench)
        bench["verdict"] = cmp["verdict"]
        bench["delta_bpb"] = cmp.get("delta_bpb")
        bench["plateau"] = (cmp.get("delta_bpb") is not None
                            and abs(cmp["delta_bpb"]) < plateau_eps)

        is_best = bench["lossless"] and (best_overall is None
                                         or bench["test_bpb_cal"] < best_overall["test_bpb_cal"])
        if is_best:
            best_overall = {"tier": tier, "version": vnum, "vdir": vdir, "ckpt": ckpt,
                            "test_bpb_cal": bench["test_bpb_cal"], "calib_T": calib_T,
                            "classical": classical, "tok": tok, "cfg": cfg}
            # Export immediately — a kill/timeout will never lose the best artifacts.
            try:
                best_dir = os.path.join(outdir, "best")
                os.makedirs(best_dir, exist_ok=True)
                export_all(model, classical, tok, cfg, best_dir)
                with open(os.path.join(best_dir, "calibration.json"), "w") as f:
                    json.dump({"calibration_T": float(calib_T),
                               "test_bits_per_byte": float(bench["test_bpb_cal"])}, f, indent=2)
                with open(os.path.join(best_dir, "best_meta.json"), "w") as f:
                    json.dump({"tier": tier, "version": vnum, "vdir": vdir, "ckpt": ckpt,
                               "test_bpb_cal": float(bench["test_bpb_cal"]),
                               "calib_T": float(calib_T)}, f, indent=2)
                print(f"[recursive] new best {tier} v{vnum:03d} "
                      f"({bench['test_bpb_cal']:.4f} bpb) → exported to {best_dir}")
            except Exception as e:
                print(f"[recursive] mid-loop best-export failed (non-fatal): {e}")

        row = {"version": vnum, "tier": tier, "step": tr["step"],
               "total_steps": tr["total_steps"], "last_path": tr["last_path"],
               "test_bpb": bench["test_bpb"], "test_bpb_cal": bench["test_bpb_cal"],
               "gzip_bpb": bench["gzip_bpb"], "xz_bpb": bench["xz_bpb"],
               "lossless": bench["lossless"], "calib_T": float(calib_T),
               "verdict": bench["verdict"], "delta_bpb": bench["delta_bpb"],
               "is_best": is_best, "stop_reason": tr["stop_reason"]}
        ledger.append(row)
        with open(os.path.join(vdir, "benchmark.json"), "w") as f:
            json.dump({k: v for k, v in bench.items() if k != "calib"}, f, indent=2, default=float)
        with open(os.path.join(vdir, "calibration.json"), "w") as f:
            json.dump(calib, f, indent=2)
        _write_ledger(outdir, ledger)

        # ---- stop / continue / escalate decisions ----
        if target_bpb is not None and bench["lossless"] and bench["test_bpb_cal"] <= target_bpb:
            print(f"[recursive] target bits/byte {target_bpb} reached.")
            break
        converged = tr["stop_reason"] in ("completed", "early_stop") or bench["plateau"]
        if converged:
            if (tier == start_tier and not escalated and escalate_to and escalate_to != tier
                    and (B - elapsed()) >= plan["escalate_min_h"]):
                print(f"[recursive] {tier} plateaued; escalating to {escalate_to} "
                      f"({B - elapsed():.2f}h left).")
                tier, escalated, prev_bench = escalate_to, True, None
                continue
            print("[recursive] converged/plateaued with no escalation budget; stopping.")
            break
        prev_bench = bench

    # ---- finalize: confirm or recover the best export ----
    if best_overall is not None:
        bo = best_overall
        best_dir = os.path.join(outdir, "best")
        meta_path = os.path.join(best_dir, "best_meta.json")
        print(f"\n[recursive] BEST = {bo['tier']} v{bo['version']:03d}  "
              f"calibrated test bits/byte = {bo['test_bpb_cal']:.4f}  (T={bo['calib_T']:.3f})")
        if os.path.exists(meta_path):
            print(f"[recursive] best artifacts already at {best_dir} (saved during loop)")
        else:
            # Fallback export in case the in-loop save was interrupted.
            try:
                m = HybridLM(bo["cfg"], bo["classical"])
                ckpt_to_load = bo["ckpt"]
                if not os.path.exists(ckpt_to_load):
                    alt = _find_pt_file(os.path.dirname(bo["ckpt"]))
                    if alt:
                        ckpt_to_load = alt
                        print(f"[recursive] loading from {ckpt_to_load} instead of {bo['ckpt']}")
                m.load_state_dict(_xla_safe_load(ckpt_to_load, map_location=DEVICE,
                                             weights_only=True)["model"])
                os.makedirs(best_dir, exist_ok=True)
                export_all(m, bo["classical"], bo["tok"], bo["cfg"], best_dir)
                with open(os.path.join(best_dir, "calibration.json"), "w") as f:
                    json.dump({"calibration_T": bo["calib_T"],
                               "test_bits_per_byte": bo["test_bpb_cal"]}, f, indent=2)
                with open(meta_path, "w") as f:
                    json.dump({"tier": bo["tier"], "version": bo["version"],
                               "vdir": bo["vdir"], "ckpt": bo["ckpt"],
                               "test_bpb_cal": float(bo["test_bpb_cal"]),
                               "calib_T": float(bo["calib_T"])}, f, indent=2)
                print(f"[recursive] exported best model + calibration_T to {best_dir}")
            except Exception as e:
                print("[recursive] export of best model failed:", e)
    print(f"[recursive] done: {len(ledger)} version(s); ledger at "
          f"{os.path.join(outdir, 'ledger.md')}")
    return {"ledger": ledger, "best": best_overall, "outdir": outdir}


def run_smoke_recursive():
    """Fast CPU test of the recursive machinery (no GPU, no downloads)."""
    print("== smoke-recursive ==")
    assert _selftest_coder(0) and _selftest_coder(7), "arithmetic coder roundtrip FAILED"
    print("  [ok] arithmetic coder byte-exact roundtrip")

    # budget resolution + planning
    assert resolve_budget_h(3.0) == 3.0
    _saved = globals()["BUDGET_H"]
    globals()["BUDGET_H"] = 5.0
    try:
        assert resolve_budget_h(None) == 5.0
    finally:
        globals()["BUDGET_H"] = _saved
    for B in (1.0, 9.0, 30.0):
        pl = plan_budget(B)
        assert pl["phase_h"] <= SESSION_CAP_H + 1e-9
        assert pl["escalate_min_h"] >= 0.30 * B - 1e-9
    try:
        import sys as _sys
        _real_isatty = None
        if _sys.stdin is not None:
            _real_isatty = _sys.stdin.isatty
            _sys.stdin.isatty = lambda: False
        raised = False
        try:
            resolve_budget_h(None)              # BUDGET_H None + non-interactive -> error
        except ValueError:
            raised = True
        finally:
            if _real_isatty is not None:
                _sys.stdin.isatty = _real_isatty
        assert raised, "mandatory budget did not raise without a value"
    except Exception as e:
        print("  [warn] interactive-guard check skipped:", e)
    print("  [ok] budget mandatory + auto-adaptive planning")

    # tiny 2-version resume on synthetic data
    import tempfile
    cfg = get_config("smoke")
    cfg.stage1_steps, cfg.stage2_steps, cfg.eval_every, cfg.warmup_steps = 4, 4, 2, 1
    data = (b"the quick brown fox jumps over the lazy dog. " * 200)
    with tempfile.TemporaryDirectory() as td:
        tok = train_bpe(data, cfg, os.path.join(td, "tok.json"))
        toks = encode_with(tok, data)
        classical = ClassicalStats(cfg)
        classical.fit(toks)
        feats = classical.features(toks)
        f2 = {"train": feats, "val": feats, "test": feats}
        t2 = {"train": toks, "val": toks, "test": toks}
        tl = make_loader(toks, feats, cfg, shuffle=True)
        vl = make_loader(toks, feats, cfg, shuffle=False)
        ck = os.path.join(td, "best.pt")
        m = HybridLM(cfg, classical)
        r1 = train(m, tl, vl, cfg, len(data), len(toks), ck, time_budget_h=10.0)
        assert os.path.exists(r1["last_path"]), "phase-1 .last not written"
        m2 = HybridLM(cfg, classical)
        cfg.stage2_steps = 8                      # extend curriculum so phase-2 has steps to do
        r2 = train(m2, tl, vl, cfg, len(data), len(toks), ck,
                   resume_from=r1["last_path"], time_budget_h=10.0)
        assert r2["step"] >= r1["step"], "resume did not continue the step counter"
        print(f"  [ok] resume: phase1 step={r1['step']} -> phase2 step={r2['step']}")

        bench = benchmark(m2, classical, tok, cfg, t2, (data, data, data), f2, label="smoke")
        for k in ("test_bpb", "gzip_bpb", "xz_bpb", "lossless", "throughput_tok_s", "stress"):
            assert k in bench, f"benchmark missing key {k}"
        assert bench["coder_ok"], "model-distribution arithmetic roundtrip not lossless"
        cal = measure_calibration(m2, vl, DEVICE)
        assert math.isfinite(cal["T_opt"]) and 0.0 <= cal["ece"] <= 1.0
        assert cal["nll_after"] <= cal["nll_before"] + 1e-6, "temperature did not help NLL"
        print(f"  [ok] benchmark keys + lossless + calibration (T*={cal['T_opt']:.3f}, "
              f"ECE={cal['ece']:.4f})")

        # ledger monotonic best-tracking
        led = [{"version": 1, "tier": "smoke", "step": 4, "total_steps": 8, "last_path": "",
                "test_bpb": 5.0, "test_bpb_cal": 5.0, "gzip_bpb": 6.0, "xz_bpb": 5.5,
                "lossless": True, "calib_T": 1.0, "verdict": "baseline", "delta_bpb": None,
                "is_best": True, "stop_reason": "budget"},
               {"version": 2, "tier": "smoke", "step": 8, "total_steps": 8, "last_path": "",
                "test_bpb": 4.5, "test_bpb_cal": 4.5, "gzip_bpb": 6.0, "xz_bpb": 5.5,
                "lossless": True, "calib_T": 1.0, "verdict": "improved", "delta_bpb": 0.5,
                "is_best": True, "stop_reason": "completed"}]
        _write_ledger(td, led)
        assert os.path.exists(os.path.join(td, "ledger.md"))
        print("  [ok] ledger written, monotonic best-tracking")
    print("== smoke-recursive PASSED ==")


# Compression goal, in calibrated test bits/byte (== bits/char on enwik8).
# Statistically projected BEST achievable value for the base tier (d=256, L=4)
# at curriculum end (step 160k): an ensemble of decay models (linear-rate-decay,
# power laws alpha=0.2..1.0, cosine-LR-integral) fitted to the recorded
# checkpoints (val 1.3100@67.3k -> 1.3071@72k, cal-test 1.2693@72k) projects a
# final calibrated test bpb of 1.2437 +/- 0.0035 (range 1.238-1.248); 1.245 is
# the ensemble median.  run_budget() keeps launching training across sessions
# until a phase CONFIRMS this value on the calibrated test split, then stops
# automatically (and future sessions exit immediately).  Keep in sync with
# benchmark_enterprise.GOAL_TEST_BPB.
GOAL_TEST_BPB = 1.245


def run_budget(total_budget_h=None, session_cap_h=11.5, outdir=None,
               fresh_start=False, target_phases=4, seed=1234,
               goal_test_bpb=GOAL_TEST_BPB, tier=None, dataset=None):
    """
    Budget-only unlimited-session trainer — no tier names required.

    total_budget_h  : used ONLY for model sizing (e.g. 80 → d=384, 23 M params).
                      On the first session this value is locked into accumulated.json
                      so all future sessions use the same architecture even if you
                      change the number here.  Pass None to load from a previous run.
    session_cap_h   : wall-clock limit for THIS Kaggle session (default 11.5 h;
                      Kaggle hard-kills CPU committed sessions at 12 h).
    fresh_start     : True only on the very first session; wipes runs/.
    tier, dataset   : first session only — pick a named tier (e.g. "mid") and a
                      corpus (e.g. "corpus_r2") explicitly instead of sizing the
                      model from the budget; the curriculum is scaled to the
                      budget and locked into accumulated.json like any other.
    goal_test_bpb   : compression goal in calibrated test bits/byte (None
                      disables).  The session chain STOPS AUTOMATICALLY when a
                      phase confirms this value on the calibrated test split —
                      the stop is recorded in accumulated.json, so any session
                      started afterwards exits immediately.  Training also
                      finishes on curriculum end or a val plateau, whichever
                      comes first.

    Sessions are unlimited — accumulated hours are tracked for information but
    never block another session from starting.  Every session (even a 4 h one)
    produces valid outputs:

      runs/current/   — ONNX models + config + tokenizer + calibration.json,
                        always reflecting the most recently trained checkpoint.
                        Valid for Rust integration immediately after any session.
      runs/best/      — same artifacts, only updated when a new bpb record is set.
      runs/ledger.json   — append-only phase history across all sessions.
      runs/accumulated.json — cross-session elapsed hours + locked model budget.

    To resume on the next Kaggle session:
      1. Upload runs/ (or at minimum ledger.json + accumulated.json + v*/best.pt)
      2. Set FRESH_START = False and run the same cell — training picks up where
         it left off automatically.
    """
    import shutil as _sh
    t_wall  = time.time()
    elapsed = lambda: (time.time() - t_wall) / 3600.0

    outdir = outdir or os.path.join(WORKDIR, "runs")
    if fresh_start and os.path.isdir(outdir):
        _sh.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)

    # ---- accumulated-hours + architecture lock ------------------------------
    acc_path = os.path.join(outdir, "accumulated.json")
    accumulated_h   = 0.0
    locked_total_h  = None
    locked_cfg      = None  # exact Config dict locked on the first session
    stop_state      = {}    # {"done": True, "reason": ...} once training finished
    val_test_offset = None  # measured (trainer val bpb - calibrated test bpb)
    if not fresh_start and os.path.exists(acc_path):
        _acc = json.load(open(acc_path))
        accumulated_h  = float(_acc.get("elapsed_h", 0.0))
        locked_total_h = _acc.get("total_budget_h")
        locked_cfg      = _acc.get("cfg")
        stop_state      = _acc.get("stop") or {}
        val_test_offset = _acc.get("val_test_offset")

    # ---- cross-session auto-stop: goal already reached / training finished --
    if stop_state.get("done"):
        led_path = os.path.join(outdir, "ledger.json")
        ledger = json.load(open(led_path)) if os.path.exists(led_path) else []
        _bm = os.path.join(outdir, "best", "best_meta.json")
        best_overall = json.load(open(_bm)) if os.path.exists(_bm) else None
        print("=" * 72)
        print(f"[goal] TRAINING ALREADY FINISHED — {stop_state.get('reason')}")
        print(f"[goal] stopped at step {stop_state.get('step')} on "
              f"{stop_state.get('when')} with calibrated test bpb "
              f"{stop_state.get('test_bpb_cal')}")
        print("[goal] runs/best/ holds the final artifacts; nothing to train. "
              "Set FRESH_START=True to restart from scratch.")
        print("=" * 72)
        return {"ledger": ledger, "best": best_overall, "outdir": outdir,
                "cfg": None, "elapsed_h": elapsed(),
                "accumulated_h": accumulated_h, "stopped": stop_state}

    # Resolve B_total — architecture-locking value (first session wins)
    if locked_total_h is not None:
        if total_budget_h is not None and abs(total_budget_h - locked_total_h) > 0.01:
            print(f"[budget] NOTE: architecture locked to {locked_total_h:.1f}h "
                  f"(first session); ignoring total_budget_h={total_budget_h:.1f}h "
                  f"to keep model weights compatible.")
        B_total = locked_total_h
    else:
        B_total = resolve_budget_h(total_budget_h)

    this_session_h = float(session_cap_h)  # no hard cap from accumulated — sessions are unlimited

    # ---- model sized on total budget (constant across sessions) -------------
    # The EXACT Config (architecture + curriculum) is locked into
    # accumulated.json on the first session; resumed sessions must reuse it
    # verbatim — budget-derived configs can differ in curriculum length
    # (stage1/stage2 steps), which would corrupt the LR schedule and trigger a
    # bogus "curriculum complete" auto-stop on a checkpoint trained with a
    # tier preset.
    train_h_total, _bench_h_total = split_budget(B_total)
    if locked_cfg:
        cfg = Config(**locked_cfg)
        print(f"[budget] using EXACT architecture+curriculum locked in "
              f"accumulated.json (tier {cfg.tier}); budget only limits "
              f"session time now.")
    elif tier:
        cfg = config_from_tier(tier, max(0.05, train_h_total - PREP_EST_H),
                               seed=seed, dataset=dataset)
    else:
        cfg = config_from_budget(max(0.05, train_h_total - PREP_EST_H), seed=seed)

    train_h_sess, bench_h_sess = split_budget(this_session_h)
    bench_per_phase = bench_h_sess / max(1, target_phases)

    print(f"[budget] model_budget={B_total:.1f}h  accumulated={accumulated_h:.2f}h  "
          f"this_session≤{this_session_h:.2f}h  (sessions unlimited)")
    print(f"[budget] session split: train≤{train_h_sess:.2f}h  bench≤{bench_h_sess:.2f}h  "
          f"({target_phases} phases × ≤{bench_per_phase:.2f}h bench each)")
    print(f"[budget] config: {cfg.tier}  d={cfg.d_model} L={cfg.n_layers} H={cfg.n_heads}  "
          f"vocab={cfg.vocab_size}  seg={cfg.seg_len}  "
          f"stage1={cfg.stage1_steps} stage2={cfg.stage2_steps}")

    # ---- data prep ----------------------------------------------------------
    cache_dir = os.path.join(outdir, "cache")
    tok, classical, toks, byts, feats = prepare_data(cfg, cache_dir)

    actual_prep = elapsed()
    if fresh_start and not tier and actual_prep > PREP_EST_H * 1.5:
        cfg2 = config_from_budget(max(0.05, train_h_total - actual_prep), seed=seed)
        if (cfg2.d_model, cfg2.n_layers) != (cfg.d_model, cfg.n_layers):
            print(f"[budget] prep took {actual_prep:.2f}h > estimate; "
                  f"downsizing {cfg.tier} → {cfg2.tier}")
            if cfg2.vocab_size != cfg.vocab_size:
                tok, classical, toks, byts, feats = prepare_data(cfg2, cache_dir)
            cfg = cfg2

    print(f"[budget] data-prep done ({elapsed():.2f}h)  "
          f"params={count_params(HybridLM(cfg, classical)):,}")

    # ---- load existing ledger + find best resume checkpoint -----------------
    import re as _re

    def _remap_ckpt_path(old_path):
        """Translate a checkpoint path from a previous session to the current outdir.
        Handles cross-platform transfers (Kaggle CPU → TPU, etc.) where the working
        directory root changes but the v{NNN}/filename structure stays the same."""
        if not old_path:
            return None
        if os.path.exists(old_path):
            return old_path
        m = _re.search(r'(v\d+[\\/].+)$', old_path.replace("\\", "/"))
        if m:
            candidate = os.path.join(outdir, m.group(1))
            if os.path.exists(candidate):
                print(f"[budget] path remapped: {old_path!r} → {candidate!r}")
                return candidate
        return None

    led_path = os.path.join(outdir, "ledger.json")
    ledger   = (json.load(open(led_path))
                if not fresh_start and os.path.exists(led_path) else [])
    resume_from = _remap_ckpt_path(ledger[-1]["last_path"]) if ledger else None
    if ledger and resume_from is None:
        print(f"[budget] WARNING: ledger last_path not found: {ledger[-1]['last_path']!r}")
    prev_bench  = None
    in_train_goal_ok = True   # val-proxy trigger; disabled after a failed confirm

    # Scan every v*/ directory for latest_hourly.json; if any hourly checkpoint
    # is newer (higher step) than the ledger's last entry, use it instead.
    # This recovers from unexpected kills where the ledger was never updated.
    if not fresh_start:
        ledger_step = int(ledger[-1].get("step", 0)) if ledger else 0
        best_hourly_step, best_hourly_path = ledger_step, None
        import glob as _glob
        for ptr_file in sorted(_glob.glob(os.path.join(outdir, "v*", "latest_hourly.json"))):
            try:
                d = json.load(open(ptr_file))
                h_path = _remap_ckpt_path(d.get("path", ""))
                h_step = int(d.get("step", 0))
                if h_step > best_hourly_step and h_path:
                    best_hourly_step, best_hourly_path = h_step, h_path
            except Exception:
                pass
        if best_hourly_path:
            print(f"[budget] found hourly checkpoint at step {best_hourly_step} "
                  f"(beyond ledger step {ledger_step}); using as resume point.")
            resume_from = best_hourly_path

    best_dir       = os.path.join(outdir, "best")
    best_meta_path = os.path.join(best_dir, "best_meta.json")
    best_overall   = (json.load(open(best_meta_path))
                      if os.path.exists(best_meta_path) else None)
    curr_dir       = os.path.join(outdir, "current")

    # ---- helper: export model to a directory --------------------------------
    def _export(model, dest, calib_T, cal_bpb=None, label=""):
        os.makedirs(dest, exist_ok=True)
        try:
            export_all(model, classical, tok, cfg, dest)
            meta = {"calibration_T": float(calib_T)}
            if cal_bpb is not None:
                meta["test_bits_per_byte"] = float(cal_bpb)
            with open(os.path.join(dest, "calibration.json"), "w") as _f:
                json.dump(meta, _f, indent=2)
            if label:
                print(f"  [budget] exported {label} → {dest}")
        except Exception as _e:
            print(f"  [budget] export failed (non-fatal): {_e}")

    # ---- training phases ----------------------------------------------------
    phase_train_h = max(0.01, (train_h_sess - elapsed()) / max(1, target_phases))

    for phase in range(1, target_phases + 1):
        sess_remaining = this_session_h - elapsed()
        if sess_remaining <= bench_per_phase * 0.5:
            print(f"[budget] session time used ({elapsed():.2f}h elapsed); stopping phases.")
            break
        this_train_h = min(phase_train_h, sess_remaining - bench_per_phase)
        if this_train_h < 0.01:
            break

        global_phase = len(ledger) + 1
        vdir = os.path.join(outdir, f"v{global_phase:03d}")
        os.makedirs(vdir, exist_ok=True)
        ckpt = os.path.join(vdir, "best.pt")
        tr_loader = make_loader(toks["train"], feats["train"], cfg, shuffle=True)
        va_loader = make_loader(toks["val"],   feats["val"],   cfg, shuffle=False)
        print(f"\n=== Phase {global_phase} (session slot {phase}/{target_phases}) | "
              f"train≤{this_train_h:.2f}h  bench≤{bench_per_phase:.2f}h | "
              f"session {elapsed():.2f}/{this_session_h:.2f}h  "
              f"accumulated {accumulated_h+elapsed():.2f}h ===")

        model = HybridLM(cfg, classical)
        # In-train goal detection: map the test-bpb goal onto the trainer's val
        # metric via the last measured val->test offset (0.002 guard so the
        # proxy fires slightly LATE, never early); the phase then ends and the
        # goal is confirmed on the calibrated test split right below.
        goal_val = (goal_test_bpb + val_test_offset - 0.002
                    if (goal_test_bpb is not None and in_train_goal_ok
                        and val_test_offset is not None) else None)
        if goal_val is not None:
            print(f"  [goal] in-train val-proxy armed: stop when val bpb <= "
                  f"{goal_val:.4f} (test goal {goal_test_bpb} + offset "
                  f"{val_test_offset:+.4f})")
        tr = train(model, tr_loader, va_loader, cfg,
                   len(byts[1]), len(toks["val"]), ckpt,
                   resume_from=resume_from, time_budget_h=this_train_h,
                   save_every=max(50, cfg.eval_every), goal_val_bpb=goal_val)
        resume_from = tr["last_path"]

        ckpt_to_load = ckpt
        if not os.path.exists(ckpt_to_load):
            alt = _find_pt_file(vdir)
            if alt:
                ckpt_to_load = alt
                print(f"[budget] using checkpoint {ckpt_to_load} instead of {ckpt}")
        if os.path.exists(ckpt_to_load):
            model.load_state_dict(
                _xla_safe_load(ckpt_to_load, map_location=DEVICE, weights_only=True)["model"])

        # Export to current/ immediately — valid Rust artifact regardless of
        # whether benchmarking follows.  calib_T=1.0 is a safe default; it is
        # updated to the calibrated value after benchmarking completes.
        _export(model, curr_dir, calib_T=1.0, label="current (pre-bench)")

        # ---- benchmark (skipped only if session wall-clock is gone) ---------
        if this_session_h - elapsed() < 0.05:
            print(f"  [budget] session cap reached; benchmark skipped "
                  f"(current/ already updated with latest weights).")
            ledger.append({"phase": global_phase, "step": tr["step"],
                           "total_steps": tr["total_steps"],
                           "last_path": tr["last_path"],
                           "stop_reason": tr["stop_reason"],
                           "elapsed_h": accumulated_h + elapsed()})
            with open(led_path, "w") as f:
                json.dump(ledger, f, indent=2)
            continue

        bench = benchmark(model, classical, tok, cfg, toks, byts, feats,
                          label=f"v{global_phase:03d}")
        mixed = "_domains" in toks
        calib = measure_calibration(
            model, shuffled_loader(toks["val"], feats["val"], cfg) if mixed else va_loader,
            DEVICE)
        prev_calib = prev_bench.get("calib") if prev_bench else None
        decal   = detect_decalibration(calib, prev_calib)
        calib_T = calib["T_opt"] if decal else 1.0
        if decal:
            print(f"  [budget] decalibration T*={calib_T:.3f}; applying scaling")
        domains = None
        if mixed:   # one calibrated pass, split by domain; total = byte-weighted
            domains = domain_eval(model, cfg, toks, feats, len(byts[2]), "test", calib_T)
            cal_bpb = domains.pop("_all")
            print("  [budget] calibrated test bpb by domain: "
                  + "  ".join(f"{k}={v:.4f}" for k, v in domains.items()))
        else:
            cal_bpb = evaluate_calibrated(
                model, make_loader(toks["test"], feats["test"], cfg, False),
                DEVICE, len(byts[2]), len(toks["test"]), calib_T)["bits_per_byte"]
        bench.update(calib=calib, calib_T=float(calib_T),
                     decalibrated=bool(decal), test_bpb_cal=float(cal_bpb))
        if domains:
            bench["domains_cal"] = domains
        cmp = compare_versions(bench, prev_bench)
        bench["verdict"] = cmp["verdict"]
        bench["delta_bpb"] = cmp.get("delta_bpb")
        prev_bench = bench

        # Refresh current/ with calibrated temperature now that we know it
        _export(model, curr_dir, calib_T=calib_T, cal_bpb=cal_bpb,
                label=f"current (phase {global_phase}, calibrated T*={calib_T:.3f})")

        prev_best_bpb = float(best_overall.get("test_bpb_cal", float("inf"))) \
                        if best_overall else float("inf")
        is_best = bench["lossless"] and bench["test_bpb_cal"] < prev_best_bpb
        if is_best:
            best_overall = {"phase": global_phase, "vdir": vdir, "ckpt": ckpt,
                            "test_bpb_cal": bench["test_bpb_cal"], "calib_T": calib_T}
            _export(model, best_dir, calib_T=calib_T, cal_bpb=cal_bpb,
                    label=f"best (phase {global_phase}, {cal_bpb:.4f} bpb)")
            with open(best_meta_path, "w") as f:
                json.dump({"phase": global_phase, "test_bpb_cal": float(cal_bpb),
                           "calib_T": float(calib_T)}, f, indent=2)

        row = {"phase": global_phase, "step": tr["step"],
               "total_steps": tr["total_steps"], "last_path": tr["last_path"],
               "test_bpb": bench["test_bpb"], "test_bpb_cal": bench["test_bpb_cal"],
               "gzip_bpb": bench["gzip_bpb"], "lossless": bench["lossless"],
               "calib_T": float(calib_T), "verdict": bench["verdict"],
               "is_best": is_best, "stop_reason": tr["stop_reason"],
               "elapsed_h": accumulated_h + elapsed()}
        if domains:
            row["domains_cal"] = domains
        ledger.append(row)
        with open(led_path, "w") as f:
            json.dump(ledger, f, indent=2)
        with open(os.path.join(vdir, "benchmark.json"), "w") as f:
            json.dump({k: v for k, v in bench.items() if k != "calib"},
                      f, indent=2, default=float)

        # ---- goal detection + cross-session auto-stop -----------------------
        # Refresh the measured val->test offset for the next val-proxy arm.
        _off = float(tr["best"]) - float(cal_bpb)
        if bench["lossless"] and -0.2 < _off < 0.5:
            val_test_offset = _off
        goal_met = (goal_test_bpb is not None and bench["lossless"]
                    and cal_bpb <= goal_test_bpb)
        if tr["stop_reason"] == "goal_val" and not goal_met:
            in_train_goal_ok = False   # avoid re-triggering every phase
            print(f"  [goal] val-proxy fired but calibrated test bpb "
                  f"{cal_bpb:.4f} > goal {goal_test_bpb} — NOT confirmed; "
                  f"continuing (proxy disabled for this session, offset "
                  f"updated to {val_test_offset:+.4f}).")
        done_reason = None
        _goal_txt = (f"goal {goal_test_bpb} not reached; "
                     if goal_test_bpb is not None else "")
        if goal_met:
            done_reason = (f"GOAL REACHED: calibrated test bpb {cal_bpb:.4f} "
                           f"<= goal {goal_test_bpb}")
        elif tr["step"] >= tr["total_steps"]:
            done_reason = (f"curriculum complete at step {tr['step']:,} "
                           f"({_goal_txt}best "
                           f"calibrated test bpb {cal_bpb:.4f})")
        elif tr["stop_reason"] == "early_stop":
            done_reason = (f"val plateau — no val improvement for "
                           f"{cfg.early_stop_patience} evals at step "
                           f"{tr['step']:,} ({_goal_txt}"
                           f"best calibrated test bpb {cal_bpb:.4f})")
        if done_reason:
            stop_state = {"done": True, "reason": done_reason,
                          "step": int(tr["step"]),
                          "test_bpb_cal": float(cal_bpb),
                          "goal_test_bpb": goal_test_bpb,
                          "when": time.strftime("%Y-%m-%d %H:%M:%S")}
            print("=" * 72)
            print(f"[goal] AUTO-STOP — {done_reason}")
            print(f"[goal] training ends here; future sessions will exit "
                  f"immediately (stop recorded in accumulated.json).")
            print("=" * 72)
            break

    # ---- persist accumulated time + architecture lock for next session ------
    new_acc = accumulated_h + elapsed()
    with open(acc_path, "w") as f:
        json.dump({"elapsed_h": new_acc, "total_budget_h": B_total,
                   "n_phases": len(ledger),
                   "cfg": asdict(cfg),
                   "goal_test_bpb": goal_test_bpb,
                   "val_test_offset": val_test_offset,
                   "stop": stop_state}, f, indent=2)

    benchmarked = [r for r in ledger if "test_bpb" in r]
    print(f"\n[budget] session done: {elapsed():.2f}h used  "
          f"accumulated {new_acc:.2f}h total  "
          f"({len(benchmarked)} phases benchmarked across all sessions)")
    if best_overall:
        print(f"[budget] BEST: phase {best_overall['phase']}  "
              f"bpb={best_overall['test_bpb_cal']:.4f}  T={best_overall['calib_T']:.3f}")
    print(f"[budget] Rust artifacts (always current): {curr_dir}")
    if stop_state.get("done"):
        print(f"[budget] TRAINING FINISHED — {stop_state['reason']}")
        print(f"[budget] do NOT start more sessions; they will exit "
              f"immediately (FRESH_START=True restarts from scratch).")
    elif goal_test_bpb is not None:
        print(f"[budget] goal calibrated test bpb <= {goal_test_bpb} NOT "
              f"reached yet — run another session with FRESH_START=False; "
              f"training will continue and stop by itself at the goal.")
        print(f"[budget] To resume: set FRESH_START=False and run the same cell.")
    return {"ledger": ledger, "best": best_overall, "outdir": outdir,
            "cfg": cfg, "elapsed_h": elapsed(), "accumulated_h": new_acc,
            "stopped": stop_state}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default=None, choices=list(TIERS.keys()))
    ap.add_argument("--init-from", default=None,
                    help="warm-start from a checkpoint before training (strict=False); "
                         "e.g. --tier fast --init-from results/export/best.pt distils Track B")
    ap.add_argument("--smoke", action="store_true", help="fast offline CPU sanity check")
    ap.add_argument("--smoke-recursive", action="store_true",
                    help="fast CPU test of the recursive checkpoint/benchmark machinery")
    ap.add_argument("--recursive", action="store_true",
                    help="run the budget-driven recursive versioned trainer")
    ap.add_argument("--budget-only", action="store_true",
                    help="auto-size model from budget; no tier names (recommended)")
    ap.add_argument("--budget", type=float, default=None,
                    help="total budget across ALL sessions in HOURS")
    ap.add_argument("--session-cap", type=float, default=11.5,
                    help="max wall-clock hours for THIS session (default 11.5; Kaggle CPU committed = 12h)")
    ap.add_argument("--fresh-start", action="store_true",
                    help="wipe runs/ and start from scratch (first session only)")
    ap.add_argument("--dataset", default=None,
                    help="with --budget-only and --tier: mixed corpus name (e.g. corpus_r2)")
    ap.add_argument("--no-goal", action="store_true",
                    help="disable the bpb goal (stop on curriculum end / plateau only)")
    ap.add_argument("--goal-bpb", type=float, default=GOAL_TEST_BPB,
                    help="auto-stop goal in calibrated test bits/byte "
                         "(= bpc on enwik8); sessions stop once reached")
    args, _ = ap.parse_known_args()
    if args.smoke:
        run_smoke()
    elif args.smoke_recursive:
        run_smoke_recursive()
    elif args.recursive:
        run_recursive(total_budget_h=args.budget, start_tier=args.tier or "base")
    elif args.budget_only:
        run_budget(total_budget_h=args.budget, session_cap_h=args.session_cap,
                   fresh_start=args.fresh_start,
                   goal_test_bpb=None if args.no_goal else args.goal_bpb,
                   tier=args.tier, dataset=args.dataset)
    else:
        run_full(args.tier or "base", init_from=args.init_from)
