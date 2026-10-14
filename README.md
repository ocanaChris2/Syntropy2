# Syntropy2 — Neuro-Symbolic Hybrid Lossless Compressor

A lossless compressor built on a **deeply-intertwined** neuro-symbolic architecture.
Classical context models (order-0/1/2 + match) produce a base probability distribution;
neural experts (a local GRU and a Transformer-XL-style long-range model) predict a
**residual logit delta** *on top of* that distribution, **conditioned on the classical
model's confidence state** (FiLM). A **context-gated attention fusion** engine routes
between experts per token, and a **confidence-gated neural-skip** path drops the
Transformer on redundant data for speed. This is *not* an ensemble average — see
`docs/EXECUTION_PLAN.md` §4–§5 for the fusion math and the novelty justification.

Training runs in a Kaggle notebook (≤30 h GPU); the trained predictor is exported to
ONNX + JSON + binary tables and consumed by a streaming **Rust** compressor with a range
/ rANS entropy coder (`docs/RUST_INTEGRATION_BLUEPRINT.md`).

---

## Project layout

```
Syntropy2/
├── python/              Research side (PyTorch): model, training, benchmark, integrity test
│   ├── neuro_symbolic_compressor.py   model + trainer + export (Kaggle script, `# %%` cells)
│   ├── benchmark_enterprise.py        benchmark harness (Part 1 A–G, Part 2 H–K)
│   ├── test_compression_integrity.py  true encode/decode + tamper suite
│   ├── int8_codec.py / int8_codec_test.py   integer-exact reference codec (spec for Rust)
├── nsc-codec/           Rust codec `nsc` (CM engine, recoders, NSC3 container) + `scripts/` (exporters, tok.py)
├── nsc-codec/gui/       Native desktop app `nsc-gui`
├── notebooks/           Jupyter versions (GPU, TPU, benchmark); r2 kernel in kaggle_r2_kernel/
├── docs/                WHITEPAPER, EXECUTION_PLAN, RUST_INTEGRATION_BLUEPRINT, HYBRID_MODEL_ANALYSIS, training strategy
├── tools/               Corpus build, model install, Kaggle push scripts
├── data/ results*/ kaggle_*/ benchmark_output/   Generated data, checkpoints, Kaggle staging (git-ignored, large)
```

Python scripts resolve the project root themselves; run them from anywhere, e.g.
`python python/neuro_symbolic_compressor.py --smoke`.

---

## Quick start (local sanity check — no GPU, no downloads)

The script has **no import-time side effects**; all orchestration lives under `__main__`.
A fast, fully offline CPU smoke test exercises the entire model wiring (both curriculum
stages, valid-distribution checks, gating shapes, a reference streaming decoder, and ONNX
export parity) at a tiny `smoke` tier:

```bash
cd Syntropy2
python python/neuro_symbolic_compressor.py --smoke
```

Expected tail:

```
  stage 1: loss=...  OK
  stage 2: loss=...  OK
  backward+step OK (grad norm=...)
  ref-decoder bits/token = ...
== SMOKE TEST PASSED ==
```

The smoke test falls back to a built-in byte-level tokenizer if the `tokenizers`
package is absent, so it runs anywhere torch is installed. ONNX export parity is
checked only if `onnx`/`onnxruntime` are present (skipped gracefully otherwise).

---

## Running on Kaggle (the real workload)

1. Create a **GPU notebook** (P100 or T4 ×2) with internet **on** (needed once to fetch
   `enwik8` and, optionally, `pip install tokenizers onnx onnxruntime`).
2. Either:
   - **Upload `notebooks/neuro_symbolic_compressor_gpu.ipynb`** and *Run All*, or
   - paste `neuro_symbolic_compressor.py` into a cell — it is split with `# %%` markers,
     so Kaggle/Jupytext/VS Code render it as ordered cells.
3. The first cell installs missing deps (`ensure_deps()` — call it explicitly in a
   notebook cell; it never runs at import). Then drive the run:

   ```python
   run_full("base")     # or "nano" / "large"
   ```

   or from a terminal cell:

   ```bash
   python python/neuro_symbolic_compressor.py --tier base
   ```

`run_full` performs the whole pipeline: download `enwik8` (100 MB Wikipedia XML, split
90/5/5 train/val/test by megabytes) → train byte-level BPE → fit the static classical
tables and precompute per-position features → two-stage curriculum training under AMP
(`stage 1`: pretrain neural predictors + classical gating; `stage 2`: joint residual +
gating fine-tune) with cosine LR, grad-accum, early stopping and a **wall-clock budget
guard** that stops gracefully before 30 h → print a bits-per-byte report **vs `gzip -9`
and `xz -9`** → verify a pure-Python reference decoder matches the batched math →
**export everything and assert ONNX↔PyTorch parity**.

> Set `NSC_WORKDIR=/kaggle/working` (or any writable dir) to control where `data/` and
> `export/` land. Defaults to the script's own directory.

---

## Scaling resources: the tier ladder

Governing principle: **closest approach with minimal resources, scaling up when more are
available.** One config knob (`--tier`) moves the whole system along the speed↔ratio
curve. Pick the largest tier that fits your GPU memory and 30 h budget.

| Tier | Vocab | `d_model` | Layers | Seg/Mem | Batch | Steps (s1+s2) | Params (total / transformer) | Use when |
|------|------:|----------:|-------:|--------:|------:|--------------:|------------------------------|----------|
| `nano`  | 4096 | 128 | 2 | 512  | 16 | 20k + 60k  | **2.16 M** | Speed-priority / quick iteration; the minimal-resource path. |
| `base`  | 8192 | 256 | 4 | 1024 | 8  | 40k + 120k | **10.56 M** | Default. Best balance on a single P100 in 30 h. |
| `large` | 8192 | 512 | 6 | 1024 | 6  | 50k + 160k | **35.79 M** (transformer **27.84 M**) | Ratio-priority; transformer hits the spec's 20–30 M target. |
| `smoke` | 256  | 32  | 2 | 32   | 2  | 2 + 2      | tiny | CPU wiring test only (`--smoke`). Not for real data. |

Override any single field without editing the file, e.g.:

```python
cfg = get_config("base", seg_len=2048, lr_stage2=1.5e-4)
```

Higher tiers raise the **neural-skip threshold `τ`** payoff differently: `nano`+int8 is
the configuration that *approaches* the speed target on redundant data; `large` maximizes
compression ratio at the cost of throughput.

---

## Exported artifacts (`<workdir>/export/`)

The Rust/C compressor consumes exactly these (see `docs/RUST_INTEGRATION_BLUEPRINT.md`):

| Artifact | Purpose |
|----------|---------|
| `tokenizer.json` | Byte-level BPE — loaded by the Rust `tokenizers` crate for **exact** encode/decode parity. |
| `config.json` | Tier hyper-parameters, vocab size, ε, skip threshold `τ`, fusion dims. |
| `order0.f16` + `order1.f16` + `classical_meta.json` | The **static** frozen order-0/order-1 tables (bit-exact Python↔Rust). **Order-2 and the match model are *not* exported as tables** — Rust recomputes them online from the decoded prefix; `classical_meta.json` carries their hyper-params (`match_order`, `match_len_cap`, smoothing). |
| `gru_step.onnx` | Single-step GRU residual predictor. |
| `xformer_window.onnx` | Fixed-window Transformer step returning last-position residual logits (tract-friendly). |
| `gate.onnx` | Context-gated attention fusion (emits classical mixture weights `w` and neural gates `g`). |
| `best.pt` | Best checkpoint (loaded with `weights_only=True`). |

---

## Rust integration

`docs/RUST_INTEGRATION_BLUEPRINT.md` is the build guide: a `tract-onnx` + `tokenizers` +
`byteorder` Cargo project, a streaming engine where **encode and decode share one
`step()`** (classical → confidence → neural residual w/ skip gate → FiLM → gated fusion →
range-coder), a 32-bit carryless range coder plus an rANS alternative, a bit-identical
round-trip `cargo test`, and a C/onnxruntime equivalent. Classical order-2/match tables
are rebuilt online from the decoded prefix so the decoder needs only the small static
tables, not the full corpus.

---

## Honest performance caveat

**≥1 MB/s single-threaded CPU decode is not physically attainable** with a per-token
Transformer forward pass (hundreds–few-thousand passes/sec ⇒ ~KB/s with multi-byte BPE
tokens). This is stated up front, not hidden. The design's answers:

- **Neural-skip** makes average throughput **data-dependent** — redundant regions run on
  the classical fast path and can approach multi-MB/s.
- **`nano` + int8** is the explicit speed-priority configuration.
- The execution plan publishes a **throughput-vs-ratio curve** and documents the gap
  rather than claiming a met SLA.

Compression *ratio*, by contrast, is expected to comfortably beat `gzip -9` and be
competitive with / better than `xz -9` on `enwik8`, which the notebook measures directly.

---

## Verification status

- **Local (CPU):** `python python/neuro_symbolic_compressor.py --smoke` — full model wiring,
  both curriculum stages (finite loss, valid simplex distributions, correct gating
  shapes), backward + optimizer step, reference streaming decoder, and ONNX export parity
  (GRU ≈ 1e-7, Transformer ≈ 3e-7). ✅ Passing.
- **Kaggle (you):** full `run_full(tier)` prints bits/byte beating `gzip -9`; the export
  cell asserts ONNX↔PyTorch `allclose`.
- **Rust (you):** `cargo test` round-trip asserts bit-identical decompression.

> The 30 h Kaggle GPU training and the Rust build are **out of scope for local
> execution** in this environment — runnable code, a CPU smoke test, and a complete
> integration blueprint are provided so you can execute both.
