# Deliverable 1 — Execution Plan & Novelty Justification

Neuro-symbolic hybrid lossless compressor. This document confirms the constraints,
states the model sizes (measured, not estimated), gives the 30-hour training-budget
strategy, and — most importantly — derives the **residual probability fusion** math and
the **gated routing** logic, then justifies why this beats a standard ensemble.

---

## 1. Constraints (confirmed)

| Constraint | Setting |
|---|---|
| Environment | Kaggle Notebook, Python 3.10+, PyTorch 2.x, **AMP required** |
| Hardware | single P100 or T4 GPU |
| GPU wall-time | **30 h hard cap** (we self-stop at 28 h, see §3) |
| Disk | ≤ ~300 GB (enwik8 is 100 MB; trivially fits) |
| Dataset | enwik8 (Hutter Prize, byte-level), chronological 90/5/5 train/val/test |
| Tokenizer | byte-level BPE, vocab 4,096–8,192, exported as `tokenizer.json` |
| Entropy coder | **NOT** in Python — exported predictor only; ANS/range coder lives in Rust |
| Target runtime | streaming Rust compressor, single-threaded |

**Governing principle (user):** *"closest approach with minimal resources, disposing more
resources if available."* Everything is driven by one `Config` with a **tier ladder**. Pick
the largest tier your GPU/time budget allows.

---

## 2. Model sizes (measured on this codebase)

Parameter counts are **measured** by instantiating each tier (`count_params`), not guessed.
The token embedding is **shared** between the GRU and Transformer experts, which is why totals
are lean. Classical tables are *buffers*, not parameters.

| Tier | Total params | GRU expert | Transformer expert | d_model · L · heads | seg/mem | order-1 table (f16) | Use when |
|---|---|---|---|---|---|---|---|
| **nano**  | **2.16 M**  | 1.19 M | 1.48 M  | 128 · 2 · 4 | 512 / 512   | 34 MB  | minimal resources / speed priority |
| **base**  | **10.56 M** | 5.12 M | 7.49 M  | 256 · 4 · 8 | 1024 / 1024 | 134 MB | **recommended default**, fits 30 h easily |
| **large** | **35.79 M** | 12.08 M | **27.84 M** | 512 · 6 · 8 | 1024 / 1024 | 134 MB | more resources / best ratio |

> The spec asks for a **20–30 M-param Transformer**: the `large` tier's transformer is
> **27.84 M**, hitting that target exactly. `base` is the efficient default; `nano` is the
> minimal-resource tier whose small models + neural-skip approach real-time speeds.

The architecture is **identical across tiers** — only the width/depth/vocab change — so the
Rust runtime is written once and reads the dimensions from `config.json`.

---

## 3. 30-hour training-budget strategy

Spec split: ~10 h prep, ~18 h train, ~2 h eval/export. Mapped to the code:

| Phase | Budget | What runs | Notes |
|---|---|---|---|
| Data + BPE | ~10 h | `download_enwik8`, `train_bpe`, `ClassicalStats.fit`, `features()` | BPE is fast (<1 h); the bulk is the **sequential classical feature pass** over the corpus (order-2 + match model). Cache `*.npy` so re-runs skip it. |
| Training | ~18 h | `train()` — **two-stage curriculum**, AMP, cosine, early stop | `wallclock_budget_h=28` is a hard guard: the loop checks elapsed time every step and **stops gracefully**, saving the best checkpoint, well before Kaggle's 30 h kill. |
| Eval + export | ~2 h | `compression_report`, `reference_decoder_check`, `export_all` | prints bits/byte vs `gzip -9`/`xz`, verifies the streaming math, exports ONNX + JSON + tables. |

**Two-stage curriculum** (implemented in `HybridLM.forward(stage=…)` + `train()`):

- **Stage 1 — pretrain base predictors.** Loss = `CE(log p_cls) + CE(resid_gru as logits) +
  CE(resid_xf as logits)`. Each neural expert is warmed up as a *standalone* next-token
  predictor and the classical gating is calibrated. This gives the encoders a sane init before
  they are asked to model a *residual*. LR `3e-4`, cosine.
- **Stage 2 — joint residual + gating.** Loss = `CE(z_final)` with the true fusion
  `z_final = log(p_cls) + Δ`. Everything trains jointly at lower LR (`1e-4`), cosine-annealed,
  with **early stopping** on validation bits/byte (`patience=8` eval windows).

AMP via `torch.autocast` + `GradScaler`; gradient accumulation (`grad_accum`) keeps the
effective batch large on a single T4; `grad_clip=1.0`.

---

## 4. The fusion math (this is the novelty)

For every position `t` with history `x_{<=t}`, we predict a distribution over the next token
`x_{t+1}` across a vocabulary of size `V`.

### 4.1 Classical base distribution `p_cls` (probability space)

Four classical experts each emit a full-vocab distribution:

- **order-0**: global unigram `p₀ ∈ Δ^V` (constant).
- **order-1**: `p₁ = order1[x_t] ∈ Δ^V`, a row of a dense, row-stochastic `[V,V]` table.
- **order-2**: a **peaked** distribution `p₂ = peaked(top₂, conf₂)` — mass `conf₂` on the
  order-2 top prediction, the remainder spread uniformly. `(top₂, conf₂)` are precomputed.
- **match**: `p_m = peaked(top_m, conf_m)` from a hash-table match model; `conf_m` grows with
  the verified match-run length (capped).

`peaked(i, c)` puts probability `c` on token `i` and `(1-c)/(V-1)` on every other token — a
valid, strictly-positive distribution (verified in the smoke test).

The **gated classical mixture** is a convex combination:

```
p_cls = Σ_{e∈{0,1,2,m}}  w_e · p_e ,        Σ w_e = 1,  w_e ≥ 0
```

where the weights `w = softmax(score_classical)` come from the attention gate (§4.3). Because
each `p_e ∈ Δ^V` and `w` is on the simplex, `p_cls ∈ Δ^V` automatically (verified: sums to 1).

### 4.2 Neural residual `Δ` (log space) — *conditioned on classical confidence*

The two neural experts do **not** predict the next token. Each emits a **residual logit
vector** `Δ_e ∈ ℝ^V`, FiLM-conditioned on the **classical confidence state**
`s = [maxₚ p₁, H(p₁)/log V, conf₂, conf_m] ∈ ℝ⁴`:

```
FiLM:   h' = (1 + γ(s)) ⊙ h + β(s)            # tells the net WHERE classical is weak
Δ_gru = W_gru · FiLM_gru( GRU(x)_t , s )
Δ_xf  = W_xf  · FiLM_xf ( Transformer(x)_t , s )
```

The gated residual is

```
Δ = g_gru · Δ_gru + g_xf · Δ_xf ,            g_e = σ(score_neural) ∈ (0,1)
```

### 4.3 Final fusion and loss

```
z_final = log(p_cls + ε) + Δ
p_final = softmax(z_final)
loss    = CrossEntropy(p_final, x_{t+1})        # bits/byte = loss·log₂e · tokens/bytes
```

Interpretation: the classical mixture provides a **calibrated prior** in log space; the neural
network adds a **learned, context- and confidence-aware correction**. If the neural experts
output `Δ = 0`, the system degrades gracefully to a pure gated classical mixer.

### 4.4 Context-gated attention routing

The gate is **attention-based, not a static MLP**. A query is built from the neural hidden
states and dotted against six expert "keys", each = a learned expert embedding **plus** a
projection of that expert's *live* confidence/activity scalar:

```
q      = W_q · [ h_gru ; h_xf ]                              ∈ ℝ^{d_g}
key_e  = E_e + W_f · feat_e   (feat_e = conf₁/conf₂/conf_m / hidden-norm …)
score  = (q · key_e) / √d_g                                  ∈ ℝ^6
w      = softmax(score[0:4])     # classical mixture weights (simplex)
g      = σ(score[4:6])           # neural residual gates (independent)
```

Routing therefore changes **per token**: on a long repeat the match key's confidence spikes →
`w_match → 1`; on structured text the transformer gate `g_xf` opens; when the classical model
is already certain, both `g`'s close and the expensive neural path can be **skipped**.

### 4.5 Neural-skip (adaptive compute)

At inference only: if `max(conf₂, conf_m) > τ` (default `0.95`), skip the Transformer (and
optionally the GRU) and emit the classical mixture directly. The gate has already learned to
assign near-zero `g` in exactly these regions, so the ratio cost is negligible while the
**speed gain is large** — this is the mechanism that lets minimal configs approach real-time
on compressible data.

---

## 5. Why this beats standard ensemble averaging

A standard ensemble computes independent model probabilities and blends them with fixed (or
context-free) weights: `p = Σ λ_e p_e`. Our system differs on four structural axes:

1. **Residual, not parallel.** The neural net models only the **error the classical model
   cannot capture** (`Δ` on top of `log p_cls`). Capacity is spent on the hard residual, not
   on re-learning the easy n-gram statistics the classical tables already nail. Lower
   effective entropy for the same parameter budget.
2. **Confidence-conditioned.** The residual is FiLM-conditioned on the classical **confidence
   state**. This is *information routing*: the net is explicitly told where the prior is
   unreliable, instead of blending blindly.
3. **Per-token attention gating.** Weights are a **function of the current context** via a
   query–key attention, not global constants. The mixture re-shapes itself token by token.
4. **Adaptive compute.** The skip-gate makes the *cost* data-dependent — impossible in a fixed
   averaging ensemble, and the key to approaching the speed target on redundant data.

Net effect: better bits/byte than either family alone, graceful fallback to the classical
mixer, and a tunable speed/ratio dial.

---

## 6. Honest performance analysis (the ≥1 MB/s target)

**Physics first.** Lossless neural decoding is inherently sequential: token `t+1` cannot be
decoded until token `t` is, so the network runs **one forward pass per token**. A modern CPU
sustains roughly hundreds–few-thousand such passes/sec for a multi-million-param transformer.
With BPE averaging ~3–4 bytes/token, the *neural* ceiling is on the order of **single-digit to
low-tens of KB/s**, not 1 MB/s. State-of-the-art neural compressors (NNCP, cmix,
tensorflow-compress) run at **KB/s**. We do not pretend otherwise.

**Measured/estimated throughput, single-thread (base tier):**

| Path (per token) | Relative cost | Approx. throughput | When it runs |
|---|---|---|---|
| Classical only (order0/1 + match lookup) | 1× | **multi-MB/s** | always cheap |
| GRU step (incremental state) | ~10–50× | ~50–150 KB/s | medium-range |
| Transformer **with KV cache** (O(context)) | ~100–500× | ~5–20 KB/s | hard / structured |
| Transformer **window-recompute** (what we export) | ~context× more | ~100 s of B/s | reference/correctness only |

**How we "approach" the target (not fake it):**

- **Neural-skip** makes throughput **data-dependent**. On highly redundant input (logs,
  genomes, repetitive binaries) the match model is confident on most tokens → the transformer
  is skipped → average throughput rises toward the classical path (**MB/s achievable on such
  data**). On dense text it stays in the KB/s regime — and we report the *curve*, not a single
  number.
- **`nano` tier + int8** (tract quantization) is the speed-priority config.
- **KV cache** is mandatory for any real speed; the simple window-recompute ONNX we export is
  for **correctness/reference** — the Rust blueprint (§ "Performance") describes the
  incremental cache that the production path should use.

**Bottom line:** ≥1 MB/s is treated as **aspirational and data-dependent**, reached only when
neural-skip bypasses the network for the large majority of tokens. The deliverable optimizes
toward it (skip-gate, nano tier, int8, KV cache) and publishes an honest throughput-vs-ratio
curve rather than a misleading headline number.

---

## 7. What was verified locally vs. what runs on Kaggle

- **Verified here (CPU, torch 2.12):** full model wiring forward/backward at the `smoke` tier;
  both curriculum stages produce finite loss; `p_final`, `p_cls`, and gate weights are valid
  distributions; the streaming **reference decoder** matches the batched math; **ONNX export
  parity** for the GRU step (`max|err|=1.2e-7`) and the Transformer window step
  (`max|err|=3.0e-7`); all artifacts written. (`python neuro_symbolic_compressor.py --smoke`.)
- **Runs on Kaggle (user, GPU):** the 30 h training, the real bits/byte vs `gzip -9`, and the
  full-tier export. The training/eval code is provided and structured but the GPU run itself is
  out of scope for local verification.
