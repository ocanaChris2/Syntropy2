# Deliverable 3 — Rust Integration Blueprint

How to turn the exported predictor (`export/`) into a streaming, bit-exact, single-threaded
neuro-symbolic compressor in **Rust**. The compressor must reproduce the Python inference
pipeline *exactly*, because the arithmetic/ANS coder is only reversible if the encoder and
decoder see the **identical** probability distribution at every step.

> Language choice: **Rust + `tract`** (pure-Rust ONNX, no C++ build, deterministic). A C
> alternative (ONNX Runtime C API) is sketched at the end.

---

## 0. Exported artifacts (input to Rust)

```
export/
  tokenizer.json          # byte-level BPE — load verbatim with the `tokenizers` crate
  config.json             # all dims: vocab, d_model, n_layers, n_heads, mem_len, conf_dim, …
  classical_meta.json     # dtype/shapes/smoothing/match_order/match_len_cap
  order0.f16              # [V]      unigram probabilities      (float16, little-endian)
  order1.f16              # [V, V]   row-stochastic p(next|prev) (float16, row-major)
  gru_step.onnx          # (token, conf_state, hx) -> (resid_logits, hx_out)
  xformer_window.onnx    # (window_ids[1,W], conf_last[1,1,Cf]) -> resid_logits[1,V]
  gate.onnx              # (gru_h, xf_h, expert_feats[1,1,6]) -> (w[1,1,4], g[1,1,2])
```

**Determinism rule:** do every probability computation in **f32** in Rust, in the *same order*
as Python (`p_cls = w0·p0 + w1·p1 + w2·p2 + w3·pm`, then `z = log(p_cls+ε) + g0·Δgru +
g1·Δxf`). Quantize `p_final` to integer frequencies *once*, with a fixed total, and use the
**same** quantization in encode and decode. This is what guarantees bit-identical round-trips.

---

## 1. Project setup

```
cargo new nsc-codec
```

`Cargo.toml`:

```toml
[package]
name = "nsc-codec"
edition = "2021"

[dependencies]
tract-onnx   = "0.21"        # pure-Rust ONNX inference
tokenizers   = "0.20"        # loads the exact tokenizer.json from Python
serde        = { version = "1", features = ["derive"] }
serde_json   = "1"
half         = "2"           # f16 table loading
byteorder    = "1"
anyhow       = "1"
clap         = { version = "4", features = ["derive"] }

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
```

Module layout:

```
src/
  main.rs          # CLI: compress / decompress
  config.rs        # serde structs for config.json + classical_meta.json
  tok.rs           # tokenizer wrapper (encode bytes -> ids, ids -> bytes)
  classical.rs     # order0/order1 tables, order2 + match models, confidence state
  neural.rs        # tract sessions: gru_step, xformer_window, gate
  fusion.rs        # p_cls mixture, residual application, neural-skip, quantization
  coder.rs         # range coder (primary) + rANS (alternative)
  engine.rs        # the streaming neuro-symbolic loop (encode & decode share it)
  tests.rs         # bit-identical round-trip harness
```

---

## 2. Tokenization (exact parity, zero re-implementation)

Because Python saved a standard `tokenizer.json`, load it directly — no custom BPE in Rust:

```rust
use tokenizers::Tokenizer;
let tok = Tokenizer::from_file("export/tokenizer.json")?;
// encode: bytes -> latin-1 string -> ids   (matches the Python latin-1 bijection)
let s: String = input_bytes.iter().map(|&b| b as char).collect();
let ids: Vec<u32> = tok.encode(s, false)?.get_ids().to_vec();
// decode: ids -> latin-1 string -> bytes
let out = tok.decode(&ids, false)?;
let bytes: Vec<u8> = out.chars().map(|c| c as u8).collect();
```

This is the single most important parity win: the Rust and Python token streams are identical
by construction.

---

## 3. Classical models in Rust (mirror `ClassicalStats`)

```rust
struct Classical {
    v: usize,
    order0: Vec<f32>,            // [V]
    order1: Vec<f32>,            // [V*V] row-major, row = prev token
    // order-2 + match are recomputed online from the decoded token stream so the
    // decoder reproduces the encoder's features without storing them:
    o2: HashMap<u64, (u32 /*top*/, f32 /*conf*/)>,   // built like Python's _o2_counts
    match_table: HashMap<u64, usize>,                // k-gram hash -> last index
    match_run: usize,
    k: usize, match_cap: f32,
}
```

Load `order0.f16` / `order1.f16` with the `half` crate and widen to f32 once at startup.

> **Static vs adaptive:** Python froze the classical tables from the *training* split. The Rust
> order-0/order-1 are loaded verbatim (static). The **order-2 and match models are rebuilt
> online** from the *already-decoded* prefix — both encoder and decoder see the same prefix, so
> their `(top, conf)` features match exactly. (Order-2 here is the online variant; if you want
> *bit-identical* parity with the Python static order-2, instead export the order-2 table too
> and load it. For a self-contained codec, online is simpler and still reversible.)

Per step, build the four classical distributions and the confidence state exactly as
`HybridLM.classical()` does:

```rust
// p0 = order0 ; p1 = order1[prev*V ..] ; p2 = peaked(o2_top, o2_conf) ; pm = peaked(m_top, m_conf)
// conf_state = [max(p1), entropy(p1)/ln(V), o2_conf, m_conf]
fn peaked(top: u32, conf: f32, v: usize) -> Vec<f32> {
    let c = conf.clamp(1e-4, 1.0 - 1e-4);
    let base = (1.0 - c) / (v as f32 - 1.0);
    let mut p = vec![base; v];
    p[top as usize] = c;
    p
}
```

---

## 4. Neural experts via `tract`

```rust
use tract_onnx::prelude::*;
let gru = tract_onnx::onnx().model_for_path("export/gru_step.onnx")?
        .into_optimized()?.into_runnable()?;
let xfw = tract_onnx::onnx().model_for_path("export/xformer_window.onnx")?
        .into_optimized()?.into_runnable()?;
let gate = tract_onnx::onnx().model_for_path("export/gate.onnx")?
        .into_optimized()?.into_runnable()?;
```

- **GRU** carries an explicit hidden state `hx: [n_layers, 1, gru_hidden]` between steps
  (O(1) per token — fast).
- **Transformer (window)**: keep a ring buffer of the last `W = mem_len` token ids; each step
  feed `window_ids[1,W]` (left-padded with 0) + `conf_last[1,1,Cf]`; read `resid[1,V]`.
  *(This is O(W) recompute per token — correct but slow; see §8 for the KV-cache speedup.)*
- **Gate** takes the two neural hidden states + the 6 live expert features and returns
  `w[1,1,4]` and `g[1,1,2]`.

---

## 5. Fusion + quantization (`fusion.rs`)

```rust
// 1. classical mixture (probability space) — SAME ORDER as Python
let mut p_cls = vec![0f32; v];
for i in 0..v { p_cls[i] = w0*p0[i] + w1*p1[i] + w2*p2[i] + w3*pm[i]; }

// 2. residual in log space (only the experts the gate/skip kept)
//    z = ln(p_cls + eps) + g0*resid_gru + g1*resid_xf
let mut z = vec![0f32; v];
for i in 0..v { z[i] = (p_cls[i] + EPS).ln() + g0*rg[i] + g1*rx[i]; }

// 3. stable softmax -> p_final
softmax_inplace(&mut z);                       // now z == p_final

// 4. quantize to integer frequencies with a FIXED total (e.g. 1<<16),
//    guaranteeing every symbol has freq >= 1 (coder must never see prob 0)
let freqs = quantize_to_cum(&z, TOTAL_BITS);   // identical in encode & decode
```

`quantize_to_cum` must be **deterministic and identical** on both sides: scale by `2^TOTAL_BITS`,
floor, bump every zero to 1, then add the leftover to the argmax bucket. Build the cumulative
table once; the coder indexes into it.

**Neural-skip in the loop:** before running the transformer, check
`max(o2_conf, m_conf) > tau` (from `config.json`); if so set `g1 = 0`, skip the `xfw` session
entirely, and (optionally) skip the GRU too. The distribution is then the classical mixture,
computed identically on encode and decode.

---

## 6. Streaming neuro-symbolic engine (`engine.rs`)

Encoder and decoder share one function so they can **never** diverge — the only difference is
whether the actual token is *known* (encode) or *pulled from the coder* (decode).

```text
fn step(state, coder, mode) -> token:
    prev      = state.context.last()
    # --- classical ---
    p0,p1,p2,pm, conf_state, o2_conf, m_conf = classical.predict(state.context)
    # --- gating needs neural hidden states; get them (respecting skip) ---
    (resid_gru, hx2) = gru.run(prev_token, conf_state, state.hx)        # O(1)
    skip = max(o2_conf, m_conf) > tau
    if skip:
        resid_xf = 0 ; xf_h = 0
    else:
        window  = state.context.last(mem_len)            # ring buffer, left-pad
        resid_xf, xf_h = xfw.run(window, conf_state)      # O(mem_len)
    (w, g) = gate.run(gru_h, xf_h, expert_feats)
    if skip: g[1] = 0
    # --- fusion + quantize (identical both sides) ---
    p_final = fuse(p0,p1,p2,pm, resid_gru, resid_xf, w, g)
    cum     = quantize_to_cum(p_final)
    # --- entropy coding ---
    if mode == ENCODE:
        token = state.next_input_token            # known
        coder.encode(cum, token)
    else: # DECODE
        token = coder.decode(cum)                 # pulled from the stream
    # --- update ALL state with the (now known) token ---
    state.hx = hx2
    state.context.push(token)
    classical.update_online(token)                # order-2 counts, match table, match_run
    return token
```

**Memory management:**
- `context`: a `VecDeque<u32>` (or ring buffer) capped at `mem_len`.
- `hx`: a single `Tensor` reused in place across steps.
- classical online tables grow with the stream; cap the match table with an LRU if compressing
  huge files.
- No per-step heap churn in the hot path: preallocate `p_cls`, `z`, `cum` buffers and reuse.

**Bit-exactness checklist** (each must match Python *and* match between encode/decode):
1. token stream (guaranteed by shared `tokenizer.json`);
2. classical features (order-2/match rebuilt from the same prefix);
3. f32 fusion done in the **same arithmetic order**;
4. `quantize_to_cum` deterministic with a fixed total and no zero frequencies;
5. coder renormalization identical.

---

## 7. Entropy coding (`coder.rs`)

### 7.1 Range coder (primary — easiest to make streaming + bit-exact)

A classic 32-bit carryless range coder. State: `low: u32`, `range: u32`. `cum` is the
cumulative-frequency table with total `T = 1<<TOTAL_BITS`.

```text
ENCODE(cum, sym):
    r = range / T
    low   += r * cum.start(sym)
    range  = r * cum.freq(sym)
    while (low ^ (low+range)) < TOP  ||  (range < BOT && {range = -low & (BOT-1); true}):
        emit_byte(low >> 24); low <<= 8; range <<= 8     # renormalize, carry-safe

DECODE(cum):
    r   = range / T
    val = (code - low) / r                  # code holds the next 32 bits of the stream
    sym = cum.find(val)                      # binary search in the cumulative table
    low   += r * cum.start(sym)
    range  = r * cum.freq(sym)
    renormalize (mirror of encode, refilling `code` from the byte stream)
    return sym
```

- `TOP = 1<<24`, `BOT = 1<<16`, `TOTAL_BITS = 16` (so `range/T` never underflows).
- `cum.find` is a binary search over the cumulative table (`O(log V)`).
- Flush 4 bytes of `low` at the end; the decoder primes `code` with the first 4 bytes.

### 7.2 rANS (alternative — faster, table-friendly)

Interleaved/streaming rANS (à la Fabian Giesen's `ryg_rans`) with the **same** `(freq, start,
total=1<<16)` tables:

```text
state x : u32, normalized to [2^23, 2^31)
ENCODE (process tokens in REVERSE, emit bytes):
    while x >= ((2^23 >> 16) << 8) * freq(sym): { emit(x & 0xFF); x >>= 8 }
    x = ((x / freq) << 16) + (x % freq) + start
DECODE (forward):
    sym  = find( x & 0xFFFF )
    x    = freq(sym) * (x >> 16) + (x & 0xFFFF) - start
    while x < 2^23 { x = (x << 8) | next_byte() }
```

rANS needs the symbol stream reversed at encode time (buffer tokens, encode back-to-front,
output is read front-to-back at decode). The range coder avoids that and is the recommended
default for a first correct implementation; switch to rANS for throughput once correct.

---

## 8. Performance

**Reality (see EXECUTION_PLAN §6):** ≥1 MB/s single-threaded with a per-token transformer is
not physically attainable; expect **KB/s** when the neural path runs, and **MB/s only when
neural-skip bypasses it** on redundant data. To get as close as possible:

1. **Incremental KV cache (biggest win).** The exported `xformer_window.onnx` recomputes the
   whole window every token (O(mem_len) layers·positions). For production, replace it with an
   incremental cache: export a second ONNX `xformer_step` that takes a **fixed-size** KV cache
   `(K,V: [L,1,H,mem_len,hd])` + write position + the single new token, applies RoPE at the
   absolute position, writes into the ring slot, and returns `resid[1,V]` + updated cache. This
   turns per-token cost from O(mem_len·L·...) into O(context) attention only. (Verify parity
   against PyTorch the same way `export_all` does.)
2. **Neural-skip aggressively.** Raise `neural_skip_tau` to skip the transformer on more tokens
   (ratio/speed dial). Track and report the skip rate — it *is* the throughput story.
3. **int8 quantization** via tract for the GRU/Transformer (`nano` tier benefits most).
4. **Reuse buffers**, avoid allocations in the hot loop, and keep tables in f32 contiguous
   arrays for cache locality. Binary-search the cumulative table, don't linear-scan.
5. **GRU-only fast mode**: for a speed-priority build, gate the transformer off entirely and
   rely on classical mixture + GRU residual + match model — close to the `nano` design point.

Report a **throughput-vs-ratio curve** across `tau` values rather than a single number.

---

## 9. Test harness (`tests.rs`) — bit-identical round-trip

```rust
#[test]
fn roundtrip_bit_identical() {
    for path in ["tests/text.txt", "tests/binary.bin", "tests/repetitive.log"] {
        let original = std::fs::read(path).unwrap();
        let mut codec = Codec::load("export/").unwrap();
        let packed   = codec.compress(&original);      // runs engine in ENCODE mode
        let restored = codec.decompress(&packed);      // runs the SAME engine in DECODE mode
        assert_eq!(original, restored, "round-trip mismatch for {path}");
        let bpb = packed.len() as f64 * 8.0 / original.len() as f64;
        eprintln!("{path}: {bpb:.4} bits/byte");
    }
}
```

Additionally, a **cross-language parity test**: dump Python's `p_final` for the first N tokens
of a fixed input to JSON, load it in Rust, and assert `max|p_rust − p_py| < 1e-4` *before* the
coder. If the round-trip ever fails, this isolates whether the divergence is in the model math
(parity test fails) or the coder (parity passes, round-trip fails).

---

## 10. C alternative (ONNX Runtime C API)

If C is required instead of Rust:

- **Inference:** ONNX Runtime C API (`OrtCreateSession` per `.onnx`, `OrtRun` per step). Heavier
  dependency (prebuilt `libonnxruntime`), manual `OrtValue` lifetime management.
- **Tokenizer:** no official C BPE that reads `tokenizer.json`; either (a) link the Rust
  `tokenizers` crate via its C FFI, or (b) re-implement byte-level BPE merges from the JSON
  (merge ranks + vocab) — fully specified by the file, but you must replicate the byte-level
  pre-tokenizer mapping exactly.
- **Classical tables:** `fread` the `.f16` blobs, widen to `float` with an f16→f32 helper.
- **Coder:** the range/rANS pseudo-code in §7 is already C-shaped; port directly.
- **Determinism:** compile the fusion math with `-ffp-contract=off` (or compute in a fixed
  order) so FMA reassociation can't break encode/decode parity across builds.

The streaming loop, memory model, neural-skip, and bit-exactness checklist are identical to the
Rust design.
