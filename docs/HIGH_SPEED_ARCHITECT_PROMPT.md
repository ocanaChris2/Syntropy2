# Architect Prompt — High-Speed Syntropy2 / `nsc` Neural Path

A self-contained prompt to hand to an architect (human or LLM) to design the high-speed
version of the neural path **without losing the 6–10× compression ratios**. It encodes the
objective, the hard invariants, the measured starting point, and the decision structure, so
the architect produces a concrete plan rather than a generic essay.

---

## How to use this file

- **Primary prompt** (§"The Prompt") — paste verbatim into the architect session.
- **Review prompt** (§"Review Prompt") — hand the returned design to a second pass to stress
  it for correctness and ratio regressions.
- **Acceptance checklist** (§"Acceptance Criteria") — the objective bar the design must clear.

Attach or reference: `nsc-codec/README.md`, `docs/WHITEPAPER.md`,
`docs/HYBRID_MODEL_ANALYSIS.md`, `docs/RUST_INTEGRATION_BLUEPRINT.md`,
`nsc-codec/src/forward.rs`, `nsc-codec/src/codec.rs`, `nsc-codec/src/main.rs`.

---

## The Prompt

> You are a compression-systems architect. Design the **high-speed evolution of the neural
> text path** in the `nsc` codec (`nsc-codec/`, pure Rust, zero third-party crates in the
> codec crate), so that it runs at **multi-MB/s class throughput while preserving the
> existing 6–10× compression ratios** on the data where those ratios are achieved today.
> Do **not** assume a full rewrite is required — determine, with evidence, the minimum set
> of changes. Where a rewrite *is* required, justify it against a cheaper alternative.
>
> ### Context you must internalize before designing
>
> 1. **Where the ratio comes from.** Per `nsc-codec/README.md`, the 6–10× figures are
>    delivered by the **classical context-mixing (CM) engine and the format-aware
>    recompressors** (deflate/PNG/JPEG/…), which already run at MB/s-class speed:
>    Text 1.292 bits/byte (6.2×), Images 1.296 (6.2×), Silesia 1.353 (5.9×), and 3–10× on
>    PDF/Office via deflate recompression. The **neural model's own trained figure is
>    bpc ≈ 1.28** — essentially the same as the CM text result. The neural model is therefore
>    a **marginal, optional contributor** to ratio and the **only KB/s-class component**.
>    Design must not sacrifice the cheap ratio to speed up the expensive marginal part.
>
> 2. **The fundamental wall.** The neural path is **autoregressive, batch-1**, feeding a
>    range coder: token *t+1*'s network input is the *reconstructed* token *t*, and encoder
>    and decoder must emit bit-identical probability sequences. Positions inside a window
>    cannot be parallelized; only the 1024-token windows parallelize
>    (`codec.rs::par_windows`), and the coder then serializes. Throughput is capped by
>    single-token latency and weight bandwidth — not by core count.
>
> 3. **Starting point (measured).** ~700 tokens/s/core decode; ~30 s per 100 KB; ≥1 MB/s
>    single-thread from the per-token transformer is stated in the repo as not physically
>    attainable. The whitepaper's **per-token confidence-skip gate (`τ`) is specified but
>    NOT implemented** in Rust — `forward.rs::step()` always runs full GRU + transformer +
>    gate + full-vocab fusion. Per-*file* routing only exists (`main.rs::neural_limit`).
>
> 4. **Estimation machinery already present.** `main.rs::neural_estimate` and
>    `codec::window_bits_m` already sample windows to decide whether the neural pass can
>    beat CM. Use and extend this rather than inventing new estimation.
>
> 5. **Hot-path costs (candidate constant factors).** In `forward.rs::step` / `mm`: a
>    `format!("blk{bi}.qkv")` string allocation + hashmap lookup **per layer, per token**;
>    per-matmul output `Vec` allocation and repeated `qrow` activation quantization; and an
>    O(V) loop over V=8192 doing `ln` + `softmax` (±16k transcendentals) per token. The
>    int8×int8→i32 design uses **associative** integer accumulation, so SIMD/instruction
>    reordering is bit-exact — exploit this.
>
> ### Hard invariants (must not be violated)
>
> - **Lossless and bit-exact.** Every change must round-trip byte-for-byte; the encoder and
>   decoder must produce identical coder tables. No approximation that changes output.
> - **Determinism.** Output is a pure function of input bytes + options — independent of
>   `--jobs`, `--mem`, stdin vs file, and SIMD level (`README.md`, "Determinism").
> - **Format compatibility.** `NSC3` format 1 stays frozen; new behavior ships as new
>   segment kinds / decoder levels, never by silently changing the meaning of existing
>   bytes. Legacy `NSC2`/`NSCZ`/`NSCR` stay readable.
> - **Zero third-party crates** in the codec crate.
> - **Never worse, never corrupt.** A region that does not shrink or fails decode-back is
>   stored raw; corrupt input errors cleanly.
> - **Existing ratios preserved** on the categories that hit 6–10× today (see acceptance).
>
> ### Design questions to answer (with numbers, not adjectives)
>
> 1. **Routing.** Design **per-region / per-token** neural routing: when does the net run,
>    and how is the decision made *identically* on encoder and decoder (decoder sees only
>    decoded prefix)? Specify the decision signal, its bit-exact computability, and the
>    tie-breaking. Extend `neural_estimate`/`window_bits_m` into the codec's segment
>    planner.
> 2. **Skip gate.** Specify the Rust implementation of the whitepaper's `max(conf₂, conf_m) > τ`
>    skip, including how `τ` is recorded in the file so decode reproduces the encoder's path.
> 3. **Cheapening the residual.** Which of {distilled `nano` tier, GRU-only mode, int4/finer
>    weights, hierarchical/adaptive softmax, tabulated log-softmax, VNNI, buffer reuse,
>    weight-slice caching} preserve the residual's ratio contribution, and at what
>    measured cost? Which are free (bit-exact refactors) vs ratio-affecting (need a new
>    model version + level)?
> 4. **Breaking the AR dependency (only if 1–3 cannot reach the target).** Design a
>    *frozen, block-level featurizer* (embedding/attention over token history computed once
>    per block, independent of decoded symbols) feeding a cheap sequential residual — the
>    idea sketched in `HYBRID_MODEL_ANALYSIS.md`. Prove it is decodable (the featurizer must
>    not depend on the coder), state the ratio delta, and compare against GPU execution.
> 5. **Coder.** Whether rANS (batch-friendly) buys anything once model recurrence is
>    handled; justify per blueprint §7–8.
>
> ### Deliverables
>
> 1. A **decision**: minimal-change plan vs. partial re-architecture vs. full rewrite, with
>    the evidence that rules out the cheaper options.
> 2. A **staged roadmap**: Stage 0 (routing + skip + instrumentation), Stage 1 (hot-path
>    optimization), Stage 2 (residual cheapening), Stage 3 (conditional re-architecture) —
>    each stage independently shippable and ratio-guarded.
> 3. **Per-stage files/interfaces** to touch (`forward.rs`, `codec.rs`, `nsc3/block.rs`,
>    `nsc3/plan.rs`, `main.rs` routing, registry/level numbering), including any new
>    segment kind or decoder level.
> 4. A **measurement plan**: throughput-vs-ratio curve across `τ`/budget, skip rate,
>    per-token cost breakdown (matmul vs fusion vs softmax vs coder), peak RSS, and the
>    bit-exact round-trip + golden-file checks that gate each stage.
> 5. **Explicit non-goals and risks**, including where ratio and speed genuinely conflict
>    (e.g. a hard 10× on general text).
>
> ### Constraints on the answer
>
> - Prefer the smallest change that meets the acceptance bar.
> - Every performance claim must name how it will be measured.
> - Every ratio-regressing change must be versioned and reversible.
> - Do not replace the classical/format path — it is where the ratio and the speed already
>   coexist.

---

## Review Prompt

> You are reviewing an architecture for a high-speed evolution of the `nsc` neural path.
> Attack it on these axes and report concrete failures, not opinions:
>
> 1. **Bit-exactness.** For every proposed speedup, can the decoder reproduce the encoder's
>    probability sequence exactly? Flag any float reordering, approximation, or state that
>    depends on symbols the decoder does not yet have.
> 2. **Determinism.** Does the result depend on `--jobs`, `--mem`, thread scheduling, or SIMD
>    width? If any is `yes`, it is a defect.
> 3. **Ratio guard.** Which change can reduce ratio on the 6–10× categories, by how much, and
>    is it versioned as a new level/segment kind?
> 4. **Format.** Does any change alter the meaning of existing `NSC3` format-1 bytes or break
>    `--compat 1` / golden files?
> 5. **Claim discipline.** For each speed claim, is there a named measurement? Reject any
>    "should be faster."
> 6. **Missed cheaper path.** Is there a lower-effort option the design skipped (e.g. routing
>    alone) that meets the bar?

---

## Acceptance Criteria

A design is acceptable when it can be shown to satisfy:

- **Ratio preserved:** on the datatypes achieving 6–10× today (Text, Images, PDF/Office,
  and comparable categories in `bench/`), the shipped output grows by an agreed bound
  (state it; a sensible default is ≤ ~2–3% larger) or is unchanged.
- **Throughput:** the neural path no longer gates overall speed — target MB/s-class on the
  categories that already reach the ratio, with the neural residual running data-dependently
  and its **skip rate reported** as the throughput story.
- **Correctness:** byte-exact round-trip on the full `bench/corpus`; all golden files decode
  and re-encode identically; `cargo test --release` and `nsc selftest` pass.
- **Determinism:** identical output across `--jobs 1..N`, `--mem`, stdin vs file, and
  `NSC_SIMD=scalar` vs AVX2.
- **Compatibility:** `NSC3` format 1 unchanged; any new model/config ships as a new versioned
  level readable by the intended decoder level and refused cleanly by older ones.
- **Honesty:** a published throughput-vs-ratio curve over the speed dial (`τ`/budget), with
  the ratio/speed tradeoff stated, not a single headline number.

---

## Starting Facts (for the architect's reference)

| Fact | Value | Source |
|---|---|---|
| Text ratio / bpb | 6.2× / 1.292 | `nsc-codec/README.md` |
| Images ratio / bpb | 6.2× / 1.296 | `nsc-codec/README.md` |
| Silesia ratio / bpb | 5.9× / 1.353 | `nsc-codec/README.md` |
| Neural model trained bpc | ≈1.28 | `nsc models` manifest |
| Neural decode speed | ~700 tok/s/core, ~30 s/100 KB | `nsc-codec/README.md` |
| Window size (neural) | 1024 tokens, parallel windows | `codec.rs` |
| Vocab | 8192 (`base`), 4096 (`nano`) | `README.md` (root) |
| Model size on disk | ~273 MiB (mostly `order1` f32 lookup) | `nsc-codec/README.md` |
| Per-token skip gate | specified, **not implemented in Rust** | `WHITEPAPER.md` §8 vs `forward.rs` |
| Associative int8 accumulation | enables bit-exact SIMD/VNNI | `forward.rs::idot` |
