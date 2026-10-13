# Syntropy2: Hybrid Model — Full Analysis, Understanding Prompt & LLM Upgrade

---

## Complete Description of the Hybrid Model

Syntropy2 is a **neuro-symbolic hybrid lossless compressor**. The "hybridity" is not a blend or ensemble — it is a **residual coupling in log-space** where neural experts learn only what the classical models get wrong.

### Architecture Map

```
Input token x_t
     │
     ▼
┌─────────────────────────────────────────┐
│         CLASSICAL SIDE (prior)          │
│  p₀ (unigram) ─┐                       │
│  p₁ (bigram)  ─┼─→ p_cls ∈ Δᵛ         │
│  p₂ (order-2) ─┤   (gated mixture)     │
│  pₘ (LZ-match)─┘                       │
│                                         │
│  confidence state s ∈ ℝ⁴               │
│  [max p₁, H(p₁)/ln V, conf₂, confₘ]   │
└──────────────────┬──────────────────────┘
                   │ s (FiLM conditioning)
     ┌─────────────┴──────────────┐
     ▼                            ▼
┌──────────┐              ┌──────────────────┐
│ GRU      │              │ Transformer-XL   │
│ (local)  │              │ (long-range)     │
│  Δ_GRU   │              │  Δ_XF            │
└────┬─────┘              └──────┬───────────┘
     │                           │
     └────────┬──────────────────┘
              ▼
     ┌─────────────────────┐
     │ Context-Gated Attn  │ ← live features f ∈ ℝ⁶
     │ (attention routing) │
     │  w ∈ Δ⁴, g_gru, g_xf│
     └──────────┬──────────┘
                │
                ▼
     ┌────────────────────────────────┐
     │  FUSION (Bayesian update)      │
     │  Δ = g_gru·Δ_GRU + g_xf·Δ_XF │
     │  z = log(p_cls + ε) + Δ       │
     │  p_final = softmax(z) ∈ Δᵛ    │
     └────────────┬───────────────────┘
                  │
                  ▼
          Range/rANS entropy coder
```

---

## The 5 Core Principles

### 1. Log-space residual (not probability-space ensemble)

$$\mathbf{z}_{\text{final}} = \underbrace{\log(p_{\text{cls}} + \varepsilon)}_{\text{log-prior (classical)}} + \underbrace{\Delta}_{\text{log-likelihood correction (neural)}}$$

The neural network models only the *error* $\Delta = \log p_{\text{final}} - \log p_{\text{cls}}$. If $\Delta = 0$, the system degrades gracefully to pure classical. This is a Bayesian update: classical prior + neural likelihood ratio.

### 2. FiLM conditioning on classical confidence

$$\text{FiLM}(\mathbf{h}, \mathbf{s}) = (1 + \boldsymbol{\gamma}(\mathbf{s})) \odot \mathbf{h} + \boldsymbol{\beta}(\mathbf{s})$$

Neural experts scale/shift their hidden representations based on how uncertain the classical models are. When $\boldsymbol{\gamma}(\mathbf{s}) \approx -1$, neural channels suppress themselves — the classical model has this case covered.

### 3. Per-token attention gate

The gate is not a static MLP. It builds a query from neural hidden states and matches it against 6 expert keys (4 classical + 2 neural), each augmented by live confidence scalars:

$$\mathbf{w} = \text{softmax}([\text{score}_0, \text{score}_1, \text{score}_2, \text{score}_m]) \in \Delta^4$$
$$g_{\text{gru}}, g_{\text{xf}} = \sigma(\text{score}_e) \in (0,1)$$

### 4. Adaptive neural-skip

$$\text{skip} = \bigl(\max(\text{conf}_2, \text{conf}_m) > \tau\bigr)$$

When the classical model is sufficiently confident ($\tau = 0.95$ by default), the Transformer (and optionally GRU) are bypassed entirely. Throughput becomes *data-dependent*: repetitive regions run at classical speed ($>10$ MB/s), novel content at neural depth (~10–80 KB/s).

### 5. Two-stage curriculum

- **Stage 1**: Each expert trained independently (avoids cold-start residual collapse)
- **Stage 2**: Full end-to-end joint optimization of the fused loss only

---

## Conceptual + Mathematical Understanding Prompt

Use this prompt with any LLM to deeply understand Syntropy2 before building on it:

---

> **Prompt:**
>
> You are an expert in information theory, probabilistic modeling, and deep learning for compression. I will describe a hybrid compression architecture called **Syntropy2**. Help me understand it both conceptually and mathematically, layer by layer. After your full explanation, pose one targeted research question.
>
> **Architecture summary:**
> - Four classical experts (unigram $p_0$, bigram $p_1$, order-2 $p_2$, LZ-match $p_m$) are combined into a prior $p_{\text{cls}} = \sum_e w_e p_e \in \Delta^V$.
> - A 4-dimensional confidence state $\mathbf{s} = [\max p_1, H(p_1)/\ln V, \text{conf}_2, \text{conf}_m]$ is extracted.
> - Two neural experts (GRU and Transformer-XL) are FiLM-conditioned on $\mathbf{s}$ and emit residual logit vectors $\Delta_{\text{GRU}}, \Delta_{\text{XF}} \in \mathbb{R}^V$.
> - A context-gated attention mechanism computes per-token mixture weights $\mathbf{w}$ and neural gate scalars $g_{\text{gru}}, g_{\text{xf}}$.
> - Fusion: $\mathbf{z}_{\text{final}} = \log(p_{\text{cls}} + \varepsilon) + g_{\text{gru}} \Delta_{\text{GRU}} + g_{\text{xf}} \Delta_{\text{XF}}$; $p_{\text{final}} = \text{softmax}(\mathbf{z}_{\text{final}})$.
> - An adaptive skip gate bypasses the Transformer when $\max(\text{conf}_2, \text{conf}_m) > \tau$.
> - Training: Stage 1 warms up each expert independently; Stage 2 jointly optimizes $\mathcal{L} = \text{CE}(\mathbf{z}_{\text{final}}, x_{t+1})$.
>
> **Explain the following:**
> 1. Why does adding $\Delta$ to $\log p_{\text{cls}}$ (rather than averaging probabilities) produce a strictly better decomposition of the learning problem? Show this from the KL-divergence factorization and the bits-per-byte identity.
> 2. What is FiLM doing geometrically to the neural hidden space? When does $\boldsymbol{\gamma}(\mathbf{s}) \to -1$ help and when does it risk suppressing useful signal?
> 3. Why does the skip gate not degrade compression ratio even though it ignores the neural path? Show the gradient argument for why the gate trains to be nearly cost-free to apply.
> 4. What failure mode arises if Stage 2 joint training begins without Stage 1 warm-up (cold-start residual collapse)?
>
> After answering all four, pose **one precise, open research question** about the theoretical or empirical limits of this architecture that is not yet answered by the design itself.

---

## The Single Upgrade Question

> **How can Syntropy2's confidence-state vector $\mathbf{s} \in \mathbb{R}^4$ be replaced by a frozen intermediate-layer representation from a pretrained LLM — extracted at the point of maximum uncertainty — and used as a richer FiLM conditioning signal, and what training protocol, architectural interface, and information-theoretic constraints govern whether this substitution improves the bits-per-byte objective without requiring the LLM to run autoregressively per token?**

---

## Implementing the LLM Intermediate Layer

### Concept

An LLM like a small GPT-2 or Mistral already has internal attention patterns that signal *when it's uncertain*. Layer $l^*$ (typically the middle layers) contains the richest uncertainty geometry. Extract the hidden state $\mathbf{h}_{t}^{(l^*)} \in \mathbb{R}^{d_{\text{LLM}}}$ and project it down:

$$\mathbf{s}_{\text{LLM}} = W_{\text{proj}} \cdot \mathbf{h}_{t}^{(l^*)} + \mathbf{b} \in \mathbb{R}^{d_s}$$

Replace $\mathbf{s}$ in all FiLM calls and the gate feature vector $\mathbf{f}$ with $\mathbf{s}_{\text{LLM}}$.

### Architectural Changes

```python
class LLMConditioningAdapter(nn.Module):
    def __init__(self, llm_hidden_dim: int, film_dim: int, layer_idx: int):
        self.layer_idx = layer_idx
        self.proj = nn.Linear(llm_hidden_dim, film_dim)
        self.norm = nn.LayerNorm(film_dim)

    def forward(self, llm_hidden_states: list[Tensor]) -> Tensor:
        # Extract from chosen intermediate layer
        h = llm_hidden_states[self.layer_idx]   # (B, T, d_LLM)
        return self.norm(self.proj(h))           # (B, T, film_dim)
```

The FiLM modules receive `s_llm` instead of the 4-scalar `s`. Gate feature vector `f` extends from $\mathbb{R}^6$ to $\mathbb{R}^{4 + d_s}$ (classical scalars + projected LLM signal).

### Key Constraints

| Constraint | Reason |
|---|---|
| LLM must run **non-autoregressively** (or once per segment) | Per-token LLM forward pass would kill throughput — use segment-level extraction with memory |
| LLM weights must be **frozen** during Stage 1 | Avoids catastrophic interference with the residual learning objective |
| Only `layer_idx` and `proj` are trained end-to-end in Stage 2 | Keeps the LLM as a fixed feature extractor, not a generative model |
| `d_s` should match the current FiLM input dim (4) or be small | Scaling FiLM conditioning arbitrarily wide risks overfitting; 32–64 is practical |

### Training Protocol Delta

```
Stage 0 (new):  Probe LLM layers on enwik8 to find l* with
                highest MI to compression residual (classical bpb - oracle bpb).
                Freeze LLM entirely.

Stage 1:        Same as before, but FiLM receives s_LLM (proj only trains).

Stage 2:        Joint residual fine-tune. Optionally unfreeze top 2 LLM layers
                with lr × 0.01 (careful — risk of representation collapse).
```

### Expected Outcomes

- **If l* is well-chosen**: `s_LLM` captures syntactic/semantic uncertainty the 4-scalar `s` misses entirely (e.g., rare proper nouns, code identifiers) → measurable bpb improvement on novel-content regions.
- **Risk**: LLM inference latency per segment becomes the new bottleneck. Mitigate with a lightweight distilled probe network trained to approximate `s_LLM` at 10× speed.
- **Theoretical bound**: The LLM intermediate representation cannot reduce bpb below the true Shannon entropy of the data. The gain is bounded by $I(\mathbf{h}^{(l^*)}; x_{t+1} \mid p_{\text{cls}})$ — the mutual information between the LLM's hidden state and the next token, given what the classical model already knows.

### First Implementation Step

Run a **probing experiment**: train a linear classifier from each LLM layer to predict the classical model's residual error, and pick the layer with highest probing accuracy as $l^*$.
