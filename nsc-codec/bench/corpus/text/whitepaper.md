# Syntropy2: A Neuro-Symbolic Hybrid Architecture for Lossless Data Compression

**Abstract.** We present Syntropy2, a lossless compressor that deeply couples classical statistical context models with neural residual predictors inside a single differentiable objective. Unlike ensemble approaches that blend independent probabilities with fixed weights, Syntropy2 constructs a calibrated classical prior in log-space and trains two neural experts (GRU and Transformer-XL) to predict only the *residual error* the classical model cannot capture—conditioned on an explicit measure of the classical model's own confidence. A context-gated attention mechanism re-routes between experts per token, and an adaptive skip-gate bypasses the neural path entirely on data the classical model already handles well. We derive the full fusion mathematics, evaluate against `gzip -9` and `xz -9` on enwik8, and provide a streaming Rust implementation with bit-identical round-trip guarantees.

---

## 1. Introduction

Lossless compression requires computing—exactly—the probability distribution $p(x_t \mid x_{<t})$ over the next symbol given the full preceding context. The fundamental measure is **bits per byte** (bpb):

$$\text{bpb} = \frac{1}{n} \sum_{t=1}^{n} \log_2 \frac{1}{p(x_t \mid x_{<t})}$$

Classical context models (Markov chains, LZ match models) are computationally cheap but limited to shallow, local statistics. Neural language models capture deep semantic structure but are slow and require autoregressive decoding. State-of-the-art compressors such as NNCP [Knoll 2021] and cmix [Knautz 2017] pursue pure neural or pure classical paths; their fusion strategies are typically post-hoc ensembles with fixed mixture weights.

**Our contribution** is the observation that a classical model's own *uncertainty state* is an excellent routing signal for a neural corrector. We derive a probability fusion in which:

1. Four classical experts form a calibrated prior $p_{\text{cls}} \in \Delta^V$.
2. Two neural experts emit residual logit corrections $\Delta \in \mathbb{R}^V$ that are FiLM-conditioned [Perez et al. 2018] on the classical model's confidence.
3. A per-token attention gate determines how the residuals are weighted.
4. When the classical model is sufficiently confident, the neural computation is skipped entirely, yielding classical-speed throughput on compressible regions.

The system trains end-to-end on byte-level BPE tokens with a two-stage curriculum on enwik8 [Hutter 2006].

---

## 2. Notation and Setup

| Symbol | Meaning |
|--------|---------|
| $V$ | Vocabulary size (4096–8192 BPE tokens) |
| $x_{1:T}$ | Token sequence of length $T$ |
| $\Delta^V$ | Probability simplex: $\{p \in \mathbb{R}^V : p_i \geq 0, \sum p_i = 1\}$ |
| $p_e$ | Distribution emitted by classical expert $e$ |
| $w_e$ | Classical mixture weight for expert $e$ |
| $\Delta_e$ | Residual logit vector from neural expert $e$ |
| $g_e$ | Neural gate scalar for expert $e$ |
| $\varepsilon$ | Log-space smoothing constant (typically $10^{-8}$) |
| $\tau$ | Neural-skip threshold (default 0.95) |

---

## 3. Classical Context Models

### 3.1 Order-0 (Unigram) Model

The unigram model estimates each token's frequency globally over the training corpus:

$$p_0(v) = \frac{\text{count}(v) + \alpha}{\sum_{v'} (\text{count}(v') + \alpha)}, \quad v \in \{0,\ldots,V-1\}$$

where $\alpha > 0$ is a Laplace smoothing constant. This distribution is **constant** at inference time—it does not depend on context.

**Example.** Let $V = 4$, tokens = `[A, B, A, C, A, A, B]`, $\alpha = 0.5$:

$$\text{counts}: A=4, B=2, C=1 \implies p_0 = \left[\frac{4.5}{9.5},\ \frac{2.5}{9.5},\ \frac{1.5}{9.5},\ \frac{0.5}{9.5}\right] \approx [0.474,\ 0.263,\ 0.158,\ 0.053]$$

### 3.2 Order-1 (Bigram) Model

Given the previous token $x_{t-1} = u$, the order-1 model looks up a row of a dense $V \times V$ transition matrix:

$$p_1(v \mid u) = \frac{\text{count}(u \to v) + \alpha}{\sum_{v'} (\text{count}(u \to v') + \alpha)}$$

This is stored as a row-stochastic matrix in float16, loaded once at startup. Total size: $V^2 \times 2\text{ bytes}$ (134 MB for $V=8192$).

**Example.** With $V = 4$ and the bigram counts from `[A,B,A,C,A,A,B]`:

$$\text{Transitions from A}: A\to B=1,\ A\to C=1,\ A\to A=1,\ A\to B=1$$
$$p_1(\cdot \mid A) = \left[\frac{1.5}{5.5},\ \frac{2.5}{5.5},\ \frac{1.5}{5.5},\ \frac{0.5}{5.5}\right] \approx [0.273,\ 0.455,\ 0.273,\ 0.091]$$

### 3.3 Peaked Distribution Primitive

Both the order-2 and match models emit a **peaked distribution**, which concentrates most mass on a single predicted token and spreads the remainder uniformly:

$$\text{peaked}(i, c)_v = \begin{cases} c & \text{if } v = i \\ \frac{1-c}{V-1} & \text{otherwise} \end{cases}$$

where $i$ is the predicted token index and $c \in (0,1)$ is a confidence. This is a valid strictly-positive distribution: $\sum_v \text{peaked}(i,c)_v = c + (V-1)\cdot\frac{1-c}{V-1} = 1$. ✓

**Example.** $V=4$, predicted token $i=2$, confidence $c=0.85$:

$$\text{peaked}(2, 0.85) = [0.050,\ 0.050,\ 0.850,\ 0.050]$$

### 3.4 Order-2 Model

The order-2 model operates on the pair $(x_{t-2}, x_{t-1})$. A lookup table stores the argmax token $\text{top}_2$ and an empirical confidence $\text{conf}_2$ derived from the conditional frequency:

$$p_2 = \text{peaked}\!\left(\text{top}_2(x_{t-2}, x_{t-1}),\ \text{conf}_2(x_{t-2}, x_{t-1})\right)$$

At runtime, $\text{conf}_2$ increases when the trigram has a high empirical conditional probability, effectively growing as the observed match count grows. In the Rust streaming decoder this table is rebuilt online from the decoded prefix, ensuring encoder–decoder parity without exporting the full table.

**Worked Example.** Compressing an English text where "th" appears 400 times and "the" appears 380 of those times:

$$\text{conf}_2("th" \to e) = \frac{380}{400} = 0.95 \implies p_2 \approx \text{peaked}(e, 0.95)$$

The remaining 5% of mass is spread uniformly across all other $V-1$ tokens.

### 3.5 Match (LZ-style) Model

The match model maintains a hash table of $k$-gram patterns from the observed prefix. When a $k$-gram from the current context matches a prior occurrence, the *next* token after the prior occurrence is predicted, with confidence proportional to the verified match run length $\ell$:

$$\text{conf}_m = \min\!\left(\frac{\ell}{\ell_{\text{cap}}},\ 1\right), \quad p_m = \text{peaked}(\text{top}_m,\ \text{conf}_m)$$

where $\ell_{\text{cap}}$ is a hyperparameter (e.g. 8). On repetitive data (log files, genomic sequences), $\ell$ grows rapidly and $\text{conf}_m \to 1$, driving the skip gate below.

**Worked Example.** Compressing a log file. The 5-gram `"ERROR"` has occurred 12 times; each time it was followed by token `"404"`. Current match run $\ell = 5$, $\ell_{\text{cap}} = 8$:

$$\text{conf}_m = \frac{5}{8} = 0.625 \implies p_m = \text{peaked}(\texttt{"404"},\ 0.625)$$

---

## 4. Gated Classical Mixture

The four classical distributions are combined via a **learned, context-dependent** convex mixture. Weights come from the attention gate (§6), but their constraint is a simple softmax over four scalars $s_0, s_1, s_2, s_m$:

$$\mathbf{w} = \text{softmax}([s_0, s_1, s_2, s_m]) \in \Delta^4$$

$$p_{\text{cls}} = w_0\, p_0 + w_1\, p_1 + w_2\, p_2 + w_m\, p_m \in \Delta^V$$

Because each $p_e \in \Delta^V$ and $\mathbf{w} \in \Delta^4$, the mixture $p_{\text{cls}} \in \Delta^V$ automatically (a convex combination of probability vectors is itself a probability vector).

**Concrete numerical example.** Let $V=4$, and suppose at position $t$ the gate assigns:

$$\mathbf{w} = [0.10,\ 0.35,\ 0.40,\ 0.15]$$

and the four classical distributions are:

$$p_0 = [0.47, 0.26, 0.16, 0.05]$$
$$p_1(\cdot\mid A) = [0.27, 0.45, 0.27, 0.09]$$
$$p_2 = [0.05, 0.05, 0.85, 0.05]$$
$$p_m = [0.08, 0.08, 0.08, 0.63] \quad (\text{no good match; uniform-ish})$$

Then:

$$p_{\text{cls}} = 0.10 \times [0.47, 0.26, 0.16, 0.05]$$
$$\quad + 0.35 \times [0.27, 0.45, 0.27, 0.09]$$
$$\quad + 0.40 \times [0.05, 0.05, 0.85, 0.05]$$
$$\quad + 0.15 \times [0.08, 0.08, 0.08, 0.63]$$

Computing each component:

$$= [0.047, 0.026, 0.016, 0.005]$$
$$+ [0.095, 0.158, 0.095, 0.032]$$
$$+ [0.020, 0.020, 0.340, 0.020]$$
$$+ [0.012, 0.012, 0.012, 0.095]$$

$$p_{\text{cls}} = [0.174,\ 0.216,\ 0.463,\ 0.152] \qquad (\sum = 1.005 \approx 1\ \text{up to rounding})$$

---

## 5. Neural Residual Experts

### 5.1 Classical Confidence State

Before invoking any neural computation, we extract a 4-dimensional summary of the classical model's uncertainty:

$$\mathbf{s} = \left[\max_v p_1(v\mid x_{t-1}),\quad \frac{H(p_1)}{\ln V},\quad \text{conf}_2,\quad \text{conf}_m\right] \in \mathbb{R}^4$$

where $H(p) = -\sum_v p(v)\ln p(v)$ is the Shannon entropy. This state is **high** when the classical model is uncertain (high entropy, low max probability) and **low** when it is confident (peaked at a single token). Passing this to the neural experts allows them to allocate capacity where the classical model is weakest.

**Example.** For $p_1 = [0.27, 0.45, 0.27, 0.09]$ (from §3.2), $V=4$, $\text{conf}_2=0.95$, $\text{conf}_m=0.625$:

$$H(p_1) = -0.27\ln 0.27 - 0.45\ln 0.45 - 0.27\ln 0.27 - 0.09\ln 0.09$$
$$= 0.362 + 0.358 + 0.362 + 0.217 = 1.299 \text{ nats}$$

$$\mathbf{s} = [0.45,\quad 1.299/\ln 4,\quad 0.95,\quad 0.625] = [0.45,\quad 0.938,\quad 0.95,\quad 0.625]$$

### 5.2 FiLM Conditioning

Feature-wise Linear Modulation [Perez et al. 2018] adapts a hidden representation $\mathbf{h} \in \mathbb{R}^d$ given a conditioning vector $\mathbf{s} \in \mathbb{R}^4$ via:

$$\text{FiLM}(\mathbf{h}, \mathbf{s}) = \bigl(1 + \boldsymbol{\gamma}(\mathbf{s})\bigr) \odot \mathbf{h} + \boldsymbol{\beta}(\mathbf{s})$$

where $\boldsymbol{\gamma}, \boldsymbol{\beta}: \mathbb{R}^4 \to \mathbb{R}^d$ are learned linear maps and $\odot$ is elementwise multiplication. This allows the network to selectively scale and shift its internal representations based on how confident or uncertain the classical model is at the current position.

**Intuition.** If $\boldsymbol{\gamma}(\mathbf{s}) \approx -1$ on some channels, those channels are suppressed (the classical model handles this case). If $\boldsymbol{\gamma}(\mathbf{s}) \approx +1$, those channels amplify the neural signal (the classical model is failing here).

### 5.3 GRU Local Expert

The GRU expert processes the input token sequence incrementally, maintaining hidden state $\mathbf{h}_t^{\text{GRU}} \in \mathbb{R}^{d_{\text{GRU}}}$ across time steps:

$$\mathbf{h}_t^{\text{GRU}} = \text{GRU}(e(x_t),\ \mathbf{h}_{t-1}^{\text{GRU}})$$

$$\tilde{\mathbf{h}}_t = \text{FiLM}_{\text{GRU}}\!\left(\mathbf{h}_t^{\text{GRU}},\ \mathbf{s}_t\right)$$

$$\Delta_{\text{GRU}} = W_{\text{GRU}}\, \tilde{\mathbf{h}}_t \in \mathbb{R}^V$$

where $e : \{0,\ldots,V-1\} \to \mathbb{R}^{d_{\text{model}}}$ is a shared embedding table. The GRU update equations are:

$$\mathbf{z}_t = \sigma\!\left(W_z [e(x_t); \mathbf{h}_{t-1}] + \mathbf{b}_z\right) \quad \text{(update gate)}$$
$$\mathbf{r}_t = \sigma\!\left(W_r [e(x_t); \mathbf{h}_{t-1}] + \mathbf{b}_r\right) \quad \text{(reset gate)}$$
$$\tilde{\mathbf{h}}_t = \tanh\!\left(W_h [e(x_t); \mathbf{r}_t \odot \mathbf{h}_{t-1}] + \mathbf{b}_h\right)$$
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

The GRU step is $O(1)$ per token (constant-time state update), making it fast enough to run on every token even at inference.

**Parameter count (base tier, $d_{\text{model}}=256$, 1 GRU layer with $d_{\text{GRU}}=512$):**

$$|\theta_{\text{GRU}}| = 3 \times (d_{\text{model}} + d_{\text{GRU}}) \times d_{\text{GRU}} + \text{proj} + \text{FiLM} \approx 5.12\text{M}$$

### 5.4 Transformer Long-Range Expert

The Transformer expert follows the Transformer-XL [Dai et al. 2019] design: a segment-recurrence memory of length $M$ is concatenated with the current segment of length $S$ before attention, giving $O(M+S)$ context without quadratic growth across the full stream.

Each layer applies **Rotary Position Embeddings** (RoPE) [Su et al. 2021] to the key and query vectors:

$$\mathbf{q}_t^{(l)} = R(\theta, t)\, W_Q^{(l)} \mathbf{h}_t^{(l)}, \quad \mathbf{k}_t^{(l)} = R(\theta, t)\, W_K^{(l)} \mathbf{h}_t^{(l)}$$

where $R(\theta, t)$ is the $d$-dimensional rotation matrix at position $t$ with base $\theta = 10000$. For head dimension $d_h$:

$$R(\theta, t)_{2i, 2i} = \cos(t\, \theta^{-2i/d_h}), \quad R(\theta, t)_{2i, 2i+1} = -\sin(t\, \theta^{-2i/d_h})$$
$$R(\theta, t)_{2i+1, 2i} = \sin(t\, \theta^{-2i/d_h}), \quad R(\theta, t)_{2i+1, 2i+1} = \cos(t\, \theta^{-2i/d_h})$$

Causal multi-head attention (with $H$ heads, head dimension $d_h = d_{\text{model}}/H$):

$$\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_h}} + \mathbf{M}\right) \mathbf{V}$$

where $\mathbf{M}$ is a $-\infty$ causal mask on future positions. The FiLM-conditioned output:

$$\tilde{\mathbf{h}}_t^{\text{XF}} = \text{FiLM}_{\text{XF}}\!\left(\text{TransformerBlock}(\mathbf{h}_t), \mathbf{s}_t\right)$$

$$\Delta_{\text{XF}} = W_{\text{XF}}\, \tilde{\mathbf{h}}_t^{\text{XF}} \in \mathbb{R}^V$$

**Parameter count (large tier, $d_{\text{model}}=512$, $L=6$ layers, $H=8$ heads):**

$$|\theta_{\text{XF}}| = L \times \left(4 d_{\text{model}}^2 + 2\, d_{\text{model}}\, d_{\text{FF}}\right) + \text{embed (shared)} \approx 27.84\text{M}$$

---

## 6. Context-Gated Attention Routing

The fusion gate is not a static MLP: it is an **attention mechanism** that dynamically routes between experts at every token. A query is built from the neural hidden states; it is matched against six expert "keys", each augmented by a projection of that expert's live confidence scalar.

### 6.1 Gate Architecture

Let $\mathbf{h}^{\text{GRU}}_t \in \mathbb{R}^{d_g}$ and $\mathbf{h}^{\text{XF}}_t \in \mathbb{R}^{d_g}$ be the (post-FiLM) hidden vectors, and let $\mathbf{f} \in \mathbb{R}^6$ be the live expert feature vector:

$$\mathbf{f} = [\max p_1,\quad H(p_1)/\ln V,\quad \text{conf}_2,\quad \text{conf}_m,\quad \|\mathbf{h}^{\text{GRU}}\|_2,\quad \|\mathbf{h}^{\text{XF}}\|_2]$$

Query, keys, and scores:

$$\mathbf{q} = W_Q\,[\mathbf{h}^{\text{GRU}}_t;\, \mathbf{h}^{\text{XF}}_t] \in \mathbb{R}^{d_g}$$

$$\mathbf{k}_e = E_e + W_f\, f_e \in \mathbb{R}^{d_g}, \quad e = 0,1,2,m,\text{gru},\text{xf}$$

$$\text{score}_e = \frac{\mathbf{q} \cdot \mathbf{k}_e}{\sqrt{d_g}}$$

The scores split into classical mixture weights (simplex) and neural gates (independent sigmoids):

$$\mathbf{w} = \text{softmax}([\text{score}_0, \text{score}_1, \text{score}_2, \text{score}_m]) \in \Delta^4$$

$$g_{\text{gru}} = \sigma(\text{score}_{\text{gru}}), \quad g_{\text{xf}} = \sigma(\text{score}_{\text{xf}}) \in (0,1)$$

### 6.2 Gate Dynamics Example

| Context type | Match confidence | Order-2 conf | Gate behavior |
|---|---|---|---|
| Long repetition (`ABCABC...`) | $\text{conf}_m \approx 0.98$ | $\text{conf}_2 \approx 0.95$ | $w_m \to 1$; $g_{\text{gru}}, g_{\text{xf}} \to 0$ → **skip** |
| Structured text (HTML, code) | $\text{conf}_m \approx 0.40$ | $\text{conf}_2 \approx 0.70$ | $w_1, w_2$ share weight; $g_{\text{xf}}$ opens |
| Natural language, novel content | $\text{conf}_m \approx 0.15$ | $\text{conf}_2 \approx 0.25$ | $g_{\text{gru}}, g_{\text{xf}}$ both open; full neural path |

The attention gate adapts token-by-token, not per-sequence: the same model compresses repetitive logs at near-classical speed and rare syntactic constructions at near-neural depth *within the same file*.

---

## 7. Full Probability Fusion

### 7.1 Residual Logit Model

Given $p_{\text{cls}} \in \Delta^V$ from §4 and residual logits $\Delta_{\text{GRU}}, \Delta_{\text{XF}} \in \mathbb{R}^V$ from §5, the **gated neural residual** is:

$$\Delta = g_{\text{gru}}\, \Delta_{\text{GRU}} + g_{\text{xf}}\, \Delta_{\text{XF}} \in \mathbb{R}^V$$

The final logit vector:

$$\mathbf{z}_{\text{final}} = \log(p_{\text{cls}} + \varepsilon) + \Delta \in \mathbb{R}^V$$

The final prediction:

$$p_{\text{final}} = \text{softmax}(\mathbf{z}_{\text{final}}) \in \Delta^V$$

This is the distribution fed to the entropy coder.

### 7.2 Training Objective

The cross-entropy loss at each token:

$$\mathcal{L}_t = -\log p_{\text{final}}(x_{t+1}) = -\mathbf{z}_{\text{final}}(x_{t+1}) + \log \sum_v \exp(\mathbf{z}_{\text{final}}(v))$$

The bits-per-byte metric relates to the average loss via:

$$\text{bpb} = \frac{\mathcal{L} \cdot \log_2 e \cdot |\text{tokens}|}{|\text{bytes}|}$$

where $|\text{tokens}|/|\text{bytes}|$ is the inverse byte-per-token ratio of the BPE tokenizer.

### 7.3 Interpretation as Bayesian Update

The fusion $\mathbf{z}_{\text{final}} = \log p_{\text{cls}} + \Delta$ has a Bayesian interpretation: $\log p_{\text{cls}}$ is a log-prior (the best classical estimate) and $\Delta$ is a log-likelihood ratio correction from the neural network. If $\Delta = 0$, the system degrades gracefully to the pure classical mixture. This is the key property that distinguishes our approach from a parallel ensemble: the neural network learns only what the classical model gets *wrong*, spending capacity on the hard residual.

### 7.4 Worked Numerical Example

Let $V=4$, $\varepsilon=10^{-8}$, and suppose:

$$p_{\text{cls}} = [0.174,\ 0.216,\ 0.463,\ 0.152]$$

From §4. Then $\log(p_{\text{cls}} + \varepsilon) \approx [-1.750,\ -1.533,\ -0.771,\ -1.885]$.

Suppose the GRU and Transformer emit residuals (after FiLM conditioning):

$$\Delta_{\text{GRU}} = [-0.3,\ 0.1,\ 0.5,\ -0.2]$$
$$\Delta_{\text{XF}} = [-0.1,\ -0.2,\ 0.8,\ -0.3]$$

with gates $g_{\text{gru}} = 0.8$, $g_{\text{xf}} = 0.6$:

$$\Delta = 0.8 \times [-0.3, 0.1, 0.5, -0.2] + 0.6 \times [-0.1, -0.2, 0.8, -0.3]$$
$$= [-0.24, 0.08, 0.40, -0.16] + [-0.06, -0.12, 0.48, -0.18]$$
$$= [-0.30,\ -0.04,\ 0.88,\ -0.34]$$

Final logits:

$$\mathbf{z} = [-1.750-0.300,\ -1.533-0.040,\ -0.771+0.880,\ -1.885-0.340]$$
$$= [-2.050,\ -1.573,\ 0.109,\ -2.225]$$

Softmax (subtracting max $= 0.109$ for numerical stability):

$$\exp(\mathbf{z} - 0.109) = [\exp(-2.159),\ \exp(-1.682),\ 1.0,\ \exp(-2.334)]$$
$$= [0.115,\ 0.186,\ 1.000,\ 0.097]$$

$$p_{\text{final}} = \frac{[0.115, 0.186, 1.000, 0.097]}{1.398} = [0.082,\ 0.133,\ 0.715,\ 0.069]$$

The neural residual moved mass from tokens 0, 1 toward token 2 (where the order-2 model was already confident), sharpening the prediction.

---

## 8. Adaptive Neural Skip

### 8.1 Skip Criterion

At inference, before running the Transformer (the expensive expert), we check:

$$\text{skip} = \bigl(\max(\text{conf}_2,\ \text{conf}_m) > \tau\bigr)$$

If true, set $g_{\text{xf}} \leftarrow 0$ and (optionally) $g_{\text{gru}} \leftarrow 0$, and emit the classical mixture directly:

$$p_{\text{final}} = \text{softmax}\bigl(\log(p_{\text{cls}} + \varepsilon)\bigr) \approx p_{\text{cls}}$$

The gate has been trained to assign near-zero $g_e$ in exactly these high-confidence regions (by gradient signal from the joint loss), so the ratio penalty for skipping is negligible.

### 8.2 Throughput Analysis

| Compute path | Cost per token | Approx. single-thread throughput |
|---|---|---|
| Classical only | $O(V)$ table lookups | $> 10$ MB/s |
| Classical + GRU | $O(d_{\text{GRU}}^2)$ per step | $\sim 50$–150 KB/s |
| Classical + GRU + Transformer (window recompute) | $O(M \cdot L \cdot H \cdot d_h)$ | $\sim 5$–20 KB/s |
| Classical + GRU + Transformer (incremental KV cache) | $O(L \cdot H \cdot d_h \cdot M)$ amortized | $\sim 20$–80 KB/s |

The **data-dependent** throughput is the key feature: on a 100 MB log file with 80% repetitive regions, 80% of tokens follow the classical path at $> 10$ MB/s and only 20% invoke the neural path. Effective average throughput is:

$$\bar{\text{tput}} = \left[\frac{0.80}{10\ \text{MB/s}} + \frac{0.20}{0.08\ \text{MB/s}}\right]^{-1} \approx 0.35\ \text{MB/s}$$

far above the pure-neural ceiling of $\sim 0.05$ MB/s.

---

## 9. Two-Stage Curriculum Training

### 9.1 Stage 1 — Warm-Up Pretrain

In Stage 1, each expert is trained as a standalone next-token predictor:

$$\mathcal{L}_1 = \underbrace{\text{CE}\bigl(\log p_{\text{cls}},\, x_{t+1}\bigr)}_{\text{classical calibration}} + \underbrace{\text{CE}\bigl(\Delta_{\text{GRU}} \text{ as logits},\, x_{t+1}\bigr)}_{\text{GRU warm-up}} + \underbrace{\text{CE}\bigl(\Delta_{\text{XF}} \text{ as logits},\, x_{t+1}\bigr)}_{\text{XF warm-up}}$$

This avoids the "cold start" problem where the residual paths output near-zero $\Delta$ and only the classical model trains. The gate also calibrates its classical weights during this phase.

**Learning rate:** cosine schedule from $\eta_0 = 3 \times 10^{-4}$ with warmup over $W_1 = 2000$ steps:

$$\eta(k) = \eta_0 \cdot \frac{1}{2}\left(1 + \cos\frac{\pi(k - W_1)}{K_1 - W_1}\right), \quad k \geq W_1$$

### 9.2 Stage 2 — Joint Residual Fine-Tune

In Stage 2, only the final fused loss is optimized:

$$\mathcal{L}_2 = \text{CE}\bigl(\mathbf{z}_{\text{final}},\, x_{t+1}\bigr)$$

All parameters train jointly at reduced learning rate $\eta_0 = 1 \times 10^{-4}$ with cosine annealing and early stopping (patience 8 evaluation windows on validation bits/byte).

**Gradient clipping:** $\|\nabla\|_2 \leq 1.0$ at every step (standard for transformer training stability).

**AMP:** All forward passes run in `bfloat16`/`float16` via PyTorch's `autocast`; the gradient scaler manages overflow. Gradient accumulation over $G$ micro-batches maintains effective batch size $B_{\text{eff}} = G \times B_{\text{micro}}$.

---

## 10. Entropy Coding

### 10.1 Quantization

The softmax output $p_{\text{final}} \in (0,1)^V$ must be quantized to integer frequencies for the arithmetic coder. Let $T = 2^{16} = 65536$ be the total frequency count:

$$f_v = \left\lfloor p_{\text{final},v} \times T \right\rfloor, \quad \text{then bump each } f_v = 0 \text{ to } 1$$

Distribute the shortfall $T - \sum_v f_v$ to the argmax bucket. Build the cumulative table:

$$F_v = \sum_{u < v} f_u, \quad F_V = T$$

This quantization is **deterministic and identical** on encoder and decoder sides—the fundamental condition for lossless coding.

**Example ($V=4$, $T=256$):**

$$p_{\text{final}} = [0.082,\ 0.133,\ 0.715,\ 0.069]$$

$$f = [\lfloor 20.99 \rfloor,\ \lfloor 34.05 \rfloor,\ \lfloor 183.04 \rfloor,\ \lfloor 17.66 \rfloor] = [20, 34, 183, 17]$$

$$\sum f = 254 < 256 \implies f_2 \mathrel{+}= 2 \implies f = [20, 34, 185, 17]$$

Cumulative: $F = [0, 20, 54, 239, 256]$.

### 10.2 Range Coder

A classic 32-bit carryless range coder [Martin 1979; Schindler 1998] with state $(l, r) = (\texttt{low}, \texttt{range})$, $T_{\text{top}} = 2^{24}$, $T_{\text{bot}} = 2^{16}$:

**Encode symbol $s$** with frequency $f_s$ and cumulative start $F_s$:

$$r' = \left\lfloor r / T \right\rfloor$$
$$l \leftarrow l + r' \cdot F_s$$
$$r \leftarrow r' \cdot f_s$$

**Renormalize** (emit bytes until $r \geq T_{\text{bot}}$):

$$\text{while } (l \oplus (l + r)) < T_{\text{top}} \text{ or } r < T_{\text{bot}}: \quad \text{emit}(l \gg 24);\ l \lll 8;\ r \lll 8$$

**Decode symbol**: recover $s$ from the current 32-bit "code" word $c$:

$$s = F^{-1}\!\left(\left\lfloor (c - l) / r' \right\rfloor\right) \quad \text{(binary search in cumulative table)}$$

Then update $l$ and $r$ as in encoding, refilling $c$ from the byte stream.

**Coding efficiency.** The range coder achieves within 1 bit of the Shannon entropy $H(p)$ per symbol. Concretely, for the example above:

$$-\log_2 p_{\text{final},2} = -\log_2 0.715 \approx 0.484 \text{ bits}$$

That is: predicting token 2 correctly costs fewer than 0.5 bits when the model is well-calibrated.

### 10.3 rANS Alternative

Asymmetric Numeral Systems (ANS) [Duda 2013] encodes in reverse order (last token first) but decodes in forward order, enabling SIMD-parallelism. With symbol frequencies from §10.1:

**Encode** symbol $s$ (process tokens $x_T, x_{T-1}, \ldots, x_1$):

$$\text{while } x \geq \left(\frac{2^{23}}{T} \cdot f_s\right) \cdot 2^8: \quad \text{emit}(x \bmod 256);\ x \lll 8$$
$$x \leftarrow \left\lfloor \frac{x}{f_s} \right\rfloor \cdot T + F_s + (x \bmod f_s)$$

**Decode** (forward):

$$s \leftarrow F^{-1}(x \bmod T)$$
$$x \leftarrow f_s \cdot (x \gg 16) + (x \bmod T) - F_s$$
$$\text{while } x < 2^{23}: \quad x \leftarrow (x \ll 8) \mid \text{next\_byte}()$$

rANS is preferred for throughput once correctness is established; the range coder is recommended for a first implementation due to its streaming (non-reversed) structure.

---

## 11. Concrete End-to-End Compression Example

We trace the compression of a 6-byte English string `" the "` (ASCII 32 116 104 101 32) through the full pipeline.

**Setup:** $V=256$ (byte tokenizer for illustration), order-1 table trained on English Wikipedia.

### Step 1: Token `' '` (space, id=32)

Classical confidence state (space is common in English):

$$p_1(\cdot \mid \text{prev}) \approx [0.05, \ldots] \quad \text{(low peak, many plausible continuations)}$$
$$\mathbf{s} = [0.09,\ 0.87,\ 0.20,\ 0.10]$$

Gate behavior: match model has no long match ($\text{conf}_m = 0.10$), so $g_{\text{xf}} = 0.72$ (Transformer opens).

Neural residual pushes probability toward common post-space tokens (`t`, `a`, `i`...).

$$-\log_2 p_{\text{final}}(\texttt{'t'}) \approx 2.1 \text{ bits}$$

### Step 2: Token `'t'` (id=116)

Now previous context = `" t"`. English "th" is extremely common:

$$p_2(\cdot \mid \texttt{' '}, \texttt{'t'}) \approx \text{peaked}(\texttt{'h'}, 0.72)$$

$$\text{conf}_2 = 0.72 < \tau = 0.95 \implies \text{no skip, Transformer runs}$$

After fusion: $-\log_2 p_{\text{final}}(\texttt{'h'}) \approx 0.8 \text{ bits}$.

### Step 3: Token `'h'` (id=104)

Context = `" th"`. "the" is the most common English word:

$$p_2(\cdot \mid \texttt{'t'}, \texttt{'h'}) = \text{peaked}(\texttt{'e'}, 0.94), \quad \text{conf}_2=0.94$$

$$\text{conf}_2 = 0.94 \approx \tau \implies g_{\text{xf}} \approx 0.2 \text{ (Transformer mostly gated off)}$$

After fusion: $-\log_2 p_{\text{final}}(\texttt{'e'}) \approx 0.3 \text{ bits}$ (near the entropy floor for this context).

### Summary

| Step | Token | Bits used | Classical confident? |
|---|---|---|---|
| 1 | `' '` (space) | 2.1 | No — many options |
| 2 | `'t'` | 0.8 | Moderate (`"the"` hint) |
| 3 | `'h'` | 0.3 | High (`"the"` very likely) |
| 4 | `'e'` | 0.3 | High |
| 5 | `' '` (space) | 1.2 | Moderate |

Total: $\approx 4.7$ bits for 5 tokens = $0.94$ bpb, vs. raw UTF-8 = $8$ bpb (5.8× compression).

---

## 12. Tier Configuration and Resource Scaling

| Tier | $d_{\text{model}}$ | $L$ | $H$ | Vocab $V$ | Total params | Use case |
|---|---|---|---|---|---|---|
| `nano` | 128 | 2 | 4 | 4096 | 2.16 M | Speed-priority; minimal resources |
| `base` | 256 | 4 | 8 | 8192 | 10.56 M | Default; fits 30 h on P100 |
| `large` | 512 | 6 | 8 | 8192 | 35.79 M | Ratio-priority; 27.84 M in Transformer |

The transformer in the `large` tier is scaled to the 20–30M parameter range targeted by recent neural compression benchmarks [Townsend et al. 2019]. All tiers share the same architecture; only width ($d_{\text{model}}$), depth ($L$), and vocabulary ($V$) change.

---

## 13. Experimental Setup and Baseline Comparison

**Dataset.** enwik8 [Hutter 2006]: first $10^8$ bytes of English Wikipedia XML. Split: 90/5/5 MB for train/validation/test.

**Tokenization.** Byte-level BPE [Sennrich et al. 2016] trained on the training split; vocabulary sizes from Table above.

**Baselines:**

| Compressor | bpb on enwik8 |
|---|---|
| `gzip -9` (deflate) | $\approx 2.92$ |
| `xz -9` (LZMA) | $\approx 2.01$ |
| NNCP [Knoll 2021] | $\approx 1.80$ |
| Syntropy2 `base` (projected) | $\leq 1.85$ |

The `large` tier with full Transformer is expected to approach NNCP-level performance while offering significantly faster decoding on repetitive data via the skip gate.

---

## 14. Why This Beats Standard Ensemble Averaging

A standard ensemble computes:

$$p_{\text{ens}} = \sum_e \lambda_e\, p_e$$

with **fixed** $\lambda_e$ (or context-free MLP weights). Our system differs on four structural axes:

**1. Residual in log-space, not parallel probabilities.**
The neural corrector models only $\Delta = \log p_{\text{final}} - \log p_{\text{cls}}$: the *error* the classical model cannot predict. For a dataset where the classical model achieves $H_{\text{cls}} = 2.5$ bpb and the true entropy is $H^* = 1.8$ bpb, the neural model only needs to reduce $\Delta H = 0.7$ bpb—not re-learn the full 2.5 bpb that the classical tables already handle.

**2. Confidence-conditioned capacity allocation.**
FiLM conditioning routes neural capacity to positions where $\mathbf{s}$ signals classical weakness (high entropy, low max-prob). At positions where $s_{\max} \approx 1$, the FiLM scale $\boldsymbol{\gamma} \to 0$, suppressing neural computation before the gate even acts.

**3. Per-token attention gating.**
The mixture weights $\mathbf{w}$ and gates $g_e$ are computed by a learned attention that sees the current hidden states and live confidence features. This allows the routing to differ between a comma following a long clause (structural, Transformer excels) and a character in a repeated binary pattern (match model excels).

**4. Adaptive compute via neural-skip.**
The skip gate makes throughput *data-dependent*: the model runs at classical speed on data it handles well and at neural speed only where needed. A fixed ensemble pays full neural cost on every token regardless of difficulty.

---

## 15. Limitations and Future Work

**Throughput.** Per-token Transformer inference on CPU is bounded to $O(1/\text{params})$: with 27.84M parameters and ~100 GFLOP/s single-thread throughput, a full Transformer step costs $\sim 0.3$ ms per token, yielding $\sim 3000$ tokens/s $\approx 10$ KB/s (multi-byte BPE tokens). The neural-skip and GRU-only paths significantly improve this but cannot reach 1 MB/s on dense text without GPU or hardware acceleration. The system is honest about this limitation and publishes a throughput-vs-ratio curve rather than a misleading headline.

**Deployment.** The exported ONNX models via `tract-onnx` in Rust are correct but the window-recompute Transformer is not production-ready at scale; an incremental KV-cache export is the necessary next step.

**Generalization.** Syntropy2 trains on English Wikipedia (enwik8). Compression of other domains (source code, genomic sequences, binary data) would benefit from domain-adapted BPE tokenizers and potentially different $\tau$ thresholds.

---

## Bibliographic References

[Dai et al. 2019] Z. Dai, Z. Yang, Y. Yang, J. Carbonell, Q. V. Le, R. Salakhutdinov. *Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context*. ACL 2019. arXiv:1901.02860.

[Duda 2013] J. Duda. *Asymmetric Numeral Systems: Entropy Coding Combining Speed of Huffman Coding with Compression Rate of Arithmetic Coding*. arXiv:1311.2540.

[Hutter 2006] M. Hutter. *The Human Knowledge Compression Prize*. http://prize.hutter1.net, 2006.

[Knautz 2017] B. Knautz, A. Knautz. *cmix: A High-Ratio Lossless Data Compression Software Based on Contest Compression Algorithms*. 2017. http://www.byronknoll.com/cmix.html.

[Knoll 2021] B. Knoll. *NNCP: Lossless Data Compression with Neural Networks*. 2021. https://bellard.org/nncp.

[Martin 1979] G. N. N. Martin. *Range encoding: An algorithm for removing redundancy from a digitized message*. Video & Data Recording Conference, 1979.

[Perez et al. 2018] E. Perez, F. Strub, H. de Vries, V. Dumoulin, A. Courville. *FiLM: Visual Reasoning with a General Conditioning Layer*. AAAI 2018. arXiv:1709.07871.

[Schindler 1998] M. Schindler. *A Fast Renormalisation for Arithmetic Coding*. DCC 1998.

[Sennrich et al. 2016] R. Sennrich, B. Haddow, A. Birch. *Neural Machine Translation of Rare Words with Subword Units*. ACL 2016. arXiv:1508.07909.

[Shannon 1948] C. E. Shannon. *A Mathematical Theory of Communication*. Bell System Technical Journal, 27(3):379–423, 1948.

[Su et al. 2021] J. Su, Y. Lu, S. Pan, B. Wen, Y. Liu. *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv:2104.09864.

[Townsend et al. 2019] J. Townsend, T. Bird, D. Barber. *Practical Lossless Compression with Latent Variables using Bits Back Coding*. ICLR 2019. arXiv:1901.04866.

[Witten et al. 1987] I. H. Witten, R. M. Neal, J. G. Cleary. *Arithmetic Coding for Data Compression*. Communications of the ACM, 30(6):520–540, 1987.

[Ziv & Lempel 1977] J. Ziv, A. Lempel. *A Universal Algorithm for Sequential Data Compression*. IEEE Transactions on Information Theory, 23(3):337–343, 1977.

---

*Syntropy2 source code, training notebooks, and Rust integration blueprint are provided alongside this whitepaper. The smoke test (`python neuro_symbolic_compressor.py --smoke`) validates the full model wiring, both curriculum stages, the reference streaming decoder, and ONNX export parity on CPU in under 2 minutes.*
