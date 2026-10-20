# How deployed long-context models avoid all-to-all attention

**Status:** literature research, 2026-10-20. No method here is implemented as a model in this repo;
the retention curves derived below *are* now in the FLOP notebook, as `ATTENTION_METHODS` behind its
last figure (added 2026-10-20, after this note). The question: a 200k-token context model almost
certainly does not run dense causal attention over 200k tokens — what do the published methods
actually do, which of them claim to be quality-lossless, and what would each one do to the curves in
[`../scripts/h_balanced_flops.ipynb`](../scripts/h_balanced_flops.ipynb)?

Numbers below were recomputed from each paper's stated hyperparameters against *our* cost model
(see [Translating a method into our charts](#translating-a-method-into-our-charts)); every quality
claim is attributed and dated. Anthropic has published nothing about Claude's attention
implementation — see [What Anthropic actually tells us](#what-anthropic-actually-tells-us); that
section separates the evidence from the inference.

## Summary of findings

1. **Exactly one class is provably lossless: the exact algorithms.** FlashAttention and Ring/Blockwise
   Attention compute the *same* attention output, bit-for-bit up to reassociation. They change memory
   traffic and parallel decomposition, **not** FLOPs. On our charts they move nothing. Every method
   that actually removes the quadratic term is an approximation; "lossless" in that literature always
   means "no measured degradation on our benchmark suite", never "same function".
2. **The strongest near-lossless evidence comes from sparsity that is trained in, not bolted on.**
   NSA (DeepSeek, ACL 2025) *beats* its full-attention baseline on 7 of 9 general benchmarks
   (avg 0.456 vs 0.443) and on LongBench (0.469 vs 0.437). DeepSeek-V3.2 reports benchmark parity
   with the dense V3.1-Terminus it was continued-trained from. MoBA's validation-loss gap against
   full attention stays within 1e-3 at 8k. These are the three to take seriously.
3. **Training-free methods are near-lossless on average but not uniformly.** MInference keeps ~95 % of
   the attention FLOPs off the table at 128k–1M, and *improves* RULER for two of three models tested —
   but drops Yi-9B-200K from 78.1 to 74.7 average. Query-aware KV selection (Quest) reports
   "negligible accuracy loss". Treat per-model, not per-method.
4. **Every one of these methods is a specific setting of the `SAM_ATTENTION_COST` knob already in our
   notebook**, with the difference that their effective cost fraction `c` is a *function of the
   context length* and usually falls as `L` grows. Converted to our model: 0.2 % (pure local) to
   50 % (Gemma-2-style 1:1) at 1M tokens, with the serious long-context designs clustering at
   **6–17 %**.
5. **Consequence for our crossover.** The context at which the score path overtakes the weights is
   `L* = (960H + V/H)/c`. At `H = 50` that is 49k tokens dense, 294k under a Gemma-3 5:1 schedule,
   784k under NSA. With a *fixed per-query budget* `B` (top-k methods), the score path never overtakes
   the weights at any context as long as **`B < 480H + V/(2H)`** — 24.5k tokens at `H = 50`,
   96.1k at `H = 200`. Every deployed budget we found (2048–49152) is below that line.
6. **Consequence for SAM.** At 1M tokens with a top-k budget the score path is already 4–25 % of a
   prefill, so "SAM removes the attention cost" stops being the interesting claim at deployment
   settings — the *parameter* saving (1.5·d² vs 3·d² of attention weights) is. There is also a real
   design problem: **MoBA, NSA, DSA and Quest all select blocks with a query–key score, and SAM has
   neither a Q nor a K projection.** See [What this means for SAM](#what-this-means-for-sam).

## Translating a method into our charts

Our model ([sam-model.md](../concepts/sam-model.md), notebook cell `inference_flops`) charges the
score path exactly:

```
pairs(L) = L(L+1)/2                 prefill        pairs = L        one decode step
per layer: 4 * datt * pairs         (QK^T + A@V, 2*head_dim per query-key-head)
total score FLOPs = 256 * H^2 * pairs           (datt = 64H, n_blocks = H, q_heads = H)
prefill weights   = 2L(61440H^3 + 64VH)
```

Define a method's **retention** `c(L)` = (query–key pairs it actually scores) / `L(L+1)/2`. That is
precisely our notebook's `attention_cost`, so any method below can be drawn today by setting
`SAM_ATTENTION_COST = c` — with the caveat that these `c` are *not* constants:

| Pattern | pairs(L) | retention `c(L)` |
|---|---|---|
| dense causal | `L(L+1)/2` | 1 |
| sliding window `w` | `Lw − w(w−1)/2` | `≈ 2w/L` |
| `r` local layers per global layer | mixture | `(r·2w/L + 1)/(r+1)` → floor `1/(r+1)` |
| fixed per-query budget `B` (top-k) | `≈ LB` | `≈ 2B/L` |
| stride-`d` compressed keys | `L(L+1)/(2d)` | `1/d` |
| linear attention layer | 0 pairs, `O(1)` state per token | 0 (plus a linear term, see below) |

The notebook carries this table as code: `ATTENTION_METHODS` in
[`../scripts/h_balanced_flops.ipynb`](../scripts/h_balanced_flops.ipynb) lists each method with its
`c(L)`, and the final figure draws the vanilla model under all nine against SAM.

Two structural facts follow and are worth stating before the individual papers:

- **A fixed budget makes attention linear in `L`, not sub-quadratic-but-still-quadratic.** The score
  path becomes `256H²·B·L`, the weights are `(122880H³ + 128VH)·L`, so their *ratio is constant in
  `L`*. Attention either is or is not the dominant term, for all context lengths at once, decided by
  `B` vs `B* = 480H + V/(2H)`.
- **A per-layer floor beats a per-token budget asymptotically.** Any schedule that keeps a fraction
  `1/(r+1)` of layers fully dense has `c → 1/(r+1)` no matter how long the context. The dense layers,
  not the local ones, set the 1M-token bill. This is visible in the MoBA row of the table below: 3
  dense layers out of 32 contribute 9.4 % of the dense cost at 1M, roughly as much as the 29 sparse
  layers together.

## The methods

### 0. Exact: FlashAttention and Ring Attention — the only genuinely lossless ones

FlashAttention (Dao et al., 2022) tiles the softmax so the `L × L` score matrix is never
materialised; Ring/Blockwise Attention (Liu, Zaharia & Abbeel, 2023) splits the sequence across
devices and rotates KV blocks so context scales with device count. Both compute exact attention.
**Effect on our charts: none.** `c = 1`. They are the reason a 1M-token prefill is *possible* at all,
not the reason it is *cheap*. Worth stating explicitly in any discussion with the group, because
"we use FlashAttention" is often offered as an answer to the quadratic-cost question and is not one.

### 1. Static locality with a global escape hatch — Gemma 2 / Gemma 3

Google's Gemma 3 technical report (arXiv 2503.19786) interleaves **5 local layers per global layer,
starting with a local layer**, with a **1024-token** window on the local layers; Gemma 2 used 1:1
with a 4096-token window. The report's ablations say the quality price is small: *"We observe minimal
impact on perplexity when changing this ratio"*, and its Figure 3 caption adds *"The impact is
minimal, even with 7-to-1 local to global"*; on window size, *"The sliding window can be reduced
significantly without impacting perplexity."* The motivation is memory, not FLOPs: at a 32k context
the KV cache is *"a memory overhead of 60 %"* for global-only, *"reduced to less than 15 % with 1:3
and sliding windows of 1024"*.

Quality evidence is perplexity-level on a validation set only — the report gives no benchmark-level
comparison for this change. That is weaker evidence than NSA's or DSA's, and the known failure mode
of static locality (information outside the window reaches later layers only through the periodic
global layers) is exactly what long-range retrieval stresses.

Retention: `c = (5·2w/L + 1)/6` → **17.96 % at 128k, 16.83 % at 1M**, floor 1/6 = 16.7 %.

### 2. Static locality taken further — MixAttention / the Character.AI recipe

MixAttention (Rajput et al., NeurIPS 2024 ENLSP, arXiv 2409.15012) is the published study of the
Character.AI serving architecture: most layers sliding-window, MQA everywhere, and **the KV cache
shared across layers**. Findings: KV cache *"of standard attention layers can be shared between
non-consecutive layers without any observed degradation in long context abilities"*, while sharing
too aggressively *between sliding-window layers* does hurt; and dense layers matter more when placed
deep than at the start. Some configurations *"match the standard transformer model in most short and
long context evals"*.

Retention: same algebra as Gemma. Cross-layer KV sharing changes **no FLOPs in our model at all** —
it divides the cache bytes. Our charts cannot show its main benefit, which is the point of §
[What the FLOP charts cannot show](#what-the-flop-charts-cannot-show).

### 3. Hybrid linear attention — MiniMax-01

MiniMax-01 (arXiv 2501.08313, 456B total / 45.9B active, 80 layers) places **one softmax block after
every seven lightning-attention (linear) blocks**, trained to 1M and serving up to 4M tokens. Pure
linear attention was rejected on evidence, not taste: it *"demonstrates limited retrieval
capabilities"* and performs comparably *"with the exception of NIAH"*, which the authors call
*"unsuitable for LLMs"*. The hybrid, by contrast, fits a *better* scaling law than softmax
(`3.4797·C^−0.0763` vs `3.7087·C^−0.0798`) and scores 95.7 NIAH at 1B / 98.0 at 3B against 43.6 for a
hybrid-cosformer2 control; the MoE ablation is a wash to slightly positive against softmax
(MMLU 49.5 vs 49.3, BBH 32.2 vs 28.2, CMMLU 46.0 vs 47.3).

Retention: the linear layers score **zero** pairs, so `c = 1/8 = 12.5 %` flat, context-independent.
The recurrent state costs extra linear FLOPs: a `64 × 64` state per head is `≈ 4·head_dim² = 16384`
FLOPs per token per head, i.e. `16384·H²` per token against `122880·H³` of weight work — **0.27 % at
`H = 50`**, negligible. In FLOP terms a linear layer is free; its cost is kernel efficiency and state
capacity (`O(d²/h)` vs softmax's `O(d)` per the paper's own framing).

### 4. Trained-in dynamic sparsity — NSA (DeepSeek, ACL 2025)

NSA (arXiv 2502.11089) is the strongest near-lossless result found. Three branches, gated and summed:
**compression** (blocks of `l = 32` with stride `d = 16`, pooled by an MLP), **selection** (top
`n = 16` blocks of `l' = 64`, scored for free from the compression branch's attention weights, 1
initial + 2 local blocks forced), and a **sliding window** of `w = 512`. Trained from scratch, 27B
MoE / 3B active, 30 layers, 64 heads in 4 GQA groups, 8k pretraining then 32k with YaRN.

Quality, trained end-to-end rather than retrofitted (Tables 1–3): general average **0.456 vs 0.443**
for full attention, ahead on 7 of 9 benchmarks (GSM8K 0.520 vs 0.486, DROP 0.545 vs 0.503, MMLU 0.565
vs 0.567); LongBench average **0.469 vs 0.437**, against 0.423 for exact top-k, 0.392 Quest, 0.383
InfLLM, 0.303 H2O; perfect needle retrieval at 64k; AIME-24 after SFT 0.121 vs 0.046 at an 8k
generation limit. Measured speed at 64k: **9.0× forward, 6.0× backward** vs Triton FlashAttention-2,
and an *expected* 11.6× decode derived from memory-access volume (5632 vs 65536 tokens), not measured
latency.

Retention is the interesting part, because NSA is **not** a fixed budget: the compression branch
attends to every preceding compressed block, so it keeps a quadratic term reduced by the stride.
`c ≈ 1/16 + 2(n·l' + w)/L` = **8.58 % at 128k, 6.54 % at 1M**, with a hard floor at `1/d = 6.25 %`.

### 5. Trained-in dynamic sparsity — MoBA (Moonshot, deployed in Kimi)

MoBA (arXiv 2502.13189) applies MoE routing to attention: split the context into blocks, let a
parameter-free top-k gate pick which blocks each query attends to. Configurations: block 512 / top-3
at 8k (81.25 % sparsity), block 2048 / top-3 at 32k, **block 4096 / top-12 for Llama-8B-1M-MoBA**
(95.31 % sparsity at 1M, 62.5 % on RULER at 128k). Two hybrid recipes matter: **the last 3 of 32
layers stay full attention**, and training runs MoBA for the first 90 % of tokens then full attention
for the last 10 %.

Quality: the 8k scaling-law fit is `2.625·C^−0.063` against `2.622·C^−0.063` for full attention, with
*"the validation loss gap stays within 1e-3"*. The honest caveat is in the same paper: on the
**trailing 2k tokens at 32k**, MoBA fits `1.546·C^−0.108` against full attention's `1.464·C^−0.097` —
a real gap, which the authors argue is narrowing with scale. Reported **6.5× prefill speedup at 1M**
and 16× less attention compute at 10M. Deployed: *"MoBA has already been deployed to support Kimi's
long-context requests."*

Retention with the 1M config and the 3 dense layers: **64.60 % at 128k, 17.67 % at 1M** — and of that
17.67 %, 9.4 points are the three dense layers. Without them the sparse layers alone are 9.16 % at 1M.
At 32k the 1M config's budget (12 × 4096 = 49152) exceeds the context, so it degenerates to dense;
MoBA is a *very* long context design and its own per-length configs reflect that.

### 6. Trained-in dynamic sparsity — DSA (DeepSeek-V3.2, the most recent production datapoint)

DSA (DeepSeek-V3.2 report, arXiv 2512.02556, and the identical V3.2-Exp architecture from 2025-09)
adds a **lightning indexer** — `I(t,s) = Σ_j w_j · ReLU(q_j · k_s)` over `H_I` small FP8 heads — and
then runs attention over only the **top-k = 2048** key-value entries. Published config
(`index_n_heads = 64`, `index_head_dim = 128`, `index_topk = 2048`, 128 query heads, MLA with
`kv_lora_rank = 512`, `qk_rope_head_dim = 64`). Instantiated under MLA in MQA mode so one latent
entry serves all query heads.

Training: a **dense warm-up** of 1000 steps / 2.1B tokens that trains only the indexer to match the
dense attention distribution by KL, then a **sparse stage** of 15000 steps / **943.7B tokens** at
k = 2048 with the indexer's gradient detached from the main loss.

Quality: *"we do not observe substantial performance degradation compared with DeepSeek-V3.1-Terminus,
on both short- and long-context tasks"*, closely matched ChatbotArena Elo (2025-11-10), and
independent long-context evaluations that go the *right* way — AA-LCR four points higher in reasoning
mode, Fiction.liveBench consistently better. Note this is still a vendor-run comparison; the vLLM
team was still verifying its own implementation at release.

Cost, in the paper's own words: *"DSA reduces the core attention complexity of the main model from
O(L²) to O(Lk) ... Although the lightning indexer still has a complexity of O(L²), it requires much
less computation compared with MLA."* Quantified with their shapes: the indexer costs
`2·64·128 = 16,384` FLOPs per pair against `2·128·(576 + 512) ≈ 278,528` for the MLA core, i.e.
**5.9 %** — so `c ≈ 0.059 + 2k/L` = **9.00 % at 128k, 6.29 % at 1M**, with a floor at the indexer's
5.9 %. *This floor is architecture-specific and does not transfer to us*: DeepSeek can afford a
64-head indexer because its dense per-pair cost is enormous (128 heads × 576 dims). In an H-balanced
model the dense per-pair cost is `4·datt = 256H` FLOPs per layer, so an indexer of `H_I` heads of
dim 64 costs a fraction `φ = H_I/(2H)` — to stay under a 5 % floor at `H = 50` it may have at most
**5 heads**. Design rule: `H_I ≤ 2·c_target·H`.

A December 2025 study (arXiv 2512.03494) measured the indexer's fidelity: DeepSeek-V3.2-Exp's
lightning indexer has *"an average precision of about 60 %"* against exact top-2048 on HELMET-128k,
and downstream quality tracks that precision until it saturates. So the parity result is **not**
evidence that approximate selection is harmless in general — the same study finds that top-k
*decoding* "rivals or outperforms full attention even at low Top-k ratios" when training and
inference use the same top-k, which is the condition DeepSeek satisfies and a retrofit does not.

### 7. Training-free prefill sparsity — MInference 1.0 (NeurIPS 2024 spotlight)

MInference (arXiv 2407.02490) assigns each head, **offline and once** (~15 min on an A100), one of
three patterns: **A-shape** (1024 global + 4096 local), **Vertical-Slash** (a few hundred vertical
columns + slash diagonals, searched over (30, 2048), (100, 1800), (500, 1500), (3000, 200)), or
**Block-Sparse** (top 100 blocks of 64). Indices are then built dynamically per input. No fine-tuning
of the base model.

Evidence for "near-lossless": RULER average improves for LLaMA-3-8B-262K (84.4 → 87.0) and
GLM-4-9B-1M (88.0 → 89.6) but **degrades for Yi-9B-200K (78.1 → 74.7)**; InfiniteBench averages are
flat (38.2 → 38.8, 37.5 → 37.7, 46.7 → 47.0). Measured prefill speedup on one A100: 1.8× at 100k,
4.1× at 300k, 6.8× at 500k, **10× at 1M** (~30 min → ~3 min). Sparsity rationale: for a 128k × 128k
attention matrix, the **top 4k columns recall 96.8 %** of total attention.

Retention with an effective budget of ~4k retained columns/diagonals per query: **6.15 % at 128k,
0.78 % at 1M** (consistent with their "~95 % of attention FLOPs" claim). Caveat from the same
literature: below ~10k tokens the index-building overhead can reach 30 % of runtime and erase the
gain — this is a long-prompt method only.

### 8. Training-free decode sparsity — Quest, and the KV-eviction family

Quest (ICML 2024, arXiv 2406.10774) keeps per-page min/max key vectors, scores each page against the
current query, and loads only the top-K pages: *"up to 2.23× self-attention speedup"*, *"reduces
inference latency by 7.03×"*, with *"negligible accuracy loss"* on long-dependency tasks. Crucially
it is **query-aware and non-destructive** — the full cache stays, only the read set shrinks. The
older eviction family (StreamingLLM's sinks + window, H2O's cumulative-score eviction, SnapKV's
observation-window selection) *deletes* entries, which is why SnapKV's own critique of StreamingLLM
is that discarding middle tokens loses what they carried, and why NSA's LongBench table puts H2O at
0.303 against 0.437 for full attention. For near-lossless behaviour, prefer selection over eviction.

Retention (decode only; prefill unchanged): pairs per step `min(L, B)` instead of `L`, so
`c = min(1, B/L)` — **3.10 % at 128k, 0.39 % at 1M** for B = 2048.

### Retention summary

Recomputed against our cost model, each paper's own hyperparameters, exact causal pairs:

| Method | `c(32k)` | `c(128k)` | `c(200k)` | `c(1M)` | asymptotic floor |
|---|---:|---:|---:|---:|---:|
| dense (FlashAttention / Ring) | 100 % | 100 % | 100 % | 100 % | 1 |
| Gemma 2, 1:1, `w = 4096` | 61.72 % | 53.08 % | 52.03 % | 50.39 % | 1/2 |
| Gemma 3, 5:1, `w = 1024` | 21.79 % | 17.96 % | 17.52 % | 16.83 % | 1/6 |
| MiniMax-01, 7:1 linear:softmax | 12.50 % | 12.50 % | 12.50 % | 12.50 % | 1/8 |
| Kimi Linear / Kimi K3 / Qwen3-Next, 3:1 linear:full | 25.00 % | 25.00 % | 25.00 % | 25.00 % | 1/4 |
| MoBA, block 4096 / top-12, 29 of 32 layers | n/a | 64.60 % | 48.45 % | 17.67 % | 3/32 |
| DSA, top-k 2048 + 5.9 % indexer | 18.01 % | 9.00 % | 7.94 % | 6.29 % | 0.059 |
| NSA, cmp(32/16) + 16×64 + win 512 | 15.41 % | 8.58 % | 7.78 % | 6.54 % | 1/16 |
| MInference, ~4k lines per query | 23.44 % | 6.15 % | 4.05 % | 0.78 % | 0 |
| pure sliding window, `w = 1024` | 6.15 % | 1.56 % | 1.02 % | 0.20 % | 0 |

### Who ships what, as of 2026-10-20

The notebook's legend names models per method; these are the attributions behind it. Primary
sources are the model card or technical report; everything marked *(secondary)* is press or
third-party analysis and should be re-checked before it is quoted in a paper.

| Method | Models | Evidence |
|---|---|---|
| dense | Llama 3.1, DeepSeek-V3.1-Terminus, MiniMax-M2 | V3.1-Terminus is the dense baseline in the DSA report; MiniMax is reported to have dropped linear attention after M1, because it "had poor accuracy in reasoning and multi-turn tasks" *(secondary)* |
| 1:1 local:global, w=4096 | Gemma 2 | Gemma 3 report, describing its predecessor |
| 5:1 local:global, w=1024 | Gemma 3, Gemma 4 | Gemma 3 technical report; Gemma 4 is 5:1 for all variants but E2B (4:1), last layer global *(secondary)* |
| 7:1 linear:softmax | MiniMax-01, MiniMax-M1 | MiniMax-01 paper: "one transformer block with softmax attention follows every seven transnormer blocks" |
| 3:1 linear:full | Kimi Linear, Kimi K3, Qwen3-Next | Kimi Linear: 3:1 KDA to global MLA, 48B-A3B. Kimi K3 model card: 93 layers = **69 KDA + 24 gated MLA**, 2.8T total / 104B active, 1M context. Qwen3-Next-80B-A3B card: 48 layers = **36 Gated DeltaNet + 12 Gated Attention**, 262k native |
| MoBA | Kimi | the MoBA paper: "deployed to support Kimi's long-context requests" |
| NSA | DeepSeek 27B research model | no shipped model names it; the 27B MoE of the NSA paper is the only trained instance published |
| DSA | DeepSeek-V3.2, GLM-5, GLM-5.2, GLM-5.3 | V3.2 report; GLM-5 report: it "adopts DSA to significantly reduce training and inference costs while maintaining long-context fidelity". GLM-5.2/5.3 add **IndexShare** - the indexer runs once per four layers instead of every layer, claimed at 2.9x less per-token compute at 1M *(secondary)*, which would put their indexer floor near 1.5% rather than DSA's 5.9% |
| MInference | retrofit onto Yi-200K, GLM-4-9B-1M | a serving-time retrofit, not a model family: the paper applies it to LLaMA-3-8B-262K, GLM-4-1M, Yi-200K, Phi-3-128K and Qwen2-128K |
| local only, w=1024 | Mistral 7B, Character.AI | both use wider windows than the 1024 drawn here; listed as the shape of the curve, not its exact setting |

Two readings worth keeping: **DSA is the first of these to be adopted by a second lab** - the GLM-5
line is DeepSeek's mechanism, not its own - and **the linear hybrids have converged on 3:1**
(Kimi Linear, Kimi K3, Qwen3-Next) rather than MiniMax's 7:1, which costs 25% of the dense pair
work instead of 12.5% and is the ratio to use when arguing from this family.

"n/a": the 1M MoBA config's budget (49152) exceeds a 32k context, so it degenerates to dense; MoBA
uses smaller blocks at smaller contexts. Computed by
`/tmp/claude-1000/-home-maciej-code-nano/1baa3f0c-88a9-404a-b395-d7a00856d145/scratchpad/lc2.py`
(regenerate: the formulas are in [Translating a method into our charts](#translating-a-method-into-our-charts)).

## What this does to our charts

### Prefill

Vanilla H-balanced, gated FF, `V = 50257`, exact causal pairs. TFLOPs for one full prefill:

**`H = 200` (492.81B), weights 196,865 TFLOPs at 200k / 1,032,141 at 1M:**

| Method | scores @200k | total @200k | attn share | scores @1M | total @1M | attn share | speedup @1M |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 204,801 | 401,666 | 51.0 % | 5,629,505 | 6,661,646 | 84.5 % | 1.00× |
| Gemma 2 1:1 | 106,552 | 303,417 | 35.1 % | 2,836,700 | 3,868,841 | 73.3 % | 1.72× |
| Gemma 3 5:1 | 35,877 | 232,742 | 15.4 % | 947,409 | 1,979,550 | 47.9 % | 3.37× |
| MiniMax 7:1 | 25,600 | 222,465 | 11.5 % | 703,688 | 1,735,829 | 40.5 % | 3.84× |
| linear 3:1 | 51,200 | 248,066 | 20.6 % | 1,407,376 | 2,439,517 | 57.7 % | 2.73× |
| MoBA 1M config | 99,217 | 296,082 | 33.5 % | 994,844 | 2,026,985 | 49.1 % | 3.29× |
| NSA | 15,934 | 212,799 | 7.5 % | 368,325 | 1,400,466 | 26.3 % | 4.76× |
| DSA | 16,256 | 213,121 | 7.6 % | 354,110 | 1,386,251 | 25.5 % | 4.81× |
| MInference | 8,303 | 205,168 | 4.0 % | 43,895 | 1,076,036 | 4.1 % | 6.19× |
| local only `w=1024` | 2,092 | 198,957 | 1.1 % | 10,990 | 1,043,131 | 1.1 % | 6.39× |

**`H = 50` (8.00B), weights 3,136 TFLOPs at 200k / 16,443 at 1M:** the same `c` hurts more, because
the weight term is `H³` and the score term only `H²` — dense attention is 80.3 % of a 200k prefill
and 95.5 % of a 1M prefill. Under NSA/DSA the 1M prefill falls from 368,287 to ~38,600–39,500 TFLOPs
(**9.3–9.6×**), under MInference to 19,187 (**19.2×**), against a floor of 17,130 (21.5×) set by the
weights alone. This is the mirror image of the trend already in the notebook's fourth sweep: the
smaller the model, the more a context-length intervention buys.

### The crossover moves by exactly `1/c`

The notebook's crossover (score path = weight path) is `L* = 960H + V/H`. Any constant-`c` method
multiplies it by `1/c`:

| | dense | Gemma 2 1:1 | linear 3:1 | Gemma 3 5:1 | MiniMax 7:1 | NSA floor | DSA floor |
|---|---:|---:|---:|---:|---:|---:|---:|
| `H = 50` | 49,005 | 97,252 | 196,021 | 294,031 | 392,041 | 784,082 | 830,596 |
| `H = 200` | 192,251 | 381,527 | 769,005 | 1,153,508 | 1,538,010 | 3,076,021 | 3,258,496 |

For fixed-budget methods there is no crossover at all: score and weight work are both linear in `L`,
and attention stays the smaller term for every context iff `B < B* = 480H + V/(2H)`:

| `H` | 12 | 50 | 100 | 200 | 252 |
|---|---:|---:|---:|---:|---:|
| `B*` (tokens) | 7,854 | 24,503 | 48,251 | 96,126 | 121,060 |

Every budget in the surveyed papers (Quest 2048, DSA 2048, MInference ~4096, NSA 1536 + compression,
MoBA 49152) sits below `B*` at `H ≥ 50`. **At the scale of our table rows, a top-k budget removes
attention from the cost equation entirely and permanently.**

### Decode is where it is decisive

Per generated token, TFLOPs (weights are constant in `L`; this is the notebook's second figure):

| | weights | dense scores @128k | @1M | top-k 2048 scores (any `L`) |
|---|---:|---:|---:|---:|
| `H = 50` | 0.0157 | 0.0839 (84.3 %) | 0.6711 (**97.7 %**) | 0.0013 (7.7 %) |
| `H = 200` | 0.9843 | 1.3422 (57.7 %) | 10.7374 (**91.6 %**) | 0.0210 (2.1 %) |

At 1M tokens dense decode attention is **43×** the weight cost at `H = 50` and **11×** at `H = 200`.
No production system pays this. A 2048-token budget turns the curve from linear-in-`L` into a flat
line a few percent above the weights — which is the same shape as the SAM decode curve in our
notebook, reached by a different route.

### What the FLOP charts cannot show

Three of the surveyed designs are chosen for reasons our figures are blind to, and this limits how
far our charts can be used to argue about deployment:

- **KV-cache bytes.** At bf16, an H-balanced model stores `2 · H · (H/2) · 64 · 2 = 128H²` bytes per
  token: 312.5 KiB at `H = 50` (**335 GB** at 1M) and 4.88 MiB at `H = 200` (**5.4 TB** at 1M). A
  Gemma-3 5:1 schedule cuts that to 56 GB / 899 GB; a 7:1 linear hybrid to 42 GB / 671 GB;
  cross-layer sharing divides it again at **zero** FLOP cost. Decode is bandwidth-bound, so this —
  not FLOPs — is usually the binding constraint, and it is why MQA/GQA/MLA/MixAttention exist.
- **Arithmetic intensity.** NSA's whole framing is "hardware-aligned": a sparsity pattern that does
  not map onto contiguous blocks buys FLOPs you cannot cash. MInference loses its advantage below
  ~10k tokens to index-building overhead. Our charts assume every FLOP removed is a FLOP saved.
- **Prefix reuse.** Prompt caching makes the *amortised* prefill cost of a long conversation far
  below what a per-request chart suggests.

## What Anthropic actually tells us

**Evidence:** Anthropic has published no architectural description of Claude's attention. What is
public is behavioural and commercial: a 200k window since Claude 2.1 (2023-11); a 1M-token beta for
Sonnet 4 / 4.5 behind the `context-1m-2025-08-07` header, tier-4 only; and a documented pricing step
— *"requests that exceed 200K input tokens are automatically charged at premium long context rates"*
(2× input, 1.5× output per gateway listings). Also public: prompt caching with cache-read pricing at
10 % of input, and context editing / compaction features.

**Inference, flagged as such:** a *flat* 2× multiplier covering everything from 200k to 1M is hard to
reconcile with a cost that grows as `L²` — dense attention at `H = 200` would make a 1M prefill 16.6×
a 200k one, i.e. 3.2× per token, not 2×. A fixed-budget or floored-`c` scheme (the 6–17 % band above)
gives a near-constant per-token cost, which is what a flat premium prices. That is consistent with,
not evidence of, sparse attention; pricing is a commercial decision and the inference is weak. The
honest statement for the group: **the public record does not say, the published state of the art that
a frontier lab would plausibly draw on is sections 1–8 above, and anything more specific about Claude
is speculation.**

## What this means for SAM

1. **The score path is not the headline at deployment settings.** Our SAM figures compare SAM's `c`
   against a *dense* vanilla baseline. Against a vanilla model that uses any of these methods, the
   comparison collapses to parameters and weight FLOPs: at 1M and `c = 6 %`, the score path is 26 %
   of an `H = 200` prefill and 58 % of an `H = 50` one, and under MInference-grade sparsity 4 % and
   14 %. The fourth sweep in the notebook already shows that the SAM advantage erodes with scale;
   sparse attention erodes it again along the other axis. The notebook's last figure now makes this
   explicit for one pair: SAM at `H = 100` against vanilla `H = 80` under all nine methods. SAM
   carries 1.75x the weight work there, so it needs the vanilla model to score more than ~29,200
   keys per query to win at any context - which means it beats dense attention past 58k tokens,
   NSA past 885k, DSA past 920k, and MInference-grade sparsity or a plain 1k window **never**.
2. **SAM and sparse attention are composable, not competing** — SAM removes `q_proj`/`k_proj`
   (1.5·d² per layer); sparsity removes pairs. Nothing stops both.
3. **But SAM may structurally exclude the best-evidenced methods.** NSA, MoBA, DSA and Quest all
   decide *what to attend to* from a query–key similarity, and SAM has neither projection. The
   options left are (a) static locality (Gemma/MixAttention-style, the weakest quality evidence),
   (b) a learned selector that is not the attention's own Q/K — which is exactly what DSA's lightning
   indexer is, and reintroduces a small Q/K pair, costing `φ = H_I/(2H)` of the dense score path, or
   (c) whatever selection mechanism SAM's own memory formulation implies. **This is the open question
   worth answering before the SAM cost story is presented externally**, and `H_I ≤ 2·c_target·H` is
   the budget it has to fit in.

## Sources

Papers, in the order used:

- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Liu, Zaharia & Abbeel, *Ring Attention with Blockwise Transformers for Near-Infinite Context*, [arXiv:2310.01889](https://arxiv.org/abs/2310.01889)
- Gemma Team, *Gemma 3 Technical Report*, [arXiv:2503.19786](https://arxiv.org/abs/2503.19786)
- Rajput et al., *Inference-Friendly Models With MixAttention*, [arXiv:2409.15012](https://arxiv.org/abs/2409.15012)
- MiniMax, *MiniMax-01: Scaling Foundation Models with Lightning Attention*, [arXiv:2501.08313](https://arxiv.org/abs/2501.08313)
- Yuan et al. (DeepSeek-AI), *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*, ACL 2025, [arXiv:2502.11089](https://arxiv.org/abs/2502.11089)
- Lu et al. (Moonshot AI), *MoBA: Mixture of Block Attention for Long-Context LLMs*, [arXiv:2502.13189](https://arxiv.org/abs/2502.13189)
- DeepSeek-AI, *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models*, [arXiv:2512.02556](https://arxiv.org/abs/2512.02556); config values from [the V3.2-Exp weights repo](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp)
- Jiang et al. (Microsoft), *MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention*, NeurIPS 2024, [arXiv:2407.02490](https://arxiv.org/abs/2407.02490)
- Tang et al., *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference*, ICML 2024, [arXiv:2406.10774](https://arxiv.org/abs/2406.10774)
- Gao et al. (Microsoft), *SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs*, [arXiv:2410.13276](https://arxiv.org/abs/2410.13276) — learned block-sparse gate, self-distilled from the dense model; same family as NSA's selection branch, retrofitted
- *A Preliminary Study on the Promises and Challenges of Native Top-k Sparse Attention*, [arXiv:2512.03494](https://arxiv.org/abs/2512.03494) — the ~60 % indexer-precision measurement
- Li et al., *SnapKV*, [arXiv:2404.14469](https://arxiv.org/abs/2404.14469); Zhang et al., *H2O*, NeurIPS 2023; Xiao et al., *StreamingLLM*, [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) — the eviction family, included as the negative control

Model cards and reports behind the attribution table:

- GLM-5 Team, *GLM-5: from Vibe Coding to Agentic Engineering*, [arXiv:2602.15763](https://arxiv.org/abs/2602.15763); GLM-5/5.1/5.2 as MLA + DSA in [NVIDIA NeMo's model coverage](https://docs.nvidia.com/nemo/automodel/latest/model-coverage/large-language-models/glm-5-moe-dsa)
- [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) (93 layers: 69 KDA + 24 gated MLA) and [MoonshotAI/Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear) (3:1 KDA : global MLA)
- [Qwen/Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) (12 x (3 x Gated DeltaNet -> 1 x Gated Attention))

Vendor documentation: [Anthropic pricing / 1M context beta](https://docs.claude.com/en/docs/about-claude/pricing).
