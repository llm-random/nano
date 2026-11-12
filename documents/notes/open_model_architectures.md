# Open-model architectures, November 2026: which are best and what we'd need

As of 2026-11-10. Research agents collected these facts from primary sources: model cards, `config.json`,
tech reports and official code (Sources at the end). The numbers were not re-checked against each source by
hand. The ratings are my own judgement.

## Short answer

**DeepSeek V4/V4.1 and Kimi K3 have the best architectures**, and Qwen3.8 is close behind. GLM-5.3 and Mistral
Large 4 are mostly DeepSeek's design (MLA + DSA + MoE) with additions. Gemma 4 is the best *small dense* design.
Olmo 3 and SmolLM3 are architecturally behind, but they are the only recipes that are fully reproducible
(code, data, configs, failures).

| Family | Latest open weights | Architecture (1-5) | Training disclosed | Use for us |
| --- | --- | --- | --- | --- |
| DeepSeek | V4.1-Flash (2026-09), V4-Pro (2026-04) | 5 | High: optimizer, LR, batch, tokens, stages; not the data mix | Best source of ideas; the V4 report is close to a recipe |
| Kimi | K3 (2026-07), K3.1-Pro (2026-11) | 5 | Medium: optimizer, schedule shape, context stages; no tokens/LR/batch | 3:1 linear hybrid, NoPE, per-head Muon |
| Qwen | Qwen3.8 2.4T-A95B, 27B dense, Flash-Next (2026-08) | 4 | Flagship none; Flash-Next has a full report | 3:1 GDN hybrid that also ships dense (27B) |
| Gemma | Gemma 4 (2026-03), 12B Unified (2026-06) | 4 | Low: no tokens, optimizer or schedule | Small dense design: asymmetric local/global layers |
| GLM | GLM-5.3, GLM-5.3-Flash (2026-08) | 3 | Medium: GLM-5 report; schedule inherited from GLM-4.5 | Muon Split, cosine over WSD |
| Mistral | Large 4 (2026-10), Medium 3.5, Ministral 3 | 3 | Medium: 30-page report, but no tokens, LR or schedule | Negative result on SWA hybrids; cascade distillation |
| OLMo | Olmo 3 32B (2025-11), Olmo Hybrid 7B (2026) | 3 (Hybrid 4) | Full: code, data, every config | Most reproducible baseline at our scale |
| SmolLM | SmolLM3-3B (2025-07) | 2 | Full, plus the playbook of what failed | Small-scale recipe: NoPE, doc masking, WSD, curriculum |

**Two trends worth noting:**
- At the frontier, attention has split into two camps:
  - **3:1 linear-attention hybrids:** Kimi KDA, Qwen Gated DeltaNet, Olmo Hybrid, GLM-5.3-Flash.
  - **Indexer-based sparse attention:** DeepSeek CSA/DSA, GLM, Mistral Large 4, Kimi K3.1, Qwen Flash-Next QSA.
- **Muon has replaced AdamW** at DeepSeek, Kimi, GLM and Qwen Flash-Next. The small fully-open models still use
  AdamW; SmolLM3 reports that Muon diverged at 3B.

## Our baseline (pretraining-jm)

- **Architecture:** pre-norm RMSNorm; RoPE on all head dims with θ = 1e4; SwiGLU; GQA/MHA/MQA; untied embeddings;
  GPT-NeoX 50k vocab.
- **Optimizer and schedule:** AdamW as one param group, with weight decay 0.1 on everything including embeddings
  and norms; cosine schedule with 1% warmup.
- **Precision and data:** bf16 with FSDP2. Documents are joined with EOS and attend across each other: no
  document masking, no z-loss.
- **Registered stubs:** `mla` and `gated_deltanet` exist in `src/core/attention.py` but raise on forward.

The "we'd need" lists below are deltas from this baseline. The switch list is in
`documents/notes/pretraining_switches.md`.

## Feature matrix (latest flagship of each family)

Legend: ✓ uses · — does not · ? not disclosed. The last column lists what
`pretraining_switches.md` plans; there, — means the feature is not on the switch list.

| Feature | DeepSeek V4 / V4.1 | Kimi K3 | Qwen3.8 | GLM-5.3 | Mistral Large 4 | Gemma 4 | Olmo 3 | SmolLM3 | pretraining-jm | pretraining-jm plan |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MoE | 384+1, top-6 | 896+2, top-16, latent | 512, 10+1 shared | 256+1, top-8 | 128+1, top-4 | 26B only: 128 top-8 + dense branch | — | — | — (removed) | planned: port `moe.py` from `main` or rewrite |
| MLA / low-rank KV | 1-head MQA, low-rank | ✓ gated MLA | — (GQA 64/4) | ✓ | ✓ | — | — | — | stub | — (stub only) |
| Indexer sparse attention | CSA/HCA; V4.1 CSA2 | K3.1: DSA | Flash-Next: QSA | DSA + IndexShare | DSA, indexer shared | — | — | — | — | planned? |
| 3:1 linear-attn hybrid | — | KDA | Gated DeltaNet | 5.3-Flash: KDA | — | — | Hybrid 7B: GDN | — | stub | — (stub only) |
| Sliding-window layers | SWA branch (w 128) | — | — | — | — (tried, dropped) | 5:1, w 512/1024 | 3:1, w 4096 | — (tried, worse) | — | — |
| Positions | partial RoPE (64 dims), YaRN | NoPE everywhere | partial RoPE 25%, θ 1e7 | partial RoPE (64 of 256), θ 8e6 | NoPE every 4th, YaRN, LeRoPE | p-RoPE 25% on global layers | RoPE 5e5, YaRN on full layers | NoPE every 4th, θ 50k→5M | full RoPE, θ 1e4 | RoPE exists; a NoPE switch is an open question |
| QK-norm | ✓ per head (q, KV) | L2 q,k in KDA + QK-clip | ✓ | — (latent norms) | — | ✓ (+ V-norm) | ✓ full width | — | — | planned: port from `origin/ctx_scl` |
| Norm placement | pre-norm inside mHC | pre-norm + AttnRes | zero-centered pre-norm | pre-norm | pre-norm | sandwich | post-sublayer ("reordered") | pre-norm | pre-norm | nGPT planned (normalized updates replace pre-norm) |
| Wider residual stream | mHC, 4 streams | Block AttnRes | Flash-Next: Gated Residual | 5.3-Flash: mHC | — | — | — | — | — | planned? |
| Attention output gate | — | ✓ sigmoid | ✓ sigmoid | — | — | — | — | — | — | planned |
| FFN | SwiGLU, clamped | SiTU-GLU | SwiGLU | SwiGLU | SwiGLU | GeGLU | SwiGLU | SwiGLU | SwiGLU | SwiGLU exists; GELU MLP via `_target_` |
| MTP | V4: depth 1; V4.1: post-hoc drafter | 1 layer | ✓ | 3 shared layers | 3-layer draft head | separate drafter | — | — | — | — |
| Logit soft-cap | — | — | — | — | 30 | 30 | — | — | — | — (nGPT adds a learned logit scale) |
| Tied embeddings | — | — | — | — | — | ✓ (×√d) | — | ✓ | — | — |
| Optimizer | Muon + AdamW | per-head Muon + QK-clip | Flash-Next: Muon (3.5: AdamW) | Muon Split | AdamW, β 0.975/0.975 | ? | AdamW + skip-step | AdamW | AdamW | planned: Muon (+ AdamW for embeddings, head, norms) |
| LR schedule | constant → cosine to 10% | cosine (beat WSD) | ? | cosine (GLM-4.5) | ? | ? | truncated cosine + linear anneals | WSD | cosine | muP planned (per-group LR multipliers, width-scaled init) |
| Batch ramp | ✓ to 75–95M tokens | ? | no (ramp didn't help) | 16M → 64M | 40M → 160M | ? | — | — | — | — |
| Precision | FP8 (+ FP4 QAT later) | FP8 activations | ? | ? (INT4 QAT in SFT) | FP8 | ? | bf16 | bf16 | bf16 | planned: FP8 via torchao |
| Document masking | ✓ | ? | ? | ? | ? | ? | long-context stage | ✓ | — | planned |
| Training tokens | 32–33T (V4), 45T (V4.1) | ? | ? | 28.5T | ? | ? | 5.9T + 0.2T | 11.2T | — | — |

H-net and token superposition are also on the switch list but have no row here, because none of the eight
families use them.

## DeepSeek: 5/5

**Latest releases:**
- **V4-Pro:** 1.6T total / 49B active, 61 layers, d 7168.
- **V4-Flash:** 284B / 13B, 43 layers, d 4096. Report: arXiv 2606.19348, April 2026.
- **V4.1-Flash (2026-09):** 552B backbone + 196B of Engram tables. It activates 8B parameters per token in
  prefill and 16B in decode. 40 layers, d 5120.

**Verdict:** this is the most advanced published co-design of attention, residual stream, MoE, optimizer and
precision. Its report is also the most complete of any frontier lab. Every piece is built for 1M context and
huge scale, so little of it transfers to a dense model at our scale without simplification. The pieces that do
transfer are Muon, mHC, partial RoPE, QK-norm, MTP and document masking.

**Architecture:**
- **Attention (V4): CSA/HCA hybrid.**
  - Compressed sparse attention (CSA) pools every 4 tokens, then a lightning indexer picks the top-k (1024 for
    Pro).
  - Heavily compressed attention (HCA) pools every 128 tokens with no selection. CSA and HCA layers interleave.
  - The core is MQA with a single 512-dim head where K = V, plus low-rank queries.
  - Extras: a sliding-window branch (window 128), a learnable attention sink, and per-head RMSNorm on q and KV.
- **Attention (V4.1): CSA2.** Some layers reuse KV and top-k indices from earlier layers. The 40 layers are
  split into 20 encoder and 20 decoder layers (a causal encoder-decoder, CED), which roughly halves prefill.
  The KV cache is stored in FP4.
- **Positions:** partial RoPE on 64 dims plus inverse RoPE on the attention output; YaRN ×16 from 64k.
- **Residual stream:** mHC with 4 streams and 20 Sinkhorn iterations. V4.1 uses a single-pass mHC.
- **MoE:** 384 routed + 1 shared expert, top-6.
  - The router score is sqrt(softplus). Balancing uses aux-free biases plus a 1e-4 sequence-level loss.
  - V4 routes its first 3 layers by token-ID hash; V4.1 drops that.
- **Engram (V4.1):** hashed n-gram memory tables at layers 1 and 14.
- **MTP:** depth 1 in V4. V4.1 has no MTP in pretraining; its drafter is trained afterwards.

**Training (V4):**
- **Optimizer:** Muon (momentum 0.95, WD 0.1, update RMS rescaled to 0.18, Nesterov, 10 hybrid Newton-Schulz
  steps). AdamW (0.9 / 0.95, ε 1e-20) for embeddings, head, norms and mHC biases.
- **LR:** 2000 warmup steps, constant at 2.7e-4 (Pro 2.0e-4), then cosine to 1/10 near the end.
- **Batch and length:** batch ramps up to 75–95M tokens. Sequence length goes 4K → 16K → 64K → 1M; sparse
  attention starts at 64K after an indexer warmup.
- **Losses and masking:** MTP loss weight 0.3, then 0.1. Document masking; fill-in-the-middle (FIM).
- **Stability:** SwiGLU clamped (linear part to [−10, 10], gate capped at 10). "Anticipatory routing" switches on
  at loss spikes.
- **Precision:** FP8 training. FP4 QAT only in post-training.
- **V4.1 changes:** 45T tokens; batch fixed at 100.6M; head-wise Muon on Q and K; a Sinkhorn-balanced momentum
  for embeddings, head and Engram.
- **Not disclosed:** the data mix. No V4.1 base model is released.

**What we'd need:**
- *Architecture:* MoE + expert parallelism; the `mla` stub made real (low-rank MQA); a compressed/sparse
  attention kernel with an indexer; SWA + attention sinks; partial RoPE; mHC; MTP; YaRN. For V4.1 also
  cross-layer KV reuse, CED and Engram.
- *Training:* Muon (needs the multi-param-group fix from the switch list); FP8; batch ramp; document masking;
  staged context extension.

## Kimi (Moonshot): 5/5

**Latest releases:**
- **K3:** 2.78T total / 104B active, 93 layers, d 7168, 1M context, weights in July 2026.
- **K3.1-Pro (2026-11):** the same model post-trained, with its 24 global layers swapped from MLA to gated DSA.
  How that swap was trained is not disclosed, and there is no report.
- **Kimi Linear 48B-A3B (Oct 2025):** the small research version, 20 KDA + 7 MLA layers.

**Verdict:** this is the boldest frontier architecture. It drops positional encoding entirely (NoPE in every layer), replaces the
residual add with attention over block summaries, and runs 896 experts in a latent space. It sits level with
DeepSeek on design, but the training disclosure is thinner. Two findings are directly useful to us:
- the 3:1 KDA:MLA hybrid pattern;
- in their scaling-law study, cosine beat WSD once each schedule had its own hyperparameter search.

**Architecture:**
- **Layout:** 69 KDA (Kimi Delta Attention) + 24 gated MLA layers. Each block is 3 KDA + 1 MLA, and one extra
  global layer comes last.
- **KDA:** short conv (kernel 4), L2-normalized q and k, a lower-bounded decay, and a full-rank output gate.
- **Positions:** NoPE everywhere, so long-context extension needs no RoPE changes.
- **Residual: Block Attention Residuals.** Layers are grouped into blocks of 12. Each sublayer attends over
  block summaries instead of adding to a plain residual.
- **MoE ("Stable LatentMoE"):** 896 experts with 16 active, run in a d/2 latent space. 2 shared experts and
  1 dense first layer.
  - Activation is SiTU-GLU, a GLU with tanh-bounded gate and input.
  - The router is sigmoid. Balancing uses "Quantile Balancing" instead of a sign-step bias update.
- **MTP:** 1 layer during pretraining, later converted into a drafter. The released config has none.

**Training:**
- **Optimizer:** per-head Muon on Q/K/V, plus the weight clipping from K2 (MuonClip / QK-clip).
- **Schedule:** cosine with 1% warmup, WD 0.1.
- **Context:** 8K → 64K during pretraining, then 256K → 1M in the cooldown.
- **Precision:** FP8 block-wise activations; MXFP4/MXFP8 QAT from SFT onward.
- **Not disclosed:** tokens, batch size, peak LR, data mix. No base model is released.

**What we'd need:**
- *Architecture:* a real `gated_deltanet`/KDA layer using FLA kernels (our `mix: every_4th` pattern already
  gives 3 linear + 1 attention); MLA; a NoPE switch; output-gated attention; AttnRes; latent MoE; SiTU-GLU; MTP.
- *Training:* per-head Muon + QK-clip; FP8; staged context.

## Qwen (Alibaba): 4/5

**Latest releases (August 2026):**
- **Qwen3.8-2.4T-A95B:** 92 layers, d 8192.
- **Qwen3.8-27B:** the latest dense model, 64 layers, d 5120.
- **Qwen3.8-Flash-Next:** 125B total / 6B active, an "experimental preview" of the architecture planned for
  Qwen4.

**Verdict:** the production line is conservative and proven. Qwen3-Next → 3.5 → 3.8 all use 3:1 Gated
DeltaNet + gated attention, with QK-norm and a zero-centered RMSNorm. It is the only family that also ships
this hybrid as a *dense* model. Flash-Next is the interesting part, and it has a real report: Muon, a widened
"Gated Residual", n-gram embeddings and block-sparse QSA. The flagship's training is undisclosed.

**Architecture (3.8):**
- **Layout:** 23 × (3 Gated DeltaNet + 1 gated attention), each followed by MoE.
- **Gated attention:** 64 q / 4 KV heads × 256; partial RoPE (25%, θ 1e7); per-head QK-norm; sigmoid output
  gate.
- **Norms:** zero-centered RMSNorm (weight stored as w, applied as 1 + w).
- **MoE:** 512 experts, 10 routed + 1 shared expert with its own sigmoid gate.
- **Other:** 1 MTP layer; vocab 248,320; 262K native context.

**Flash-Next:**
- **QSA:** keys are pooled into blocks of 4 and an indexer selects 2048 tokens. It was added only during
  256K-context continued pretraining: about 2B tokens of indexer KL warmup, then about 200B tokens of joint
  training.
- **Gated Residual:** 4 residual branches with sigmoid read gates; this replaces pre-norm.
- **N-gram embeddings:** a 51B-parameter table at layer 2.
- **They kept RoPE:** NoPE caused "endless generation" after post-training.
- **Optimizer:** Muon (Nesterov 0.95, update scale 0.2·√max(A,B), 8 Newton-Schulz steps, fused projections split
  per head). AdamW for embeddings, head, router and gates.
- **Batch and stability:** no batch ramp ("cost 18.8% more steps for no gain"); LR and batch from a fitted
  scaling law; gradient clipping 0.5; no loss spikes without QK-clip.

**What we'd need:**
- *Architecture:* real `gated_deltanet` + `mix`; output gate; QK-norm (port it); zero-centered norm; partial
  RoPE; MTP; MoE for the flagship.
- *Training:* Muon with per-head splitting (Flash-Next); scaling-law-fitted LR and batch.

## Gemma (Google): 4/5

**Latest releases:**
- **Gemma 4 (2026-03-31):** E2B, E4B, 26B-A4B (MoE) and 31B dense (60 layers, d 5376).
- **Gemma 4 12B Unified (2026-06-03):** takes raw image patches and audio directly, with no encoders.

**Verdict:** the best design for small dense models. It is clever about the KV cache:
- local:global is 5:1 (4:1 on E2B), with a 512/1024 window and the last layer global;
- global layers differ from local ones: head dim 512 vs 256, far fewer KV heads (4 vs 16 on 31B), K = V, and
  p-RoPE on 25% of dims.

It is weak as a reference because training is nearly undisclosed. The MatFormer, AltUp and LAuReL pieces from
Gemma 3n are gone.

**Architecture:**
- **Blocks:** sandwich RMSNorm (pre and post, around both attention and FFN); QK-norm plus V-norm; GeGLU.
- **Embeddings and logits:** final-logit soft-cap 30; tied embeddings scaled by √d; vocab 262,144.
- **26B MoE:** a dense GeGLU branch runs in parallel with 128 experts (top-8) in every layer.
- **E2B/E4B:** per-layer embeddings (256 dims per layer, about 2.3–2.8B parameters) and KV sharing across the
  last layers.
- **Drafter:** a separate MTP drafter that cross-attends to the main model's KV.

**Training:**
- "Similar pre-training as Gemma 3"; TPU counts; QAT checkpoints.
- **Not disclosed:** tokens, optimizer, schedule, teacher.
- **Gemma 3 precedent (not confirmed for Gemma 4):** distillation from 256 sampled teacher logits per token;
  14T tokens for 27B.

**What we'd need:**
- *Architecture:* per-layer-type attention: SWA, a different head dim and KV-head count on global layers, K = V,
  p-RoPE. Also sandwich norm, QK- and V-norm, GeGLU (one more FFN target), soft-cap, √d-scaled tied embeddings.
- *Training:* a logit-distillation pipeline.

## GLM (Zhipu / Z.ai): 3/5

**Latest releases (August 2026):**
- **GLM-5.3:** 744B total / 40B active, 78 layers, d 6144. It shares a base with 5.2; every gain is from
  post-training.
- **GLM-5.3-Flash:** 320B / 18B, 45 layers = 34 KDA + 11 DSA, trained as a new base on 30T multimodal tokens.

**Verdict:** a strong follower. GLM-5.x is DeepSeek V3.2's design (MLA + DSA + sigmoid aux-free MoE + MTP) plus
IndexShare, where one indexer serves four layers (21 of 78 layers keep one). 5.3-Flash combines Kimi's KDA with
DeepSeek's mHC. Their original contributions are training findings:
- **Muon Split:** per-head Muon on MLA up-projections. It brings MLA up to GQA-8 quality and keeps attention
  logits stable "without any clipping".
- **Cosine over WSD:** GLM-4.5 preferred cosine to WSD.

**Architecture (5.3):**
- **MLA:** 64 heads, qk dim 256 (192 without position + 64 with RoPE).
- **No QK-norm:** RMSNorm on the latents instead.
- **DSA:** top-2048.
- **MoE:** first 3 layers dense, then 256 + 1 experts, top-8, sigmoid aux-free routing.
- **Context:** RoPE θ 8e6 for 1M context.

**Training:**
- **Tokens (GLM-5):** 28.5T. 27T pretraining at 4K, then 32K (1T), 128K (500B), 200K (50B).
- **Optimizer:** Muon Split.
- **MTP:** 3 shared layers.
- **DSA adaptation:** an indexer warmup, then 20B tokens of sparse adaptation.
- **Inherited from GLM-4.5:**
  - Muon (5 Newton-Schulz steps, momentum 0.95, update RMS 0.2).
  - Cosine from 2.5e-4 to 2.5e-5.
  - Batch 16M → 64M over the first 500B tokens; WD 0.1.
  - MTP weight 0.3 → 0.1.
- **Not disclosed for GLM-5:** schedule, batch, precision.

**What we'd need:** MLA, DSA + indexer KL warmup, MoE, MTP and Muon Split. 5.3-Flash would also need KDA, mHC
and the SwiGLU clamp.

## Mistral: 3/5

**Latest releases:**
- **Large 4 (weights 2026-10-29):** 1.05T total / 52B active, 72 layers, d 6144, and a new 204,800 vocab.
- **Medium 3.5 (2026-05):** 128B dense with the Llama-style shape of Large 2411. Training undisclosed.
- **Small 4 (2026-03):** 119B MoE using MLA.
- **Ministral 3 (2025-12):** dense 3/8/14B.

**Verdict:** Large 4 is DeepSeek's design (MLA + DSA + MoE with a shared expert) plus extras:
- NoPE every 4th layer;
- YOCO KV reuse;
- LeRoPE (learnable per-frequency RoPE scaling);
- SSMax (a position-dependent query scale);
- a logit soft-cap.

It is competent but derivative. Two parts of the report are valuable:
- **A negative result:** they first pretrained a 1 full : 3 SWA design with cross-layer KV sharing. They dropped
  it because SWE-bench fell even at matched perplexity, an "agentic penalty".
- **Ministral 3's cascade distillation:** prune 24B → 14B → 8B → 3B, training after each step with pure
  forward KL on teacher logits. Pure KL beat mixing in the next-token loss.

**Training (Large 4):**
- **Compute:** 3,840 GB200 for 45 days at 36% MFU.
- **Optimizer:** AdamW with β1 = β2 = 0.975. LR and batch come from log-linear predictors.
- **Batch and precision:** batch ramp 40M → 80M → 160M (unit probably tokens); FP8.
- **Data:** two phases, with code rising to 39% and sequence length going 4K → 40K in phase 2.
- **DSA adaptation:** a 15B-token indexer warmup, then 198B tokens of joint training.
- **Post-training:** Muon is used only for RL.
- **Not disclosed:** tokens, peak LR, schedule shape, weight decay.

**What we'd need:**
- *Large 4:* MoE, MLA, DSA, NoPE layers, YOCO, soft-cap, FP8, batch ramp.
- *Ministral 3:* only YaRN plus a pruning and distillation pipeline. It is otherwise our stack.

## OLMo (Ai2): 3/5, Olmo Hybrid 4/5

**Latest releases:**
- **Olmo 3 32B (2025-11):** the flagship. Olmo 3.1 changes only post-training.
- **Olmo 3 7B:** MHA, not GQA.
- **Olmo Hybrid 7B (2026):** 3 GDN : 1 attention.

**Verdict:** a solid but conservative dense design. Every config, data mix and training script is public, so
it is the best baseline to reproduce at our scale. Olmo Hybrid is the most relevant open model to our attention
registry: GDN in a 3:1 pattern, trained fully openly. It also drops RoPE from all layers (DroPE) before the
long-context stage.

**Architecture (32B):**
- **Attention:** GQA 40/8, SWA with a 4096 window on 3 of every 4 layers, the last layer full.
- **Norms:** "reordered" norm with no input norm: `h = x + RMSNorm(Attn(x))`. QK-norm over the full q and k
  width.
- **Positions:** RoPE θ 5e5, with YaRN only on the full-attention layers.
- **Other:** z-loss 1e-5; 65K context.

**Training:**
- **Optimizer:** AdamW (0.9/0.95, WD 0.1, no WD on embeddings) with update skipping on 6σ loss or grad-norm
  spikes; gradient clipping 1.0.
- **Pretraining:** 5.5T tokens. The cosine was planned for 5.93T and truncated. Peak LR 6e-4, batch about 8.4M
  tokens.
- **Mid-training:** two 100B-token runs with different data orders, linear decay to 0, weights averaged.
- **Long-context stage:** 100B tokens at 65K with document masking; the final checkpoints are averaged.
- **Data:** a repetition filter on training instances; a published mix (Common Crawl 76%, PDFs 14%, code 7%,
  math 3%).
- **Compute:** bf16, 1024 H100s, about 41% MFU.

**What we'd need:**
- *Architecture:* reordered norm, QK-norm, per-layer SWA, z-loss, YaRN on full-attention layers only, document
  masking, and context parallelism for 65K.
- *Training:* no WD on embeddings; skip-step updates; checkpoint averaging; truncated cosine plus anneal stages.
- *Hybrid adds:* a real GDN with FLA kernels, and DroPE.

## SmolLM (Hugging Face): 2/5

**Latest releases:**
- **SmolLM3-3B (2025-07):** there is no SmolLM4.
- **SmolLM2-135M/360M (2024-10):** still the latest sub-1B models.

**Verdict:** Llama-style with two additions: NoPE every 4th layer and intra-document masking. It skips QK-norm
on purpose, citing a paper that found QK-norm hurts long context. The architecture is unremarkable, but the
recipe and the playbook are the most honest small-scale record:
- Muon and AdEMAMix diverged at 3B;
- a TP seeding bug forced a restart after 1T tokens;
- unshuffled sequences caused loss spikes.

**Architecture:** GQA 16/4, 36 layers, d 2048, tied embeddings, 128k vocab.

**Training:**
- **Optimizer:** AdamW (0.9/0.95, WD 0.1, no WD on embeddings), gradient clipping 1.0, bf16.
- **Schedule:** WSD with peak 2e-4 and 2000 warmup steps; linear decay over the last 10%.
- **Tokens and batch:** 11.2T tokens, 2.36M-token batch, sequence length 4096.
- **Curriculum (web/code/math):** 85/12/3 → 75/15/10 → 63/24/13.
- **Long context:** two 50B-token stages, 4K → 32K → 64K, with RoPE θ raised to 2M and then 5M. YaRN to 128K
  at inference.
- **Post-training:** 140B tokens of reasoning mid-training. The final model is a linear merge of a soup of
  checkpoints.

**What we'd need:**
- *Architecture:* a per-layer NoPE switch, document masking (varlen attention), tied embeddings.
- *Training:* no WD on embeddings, a WSD scheduler, a staged data mixer, per-stage RoPE θ, checkpoint merging.
- *SmolLM2-small:* only tied embeddings and WSD.

## What this means for our switch list

- **Muon:** 4 of 8 families pretrain with it (Qwen only in Flash-Next). Each of the four uses a per-head variant for attention
  projections:
  - Kimi: per-head Muon + clip;
  - GLM: Muon Split;
  - Qwen Flash-Next: per-head split;
  - DeepSeek V4.1: head-wise Muon on Q and K.

  This is the top item. It needs the multi-param-group fix first.
- **muP:** none of the eight reports we read uses muP. Hyperparameter transfer instead comes from Muon's
  RMS-matched update scale (DeepSeek, GLM, Qwen) or fitted scaling laws for LR and batch (Qwen Flash-Next,
  Mistral). That is worth weighing before building muP.
- **FP8:** used by DeepSeek, Kimi (activations) and Mistral. Only the large-scale labs use it.
- **MoE:** 7 of 8 families ship it; only Olmo and SmolLM are dense-only. Gemma's design (a dense branch alongside
  the experts) is the gentlest place to start.
- **QK-norm:** used by DeepSeek, Qwen, Gemma and Olmo. GLM replaces it with Muon Split and Kimi with QK-clip.
  Porting it is cheap.
- **H-net, token superposition, nGPT:** none of the eight ship these. The closest shipped ideas are the
  hash-indexed n-gram tables: DeepSeek V4.1's Engram and Qwen Flash-Next's n-gram embeddings.

**Added to the switch list after this survey:**
- **Document masking:** DeepSeek, SmolLM3, Olmo.
- **Output-gated attention:** Qwen, Kimi.
- **Wider residual streams (mHC / AttnRes / Gated Residual), tentative:** DeepSeek, Kimi, Qwen Flash-Next,
  GLM-Flash.
- **Indexer sparse attention, tentative:** DeepSeek, GLM, Mistral, Kimi K3.1, Qwen Flash-Next.

**Not on the switch list, but common enough to add:**
- **Partial RoPE / NoPE layers:** 7 of 8. A `rope_fraction` and a `no_rope_layers` setting in `build_attention`
  would cover them.
- **MTP:** DeepSeek, Kimi, Qwen, GLM.
- **3:1 linear-attention hybrid:** Kimi, Qwen, Olmo Hybrid, GLM-Flash. Our `gated_deltanet` stub plus
  `mix: {type: gqa, layers: every_4th}` already expresses this layout.
- **No WD on embeddings:** Olmo, SmolLM3.
- **z-loss:** Olmo.
- **Checkpoint averaging:** Olmo, SmolLM3.
- **WSD vs cosine:** Kimi and GLM found cosine better when tuned; SmolLM3 uses WSD.

## Open questions and conflicts in the sources

- **Olmo 3 32B long-context soup:** the report says the last 3 checkpoints are averaged; the model card says 4.
- **SmolLM3 32K-stage RoPE θ:** 2M in the config and playbook, 1.5M in the blog.
- **Kimi K3 MTP:** the report says K3 was pretrained with an MTP layer; the released config has 0 MTP layers.
- **Mistral Large 4 dates:** the weight-release date (2026-10-29) comes from press only (secondary).
- **Qwen3.8 global-batch balancing:** whether 3.5/3.8 use it is unverified.
- **Unreadable blogs:** the Qwen and Z.ai blogs render client-side and could not be read.

## Sources

**DeepSeek**
- V4 report: https://arxiv.org/pdf/2606.19348
- V4-Pro config: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/config.json
- V4-Flash config: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main/config.json
- V4.1-Flash model card: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
- V4.1 tech report: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main/DeepSeek_V41_Tech_Report.pdf
- V4.1-Flash config: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main/config.json

**Kimi**
- K3 report: https://github.com/MoonshotAI/Kimi-K3/raw/main/k3_tech_report.pdf
- K3 config: https://huggingface.co/moonshotai/Kimi-K3/resolve/main/config.json
- K3.1-Pro model card: https://huggingface.co/moonshotai/Kimi-K3.1-Pro
- K3.1-Pro blog: https://www.kimi.ai/blog/kimi-k3-1-pro
- Kimi Linear paper: https://arxiv.org/html/2510.26692
- K2 paper (MuonClip): https://arxiv.org/abs/2507.20534

**Qwen**
- Qwen3.8 flagship: https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B
- Qwen3.8 27B dense: https://huggingface.co/Qwen/Qwen3.8-27B
- Flash-Next model card: https://huggingface.co/Qwen/Qwen3.8-Flash-Next
- Flash-Next report: https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf
- Qwen3.5 modeling code (transformers): https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py

**Gemma**
- Gemma 4 report: https://arxiv.org/pdf/2607.02770
- 31B config: https://huggingface.co/google/gemma-4-31B/resolve/main/config.json
- 26B-A4B config: https://huggingface.co/google/gemma-4-26B-A4B/resolve/main/config.json
- Model card: https://ai.google.dev/gemma/docs/core/model_card_4
- Modeling code (transformers): https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gemma4/modeling_gemma4.py
- Gemma 3 report: https://arxiv.org/html/2503.19786v1

**GLM**
- GLM-5.3 model card: https://huggingface.co/zai-org/GLM-5.3
- GLM-5.3-Flash model card: https://huggingface.co/zai-org/GLM-5.3-Flash
- GLM-5 report: https://arxiv.org/abs/2602.15763
- IndexCache paper: https://arxiv.org/abs/2603.12201
- GLM-4.5 report: https://arxiv.org/abs/2508.06471

**Mistral**
- Large 4 model card: https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B
- Large 4 technical report: https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B/resolve/main/assets/technical_report.pdf
- Medium 3.5 model card: https://huggingface.co/mistralai/Mistral-Medium-3.5-128B
- Small 4 model card: https://huggingface.co/mistralai/Mistral-Small-4-119B-2603
- Ministral 3 paper: https://arxiv.org/html/2601.08584v1
- Large 4 weights date (secondary): https://techcrunch.com/2026/10/29/mistral-releases-weights-for-its-trillion-parameter-ai-model-targeting-cybersecurity/

**OLMo**
- Olmo 3 report: https://arxiv.org/abs/2512.13961
- Olmo 3 32B model card: https://huggingface.co/allenai/Olmo-3-1125-32B
- Olmo 3 training scripts (OLMo-core): https://github.com/allenai/OLMo-core/tree/main/src/scripts/official/OLMo3
- Olmo Hybrid report: https://arxiv.org/abs/2604.03444
- Olmo Hybrid 7B model card: https://huggingface.co/allenai/Olmo-Hybrid-7B

**SmolLM**
- SmolLM3 blog: https://huggingface.co/blog/smollm3
- SmolLM3 base model card: https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base
- SmolLM3 pretraining configs: https://github.com/huggingface/smollm/tree/main/text/pretraining/smollm3
- Smol training playbook: https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook
- SmolLM2 paper: https://arxiv.org/abs/2502.02737
