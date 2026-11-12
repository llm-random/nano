# Pretraining switches: status on pretraining-jm

As of 2026-11-10, commit `4847066b`.

## Summary

3 of the 15 exist on pretraining-jm today: RoPE, SwiGLU and GQA. The other 12 are
missing. QK-norm and MoE have working code on other branches to port; the rest
start from scratch. Items marked ? are tentative. The last four rows were added
after the survey in `open_model_architectures.md`.

| Item | Status | On pretraining-jm | Code elsewhere |
| --- | --- | --- | --- |
| Muon | Missing | AdamW is hard-coded in `main.py` | None found |
| muP | Missing | Fan-in init only, no width multipliers | Related simpleP config on `origin/ctx_scl-simpleP` |
| FP8 | Missing | torchao 0.11.0 is a dependency, never imported | None found |
| MoE | Missing | Removed in the 2026-10-30 reset | `src/core/moe.py` on `main`, `origin/ctx_scl` |
| H-net | Missing | No code | None found |
| Token superposition | Missing | No code | None found |
| nGPT (normalized activations) | Missing | No code | None found |
| RoPE | Exists | Always on; base and llama-3 scaling configurable | |
| QK-norm | Missing | No code | `pre_attn_fn` on `origin/ctx_scl`; `origin/ame_improve` |
| SwiGLU | Exists | Default FF; GELU MLP via `_target_` swap | |
| GQA | Exists | Default; `common.attention.type` picks gqa, mha or mqa | |
| Document masking | Missing | Documents are joined with EOS and attend across each other | Compiled `flex_attention` wrapper in `src/core/ame.py` on `origin/ame_improve` |
| Attention output gate | Missing | No code | None found |
| Wider residual stream? | Missing | One residual stream, in `Residual` | None found |
| Indexer sparse attention? | Missing | No code | None found |

Other branches were checked by keyword grep over their Python files, not read in full.

## Already present

GQA is the only one with a real switch; RoPE has no off switch and SwiGLU is
switched by swapping a Hydra `_target_`.

| Item | Where | How it is switched today |
| --- | --- | --- |
| RoPE | `RoPE` and `RoPEAttention` in `src/core/model.py` | Always on: every attention type in `src/core/attention.py` builds a `RoPEAttention`. `attention_fn.rope_base` (10000 in H-balanced) and `rope_scale_freqs` (llama-3 frequency scaling, off) are configurable. A NoPE ablation needs a new flag. |
| SwiGLU | `GatedFeedForward` in `src/core/model.py` | Default `ff_layer_fn` in `configs/_model/h_balanced/base.yaml`. The alternative is the GPT-2 GELU MLP: set `ff_layer_fn._target_=src.core.model.GeluFeedForward` and drop `gate_fn`, as in the comment in `configs/mfu_audit/h20_mgpu.yaml`. |
| GQA | `ATTENTION_TYPES` registry in `src/core/attention.py` | `common.attention.type`: `gqa` (default, kv_heads = H/2), `mha` or `mqa`. An optional `common.attention.mix` swaps the type on every 2nd, every 4th or listed layers. |

## Missing

Muon and muP share one blocker: the optimizer is a single AdamW param group, built
twice in `main.py` (lines 208 and 286), and `get_cosine_scheduler_with_warmup`
asserts exactly one param group (`src/core/schedulers.py:65`).

| Item | What a switch would touch | Start from |
| --- | --- | --- |
| Muon | Optimizer construction in `main.py`: Muon on hidden 2D matrices, AdamW on embedding, head and norms. Scheduler must accept several groups. Under FSDP2 the Newton-Schulz step needs full matrices, so gather or use a distributed Muon. | Nothing in the repo |
| muP | Init is `llm_random_weight_init` (std = scale/sqrt(fan_in)), i.e. standard parametrization. Needs width-dependent init and per-group LR multipliers keyed on dmodel = 64·H, so the same param-group work as Muon. Its LR rules differ under Muon. | `configs/_parameterization/simpleP_k12.yaml` on `origin/ctx_scl-simpleP` (a related parametrization, not muP) |
| FP8 | torchao `float8` conversion of the `Linear` layers in `main.py`, before FSDP2 wrapping, as a top-level flag like `compile_blocks`. Needs Hopper/Ada GPUs. | `torchao==0.11.0` already in `pixi.toml` |
| MoE | A new `ff_layer_fn` target (the SwiGLU slot) plus the aux-loss logging that was removed from `trainer.py`. The reset log plans a rewrite with efficient granularity kernels. | `src/core/moe.py` on `main` (token-choice, capacity factor, 169 lines); `documents/notes/pretraining_reset_log.md` |
| H-net | Byte-level input with learned dynamic chunking: replaces the GPT-NeoX tokenizer in the data pipeline and the flat `LLM` with encoder, main network and decoder. Loss per token stops being comparable, so eval needs bits per byte. The largest change on the list. | [goombalab/hnet](https://github.com/goombalab/hnet) |
| Token superposition | Not assessed: the paper hasn't been read yet, so what it touches is an open question. | [arXiv 2605.06546](https://arxiv.org/abs/2605.06546) |
| nGPT | Unit-norm weights and hidden states: renormalize weights after every optimizer step, replace `RMSNorm` + `Residual` with normalized updates and learned step sizes, add a learned logit scale in `TransformerHead`. nGPT drops weight decay and warmup. | Nothing in the repo |
| QK-norm | A `pre_attn_fn` (or `qk_norm`) argument on `RoPEAttention`, threaded through the shared `build_attention` contract so every attention type gets it. Pick ℓ2 + learned temperature or RMS; the AME notes warn against mixing them. | `RoPEAttention(pre_attn_fn=...)` on `origin/ctx_scl`; `tests/test_qk_norm.py` on `origin/ame_improve`; `documents/notes/ame_implementation.md:101` |
| Document masking | `src/core/datasets.py` only appends EOS, so the dataloader must also emit document boundaries. `attention_mechanism` in `src/core/model.py` calls SDPA with `is_causal=True` and no mask; it needs a FlexAttention document block mask (compiles) or flash-attn varlen with `cu_seqlens`. Optionally reset RoPE positions per document. | SmolLM3 (`_use_doc_masking`) and Olmo-core long-context configs; the flex wrapper on `origin/ame_improve` |
| Attention output gate | In `RoPEAttention.forward`, multiply the attention output by `sigmoid(W_g x)` before `o_proj`. One extra dmodel → q_heads·dhead projection (Qwen fuses it into a doubled `q_proj`); a per-head scalar gate is the cheap variant. Thread a flag through `build_attention`, as for QK-norm. | Qwen3.5 modeling code in transformers; Qwen's "Gated Attention for LLMs" paper |
| Wider residual stream? | Replace the single stream carried through `Residual` and `TransformerEncoder` with n streams (DeepSeek and GLM use 4). Expand after the embedding, collapse before the head, and give each sublayer learned read and write mixing; mHC keeps the mixing matrix doubly stochastic with Sinkhorn iterations. Activation memory grows ×n. Collides with nGPT (both replace `Residual`) and with the residual-norm metrics logged there. | DeepSeek-V4 `inference/model.py` (mHC); Kimi K3 AttnRes; Qwen3.8-Flash-Next Gated Residual (report) |
| Indexer sparse attention? | A new type in `ATTENTION_TYPES` (e.g. `dsa`): a small indexer (few heads, ReLU scores) picks the top-k keys per query and attention runs on those only. Needs a gather/sparse kernel, which SDPA can't do. Training: dense warmup, an indexer KL warmup against dense attention, then sparse. `src/core/mfu.py` counts dense score FLOPs per attention module, so MFU would need a sparse-aware count. At our 256–1024-token sequences a top-2048 selection keeps every key, so it pays off only for long-context runs or a much smaller k. | DeepSeek-V3.2 report (arXiv 2512.02556); `documents/notes/long-context-attention.md` §6 |

## Open questions

- Should RoPE get a real on/off switch (NoPE), or does configurable RoPE count as done?
