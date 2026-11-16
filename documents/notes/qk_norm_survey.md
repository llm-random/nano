# QK-norm survey (line 1, task 1)

Latest open release of each family as of 2026-11-16. Read from config.json and modeling code
only (transformers main at `696c46ba` unless the row says otherwise; `TF` = its `src/transformers/models/`).
"Before RoPE, kv" means q and k are normalized before RoPE, k on its kv heads before the repeat.

| Model (HF release) | QK-norm: operator, over | Gain | Placement, softmax scale | Source |
| --- | --- | --- | --- | --- |
| Qwen3.8-27B (2026-08); Flash-Next and 2.4T the same | RMSNorm per head over head_dim, eps 1e-6 | `1+w`, w=0, [256] | before RoPE, kv; 1/√d | [config](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json), [code](https://github.com/huggingface/transformers/blob/696c46bacf291909624c4f8d62e8016281aa762e/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L742-L852) |
| MiniMax-M3.1-Flash (2026-10) | same as Qwen (`qk_norm_type: per_head`; M2 was whole-projection) | `1+w`, w=0, [128] | before RoPE, kv; 1/√d | [config](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/raw/main/config.json), [code](https://github.com/huggingface/transformers/blob/696c46bacf291909624c4f8d62e8016281aa762e/src/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py#L397-L479) |
| Gemma 4 31B (2026-03; the 12B of 2026-05 the same) | RMSNorm per head, eps 1e-6 (also a gainless v norm) | `w`, ones, [256] / [512] | before RoPE, kv; **1.0** | [config](https://huggingface.co/google/gemma-4-31B/raw/main/config.json), [code](https://github.com/huggingface/transformers/blob/696c46bacf291909624c4f8d62e8016281aa762e/src/transformers/models/gemma4/modeling_gemma4.py#L1153-L1264) |
| Olmo 3.1 32B (2025-12; no newer plain Olmo) | RMSNorm over the **whole** q / k projection, eps 1e-6 | `w`, ones, [5120] / [1024] | before head split and RoPE; 1/√d | [config](https://huggingface.co/allenai/Olmo-3.1-32B-Instruct/raw/main/config.json), [code](https://github.com/huggingface/transformers/blob/696c46bacf291909624c4f8d62e8016281aa762e/src/transformers/models/olmo3/modeling_olmo3.py#L137-L211) |
| DeepSeek-V4.1-Flash (2026-09) | **k only**: RMSNorm over the one shared 512-d K=V head; V4-Pro adds a gainless per-head q norm | `w`, ones, [512] | before RoPE; 1/√512, plus sinks | [config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main/config.json), [code](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/model.py#L640-L707) |
| Kimi-K3.1-Pro (2026-11) | none; MLA latent norms only, QK-Clip in the optimizer (K2 report) | – | 1/√192 | [config](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/resolve/main/config.json), no public code, [K3 code](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py#L335-L475) |
| GLM-5.3 (2026-08) | none; MLA + DSA latent norms, "Muon Split" keeps logits stable (GLM-5 report) | – | 1/√256 | [config](https://huggingface.co/zai-org/GLM-5.3/raw/main/config.json), [code](https://github.com/huggingface/transformers/blob/696c46bacf291909624c4f8d62e8016281aa762e/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py#L299-L453) |
| Mistral-Large-4 1T-A52B (2026-09) | none; MLA latent norm, learned log-position q temperature, AdamW | – | 1/√128 × SSMax | [config](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B/resolve/main/config.json), [code](https://github.com/huggingface/transformers/blob/696c46bacf291909624c4f8d62e8016281aa762e/src/transformers/models/mistral_large4/modeling_mistral_large4.py#L331-L497) |
| Llama-4-Maverick (2025-04; config gated) | none ("the 128E model does not use qk_norm"); Scout: gainless per-head RMS after RoPE | – | 1/√128 | [code](https://github.com/huggingface/transformers/blob/696c46bacf291909624c4f8d62e8016281aa762e/src/transformers/models/llama4/modeling_llama4.py#L302-L393) |
| gpt-oss-120b (2025-08) | none; learned per-head attention sink, all layers | – | 1/√64 (×1.81 YaRN) | [config](https://huggingface.co/openai/gpt-oss-120b/raw/main/config.json), [code](https://github.com/openai/gpt-oss/blob/277020131d524f06bed5f66b694a92ed7f6e5224/gpt_oss/torch/model.py#L153-L247) |
| Nemotron-3.5-Lightning-30B-A3B (2026-08) | none; NoPE GQA in 6 of 52 blocks of a Mamba-2 hybrid | – | 1/√128 | [config](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/raw/main/config.json), [code](https://github.com/huggingface/transformers/blob/696c46bacf291909624c4f8d62e8016281aa762e/src/transformers/models/nemotron_h/modeling_nemotron_h.py#L870-L924) |
| MiMo-V2.6-Flash (2026-09) | none; learned sink on the sliding-window layers | – | 1/√192 | [config](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD/raw/main/config.json), [code](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD/blob/main/modeling_mimo_v2.py#L220-L391) |

Linear-attention layers (Qwen, Kimi, GLM-Flash and Olmo hybrids) ℓ2-normalize q and k per head with no
gain; that is the delta rule's requirement, not softmax QK-norm, so it doesn't count above.

**Models without QK-norm, and whether their substitute transfers to our GQA `RoPEAttention`:**
- MLA models (Kimi, GLM, Mistral) never materialize per-head k at inference, which is why they skip
  QK-norm. Their substitutes live in the optimizer: QK-Clip rescales W_q, W_k by the per-head max logit
  (SDPA doesn't return it), and Muon Split only means something under Muon. Neither fits AdamW + SDPA.
- Sinks (gpt-oss, MiMo, DeepSeek) are an extra softmax logit per head; SDPA can't express one without
  a float mask that drops the flash backend. Mistral's SSMax is a long-context temperature.

## Pick

Per-head RMSNorm over `dhead` with one `[dhead]` gain shared across heads, eps 1e-6, on q and k
before RoPE, k on its kv heads before `repeat_kv`, softmax scale 1/√dhead (SDPA's default, so
`attention_mechanism` gets no `scale` argument). Gain `(1 + w)` with w zero-initialized.
- **Operator:** 3 of the 4 models with full QK-norm (Qwen, MiniMax, Gemma); Olmo alone norms the whole
  projection, and MiniMax moved from that (M2) to per head (M3). All five norms sit before RoPE.
- **Scale:** 4 of 5 keep 1/√d; Gemma 4 alone uses 1.0.
- **Gain:** `1+w` as in Qwen3.8 and MiniMax-M3.1 (Gemma 4 and Olmo use `w`). Same forward, and the same
  AdamW trajectory without weight decay. Under h12_chinchilla's one AdamW group (decay 0.1 on every
  parameter), decay pulls `1+w` toward 1 instead of pulling a `w` gain, and so the logit scale, toward 0.
- MFU doesn't argue against it (below).

## MFU estimate (h12 on one H100 SXM, no GPU runs)

- h12: dmodel 768, 12 q / 6 kv heads of 64, seq 1024, micro 64×1024 tokens per GPU. A block is
  8.85M matmul parameters, 62.5 MFLOP per token in training; at 35% MFU its fwd+bwd per micro-step
  takes ~11.8 ms. `mfu.py` counts the 1-D gains as nothing, so all QK-norm time is lost MFU.
- QK-norm is memory-bound on 768 + 384 bf16 values per token per block. Eager it makes ~5 passes
  (fwd read+write; bwd read grad, read saved input, write grad): 755 MB per block per micro-step,
  ~0.25 ms at 3 TB/s, **~2.1%**.
- Under `compile_blocks_full`: torch 2.7.1 decomposes `aten.rms_norm`, so Inductor sees a reduction
  over 64 contiguous values plus pointwise ops. The normalize-and-scale inlines into the RoPE kernel
  (CPU Inductor, forward: 5 → 6 kernels, the extra one the per-row rstd reduction); no reshape or copy,
  since the norm runs on the transposed view, whose last dim stays contiguous. Backward adds the row
  reductions and two gain-gradient column sums: ~2–3 passes, **~1%** of block time.
- **Estimate: MFU drops ~1% relative (35% → ~34.6%), at most ~2% if Inductor doesn't fuse it.**
- Alternatives: k after `repeat_kv` doubles the k work (~2.8% eager) for the same math with a shared
  gain. A whole-projection norm moves the same bytes (one reduction over 768 / 384 instead of 64) and
  needs no communication without tensor parallelism, so MFU doesn't separate it from per head.
- Memory: pre-norm q and k are saved for backward, about +1.7 GiB per GPU at micro 64 × 12 blocks.
