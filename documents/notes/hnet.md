# H-net: scoping note

**Decision 2026-11-12: abandoned.** At our Chinchilla budget a likely tie isn't worth the cost.

Line 5 of the pretraining-switches plan, written 2026-11-12 on `pretraining-jm` at `581e2f73`.
Sources: the paper, arXiv 2507.07955v2 (§ refers to its sections), and the code at
goombalab/hnet `main` (paths `hnet/...`). No code was written for this note.

**Summary.** The premise holds: H-net replaces the tokenizer and the flat `LLM`, and it needs
bits-per-byte eval. It still fits next to `LLM` as a separate model, not as a fork, because its
main network is an ordinary Transformer stack and our h16 blocks work unchanged. What is new is a
byte-level outer stage made of Mamba-2 layers, plus the chunking module.

Recommendation: a reduced version, built after lines 1–3 land. At h16_chinchilla's data budget the
paper points to a tie for a 1-stage model, so the decision is whether a tokenizer-free tie, or a bet
on the 2-stage model, is worth ~800 lines and a Mamba-2 dependency.

## 1. What H-net changes

An encoder E reads bytes (vocab 256). A router marks boundaries, and only boundary positions,
about 1 byte in 5, enter the main network M, which is a normal LM stack. M's outputs are
smoothed, expanded back to bytes, and added to a projected residual from E. A decoder D then
predicts the next byte (§2.1, Eqs. 1–3). Everything trains end to end on byte cross-entropy
plus a ratio loss. M can itself be an H-net, which gives the 2-stage model.

- **Router** (§2.2.1, Eq. 4; `hnet/modules/dc.py: RoutingModule`): q_t = W_q x̂_t,
  k_t = W_k x̂_t, p_t = ½(1 − cos(q_t, k_{t−1})), b_t = 1{p_t ≥ 0.5}, p_1 = 1. W_q and W_k start
  as the identity. The downsampler keeps x̂_t where b_t = 1; pooling was no better (App. E.1).
- **Smoothing** (§2.2.2, Eqs. 5–9; `DeChunkLayer`): an EMA over M's outputs,
  z̄_t = P_t ẑ_t + (1 − P_t) z̄_{t−1}, run with the Mamba-2 scan kernel. The upsampler repeats
  each chunk up to the next boundary and multiplies by STE(c_t). Here c_t is p_t at a boundary
  and 1 − p_t elsewhere; STE makes it 1 in the forward pass and keeps its gradient. Without
  smoothing, compression oscillates and BPB degrades (§3.3, Fig. 7).
- **Ratio loss** (§2.2.3, Eq. 10; `hnet/utils/train.py: load_balancing_loss`):
  N/(N−1)·((N−1)FG + (1−F)(1−G)), where F = mean b_t (not differentiable) and G = mean p_t.
  The total loss is L_AR + 0.03·Σ_s L_ratio, with α = 0.03 in every run. The target is N = 6
  for 1-stage and (3,3) for 2-stage. The learned 1-stage chunk averages 4.8 bytes at L and 4.7
  at XL, close to GPT-2's 4.6 bytes/token (Table 1). It is computed per device micro-batch.
- **Layers** (§2.1.2, §3, Table 1): E and D are 4 Mamba-2 layers each with no MLP, and M is a
  SwiGLU Transformer. Widths are monotone (D⁰ ≤ D¹), joined by padding with a learned vector and
  by truncation. Other parts:
  - an RMSNorm at the end of each of E, M and D;
  - a zero-initialized fp32 residual Linear (§2.3);
  - LR multipliers per stage (App. C, Eq. 11).

  For scale, the 1-stage L model is M4 + T22 + M4 at 1024/1536 (680M), against a BPE T24 at 1536
  (760M).
- **Transformer-only** trains, but loses. Of four E/D layouts at L, attention-only (T2-T2) is by
  far the worst. It compresses poorly, so it also spends more FLOPs in M, and results improve
  monotonically with more Mamba (§3.3, Fig. 8). The same holds on BPE inputs (Fig. 10), so
  Mamba-2 is load-bearing.

## 2. What it touches here

| Where | What H-net needs | Owner |
| --- | --- | --- |
| `src/core/attention.py:107` `Mamba2Mixer` (`mamba2`, planned) | The real mixer for E and D | line 1 |
| `src/core/model.py:359` `TransformerEncoder`, `:329` `TransformerBlock` | M reuses h16 blocks. E and D use MLP-free blocks built from `Residual` in `hnet.py` | line 1 |
| `src/core/model.py:49` `Residual` metrics | Its all-gather needs equal (batch, seq) on every rank, so M needs a fixed capacity | line 1 |
| line 1 task 4 `block_mask` path | Masks M's packed chunk row per sequence | line 1 |
| `src/core/datasets.py:90` packer | No change, since it packs any list of ints. Add `byte_tokenize_fn` (UTF-8 + EOS 255) next to `:38` | line 1 / H-net |
| `main.py:208,286` single-group AdamW; `src/core/schedulers.py:80` one-group assert (cosine only; WSD handles groups) | Per-stage LR needs groups by stage × role | line 2 |
| `src/core/trainer.py:580` `_forward_loss`, `:622` `calculate_loss`, `:746` `log_metrics` | Model returns (logits, ratio losses). Add α·L_ratio and log L¹/L⁰ | line 3 aux-loss plumbing |
| FFN slot, `moe.py` | M can take MoE as is (§4 "Sparsity") | line 3 |
| float8 conversion in `main.py` | Skip the router's q/k, the residual Linear and the byte head | line 6 |

Not planned by any line:
- `main.py:166,192`: the compile flags walk `model.encoder.blocks`.
- `configs/_trainer/llama.yaml:20`: the FSDP2 list interpolates `${model.encoder.block_fn._target_}`.
- `src/core/eval.py:33`: `NanoLM` is built around a tokenizer.
- `src/core/mfu.py:44` (owned by line 1): FLOPs are counted per input position.
- `trainer.py:743`: input positions are counted as tokens.

**Fit:** a separate model, not a fork.
- New: `src/core/hnet.py`, `configs/_model/hnet/`, `configs/_dataset/fineweb_bytes.yaml`, and a
  `ByteLM` beside `NanoLM`.
- Unchanged: the dataset class, holdout, packer, checkpointing and logging.

## 3. How to evaluate it against our runs

- **BPB.** For H-net, eval/loss / ln 2. For token runs, eval/loss / (ln 2 · B), with B the bytes
  per target token on the eval set.
  - On 3,000 FineWeb-Edu docs, B = 4.605 for GPT-NeoX. GPT-2 gives the same, matching the paper's
    4.6 (App. C).
  - Measure B on our holdout, counting EOS as 1 byte on both sides.
  - For identical text, decode each 1024-token eval window into the byte eval set.
  - Rescaling slightly overstates a token model's BPB, which favours H-net (§4 "BPB Calculation").
- **Downstream without a tokenizer.** A `ByteLM` that:
  - encodes with `tok_encode = list(s.encode())`, with EOT = 255;
  - counts `max_length` in bytes;
  - takes the logits from the (logits, aux) output.

  All nine `joint.yaml` tasks are loglikelihood tasks, and none of their metrics depend on the
  tokenizer:
  - acc sums log-probabilities;
  - acc_norm divides by character length (`lm_eval/api/task.py:1489`);
  - lambada's acc is a greedy check.

  Right-padding is safe because the router and the EMA are causal. The paper uses lm-eval the same
  way (§3.1, Table 2). Each request is about 4.6× longer.
- **Matching compute,** as the paper does (§3 "Experimental Setup"): equal bytes per batch and
  equal FLOPs per byte.
  - h16 costs 2.020 GFLOPs/token in `mfu.py`'s convention (6P plus scores), i.e. 0.439
    GFLOPs/byte. 15% of that is the 50k-vocab head, which H-net drops.
  - The matched H-net is E = M4 @ 512, M = 15 h16 blocks @ 1024, D = M4 @ 512, N = 6.
  - It costs 0.439 GFLOPs/byte at 4.8 bytes/chunk, with 251M params (h16 has 355M, 252M of them
    non-embedding). E and D take 24% of FLOPs, against ≈28% for the paper's L model.
  - An outer width of 768 would cost 1.26× and the paper's 1024 would cost 1.6×, so 512 is forced.
  - Data: 4608-byte sequences (≈1000 tokens, ≈960 chunks), batch 256, 30,700 steps, which is
    h16_chinchilla's 36.2B bytes.
- **Reference:** h16_chinchilla at its best LR.
  - Paper: at XL, the 1-stage crosses the BPE Transformer after 100B bytes, the 2-stage after
    30B (§3.1, Fig. 3).
  - Paper: at L, the 1-stage only ties after ~375B bytes (0.755 vs 0.756, Table 2).
  - We train 36B bytes, so expect the 1-stage to tie or trail.

## 4. What it costs

- **Size.** goombalab/hnet has model and inference code only: no training loop, no data pipeline.
  `hnet/utils/train.py` gives the ratio loss and the param groups. In lines of code:

  | Part | Lines |
  | --- | --- |
  | Port router, chunk, dechunk, STE and the stage wrapper, forward only | ~300 |
  | Mamba-2 mixer | ~80 |
  | Byte tokenizer and `ByteLM` | ~60 |
  | Hooks in trainer, mfu and main | ~80 |
  | Configs and tests | rest |
  | Total | ~800 |

  `mha.py`, `rotary.py` and `isotropic.py` are flash-attn blocks that our `TransformerEncoder`
  replaces. A parity test of our DC module against `dc.py` is cheap. Loading the HF checkpoints is
  not, since it needs hnet's own blocks.
- **Mamba-2 dependency.** hnet pins `mamba_ssm`, `causal_conv1d`, `flash_attn` and triton. We
  need only `mamba_ssm`:
  - PyPI has only an sdist (2.3.2.post1). By default it builds no CUDA extensions, because those
    are Mamba-1's; Mamba-2 runs on Triton.
  - Its `setup.py` imports torch, so pixi needs `no-build-isolation`.
  - einops is not in the lock.
  - The lock is pinned at v6 (`tests/test_repo_hygiene.py:655`): add entries by hand, then
    `pixi install --locked`.
  - The aarch64 env (torch 2.6) has no Triton in the lock, so H-net would be entropy-only.
  - The fused path needs `causal_conv1d`, a CUDA extension. Without it the conv falls back to
    `nn.Conv1d`, which has no varlen `seq_idx`. That is fine for plain (B, L⁰) rows.
  - Triton is CUDA-only, so CPU `tiny_local` needs a pure-PyTorch SSD and EMA anyway (the Mamba-2
    paper's ~40-line minimal SSD). I'd make that the default and `mamba_ssm` an optional fast
    path.
- **Data-dependent chunk lengths.**
  - The problem: `ChunkLayer` uses boolean or argsort selection and `int(...max())`, a host sync,
    so it graph-breaks. If M's length varied, `compile_blocks_full` would recompile every step.
  - The plan: keep DC eager. Pack the micro-batch's chunks into one row of fixed capacity, ~1.1×
    the expected count; a sum over 64 sequences varies far less than a per-sequence max. Mask that
    row per sequence with line 1's FlexAttention `block_mask`. M's shapes are then static, the
    same rule line 3 follows for MoE.
  - What fixed capacity also buys:
    - `Residual` gathers line up across ranks;
    - FSDP2 ranks stay balanced;
    - the paper's out-of-memory edge case goes away (§4 "Efficiency").
  - FSDP2 shards the Mamba and h16 blocks. The fp32 residual Linear needs its own mixed-precision
    policy, or it runs in bf16.
- **MFU.**
  - `mfu.py` counts each 2-D weight once per input position. For H-net that overcounts M by ~4.8×
    and misses the SSD scan, which is ~19% of a width-512 Mamba layer.
  - The right count is FLOPs/byte = E/D + (L¹/L⁰)·(M per chunk), with L¹/L⁰ logged each step.
  - `processed_tokens` becomes bytes.
  - Count real chunks, not the capacity, so padding shows up as lost MFU.
- **Throughput.** The authors' code, varlen kernels and all, is "approximately up to 2× slower than
  an isotropic model during training" (§4 "Efficiency"). E and D add width-512 matmuls over 4.6×
  the positions, plus a memory-bound scan. Expect lower MFU than h16, and measure it.
- **Risks.**
  - The paper's smallest LM is matched to 760M, and an outer width of 512 is untested.
  - At Chinchilla data the paper predicts a 1-stage tie.
  - There are new hyperparameters: α, N, and the per-stage LR (Eq. 11 gives ×3.0 for E and D,
    ×0.88 for M).
  - The paper saw no instability up to 1.3B but flags it for larger scale (§4 "Scale").
- **Reported gains** (FineWeb-Edu, ~375B bytes, FLOP-matched to 760M at L and to 1.3B at XL):

  | Scale | Model | BPB | Avg acc |
  | --- | --- | --- | --- |
  | L | BPE Transformer | 0.756 | 53.3 |
  | L | 1-stage | 0.755 | 53.6 |
  | L | 2-stage | 0.743 | 55.5 (= XL BPE) |
  | XL | BPE Transformer | 0.730 | 55.5 |
  | XL | 1-stage | 0.728 | 56.2 |
  | XL | 2-stage | 0.715 | 58.2 |

  The paper also reports much higher robustness on perturbed HellaSwag (Table 3) and larger gains
  on Chinese and code (§3.2, Table 4).

## 5. Recommendation and open questions

**Implement a reduced version**, once these land, because it reuses all three:
- line 1 task 4 (document masking);
- line 2 task 1 (param groups);
- line 3 (aux loss).

Scope:
- **In:**
  - stages written recursively, so 2-stage is a config change;
  - PyTorch Mamba-2, with `mamba_ssm` as an optional fast path;
  - fixed-capacity packed chunks in M;
  - byte BPB and `ByteLM`.
- **Out:** generation and inference state (`step`), flash_attn, HF checkpoint parity, aarch64.

Runs, which Janek launches:
1. `tiny_local` on CPU.
2. A short h12 smoke run.
3. 1-stage and 2-stage, each at h16_chinchilla's bytes and FLOPs.

Stop if the 2-stage doesn't beat h16 on BPB. Skip H-net if only a BPB win at Chinchilla data
counts, since there the 1-stage only ties.

Open questions, each with my working assumption:

1. Is a tokenizer-free tie worth ~800 lines, or is the 2-stage model the real target? [2-stage is
   the headline run.]
2. Can `mamba_ssm` go into the lock, or PyTorch SSD only? [PyTorch default, `mamba_ssm` as an
   optional fast path on x86.]
3. Who fills the `mamba2` slot in line 1's `attention.py`? [H-net, after line 1 is done.]
4. Byte-match with 4608-byte sequences and 30,700 steps, or keep 30k steps and train on 2% fewer
   bytes? [Byte-match.]
5. What happens on capacity overflow? [Capacity 1.1×; drop the extra chunks and log it.]
6. Is our cluster `fineweb` FineWeb or FineWeb-Edu? B was measured on -Edu. [Measure B on our
   holdout.]
7. Work out a 2-stage FLOP match before building? [Yes, as part of the implementation task.]
