# Token superposition (TST): scoping note

As of 2026-11-12, commit `581e2f73`. Source: Peng, Gigant, Quesnelle (Nous Research),
*Efficient Pre-Training with Token Superposition*, [arXiv 2605.06546](https://arxiv.org/abs/2605.06546)
v2 (19 May 2026), read in full. It links no code of its own; App. A's three listings are the
whole implementation. Shao et al. (ICLR 2025, "patch-level training", §2.4) published the same
mechanism independently, at up to 2.7B params / 360B tokens for 0.5× the training cost.

## 1. What the method changes

One model, two phases (§3). In phase (i), the first r·N of N steps, each sample holds s× more
tokens, cut into non-overlapping bags of s contiguous tokens. The model reads one averaged
embedding per bag and predicts the next bag's s tokens with a multi-hot cross-entropy. In phase
(ii), the remaining (1−r)·N steps are ordinary next-token training, continued from phase (i).
The model still sees L positions per sample, so a step costs the baseline's FLOPs while reading
s× the data (§3.1). The paper credits the gain to coarse-to-fine throughput, and it holds only
if the embedding and head carry over. Re-initialising them at the switch ends worse than the
baseline (Table 2: 2.938 vs baseline 2.808 vs TST 2.676; §5.3).

- **Inputs** (§3.1, Listings 1–2): s·L tokens become `[B, L, s]`. Each position's embedding is
  the mean of its bag's s embeddings, summed in fp32.
- **Model:** unchanged; no new parameters.
- **Loss** (§3.2, Eqs. 1–3, Listing 3): bag j (tokens js … js+s−1) targets bag j+1 through the
  one head. Eq. 2 is the KL to a uniform target over the bag (App. C.1, Eqs. 4–5). Dropping its
  constant log s gives Eq. 3, the one they train with: the mean of s ordinary CEs on the same
  logits.
  - Variants: hinge and BCE end worse than the baseline (§3.2), the sum-to-one Eq. 6 ties, and
    power-law weights (App. D, Eq. 7) help only for s ≥ 8.
  - Input-only and output-only superposition each beat the baseline; both together are best
    (Fig. 6).
- **Inference/generation:** unchanged after phase (ii). A phase-(i) model is not an LM: r = 1
  runs end at loss 4.4–6.6 (Tables 4–6).
- **Settings:** robust for s ∈ [4, 8] and r ∈ [0.2, 0.4] (§6). Table 1 uses s = 6, r = 0.3 for
  the dense models and s = 16, r ≈ 0.25 for the MoE. Training uses AdamW (0.9, 0.95) and WSD
  with 2000 warmup steps and a 10% decay, over the whole run (§4).

## 2. What it touches here

| Where | Change | Owner |
| --- | --- | --- |
| `model.py` `TransformerEmbedding.forward` | `[B, l, s]` ids → mean of the s embeddings in fp32 (~5 lines) | line 1 (theirs for nGPT) |
| `trainer.py` `train()` | for step < `n_tst`, pull s loader batches and concatenate along the sequence → `[B, s·1024]` | — |
| `trainer.py` `_preprocess_input` | inputs `x[:, :-s].view(B, l, s)`, targets `x[:, s:].view(B, l, s)`; l = 1023, today's position count | — |
| `trainer.py` `_cross_entropy` | bag variant: one `log_softmax`, gather the s targets, mean (= Eq. 3) | — |
| `trainer.py` `calculate_loss`, `log_metrics` | bag path only when `model.training` and step < `n_tst`; log the phase-(i) loss under its own key | line 3 edits both (aux loss) |
| `trainer.py` `_batches_from_start_step` | a resume at step k skips s·min(k, n_tst) + max(0, k − n_tst) batches | — |
| `trainer.py` `train()` (`_seq_len_for_flops`), `_log_throughput` | MFU counts positions, not data tokens | — |
| `definitions.py` `TrainerSchema`, `configs/_trainer/llama.yaml` | `token_superposition: null` or `{bag_size, steps}` | — |

Unchanged:
- `mfu.py` (line 1), if the trainer passes position counts; `eval.py` (`NanoLM` sends 2-D ids).
- The packer in `datasets.py` (line 1): the loader still yields 1024-token rows.
- `main.py` optimizer construction and `schedulers.py` (line 2): one optimizer and one schedule
  span both phases.
- The FFN slot and `moe.py` (line 3); the float8 conversion (line 6), which swaps `Linear`s
  while the bag mean is in the embedding; RoPE tables, since l = 1023 fits.

Collisions:
- Line 1's document masking builds `doc_ids` from 2-D `input_ids` in `LLM.forward`, but
  phase-(i) ids are 3-D.
- Line 3 also edits `calculate_loss` and `log_metrics`. Keeping TST in its own helpers keeps
  that merge mechanical.

**Switch or separate model:** a switch, off by default. The model class, parameters, checkpoint
format and inference are unchanged. With `token_superposition: null` every path is today's, so
the CPU bit-identity check applies. It belongs under `trainer.*`, not `common.*`, because no
architecture changes.

## 3. How to evaluate it against our runs

- **Metric: loss per token on the fixed holdout.** That is `final_ce` (8192 × 1024) and
  `eval/loss` after the switch. Tokenizer and eval set are unchanged, so per-token CE compares
  as is. Bits per byte would only rescale it by the eval set's fixed bytes per token.
- **Not comparable:** phase-(i) `train/loss` (Eq. 3, floor log s = 1.79 at s = 6), phase-(i)
  `eval/loss` (a bag model scored on next tokens), and phase-(i) checkpoints and downstream
  scores (h16's `step_2999` to `step_8999`).
- **In-loop downstream (`joint`):** comparable after the switch, but secondary at our sizes. At
  the paper's 270M, HellaSwag gains 2.3 points and BoolQ loses 2.7 (Table 3).
- **x-axis: steps.** Every step costs the same FLOPs (Table 1: equal B200-hours), while
  processed tokens grow s× during phase (i).
- **h12 first**, on top of `h12_chinchilla` (12k steps, 262k tokens/step, 2 GPUs).
  - h12's 106.2M non-embedding params match the paper's "270M". That model is SmolLM2-135M with
    untied 128k embeddings, ≈106M non-embedding by my count.
  - Run s = 6, r = 0.3 (`n_tst` = 3600) at h12_chinchilla's best LR, plus s = 4 and s = 8.
- **Then h16 against `h16_chinchilla`** (30k steps, `n_tst` = 9000), at the best LR of its
  [8, 9, 10, 11] sweep. The nearest paper point is 600M (≈315M non-embedding, between h16 and
  h20).
- **Regime gap.**
  - The paper's 270M and 600M runs see 42B tokens (≈400 per non-embedding param at 270M) in
    2M-token batches.
  - Our chinchilla runs see ~20 tokens/param in 262k-token batches.
  - Only the 3B run (≈14 tokens/param) is near Chinchilla. h12/h16 test a corner the paper
    didn't run.

## 4. What it costs

- **Implementation:** ~60–80 lines in `trainer.py`, ~5 in `model.py`, plus schema and configs.
  About a day including `tiny_local`. CPU tests:
  - Eq. 3 equals the mean of s `F.cross_entropy` calls.
  - Bags and targets align on a toy sequence.
  - The resume skip count is right.
  - The switch is bit-identical when off.
- **Compute and memory per step:** model FLOPs are unchanged (1023 positions).
  - The extras are s embedding lookups and s gathers, both negligible.
  - With one `log_softmax`, memory is the baseline's.
  - The paper's loop of s `F.cross_entropy` calls on `pred.float()` (Listing 3) keeps s
    log-softmax copies for backward in eager, contrary to §3.2's "no additional memory". At
    h16 micro-batch 64, one fp32 copy is 64·1023·50304·4 B ≈ 13 GB, so s = 6 adds ~65 GB.
- **Data loader:** it reads s× the raw tokens per step. h16 runs at 0.32 s/step on 4×H100
  (line 6 prompt), ~205k tokens/s per rank, so 2 workers must tokenize ~1.2M tokens/s per rank
  at s = 6 (~3.3M at s = 16). `data_fetch` is 0.1–0.2 ms today (`multigpu_efficiency.md`) but
  unmeasured at 6×: run h12 with `trainer.profile_timers=true` and raise `num_workers` if it
  stalls.
- **MFU:** phase-(i) MFU would read more than s× too high. `train()` sets
  `_seq_len_for_flops = batch.shape[-1] - 1` (s·1024 − 1 in phase i), and `_log_throughput`
  caches that FLOPs count at the first logged step and multiplies it by data tokens/s. The
  cached count keeps phase (ii) inflated through the attention-score term. Fix: count positions
  for MFU, and keep `processed_tokens` and `train/tokens_per_s` as data tokens. `wandb_mfu.py`'s
  formula path overstates TST runs the same way, so use the logged `train/mfu`.
- **Risks:**
  - Regime transfer (§3 above).
  - Data repeats at the switch: a fresh 1024-token loader with today's batch-count skip re-reads
    ~(s−1)/s of phase (i)'s data. The splice design avoids that, at the cost of s−1 joins
    without EOS per phase-(i) row.
  - Untested combinations: Muon (the paper runs only AdamW), document masking, and MoE routing
    on averaged tokens (the 2.5× result is an MoE, with nothing on router balance).
  - Evidence: single runs, no seeds (§7); Table 1's loss is training loss.
  - Data budget: (r·s + 1 − r)× the baseline's tokens, 2.5× at s = 6, r = 0.3. FineWeb's 1.28e9
    rows cover it.
  - Compile: `compile_model` and `compile_loss` recompile once at the switch;
    `compile_blocks_full` sees the same block shapes.
- **Gains reported** (training loss at equal steps, Table 1):

  | Model | Steps | Baseline → TST | Equal-loss speedup |
  | --- | --- | --- | --- |
  | 270M | 20k (42B tokens) | 3.212 → 3.142 | — |
  | 270M | 100k (209B) | 3.092 → 3.048 | — |
  | 600M | 20k (42B) | 3.019 → 2.943 | — |
  | 3B | 20k (42B) | 2.808 → 2.676 | 1.8× (baseline needs 36k steps for 2.677) |
  | 10B-A1B MoE, s = 16 | baseline 125k, TST stopped at 50k | 2.252 vs 2.236 | 2.5× (12,311 vs 4,768 B200-h) |

  - The surface is flat. At 270M/20k, every s in 4–16 with r in 0.1–0.5 is within 0.022 of the
    best (3.139 at s = 7, r = 0.3; Table 4).
  - At equal data TST loses (Fig. 3c: 2.676 vs 2.640), so it trades data for compute (§7).

## 5. Recommendation

**Implement, core only.** That means uniform Eq. 3 with one `log_softmax`, a hard in-loop switch
at `n_tst`, and per-position MFU, without the power-law weights or Eq. 6.
- It's about a day of work, and it touches another line's code only in
  `TransformerEmbedding.forward`.
- The paper's effects (0.04–0.13 loss) should show in one h12 run per setting.
- Go to h16 only if h12 at s = 6 beats `h12_chinchilla`'s `final_ce` clearly. There is no h12
  seed-noise number yet; the paper's 270M gap is 0.07.

Open questions, each with this note's assumption:
1. Phase switch: in-loop, as one W&B run with the optimizer state kept (assumed), or two runs as
   in §3? The paper restarts with the TST code removed and doesn't say whether it keeps the
   optimizer state.
2. Phase-(i) rows: splice s rows of 1024 tokens (assumed; exact resume, no `datasets.py`
   change)? Or pack native s·1024 rows with a second loader (line 1's packer config, plus a
   scaled skip)?
3. Document masking in phase (i): off (assumed), or a bag rule such as "a bag belongs to the
   document of its last token"?
4. LR: reuse each baseline's best LR (assumed; the paper uses one LR for both arms), or sweep
   TST too?
5. Equal-loss: is it wanted? It needs a longer baseline, e.g. h12 at 1.5–2× steps (the paper's
   3B compared 36k baseline steps with 20k TST steps).
6. Config: `trainer.token_superposition: {bag_size, steps}` with absolute steps (assumed), or r
   as a fraction of `n_steps`?

## 6. Status (2026-11-23): basic version implemented, rebased onto `46a5eb89`

Implemented with assumptions 1, 2 and 6: an in-loop switch in one run, phase-(i) rows spliced from
s loader batches, and absolute `steps`. Uncommitted on branch `worktree-line4-token-superposition`.

- **Key:** `trainer.token_superposition: {bag_size, steps}`, with `steps: 0` meaning off (the
  default in `configs/_trainer/{llama,nano}.yaml`). It is not `null` because Hydra's CLI can't fill
  a null structured node. With this form, `trainer.token_superposition.steps=15000` works.
- **Batch size warmup:** `validate()` rejects it together with token superposition: bag steps
  splice several loader batches, warmup steps slice one.
- **FLOPs:** a bag step computes h12_chinchilla's 64 × 1023 positions (6 rows of 1024 spliced), so
  FLOPs per step equal a plain step's; only FLOPs per data token drop 6×. The FLOPs-matched run
  therefore keeps 50_000 steps: 15_000 bag steps (30%) and 35_000 plain ones.
- **Wall clock:** the GPU extras are an `embedding_bag` over s× ids and one `gather` in the
  loss. The real cost is on the CPU:
  - the loader must deliver s× the tokens per step;
  - one loader process tokenizes 0.09M tokens/s on a laptop i7 (73% of that time is the Rust BPE
    `encode_batch`), and the packer tokenizes ~4 documents per 1024-token row and keeps one row;
  - h12 on 2 GPUs needs ~0.35M tokens/s per rank today and ~2.1M at s = 6;
  - lem grants 16 CPUs per GPU (`cpus_per_gpu` in `_cluster/lem.yaml`), so more workers may be
    enough; otherwise use pre-tokenized data.
- **Token and FLOPs axes (since 2026-11-25):** `processed_tokens`, W&B's `token_count` and
  `train/tokens_per_s` count data tokens. A bag step reads 6× the tokens of a plain step.
  `processed_flops` (W&B: `flops`, with every metric also under `flops/*`) counts model FLOPs:
  positions × `model_flops_per_token`, so a bag of 6 costs one position. MFU is that FLOPs
  delta over step time. Compare runs on `flops/*` at equal compute and on `tokens/*` at equal
  data. `processed_flops` is checkpointed. A checkpoint saved before the FLOPs axis starts it
  at `processed_tokens` × FLOPs per token, which is exact for runs without superposition.
  - `xutnbfji` counted data tokens and has no FLOPs axis.
  - `okgg20fp` counted positions (data tokens ÷ 6 in the bag phase).
- **LR re-warmup:** `token_superposition.rewarmup_steps` scales the LR linearly from
  1/rewarmup_steps to 1 over that many steps after the switch. The scheduler never sees it;
  `train/lr` logs the scaled value.
- **Checked on CPU:**
  - switch off: all logged metrics of a 200-step `tiny_local` run are bit-identical to the base;
  - `tiny_local_token_superposition` runs both phases, then evals and saves;
  - resuming inside phase (i) reproduces the uninterrupted run (`test_training_smoke.py`);
  - 2-rank FSDP2 on gloo with bf16 params matches an unsharded fp32 reference to bf16 precision.
- **To launch** (`run_exp.py --config-path configs/jm/feature`):
  - `h12_token_superposition_mfu`: 600 steps on lem, phase (i) for 0–299 and next-token for
    300–599 in the same job, `profile_timers` on, `num_workers` ∈ {2, 6, 14}. Compare
    `train/step_time` and the `timers_ms: data_fetch=` lines across phases.
  - `h12_token_superposition`: the FLOPs-matched comparison against `h12_chinchilla` at LR 2^-10,
    with 14 loader workers and a 1_000-step LR re-warmup after the switch (since 2026-11-24).

## 7. First h12 result (2026-11-24): W&B `xutnbfji` vs baseline `kiu7rawb`

- **Per step it wins:** `final_ce` 3.031 vs 3.106, −0.075. That's about the paper's −0.070 at 270M,
  and 4–6× the effect of one LR notch in the baseline sweep (LR 8–12: 3.178, 3.125, 3.106, 3.118,
  3.159). TST's train loss drops below the baseline at step ~17_800 and stays 0.06–0.08 lower.
- **The switch shock:** at step 15_000 train loss jumps to 5.12 (50-step mean) and grad norm to
  0.93, 13× the bag phase's 0.07, clipped at 1.0 for ~20 steps. It recovers in ~2_000 steps.
  Two likely causes:
  - AdamW β2 = 0.999 (torch default; the paper uses 0.95), so Adam's v remembers the small
    bag-phase gradients for ~1_000 steps;
  - the input scale jumps by √6: the block-0 residual norm goes from 2.47 to 5.94, because a mean
    of 6 embeddings has ~1/√6 the RMS of one.
- **Wall clock:** with 2 loader workers a bag step takes 0.291 s against 0.100 s, so the run took
  9_655 s against 6_566 s. From the `_mfu` runs (`31oc8vs1`, `ym3egoz0`, `kpxb2dmf`):
  - bag steps are 2.9× plain steps at 2 workers, 1.43× at 6 and 1.24× at 14;
  - the loader saturates at ~3.1M tokens/s and would need ~3.85M;
  - at 14 workers the whole run would be about 8% slower than baseline.
- **Logging:** works as designed but reads oddly:
  - W&B's `token_count` is data tokens, so on `tokens/*` panels phase (ii) starts at 5.89B, after
    the baseline ends at 3.27B. That is right for data; for equal compute use `steps/*` (or
    `flops/*` in runs from 2026-11-25 on).
  - phase-(i) `eval/loss` (6.7–7.3) scores a bag model as a next-token LM;
  - the duplicated `train/loss` rows at round steps are also in the baseline.
