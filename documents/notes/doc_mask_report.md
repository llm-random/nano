# Document masking for packed rows — implementation and h12 runs (plan and log)

**Decision (Maciej, 2026-11-21): no document mask in either model.**
- The gain is small, about −0.02.
- The mask costs about 12% more time per step. The settle runs measured this within one
  cluster ([settle_runs_report.md](settle_runs_report.md), item 4). The 1.5% in Results
  below compared entropy with LEM, and entropy runs this workload about 10% faster.
- Unmasked packed rows are a valid way to train.

The record: [standard_model_improvements.md](standard_model_improvements.md#the-document-mask-both-models).

**Status (2026-11-10 23:35): done. Document masking helps both methods by about 0.02
nats at h12 and 4B tokens: AME `final_ce` −0.017 (3.1084 against 3.1250), dense −0.019
(2.9639 against 2.9824); paired train loss −0.016 and −0.020. Each gain is under its tie
threshold, and the AME − dense gap is unchanged (+0.143 without, +0.145 with), so the
unmasked comparison stands (Results). Runs from `doc_mask` 3974addd, after a fix to the
compiled eval path (Log 16:27-17:12).** The configs and the code are on branch `doc_mask`, not on
`ame_main`.

Maciej set the task by `/goal` on 2026-11-10: implement document masking on a branch
from `ame_main` in its own worktree, test it on CPU, then run dense h12 and AME
(KV shift + positional encoding, S2-PS) for 4B tokens at n = 8 on LEM, to measure the
mask's effect on both methods. Times are CET.

## Why

The packer (`src/core/datasets.py`, `sample_packer`) fills each 2049-token row with
documents back to back, each ending in `<|endoftext|>`, and cuts the row at the
length. Until now nothing masked across them: dense attention was SDPA with
`is_causal=True` and no mask, and AME's flex mask was causal plus the sink, the same
for every row. So every token also attended to the earlier documents in its row, and
AME's KV shift mixed each document's first key and value with the previous
document's EOS. That was a deliberate, documented choice: both methods saw the same
cross-document context, so the comparison was fair
([ame_implementation.md](ame_implementation.md), "Repo-specific findings"). These
runs measure what the choice costs or gains, for each method.

## The feature

`common.attention.document_mask: true` (default false). Branch `doc_mask`, worktree
`~/code/nano-doc-mask`, cut from `ame_main` 3ef6e363.

| Piece | What it does | Where |
|---|---|---|
| Documents | A position's document = number of EOS tokens strictly before it. An EOS belongs to the document it ends | `src/core/document_mask.py` `document_ids` |
| Mask | Query i sees key j iff j ≤ i and same document. Every query sees itself, so no row is empty | `allowed_keys` (materialized), `document_score_mod` (flex) |
| Dense (mha/mqa/gqa) | Mask off: the old SDPA call, unchanged. Mask on: the layer asks for token ids (`wants_token_ids`, as AME does); flex with a cached causal block mask and a score_mod that sends cross-document scores to −inf on CUDA, SDPA with the `[B, 1, n, n]` mask on CPU | `src/core/model.py` `RoPEAttention` |
| AME | The same score_mod wraps the ALiBi one in the flex kernel; the reference path masks the materialized scores. The sink stays visible to every query, and the LSE merge is unchanged. The KV shift's predecessor term is zero at every document start, as at position 0 of a row | `src/core/ame.py` |
| AME diagnostics | `score_matrix` and `attention_diagnostics` mask too; normalized entropy divides by the keys a row can see in its own document | `src/core/ame.py` |
| Positions | Not reset per document, and not needed: RoPE's score depends on q_pos − k_pos only, ALiBi's bias on i − j only | — |
| Loss | Unchanged: every position trains, including the EOS that predicts the next document's first token | — |
| Config | `common.eos_token_id: 0` (GPT-NeoX `<|endoftext|>`) in the h_balanced model config, passed to `build_attention`. `validate` refuses the mask without an EOS id before anything is submitted; a planned type refuses the mask at build | `configs/_model/h_balanced/base.yaml`, `src/definitions.py`, `src/core/attention.py` |

**Why a score_mod and not a mask_mod.** A mask_mod would need a block mask built per
batch from the document ids: either traced inside the compiled block, where
`create_block_mask` breaks the graph, or built outside and passed in, where the
BlockMask becomes a graph input (AME's `_block_mask` docstring records both failures).
The score_mod keeps the cached causal block mask, so the only new input is the
document ids, computed in the graph from the token ids it already has. The cost:
cross-document blocks are computed and masked rather than skipped.

## Verification

**CPU** (`tests/test_document_mask.py`, 36 tests; the full CPU suite, 447 passed in 10.6 min):

- Every document of a packed row gets exactly the output it gets alone. Checked per
  layer type (mha, mqa, gqa, gqa NoPE, AME S2-PS, AME without sink), and through a
  whole h4 model (dense, AME, and a dense/AME mix): each document's logits do not
  depend on what precedes it in the row. Two rows with different boundaries,
  including a document that is only its EOS. Unmasked, the same check fails, so it
  has teeth.
- The AME shift stops at a boundary: pinned fully shifted, every document's first key
  and value are zero.
- Off is the old layer bit for bit: same parameters, same outputs, for every type.
- Flex (uncompiled on CPU) matches the reference path, dense and AME with and
  without the sink.
- AME's diagnostic scores are −inf exactly outside a row's own document; the sink
  column stays finite.
- Config: the switch reaches every layer; default off; no EOS id fails `validate`;
  the EOS id is the tokenizer's (`tokenize_fn` appends exactly one).
- The composed run configs equal the baselines' logged W&B configs except the mask
  keys, schema defaults (all equal to the Trainer's), names, tags and LEM's newer
  `mem_per_gpu` (228G against 220G).

**GPU** (`pytest -m gpu`, LEM): the compiled flex kernel against the CPU reference,
forward and backward, eager and inside `torch.compile`, at head width 64 with
boundaries on and across flex's 128-wide blocks; bf16 autocast within bf16
resolution; AME's existing GPU tests. **Passed: all 9 document-mask GPU tests**, on LEM's
interactive node r18-15 (H100, torch 2.7.1+cu128), job 6295055, 09:22-09:24, from the
committed tree 56866314 (`.../nano/plgmstefaniak/doc_mask_gputest_56866314/`, run with
`gputest.job` under `srun --pty`). Of AME's 13 GPU tests, 12 passed. The one failure,
`test_on_cuda_every_gate_trains_under_compile_blocks_full[stem]`, predates this branch:
the ST gate's init draws from a CPU generator (`_private_generator`) while the test
builds the model under a CUDA default device ("Expected a 'cuda' device type for
generator but found 'cpu'"). This branch does not touch that code, and no run here uses
a bank gate ("For Maciej").

**The eval path was not covered, and the first runs died of it.** Every GPU test above
compiles with gradients on. The trainer's eval and final_ce compile a second graph,
under no_grad, and with the mask that graph did not lower (Log 16:27-16:28). Since
3974addd the document ids come from a custom op, and two checks cover the path: the
GPU test `test_on_cuda_compile_blocks_full_trains_then_evaluates` (whole h4 and h12
models, every block compiled, rows of 2049, a training step, then the eval forward
under no_grad and bf16 autocast, against the uncompiled model) and a smoke run of both
run configs on entropy (Log 16:45-17:07).

## Runs

| Run | Config | Baseline (same seed, data order, steps, schedule) | Baseline `final_ce` | Baseline s/step |
|---|---|---|---:|---:|
| `dm_dense_h12_lr8` | [dense_h12_docmask.yaml](../../configs/doc_mask/dense_h12_docmask.yaml) | S1-D n = 8, [`x0xlqroj`](https://wandb.ai/ideas_cv/nano-ame/runs/x0xlqroj), job 6252398_0 | 2.9824 | 3.272 |
| `dm_ame_h12_ps_lr8` | [ame_h12_ps_docmask.yaml](../../configs/doc_mask/ame_h12_ps_docmask.yaml) | S2-PS n = 8, [`doi36p8k`](https://wandb.ai/ideas_cv/nano-ame/runs/doi36p8k), job 6256577 | 3.1250 | 3.746 |

- **Recipe**, both: entropy's h100a, 1 H100 (moved from LEM, Log 13:15-13:24); h12;
  3,815 steps of 512 × 2049 tokens (4B); LR 2⁻⁸; seed 123; micro-batch 32; no FSDP2;
  autocast bf16; `compile_blocks_full`; final checkpoint only; 7 h limit. AME: Tier 2
  banks (`mem_parallel_size: 1`), bank init N(0, 1), banks at the backbone's LR, as the
  baseline. W&B project `ideas_cv/nano-ame`.
- **Jobs (entropy), from `doc_mask` 3974addd:** AME 37694, dense 37695. Clones in
  `~/llmrandom_cemetery/`, named like their cemetery branches:
  `dm_ame_h12_ps_lr8_2026-11-10_17-07-17` and `dm_dense_h12_lr8_2026-11-10_17-07-54`.
- **Ignore:** dense 37660 (W&B `28sjg3ii`) and AME 37661 (W&B `ts67c0g3`), from
  da59f08d, which died at step 100 (Log 16:27); the LEM pair 6294854/6294855 (from
  56866314), cancelled at 16:29 before it started, with no W&B runs.
- **The only difference from each baseline is `document_mask: true`**, and the
  cluster: the baselines ran on LEM. Entropy's FineWeb copy is LEM's
  ([ame_efficiency_parity_report.md](ame_efficiency_parity_report.md), item 6), so the
  batches are the same.

## Results

Masked minus baseline; negative means the mask helps. Paired gaps are train-loss means
at the same steps, per part of training (0-10 / 10-25 / 25-50 / 50-75 / 75-100 %).
Numbers from `documents/scripts/doc_mask_results.py <run> <baseline>`.

| Arm | Masked run | `final_ce` masked / baseline | Δ `final_ce` | Paired gap, last 500 steps | Paired gap by part | s/step masked / baseline |
|---|---|---:|---:|---:|---|---:|
| AME (S2-PS) | [`j6wxx6h2`](https://wandb.ai/ideas_cv/nano-ame/runs/j6wxx6h2) vs `doi36p8k` | 3.1084 / 3.1250 | **−0.0166** | **−0.0163** | −0.014 / −0.040 / −0.025 / −0.018 / −0.016 | 3.746 / 3.746 |
| Dense (S1-D) | [`k3xx6w71`](https://wandb.ai/ideas_cv/nano-ame/runs/k3xx6w71) vs `x0xlqroj` | 2.9639 / 2.9824 | **−0.0186** | **−0.0203** | +0.012 / −0.092 / −0.045 / −0.026 / −0.021 | 3.320 / 3.272 |

**The mask helps both methods by about the same small amount, so it leaves the
comparison between them where it was.** The AME − dense `final_ce` gap is +0.1426
without the mask and +0.1445 with it; in paired train loss AME gains 0.016, dense 0.020.
Packing without a mask was a deliberate choice: both methods saw the same cross-document
context ([ame_implementation.md](ame_implementation.md), "Repo-specific findings").
These runs show it cost each method about 0.02 nats at h12 and 4B tokens, and did not
favour either one. Both gains are under the tie thresholds, so neither is established
alone; that the two agree in sign and size is the stronger evidence.

**AME: a small, steady gain, inside the tie band.** −0.017 in `final_ce` and −0.016 in
the paired gap, which stays under the 0.022 threshold, so by this report's rule it is a
tie. The gap's sign never changed, though. It is negative in every 250-step window, peaks
at −0.043 around step 500, and sits flat at −0.016 from step 3000 to the end. The eval
gaps are −0.0156 (one bf16 step) at every eval from 2000 on. A same-seed rerun of
either arm would say whether 0.016 is real. The mask costs AME nothing in speed.

AME's readouts at the end (`stage_results.py readouts`, mean over the 12 layers, masked /
baseline): τ 8.51 / 8.47; KV shift α₂ |mean| 0.49 / 0.50, β₂ 0.73 / 0.71; head horizons
the same. Attention is more concentrated with the mask: entropy 3.31 / 3.43 nats,
normalized 0.506 / 0.523, top-8 mass 0.593 / 0.572. Part of that is mechanical, because a
masked query has fewer keys to spread over (only its own document's). The learned
temperature and the shift's use did not change.

**Dense: the same small gain, less certain.** −0.019 in `final_ce` and −0.020 in the
paired gap, which is negative from step 250 on and flat at −0.020 to −0.021 over the last
1000 steps. Eval gaps are −0.0156 from step 3100 on (−0.0312 at 3800). The early
−0.092 (10-25 %) is partly the baseline's own trouble: it spiked at steps 196-198 (train
loss 6.19 and 6.5), while the masked run had no spike. Dense runs spread 6-24× more than
AME's, so this gain is less certain than AME's. The mask's flex path costs dense 1.5%
(3.320 against SDPA's 3.272 s/step, across two clusters, which AME's identical s/step
suggests run at the same speed).

## How the verdict is made

As in [ame_improve_runs_report.md](ame_improve_runs_report.md): each pair shares the
seed, data order and every setting but the mask, so the train losses pair step by
step. Two measures:

- `final_ce` difference. Each run's value is a mean of 16 bf16 batch losses, so it can
  be off by up to 0.008, a difference by up to 0.016
  ([reference: final_ce precision](ame_improve_runs_report.md#results)).
- The paired train-loss gap (masked − baseline at the same step), averaged over the
  last 500 steps and over each part of training. This is the finer measure.

AME's same-seed spread at h12 4B is sd 0.0072, so a difference under 0.022 is a tie.
Dense runs spread 6–24× more, so a dense difference is less certain than an AME one.

The pairs span two clusters (masked runs on entropy, baselines on LEM). The batches
are the same; what can differ is floating point (kernel choices, non-deterministic
sums), which non-determinism already varies between same-seed reruns on one cluster.
The tie threshold is kept, with that caveat. s/step across clusters is only indicative.

Both runs are evaluated as trained: the masked model with the mask, on the same
packed held-out rows.

## For Maciej

1. **A GPU test on `ame_main` fails, unrelated to this task:**
   `test_on_cuda_every_gate_trains_under_compile_blocks_full[stem]` (`tests/test_ame.py`).
   The ST gate's init draws from `_private_generator()`, a CPU generator, which fails when
   the model is built under `torch.device("cuda")`. Only a run that builds an ST-gated AME
   model directly on CUDA would hit it.
2. **Branch `doc_mask` is local.** The only pushes were the snapshot branches that each
   submission pushes to the cemetery repo, one per job (two for LEM, four for entropy),
   named like the clones. Pushing `doc_mask` to origin waits for your OK.
3. **The LEM cluster env has no pytest.** For GPU tests there, `pytest` and its pure-Python
   dependencies go on `PYTHONPATH` from a copy, as in the log (09:16). Adding pytest to the
   env built by `update_pixi.py` would remove the step.

## Check procedure

- **Start (about 2 min after training begins):** job running, first steps logged, the
  W&B config has `document_mask: true`, s/step within 1.5× the baseline's, first
  losses level with the baseline's at the same steps.
- **Then one check per decision point:** mid-run (paired gap so far, s/step, NaN or
  spikes: a train loss more than 0.5 above the median of the previous 50 steps), and
  the end.
- Cancelled or failed runs are recorded below with their W&B ids as "ignore".

## Log

| Time | Event |
|---|---|
| 08:45 | LEM busy: 21 nodes in maintenance, 12 allocated; the other session's two AME jobs pending on Priority since 08:23; `sbatch --test-only` forecasts a 1-GPU start on 11-14 (forecasts have been days off before) |
| 08:51 | GPU test job 6294848 submitted (45 min limit), forecast start 10:30 |
| 08:55 | Both runs submitted from 56866314, in two halves as in [stage_1c_report.md](stage_1c_report.md) ("Submission": the cemetery is world-readable, so the W&B key goes to `sbatch` on stdin, never into `exp.job`). Dense: job 6294854, clone `dm_dense_h12_lr8_2026-11-10_08-54-06`. AME: job 6294855, clone `dm_ame_h12_ps_lr8_2026-11-10_08-54-34`. Both clones checked first: `document_mask: true`, LR 8, 3,815 steps, the W&B export line commented out. Forecast start 11-12 12:14; the partition has 507 jobs pending (425 from one user) and 29 running |
| 09:00-09:15 | LEM refused new connections (TCP timeouts on port 22) for about 17 min, after about ten fresh SSH connections in 20 min; a stale shared ControlMaster socket from 08:17 had been hanging every connection that reused it. Since then one multiplexed connection of this task's own is reused |
| 09:16-09:21 | First GPU-test attempt on the interactive node: the cluster env has no pytest (and no pip). pytest 8.4.2 and its pure-Python dependencies copied from the local env into the test directory's `.pylib`, on `PYTHONPATH` for the test job only |
| 09:20 | Batch GPU-test job 6294848 cancelled before it started: it lacked the pytest path, and the interactive node made it redundant |
| 09:22-09:24 | GPU tests pass (job 6295055 on r18-15): all 9 document-mask tests, 12 of AME's 13; the one failure predates this branch (see Verification) |
| 12:15 | Both still pending (Priority). Each job's priority is 673: age 195 (grows with waiting), fair-share 300, job size 79, partition 100. 370 of the partition's 382 pending jobs rank above them, about 90 of those waiting on dependencies. 30 jobs running; 31 nodes are partly used and planned by the backfill scheduler for other jobs; 20 idle nodes are powered down. Forecast start now 11-13 05:23 (11-12 12:14 at submission). A shorter time limit would not help: `sbatch --test-only` gives a new job of this shape the same start (11-16 18:46) at a 4, 5, 6 or 7 h limit. Watcher re-armed (every 20 min, up to 5 days); it also catches a job that starts and leaves the queue between checks |
| 13:15-13:24 | **Moved to entropy**, Maciej: "move them if they are still pending to entropy, there will be a chance to run sooner". The LEM pair was still pending. h100a was full: three AME jobs of ours from other sessions (since about 12:25), wmasarczyk's 4-GPU and wojszalek's 1-GPU jobs (up to 2 days). Our QOS preempts only lukaskoz's jobs, and none were running. Configs switched (da59f08d: cluster, gres `gpu:h100:1`, no exclusion) and checked again against the baselines' logged configs: besides the mask keys and schema defaults, only cluster settings and dataset paths differ. Submitted in two halves, the W&B key on stdin, after checking both clones: dense job 37660, AME job 37661. Both pending (Priority) behind our pkmflin_dense job 37650. Expected start when the three AME jobs finish, about 16:30 (Slurm's forecast assumes full time limits: 11-11 00:24). One watcher covers all four jobs every 10 min |
| 16:19 | Both start on h100a (H100 80GB HBM3), W&B dense `28sjg3ii`, AME `ts67c0g3`. FineWeb opens took about 40 s each; the dataset log has the same 1,279,107,432 rows and the same held-out cut as LEM's runs. Step 0 at about 16:21:30 |
| 16:24-16:29 | **Start check passed.** Both configs have `document_mask: True`; no NaN or inf. s/step from W&B: dense 3.36 (LEM baseline 3.272), AME 3.77 (3.746). Step 0's loss equals the baseline's in both pairs (AME 11.25, dense 11.1875), and AME's steps 1, 3 and 4 too: same initial weights and same batches on entropy. The script flags AME step 11 (9.56, grad norm 16.3) as a spike: it is the warmup transient both baselines have at step 10 (8.25, grad norm 46.5 AME / 15.9 dense), one step later, and the loss is level again at step 12. Paired gap so far: dense +0.051 (steps 0-55), AME -0.003 (steps 0-48), at 0.03 bf16 loss resolution |
| 16:29 | LEM fallback pair cancelled (6294854 `dm_dense`, 6294855 `dm_ame`, both still pending, checked by name, user and state first) |
| 16:27-16:28 | **Both runs died at their first eval, step 100** (found 16:44): dense at 16:27:16, AME at 16:28:19, exit 1, `InductorError: LoweringException: AssertionError: convert FlexibleLayout to FixedLayout first` in `flex_attention`, compiling the blocks' no_grad forward (`Trainer.eval`). The score_mod's captured buffer (`args[7]`) is the document ids, `[32, 2048]` int32, written by inductor to a buffer whose layout was still flexible; torch 2.7.1 cannot lower the score_mod's index into it. Training compiles with gradients, where the forward keeps the ids for the backward and that fixes their layout. Every GPU test compiled that way, so none reached this graph. So the start check passed, and the LEM pair was cancelled a minute after the failure. W&B `28sjg3ii`, `ts67c0g3`: **ignore** |
| 16:45-17:07 | **Fix, 3974addd:** `document_ids` became a `torch.library` custom op, `nano::document_ids`. Inductor does not fuse it; it allocates the output from the op's fake (contiguous int32), a fixed layout in every compile mode. On CPU the same no_grad compile failed before ("No choices to select") and matches eager to 8e-7 after. Entropy job 37688 on the fixed tree: AME's GPU tests 27 of 28 pass (the 28th is the stem-gate failure that predates this branch), and both run configs pass a smoke run: 100 steps at batch 64 (micro-batch 32 × 2), eval every 10 steps, final_ce on 256 samples. Dense: eval 8.375 → 6.906, final_ce 6.875, peak 46.8 GiB. AME: eval 8.375 → 6.8125, final_ce 6.781, peak 70.6 GiB. The first regression test (one layer compiled under no_grad at 256 tokens) passed on the unfixed tree too, so it was replaced with a production-shaped one: whole h4 and h12 models, every block compiled, rows of 2049, a training step, then the eval forward. Job 37693 checks that it fails unfixed and passes fixed. Logs: entropy `~/dm_fixtest_1110_1651/` (`slurm-37688.out`, the two smoke logs) |
| 17:08-17:10 | Resubmitted from 3974addd, AME first, as before (two halves, key on stdin; clones checked, fix present): AME job 37694 (clone `dm_ame_h12_ps_lr8_2026-11-10_17-07-17`), dense job 37695 (`dm_dense_h12_lr8_2026-11-10_17-07-54`). One h100a GPU is free for us at a time: AME starts when test job 37693 ends; dense waits for the next GPU (our pkmflin_dense job 37650 should end about 19:50) |
| 17:06-17:12 | **The new test has teeth** (entropy job 37693). On the unfixed tree, `test_on_cuda_compile_blocks_full_trains_then_evaluates` fails with the runs' error on the `[4, 2048]` int32 document-id buffer, and the dense run config dies at its first eval (step 10) with the runs' error on the `[32, 2048]` buffer. On the fixed tree all 13 document-mask GPU tests pass. CPU suite on the fixed tree: 451 passed, none failed or skipped; black clean. Logs: entropy `~/dm_fixtest2_1110/` (`slurm-37693.out`, `smoke_before_dense.log`) |
| 17:12 | AME job 37694 starts on h100a; dense job 37695 waits for a GPU (Resources) |
| 17:21-17:24 | **AME (W&B [`j6wxx6h2`](https://wandb.ai/ideas_cv/nano-ame/runs/j6wxx6h2)) is past its first eval**: eval/loss 5.9375 at step 100, equal to the baseline's. Start check passed: `document_mask: True`, 3.746 s/step (baseline 3.746), no NaN or inf, the step-11 warmup bump (9.5625) identical to the first attempt's. Paired train-loss gap +0.008 over steps 0-118, -0.005 over the last 100. Expected end about 21:15 |
| ~19:38 | Dense job 37695 starts (W&B [`k3xx6w71`](https://wandb.ai/ideas_cv/nano-ame/runs/k3xx6w71)); the watcher saw it at 20:15 |
| 20:15-20:20 | **Mid-run check.** AME at step 2780: 3.746 s/step, no NaN; paired gap (masked - baseline) by part of training -0.014 / -0.040 / -0.025 / -0.018 (0-10 / 10-25 / 25-50 / 50-75 %, the last part still running), last 500 steps -0.017; eval gaps -0.016 at every eval from 2000 to 2700. Dense at step 605, past its first eval: `document_mask: True`, 3.322 s/step (baseline 3.272), no NaN and no spikes. Its gap is -0.053 over the last 500 steps, but in part that is the baseline's instability: the baseline spiked at steps 196-198 (eval gap -0.44 at step 200) |
| 21:26-21:40 | **AME done** (job 37694 COMPLETED at 21:26, 4 h 14 min; final checkpoint saved). `final_ce` 3.1084 against the baseline's 3.1250: -0.0166. Paired gap -0.014 / -0.040 / -0.025 / -0.018 / -0.016 by part, last 500 steps -0.0163, negative in every 250-step window. Readouts in Results. Dense at about step 2000 |
| 23:23-23:35 | **Dense done** (job 37695 COMPLETED at 23:23, 3 h 45 min; final checkpoint saved). `final_ce` 2.9639 against the baseline's 2.9824: -0.0186. Paired gap +0.012 / -0.092 / -0.045 / -0.026 / -0.021 by part, last 500 steps -0.0203. AME - dense gap +0.1445 masked, +0.1426 unmasked. Results written |
