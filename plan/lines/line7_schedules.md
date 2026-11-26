# Line 7: schedules

## Human

Two schedule switches, each off by default; `jm/h12_chinchilla` turns the batch warmup on (2026-11-17). Now: both written and checked locally; h12 runs not launched.

- [x] WSD: decay to a fraction of the peak LR, as cosine does, not always to 0
- [x] Batch size warmup: small batches early, more steps for the unstable start

**Needs Janek:**
- This file didn't exist when the line was started; it was written from the two lines in `today.md`. Check the decisions below.
- During the small-batch stages the LR is the schedule's LR, not scaled with batch size. Keep that, or scale it?
- `h12_wsd_final_lr` and `h12_batch_warmup` in `configs/jm/feature/` are written and ready to launch (`run_exp.py --config-path configs/jm/feature --config-name <name>`).

**Last change:** both switches landed; with them off, local losses are unchanged.

## AI

### Status

| # | Task | Status | What landed |
| --- | --- | --- | --- |
| 1 | WSD final LR | done | `WSDScheduler(final_lr_fraction)` is `LinearLR`'s `end_factor`. `_scheduler/wsd.yaml` sets 0.0 (today's decay to 0). Typed in `WSDSchedulerSchema`, `validate` keeps it in [0, 1]. Configs `jm/feature/tiny_local_wsd_final_lr`, `jm/feature/h12_wsd_final_lr` (0.1, cosine's value). Cluster: not run yet. |
| 2 | Batch size warmup | done | `trainer.batch_size_warmup`: null, or `batch_sizes` and `steps` (one entry per stage) plus `final_batch_size`, which must equal `common.batch_size` (the feature configs set `common.batch_size` from it). `Trainer._warmup_split` gives the steps per loader batch; `_warmup_batches` slices the loader's batch; `train()` sets accumulation from the split and logs `train/batch_size` while it's on. `validate` checks the stages. Configs `jm/feature/tiny_local_batch_warmup`, `jm/feature/h12_batch_warmup`. Tests `tests/test_schedules.py`. Cluster: not run yet. On in `jm/h12_chinchilla` since 2026-11-17: batch 64, then 128, over its LR warmup (360 steps, the tokens of the old 120); n_steps 12_240, decay and checkpoint grid 1_224. |

### Task 1: WSD final LR

- The fraction is relative to each param group's own peak LR (`LinearLR` scales each group), so it
  keeps working when line 2 gives groups different LRs. Cosine's `eta_min` is now per group
  too (line 2, `PerGroupCosineAnnealingLR`; bit-identical with one LR).
- Off by default: `wsd.yaml` keeps 0.0. Making 0.1 the default is a one-line change in `wsd.yaml`,
  and it changes every WSD run (nano, h12, h16).
- `LinearLR` reaches the fraction one step after the last training step, as the old decay reached 0:
  the last step trains at `f + (1 - f)/decay_steps` of peak.

### Task 2: batch size warmup

How it works:
- The whole ramp is in one node: `batch_sizes`, `steps` and `final_batch_size`. `validate` makes
  `final_batch_size` equal `common.batch_size`, the loader's batch size.
- The loader is unchanged: it always yields the final batch. A stage at batch `b` cuts each loader
  batch into `split = final_batch_size / b` steps, so the samples arrive in the same order as without
  warmup. After the last stage `split` is 1 and steps are whole loader batches, as today.
- `validate` makes each stage divide `final_batch_size` and end on a loader batch (`steps` a
  multiple of `split`); otherwise a stage would end part-way through a loader batch, and the rest of
  it would need a different slice size. It also makes `split` and `gradient_accumulation_steps`
  divide one another.
- Micro-batches: a warmup step runs `max(1, gradient_accumulation_steps // split)` chunks, so it keeps
  the full step's micro-batch, or runs as one chunk when it is smaller than that.
- `n_steps` counts optimizer steps. To keep the token budget, raise it by what the warmup saves:
  `h12_batch_warmup` runs 13_000 steps (800 at 64, 800 at 128, then 11_400 at 256) for h12_chinchilla's
  3.15B tokens. LR schedule, eval, checkpoints and downstream eval stay in steps; h12_batch_warmup
  pins WSD's warmup 120 and decay 1_200 to h12_chinchilla's values.
- Resume replays the slicing from step 0, so it yields the steps an uninterrupted run would.
- Compile: the smaller warmup micro-batches make dynamo recompile the blocks with a dynamic batch
  dim. At the first full-batch step the trainer calls `torch._dynamo.reset()`, so the rest of the run
  compiles static, as a run without warmup does. (Dynamo's PGO profile is only kept across a reset
  with `torch.compiler.config.job_id` set; nothing sets it.)

Checks (CPU, `tiny_local`):
- Off: `tiny_local` and `tiny_local_wsd`, and `tiny_local` at batch 8 with
  `gradient_accumulation_steps=2`, give loss and LR lines byte-identical to `d8d5f28d`.
- On: `tiny_local_batch_warmup` logs batch 2, then 4, then 8, and ends at the same 24_000 tokens as
  200 full steps. After the rebase onto `c7593088` and the `final_batch_size` change, the off checks
  match `c7593088` and the warmup run's losses are unchanged.
- Resume from `step_19` (stage boundary), `step_29` (mid-stage) and `step_59` (after warmup) gives
  identical loss, LR and batch-size lines from the next step on.
- `compile_blocks_full` on CPU: 10 symbolic sizes during warmup, none after the reset.
- Not checked: GPU s/step after the reset, and FSDP2 with `torch._dynamo.reset()` mid-run (CPU runs
  aren't sharded). The h12 run shows both: compare s/step after step 1_600 with h12_chinchilla.

Open:
- LR is not scaled with batch size during warmup.
- The checkpoint and downstream-eval grid (every 1_200 steps) doesn't line up with h12_chinchilla's
  in tokens: step `s >= 1_600` here has the tokens of step `s - 1_000` there. Compare by tokens.
