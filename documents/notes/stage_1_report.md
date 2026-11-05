# Stage 1 — first models: report

**Status: the learning-rate grid is queued on LEM (submitted 2026-11-06 00:43); the smoke runs passed.**

The plan is [ame_experiments.md, "Stage 1 — first models"](ame_experiments.md#stage-1--first-models).
This file is its record: every job, every result, and every decision the plan left open,
each with its reason. Times are CET.

## Setup

| | |
|---|---|
| Question | With a tuned learning rate for each, how far is the simplest AME (S1-A0) from a dense baseline (S1-D) at h12 and 4B tokens? |
| Code and configs | Commit **`8a240816`** on `ame_main`, for every Stage 1 run. Jobs are submitted from the clean worktree `~/code/nano-stage12` (branch `stage12_lem`, held at that commit), so each cemetery snapshot is that commit plus its generated configs and `exp.job` |
| Configs | [configs/stage_1/dense_h12_lr_grid.yaml](../../configs/stage_1/dense_h12_lr_grid.yaml) (S1-D), [configs/stage_1/ame_h12_simplest_lr_grid.yaml](../../configs/stage_1/ame_h12_simplest_lr_grid.yaml) (S1-A0) |
| Cluster | LEM, partition `plgrid-lem-gpu-h100`, account `plgllmefficont3`; 1 GPU per run (`gres: gpu:hopper:1`), 16 CPUs, 220 GB RAM. Switched from entropy in `8a240816`, so every Stage 1–2 run shares one hardware |
| W&B | `ideas_cv/nano-ame` |
| Recipe | As in the plan: h_balanced h12, FineWeb-Edu seed 123, 2,048 tokens seen × 512 sequences per step, 3,815 steps (4.0B tokens), AdamW wd 0.1 (AME banks exempt), cosine with 38 warmup steps to 10% of peak, eval every 100 steps on 10 batches, `final_ce` on the fixed 8192 × 2048 held-out set, micro-batch 32 × 16, no FSDP2, autocast bf16, final checkpoint only |

What the comparison matches and what it does not (from the plan):

| Matched | Not matched |
|---|---|
| tokens | total parameters: AME has about 6×, almost all of it banks |
| data order | FLOPs per token: AME skips the W_Q and W_K matmuls |
| recipe | positional signal: RoPE vs none |

**Weight decay differs between the arms.** AME's Q/K banks are exempt from weight decay,
while the dense W_Q and W_K are not
([archive/ame_review_and_fixes.md](archive/ame_review_and_fixes.md) 2.6). Every AME-vs-dense
number in this file carries that difference.

## Smoke runs — LEM's baseline

The plan's step 1 "Before the grid": `trainer.n_steps=60 trainer.eval_interval=20`, with
learning rate n = 12 for the AME pair. All six smoke runs finished on 2026-11-06 by 00:40,
and every check passed. Their checkpoints were then deleted, as the plan asks.

| Job | Config | n | W&B run | Node | Result |
|---|---|---:|---|---|---|
| 6251533_0 | stage_1/dense_h12_lr_grid | 10 | `auyt41sd` — **ignore** | r16-11 | FAILED at 20 s: `CUDA-capable device(s) is/are busy or unavailable` at model creation, a bad GPU on r16-11 |
| 6251533_1 | stage_1/dense_h12_lr_grid | 11 | `1peeepdd` — **ignore** | r16-11 | FAILED at 13 s, the same error on the same node |
| 6251543_0 | the same clone, resubmitted with `--exclude=r16-11` | 10 | `bm41vdbc` | r16-14 | completed |
| 6251543_1 | the same | 11 | `ymolkx78` | r16-14 | completed |
| 6251533_2 | stage_1/dense_h12_lr_grid | 12 | `rzjcmjgx` | r12-15 | completed |
| 6251533_3 | stage_1/dense_h12_lr_grid | 13 | `b1vhd7hb` | r18-1 | completed |
| 6251535 | stage_2/ame_h12_simplest_seed2 (the S1-A0 code path, seed 124) | 12 | `ns484ak5` | r16-2 | completed |
| 6251536 | stage_2/ame_h12_pe_shift (ALiBi and the shift) | 12 | `k3ydhc0h` | r16-14 | completed |

**The checks:**

| Check | Dense | AME, S1-A0 path | AME, S2-PS |
|---|---|---|---|
| Micro-batch 32 fits without FSDP2 | yes, peak 46.80 GiB | yes, 65.71 GiB | yes, 70.21 GiB |
| Parameters, as logged | 183,454,464 | 1,103,598,480 | 1,103,599,056 (the 576 extra are α and β) |
| FLOPs per token (training, seq 2048, as logged) | 1.095 GFLOP | 1.053 GFLOP | 1.053 GFLOP |
| s/step on LEM (median over steps 15–59) | 3.271 (mean of 4 runs, 3.265–3.281) | 3.593 | 3.771 |
| s/step in the plan's table (entropy, FSDP2 on) | 2.879 | — | 3.240 |
| AME readouts in W&B | — | yes: 15 keys per layer; α₂ = β₂ = 0, horizon = Infinity (W&B takes it), τ = 8 | yes: α₂ ≈ 0.48, β₂ ≈ 0.38 (layer 0), horizons 16–8,196 |
| `final_ce` logged | yes | yes | yes |
| Checkpoint written | yes, 2.1 GB | yes, 13 GB with the bank state | yes, 13 GB |

The parameter counts match the plan's arithmetic. Banks: 2 × 50,304 × 768 × 12 =
927,277,056; embedding and head: 77,266,944; so the backbone is 106,187,520 (dense) and
99,054,480 (AME).

**What LEM is.** The GPUs report as `NVIDIA H100`, 93.6 GiB, 16,896 CUDA cores: H100 SXM5
94 GB. The trainer's peak table does not know that name, so **W&B logs tokens/s but no MFU
on LEM**. This file computes MFU offline with the H100 SXM dense BF16 peak, 989 TFLOP/s.

| | tokens/s | MFU | 3,815 steps | Whole run, with evals and startup |
|---|---:|---:|---:|---:|
| S1-D | 320,567 | 35.5% | 3.47 h | about 3.9 h |
| S1-A0 | 291,839 | 31.1% | 3.81 h | about 4.25 h |
| S2-PS | 278,063 | 29.6% | 4.00 h | about 4.45 h |

LEM is 12–16% slower per step than the plan's entropy figures. FSDP2 being off explains
only part of it (−2.5% MFU at h16). Each eval of 10 batches adds about 18 s, so 38 evals add
about 12 minutes.

**Startup is about 15 minutes, and concurrent jobs share it.** Hugging Face `datasets`
spends about 14 minutes indexing the 1.28 billion FineWeb rows before the holdout split.
The four dense smokes started between 00:19 and 00:24, yet all of them finished that phase
at 00:33:50 and reached step 2 at 00:35:03. That looks like a shared cache lock: jobs that
start together pay it once.

## Decisions

Each one is a point the plan left open, or a change forced by the switch to LEM.

1. **The dense smoke ran its 4 points at once, not one at a time.** The plan ran it with
   `infrastructure.max_concurrent_jobs=1`. That limit belongs to entropy, whose 8-GPU cap
   and live preemption the plan's own step 3 ties it to. LEM has neither for us, and on
   LEM every run starts with about 15 minutes of dataset indexing, so four serial runs
   would have cost an hour or more of wall clock for nothing. The GPU-hours are the same
   either way.
2. **Smoke runs are named `smoke_*` and tagged `smoke`**, with `job-name` `smoke_s1_d`,
   `smoke_s2_n` and `smoke_s2_ps`, and a 1.5 h time limit. The plan names only the grid
   runs; without this, the AME smokes would log under the real Stage 2 names
   (`s2_n_h12_lr12.0_seed124`, `s2_ps_h12_lr12.0`) and could later be mistaken for them.
3. **Node r16-11 is excluded** from every Stage 1–2 job, after two dense smoke tasks died
   there in seconds with a busy or unavailable CUDA device. Pending jobs were updated with
   `scontrol update ExcNodeList=r16-11`; later submits pass `+infrastructure.slurm.exclude=r16-11`.
   It is a scheduling setting, not part of the recipe.
4. **Every Stage 1–2 run gets a 7 h time limit instead of 5 h**, passed at submit as
   `infrastructure.slurm.time=0-07:00:00`. On LEM an AME run needs about 4.25–4.45 h, which
   would leave 35–45 minutes of margin under 5 h. There are no mid-run checkpoints, so a
   timeout loses the whole run. The time limit is not part of the recipe, and the
   comparison commit stays `8a240816`.
5. **The grid keeps micro-batch 32 × 16.** It fits without FSDP2 on LEM with room to
   spare (70.21 GiB of 93.6 at the heaviest arm), so the plan's fallback of 32 accumulation
   steps is not needed.

Two facts for whoever reads the checkpoints:
- The checkpoint path uses each array task's own Slurm job id, not the array id: dense
  smoke task 6251543_0 wrote to `checkpoint/6251633/0/`. The plan's
  `checkpoint/<job id>/<array task id>/` means that per-task id.
- MFU is missing from every LEM run in W&B, for the reason above. Use tokens/s and this
  file's peak.

## The learning-rate grid

Submitted 2026-11-06 00:43 from `~/code/nano-stage12` at `8a240816`, with
`+infrastructure.slurm.exclude=r16-11 infrastructure.slurm.time=0-07:00:00`. Array task
*i* trains at n = 10 + *i* (checked in the clones' `generated_configs/`).

| Arm | Job | Cemetery clone on LEM (under `.../nano/plgmstefaniak/llmrandom_cemetery/`) |
|---|---|---|
| S1-D | 6251667_[0-3] | `s1_d_h12_lrNone_2026-11-06_00-42-57` |
| S1-A0 | 6251668_[0-3] | `s1_a0_h12_lrNone_2026-11-06_00-43-29` |

Results: pending.

## W&B runs to ignore

| Run | Job | Why |
|---|---|---|
| `auyt41sd` | 6251533_0 | died at model creation on node r16-11 (bad GPU) |
| `1peeepdd` | 6251533_1 | the same |
