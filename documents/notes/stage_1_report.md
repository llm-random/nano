# Stage 1 — first models: report

**Status: the learning-rate grid is training on LEM (n = 10–13 since 00:44, n = 9 since 01:41, n = 8 since 03:36, both arms); the smoke runs passed.**

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
6. **n = 9 was started early, at 01:41, for both arms**, while the grid was at step ~850.
   The plan adds the next n beyond an edge only once the edge wins. But at that point n = 10
   led both arms, on eval and on train loss averaged over steps 800–899:
   dense 3.894 against 3.933 at n = 11; AME 5.153 against 5.259 at n = 12. A run's
   advantage at high LR usually grows as the cosine schedule decays, so the edge rule was
   likely to fire for both arms. Waiting for it would have cost a whole second wave
   (about 4.5 h) on the critical path, and Stage 2 needs S1-A0's n\*. If n = 10 does not
   stay best, the two n = 9 runs (about 9 GPU-h) are simply an extra grid point. The
   selection rule is unchanged: the lowest `final_ce` over every n run.
7. **n = 8 was started early, at 03:36, for both arms**, by the same reasoning, one step
   further. Train loss averaged over 200-step windows at matched steps, step 1,700:
   - **dense:** n = 9 is at 3.315 against 3.513 at n = 10, a lead of 0.20, so n = 9 will
     almost surely win dense and the rule will ask for n = 8;
   - **AME:** n = 9 and n = 10 are tied, 4.943 against 4.947 at step 1,300 and 4.901 against
     4.888 at step 1,500.

   An AME win at n = 9 would put n = 8 on Stage 2's critical path, since Stage 2 needs
   S1-A0's n\*. Starting it now saves about 2.7 h if it is needed; otherwise it costs about
   4.5 GPU-h. Dense n = 8 is off that path, but its case is strong, and it finishes Stage 1
   sooner.

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

All eight started at 00:44 and passed the two-minute check: the model built with the
expected parameter count, and there was no error in the log.

| Arm | n | LR | Job | W&B run | Node |
|---|---:|---:|---|---|---|
| S1-D | 10 | 2⁻¹⁰ = 9.77e-4 | 6251667_0 | `y451s5xq` | r12-8 |
| S1-D | 11 | 2⁻¹¹ = 4.88e-4 | 6251667_1 | `v34fc3g7` | r12-11 |
| S1-D | 12 | 2⁻¹² = 2.44e-4 | 6251667_2 | `4k9jtp9b` | r12-15 |
| S1-D | 13 | 2⁻¹³ = 1.22e-4 | 6251667_3 | `l7iznfkf` | r16-2 |
| S1-A0 | 10 | 2⁻¹⁰ | 6251668_0 | `ncfg11gm` | r16-2 |
| S1-A0 | 11 | 2⁻¹¹ | 6251668_1 | `0cdw01vh` | r16-15 |
| S1-A0 | 12 | 2⁻¹² | 6251668_2 | `h45qwv5g` | r18-1 |
| S1-A0 | 13 | 2⁻¹³ | 6251668_3 | `ssdlz0nk` | r18-1 |
| S1-D | 9 | 2⁻⁹ = 1.95e-3 | 6251812_0 | `ovy19sqp` | r16-5 |
| S1-A0 | 9 | 2⁻⁹ | 6251816_0 | `p5u294hj` | r16-7 |
| S1-D | 8 | 2⁻⁸ = 3.91e-3 | 6252398_0 | `x0xlqroj` | r12-3 |
| S1-A0 | 8 | 2⁻⁸ | 6252402_0 | pending at 03:40 | |

The n = 9 runs (decision 6) were submitted at 01:41 from the same worktree. The only change
was each grid file's `^learning_rate` list, edited to `[9]` and left uncommitted, as the
plan prescribes for running a subset. Their cemetery clones are
`s1_d_h12_lrNone_2026-11-06_01-41-09` and `s1_a0_h12_lrNone_2026-11-06_01-41-49`. For the
dense n = 9 job, `run_exp.py` gave up waiting for the job id, but `sbatch` had succeeded:
the tmux pane on LEM shows `Submitted batch job 6251812`.

**Every snapshot is `8a240816` plus generated files.** Each cemetery commit has
`8a240816` as its parent. It adds `exp.job` and `generated_configs/`, and drops `pixi.*`,
as `run_exp.py` always does.

| Cemetery commit | Jobs | Differs from `8a240816` in `configs/` and `src/` |
|---|---|---|
| `38187b84` | S1-D grid, 6251667 | nothing |
| `a162a92f` | S1-A0 grid, 6251668 | nothing |
| `26688afc` | S1-D n = 9, 6251812 | both grid lists read `[9]` |
| `539eddce` | S1-A0 n = 9, 6251816 | both grid lists read `[9]` |
| `489342e2` | S1-D n = 8, 6252398 | both grid lists read `[8]` |
| `76bd1630` | S1-A0 n = 8, 6252402 | both grid lists read `[8]` |

The two n = 9 snapshots also carry stale `generated_configs/config_1–3.yaml` left in the
worktree by the grid submit. Their `exp.job` runs `--array=0-0`, so only `config_0`
(n = 9, the right arm) trains. The worktree's `generated_configs/` is now cleared before
each submit.

**Progress at step ~850 (01:40).** Eval loss on 10 batches:

| Arm | n = 10 | n = 11 | n = 12 | n = 13 |
|---|---:|---:|---:|---:|
| S1-D, step 800 | 3.797 | 3.828 | 4.188 | 4.594 |
| S1-A0, step 800 | 5.063 | 5.094 | 5.188 | 5.375 |

The high learning rates are unstable early:
- **dense n = 10:** train loss spiked to 8.94 around steps 300–500;
- **AME n = 10:** spiked to 9.56 around steps 300–500;
- **AME n = 11:** spiked to 7.69 around steps 500–600.

All three recovered, and n = 10 still led both arms afterwards. Throughput matches the
smoke baseline: 3.26–3.28 s/step dense and 3.59–3.61 AME, i.e. 35.5% and 31.0% MFU.

**The 10-batch eval loss is logged in bf16.** Its values step by 1/64 below a loss of 4
and by 1/32 between 4 and 8, so a difference of 0.03 in an AME eval is one rounding step.
Mid-run comparisons in this file therefore use train loss averaged over 100–200-step
windows. `final_ce` does not have this problem: it averages 256 chunk losses in float64,
which is good to about 0.001.

**Progress at 03:34.** Grid runs at step ~2,600–2,900, n = 9 at ~1,700–1,800. Train loss
averaged over 200-step windows:

| Arm | Window from step | n = 9 | n = 10 | n = 11 | n = 12 |
|---|---:|---:|---:|---:|---:|
| S1-D | 1,700 | 3.315 | 3.513 | 3.604 | — |
| S1-A0 | 1,500 | 4.901 | 4.888 | 4.946 | 5.046 |

Results: pending.

## W&B runs to ignore

| Run | Job | Why |
|---|---|---|
| `auyt41sd` | 6251533_0 | died at model creation on node r16-11 (bad GPU) |
| `1peeepdd` | 6251533_1 | the same |
