# Stage 1 — first models: report

**Status: n\* = 9 for both arms, by Maciej's decision (decision 9). Dense's best (n = 8) is inside its grid. AME improves all the way to n = 6, so n = 5 and 4 run (job 6255094, from 11:52).**

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
8. **After n = 8 won both arms (07:41): dense gets n = 7; AME gets n = 7 and n = 6 at
   once.** The rule needs n = 7 in both arms. For dense, n = 7 is all the rule asks, and
   dense is off Stage 2's critical path. For AME, the gain per step of n is shrinking
   slowly: 0.212, 0.133, 0.125, 0.082 and 0.061, from n = 13 down to n = 8. Extrapolated,
   that leaves the optimum around n = 6–7, which could take two more waves before Stage 2
   can start. Running n = 6 beside n = 7 brackets it in one wave, for about 4 GPU-h. If
   n = 6 is not needed, it is simply an extra grid point.
9. **n\* = 9 for both h12 arms, by Maciej's decision (2026-11-06, about 10:00).** His
   words: "lets set lr = 9 for h12 as a optimal, it is very close to lr 8 but i prefere
   lr 9 for stability".
   - **This overrides the plan's selection rule**, which takes the lowest `final_ce`. On
     that rule n = 8 led: by 0.042 in dense (2.9824 against 3.0244) and by 0.061 in AME
     (4.3711 against 4.4316), with n = 7 and 6 still running.
   - **The stability record:**

     | Arm | n = 9 | n = 8 |
     |---|---|---|
     | Dense | no spike | no spike |
     | AME | one spike, steps 609–635, peak 14.69 | one spike, steps 1,117–1,141, peak 7.50 |

     Every n ≤ 11 AME run spiked once early, as did dense n = 10.
   - **Consequences.** Stage 2 already runs at n = 9 (stage_2_report.md, decision 1),
     so no re-run is needed. n = 9 lies inside the measured grid, with n = 8 and n = 10
     on either side.
   - **n = 7 and 6 still finish.** Maciej asked for AME n = 7 "just to be sure". It was
     already training as job 6253086_0, so it was not submitted twice. Their final
     losses complete the grid record.
10. **AME n = 5 and n = 4, submitted together at 11:52 (job 6255094).** n = 7 and 6
    finished at 11:51 with AME still improving: 4.3262 at n = 7, 4.2656 at n = 6. So n = 6
    is the best AME point and sits at the grid's edge. The plan's done-criterion needs the
    best inside the grid, so its rule calls for n = 5. The gain did not shrink from 8→7
    (0.045) to 7→6 (0.061), so n = 4 runs beside n = 5 in the same wave, as in
    decision 8.
    - **Cost.** About 8 GPU-h. If n = 5 wins, this saves the next 4 h wave.
    - **What it does not change.** Maciej's n\* = 9 (decision 9) still governs Stage 2;
      these runs only complete the grid record.
    - **The divergence question.** If n = 5 and 4 diverge or end worse, n = 6 is inside
      the grid and the AME arm is done.
    - **Dense is done.** Its best, n = 8, is already inside the grid: n = 7 ended at
      4.2734, far behind.

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
| S1-A0 | 8 | 2⁻⁸ | 6252402_0 | `tuvcps1y` | r10-7 |
| S1-A0 | 7 | 2⁻⁷ = 7.81e-3 | 6253086_0 | `5bvmqr0l` | r10-4 |
| S1-A0 | 6 | 2⁻⁶ = 1.56e-2 | 6253086_1 | `lr528luy` | r10-7 |
| S1-D | 7 | 2⁻⁷ | 6253087_0 | `6ldjhbyd` | r12-3 |
| S1-A0 | 5 | 2⁻⁵ = 3.13e-2 | 6255094_0 | | |
| S1-A0 | 4 | 2⁻⁴ = 6.25e-2 | 6255094_1 | | |

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
| `04316cd4` | S1-A0 n = 7 and 6, 6253086 | AME list `[7, 6]`, dense list `[7]` |
| `66ececd1` | S1-D n = 7, 6253087 | the same |
| `4941a4d6` | S1-A0 n = 5 and 4, 6255094 | AME list `[5, 4]` |

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

### Results so far (06:58): n = 9–13 complete in both arms, n = 8 still training

All ten runs finished their 3,815 steps, logged `final_ce` and wrote their final
checkpoints. The dense runs took 3 h 40 m – 3 h 43 m and the AME runs 4 h 01 m – 4 h 05 m,
well inside the 7 h limit.

**Every row** has seed 123 and 3,815 steps × 1,048,576 tokens = 4.00B tokens, and every
config is the arm's grid file. The commit is `8a240816`; for n = 9 it is `8a240816` with
the grid list set to `[9]` (cemetery `26688afc` dense, `539eddce` AME).

| Arm | n | Parameters: total / backbone / banks | GFLOP/token | `final_ce` | W&B run | Config |
|---|---:|---|---:|---:|---|---|
| S1-D | 9 | 183.45M / 106.19M / — | 1.095 | **3.0244** | `ovy19sqp` | [dense_h12_lr_grid.yaml](../../configs/stage_1/dense_h12_lr_grid.yaml) |
| S1-D | 10 | 183.45M / 106.19M / — | 1.095 | 3.2002 | `y451s5xq` | the same |
| S1-D | 11 | 183.45M / 106.19M / — | 1.095 | 3.3027 | `v34fc3g7` | the same |
| S1-D | 12 | 183.45M / 106.19M / — | 1.095 | 3.5215 | `4k9jtp9b` | the same |
| S1-D | 13 | 183.45M / 106.19M / — | 1.095 | 3.8301 | `l7iznfkf` | the same |
| S1-A0 | 9 | 1,103.60M / 99.05M / 927.28M | 1.053 | **4.4316** | `p5u294hj` | [ame_h12_simplest_lr_grid.yaml](../../configs/stage_1/ame_h12_simplest_lr_grid.yaml) |
| S1-A0 | 10 | 1,103.60M / 99.05M / 927.28M | 1.053 | 4.5137 | `ncfg11gm` | the same |
| S1-A0 | 11 | 1,103.60M / 99.05M / 927.28M | 1.053 | 4.6387 | `0cdw01vh` | the same |
| S1-A0 | 12 | 1,103.60M / 99.05M / 927.28M | 1.053 | 4.7715 | `h45qwv5g` | the same |
| S1-A0 | 13 | 1,103.60M / 99.05M / 927.28M | 1.053 | 4.9844 | `ssdlz0nk` | the same |

Total parameters include the 77.27M of the untied embedding and head. Backbone means
non-embedding parameters without the banks. FLOPs per token are the trainer's PaLM-convention
training FLOPs at seq 2048; they exclude the bank gathers, which are lookups.

| Arm | n | Last 10-batch eval | Loss spike (train) | s/step | MFU | Peak GiB | Job |
|---|---:|---:|---|---:|---:|---:|---|
| S1-D | 9 | 3.0156 | none | 3.272 | 35.5% | 46.99 | 6251812_0 |
| S1-D | 10 | 3.1875 | steps 354–386: 4.41 → 8.94 | 3.263 | 35.6% | 46.99 | 6251667_0 |
| S1-D | 11 | 3.2969 | none | 3.266 | 35.5% | 46.99 | 6251667_1 |
| S1-D | 12 | 3.5156 | none | 3.283 | 35.4% | 46.99 | 6251667_2 |
| S1-D | 13 | 3.8281 | none | 3.259 | 35.6% | 46.99 | 6251667_3 |
| S1-A0 | 9 | 4.4375 | steps 609–635: 5.12 → 14.69 | 3.603 | 31.0% | 65.89 | 6251816_0 |
| S1-A0 | 10 | 4.5000 | steps 359–387: 5.34 → 9.56 | 3.589 | 31.1% | 65.89 | 6251668_0 |
| S1-A0 | 11 | 4.6250 | steps 573–595: 5.25 → 7.69 | 3.595 | 31.1% | 66.08 | 6251668_1 |
| S1-A0 | 12 | 4.7812 | none | 3.609 | 30.9% | 65.89 | 6251668_2 |
| S1-A0 | 13 | 4.9688 | none | 3.592 | 31.1% | 65.89 | 6251668_3 |

A spike here is a step whose train loss exceeds the median of the previous 50 steps by
more than 0.5. Every spike came in the first 650 steps, soon after warmup, and every run
recovered from it.

**The best n is at the edge in both arms.** `final_ce` falls monotonically as the
learning rate rises, in both arms, so n = 9 wins dense and AME alike. By the plan's rule,
n = 8 is next. It has been training since 03:36 (decision 7), with these runs:
- dense n = 8: job 6252398, W&B `x0xlqroj`;
- AME n = 8: job 6252402, W&B `tuvcps1y`.

At matched steps, dense n = 8 leads n = 9 in 200-step train loss by 0.035 at step 3,100,
and the lead is growing. AME n = 8 led n = 9 by about 0.06 at steps 1,900–2,300 and was
tied with it at 2,700.

**AME vs dense, so far.** At each arm's best n so far (n = 9), S1-A0 is **1.407 nats**
behind S1-D: 4.4316 against 3.0244. The comparison matches tokens, data order and recipe.
It does not match:
- total parameters: AME has 6.0×, almost all of it banks;
- FLOPs per token: AME is 0.96× dense;
- positional signal: dense has RoPE, S1-A0 has none;
- weight decay: the AME banks are exempt, while the dense W_Q and W_K get 0.1.

**AME readouts at the end of training.** τ starts at 8 = √64 and is learned; the attention
entropy is normalized by the log of the number of keys in view.

| n | τ, mean over heads and layers | τ, per-layer means, min – max | Attention entropy, normalized | Top-8 mass |
|---:|---:|---|---:|---:|
| 9 | 6.89 | 5.39 – 8.54 | 0.909 | 0.137 |
| 10 | 6.82 | 5.98 – 7.93 | 0.930 | 0.113 |
| 11 | 7.15 | 6.80 – 7.84 | 0.932 | 0.108 |
| 12 | 7.49 | 7.36 – 7.79 | 0.933 | 0.105 |
| 13 | 7.72 | 7.67 – 7.80 | 0.932 | 0.105 |

S1-A0's attention stays close to uniform over its context. Its normalized entropy is
0.91–0.93 in every layer, and the 8 strongest keys hold only 10–14% of the mass. τ
falls further the higher the learning rate, most in the middle layers: at n = 9 the
per-layer means form a U, 8.1 in layer 0, 5.4 in layer 5 and 8.5 in layer 11. That fits
the plan's analysis that S1-A0 cannot attend by position. Its attention is a weighted
average over the previous tokens, where the weights depend only on the token-id pair.

### n = 8 (07:41): best in both arms, at the edge again

| Arm | n | `final_ce` | Last 10-batch eval | Loss spike (train) | s/step | MFU | Peak GiB | Job | W&B run | Commit |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---|
| S1-D | 8 | **2.9824** | 2.9844 | none | 3.272 | 35.5% | 46.99 | 6252398_0 | `x0xlqroj` | `8a240816` + list `[8]` (`489342e2`) |
| S1-A0 | 8 | **4.3711** | 4.3750 | steps 1,117–1,141: 4.88 → 7.50 | 3.595 | 31.1% | 66.08 | 6252402_0 | `tuvcps1y` | `8a240816` + list `[8]` (`76bd1630`) |

Seed, tokens, parameters and FLOPs per token are as in the table above. n = 8 beats n = 9
in both arms: by 0.042 in dense (2.9824 against 3.0244) and by 0.061 in AME (4.3711
against 4.4316). So the best n is still at the edge, and decision 8 extends the grid.

`final_ce` so far, by n:

| Arm | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---:|---:|---:|---:|---:|---:|
| S1-D | **2.9824** | 3.0244 | 3.2002 | 3.3027 | 3.5215 | 3.8301 |
| S1-A0 | **4.3711** | 4.4316 | 4.5137 | 4.6387 | 4.7715 | 4.9844 |

At n = 8, S1-A0 is 1.411 nats behind S1-D.

### n = 7 and 6 (11:51): dense's best is inside the grid; AME's is still at the edge

| Arm | n | `final_ce` | Last 10-batch eval | Loss spike (train) | s/step | MFU | Peak GiB | Job | W&B run | Commit |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---|
| S1-D | 7 | 4.2734 | 4.2812 | stuck near 6.0 from step ~280 to ~1,200, then learning as the LR decayed | 3.279 | 35.4% | 46.99 | 6253087_0 | `6ldjhbyd` | `8a240816` + list `[7]` (`66ececd1`) |
| S1-A0 | 7 | 4.3262 | 4.3125 | | 3.623 | 30.8% | 65.89 | 6253086_0 | `5bvmqr0l` | `8a240816` + list `[7, 6]` (`04316cd4`) |
| S1-A0 | 6 | **4.2656** | 4.2500 | | 3.604 | 31.0% | 65.89 | 6253086_1 | `lr528luy` | the same |

Seed, tokens, parameters and FLOPs per token are as above. Train loss over steps
1,900–2,099 already ranked AME as n = 6 < 8 < 7 < 9 (4.639, 4.704, 4.751, 4.764), with
n = 7 trailing n = 8 mid-run before passing it by the end.

`final_ce` by n, every finished point:

| Arm | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S1-D | — | 4.2734 | **2.9824** | 3.0244 | 3.2002 | 3.3027 | 3.5215 | 3.8301 |
| S1-A0 | **4.2656** | 4.3262 | 4.3711 | 4.4316 | 4.5137 | 4.6387 | 4.7715 | 4.9844 |

- **S1-D: done by the plan's rule.** Its lowest loss is n = 8, with a worse point on
  each side. n = 7 is the instability the spikes at n ≤ 10 hinted at.
- **S1-A0: not done.** It improves monotonically all the way to n = 6, so n = 5 and 4
  were started (decision 10).
- **The learning rates point apart.** AME prefers a far higher learning rate than dense:
  ≥ 4× (n ≤ 6 against dense's n = 8). Part of this may be the banks. Each bank row only
  updates when its token appears in the batch, so it sees few updates, and its weight
  decay is exempt.
- **The cost of n\* = 9.** At Maciej's n\* = 9, S1-A0 sits 0.166 nats above its best
  measured point (4.4316 against 4.2656), and S1-D 0.042 above its best.
- **The AME–dense gap depends on the n compared.** At the best measured points it is
  4.2656 − 2.9824 = **1.283 nats**; at n = 9 it is 1.407.

## W&B runs to ignore

| Run | Job | Why |
|---|---|---|
| `auyt41sd` | 6251533_0 | died at model creation on node r16-11 (bad GPU) |
| `1peeepdd` | 6251533_1 | the same |
