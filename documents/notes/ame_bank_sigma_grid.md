# AME bank-init σ grid at h12 (plan and log)

**Status:** RUNNING, 2026-11-12: round 1 of 3.

**Question:** which bank init std σ is best for S2-PS at h12, 4B tokens? Bank-gates
stage 2 showed σ = 0.01 beats σ = 1 (−0.0895 in final_ce,
[ame_bank_gates_stage2_report.md](ame_bank_gates_stage2_report.md)); nothing else
was tried.

Maciej, 2026-11-12: "do a sequential grid to find optimal σ for h12 4B tokens
training, submit at most 4 x 1 gpu jobs at once for entropy, so no more than 2 gpus
are occupied by my jobs, use optimal lr for this grid search, stop after 12
experiments and draw a chart of results".

## The runs

- **Recipe: `kf4q4t4s`'s** ([configs/bank_gates/stage2_none_std001.yaml](../../configs/bank_gates/stage2_none_std001.yaml)):
  S2-PS (AME with the KV shift and ALiBi), h12, 3,815 steps = 4B tokens, no gate,
  bank LR ratio 1, seed 123.
  - **n = 8 (LR 2⁻⁸),** S2-PS's optimum and the LR of both earlier σ points.
  - **Only σ changes.** With one seed, every run sees the same batches, and every
    weight but the banks starts the same. The banks start as σ × the same standard
    normals. So any two runs are an exact pair.
- **Three differences from `kf4q4t4s`, none in the training:**
  - entropy (h100a, one H100 per run) instead of LEM;
  - no final checkpoint, so no `token_counts.pt`: 12 checkpoints of 13 GB don't
    fit entropy's 120 GB `/home` quota (83 GB in use);
  - names: σ = 10^(−m/100), run `bank_sigma_m<m>` (m200 is σ = 0.01), so names
    carry no ".".
- **Concurrency, Maciej's limit:** each round is one sbatch array of 4 runs with at
  most 2 running (`--array=0-3%2`). One round takes about 7.8 h.
- **Configs:** `configs/bank_sigma/round<k>.yaml` on branch `bank_sigma_grid`
  (worktree `~/code/nano-bank-sigma`). Tools on the branch:
  [scripts/grid_points.py](../../scripts/grid_points.py) (compose and check a grid as
  `run_exp.py` does), [scripts/paired_gap.py](../../scripts/paired_gap.py),
  [scripts/watch_array.py](../../scripts/watch_array.py),
  [scripts/bank_sigma_next.py](../../scripts/bank_sigma_next.py) (the next round's
  picks), [scripts/bank_sigma_chart.py](../../scripts/bank_sigma_chart.py) (the chart
  and the Results table).

## The measure

- **g(σ):** the last-500 paired train-loss gap against `bank_sigma_m200`, the σ =
  0.01 run on entropy. Lower is better.
- **A difference inside ±0.022 is a tie** (AME's same-seed floor).
- **Also logged:** final_ce (bf16, ±0.008), spikes, the grad-norm max, s/step, and
  the banks' share of the clipped gradient norm.
- **Points for free, shown on the chart but not used to pick:**
  - σ = 1 on entropy: the parity study's three same-seed runs of `doi36p8k`'s
    config (W&B `nano-ame-parity`: `u1zpx7lt`, `s65ebe41`, `w4hmw00o`), their mean;
  - `kf4q4t4s` (σ = 0.01) and `doi36p8k` (σ = 1), both on LEM.

## The rounds

x = log10 σ; x* is the best measured point so far (lowest g). A point already
measured is skipped, and the next rule fills the slot.

- **Round 1, decades:** x = −2 (σ = 0.01 again, on entropy: every pair then stays on
  one cluster), −4, −3, −1. The array runs −2 and −4 first.
- **Round 2, half-decades.** In order, until there are 4:
  1. if x* is the lowest x measured, x* − 1: the curve may still be falling;
  2. x* − 0.5 and x* + 0.5;
  3. the midpoint of the interval between measured neighbours with the largest
     |Δg|, the knee; repeat.
- **Round 3, quarter-decades.** In order, until there are 4:
  1. if x* is the lowest x measured, x* − 0.5;
  2. x* − 0.25 and x* + 0.25;
  3. on each side, the quarter-decade point that bisects the gap between the
     outermost point of the tie band (g ≤ g(x*) + 0.022) and the first point
     outside it;
  4. rule 3 of round 2.
- **When the rules run short** (added 11-13, before any result): one step, then two,
  beyond the end of the measured range with the lower g, the end nearer the optimum.
  Round 2 can run short: 4 decades have only 3 interior half-decades, and x* ± 0.5
  are two of them.
- **[scripts/bank_sigma_next.py](../../scripts/bank_sigma_next.py) applies these
  rules** to the measured g and prints the picks in task order, with the rule behind
  each.
- **The answer:** the best σ and the tie band, as σ ranges. The chart plots g and
  final_ce against σ on a log axis.

## What to expect

- **The mechanism** (stage 2's reading): under the hard QK-norm a bank row's
  direction is all the scores see. Adam moves each element by about lr per step, so
  a row turns at about lr/σ. A small σ turns fast, and that helped rare tokens most
  (stage 2: CE on tokens seen 101-1,000 times 2.962, against 3.305).
- **A plateau is likely below σ ≈ lr = 0.004,** where the first updates overwrite
  the draw.
- **A risk for tiny σ:** a bank row's gradient scales as 1/σ through the norm, and
  the clip (1.0) acts on the joint norm, banks included. If the banks dominate it,
  every other weight's step is cut as well. The logged bank share of the norm and
  clip factor show it.

## Log

### Implementation

2026-11-12, branch `bank_sigma_grid` from `ame_main` at `2299c1ee`. No code change:
the training code and the configs it reads are unchanged since `03317133`, the
commit `kf4q4t4s` ran.
- **Round 1, [round1.yaml](../../configs/bank_sigma/round1.yaml):** composed and
  expanded by `scripts/grid_points.py` as `run_exp.py` does:
  - 4 points, σ = 0.01, 1e-4, 1e-3, 0.1 (m200, m400, m300, m100), each passing
    `check_point`;
  - sbatch header `--array=0-3%2`, `--gres=gpu:h100:1`, 7 h limit.
- **Point 0 (σ = 0.01) against `kf4q4t4s`'s logged config:** the only training keys
  that differ are the dataset paths (entropy's copy of the same FineWeb), the
  checkpoint path (null) and `count_token_updates` (false). The rest is names and
  runtime keys.

### Runs

Round 1 was submitted from `~/code/nano-bank-sigma` at `3a2a83c0`: array 38422, clone
`/home/mstefaniak/llmrandom_cemetery/bank_sigma_m200_2026-11-12_18-29-53`.

| Round | σ | m | Job | W&B | Started | Finished | State |
|---|---|---|---|---|---|---|---|
| 1 | 0.01 | 200 | 38422_0 | | | | PENDING |
| 1 | 1e-4 | 400 | 38422_1 | | | | PENDING |
| 1 | 1e-3 | 300 | 38422_2 | | | | PENDING |
| 1 | 0.1 | 100 | 38422_3 | | | | PENDING |

### Checks

- **Round 1, submit, 11-12 18:30:** PENDING (Priority). No other job of Maciej's is
  on entropy.
  - h100a is full: j321m holds 4 GPUs (8 h limit), and mateuszwyszynski holds 4
    single-GPU jobs (2-day limit, started 09:08).
  - Nine pending jobs rank above ours, among them 1-GPU jobs of 1.4-2 days (hegel,
    nauman ×3). Our priority, 2,891, is the queue's lowest: our fairshare has fallen
    after this week's runs. Our QOS preempts only `lukaskoz_h100`, which has nothing
    running. Slurm's start estimate is 11-21 08:00; such estimates assume full time
    limits and have been days off.
  - **The time limit went from 7 h to 5 h** (`scontrol update`, 18:33). A run takes
    about 3.8 h here, and a shorter job backfills more easily.
- **Next check: when round 1 starts.** One watcher polls every 10 min. It wakes at
  step 20 of the first runs, or at once on a failed task or a NaN.
- **11-13 01:20: still PENDING (Priority).** Age has lifted the priority to 3,453,
  still the lowest on h100.
  - All 8 GPUs are held by single-GPU jobs with 1.4-2-day limits. The first four end
    by 11-14 09:10 at the latest (mateuszwyszynski's, started 11-12 09:08).
  - Three jobs rank above ours: wmasarczyk (1.5 h), crewtool (2 GPUs, 1 day),
    lukaskoz (8 h).
  - Slurm's estimate is now 11-15 09:50.
  - LEM still refuses the key (`Permission denied (publickey)`), so entropy stays
    the only option.

### Results

Bank share is the banks' median share of the gradient norm after step 100. Clipped
is the fraction of steps whose norm was clipped.

| σ | m | Round | g, last 500 | final_ce | Spikes | Bank share | Clipped | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---|

### Analysis

### Decisions and issues

1. **LR n = 8.** "Use optimal lr": n = 8 is S2-PS's tuned optimum
   (stage_2_report.md), and both earlier σ points ran there. The bank LR ratio stays
   1, as in both; ratio 4 is a separate gain (ame_improve_runs_report.md), and
   stacking it with a small σ is untested.
2. **No checkpoints:** the grid's answer needs only the training curves and final_ce,
   which is computed before the save. A per-frequency analysis would need them;
   none is planned.
3. **σ = 0.01 is rerun on entropy** (1 of the 12): it anchors every comparison on
   one cluster. Against `kf4q4t4s` it also measures the cluster offset at σ = 0.01.
4. **Scratch tools went into the repo:** the earlier session's scratchpad was wiped
   on 2026-11-11.
5. **A fallback rule for picks** (11-13, while round 1 was still queued): the plan's
   round-2 rules give only 3 points when x* is interior. The fallback is in "The
   rounds".
6. **The chart, ready before the results:** `scripts/bank_sigma_chart.py` plots g
   and final_ce against σ on one log axis, with the tie band. The σ = 1 parity runs
   and the two LEM runs are gray context. It was tested on `kf4q4t4s` and
   `doi36p8k`. It also adds two columns to the Results table:
   - the banks' median share of the gradient norm after step 100;
   - the fraction of steps clipped.

   The tiny-σ risk shows in these columns. `kf4q4t4s` reads 0.18 and 1%.
