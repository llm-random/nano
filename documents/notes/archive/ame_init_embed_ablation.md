# AME init follow-up: the new init with the old embedding, then n ± 1 (plan and log)

**Archived 2026-11-27 (historical).** Done on 2026-11-10. The bundle stayed rejected; its
one real gain, a small bank σ, reached the standard models as σ = 1e-4. Code on the unmerged
branch `ame_initialization`. **Today:**
[standard_model_improvements.md](../standard_model_improvements.md) ("Other rejected
changes").

**Status:** DONE, 2026-11-10.
- **Step 1 and its extra arm:** AME survived and dense did not (*Results*, *Analysis,
  step 1*).
- **Step 2, AME only, on entropy** (moved from LEM at Maciej's request; *Decisions
  and issues*): the best n is 7, at −0.0285 against the old init at its own best,
  n = 8. That is a win just beyond the tie floor (*Analysis, step 2*).
- No job from this plan is running.
Background: [ame_init_exp.md](ame_init_exp.md), its Results and Analysis.

**Question:** is the Xavier embedding the whole problem with the new init? If the new
init minus that one change ties or beats the old init at h12, does it win at its own
best learning rate?

## Why

- **The new init lost at the references' n.** In ame_init_exp.md, the bundle is:
  - Xavier embedding, head and banks;
  - pkmf's √(0.1/fan_in) `Linear`s;
  - no weight decay on the embedding.

  Its last-500 paired gaps against the old init: dense h12 +1.271, AME h12 +0.014
  (a tie), dense h16 +0.213, AME h16 +0.032.
- **The embedding is the prime suspect.** Its vectors start at norm 0.17, against 24
  before, and block 0's attention output drowns them: 1,800-2,000:1 by step 100 at
  h12, against 0.3:1.
- **One global LR can't compensate.** Adam moves every weight by about lr per step.
  The new init raised that step's ratio to the init std 140× for the embedding and
  only 3-5× for the head and the `Linear`s.
- **The bundle hides a known gain.** In bank-gates stage 2, σ = 0.01 banks alone beat
  `doi36p8k` by −0.0895 at h12
  ([ame_bank_gates_stage2_report.md](ame_bank_gates_stage2_report.md)). The bundle's
  Xavier banks (σ ≈ 0.006) are close to that, yet AME only tied.

## Step 1: the old embedding, at n = 8

| Run | Model | Config copied | n | Old-init reference, final_ce | New-init run (Xavier embedding), final_ce |
|---|---|---|---|---|---|
| `oldemb_ame_h12` | AME with KV shift and ALiBi (S2-PS) | [configs/ame_init/ame_h12.yaml](../../../configs/ame_init/ame_h12.yaml) | 8 | `doi36p8k`, 3.1250 | `ycwqa7k9`, 3.1504 |
| `oldemb_dense_h12` | Dense GQA with RoPE (S1-D) | [configs/ame_init/dense_h12.yaml](../../../configs/ame_init/dense_h12.yaml) | 8 | `x0xlqroj`, 2.9824 | `355fr6rn`, 4.2188 |

- **One change from the `ame_init` runs:** the embedding's draw goes back to the old
  one, a cut N(0, 1) with std 0.88, set in each run config:

  ```yaml
  model:
    embedding:
      init_fn:
        _target_: src.core.model.trunc_normal_ # N(0, 1), cut at +-2, as ame_main
        _partial_: true
  ```

- **Everything else stays the new init:**
  - the head and the banks are Xavier;
  - the `Linear`s are √(0.1/fan_in), cut at ±2σ;
  - the embedding is still not decayed. That exemption is a code attribute on this
    branch (`TransformerEmbedding.no_weight_decay_parameters`), not part of the init.
- **Two paired comparisons per run,** both same-seed:
  - **against the old-init reference:** is the rest of the bundle a gain once the
    embedding is normal? The rest is the head, banks, `Linear`s and the undecayed
    embedding.
  - **against the `ame_init` run:** how much did the Xavier embedding cost?
- **Where:**
  - branch `ame_initialization`, worktree `~/code/nano-ame-init`;
  - configs `configs/ame_init/oldemb_<model>_h12.yaml`;
  - W&B project `nano-ame`, run name `ame_init_oldemb_<model>_h12`;
  - seed 123.
- **Submitting:**
  - Submit both at once, the same way as ame_init_exp.md, *Submitting*.
  - Commit everything on the branch first.
  - The two `ame_init` h12 runs took 4 h 14 min (AME) and 3 h 42 min (dense).

**A model survives step 1** if its run's last-500 paired gap against the old-init
reference is below +0.022: a tie or a win. The threshold is AME's tie floor; dense has
no replicate floor of its own and borrows it.

### Step 1's extra arm: the old init without the embedding's decay

Added 2026-11-09 on Maciej's call: the decay may be the embedding's problem. In the
old-init references, the decay alone shrinks the embedding from norm 24.35 to about
10.8 over the run (*Decisions and issues*).

| Run | Model | Config | n | Reference, final_ce |
|---|---|---|---|---|
| `oldinit_embnowd_ame_h12` | AME with KV shift and ALiBi (S2-PS) | [configs/ame_init/oldinit_embnowd_ame_h12.yaml](../../../configs/ame_init/oldinit_embnowd_ame_h12.yaml) | 8 | `doi36p8k`, 3.1250 |
| `oldinit_embnowd_dense_h12` | Dense GQA with RoPE (S1-D) | [configs/ame_init/oldinit_embnowd_dense_h12.yaml](../../../configs/ame_init/oldinit_embnowd_dense_h12.yaml) | 8 | `x0xlqroj`, 2.9824 |

- **One change from the reference:** the embedding is not decayed, which is this
  branch's default. Everything else is the old init, restored in the config:
  - the embedding draw;
  - scale 1 for the attention and FF `Linear`s;
  - the head's `llm_random_weight_init` at scale 1;
  - AME's `bank_init_std: 1.0`.
- **Two paired comparisons per run,** both same-seed:
  - **against the reference:** the decay alone, on the tuned old init;
  - **against `oldemb_<model>_h12`:** the rest of the new init (the head, banks and
    `Linear`s), with both embeddings the old draw, undecayed.
- **Outside the plan's rules:** the arm decides neither survival nor step 2. The
  plan's rule still judges `oldemb_<model>_h12` against the reference. If the arm
  clearly beats the reference (a gap below −0.022), the decay hurt the old embedding,
  and the survival gap carries that gain; the Analysis then says so.
- **Submitting:** as step 1, beside the pending pair.

## Step 2: n ± 1 for what survives

- **Grid:** for each surviving model, two runs at n = 7 and n = 9. Each is its step-1
  config with only `learning_rate` and the names changed (`..._n7`, `..._n9`), at the
  same seed.
- **Edge rule:** if the best of n = 7, 8 and 9 is an edge (7 or 9), add one run a step
  further (6 or 10). That matches how the references were tuned: each one's n = 8 is the
  best of its own grid (S1-D n = 7-13, S2-PS n = 6-10). Under the old init, dense n = 7
  broke down, so it may here too; it stays a grid point.
- **Nothing survives:** there is no step 2. The Log says so, and the plan ends at
  step 1.
- **Size:** at most 2 runs per surviving model, plus 1 at an edge. Submit each
  model's runs together.

## Checks

As in ame_init_exp.md, *Checks*:
- **About 2 minutes after each start:**
  - the job is running;
  - the first loss is ≈ ln V = 10.83. The Xavier head and the final norm set the
    logits, whatever the embedding;
  - s/step is within 10% of the reference's (AME 3.746, dense 3.272).
- **At step 500 and mid-run:**
  - no NaN;
  - loss spikes and grad norm;
  - the paired gap.
- **The watchers also stop on a loss spike** (a step more than 0.1 above the median of
  the 50 before it, after step 100). The `ame_h16` spike went unseen until its mid-run
  check.
- **Never cancel a job without asking.**

## How to analyze

- **Each run:**
  - final_ce and its Δ against the reference;
  - the paired train-loss gap per 500 steps and over the last 500 steps
    ([scripts/paired_gap.py](../../../scripts/paired_gap.py), on the branch);
  - health: spikes, the grad-norm max after step 100, s/step, peak memory;
  - block 0's embedding norm against its attention update norm, the failure seen in
    ame_init_exp.md.
- **The paired gap decides.** AME ties inside ±0.022; dense borrows that threshold.
- **Step 1 verdict, per model:**
  - A tie or a win means the embedding was the whole problem for that model, and the
    rest of the bundle is neutral or a gain.
  - Still clearly worse means it is not the whole problem: the rest hurts too.
- **For AME, also set the gap against bank-gates' −0.0895** for σ = 0.01 banks alone:
  - near −0.09: the head, `Linear`s and undecayed embedding add nothing;
  - well above it: they cost;
  - well below it: they gain.
- **The extra arm:** its gap against the reference is the decay's effect. Below
  −0.022, the decay hurt; inside ±0.022, it made no difference. Its gap against
  `oldemb_<model>_h12` prices the rest of the new init with neither embedding
  decayed.
- **Step 2:** the best n for each surviving model, its gap against the reference
  (n = 8), and whether the new init, at its best n, beats the old.

## Log

### Implementation

2026-11-09, on branch `ame_initialization` (worktree `~/code/nano-ame-init`). No code
change; two run configs:
[oldemb_ame_h12.yaml](../../../configs/ame_init/oldemb_ame_h12.yaml) and
[oldemb_dense_h12.yaml](../../../configs/ame_init/oldemb_dense_h12.yaml).
- **Each is its `ame_init` h12 config plus the plan's `model.embedding.init_fn`
  block,** with the names and tags changed (W&B tag `ame_init_oldemb` on both).
- **Composed as `run_exp.py` does** (grid point, `check_point`), each differs from its
  `ame_init` original in exactly four keys:
  - `model.embedding.init_fn._target_`, from `torch.nn.init.xavier_normal_` to
    `src.core.model.trunc_normal_`;
  - the W&B name, the tags and the Slurm job name.

  No missing keys.
- **Built on CPU with the real init** (seed 123):

  | | `oldemb_dense_h12` | `oldemb_ame_h12` | Target |
  |---|---:|---:|---:|
  | Parameters | 183,454,464 | 1,103,599,056 | as the `ame_init` runs |
  | Embedding std, max \|w\| | 0.8798, 2.000 | 0.8798, 2.000 | 0.8796, cut at ±2 |
  | Head std | 0.00626 | 0.00626 | Xavier 0.00626 |
  | `q_bank`, `k_bank` std | — | 0.00626, 0.00626 | Xavier 0.00626 |
  | Attention and FF-in `Linear` std | 0.0100 | 0.0100 | 0.0100 |
  | FF-out std | 0.00502 | 0.00502 | 0.00502 |
  | Weight decay 0.1 / 0 (numel) | 144,820,992 / 38,633,472 | 137,743,104 / 965,855,952 | as the `ame_init` runs; the embedding is in the 0 group |
  | Logits std at init | 0.173 | 0.173 | |
  | CE against random targets | 10.840 | 10.831 | ln V = 10.826 |

- **No test run:** nothing in `src/` changed. The init itself is covered by
  `tests/test_init_defaults.py`, and the old embedding draw is `ame_main`'s.

**The extra arm, 2026-11-09:**
[oldinit_embnowd_ame_h12.yaml](../../../configs/ame_init/oldinit_embnowd_ame_h12.yaml)
and [oldinit_embnowd_dense_h12.yaml](../../../configs/ame_init/oldinit_embnowd_dense_h12.yaml).
Each is its `oldemb_` config with the old init also restored for the `Linear`s, the
head and AME's banks. Checked against the reference's own composition on `ame_main`:
`configs/bank_gates/base.yaml`, which composes `doi36p8k`, and
`configs/stage_1/dense_h12_lr_grid.yaml` set to n = 8 (its grid now lists 10-13; the
init doesn't read n).
- **Composed config:** identical except the names and tags. AME's `bank_init_std` is
  set explicitly to 1.0, the default it had on `ame_main`.
- **Init, built on CPU with seed 123:** every parameter is bit-identical to the
  reference's (dense 111 of 111, AME 171 of 171).
- **Optimizer groups:** the only difference is `embedding.weight`, decay 0.1 → 0.

**Step 2, 2026-11-10:**
[oldemb_ame_h12_n7.yaml](../../../configs/ame_init/oldemb_ame_h12_n7.yaml) and
[oldemb_ame_h12_n9.yaml](../../../configs/ame_init/oldemb_ame_h12_n9.yaml). Each is
`oldemb_ame_h12.yaml` with a new header. Composed as `run_exp.py` does, each differs
from it only in:
- `trainer.learning_rate`: 8 → 7, and 8 → 9;
- the W&B name, the Slurm job name and the tags (step-2 tag `ame_init_oldemb_step2`).

No missing keys; `check_point` passes.

**Step 2's edge run, 2026-11-10:**
[oldemb_ame_h12_n6.yaml](../../../configs/ame_init/oldemb_ame_h12_n6.yaml), a copy of
`oldemb_ame_h12_n7.yaml`, so on entropy too. Composed, it differs from n = 7 only in
`trainer.learning_rate` (7 → 6), the W&B name, the Slurm job name and the run's own
tag. No missing keys; `check_point` passes.

### Runs

Step 1 was submitted from `~/code/nano-ame-init` at `7a060dd1` (configs plus this
doc), with the plan's command, one `run_exp.py` after the other; the extra arm the
same way at `e5fe9561`, and step 2 at `8461a1da`. Each clone is under
`/lustre/pd03/plgrid/plgllmefficont3/nano/plgmstefaniak/llmrandom_cemetery/`.
Step 2 was moved to entropy at `0325e7dc` (*Decisions and issues*); its clones there
are under `/home/mstefaniak/llmrandom_cemetery/`. Its edge run, n = 6, went to
entropy at `10737ba1`.

| Run | Job | Clone | W&B | Node | Submitted | Finished | State |
|---|---|---|---|---|---|---|---|
| `oldemb_ame_h12` | 6287404 | `ame_init_oldemb_ame_h12_2026-11-09_12-14-39` | `wbq7g4i8` | r18-7 | 11-09 12:14 | 11-09 23:57 | COMPLETED, exit 0, 4 h 15 min |
| `oldemb_dense_h12` | 6287406 | `ame_init_oldemb_dense_h12_2026-11-09_12-15-01` | `l0vosr68` | r14-7 | 11-09 12:15 | 11-09 20:54 | COMPLETED, exit 0, 3 h 47 min |
| `oldinit_embnowd_ame_h12` | 6288694 | `ame_init_oldinit_embnowd_ame_h12_2026-11-09_14-18-37` | `4d5yuylb` | r10-7 | 11-09 14:18 | 11-10 01:55 | COMPLETED, exit 0, 4 h 17 min |
| `oldinit_embnowd_dense_h12` | 6288695 | `ame_init_oldinit_embnowd_dense_h12_2026-11-09_14-19-04` | `4oj9ysgv` | r14-4 | 11-09 14:19 | 11-10 01:21 | COMPLETED, exit 0, 3 h 43 min |
| `oldemb_ame_h12_n7`, LEM | 6294683 | `ame_init_oldemb_ame_h12_n7_2026-11-10_08-23-49` | | | 11-10 08:23 | 11-10 12:35 | CANCELLED while pending: moved to entropy |
| `oldemb_ame_h12_n9`, LEM | 6294685 | `ame_init_oldemb_ame_h12_n9_2026-11-10_08-24-23` | | | 11-10 08:24 | 11-10 12:35 | CANCELLED while pending: moved to entropy |
| `oldemb_ame_h12_n7`, entropy | 37646 | `ame_init_oldemb_ame_h12_n7_2026-11-10_12-23-57` | `sho4z0q5` | h100a | 11-10 12:24 | 11-10 16:19 | COMPLETED, exit 0, 3 h 55 min |
| `oldemb_ame_h12_n9`, entropy | 37648 | `ame_init_oldemb_ame_h12_n9_2026-11-10_12-24-27` | `s7r0rfv4` | h100a | 11-10 12:24 | 11-10 16:19 | COMPLETED, exit 0, 3 h 54 min |
| `oldemb_ame_h12_n6`, entropy | 37684 | `ame_init_oldemb_ame_h12_n6_2026-11-10_16-40-05` | `ht7uhbe1` | h100a | 11-10 16:40 | 11-10 20:33 | COMPLETED, exit 0, 3 h 51 min |

### Checks

- **`oldemb_dense_h12`, start, 11-09 17:15-17:20:** running on r14-7 since 17:07,
  W&B `l0vosr68`.
  - **First loss:** 10.8125, against ln V = 10.83. The reference `x0xlqroj` started
    at 11.19, from its larger head.
  - **Speed:** 3.265 s/step over steps 5-21, against the reference's 3.269.
  - **Early losses:**
    - against the reference: within ±0.2 over steps 10-21 (mean +0.016 over steps
      10-17);
    - against the Xavier-embedding run `355fr6rn`: 0.28 ahead by step 25. That run
      sits near 7.7 from step 15, while this one keeps falling.
- **`oldemb_dense_h12`, step 500, 11-09 17:45:**
  - **Health:** no NaN and no spikes; the reference spiked over steps 194-203. The
    grad-norm max after step 100 is 0.77, at step 197 (the reference's: 2.58).
    3.278 s/step.
  - **Paired gap against the reference: +0.79** over steps 4-503. It widens from
    +0.46 (steps 0-99) to +0.91-0.96 (steps 300-512). Eval loss at step 500: 5.06
    against 4.13 (+0.94).
  - **Against the Xavier-embedding run `355fr6rn`: −0.64** over steps 10-509.
  - **Block 0's attention output against the embedding:**

    | Run | Step 100 | Step 500 |
    |---|---:|---:|
    | `oldemb_dense_h12` | 8.7× | 3.2× |
    | `x0xlqroj`, the reference | 0.25× | 0.29× |
    | `355fr6rn`, Xavier embedding | 1,750× | 1,220× |

    The old draw relieves most of the drowning, not all of it. The residual stream
    entering block 5 is 14× the reference's at step 100 (25,900 against 1,820).
  - **Provisional read:** for dense, the old embedding is far better than Xavier,
    but at step 500 it is not the whole problem. The verdict waits for the end:
    `355fr6rn`'s gap narrowed from +2.13 to +1.26 over its run.
- **Unwatched from the evening of 11-09:** the session ended, and its watchers with
  it, before the dense run's spike at about 18:40. The other three runs started
  (19:41, 21:38, 21:38) and finished unwatched. All four completed, so the checks
  below were done after the fact.
- **End checks, all four runs, 11-10 08:17-08:24:** each slurm log has final_ce and
  the step-3814 checkpoint (13 GB for AME, 2.1 GB for dense), with no errors.

  | Run | Spikes | Grad-norm max after step 100 | s/step (reference) | Peak memory |
  |---|---|---|---|---|
  | `oldemb_ame_h12` | none | 0.96, step 103 | 3.768 (3.746) | 70.6 GiB |
  | `oldemb_dense_h12` | 35 steps, 1,562-1,597 | 3.54, step 1,573 | 3.277 (3.272) | 47.0 GiB |
  | `oldinit_embnowd_ame_h12` | none | 0.45, step 159 | 3.781 (3.746) | 70.4 GiB |
  | `oldinit_embnowd_dense_h12` | none | 0.56, step 131 | 3.255 (3.272) | 47.0 GiB |

  - **The dense spike:**
    - The loss rose from 3.84 at step 1,560 to 6.38 at step 1,580; the gap to
      `x0xlqroj` peaked at +2.98. It was back to +0.64 by step 1,660.
    - Just before it, the gap was +0.38 to +0.45. Over the last two quarters it is
      +0.32 and +0.28, so the step-1 verdict doesn't hinge on the spike.
    - The reference itself spiked at steps 194-203.
- **LEM's Slurm controller went unreachable, 11-10 from about 11:09.**
  - `squeue` reports "Unable to contact slurm controller", and `sacct` times out on
    `slurm-db`. Both step-2 jobs were PENDING (Priority) at 11:08:41.
  - Both step-2 watchers then exited with "left the queue". That was a watcher bug:
    it took any reply carrying an `rc=` line as an answer, so a failed query read
    as every job gone.
  - Fixed: it now queries by user and accepts only rc=0, so an unreachable
    controller reads as unknown.
  - **Back at 11:24, after about 15 min.** Both step-2 jobs survived, still PENDING
    (Priority). The watchers were re-armed with the fix at 11:25.
- **Step 2 on entropy, start, 11-10 12:32-12:34:** both running on h100a since 12:24
  (they moved from LEM: *Decisions and issues*).
  - **Same init and data across the clusters:** both match `wbq7g4i8` (n = 8, LEM)
    at step 0, before any update. Loss 10.8125 and grad norm 0.439, both identical.
  - s/step 3.400 (n = 7) and 3.389 (n = 9), as expected on entropy (3.36-3.38).
    LEM's n = 8 ran at 3.775.
  - No NaN and no spike. The grad-norm max after step 100 is 0.59 (n = 7) and 0.76
    (n = 9).
  - Early paired gaps against n = 8, not a verdict:
    - n = 7: −0.06 over steps 100-128.
    - n = 9: −0.14 over steps 100-113, after starting +0.63 behind over steps 1-9
      (its lower LR).
  - Peak memory is logged only at the end of training, so it is checked then.
  - The LEM copies, 6294683 and 6294685, were cancelled at 12:35 after checking their
    names.
- **Next check: the end of training (Maciej, 11-10 12:40):** "check them at the end
  of their training, there is no need for you to wake up more frequently". The plan's
  step-500 and mid-run checks are dropped. One watcher waits until all three entropy
  AME jobs have left the queue (step 2's pair and `pkmflin_ame_h12`), polling every
  10 min. Only a NaN ends it earlier; spikes are read at the end.
- **Step 2, end, 11-10 16:31-16:45:** both COMPLETED with exit 0 at 16:19. Each
  slurm log has final_ce and no error, and each clone has the step-3814 checkpoint
  (13 GB).

  | Run | Spikes | Grad-norm max after step 100 | s/step (LEM reference) | Peak memory |
  |---|---|---|---|---|
  | `oldemb_ame_h12_n7` | none | 0.59, step 109 | 3.400 (3.746) | 70.4 GiB |
  | `oldemb_ame_h12_n9` | none | 0.76, step 105 | 3.390 (3.746) | 70.6 GiB |

- **The cluster doesn't flatter entropy runs.** The three same-seed entropy runs of
  `doi36p8k`'s config (*Decisions and issues*, the move) were paired against it over
  the last 500 steps (W&B project `nano-ame-parity`):

  | Run | Paired gap vs `doi36p8k`, last 500 |
  |---|---:|
  | `u1zpx7lt` | +0.0157 |
  | `s65ebe41` | +0.0002 |
  | `w4hmw00o` | +0.0053 |

  Entropy runs come out level with LEM's or slightly behind, never ahead. A gain
  measured on entropy against a LEM run is therefore no cluster artifact.
- **`oldemb_ame_h12_n6`, start, 11-10 16:47:** running on h100a since 16:41, W&B
  `ht7uhbe1`.
  - Step 0 equals n = 7's exactly: loss 10.8125, grad norm 0.439.
  - 3.390 s/step, as n = 7.
  - **A warmup blip at the higher LR:** loss 14.6 at step 5 (n = 7: 8.3), with grad
    norm 4.57. It had recovered by step 10 (+0.59 against n = 7), and by step 41 it
    was −0.16 against n = 8. It falls before step 100, where the spike rule starts.
- **Next check: the end of training,** of n = 6 and `pkmflin_dense_h12` together
  (about 20:45). One watcher polls every 10 min. A NaN, or a job that leaves the
  queue before step 3,814, ends it at once.
- **`oldemb_ame_h12_n6`, end, 11-10 20:51-21:10:** COMPLETED with exit 0 at 20:33.
  The slurm log has final_ce and no error; the clone has the step-3814 checkpoint
  (13 GB).
  - **One spike, steps 746-770** (24 steps by the plan's rule). The loss rose from
    3.89 at step 735 to 4.44 at step 760, 0.52 above n = 7, and the grad norm peaked
    at 0.44 at step 755. It was back to +0.09 against n = 7 by step 790.
  - 3.400 s/step; peak memory 70.6 GiB.

### Results

**Step 1:**

| Run | final_ce | Reference | Δ final_ce | Paired gap vs reference, last 500 | Paired gap vs the `ame_init` run, last 500 | Verdict | Survives |
|---|---:|---:|---:|---:|---:|---|---|
| `oldemb_ame_h12` (`wbq7g4i8`) | 3.1465 | 3.1250 | +0.0215 | +0.0116 | −0.0027 (vs `ycwqa7k9`) | **Tie** (inside ±0.022), on the behind side: +0.007 to +0.029 per quarter | **Yes** |
| `oldemb_dense_h12` (`l0vosr68`) | 3.2324 | 2.9824 | +0.2500 | +0.2717 | −0.9992 (vs `355fr6rn`) | **Much worse.** The gap narrows from +0.81 (first quarter) to +0.28 (last); one spike at steps 1,562-1,597 | **No** |

**Step 1, extra arm:**

| Run | final_ce | Reference | Δ final_ce | Paired gap vs reference, last 500 | Paired gap vs `oldemb_<model>_h12`, last 500 | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `oldinit_embnowd_ame_h12` (`4d5yuylb`) | 3.1377 | 3.1250 | +0.0127 | +0.0117 | +0.0001 | **Tie**, on the behind side (+0.010 to +0.038 per quarter): removing the decay gains nothing |
| `oldinit_embnowd_dense_h12` (`4oj9ysgv`) | 3.0186 | 2.9824 | +0.0361 | +0.0392 | −0.2325 | **Slightly worse** (outside ±0.022): removing the decay costs dense +0.04 |

**Step 2 (AME):** n = 8 is step 1's run, on LEM; the others ran on entropy. The
reference is `doi36p8k`, final_ce 3.1250.

| Run | n | final_ce | Paired gap vs reference, last 500 | Paired gap vs n = 8, last 500 | Verdict |
|---|---:|---:|---:|---:|---|
| `oldemb_ame_h12_n7` (`sho4z0q5`) | 7 | 3.1084 | −0.0285 | −0.0401 | **Win** (beyond −0.022). Per quarter: −0.056, +0.013, −0.013, −0.027 |
| `oldemb_ame_h12` (`wbq7g4i8`) | 8 | 3.1465 | +0.0116 | — | Tie (step 1) |
| `oldemb_ame_h12_n9` (`s7r0rfv4`) | 9 | 3.1348 | −0.0055 | −0.0171 | **Tie.** Per quarter: −0.100, −0.033, −0.025, −0.009 |
| `oldemb_ame_h12_n6` (`ht7uhbe1`) | 6 | 3.1133 | −0.0173 | −0.0290 | **Tie.** Per quarter: +0.015, +0.066 (the spike at steps 746-770), +0.023, −0.011 |

Block 0's attention output against the embedding:

| Run | Step 100 | Step 500 | Step 1,000 | Step 3,800 |
|---|---:|---:|---:|---:|
| n = 6 | 1.6× | 1.2× | 1.2× | 0.96× |
| n = 7 | 2.1× | 1.1× | 1.1× | 0.83× |
| n = 8 | 1.2× | 0.94× | 1.0× | 0.83× |
| n = 9 | 0.25× | 0.42× | 0.45× | 0.42× |
| `doi36p8k`, reference | 0.29× | 0.48× | 0.44× | 0.39× |

No n drowns the embedding, which stays at 24-27 undecayed.

Best n per surviving model: **AME, n = 7**, at −0.0285 against the reference
(*Analysis, step 2*). Dense had no step 2.

### Analysis, step 1

**The bundle, decomposed.** Each link is a same-seed pair that changes one thing.
The values are last-500 paired train-loss gaps, which add up exactly along the
chain.

| Link | What changes | Dense | AME |
|---|---|---:|---:|
| reference → `oldinit_embnowd` | the embedding's decay, 0.1 → 0 | +0.0392 | +0.0117 |
| `oldinit_embnowd` → `oldemb` | the head, `Linear`s and AME's banks, old → new init | +0.2325 | −0.0001 |
| `oldemb` → the `ame_init` run | the embedding draw, old → Xavier | +0.9992 | +0.0027 |
| **reference → the `ame_init` run** | **the whole bundle** | **+1.2709** | **+0.0143** |

- **Dense, step 1 verdict:** still clearly worse, so the embedding is not the whole
  problem. Of the bundle's +1.271:
  - the Xavier draw is +0.999;
  - the new head and `Linear`s are +0.233;
  - removing the decay is +0.039.
- **AME, step 1 verdict:** a tie. The embedding draw never mattered for AME
  (+0.003), and the head, `Linear`s and banks are neutral together (−0.000). The
  bundle's whole +0.014 is the removed decay (+0.012), itself inside the tie floor.
- **The decay hypothesis (Maciej's) is not supported:**
  - Removing the embedding's decay costs dense +0.039 (outside ±0.022) and AME +0.012
    (a tie).
  - The decay shrinks the old embedding to 45% of its initial norm over the run. That
    helps a little or not at all; it never hurts.
- **AME against bank-gates.** σ = 0.01 banks alone beat `doi36p8k` by −0.105 in paired
  train loss over steps 3,434-3,814 (−0.0895 in final CE). Here, the new head,
  `Linear`s and Xavier banks together come to −0.000 against the old ones, with
  neither embedding decayed.
  - If the banks' gain carries over, the new head and `Linear`s cost about +0.10 and
    cancel it: by the plan's rule, "well above −0.09: they cost".
  - This is an inference: no run isolates the head and `Linear`s.
- **Block 0's attention output against the embedding:**

  | Run | Step 100 | Step 500 | Step 1,000 | Step 3,800 |
  |---|---:|---:|---:|---:|
  | `x0xlqroj`, dense reference | 0.25× | 0.29× | 0.24× | 0.15× |
  | `oldinit_embnowd_dense_h12` | 0.78× | 0.27× | 0.22× | 0.13× |
  | `oldemb_dense_h12` | 8.7× | 3.2× | 0.85× | 0.32× |
  | `355fr6rn`, Xavier embedding | 1,750× | 1,220× | 970× | 108× |
  | `doi36p8k`, AME reference | 0.29× | 0.48× | 0.44× | 0.39× |
  | `oldinit_embnowd_ame_h12` | 0.27× | 0.44× | 0.41× | 0.33× |
  | `oldemb_ame_h12` | 1.2× | 0.94× | 1.0× | 0.83× |
  | `ycwqa7k9`, Xavier embedding | 1,915× | 295× | 93× | 14× |

  - With the old init and no decay, both models track their references. So the
    excess in the `oldemb` runs comes from the new head and `Linear`s.
  - The excess is large early in dense (8.7× at step 100) and fades by step 1,000,
    yet dense keeps a +0.27 gap. In AME it is mild (about 1×).
  - The undecayed embeddings stay near 24 all run; the references' fall to 10.7-10.9.

### Analysis, step 2

**AME's best n is 7, at −0.0285 against the old init at its own best n**
(`doi36p8k`, n = 8). That clears the ±0.022 tie floor by 0.0065. So the new init
with the old embedding, undecayed, at its best learning rate, edges out the old init
at its best.

| n | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|
| Paired gap vs `doi36p8k`, last 500 | −0.0173 | **−0.0285** | +0.0116 | −0.0055 |
| final_ce | 3.1133 | 3.1084 | 3.1465 | 3.1348 |
| Cluster | entropy | entropy | LEM | entropy |

- **The optimum moved one step up, to n = 6-7.**
  - n = 7 beats n = 8 by −0.040; n = 9 ties n = 8 (−0.017).
  - n = 6 and 7 tie: +0.0112 for n = 6, on one cluster. n = 6 lost ground in its
    spike at steps 746-770 and was still closing on n = 7 at the end: +0.036 in the
    third quarter, +0.016 in the fourth.
- **The cluster doesn't explain the win.** n = 7 ran on entropy and the reference on
  LEM. Entropy replicates of the reference sit +0.000 to +0.016 behind it, never ahead
  (*Checks*), so a cross-cluster gain is, if anything, understated.
- **The margin is thin.** The new init with the old embedding is at least as good as
  the old init at their best learning rates, probably slightly better. It is not a
  large gain.
- **The plan's question, answered:**
  - **AME:** the Xavier embedding was the whole problem. Without it, the rest of the
    bundle tied at n = 8 (step 1) and wins slightly at its own best n.
  - **Dense:** it was not. The rest still cost +0.27 at n = 8 (step 1), so dense
    had no step 2.

### Decisions and issues

- **The embedding's weight decay, 2026-11-09 (raised by Maciej).** Step 1 restores
  only the draw; the embedding stays undecayed. The old-init references decayed it at
  0.1, and that decay sets its size. Block 0's input norm (the embedding rows looked
  up), with the old draw's norm shrunk by the decay alone, 24.35 × ∏(1 − lr_t · 0.1):

  | Step | 0 | 1,000 | 2,000 | 3,800 |
  |---|---:|---:|---:|---:|
  | `x0xlqroj`, dense, old init | 24.35 | 17.08 | 13.00 | 10.86 |
  | `doi36p8k`, AME, old init | 24.35 | 16.83 | 12.82 | 10.72 |
  | The decay alone | 24.35 | 16.87 | 12.83 | 10.74 |
  | `355fr6rn`, dense, new init (undecayed) | 0.17 | 3.97 | 4.25 | 4.27 |
  | `ycwqa7k9`, AME, new init (undecayed) | 0.17 | 1.61 | 2.17 | 2.39 |

  - The old embedding's norm follows the decay to within 1%. Undecayed, the step-1
    embedding should stay near 24 all run, 2.2× the references' at the end. Its size
    against block 0's output is the mechanism ame_init_exp.md diagnosed.
  - So the comparison against the `ame_init` run still isolates the draw (both
    undecayed). The one against the old-init reference carries the decay along with
    the head, banks and `Linear`s, and so does the AME check against bank-gates'
    −0.0895: those runs decayed the embedding.
  - **Proposed:** a third arm per model, the old draw plus decay 0.1, i.e. the old
    embedding exactly. Step 2 would then tune, per model, whichever variant has the
    lower gap, if it survives. It needs a flag on `TransformerEmbedding`: the
    exemption is a class attribute (`src/core/model.py`), with no config knob.
    Not run: Maciej chose the arm in the next entry.
- **Maciej's call, 2026-11-09:** "run also old embedding without weight decay, the
  weight decay might be the problem for the embedding". The step-1 runs already are
  the old draw without decay, on the new init. So the new arm is the old init with
  only the embedding's decay removed: a one-change test of the decay on the tuned
  reference. It is *Step 1's extra arm*, above.
- **Survival, 2026-11-10, by the plan's rule** (last-500 paired gap against the
  reference below +0.022):
  - `oldemb_ame_h12`: +0.0116, so it **survives**;
  - `oldemb_dense_h12`: +0.2717, so it **does not**.
- **Step 2 runs AME only,** at n = 7 and n = 9. There is no step 2 for dense
  because it didn't survive: its n = 8 gap is twelve times the threshold. The extra
  arm decides nothing here, as the plan says. It didn't beat the reference, so the
  AME survival gap carries no decay gain.
- **A separate pair, 2026-11-10, Maciej's request:** pkmf-scale `Linear`s and head
  on top of `kf4q4t4s` (AME, σ = 0.01 banks) and of `x0xlqroj` (dense), with the old
  embedding decayed. It tests this doc's inference that the new head and `Linear`s
  cost about +0.10 in AME. It is outside this plan and decides nothing here. Plan
  and log: [ame_init_pkmf_linears.md](ame_init_pkmf_linears.md).
- **Step 2 moves to entropy, 2026-11-10, Maciej's call:** "Move all pending jobs to
  entropy, there they should start sooner because entropy is less occupied".
  - LEM's H100 partition had 375 jobs pending and 31 running. 237 of the pending jobs
    ranked above ours, and Slurm's start estimate was 11-13 05:23.
  - Entropy's FineWeb copy is LEM's
    ([ame_efficiency_parity_report.md](ame_efficiency_parity_report.md), item 6), and
    its node is also an H100. Only the cluster changes in the configs
    (`_cluster: entropy`, `gres: gpu:h100:1`). Composed, only cluster keys differ:
    the data path, the server and the Slurm settings.
  - **The cross-cluster caveat is inside the tie floor.** Three same-seed runs of
    the production AME config on entropy ended at 3.1396, 3.1250 and 3.1328,
    against `doi36p8k`'s 3.1250 on LEM. That spread, 0.015, is below the ±0.022
    floor, so step 2's comparisons across clusters hold at the plan's resolution.
  - Expected speed there: 3.36-3.38 s/step for AME (LEM: 3.746).
  - Entropy's `/home` has a 120 GB quota shared by every job, with 17 GB in use. The
    final checkpoints add about 13 GB per AME run and 2 GB per dense run.
  - The LEM jobs are cancelled once their entropy copies have started.
- **Step 2's edge run, 2026-11-10, by the plan's rule:** the best of n = 7, 8 and 9
  is n = 7, an edge, so one more run goes a step further, to n = 6.
  - It runs on entropy, as n = 7 and 9 did, so the call between 6 and 7 stays on one
    cluster.
  - The plan allows one edge run per model, so step 2 ends with it, whichever n wins.
