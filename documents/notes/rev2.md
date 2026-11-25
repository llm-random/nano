# Review of nine unpushed `ame_main` commits (rev2)

**Status (2026-11-25): reviewed. No correctness bug in the staged code. Two issues are worth
fixing before the push (1 and 2). Ten more are low and four are nits.**

- **Reviewed state:** the staged tree in `~/code/nano`. Its index equals the tree of `27d96faf`,
  the tip of `ame_main_backup_2026-11-25`; HEAD is `e6116135`. Line numbers refer to that tree.
- **Scope:** the nine commits below, one by one, against the notes that specify them. The last
  merge (`7597f382`) is not reviewed here, except where it changes what these commits do.
- This file is untracked and unstaged on purpose, so the staged set stays what is under review.

## The issues

| # | Severity | Commit | Issue | Fix |
|---:|---|---|---|---|
| 1 | fix before push | `03317133` | The stage 2 plan that the code, four configs and two notes cite was never committed | Commit `ame_bank_gates_stage2_plan.md`, or point the seven references at the stage 2 report |
| 2 | fix before push | `157c3122`, `c96c31a4`, `03317133`, `683c316b` | 17 AME configs now train their banks at 4× the learning rate. Their runs trained at 1×, and two headers still promise exact reproduction | `bank_lr_ratio: 1` in `base.yaml`, `h16_base.yaml` and `ame_h20_pe_shift.yaml` |
| 3 | low | `7ae928a7` | The comment says the banks are fp32 without the controller. Under FSDP2 they are bf16 | Reword the two bullets |
| 4 | low | `683c316b`, `6a5690c6` | Two texts describe the FSDP2 readout bug that `8b78b805` fixes in the same batch | Update both |
| 5 | low | `683c316b` | Stage 2c's run name contains a dot, which `run_exp.py`'s tmux step cannot handle | Say so in the header, or apply the pending tmux-name fix |
| 6 | low | `1846e395` | `sparse_exclude` given as a string is split into characters, with no error | Reject a string; refuse folders a job needs |
| 7 | low | `9d7ed99d` | `bank_gates_lem_eval.sh` exits 0 when a check fails | Exit non-zero if any check failed |
| 8 | low | `9d7ed99d` | The evaluator cannot score FSDP2 runs, though its docstring says "any run config" | State the limits in the docstring |
| 9 | low | `157c3122` | `train/grad_norm/banks` is not logged under FSDP2, which is the h20 setup | All-reduce the banks' norm on its own |
| 10 | low | `157c3122` | Built under a CUDA default device, every layer's ST gate gets the same init | Seed each layer differently, or leave it: ST is rejected |
| 11 | low | `9d7ed99d`, `6a5690c6` | `documents/README.md` lists neither the bank-gates notes nor the four new scripts | Add them |
| 12 | low | `157c3122`, `03317133` | "The config surface" in `ame_implementation.md` lacks `bank_gate` and `confidence_init` | Add them, marked rejected and off by default |
| 13 | nit | `1846e395` | `tests/test_run_exp_clone.py` fails `black --check` | Run black on it |
| 14 | nit | `9d7ed99d` | `bank_gates_eval.py` reports `--attn-seqs` as given but measures whole micro-batches of 8 | Report the measured count |
| 15 | nit | `6a5690c6` | `ame_ckpt_readouts.py` raises a bare `IndexError` on a checkpoint without AME layers | Print a clear message |
| 16 | nit | `683c316b` | A stale line reference, `src/core/ame.py:148-163` | It is 230-244 now; or drop the numbers |

## Per commit

| Commit | What | Result |
|---|---|---|
| `7ae928a7` | Comment on the q/k cast | Issue 3 |
| `157c3122` | Bank gates, token-update counts, grad-norm and clip readouts | The gate math, the RNG-neutral init, the counters and the clip readouts are correct. Issues 2, 9, 10, 12 |
| `9d7ed99d` | The common evaluator and the bucket analysis | Works end to end on the staged tree. Issues 7, 8, 11, 14 |
| `c96c31a4` | h16 bank-gates configs | A faithful copy of the Stage 2b file. Issue 2 |
| `683c316b` | Stage 1c/2c configs, Helios pixi change | The configs match the plan and the lock is sound. Issues 2, 4, 5, 16 |
| `03317133` | `confidence_init`, four stage 2 configs | Code and tests correct. Issues 1, 2 |
| `acb68ff5` | Stage 1c/2c move to LEM | Clean. It inherits 2, 4 and 5 from the files it edits |
| `6a5690c6` | `ame_ckpt_readouts.py` | Computes what `instrumentation` computes. Issues 4, 15 |
| `1846e395` | `run_exp.py`'s sparse cluster clone | The command is correct and no job reads the excluded folders. Issues 6, 13 |

## Details

### 1. The stage 2 plan is not in the repo

- **Where:** [ame.py:82](../../src/core/ame.py#L82); the headers of
  `configs/bank_gates/stage2_b1_std1.yaml`, `stage2_b1open_std001.yaml`, `stage2_b3_std1.yaml`
  and `stage2_none_std001.yaml`; [ame_bank_gates_plan.md](ame_bank_gates_plan.md) line 13;
  [ame_bank_gates_stage2_report.md](ame_bank_gates_stage2_report.md) line 15.
- **What:** all seven cite `documents/notes/ame_bank_gates_stage2_plan.md` as the spec for
  `03317133`. The file exists only as an untracked file in `~/code/nano` (dated 2026-11-08).
- **Checked:** `git log --all -- documents/notes/ame_bank_gates_stage2_plan.md` is empty.
- **Fix:** commit the plan, or change the references to the stage 2 report, which is tracked.

### 2. The old AME configs no longer reproduce their runs

- **Where:** all 15 files in `configs/bank_gates/`,
  [ame_h20_pe_shift.yaml](../../configs/stage_2c/ame_h20_pe_shift.yaml) and
  `configs/stage_1c/helios_compile_smoke_h16_ps.yaml`.
- **What:** none of them sets `trainer.bank_lr_ratio`, so after the merge each composes with the
  new default, 4.0. The runs they record predate the key and trained at 1: `doi36p8k`,
  `hjw7znbz`, the twelve bank-gates runs and Stage 2c's four.
- **Known, by your decision of 2026-11-21:**
  [standard_model_improvements.md](standard_model_improvements.md), step 4, says every AME
  config that does not set the key now trains its banks at 4×.
- **What that decision leaves open:**
  - [base.yaml](../../configs/bank_gates/base.yaml) line 3 and
    [h16_base.yaml](../../configs/bank_gates/h16_base.yaml) line 4 still say the file is the run
    "exactly as W&B `doi36p8k`" (or `hjw7znbz`) "ran it on LEM".
  - Every one of these files uses `mem_parallel_size`. That is the path where a checkpoint from
    before the merge resumes at 4× with no error (`configs/_trainer/llama.yaml`, lines 14-20).
- **Not affected:** scoring. The evaluator composes these files only to build the model.
- **Checked:** composed as `run_exp.py` composes them, all 17 report `bank_lr_ratio = 4.0`. The
  CPU smoke run logs `train/lr` 0.0021484 and `train/bank_lr` 0.0085938 at step 3.
- **Fix:** `bank_lr_ratio: 1` under `trainer:` in `base.yaml` and `h16_base.yaml` (the twelve arm
  files inherit it) and in `ame_h20_pe_shift.yaml`. The two smoke files are optional.

### 3. `7ae928a7`: the comment is wrong for FSDP2 without the controller

- **Where:** [ame.py:731-742](../../src/core/ame.py#L731-L742).
- **What:** the first bullet says that without the memory-parallel controller the banks are fp32
  and the gather, the norm and the mix run in fp32. That holds on a single GPU or under DDP with
  autocast. Under FSDP2 the policy in `setup_fsdp2_model` (`param_dtype=bf16`) casts the banks
  too, so the rows are already bf16, with or without the controller.
- **Also:** the second bullet says the controller's copy is "bf16 under autocast_bf16 or FSDP2".
  `build_memory_parallel_controller` requires CUDA for that; on CPU the copy has the model's
  dtype (the smoke log reads `compute dtype torch.float32`).
- **Checked:** an `AMEAttention` under `fully_shard` with that policy on a one-rank gloo mesh, no
  controller: the rows the forward gathers have dtype `bfloat16`.

### 4. Two texts still describe the FSDP2 readout bug

- **Where:** [ame_h20_pe_shift.yaml](../../configs/stage_2c/ame_h20_pe_shift.yaml), lines 25-29;
  [ame_ckpt_readouts.py](../scripts/ame_ckpt_readouts.py), lines 3-5.
- **What:** both say that under FSDP2 AME's readouts fail and are dropped. `8b78b805`, in this
  same batch, fixes that: α₂, β₂, τ and the horizons are logged on a sharded run, and only
  attention entropy and top-k mass are still skipped.
- **Fix:** update both. The script stays useful for the runs made before the fix.

### 5. Stage 2c's run name contains a dot

- **Where:** [ame_h20_pe_shift.yaml](../../configs/stage_2c/ame_h20_pe_shift.yaml), lines 19-23
  and the `metric_logger.name` at line 93.
- **What:** the schema types `learning_rate` as a float, so `trainer.learning_rate=9` gives the
  name `s2c_ps_h20_lr9.0`. `run_exp.py` uses that name as a tmux session and as a tmux target,
  and the dot breaks the target ([stage_2_report.md](stage_2_report.md), decision 2). The header
  gives the name but not the consequence.
- **Why it did not bite:** the Stage 2c runs were submitted by hand, in two halves
  ([stage_1c_report.md](stage_1c_report.md), "Submission").
- **Checked:** the composed name. The tmux failure is taken from the Stage 2 report, not re-run.
- **Fix:** say in the header that the file needs the two-halves submit, or apply the one-line
  tmux-name fix that the Stage 2 report proposes ("For Maciej", item 2). The bank-gates configs
  avoid the problem by keeping the learning rate out of the name.

### 6. `sparse_exclude` accepts a plain string

- **Where:** [run_exp.py:70-84](../../run_exp.py#L70-L84), `sparse_exclude_names`.
- **What:** `for name in names or []` iterates a string by character. So
  `infrastructure.git.sparse_exclude=tests` on the command line becomes
  `['t', 'e', 's', 't', 's']`: nothing is excluded and nothing fails. The list form,
  `sparse_exclude=[tests]`, works.
- **Also:** `src`, `configs`, `generated_configs`, `resolver` and `grid_generator` are accepted,
  though a job cannot start without them.
- **Checked:** both, by calling the function.

### 7. `bank_gates_lem_eval.sh` hides failures

- **Where:** [bank_gates_lem_eval.sh](../scripts/bank_gates_lem_eval.sh), line 62.
- **What:** every command is written as `python … || echo "CHECK FAILED: …"`. The job's last
  command is then an `echo`, so the job exits 0 and Slurm reports COMPLETED even if every check
  failed. A failure shows only in the log.
- **Fix:** count the failures and `exit 1` at the end if there were any.

### 8. The evaluator is narrower than "composed from any run config"

- **Where:** [parity_numerics.py](../scripts/parity_numerics.py): the docstring at lines 7-8,
  `VARIANTS` at 60-66, `build` at 123-161, `load_checkpoint` at 177-231.
- **What it cannot score:**

  | Case | Why | Checked |
  |---|---|---|
  | An FSDP2 run (Stage 2c, `standard/ame_h20`) | `ame_ref32` does not null `trainer.distributed`, so on a GPU the build reaches `setup_fsdp2_model` with no process group. `load_checkpoint` reads only the single-file checkpoint and the rank-0 sidecar; an FSDP2 run writes a distributed checkpoint and one sidecar per rank | Read from the code; not run (no CUDA here) |
  | A config with a null learning rate (`ame_h20_pe_shift.yaml`) | `float(cfg.trainer.learning_rate)` at line 146. `--override trainer.learning_rate=<n>` gets past it | Run: `TypeError` |
  | A config that gives its length as `n_tokens_b` | `build` does not call `resolve_n_steps`, so the warmup's `${trainer.n_steps}` resolves on null. No config on `ame_main` does this today | Run: `InterpolationResolutionError` |

- **Fix:** state the limits in the docstring. Scoring h20 would need more: null
  `trainer.distributed` in `ame_ref32`, and a loader for sharded checkpoints.

### 9. `train/grad_norm/banks` is not logged under FSDP2

- **Where:** [trainer.py:986-993](../../src/core/trainer.py#L986-L993).
- **What:** under FSDP2 the banks' norm is summed into the joint norm before the all-reduce, so
  the per-rank value is partial, and the code skips the readout instead of logging it wrong.
  That is correct. But FSDP2 with `mem_parallel_size: 4` is the standard h20 setup, at
  σ = 1e-4, where the banks' share of the clip is the readout the σ grid relied on.
  `train/clip_factor` is still logged there.
- **Fix:** all-reduce `bank_norm_sq` on its own in that branch. It is one scalar per step.

### 10. The ST gate's init is the same in every layer under a CUDA default device

- **Where:** [ame.py:150-159](../../src/core/ame.py#L150-L159), `_private_generator`.
- **What:** it seeds from the CPU RNG's state, on the assumption that the state differs at each
  layer. When the model is built under `with torch.device("cuda")`, as the GPU tests build it,
  every other draw uses the CUDA generator and the CPU state never advances. Each layer then
  gets the same seed and the same gate weights.
- **Scope:** ST only, and only on that build path. `main.py` builds on the CPU and then moves
  the model, so every ST run was unaffected. `41b041db` fixed the crash on the CUDA path, not
  this.
- **Checked:** by simulation on the CPU: with the CPU stream held still between layers, two
  private generators give the same draw. Not run on a GPU.

### 11. `documents/README.md` does not index this work

- **Where:** [documents/README.md](../README.md): the notes index, and the `scripts/` table at
  lines 28-36.
- **What:**
  - The notes index lacks `ame_bank_gates_plan.md`, `ame_bank_gates_report.md` and
    `ame_bank_gates_stage2_report.md`. Sixteen tracked notes are missing in all; the rest belong
    to other work.
  - The scripts table lacks `parity_numerics.py`, `bank_gates_eval.py`, `bank_gates_lem_eval.sh`
    and `ame_ckpt_readouts.py`.
  - Line 17 describes the folder as "Importable, no side effects on import".
    `parity_numerics.py` sets four environment variables and edits `sys.path` on import (lines
    43-48), and `bank_gates_eval.py` imports it.

### 12. "The config surface" lacks the two new options

- **Where:** [ame_implementation.md](ame_implementation.md), "The config surface".
- **What:** the YAML block lists the `attention.ame` options without `bank_gate` and
  `confidence_init`. `trainer.count_token_updates` is described only in the bank-gates plan.

### 13-16. Nits

- **13.** [tests/test_run_exp_clone.py](../../tests/test_run_exp_clone.py): `black --check` would
  reformat three statements (lines 20, 32 and 60, each over 88 columns). It is the only file from
  these nine commits that fails. Nine files already fail on `origin/ame_main`
  ([ame_bank_gates_report.md](ame_bank_gates_report.md), decision 6).
- **14.** [bank_gates_eval.py](../scripts/bank_gates_eval.py): line 235 collects attention
  statistics for every micro-batch of 8 that starts below `--attn-seqs`, and line 270 reports
  `--attn-seqs` as given. The two differ unless it is a multiple of 8. The default, 512, is.
- **15.** [ame_ckpt_readouts.py](../scripts/ame_ckpt_readouts.py): `rows[0]` at line 102 raises
  `IndexError` on a checkpoint with no AME layers. A dense layer with QK-norm and the KV shift
  has `tau`, `alpha` and `beta` but no `slopes`, so it trips the assertion at lines 75-76
  instead.
- **16.** [ame_h20_pe_shift.yaml](../../configs/stage_2c/ame_h20_pe_shift.yaml), line 46, cites
  `src/core/ame.py:148-163` for the ALiBi ladder. `horizon_ladder` is at lines 230-244 now.

## What was checked

All of it ran on an export of the staged tree, outside `~/code/nano`.

| Check | Result |
|---|---|
| CPU gate, `pytest -m "not gpu and not slow and not network"` | **494 passed**, 18 skipped (they need the gitignored dummy dataset), 0 failed. `tests/test_ame.py` and `tests/test_run_exp_clone.py` alone: 144 passed |
| Bank-gates CPU smoke, `smoke_local.yaml` with `bank_gate=confidence`, through `main.py` | Completes: `final_ce`, checkpoint, bank sidecar, `token_counts.pt`; trust reads 0.05 at step 0; `train/clip_factor` and `train/grad_norm/banks` are logged |
| The evaluator on that checkpoint: `dump_data`, `score`, then `bank_gates_eval.py` | `ame_ref32` and `ame_prod` both 10.290798, against the trainer's `final_ce` 10.290797. The bucket table is written |
| `pixi.lock` | Still v6. Every package listed for the three platforms has an entry, with no orphans or duplicates. linux-aarch64 resolves torch 2.7.1+cu128, triton 3.3.1 and sympy 1.14.0 |
| Configs | All 18 files from these commits compose. `base.yaml` and `h16_base.yaml` differ from their Stage 2 and 2b sources only in the learning rate, the written-out warmup, `count_token_updates`, the names and tags, and at h12 the time limit. Stage 1c/2c match the plan in [ame_experiments.md](ame_experiments.md): 15,259 steps, warmup 152, micro-batch 16 × 8 on 4 GPUs, `mem_parallel_size: 4` |
| The sparse clone | No job script, config or module reads `tests/`, `scripts/` or `documents/`, on `ame_main` or on `main`. The generated command is the classic sparse checkout the commit describes |
| The plan's parameter counts | Recomputed: B1 +14,487,552, B2 +110,736, B3 +18,432, ST +7,077,888 at h12 |

**Not checked:** the GPU tests (no CUDA here), anything on a cluster, and the numbers in the
reports.

## Already fixed in the same batch

- **The ST gate's init crashed under a CUDA default device** (`157c3122`): the draw landed on
  CUDA while its generator is a CPU one. `41b041db` fixes it. Issue 10 is what remains.
- **No readout under FSDP2:** `8b78b805` fixes it. Issue 4 is the text that still describes it.

## Outside these nine commits, seen in passing

- `documents/scripts/h_balanced_flops_quotes.py` is the other file in the staged set that newly
  fails `black --check`.
- The module docstring in [ame.py:29-32](../../src/core/ame.py#L29-L32) still calls a bank init
  of σ ≈ 1 load-bearing. [ame_bank_sigma_grid.md](ame_bank_sigma_grid.md) found the opposite:
  a plateau from 1e-5 to 0.0178, and +0.097 at σ = 1.

## Response (Maciej, 2026-11-25)

Thanks: a careful review, and every claim checked out against the staged tree. One correction
of scope, on #2, and a few notes on the fixes.

| # | Verdict | Response |
|---|---|---|
| 1 | Valid | The plan is my file, kept uncommitted on purpose so far. Commit it or point the seven references at the stage-2 report: not decided yet |
| 2 | **Valid, and wider than scoped; fix declined** | The configs stay at 4, and the two headers get reworded ([below](#2-is-wider-than-the-nine-commits)) |
| 3 | Valid | rev1 found the matching contradiction in `build_memory_parallel_controller`'s docstring ([main.py:207-210](../../main.py#L207-L210)). Both get reworded together |
| 4 | Valid | — |
| 5 | Valid | Right: `/base_schema` makes even an integer `9` into `lr9.0` (rev1 claimed integers work). The header will say so; the tmux-name fix in `run_exp.py` still waits for my OK |
| 6 | Valid; low | Confirmed: `sparse_exclude` has no schema type, so a command-line string reaches the function as a string |
| 7 | Valid; low | — |
| 8 | Valid | The docstring will state the limits |
| 9 | Valid; optional | Your fix is right, with two conditions. Reduce a copy: `bank_norm_sq` also feeds the joint norm that drives the clip. And sum over all ranks, which is exact because `local_grad_norm_sq` returns 0 on every rank at or past `mem_parallel_size` ([memory_parallel.py:538](../../src/core/memory_parallel.py#L538)) |
| 10 | Valid; not fixing | Any change to the seed also changes ST's init on the CPU path, so the trained ST runs would stop reproducing. ST is off by default; the docstring will note the limitation |
| 11, 12 | Valid | — |
| 13 | Valid; not blocking | [black.yml](../../.github/workflows/black.yml) runs only on pushes to `main` and `pretraining` and on PRs into `main`, so it doesn't gate `ame_main`. The two files get formatted anyway |
| 14, 15, 16 | Valid; nits | — |
| Outside | Valid | The σ docstring gets reworded to the σ grid's evidence. The default stays 1.0 for now |

### #2 is wider than the nine commits

Every AME config that sets no `bank_lr_ratio` now composes to 4.0, not just the 17 from these
commits. 46 more are already on GitHub:
- `configs/stage_1`, `stage_1b`, `stage_2` and `stage_2b`: 10 files that record 1× runs.
  Checked: `stage_2/ame_h12_pe_shift.yaml` composes to 4.0.
- `configs/ame/` (6) and `configs/ame_tests/` (30): speed benchmarks.

**Decision: no pin.** 4 is the standard, and every old config stays at the default, so rerunning
one gives today's recipe, not the recorded run. Instead:
- the two headers that promise an exact reproduction get reworded:
  [base.yaml:3](../../configs/bank_gates/base.yaml#L3) and
  [h16_base.yaml:4](../../configs/bank_gates/h16_base.yaml#L4). No other config makes that promise;
- reproducing or resuming a pre-merge run means setting `bank_lr_ratio: 1` by hand, as
  [llama.yaml](../../configs/_trainer/llama.yaml) says.

### Found by rev1, not here

- `steps_seen` is described as a count of bank-row updates. With AdamW's momentum it counts the
  steps in which the row had a gradient (rev1 #3).
- `bank_gates_lem_eval.sh` re-splits arguments that contain spaces (rev1 #6).
- `build_memory_parallel_controller`'s docstring contradicts its code (rev1 #9).

## Reply (rev2 reviewer, 2026-11-25)

**The response holds: I checked its new claims against the staged tree and found nothing
wrong. I agree with the decisions on #5, #9 and #10 (a header note for the dot, the banks'
grad-norm readout optional, no ST seed fix), and #2 is settled as no pin. The three rev1 findings
are valid.**

### On the response

| Point | Checked | Position |
|---|---|---|
| #1, not decided | — | Committing the plan is one file. Re-pointing the references edits `src/core/ame.py` and four configs in the reviewed set |
| #2, no pin | Composed every runnable config. 63 AME files take the default 4.0 and only the three in `configs/standard/` set the key. 46 of the 63 are on GitHub: 10 stage files, 6 in `configs/ame/`, 30 in `configs/ame_tests/`. Only `base.yaml:3` and `h16_base.yaml:4` promise an exact reproduction | The counts and the "no other config" claim hold, and the decision settles it. Two facts for whoever reproduces a run by hand: `bank_lr_ratio: 1` is exactly the pre-merge build (no bank group in `parameter_groups`, [model.py:753](../../src/core/model.py#L753); the controller steps at the backbone's lr), so an old checkpoint's optimizer state fits again. And [stage_runs.md](stage_runs.md), where a run is looked up with its config, does not carry the rule; one line under "Every run shares the following" would |
| #9, reduce a copy | [trainer.py:969-974](../../src/core/trainer.py#L969-L974): the sum at line 972 is out of place, and the joint all-reduce follows at 974 | Agree with both conditions. The hazard is the order: reduced in place before line 972, the banks' norm would enter the joint norm once per rank. A copy is safe wherever it sits. A third condition: keep it in `clip_gradient`, which every rank runs every step, because it is a collective |
| #10, not fixing | — | Agree. If the GPU tests ever need distinct gates, a salt added only under a non-CPU default device would leave the CPU path's seeds as they are |
| #5, header note | — | Agree |
| #13, black | [black.yml](../../.github/workflows/black.yml): pushes to `main` and `pretraining`, PRs into `main` | Agree. It will gate the PR that takes `ame_main` into `main` |

### On the three rev1 findings

| Finding | Checked | Note |
|---|---|---|
| `steps_seen` is not an update count (rev1 #3) | A dense `AdamW` with default betas, one element, a gradient in one step only | Valid. I saw it and left it out as wording; that was wrong. One occurrence moves an element by more than one learning rate: 2.5 lr at once and 25 lr in all when it comes at step 1,000 (3.1 and 31 at step 3,000), with 90% of it inside 23 steps. So the plan's B1 estimate, one lr per update and 754 updates to reach s = 0.5, fits a token that occurs in every step, not a rare one, whose trust moves many lr per occurrence. The same wording is in two more places than the response lists: [trainer.py:1041-1043](../../src/core/trainer.py#L1041-L1043) and the test docstring at [test_ame.py:1863](../../tests/test_ame.py#L1863) |
| The eval script re-splits arguments (rev1 #6) | Simulated line 62 with `--override "common.attention.ame={bank_gate: confidence, bank_init_std: 0.01}"`: the node sees it as four words | Valid. I noticed the `"$*"` and did not list it. No bank-gates command so far needed an argument with a space |
| The controller's docstring contradicts its code (rev1 #9) | [main.py:207-210](../../main.py#L207-L210) against the `bf16_stack` line below it | Valid. I read that docstring while checking #3 and did not flag it |
