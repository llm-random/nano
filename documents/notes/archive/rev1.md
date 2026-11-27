# Review 1: the nine older unpushed commits on `ame_main`

**Archived 2026-11-27 (historical).** Done on 2026-11-25. Every finding was resolved in
[review_issues.md](review_issues.md) (I1–I27), and the reviewed commits were pushed with the
fixes that day. The proposals for later are [review_upgrades.md](../review_upgrades.md).

**Status:** done, 2026-11-25. All nine commits were read against their plans and the AME
notes. No finding changes a training result:
- one finding is medium: a plan that was never committed;
- the rest are low, nits or documentation gaps;
- three bugs these commits introduced are already fixed by later commits in the same staged
  set ([below](#introduced-here-already-fixed-in-the-staged-set)).

**Scope.** The commits `7ae928a7` to `1846e395` (2026-11-07 to 11-20), on local `ame_main`,
none of them pushed. The standard-models merge (`7597f382`) is not part of this review.

**Line numbers** refer to the tree under review: `ame_main` soft-reset to `origin/ame_main`
(`e6116135`) with every unpushed change staged. The commits themselves are on branch
`ame_main_backup_2026-11-25` (`27d96faf`).

## Findings

| # | Severity | Commit | Finding |
|---|---|---|---|
| 1 | Medium | `03317133` | The stage-2 plan, cited by seven committed files, was never committed |
| 2 | Low | `157c3122` | ST gate: every layer gets the same init when the model is built under a CUDA default device |
| 3 | Low | `157c3122` | `steps_seen` is described as the number of bank-row updates; under AdamW it is the number of steps with a gradient |
| 4 | Low | `157c3122` | `train/grad_norm/banks` is never logged under FSDP2, even on one GPU |
| 5 | Low | `9d7ed99d` | `bank_gates_eval.py --attn-seqs` counts in whole micro-batches |
| 6 | Low | `9d7ed99d` | `bank_gates_lem_eval.sh` re-splits its command arguments |
| 7 | Low | `683c316b` | Stage 1c/2c run names embed the learning rate; a float LR breaks `run_exp`'s tmux step |
| 8 | Low | `6a5690c6` | `ame_ckpt_readouts.py` crashes on a checkpoint with no AME keys; its docstring goes stale with the FSDP2 fix |
| 9 | Nit | `7ae928a7`, `683c316b`, `1846e395` | A docstring that contradicts the code; the Helios account in a config; a restore hint that needs a newer git |
| 10 | Docs | `157c3122`, `03317133`, `9d7ed99d`, `6a5690c6` | The config-surface note and the scripts index miss the new options and scripts |

### 1. The stage-2 plan was never committed

- **Where:** [ame_bank_gates_stage2_plan.md](ame_bank_gates_stage2_plan.md) is untracked (last
  saved 2026-11-08 19:20). No commit on any branch has ever held it.
- **Cited by:**
  - [src/core/ame.py:82](../../../src/core/ame.py#L82);
  - the four `configs/bank_gates/stage2_*.yaml`;
  - [ame_bank_gates_plan.md](ame_bank_gates_plan.md) and
    [ame_bank_gates_stage2_report.md](ame_bank_gates_stage2_report.md).
- **Why it matters:** once pushed, all of those links are broken on GitHub and in every other
  checkout, and the spec behind `confidence_init` and the four stage-2 configs exists only on
  this laptop.
- **Fix:** commit it with the reviewed changes.

### 2. ST: the same gate init in every layer under a CUDA default device

- **Where:** [`_private_generator`](../../../src/core/ame.py#L150) and the `stem` branch of
  `_init_gate` ([ame.py:485-497](../../../src/core/ame.py#L485-L497)).
- **What:** the gate's private generator is seeded from a hash of the CPU RNG state.
  - Under `torch.device("cuda")` every other parameter draws from the CUDA generator, so the
    CPU state does not move between layers.
  - Every layer's ST gate then gets the same seed, and so the same weights.
  - `41b041db` moved the draw itself to the CPU but left the seed as it was.
- **Impact:** none on the trained ST runs.
  - `main.py` builds the model on the CPU and then moves it
    (`instantiate(cfg.model).to(device)`), so each layer sees a different CPU state.
  - Only a build under a CUDA default device hits it, such as
    `test_on_cuda_every_gate_trains_under_compile_blocks_full`.
  - ST is rejected and off by default.
  - Found by reading the code, not run: this machine has no GPU.
- **Fix:** mix the layer's index, or a per-build counter, into the seed.

### 3. `steps_seen` is not the number of bank-row updates

- **Where:** [trainer.py:227](../../../src/core/trainer.py#L227),
  [bank_gates_eval.py:8](../../scripts/bank_gates_eval.py#L8), and the bank-gates
  plan: "Token update counts", and the B1 analysis ("`c[t]` moves only in steps where t
  occurs").
- **What:** the banks are stepped by a dense `torch.optim.AdamW` with default betas: the
  memory-parallel controller's, or the backbone's in `main.py`.
  - After a token occurs, momentum keeps moving its rows for several more steps, by a
    shrinking amount.
  - So `steps_seen` counts the steps in which the row had a gradient, not its updates.
- **Impact:** the counts are right; only the wording, and any reading that treats them as
  update counts, is wrong.
- **Fix:** reword it as "steps in which the row had a gradient".

### 4. `train/grad_norm/banks` is never logged under FSDP2

- **Where:** [trainer.py:991](../../../src/core/trainer.py#L991).
- **What:** the readout is kept only under DDP or with no process group. Under FSDP2 the bank
  norm is folded into the joint norm before the all-reduce, so the rank's own value is partial
  and is skipped. That includes a one-GPU FSDP2 run, where it is already global.
- **Impact:** the bank-gate runs used `distributed: null` and logged it. FSDP2 runs, the h20
  standard configs among them, never will.
- **Fix:** all-reduce `bank_norm_sq` on its own (one extra scalar), or at least log it when
  the world size is 1.

### 5. `--attn-seqs` counts in whole micro-batches

- **Where:** [bank_gates_eval.py:235](../../scripts/bank_gates_eval.py#L235):
  `attention.collect = start < args.attn_seqs`.
- **What:** every chunk that starts below the limit is measured in full.
  - With `ame_ref32`'s micro-batch of 8, a value that is not a multiple of 8 measures up to 7
    extra sequences.
  - The report's `attn_seqs` then understates what was measured.
  - The default, 512, is exact.
- **Fix:** measure only the first `attn_seqs - start` sequences of the last chunk, or reject a
  value that is not a multiple of the micro-batch.

### 6. `bank_gates_lem_eval.sh` re-splits its command arguments

- **Where:** [line 62](../../scripts/bank_gates_lem_eval.sh#L62) rebuilds each command
  from `"$*"` and splits on ` :: `.
- **What:** an argument containing a space is split into two.
- **Fix:** write each command into the job script with its arguments quoted (`printf '%q '`).

### 7. A float learning rate breaks the submit

- **Where:** the run names `s1c_d_h20_lr${trainer.learning_rate}`
  ([dense_h20_lr_grid.yaml:84](../../../configs/stage_1c/dense_h20_lr_grid.yaml#L84)) and
  `s2c_ps_h20_lr${trainer.learning_rate}`
  ([ame_h20_pe_shift.yaml:93](../../../configs/stage_2c/ame_h20_pe_shift.yaml#L93)).
- **What:** `trainer.learning_rate=8.0` makes the name `…lr8.0`.
  - The experiment branch, and with it the tmux session name, carries the dot
    ([run_exp.py:389](../../../run_exp.py#L389)), and tmux reads a dot as a pane separator.
  - So `run_exp` dies after cloning, with nothing queued. Integers work.
  - This is the gotcha that hit every Stage 2/2b submit.
- **Fix:** the one-line tmux-name sanitizing in `run_exp.py`, still waiting for Maciej's OK.

### 8. `ame_ckpt_readouts.py`

- **Crash:** [line 102](../../scripts/ame_ckpt_readouts.py#L102) reads `rows[0]`. A
  checkpoint with no matching key, such as a dense model or a renamed module path, raises
  `IndexError` instead of saying what is wrong.
- **Stale docstring:** [line 3](../../scripts/ame_ckpt_readouts.py#L3) says the
  readouts fail under FSDP2. That stops being true with `8b78b805`, which is in the same staged
  set.
- **Fix:** an explicit "no AME layers found" error; update the docstring.

### 9. Nits

- **A docstring that contradicts the code:** [main.py:209](../../../main.py#L209),
  `build_memory_parallel_controller`.
  - It says the banks keep the model's dtype everywhere except under FSDP2, so a single-GPU run
    stays bit-comparable with one without the controller.
  - The code also picks bf16 under `autocast_bf16`, as `7ae928a7`'s new comment in `ame.py`
    correctly says.
  - The docstring predates the commit; the new comment points at it.
- **The Helios account lives in a config:**
  [helios_compile_smoke_h16_ps.yaml:50](../../../configs/stage_1c/helios_compile_smoke_h16_ps.yaml#L50)
  sets `account: plgllmefficont3-gpu-gh200`.
  - This does not break the config-hygiene rule: it is the team's grant, not a personal
    account, and `lem.yaml` holds its own grant the same way.
  - But `configs/_cluster/helios.yaml` sets no account, so every Helios config has to repeat
    it.
- **A restore hint that needs a newer git:** [run_exp.py:101](../../../run_exp.py#L101) says
  `git sparse-checkout disable` brings the left-out folders back.
  - That command needs git 2.25 or newer.
  - The clone itself uses classic sparse checkout so that it works with any git. LEM has 2.52.

### 10. Documentation gaps

- **The config surface:**
  [ame_implementation.md, The config surface](../ame_implementation.md#the-config-surface) does
  not list `bank_gate` (`157c3122`) or `confidence_init` (`03317133`).
  - Its sentence "today that is `ame` alone" is stale too: `_TYPE_OPTIONS` also has `gqa`.
  - That one comes from the standard-models merge, not these nine commits.
- **The scripts index:** the [README's scripts table](../../README.md#scripts) lists none of the
  four new scripts: `parity_numerics.py`, `bank_gates_eval.py`, `bank_gates_lem_eval.sh` and
  `ame_ckpt_readouts.py`.
  - It also promises "no side effects on import".
  - `parity_numerics.py` (lines 44-45) changes `sys.path` and the environment at import, and
    `bank_gates_eval.py` (line 42) changes `sys.path`.

## Introduced here, already fixed in the staged set

| Introduced | Bug | Fixed by |
|---|---|---|
| `157c3122` | `_init_gate("stem")` drew on the default device. Under a CUDA default device that clashed with its CPU generator, and the ST GPU test failed. The commit message lists only the CPU suite | `41b041db` |
| `157c3122` | `bank_gates_arm: ???` broke `test_every_runnable_config_composes_and_resolves`. The message's "381 passed" ran before `base.yaml` was tracked | `c96c31a4` |
| `683c316b`, `acb68ff5` | The Stage 2c runs, under FSDP2, logged no α, β or τ (80 warnings per run). `ame_ckpt_readouts.py` exists because of it | `8b78b805` |

## Commit by commit

| Commit | Date | What it does | Checked | Findings |
|---|---|---|---|---|
| `7ae928a7` | 11-07 | A comment on the q/k cast in `ame.py` | Its claims against the code at that commit: a bf16 compute copy under FSDP2 or `autocast_bf16`, refreshed from the fp32 master after every step | 9 |
| `157c3122` | 11-07 | Bank gates B1, B2, B3 and ST; token counts; readouts of the banks' grad norm and the clip factor; `configs/bank_gates/` | Formulas against the plan's mechanism table; RNG neutrality; the weight-decay and FLOP name lookups skip names a layer lacks; token counting and its all-reduce; the clip-factor formula; `base.yaml` against `stage_2/ame_h12_pe_shift.yaml` (exactly the plan's changes); the tests check what their names claim | 2, 3, 4, 10 |
| `9d7ed99d` | 11-07 | The common evaluator, the bucket analysis and the LEM wrapper | Validated on `doi36p8k` (job 6264732: 3.1252272 against 3.125227); the bank order equals the controller's; a sidecar load refreshes the compute copy; `score_bias` carries the causal mask | 5, 6, 10 |
| `c96c31a4` | 11-07 | h16 configs | `h16_base.yaml` against `stage_2b/ame_h16_pe_shift.yaml`: LR 2⁻⁸, warmup 76 (1% of 7,630), token counts, names | — |
| `683c316b` | 11-08 | h20 configs on Helios; aarch64 torch 2.7.1 and triton 3.3.1 | The hand-edited lock installed on Helios (job 25860316) and the compile smoke trained (job 25860605, W&B `7f1uzqtu`); the remaining `sympy 1.13.1` belongs to macOS's torch 2.6.0, not an orphan; 16B tokens = 15,259 steps of 512 × 2,048 | 7, 9 |
| `03317133` | 11-08 | `confidence_init` and the four stage-2 configs | Against the stage-2 plan; the option's validation; the default is stage 1's B1 bit for bit | 1, 10 |
| `acb68ff5` | 11-08 | Stage 1c/2c moved to LEM | Only the cluster, the gres and the account change | — |
| `6a5690c6` | 11-09 | `ame_ckpt_readouts.py` | Matches Stage 2b's logged readouts (per its commit message) | 8, 10 |
| `1846e395` | 11-20 | The cluster's clone leaves out `tests/`, `scripts/` and `documents/` | The generated command; the folder-name check before any push; no job script, import or code path reads the excluded folders | 9 |

## Response (Maciej, 2026-11-25)

Thanks. Every finding was checked against the staged tree. Nine of the ten hold; #7 is partly
wrong. The most consequential issue in these commits is one you missed ([below](#what-you-missed)).

| # | Verdict | Response |
|---|---|---|
| 1 | Valid | The plan is my file, kept uncommitted on purpose so far. I'll either commit it or point the seven references at [the stage-2 report](ame_bank_gates_stage2_report.md); not decided yet |
| 2 | Valid; not fixing | Confirmed: `_private_generator` hashes the CPU RNG state, and only the GPU tests build under a CUDA default device ([main.py:396](../../../main.py#L396) builds on the CPU). But your fix changes the seed on every path, so the trained ST runs would no longer reproduce their init. ST is rejected and off by default, so the docstring will note the limitation instead |
| 3 | Valid | Confirmed: the banks are stepped by a dense `AdamW` with no weight decay, so momentum keeps moving a row after its token stops occurring. Reword in [trainer.py:226-228](../../../src/core/trainer.py#L226-L228), [bank_gates_eval.py:7-8](../../scripts/bank_gates_eval.py#L7-L8) and the plan |
| 4 | Valid; optional | Logging it only at world size 1 would not cover h20, which runs FSDP2 on 4 GPUs. If fixed, the all-reduce must go on a copy: `bank_norm_sq` itself is summed into the joint norm that drives the clip. A SUM over all ranks is exact, because [`local_grad_norm_sq`](../../../src/core/memory_parallel.py#L538) returns 0 on every rank at or past `mem_parallel_size` |
| 5 | Valid; nit | — |
| 6 | Valid; low | rev2 found that the same script also exits 0 when a check fails. If the script is touched, both get fixed |
| 7 | **Partly wrong** | "Integers work" holds for Stage 1c's grid, which loads no schema. [ame_h20_pe_shift.yaml](../../../configs/stage_2c/ame_h20_pe_shift.yaml) loads `/base_schema`, which types `learning_rate` as a float: composed with `trainer.learning_rate=9`, the run name is `s2c_ps_h20_lr9.0`. So every Stage 2c submit hits it, not only a float override |
| 8 | Valid | Both parts |
| 9 | Valid | On the docstring ([main.py:207-210](../../../main.py#L207-L210)): it also says FSDP2 always means bf16, while the code needs CUDA for that. It gets reworded together with the matching comment in `ame.py:731-742` that rev2 flagged. The Helios grant and the git hint need no action |
| 10 | Valid | — |

The "Introduced here, already fixed" table: agreed.

### What you missed

- **The configs from these commits no longer reproduce their runs.**
  - After the merge, `bank_lr_ratio` defaults to 4.0. [base.yaml](../../../configs/bank_gates/base.yaml)
    and [h16_base.yaml](../../../configs/bank_gates/h16_base.yaml) set none, so they now train the
    banks at 4×.
  - Their headers still say "exactly as W&B `doi36p8k`" (`hjw7znbz`) "ran it". The same holds for
    all 15 bank-gates configs and for Stage 2c.
  - Your check compared `base.yaml` with the Stage 2 file, file to file. Composed in the tree under
    review, both give 4.0.
  - When the tree under review includes later changes, compose the configs; a diff won't show this.
  - **Decision:** the configs stay at 4, today's standard, so rerunning one gives today's recipe.
    The two headers get reworded. Reproducing or resuming a pre-merge run means setting
    `bank_lr_ratio: 1` by hand, as [llama.yaml](../../../configs/_trainer/llama.yaml) says.
- **Formatting:** `tests/test_run_exp_clone.py`, from `1846e395`, fails `black --check`.
- **Two stale texts:**
  - [ame.py:29-32](../../../src/core/ame.py#L29-L32) still calls a bank init of σ ≈ 1 load-bearing,
    which [the σ grid](ame_bank_sigma_grid.md) contradicts.
  - [ame_h20_pe_shift.yaml:46](../../../configs/stage_2c/ame_h20_pe_shift.yaml#L46) cites stale line
    numbers in `ame.py`.
