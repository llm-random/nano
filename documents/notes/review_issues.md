# Review issues: what to fix before the reviewed commits settle on ame_main

**Status (2026-11-25): applied.** Every item was applied on `ame_main` the same day, after
the commits were restored from `ame_main_backup_2026-11-25`: see
[Outcome](#outcome-2026-11-25). Still open: the review of the standard-models merge (I1),
which comes next, and the push, on Maciej's word. Analyses and ablations for later are in
[review_upgrades.md](review_upgrades.md).

- **Sources:** [rev1.md](rev1.md) and [rev2.md](rev2.md), each with Maciej's response, plus
  rev2's reply. Every item was re-checked against the staged tree.
- **The tree:**
  - `ame_main` is soft-reset to `origin/ame_main` (`e6116135`), with the 184 unpushed commits
    staged (85 files);
  - the commits themselves are on branch `ame_main_backup_2026-11-25` (`27d96faf`);
  - line numbers refer to the staged tree.
- **While the commits were staged,** no session could safely commit in `~/code/nano`: a
  commit would have taken all 85 staged files. The restore ended that.
- **Scope:**
  - the nine older commits (`7ae928a7` to `1846e395`);
  - what the standard-models merge (`7597f382`) changes about them;
  - the files they share with it.

  The merge itself has not been reviewed yet (I1).
- **Status values:**
  - **Apply**: agreed; no further decision needed.
  - **Decide**: Maciej's call.
  - **Optional**: fix or leave.
  - **No action**: decided or not worth fixing.
  - **Done**: already fixed in the staged set.

## Outcome (2026-11-25)

The decisions taken: I16 commit the plan, I17 make 1e-4 the default, I18 apply the fix,
I2 restore and fix on top; every Optional item applied. "This commit" is the one that
adds this section.

| # | Outcome | Commit |
|---|---|---|
| I1 | Next: the full review of the merge, by the session that applied these fixes | — |
| I2 | Done: `ame_main` restored from `ame_main_backup_2026-11-25` (`27d96faf`), the fixes committed on top. Nothing pushed | — |
| I3 | Done. The headers also name σ, whose default changed with I17, and the overrides that restore both (composed and checked) | `e76763b3` |
| I4 | Done. The bullet also names σ | this commit |
| I5 | Done | `3e1719af` |
| I6 | Done | `2f6cd9f7` (script), `e76763b3` (config) |
| I7 | Done, in seven places: the five listed, plus `ame.py`'s bank-gates paragraph and `definitions.py`, found while applying. Correction notes in the plan's B1 estimate and its "Token update counts" | `3e1719af`, this commit |
| I8 | Done | `765bca87` |
| I9 | Done, plus two stale lines: the AME intro, which said everything else was archived, and the efficiency-parity plan, still marked "not started". The index now covers every note | this commit |
| I10 | Done, with the stack-wide `qk_norm` and `shift` | this commit |
| I11 | Done | `3e1719af` |
| I12 | Done | `e76763b3` |
| I13 | Done; it also records the I18 fix | `e76763b3` |
| I14 | Done | `93fc12e3` |
| I15 | Done | `3e1719af` |
| I16 | Done: committed unchanged | this commit |
| I17 | Done: the default is 1e-4. **Correction:** "no test asserts the 1.0 default" held, but one test depended on it: `test_a_separate_clip_would_fail_this_gate`, a negative control, failed (9e-8 against 1e-4), because at 1e-4 the QK-norm makes the banks' gradient dominate the joint norm, so a bank-only clip equals the joint one. Its toy model now pins σ = 1, the regime its gates were built in | `765bca87` |
| I18 | Done, with a test | `913f4111` |
| I19 | Done, with a two-rank gloo test (`mem_parallel_size` 1 and 2) that fails on the old code (`nan`, the readout unset) | `ddfcd2fa` |
| I20 | Done: every tracked `.py` file passes black 25.1.0 | `93fc12e3` |
| I21 | Done, with tests. **Correction:** jobs do not read `grid_generator/`: only `run_exp.py`, `scripts/expand_config.py` and `update_pixi.py` import it, all on the submitting side. So the refused set is `src`, `resolver`, `configs` and `generated_configs` | `1c77e451` |
| I22 | Done. **Correction:** `printf '%q '` would have escaped `$BG` and `$CEM`, which the script documents as expanded on the node. Each argument is double-quoted instead, with only `\`, `"` and `` ` `` escaped. Checked offline with a fake `ssh` and `python`: argument boundaries, node-side expansion, and exit codes 1 and 0 | `2f6cd9f7` |
| I23 | Done: the report gives the number of sequences measured, and the one requested | `2f6cd9f7` |
| I24 | Done | `2f6cd9f7` |
| I25-I27 | No action, as decided | — |

## All items

| # | Issue | Solution | Status | Found by |
|---|---|---|---|---|
| [I1](#i1-the-standard-models-merge-is-unreviewed) | The standard-models merge (24 files) was reviewed by neither reviewer | Review it before training | Decide: who | The response |
| [I2](#i2-landing-the-reviewed-commits) | Landing the reviewed commits | Restore from the backup, put the fixes on top, run the checks, push on Maciej's word | Decide | This session |
| [I3](#i3-two-config-headers-promise-an-exact-reproduction) | Two config headers promise an exact reproduction that no longer holds | Reword both | Apply | rev2 #2 |
| [I4](#i4-stage_runsmd-lacks-the-bank_lr_ratio-rule) | `stage_runs.md` lacks the `bank_lr_ratio: 1` rule | One bullet | Apply | rev2's reply |
| [I5](#i5-the-banks-dtype-a-comment-and-a-docstring-contradict-the-code) | The banks' dtype: a comment and a docstring contradict the code | Reword both | Apply | rev2 #3, rev1 #9 |
| [I6](#i6-two-texts-describe-the-fsdp2-readout-bug-that-8b78b805-fixed) | Two texts describe the FSDP2 readout bug that `8b78b805` fixed | Reword both | Apply | rev1 #8, rev2 #4 |
| [I7](#i7-steps_seen-is-called-an-update-count) | `steps_seen` is called an update count, in five places | Reword; a correction note in the plan | Apply | rev1 #3, rev2's reply |
| [I8](#i8-amepys-docstring-calls-σ--1-load-bearing) | `ame.py`'s docstring calls σ ≈ 1 load-bearing | Reword to the σ grid | Apply | rev2 (outside the nine) |
| [I9](#i9-documentsreadmemd) | `documents/README.md`: a false import claim; 4 scripts and 16 notes missing | Update it | Apply | rev1 #10, rev2 #11 |
| [I10](#i10-the-config-surface-in-ame_implementationmd) | "The config surface" lacks options and is stale | Update it | Apply | rev1 #10, rev2 #12 |
| [I11](#i11-the-evaluators-docstring-claims-any-run-config) | The evaluator's docstring claims "any run config" | State the limits | Apply | rev2 #8 |
| [I12](#i12-a-stale-line-reference) | A stale line reference | Cite by name | Apply | rev2 #16 |
| [I13](#i13-stage-2cs-header-gives-the-dotted-run-name-without-the-consequence) | Stage 2c's header gives the run name `…lr<n>.0` but not its consequence | A header note | Apply | rev1 #7, rev2 #5 |
| [I14](#i14-two-files-newly-fail-black---check) | Two files newly fail `black --check` | Run black | Apply | rev2 #13, rev1's reply |
| [I15](#i15-st-the-same-init-in-every-layer-under-a-non-cpu-default-device) | ST: every layer gets the same init under a non-CPU default device | A docstring note only | Apply | rev1 #2, rev2 #10 |
| [I16](#i16-the-stage-2-plan-is-untracked) | The stage-2 plan is untracked, yet seven files cite it | Commit it, or re-point the references | Decide | rev1 #1, rev2 #1 |
| [I17](#i17-the-bank_init_std-default) | `bank_init_std` defaults to 1.0, against the evidence | Keep 1.0, or make 1e-4 the default | Decide | This session |
| [I18](#i18-the-tmux-name-fix-in-run_exppy) | `run_exp.py` can't submit a run whose name contains a dot | The one-line tmux-name fix | Decide | stage_2_report.md; rev1 #7, rev2 #5 |
| [I19](#i19-the-banks-grad-norm-under-fsdp2) | `train/grad_norm/banks` is never logged under FSDP2 | A separate all-reduce, to the spec below | Optional | rev1 #4, rev2 #9 |
| [I20](#i20-nine-files-already-failed-black-on-originame_main) | Nine files already failed black on `origin/ame_main` | Format them before the PR into `main` | Optional | rev2 #13, rev1's reply |
| [I21](#i21-sparse_exclude-given-as-a-string) | `sparse_exclude` given as a string silently excludes nothing | Reject a string; refuse folders a job needs | Optional | rev2 #6 |
| [I22](#i22-bank_gates_lem_evalsh) | `bank_gates_lem_eval.sh` re-splits arguments and exits 0 on failures | Quote the arguments; exit 1 on a failure | Optional | rev1 #6, rev2 #7 |
| [I23](#i23---attn-seqs-measures-whole-micro-batches) | `--attn-seqs` measures whole micro-batches | Measure, or report, the exact count | Optional | rev1 #5, rev2 #14 |
| [I24](#i24-ame_ckpt_readoutspy-raises-a-bare-indexerror) | `ame_ckpt_readouts.py` raises a bare `IndexError` | A clear error | Optional | rev1 #8, rev2 #15 |
| [I25](#i25-no-action-needed) | A Helios grant in a config; a git ≥ 2.25 hint; the Helios smoke at 4× | None | No action | rev1 #9, rev1's reply |
| [I26](#i26-the-old-ame-configs-train-their-banks-at-4-decided) | The old AME configs train their banks at 4× | They stay at 4; I3 and I4 cover the text | No action (decided) | rev2 #2 |
| [I27](#i27-the-rejected-bank-gate-code-decided) | The rejected bank-gate code | It stays, off by default | No action (decided) | The response |
| [D1-D3](#d1-d3-already-fixed-in-the-staged-set) | Three bugs these commits introduced | Fixed by later commits in the staged set | Done | rev1, rev2 |

## Before the commits can settle

### I1. The standard-models merge is unreviewed

- **What:** `7597f382` brought in 24 files (+1,984 / −86), the code that real training runs:
  - `bank_lr_ratio`;
  - dense QK-norm and the KV shift;
  - `n_tokens_b`;
  - `run_exp.py`'s missing-key fix;
  - the FSDP2 readout fix and the ST init fix;
  - `configs/standard/`.

  Both reviews scoped it out.
- **The evidence so far** ([standard_models_report.md](standard_models_report.md)):
  - the CPU tests pass on the merged tree (512 passed);
  - `pytest -m gpu` passes, 13 of 13;
  - the 60-step test runs of all six configs pass;
  - the h12 reproductions tie their references: dense +0.0075, the AME rerun +0.0017.
- **Solution:** a line-by-line review against
  [standard_model_improvements.md](standard_model_improvements.md) and the method notes,
  applying [the lessons below](#lessons-for-the-next-review).
- **Decide:** who reviews it: rev2, rev1, or the session applying these fixes.

### I2. Landing the reviewed commits

1. Finish the review, I1 included.
2. Restore the commits: `git -C ~/code/nano reset --soft ame_main_backup_2026-11-25`. That puts
   `ame_main` back at `27d96faf`. Working-tree edits survive, and the other sessions'
   unstaged files are left alone.
3. Apply every item marked Apply, plus whichever of Decide and Optional are chosen, as commits
   on top. Code, configs and notes all go on `ame_main`.
4. Run [the checks](#checks-after-applying).
5. Push only on Maciej's word.

**Why restore rather than re-commit:** committing the staged changes again as new commits
breaks the shared history with the branches cut from the old `ame_main`: `standard_models`,
`bank_gates_lem` and the `settle_runs_*` branches. Their later merges could then conflict.

## Apply: text fixes, no runtime effect

### I3. Two config headers promise an exact reproduction

- **Where and what:**
  - [base.yaml:3](../../configs/bank_gates/base.yaml#L3) says the file "on its own is S2-PS
    exactly as W&B doi36p8k ran it on LEM";
  - [h16_base.yaml:4](../../configs/bank_gates/h16_base.yaml#L4) says the same of S2b-PS h16
    and `hjw7znbz`.
- **Why it is wrong now:** both compose to `bank_lr_ratio` 4.0 (checked), and both runs
  trained the banks at 1.
- **Solution:** keep the sentence, and add the exception: since the standard-models merge the
  banks train at 4× by default, while doi36p8k (hjw7znbz) trained them at 1×. Set
  `trainer.bank_lr_ratio=1` to reproduce the run exactly.
  - No other config promises an exact reproduction (grep; rev2's reply).
  - Scoring is not affected: the evaluator composes these files only to build the model.
  - If I17 makes 1e-4 the default, the same sentence must name `bank_init_std` too. Both files
    leave it at the default, and both runs trained at σ = 1.0.

### I4. `stage_runs.md` lacks the `bank_lr_ratio` rule

- **Where:** [stage_runs.md:21](stage_runs.md#L21), "Every run shares the following". That is
  where a run is looked up together with its config.
- **Solution:** one bullet. Every AME run there trained its banks at the base learning rate,
  because `trainer.bank_lr_ratio` did not exist yet. Since the standard-models merge the
  default is 4.0, so a rerun needs `trainer.bank_lr_ratio=1`.
  - That rebuilds the pre-merge optimizer exactly: no bank group
    ([model.py:753](../../src/core/model.py#L753)), and the controller at the backbone's
    learning rate ([main.py:343](../../main.py#L343)).
  - So the run's checkpoint resumes, too.

### I5. The banks' dtype: a comment and a docstring contradict the code

- **[ame.py:731-742](../../src/core/ame.py#L731-L742)**, the comment `7ae928a7` added above
  the q/k cast:
  - **First bullet:** it says that without the memory-parallel controller the banks are fp32
    and the gather, the norm and the mix run in fp32.
    - That holds on one GPU or under DDP with autocast.
    - Under FSDP2 the `MixedPrecisionPolicy` (`param_dtype=torch.bfloat16`,
      [distributed_training.py:65-66](../../src/core/distributed_training.py#L65-L66)) casts
      the banks too, so the gathered rows are already bf16. rev2 confirmed this on a one-rank
      gloo mesh.
  - **Second bullet:** it says the controller's copy is bf16 "under autocast_bf16 or FSDP2".
    That holds only when CUDA is available; on the CPU the copy has the model's dtype.
- **[main.py:207-210](../../main.py#L207-L210)**, the docstring of
  `build_memory_parallel_controller`:
  - **It says:** the banks are bf16 under FSDP2, "and everywhere else in whatever dtype the
    model was built in -- which keeps the single-GPU and CPU paths bit-comparable".
  - **The code below it (`bf16_stack`):** picks bf16 when CUDA is available and the stack is
    FSDP2 or `autocast_bf16`. So a single-GPU autocast run with the controller is not
    bit-comparable with one without it.
- **Solution:** reword both to the code:
  - with the controller: bf16 when CUDA is available and the stack is FSDP2 or
    `autocast_bf16`, otherwise the model's dtype;
  - without it: FSDP2's policy if the banks are sharded, otherwise their own dtype.

  Drop the bit-comparable claim for the autocast path.
- **No runtime effect:** `q, k = q.to(v.dtype), k.to(v.dtype)`
  ([ame.py:743](../../src/core/ame.py#L743)) handles every case.

### I6. Two texts describe the FSDP2 readout bug that `8b78b805` fixed

- **Where:**
  - [ame_h20_pe_shift.yaml:25-29](../../configs/stage_2c/ame_h20_pe_shift.yaml#L25-L29);
  - [ame_ckpt_readouts.py:3-5](../scripts/ame_ckpt_readouts.py#L3-L5).

  Both say AME's readouts fail under FSDP2 and are dropped.
- **Now:** since `8b78b805`, in the staged set, α₂, β₂, τ and the horizons are logged under
  FSDP2. Only attention entropy and top-k mass are still skipped on a sharded layer.
- **Solution:** say that the runs from before the fix, all of Stage 2c, logged none of them,
  so the script remains the way to read those runs; runs made after the fix log them.

### I7. `steps_seen` is called an update count

| Where | What it says |
|---|---|
| [trainer.py:226-228](../../src/core/trainer.py#L226-L228) | `steps_seen` is "the number of updates its AME bank rows receive" |
| [trainer.py:1041-1043](../../src/core/trainer.py#L1041-L1043) | In `_fold_token_counts`: a token with a positive step count "gets one more update of its bank rows" |
| [bank_gates_eval.py:7-8](../scripts/bank_gates_eval.py#L7-L8) | "the number of optimizer steps a token occurred in, i.e. the updates its bank rows got" |
| [test_ame.py:1861-1863](../../tests/test_ame.py#L1861-L1863) | The test's name, `test_token_counts_count_occurrences_and_the_steps_a_row_is_updated_in`, and its docstring, "the updates its bank rows get" |
| [ame_bank_gates_plan.md:121-124](ame_bank_gates_plan.md#L121-L124) | B1's estimate: "about one learning rate per update", so at least 754 of a token's own updates to reach s = 0.5. The plan's "Token update counts" section ([line 241](ame_bank_gates_plan.md#L241)) is what the trainer comment cites |

- **Why it is wrong:** a dense AdamW with the default betas keeps moving a row after its token
  stops occurring. One occurrence at step 1,000 moves an element 2.5 lr at once and 25 lr in
  total ([review_upgrades.md, U1](review_upgrades.md#u1-bank-rows-under-dense-adamw)).
  `steps_seen` counts the steps in which the row had a gradient.
- **Solution:**
  - in the three code places and the test's docstring, write "the optimizer steps in which
    the token occurs, i.e. in which its bank rows have a gradient";
  - renaming the test is optional;
  - under the plan's B1 estimate, add a dated correction note pointing to U1. Notes are
    append-only, so the note is added rather than the estimate rewritten.
- **The counts themselves are right.** No logic changes.

### I8. `ame.py`'s docstring calls σ ≈ 1 load-bearing

- **Where:** [ame.py:29-32](../../src/core/ame.py#L29-L32), item 3 of "Five things here are
  load-bearing". It says bank rows initialize at σ ≈ 1, "not the 0.02 one reaches for by
  habit", because at 0.02 "attention is uniform at init, and symmetry may never break".
- **The evidence says otherwise** ([ame_bank_sigma_grid.md](ame_bank_sigma_grid.md)):
  - a plateau from 1e-5 to 0.0178, with 0.01 inside it;
  - σ = 1 at +0.097;
  - failure from about 1e-7 down.
- **Solution:** rewrite item 3 to the evidence and cite the σ grid. What is load-bearing is
  staying inside the plateau, away from both failure edges. The code default follows I17.

### I9. `documents/README.md`

- **The import claim:** [line 17](../README.md) says `scripts/` is "Importable, no side effects
  on import". Two scripts break that:
  - `parity_numerics.py` edits `sys.path` and sets `WORLD_SIZE`, `LOCAL_WORLD_SIZE`, `RANK` and
    `LOCAL_RANK` at import ([lines 43-48](../scripts/parity_numerics.py#L43-L48));
  - `bank_gates_eval.py` edits `sys.path` ([line 42](../scripts/bank_gates_eval.py#L42)) and
    imports it.
- **The scripts table** ([line 28](../README.md)) lacks four scripts:
  - `parity_numerics.py`;
  - `bank_gates_eval.py`;
  - `bank_gates_lem_eval.sh`;
  - `ame_ckpt_readouts.py`.
- **The notes index** ([line 38](../README.md)) lacks 16 tracked notes:
  - `ame_bank_gates_plan.md`, `ame_bank_gates_report.md`, `ame_bank_gates_stage2_report.md`;
  - `ame_bank_sigma_grid.md`, `ame_bf16_scalars_plan.md`, `ame_efficiency_parity_report.md`;
  - `ame_improve_runs_report.md`;
  - `ame_init_embed_ablation.md`, `ame_init_exp.md`, `ame_init_pkmf_linears.md`;
  - `doc_mask_report.md`, `lem_file_quota_request.md`, `pkmf_init_comparison.md`;
  - `settle_runs_report.md`, `stage_runs.md`, `standard_models_report.md`.
- **Solution:**
  - add the four scripts and the 16 notes;
  - for the import claim, name the exception in the README, which changes no code. The
    alternative is to move those side effects into the scripts' `main()`.

### I10. "The config surface" in `ame_implementation.md`

- **Where:** [ame_implementation.md:708-727](ame_implementation.md#L708-L727).
- **The YAML block lacks:**
  - `bank_gate`: `none | confidence | context | backoff | stem`
    ([ame.py:77](../../src/core/ame.py#L77)); rejected, off by default;
  - `confidence_init`: B1 only, default 0.05.
- **Line 725 is stale:** "today that is `ame` alone". In fact:
  - `_TYPE_OPTIONS` is `{ame, gqa}` ([attention.py:293](../../src/core/attention.py#L293));
  - since the merge, `gqa` takes `rope`, `qk_norm` and `shift`
    ([attention.py:283](../../src/core/attention.py#L283));
  - so do the stack-wide `common.attention.qk_norm` and `shift`.
- **Elsewhere:** `trainer.count_token_updates` ([trainer.py:232](../../src/core/trainer.py#L232),
  off by default) is described only in the bank-gates plan.
- **Solution:** add all of them, with the gate options marked rejected and off by default.

### I11. The evaluator's docstring claims "any run config"

- **Where:** [parity_numerics.py:7-8](../scripts/parity_numerics.py#L7-L8): "a model is
  composed from any run config".
- **It cannot score:**
  - FSDP2 runs;
  - configs with a null learning rate;
  - `n_tokens_b` configs.

  The details are in [review_upgrades.md, U5](review_upgrades.md#u5-the-common-evaluator).
- **Solution:** state these three limits in the docstring. Making the evaluator score them is
  U5.

### I12. A stale line reference

- **Where:** [ame_h20_pe_shift.yaml:46](../../configs/stage_2c/ame_h20_pe_shift.yaml#L46) cites
  `src/core/ame.py:148-163` for the ALiBi ladder.
- **Now:** `horizon_ladder` is at [ame.py:230-244](../../src/core/ame.py#L230-L244), and lines
  148-163 are now `_private_generator`.
- **Solution:** cite `horizon_ladder` by name, without line numbers.

### I13. Stage 2c's header gives the dotted run name without the consequence

- **Where:** [ame_h20_pe_shift.yaml:24](../../configs/stage_2c/ame_h20_pe_shift.yaml#L24): "The
  run name then reads s2c_ps_h20_lr<n>.0."
- **What it leaves out:** `run_exp.py` cannot finish such a submit, because tmux reads the dot
  in `tmux send -t <name>.0` as a pane separator ([stage_2_report.md](stage_2_report.md),
  decision 2).
  - The file loads `/base_schema`, so an integer learning rate becomes a float too. Composed,
    `trainer.learning_rate=9` gives `s2c_ps_h20_lr9.0`; rev1's "integers work" was wrong here.
  - The Stage 2c runs avoided it by being submitted in two halves.
- **Solution (agreed in the response and by rev2):** say so in the header, and point to the
  two-halves submit ([stage_1c_report.md](stage_1c_report.md), "Submission"). The code fix is
  I18.

### I14. Two files newly fail `black --check`

- **The files:**
  - `tests/test_run_exp_clone.py`, from `1846e395`: lines 20, 32 and 60 run over 88 columns;
  - `documents/scripts/h_balanced_flops_quotes.py`, from `c69b1ad8` (11-17), outside the nine.
- **Solution:** run black 25.1.0 on both. That is CI's version, and it is in the pixi env.
- **When it matters:** CI's [black.yml](../../.github/workflows/black.yml) runs on pushes to
  `main` and `pretraining` and on PRs into `main`. It does not gate `ame_main`, but it will gate
  the PR that takes `ame_main` into `main`.

### I15. ST: the same init in every layer under a non-CPU default device

- **Where:** `_private_generator` ([ame.py:150-159](../../src/core/ame.py#L150-L159)).
- **What:** it seeds from the CPU RNG's state, which does not advance between layers when the
  model is built under a non-CPU default device. Every layer's ST gate then gets the same init.
- **Scope:**
  - only the GPU tests build that way;
  - `main.py` builds on the CPU ([main.py:396](../../main.py#L396)), so every trained ST run is
    unaffected;
  - ST is rejected and off by default;
  - `41b041db` fixed the crash on that path, not this.
- **Solution (agreed in the response and by rev2): a docstring note only.** If distinct gates
  are ever needed, salt the seed only under a non-CPU default device, which keeps the CPU
  path's seeds
  ([review_upgrades.md, U8](review_upgrades.md#u8-the-st-gate-if-it-is-ever-revived)).

## Decide

### I16. The stage-2 plan is untracked

- **What:** `documents/notes/ame_bank_gates_stage2_plan.md` is untracked, last saved 2026-11-08
  19:20. No commit on any branch has ever held it.
- **Seven files cite it:**
  - [ame.py:82](../../src/core/ame.py#L82);
  - `configs/bank_gates/stage2_b1_std1.yaml:3` and `stage2_b1open_std001.yaml:3`;
  - `configs/bank_gates/stage2_b3_std1.yaml:2` and `stage2_none_std001.yaml:2`;
  - [ame_bank_gates_plan.md:13](ame_bank_gates_plan.md#L13);
  - [ame_bank_gates_stage2_report.md:15](ame_bank_gates_stage2_report.md#L15).
- **Why it matters:** once pushed, those links are dead on GitHub. The spec of
  `confidence_init` and of the four stage-2 configs then exists only on this laptop.
- **The options:**
  - commit it: one file (rev2);
  - re-point the seven references at the stage-2 report: edits `ame.py` and four configs in the
    reviewed set.
- **Recommendation:** commit it, if it is final. The file is Maciej's and has been kept
  uncommitted on purpose so far.

### I17. The `bank_init_std` default

- **Now:**
  - the default is 1.0 ([ame.py:336](../../src/core/ame.py#L336));
  - the standard configs set 1e-4;
  - on the σ grid, σ = 1 is +0.097.

  So a new AME config that forgets the key trains at σ = 1.
- **The options:** keep 1.0, so that every config must set it; or make 1e-4 the default. The
  second follows the same logic as `bank_lr_ratio`: the default is today's standard.
- **If it changes:**
  - every AME config that does not set the key changes its init. By the "leave them at 4"
    decision that is accepted, but it includes `bank_gates/base.yaml`, `h16_base.yaml` and
    their gate arms;
  - I3's sentence must then name σ too;
  - I8's docstring follows;
  - no test asserts the 1.0 default (checked), but rerun the CPU and GPU tests.
- **Recommendation:** 1e-4.

### I18. The tmux-name fix in `run_exp.py`

- **The proposal** ([stage_2_report.md:420-423](stage_2_report.md#L420-L423), waiting for
  Maciej's OK since 11-06):
  `tmux_name = experiment_branch_name.replace(".", "_").replace(":", "_")`. Use it in:
  - `tmux new` ([run_exp.py:389](../../run_exp.py#L389));
  - `tmux send` (lines 406, 409 and 414);
  - `wait_for_job_id` (line 411, which reads the pane at line 263);
  - `tmux attach-session` (line 419).
- **What it affects:** every schema-typed config with the learning rate in its run name: Stage
  2, 2b and 2c. Until it lands, those submits need the manual `sbatch` workaround.
- **If applied:** add a unit test on the generated tmux commands. It touches the tool every
  submit goes through.
- **Recommendation:** apply it, if those configs will be submitted through `run_exp.py`.

## Optional

### I19. The banks' grad norm under FSDP2

- **Where:** [trainer.py:986-993](../../src/core/trainer.py#L986-L993).
- **What:**
  - under FSDP2, the banks' norm is summed into the joint norm before the all-reduce, so the
    per-rank value is partial, and the readout is skipped;
  - that includes one GPU, where the value would already be global;
  - `train/clip_factor` is still logged.
- **Spec,** agreed by rev1, rev2, the response and rev2's reply:
  1. **Inside `clip_gradient`** ([trainer.py:932](../../src/core/trainer.py#L932)), which every
     rank runs every step. The all-reduce is a collective.
  2. **Reduce a copy of `bank_norm_sq`,** or reduce after line 972, where the joint sum is
     formed out of place. Reduced in place before line 972, the banks' norm would enter the
     joint norm once per rank.
  3. **SUM over all ranks.** That is exact, because `local_grad_norm_sq` returns 0 on every
     rank at or past `mem_parallel_size`
     ([memory_parallel.py:538](../../src/core/memory_parallel.py#L538)). The joint norm relies
     on the same convention.
- **Cost:** one scalar all-reduce per step, with no host sync.
- **Test:** extend the FSDP2 readout test in `tests/test_distributed.py`.
- **Recommendation:** add it before the h20 runs. Logging it only at world size 1 would not
  cover h20.

### I20. Nine files already failed black on `origin/ame_main`

- **The files** that already fail black 25.1.0 on `origin/ame_main` (`e6116135`):
  - `scripts/mfu_window_log.py`, `scripts/timers_by_rank.py`;
  - `src/definitions.py`;
  - `documents/scripts/stage_results.py`, `documents/scripts/ame_comparison_h12.py`;
  - `tests/test_mfu.py`, `tests/test_typed_config.py`, `tests/test_core_framework.py`,
    `tests/test_repo_hygiene.py`.
- **When it matters:** not for `ame_main`. It is needed before the PR into `main`, which
  `black.yml` gates.

### I21. `sparse_exclude` given as a string

- **Where:** `sparse_exclude_names` ([run_exp.py:70-84](../../run_exp.py#L70-L84)) iterates
  `names or []`.
- **What goes wrong:**
  - `sparse_exclude` has no schema type
    ([configs/_misc/default.yaml:19](../../configs/_misc/default.yaml#L19)). So
    `infrastructure.git.sparse_exclude=tests` on the command line arrives as a string and
    becomes `['t', 'e', 's', 't', 's']`: nothing is excluded, and nothing fails. The list form,
    `[tests]`, works;
  - it also accepts `src`, `configs`, `generated_configs`, `resolver` and `grid_generator`. A
    job cannot start without them, and the failure comes on the cluster, after the push.
- **Solution:** raise on a string, and refuse those folders. Add a CPU unit test beside
  `tests/test_run_exp_clone.py`.

### I22. `bank_gates_lem_eval.sh`

- **Where:** [bank_gates_lem_eval.sh:62](../scripts/bank_gates_lem_eval.sh#L62).
- **Argument splitting:** each command is rebuilt from `"$*"` and split on ` :: `, and the
  generated `python $cmd` word-splits it. So an argument with spaces arrives as four words,
  e.g. `--override "common.attention.ame={bank_gate: confidence, bank_init_std: 0.01}"` (rev2
  simulated this). No bank-gates command has needed such an argument so far.
- **Exit status:** each command ends in `|| echo "CHECK FAILED: …"`, so the job exits 0 and
  Slurm reports COMPLETED even when checks fail.
- **Solution:** write each command with its arguments quoted (`printf '%q '`); count the
  failures and `exit 1` at the end.

### I23. `--attn-seqs` measures whole micro-batches

- **Where:** [bank_gates_eval.py:235](../scripts/bank_gates_eval.py#L235) and
  [line 270](../scripts/bank_gates_eval.py#L270).
- **What:**
  - attention statistics are collected for every micro-batch (8 under `ame_ref32`) that starts
    below `--attn-seqs`;
  - the report gives `min(attn_seqs, held-out sequences)`, the value as given;
  - so up to 7 extra sequences are measured, unless the value is a multiple of 8. The default,
    512, is exact.
- **Solution:** measure only the first `attn_seqs − start` sequences of the last chunk; or
  reject a value that is not a multiple of the micro-batch; or report the measured count.

### I24. `ame_ckpt_readouts.py` raises a bare `IndexError`

- **Where:** [line 102](../scripts/ame_ckpt_readouts.py#L102), `for key in rows[0]`.
- **What:**
  - it raises `IndexError` when no AME layer matches, as with a dense model or a renamed
    module path;
  - a dense layer with QK-norm and the KV shift trips the assertion at
    [lines 75-76](../scripts/ame_ckpt_readouts.py#L75-L76) instead, which names the missing
    `slopes`.
- **Solution:** an explicit "no AME layers found" error. Supporting dense layers is
  [review_upgrades.md, U4d](review_upgrades.md#u4-readouts-for-future-analysis).

## No action

### I25. No action needed

- **The Helios grant:**
  [helios_compile_smoke_h16_ps.yaml:50](../../configs/stage_1c/helios_compile_smoke_h16_ps.yaml#L50)
  sets `account: plgllmefficont3-gpu-gh200`.
  - It is the team's grant, not a personal account, and `lem.yaml` holds its own grant the
    same way.
  - `configs/_cluster/helios.yaml` sets none, so every Helios config repeats it.
- **The git hint:** [run_exp.py:101](../../run_exp.py#L101) says
  `git sparse-checkout disable` brings the left-out folders back. That command needs git 2.25
  or newer, while the clone itself uses classic sparse checkout. Saying so is optional.
- **The Helios smoke at 4×:** `helios_compile_smoke_h16_ps.yaml` composes to 4× too. Its header
  says only that it is Stage 2b's file cut to 60 steps, which is still true, and it is a
  compile check (rev1's reply).

### I26. The old AME configs train their banks at 4× (decided)

- **The scope:**
  - 63 AME configs take the default 4.0: 17 in the reviewed set, and 46 on GitHub (10 stage
    files, 6 in `configs/ame/`, 30 in `configs/ame_tests/`);
  - only the three in `configs/standard/` set the key.
- **Decided (Maciej, 2026-11-25):** no pin to 1. I3 and I4 cover the text.

### I27. The rejected bank-gate code (decided)

- **What stays on `ame_main`, off by default:**
  - B1, B2, B3, ST and `confidence_init`;
  - their tests;
  - `configs/bank_gates/`.

  `bank_gate: none` runs the original forward.
- **Decided:** Maciej declined removing them (2026-11-25).

### D1-D3. Already fixed in the staged set

| # | Introduced in | Bug | Fixed by |
|---|---|---|---|
| D1 | `157c3122` | `_init_gate("stem")` drew on the default device; under a CUDA default device it clashed with its CPU generator, and the ST GPU test failed | `41b041db` |
| D2 | `157c3122` | `bank_gates_arm: ???` broke `test_every_runnable_config_composes_and_resolves` | `c96c31a4` |
| D3 | `683c316b`, `acb68ff5` | Stage 2c's FSDP2 runs logged no α, β or τ (80 warnings per run) | `8b78b805` |

## Checks after applying

| Check | How | Expect |
|---|---|---|
| CPU tests | `./.pixi/envs/default/bin/python -m pytest -m "not gpu and not slow and not network"` in `~/code/nano` | 512 passed. A checkout without `data/` and `data_eval/` gives 494 passed and 18 skipped |
| Every config composes | Part of the CPU tests: `test_every_runnable_config_composes_and_resolves` | — |
| black | `./.pixi/envs/default/bin/black --check` on every touched `.py` file | No change |
| GPU tests | `pytest -m gpu` on one cluster GPU, only if `src/` code changes beyond comments (I17, I19) | 13 passed |
| Distributed | `tests/test_distributed.py` on CUDA with NCCL, if I19 is applied | All pass |

## Lessons for the next review

From rev1's own account and rev2's reply. They apply to I1.
- **Compose the configs in the tree under review.** A file-to-file diff misses changed defaults,
  such as the 4× banks.
- **Compose derived values,** such as run names, instead of reasoning about them; that is how
  the dot was missed.
- **Run CI's own checks on the staged tree,** such as black.
- **Check the claims in comments and docstrings against the current code:** I5, I6, I8, I12.

## Checked and fine

- **`157c3122`** (rev1, rev2):
  - the gate formulas match the plan's mechanism table;
  - the init is RNG-neutral;
  - the weight-decay and FLOP name lookups skip names a layer lacks;
  - token counting and its all-reduce, and the clip-factor formula, are right;
  - the tests check what their names claim.
- **`9d7ed99d`:**
  - validated on `doi36p8k` (job 6264732: 3.1252272 against 3.125227);
  - the bank order equals the controller's;
  - a sidecar load refreshes the compute copy;
  - `score_bias` carries the causal mask (rev1);
  - it works end to end on the staged tree: `ame_ref32` and `ame_prod` both give 10.290798
    against the trainer's `final_ce` of 10.290797 on a CPU smoke run (rev2).
- **`base.yaml` and `h16_base.yaml`** differ from their Stage 2 and 2b sources only in the
  learning rate, the written-out warmup, `count_token_updates`, the names and tags, and, at
  h12, the time limit (rev2). The h16 warmup is 76, 1% of 7,630 steps (rev1).
- **`683c316b`:**
  - the hand-edited lock installed on Helios (job 25860316), and the compile smoke trained
    (job 25860605, W&B `7f1uzqtu`);
  - `pixi.lock` is still v6, with no orphans or duplicates;
  - linux-aarch64 resolves torch 2.7.1+cu128, triton 3.3.1 and sympy 1.14.0;
  - 16B tokens = 15,259 steps of 512 × 2,048.
- **`03317133`:** code and tests correct; the default is stage 1's B1 bit for bit.
- **`acb68ff5`:** only the cluster, the gres and the account change.
- **`6a5690c6`:** the script computes what `instrumentation` computes, and matches Stage 2b's
  logged readouts.
- **`1846e395`:**
  - the generated command is classic sparse checkout;
  - folder names are checked before any push;
  - no job script, config or module reads `tests/`, `scripts/` or `documents/`, on `ame_main`
    or on `main`.
- **Parameter counts at h12, recomputed (rev2):** B1 +14,487,552, B2 +110,736, B3 +18,432, ST
  +7,077,888.
