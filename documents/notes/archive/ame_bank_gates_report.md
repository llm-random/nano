# AME bank gates — report

**Archived 2026-11-27 (historical).** Scored on 2026-11-08. The replicates it proposes were
never submitted: stage 2 found that the bank init was all of the gain, and every gate was
rejected. **Still only here:** "All runs, stages 1 and 2", the twelve W&B runs of both
stages. **Today:** [standard_model_improvements.md](../standard_model_improvements.md).

**Status (2026-11-08 15:05): both waves are trained and scored. No job from this plan is
running.**
- **h12, best gate:** ST, B2 and B3 tie for best at the plan's noise floor (3.0191,
  3.0355, 3.0358; production 3.1252). B1 is last (3.1420).
- **h16, best gate:** ST, resolved against the h12 floor used as a proxy (2.7530; B3
  2.8112, B2 2.8183, B1 2.8460; reference `hjw7znbz` 2.8610). The ranking agrees with
  h12's.
- **Rare rows:** a rarely updated row is less influential only in B1, at both scales. At
  h16 that no longer costs B1 its rare tokens.
- **Replicates:** proposed for the ties at both scales ([h12](#results),
  [h16](#results-1)), not submitted.
- **Next:** [stage 2](ame_bank_gates_stage2_report.md) separates the σ = 0.01 init from
  the gates.

The plan is [ame_bank_gates_plan.md](ame_bank_gates_plan.md). This file is its record:
every job, check, result and decision, each with its reason. Times are CET.

## All runs, stages 1 and 2

Twelve runs, all in the W&B project **`ideas_cv/nano-ame`**, all on LEM with one H100
each, and all finished.
- Score is the common evaluator's `ame_ref32` on the held-out set; lower is better.
- σ is the banks' starting spread. Production uses σ = 1.
- The references are production runs, not part of this task: `doi36p8k` for h12
  (3.125227) and `hjw7znbz` for h16 (2.860982).
- The local CPU smoke runs logged to stdout, and the scoring jobs do not log to W&B.

| Stage | W&B name | ID | Setup | Slurm job | Runtime (W&B) | `ame_ref32` | vs reference |
|---|---|---|---|---|---:|---:|---:|
| 1, h12 (4B tokens) | `bank_gates_b1` | `621sjn71` | B1 confidence gate, σ = 0.01 | 6264723_0 | 4.35 h | 3.141958 | +0.0167 |
| | `bank_gates_b2` | `kgk0prvf` | B2 context gate, σ = 0.01 | 6264725_0 | 4.30 h | 3.035505 | −0.0897 |
| | `bank_gates_b3` | `5w6p64bv` | B3 backoff gate, σ = 0.01 | 6264726_0 | 4.23 h | 3.035837 | −0.0894 |
| | `bank_gates_stem` | `l54pyyqp` | ST stem gate, σ = 0.01 | 6264733_0 | 4.35 h | 3.019112 | −0.1061 |
| 1, h16 (8B tokens) | `bank_gates_h16_b1` | `h85djd57` | B1, σ = 0.01 | 6264776_0 | 15.48 h | 2.845979 | −0.0150 |
| | `bank_gates_h16_b2` | `jwirkswo` | B2, σ = 0.01 | 6264777_0 | 15.17 h | 2.818295 | −0.0427 |
| | `bank_gates_h16_b3` | `3yc7haqu` | B3, σ = 0.01 | 6264778_0 | 15.04 h | 2.811172 | −0.0498 |
| | `bank_gates_h16_stem` | `ug8g2i2x` | ST, σ = 0.01 | 6264779_0 | 15.57 h | 2.753039 | −0.1079 |
| 2, h12 (4B tokens) | `bank_gates_stage2_none_std001` | `kf4q4t4s` | no gate, σ = 0.01 | 6272065_0 | 4.28 h | **3.035771** | −0.0895 |
| | `bank_gates_stage2_b1_std1` | `1dmp8l8w` | B1, σ = 1 | 6272075_0 | 4.43 h | 3.211324 | +0.0861 |
| | `bank_gates_stage2_b3_std1` | `5hzd2fte` | B3, σ = 1 | 6272076_0 | 4.30 h | 3.117312 | −0.0079 |
| | `bank_gates_stage2_b1open_std001` | `jc9flzsz` | B1 with trust starting at 0.95, σ = 0.01 | 6272078_0 | 4.40 h | 3.107609 | −0.0176 |

Each run's cemetery clone is `<W&B name>_<timestamp>`. Stage 2's record is
[ame_bank_gates_stage2_report.md](ame_bank_gates_stage2_report.md).

## Setup

| | |
|---|---|
| Question | Which way of trusting a bank row gives the lowest final loss at h12 and 4B tokens, and which makes rarely updated rows least influential without costing loss? |
| Design | Four runs, one per mechanism, all with banks drawn at σ = 0.01 and otherwise identical: B1 per-row confidence, B2 context gate, B3 shared backoff vector, ST STEM gate. No control run (Maciej, 2026-11-07). The production reference is S2-PS, W&B `doi36p8k`, common evaluator 3.125227 |
| Code | Commit **`9d7ed99d`** on `ame_main` for every run: the mechanism in `157c3122`, the evaluator in `9d7ed99d`. Worktree `~/code/nano-bank-gates`, branch `bank_gates_lem`, equal to `ame_main` at every submit |
| Configs | [configs/bank_gates/](../../../configs/bank_gates/): `base.yaml` (S2-PS at n = 8, a copy of the Stage 2 file), `b1.yaml`, `b2.yaml`, `b3.yaml`, `stem.yaml`, and `smoke_local.yaml` for the CPU smoke runs |
| Cluster, recipe | LEM, one H100 per run, node r16-11 excluded, 7 h limit. h12 AME as S2-PS: 3,815 steps × 1,048,576 tokens = 4.00B, seed 123, n = 8, warmup 38, micro-batch 32 × 16, Tier 2 banks, autocast bf16, `compile_blocks_full` |
| Evaluation | The common evaluator ([parity_numerics.py](../../scripts/parity_numerics.py) `score`, `ame_ref32`) on the 8,192-sequence held-out set; the bucket analysis ([bank_gates_eval.py](../../scripts/bank_gates_eval.py)); both on LEM through [bank_gates_lem_eval.sh](../../scripts/bank_gates_lem_eval.sh) |

## Decisions

1. **Every gate starts every shared weight from the same numbers as `none` at the same
   seed.** Nothing an arm adds draws from the global RNG:
   - B1 widens the bank after the usual draw;
   - B2's and ST's gates are built on the meta device and then filled;
   - B3's base vectors are zeros;
   - ST's gate, the one random init, is drawn from a private generator seeded from a
     hash of the global RNG state, which it does not advance.

   **Why:** otherwise each arm would also start from a different draw of every later
   weight (the next blocks' projections, the head), as if it had another seed. The noise
   floor (±0.019) comes from same-seed replicates, so the arms must share the init for it
   to apply. A test pins it: after each arm's constructor the RNG state equals `none`'s,
   and every shared tensor is equal.
2. **B3's base vectors are cast to the rows' dtype before the add.** Under Tier 2 the
   rows arrive bf16, and an fp32 base would have promoted B3's whole prep (add, norm,
   mix) to fp32 while the other arms run it in bf16. The cast keeps the prep precision
   equal across arms.
3. **B2 and ST log their gate's weights, not its values.** Their gates read x, and the
   trainer's readout hook sees only token ids, so the plan lists no in-training readout
   for them. Each logs `gate_weight/rms`, and B2 also its bias mean, min and max. These
   are parameter readouts, as cheap as B1's and B3's. The gates' values on real
   activations come from the bucket analysis.
4. **The evaluator is a trimmed port, plus a LEM submit script.** `parity_numerics.py`
   on `ame_main` carries `parity_audit`'s `score` and `dump_data` checks and the
   helpers they use, unchanged except that a model is composed from any run config
   (`--config`, `--override`), so it builds every `bank_gate`. It needs none of
   `parity_audit`'s changes to `main.py`, `ame.py` or `model.py`, since `ame_ref32`
   uses only standard overrides. `parity_score_runs.sh` and `parity_submit.sh` target
   entropy's h100a; their LEM counterpart is `bank_gates_lem_eval.sh`.
5. **The held-out tensor is the one the references were scored on.** Entropy's parity
   dump (`~/parity/data/parity_batches.pt`, sha256 `3265f589…`) was copied to LEM, so
   every arm is scored on the same bytes as `doi36p8k`'s 3.125227. The validation job
   also rebuilds the held-out set from LEM's own FineWeb copy and compares the two.
6. **`black --check` passes on every file this plan adds or changes, but not
   tree-wide.** Nine files fail it on `ame_main` already. All predate this plan, and
   eight of them it does not touch:
   - `scripts/timers_by_rank.py`, `scripts/mfu_window_log.py`;
   - `tests/test_typed_config.py`, `tests/test_mfu.py`, `tests/test_repo_hygiene.py`,
     `tests/test_core_framework.py`;
   - `documents/scripts/stage_results.py`, `documents/scripts/ame_comparison_h12.py`;
   - `src/definitions.py`, whose only change here is a 3-line hunk that is itself
     black-clean.

   Reformatting them would rewrite other sessions' files and invite merge conflicts,
   so they are left for their owners.
7. **An h16 wave, by Maciej's request (2026-11-07, about 22:30).** Four more runs, one per
   gate: "Each is AME h16 at n = 8 (LR 2^-8) on 8B tokens, on LEM. Run them alongside the
   h12 wave without disturbing it, and keep finishing the h12 scoring and report as
   planned."
   - The plan's [h16 extension](ame_bank_gates_plan.md#h16-extension-maciej-2026-11-07)
     specifies them.
   - The reference is S2b-PS h16 at n = 8 (`hjw7znbz`).
   - The h12 noise floor does not transfer to h16 and 8B. h16 gaps are reported beside it
     as a stated proxy, with that caveat ([The h16 wave](#the-h16-wave)).
8. **The rare-row reading rests on the u = 101-1000 bucket (2026-11-08).** The held-out
   set's rarer buckets are too thin or too narrow to read:
   - u = 0 is 5 positions;
   - u = 1-10 is 99% one token, ' COVID', which the training stream almost never
     contains;
   - half of u = 11-100 comes from five documents.

   The attention sample (the first 512 sequences) holds 4, 61 and 16 positions of them.
   u = 101-1000 has 17,812 positions over 4,026 ids and 4,575 sequences. The thin buckets
   are still reported, labelled ([Results](#results)).

## Implementation and local checks

Commits `157c3122` (mechanism, trainer, tests, configs) and `9d7ed99d` (evaluator), both
fast-forwarded into `ame_main`.

- **`bank_gate: none` is bit-identical to the previous layer** (`7ae928a7`'s `ame.py`,
  loaded side by side). The state dict, the RNG state after construction, the fp32 and
  bf16 forward and every gradient are equal, at σ = 1, at σ = 0.01 and with the shift
  and sink off.
- **Tests:** 381 passed, 18 skipped (`pytest -m "not gpu and not slow and not network"`).
  The new ones cover:
  - every gate against a literal transcription of its formula;
  - B1's trust at 0.05 after `main.py`'s build path, its scores equal to s_q·s_k times
    the ungated scores, its gradient only for ids in the batch, and the fused
    memory-parallel path at width 780;
  - B2 at sigmoid(3), its scores equal to g_i times the ungated scores, and its bias
    undecayed;
  - B3 equal to the ungated forward at init, its base trained, undecayed and outside
    the FLOP count;
  - ST at about half the ungated logit spread;
  - the id-only invariants for `none`, `confidence` and `backoff`, and their break for
    `context` and `stem`;
  - the diagnostics guard, both backends, finite bf16 gradients, and every gate's
    readout through the trainer;
  - token counts, and the clip's bank norm and factor;
  - one `gpu`-marked test that each gate trains under `compile_blocks_full`. It is not
    run here (no GPU); the full runs' first steps exercise that path.
- **Startup parameter counts** (configs composed as `run_exp.py` composes them, model
  built on the meta device) match the plan's table exactly:

  | Arm | Parameters | GFLOP/token (seq 2048) |
  |---|---:|---:|
  | none (`base.yaml`, = `doi36p8k`) | 1,103,599,056 | 1.0529 |
  | B1 | 1,118,086,608 | 1.0529 |
  | B2 | 1,103,709,792 | 1.0536 |
  | B3 | 1,103,617,488 | 1.0529 |
  | ST | 1,110,676,944 | 1.0954 (+4.0%) |

- **CPU smoke run of each arm** (`smoke_local.yaml`: h4 at vocab 50,304, Tier 2 banks,
  4 steps of 4 × 32 tokens, final checkpoint on). Each one ran to completion:
  - it wrote `final_ce`, the model checkpoint, the bank sidecar and `token_counts.pt`;
  - the four `token_counts.pt` are identical: 512 occurrences = 4 steps × 4 × 32;
  - at step 0, B1's trust read 0.0500, B2's gate bias 3.0 with weight RMS 0, and B3's
    base norms 0;
  - entropy and top-k were logged for B1 and B3 and, by design, not for B2 and ST.
- **The ported evaluator on the smoke checkpoints** reproduces each run's own
  `final_ce` within 1e-6, through `ame_ref32` (banks loaded from the sidecar's fp32
  masters into `nn.Embedding`) and through `ame_prod` (through the controller). On CPU
  both paths run fp32, so this checks the loading of every gate's parameters, not the
  evaluator's precision:

  | Arm | trainer `final_ce` | `ame_ref32` | `ame_prod` |
  |---|---:|---:|---:|
  | B1 | 10.290774 | 10.290774 | 10.290774 |
  | B2 | 10.323029 | 10.323028 | 10.323028 |
  | B3 | 10.313798 | 10.313798 | 10.313798 |
  | ST | 10.314072 | 10.314072 | 10.314072 |

  `bank_gates_eval.py` ran on the same four checkpoints and its total CE equals the
  evaluator's.

## The wave — submitted 2026-11-07 22:16-22:18

All four were submitted with `run_exp.py --config-path configs/bank_gates --config-name
<arm> +infrastructure.slurm.exclude=r16-11`, from the worktree at `9d7ed99d`. Each was
running within two minutes. Every cemetery commit has `9d7ed99d` as its parent and
changes only `exp.job`, `generated_configs/config_0.yaml` and `run_exp.py`'s usual
untracking of `pixi.toml` and `pixi.lock`.

| Arm | `bank_gate` | Job | Cemetery commit | Cemetery clone (under `.../nano/plgmstefaniak/llmrandom_cemetery/`) | W&B run | Node | Parameters, as logged |
|---|---|---|---|---|---|---|---:|
| B1 | confidence | 6264723_0 | `49456e2` | `bank_gates_b1_2026-11-07_22-16-06` | `621sjn71` | r12-6 | 1,118,086,608 |
| B2 | context | 6264725_0 | `fda924b` | `bank_gates_b2_2026-11-07_22-16-34` | `kgk0prvf` | r12-7 | 1,103,709,792 |
| B3 | backoff | 6264726_0 | `12d2779` | `bank_gates_b3_2026-11-07_22-16-59` | `5w6p64bv` | r12-10 | 1,103,617,488 |
| ST | stem | 6264733_0 | `6af74f7` | `bank_gates_stem_2026-11-07_22-17-22` | `l54pyyqp` | r14-9 | 1,110,676,944 |

W&B project `nano-ame`; run names `bank_gates_<arm>`. Each run's final checkpoint lands
in its clone under `checkpoint/<job>/0/step_3814/`, with `token_counts.pt` beside it.

**Evaluator validation, job 6264732** (`bg_eval_doi36p8k`, r12-10), submitted 22:17
with `bank_gates_lem_eval.sh` from `9d7ed99d`, while the four runs train:
1. `parity_numerics.py score` on `doi36p8k`'s final checkpoint (job 6256577, clone
   `s2_ps_h12_lr8.0_2026-11-06_13-52-40`), `ame_ref32` and `ame_prod`. The plan's
   gate: `ame_ref32` = 3.125227.
2. `dump_data`: the held-out set rebuilt from LEM's FineWeb copy and compared with the
   entropy dump that every score uses.

**The evaluator is validated (22:22).** On `doi36p8k`'s checkpoint the port gives
`ame_ref32` = **3.1252272**, the plan's 3.125227 to every digit it states, in 242 s.
- The production path (`ame_prod`) gives 3.1253729 in 49 s. The parity audit had
  3.125377 for it on entropy; the 4e-6 gap is bf16 nondeterminism in that path.
- The common evaluator, fp32 and eager, is the one the plan scores with, and it agrees
  exactly.
- LEM's own held-out set, rebuilt by `dump_data` from LEM's FineWeb copy, is identical
  to the entropy dump (`torch.equal` on the `[8192, 2049]` tensor; fingerprint
  `[21440, 25411, …, 48684]`).

So a score from LEM's dataloader and a score from the copied tensor are the same
measurement. Job 6264732 completed in 5 min 44 s.

### Checks

**Start, 22:18-22:21.** All four were running within two minutes of submission and logged
their first steps by 22:21. FineWeb opened in about 3 minutes, against the 10-21 the plan
warned of. The logged parameter counts equal the plan's table. The banks run Tier 2 in
bf16 ("Memory-parallel banks enabled: mem_parallel_size=1, compute dtype torch.bfloat16").

| Arm | Step | loss | grad_norm | grad_norm/banks | clip factor | s/step |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 12 | 7.69 | 0.87 | 0.00004 | 1.00 | 3.87 |
| B2 | 13 | 7.66 | 1.51 | 0.014 | 0.66 | 3.83 |
| B3 | 12 | 7.59 | 1.41 | 0.022 | 0.71 | 3.77 |
| ST | 3 | 9.25 | 1.98 | 0.013 | 0.50 | 3.88 |

- **The banks are a small share of the joint norm, not the bulk of it.** The plan's
  worry was that σ = 0.01 would make the bank gradients dominate the clip (the stop
  rule). In these first steps the banks contribute 0.00004-0.022 of a norm of 0.9-2.0.
  The clipping during warmup is the backbone's, as in `doi36p8k`, whose `grad_norm`
  averaged 2.5-2.8 before step 38.
- **B1's banks barely move at first:** every score starts at 0.0025 of its ungated size
  (the trust is 0.05 on both sides), so the rows' gradient is scaled down by the trust
  as well.

**Step 300, 22:42.** Every check passes, and the stop rule does not apply.

Train loss and `grad_norm` are window means (min/mean/max for the loss), and `doi36p8k`'s
are at the same steps. s/step is the median, excluding the eval steps.

| Arm | Loss, steps 200-299 | `grad_norm`, 100-199 / 200-299 | `grad_norm/banks`, 200-299 | Clip factor after step 76 | s/step | vs `doi36p8k`'s 3.746 |
|---|---|---|---:|---|---:|---:|
| B1 | 4.91/5.16/5.53 | 0.243 / 0.212 | 0.0016 | 1.000 | 3.880 | +3.6% |
| B2 | 4.25/4.43/4.69 | 0.251 / 0.197 | 0.045 | 1.000 | 3.836 | +2.4% |
| B3 | 4.12/4.35/4.66 | 0.286 / 0.237 | 0.043 | 1.000 | 3.773 | +0.7% |
| ST | 4.38/4.58/4.81 | 0.262 / 0.193 | 0.059 | 1.000 | 3.892 | +3.9% |
| `doi36p8k` | 4.81/5.03/5.28 | 0.293 / 0.182 | not logged | — | 3.746 | |

- **No NaN or inf** in any run.
- **s/step** is within the plan's 10% (ST: 15%) in every arm.
- **`grad_norm`** tracks `doi36p8k`'s.
- **The banks never dominate the norm.** At most about a fifth of it, so 4% of the
  squared norm. B1's banks are a hundredth of that. The clip has been inactive since the
  warmup in all four runs, as in production. So the stop rule's condition (a clip factor
  far below 1 after the warmup) is not met.
- **The readouts move** (layer means; the range over layers in brackets):

  | Arm | Step 0 | Step 300 |
  |---|---|---|
  | B1 trust, q / k | 0.0500 / 0.0500 | 0.0509 / 0.0504; p90 0.0534 [0.0517, 0.0605] / 0.0509; none above 0.5 |
  | B2 gate | bias 3.000, weight RMS 0 | bias mean 3.055, min 2.92 [2.49, 3.05]; weight RMS 0.044 |
  | B3 base norm, q / k | 0 / 0 | 0.246 [0.131, 0.313] / 0.220 [0.166, 0.265] |
  | ST gate weight RMS | 0.0317 | 0.0556; τ falling, 8.00 → 7.69 |

  B1's trust climbs as slowly as the plan predicts: reaching s = 0.5 takes at least 754
  of a token's own updates at the peak learning rate. B3's base vectors are already
  about three times the rows' initial length per head (0.08 at σ = 0.01).
- **An early lead, not a result.** Three arms are well ahead of production at the same
  step: 4.35-4.58 against 5.03 over steps 200-299. That fits σ = 0.01 turning the banks
  about 100× faster at the start, since QK-norm makes σ an inverse learning rate for
  the banks ([ame_implementation.md](../ame_implementation.md), weight init). B1, whose
  trust mutes that, trails production slightly (5.16). The plan does not read a
  step-300 loss as quality: production's 220-step losses spread with an sd of 0.13.

**Step 500, 22:55.** Every check passes again.

| Arm | Loss, steps 400-499 | `grad_norm`, 400-499 | `grad_norm/banks` | Clip factor | s/step |
|---|---|---:|---:|---|---:|
| B1 | 4.31/4.47/4.62 | 0.155 | 0.0027 | 1.000 | 3.880 |
| B2 | 3.86/3.95/4.06 | 0.124 | 0.030 | 1.000 | 3.833 |
| B3 | 3.78/3.88/3.98 | 0.131 | 0.025 | 1.000 | 3.773 |
| ST | 3.91/4.01/4.12 | 0.130 | 0.039 | 1.000 | 3.880 |
| `doi36p8k` | 4.19/4.33/4.50 | 0.145 | not logged | — | 3.746 |

- **No NaN.** s/step and `grad_norm` are steady. The clip has stayed inactive since the
  warmup.
- **Readouts at step 500:**
  - B1's trust: mean 0.0516 (q) and 0.0514 (k), q p90 0.0564. The first two
    (token, head) entries of one layer crossed 0.5.
  - B2's bias: mean 3.00, minimum 2.36; weight RMS 0.047.
  - B3's base norms: 0.30 (q) and 0.29 (k).
  - ST's gate RMS: 0.065; τ 7.65.
- **The early lead over production is shrinking:** 0.32-0.45 over steps 400-499,
  against 0.45-0.68 over steps 200-299.

### Finish — 2026-11-08 02:31-02:39

All four completed (Slurm `COMPLETED`, exit 0) at step 3,814, after 4,000,317,440 tokens.
The watch kept to the end flagged nothing; it read W&B and Slurm every 4 minutes and
scanned the job logs.

| Arm | Job | Elapsed | s/step, median | vs `doi36p8k` | Peak memory, GiB | Train loss, last 100 steps | In-training `final_ce` (bf16) |
|---|---|---|---:|---:|---:|---:|---:|
| B1 | 6264723_0 | 4:21:18 | 3.880 | +3.6% | 70.66 | 3.225 | 3.1406 |
| B2 | 6264725_0 | 4:18:35 | 3.835 | +2.4% | 70.42 | 3.108 | 3.0352 |
| B3 | 6264726_0 | 4:14:02 | 3.771 | +0.7% | 70.58 | 3.108 | 3.0361 |
| ST | 6264733_0 | 4:21:23 | 3.880 | +3.6% | 71.65 | 3.088 | 3.0186 |
| `doi36p8k` | 6256577_0 | 4:12:54 | 3.746 | | 70.40 | 3.212 | 3.125 |

- **Clean from first step to last,** over every step of the full W&B histories:
  - no NaN or inf in any arm;
  - no loss spike in any arm or in `doi36p8k`, where a spike is a step more than 0.3
    above the median of the 50 before it;
  - after the warmup the clip acted on four steps in all: B1's 38, 40 and 41 (minimum
    0.53, as the warmup ended) and B3's 57 (0.97). It never acted in B2 or ST.
- **Checkpoints:** each `step_3814/` holds three files:
  - the model, 2.12-2.20 GB;
  - the banks' sidecar `memory_parallel_rank0.pt`, 11.13 GB (B1's is 11.30 GB, wider by
    the trust columns);
  - `token_counts.pt`.
- **The four `token_counts.pt` are byte-identical** (sha256 `ab1df510…`): every arm saw the
  same tokens in the same steps.
- **Readouts at step 3,800** (layer means; the range over layers in brackets):

  | Arm | Readouts |
  |---|---|
  | B1 | trust mean 0.055 (q) and 0.057 (k), p90 0.068 both; 0.012% of (token, head) entries above 0.5; τ 11.3 [9.7, 12.8] |
  | B2 | gate bias mean 2.60 [2.44, 2.92], minimum 2.20; weight RMS 0.065; τ 8.34 |
  | B3 | base norms 0.47 [0.27, 0.58] (q) and 0.58 [0.40, 0.83] (k); τ 8.20 |
  | ST | gate weight RMS 0.084 [0.071, 0.095]; τ 7.83 |

  B1's trust hardly left its start. Its scores end at about s_q·s_k ≈ 0.003 of their
  ungated size, and its τ rose further than any other arm's.

### Scoring — submitted 2026-11-08 02:44-02:45

Five one-GPU jobs were run through `bank_gates_lem_eval.sh`, from the worktree at
**`8b07c983`** (= `ame_main`).

`ame_main` had moved past `9d7ed99d` by then:
- my report commits;
- `c96c31a4`'s default arm name;
- other sessions' docs, Stage 1c/2c configs and Helios pins in the pixi files.

Nothing under `src/` or `documents/scripts/` changed, so the evaluator is the one validated
on `doi36p8k`.

**What each job ran:**
- **Each arm's job:** `parity_numerics.py score` (`ame_ref32`, then `ame_prod`), then
  `bank_gates_eval.py`, on its `step_3814/` checkpoint with its own counts.
- **The fifth job:** the bucket analysis on `doi36p8k`'s final checkpoint, with B1's counts.
  Its run predates the counter. Its logged config differs from the arms' in no data, seed,
  batch, step or learning-rate key, only in:
  - the gate keys and the names;
  - `count_token_updates`;
  - the Slurm memory request;
  - a `gqa: null` that its older commit did not log.

| Job | Name | Scores | Node | Elapsed | Status |
|---|---|---|---|---|---|
| 6265588 | `bg_eval_b1` | B1 | r10-3 | 9:30 | COMPLETED |
| 6265589 | `bg_eval_b2` | B2 | r10-4 | 9:26 | COMPLETED |
| 6265590 | `bg_eval_b3` | B3 | r10-9 | 9:29 | COMPLETED |
| 6265591 | `bg_eval_stem` | ST | r10-13 | 9:38 | COMPLETED |
| 6265592 | `bg_buckets_doi36p8k` | `doi36p8k`, buckets only | r10-16 | 4:34 | COMPLETED |

Outputs (`score_<tag>.json`, `buckets_<tag>.json`) are under
`.../nano/plgmstefaniak/bank_gates/out/<name>_<job>/`.

### Results

**Final CE, common evaluator** (`ame_ref32`; 8,192 held-out sequences, 16.8M
predictions). `ame_ref32` is fp32 and eager, with reference attention and the banks
loaded from the fp32 masters.

| Rank | Arm | `ame_ref32` | − `doi36p8k` | `ame_prod` |
|---|---|---:|---:|---:|
| 1 | ST, stem | **3.019112** | −0.1061 | 3.019252 |
| 2 | B2, context | 3.035505 | −0.0897 | 3.035630 |
| 3 | B3, backoff | 3.035837 | −0.0894 | 3.035979 |
| 4 | B1, confidence | 3.141958 | +0.0167 | 3.142109 |
| | `doi36p8k`, production S2-PS | 3.125227 | | 3.125373 |

Three consistency checks hold:
- each bucket pass reproduced its arm's `ame_ref32` to six digits;
- `ame_prod` sits 0.00012-0.00015 above `ame_ref32` in every arm, as in production
  (0.00015);
- the in-training `final_ce` (bf16, same held-out set) ranks the arms the same way.

**Pairwise**, row minus column. A tie is a gap inside the plan's ±0.019, the 95%
half-width of one run against another.

| | B2 | B3 | B1 | `doi36p8k` |
|---|---:|---:|---:|---:|
| ST | −0.0164, tie | −0.0167, tie | −0.1228 | −0.1061 |
| B2 | | −0.0003, tie | −0.1065 | −0.0897 |
| B3 | | | −0.1061 | −0.0894 |
| B1 | | | | +0.0167, tie |

**Best gate: a three-way tie between ST, B2 and B3.**
- ST is lowest, but its lead over B2 (0.0164) and B3 (0.0167) is inside the floor. That
  lead is 1.98 standard deviations of a one-run difference (two-sided p ≈ 0.08 at 8 df):
  suggestive, not resolved. B2 and B3 are 0.0003 apart.
- B1 is resolved last, 0.106-0.123 behind the other three. Against production it ties
  (+0.0167).
- ST, B2 and B3 beat production by 0.089-0.106, 4.7-5.6 half-widths. As the plan says,
  that gain is the mechanism and the σ = 0.01 init together.
- σ = 0.01 probably does most of that work. Three dissimilar gates land within 0.017 of
  each other, and none of them mutes rare rows (below). Only a σ = 0.01 run without a
  gate (Plan C, outside this plan) can split σ from the mechanism.

**Replicates that would break the tie (proposed; not submitted).**
- **Which runs:** two more runs each of ST and B2 at the same seed, as the plan
  prescribes. B3 too, since it is as much the runner-up as B2 (0.0003 behind).
- **What they resolve:** with three runs per arm, the 95% half-width for a difference of
  means falls to about ±0.010 (t at 12-14 df × 0.00585 × √(2/3)). ST's 0.016 lead
  resolves if it holds.
- **What else they measure:** LEM's own run-to-run spread, which the plan has so far
  assumed equals entropy's.
- **Cost:** about 4.3 h per run on one LEM H100, all in parallel. Four runs (ST, B2) take
  about 17.5 GPU-hours, six (with B3) about 26.
- **Status:** waiting for Maciej's go.

**Rare-row buckets: what each one holds.** u is the number of training steps a token
occurred in.

| u | Token ids | Held-out positions | In the attention sample (first 512 sequences) | What dominates |
|---|---:|---:|---:|---|
| 0 | 1,026 | 5 | 4 | 2 ids |
| 1-10 | 476 | 1,702 | 61 | ' COVID' (id 19314): 1,686 positions in 520 sequences |
| 11-100 | 1,143 | 502 | 16 | five sequences hold 49%; the commonest ids are byte pieces from Gujarati and Lao text |
| 101-1000 | 5,920 | 17,812 | 1,123 | nothing: 4,026 ids in 4,575 sequences, none above 1.4% |
| 1001-3814 | 32,928 | 1,884,321 | | nothing |
| 3815 | 8,811 | 14,872,874 | | nothing |

**The held-out set is not drawn like the training stream.** It is FineWeb-Edu's reserved
tail (`holdout: eval`, the last 262,144 rows).
- ' COVID' occurs twice in the runs' 4.0B training tokens and never in the 2.1M-token
  training sample stored in `parity_batches.pt`.
- It occurs 1,686 times in the held-out set's 16.8M tokens, in passages about COVID-19 and
  2022.

So u = 1-10 is in effect one token the runs never learned, and u = 0 and u = 11-100 are a
handful of documents. The reading below rests on u = 101-1000 (decision 8). The shift
affects every run's CE alike, so the comparisons above stand.

**Loss** at positions whose input token is in the bucket:

| u | `doi36p8k` | ST | B2 | B3 | B1 |
|---|---:|---:|---:|---:|---:|
| 0 (5 positions) | 6.26 | 7.64 | 5.66 | 7.09 | 5.60 |
| 1-10 (' COVID') | 3.99 | 6.35 | 5.74 | 4.73 | 4.97 |
| 11-100 | 4.44 | 4.10 | 4.08 | 4.02 | 4.34 |
| **101-1000** | **3.305** | **2.942** | **2.963** | **2.952** | **3.416** |
| 1001-3814 | 2.471 | 2.344 | 2.360 | 2.354 | 2.525 |
| 3815 | 3.208 | 3.104 | 3.121 | 3.122 | 3.220 |

**Influence as queries and as keys**, at u = 101-1000 / u = 3815 (the commonest tokens):

| | `doi36p8k` | ST | B2 | B3 | B1 |
|---|---|---|---|---|---|
| Identity spread as queries (std of q·k over the visible keys) | 0.90 / 1.00 | 1.24 / 1.10 | 1.10 / 0.91 | 1.26 / 1.19 | 0.32 / 0.64 |
| Attention entropy as queries (÷ ln of the visible keys) | 0.561 / 0.519 | 0.543 / 0.509 | 0.551 / 0.579 | 0.486 / 0.511 | 0.577 / 0.543 |
| Mass received as keys ÷ uniform share | 0.835 / 0.808 | 0.852 / 0.750 | 0.976 / 0.843 | 0.882 / 0.725 | 0.782 / 0.748 |
| q row length per head | 7.97 / 8.00 | 0.69 / 1.05 | 0.73 / 1.03 | 0.69 / 0.99 | 0.32 / 0.74 |

**Mechanism readouts by bucket.** B1's trust is per (token, head); B2's g and ST's gate
RMS are on real activations; B3's is cos(row, base).

| u | B1 trust, q / k | B2 g | ST gate RMS | B3 cos(row, base), q / k |
|---|---|---:|---:|---|
| 0 | 0.0500 / 0.0500 | 0.76 | 1.66 | 0.000 / −0.001 |
| 1-10 | 0.0501 / 0.0500 | 0.81 | 1.70 | 0.009 / −0.001 |
| 11-100 | 0.0506 / 0.0501 | 0.75 | 1.56 | 0.013 / −0.007 |
| 101-1000 | 0.0522 / 0.0509 | 0.71 | 1.51 | 0.005 / −0.015 |
| 1001-3814 | 0.0548 / 0.0551 | 0.69 | 1.47 | 0.022 / 0.009 |
| 3815 | 0.0612 / 0.0686 | 0.61 | 1.26 | 0.076 / 0.026 |

**What the buckets show:**
1. **Only B1 makes rare rows less influential, and it pays for it.**
   - Its trust rises with u but stays near its start everywhere, so identity logits are
     flat for every token: spread 0.32 at u = 101-1000 against production's 0.90, and
     0.64 even at u = 3815.
   - Its rare keys draw the least mass, and its rare queries attend the most diffusely.
   - Its loss is above production's from u = 101 up: +0.11 at u = 101-1000.
2. **ST, B2 and B3 make rare rows more influential, not less, and gain most on rare
   tokens.**
   - Their rare queries' identity logits are sharper than production's (1.10-1.26 against
     0.90).
   - Their rare keys draw as much mass or more (0.85-0.98 against 0.84).
   - Their loss gain over production grows as tokens get rarer:
     - −0.09 to −0.10 at u = 3815;
     - −0.11 to −0.13 at 1001-3814;
     - −0.34 to −0.36 at 101-1000 (and −0.34 to −0.43 at 11-100).
3. **B2's and ST's gates open wider for rare query tokens, not narrower.**
   - B2's g is 0.71 at u = 101-1000 and 0.81 at 1-10, against 0.61 at u = 3815.
   - ST's gate RMS is 1.51 and 1.70 there, against 1.26.
4. **B3 backs a rare row off to its shared vector, as designed, but that vector is a sharp
   query.**
   - The rows stay nearly orthogonal to the base (|cos| ≤ 0.08), so the backoff works
     through length.
   - At u = 1-10 a row is 0.17-0.22 long per head, against base norms of 0.47 (q) and
     0.58 (k), so the normalized sum is mostly base. At u = 3815 the row, about 1.0 long,
     dominates.
   - Rare queries in B3 attend the most sharply of any arm: entropy 0.38-0.49 at
     u = 1-1000, against 0.52 at 1001-3814.
5. **At the ' COVID' positions every σ = 0.01 arm is worse than production, by 0.73 (B3)
   to 2.35 (ST).** This is one token: an anecdote, not a bucket result.
   - Production's rows there are still their σ = 1 draws, 7.97 long.
   - In the σ = 0.01 arms, rows with 1-10 updates grew from 0.08 to 0.10-0.35 per head.
     A few updates and the Adam momentum after them outweigh the init, as the plan's
     "idle rows keep moving" risk anticipates. Production's rows, 100 times longer, barely
     turn.
   - B3, whose rare rows are mostly the base, is hurt least.

**Against the plan's requirement, no arm meets all three conditions.** u = 0 (5 positions)
is not read.

| Arm | Rare rows less influential? | Overall CE within the floor, or better | Rare loss not above production's |
|---|---|---|---|
| B1 | yes: flat identity logits, s ≈ 0.05, least key mass | yes, +0.017 | no: +0.11 at u = 101-1000 |
| B2 | no: sharper logits, most key mass, gate widest on rare tokens | yes, −0.090 | yes from u = 11 up; no at ' COVID' (+1.74) |
| B3 | no, by the plan's measures; rare rows back off to a shared vector that is itself sharp | yes, −0.089 | yes from u = 11 up; no at ' COVID' (+0.73) |
| ST | no: sharpest logits, gate widest on rare tokens | yes, −0.106 | yes from u = 11 up; no at ' COVID' (+2.35) |

**The plan's second question at h12, answered:** a rarely updated row is less influential
only in B1, which mutes every row and is the worst arm. In the three arms that tie for
best, rare rows are at least as influential as in production, and their gain is largest on
rare tokens.

## The h16 wave

Four runs, one per gate, at h16 on 8B tokens (decision 7; the plan's
[h16 extension](ame_bank_gates_plan.md#h16-extension-maciej-2026-11-07)). They train
alongside the h12 wave.

| | |
|---|---|
| Configs | `configs/bank_gates/h16_base.yaml`, a copy of [configs/stage_2b/ame_h16_pe_shift.yaml](../../../configs/stage_2b/ame_h16_pe_shift.yaml) (S2b-PS h16), plus `h16_b1.yaml`, `h16_b2.yaml`, `h16_b3.yaml`, `h16_stem.yaml` |
| Recipe | S2b-PS h16 as `hjw7znbz` ran it: one H100, micro-batch 16 × 32, Tier 2 banks, no FSDP2, autocast bf16, `compile_blocks_full`, 24 h limit. Changed: n = 8, 7,630 steps = 8.00B tokens with warmup 76 written out, `count_token_updates`, W&B names `bank_gates_h16_<arm>` |
| Reference | S2b-PS h16 at n = 8: job 6260570, W&B `hjw7znbz`, clone `s2b_ps_h16_lr8.0_2026-11-07_03-36-02`. Trainer `final_ce` 2.8594; 1,986,332,928 parameters; 6.810 s/step; peak 72.17 GiB; 15 h 06 min |
| Noise floor | None measured at h16 and 8B. The h12 floor (±0.019 at h12 and 4B) does not transfer; h16 gaps are reported beside it as a stated proxy, with that caveat |

### Before submitting

Configs committed in `c96c31a4` and fast-forwarded into `ame_main`.

**Composed as `run_exp.py` composes them**, each passed `check_point` and was built on the
meta device. The parameter counts equal the plan's arithmetic:

| Arm | Parameters | GFLOP/token (seq 2048) |
|---|---:|---:|
| none (`h16_base.yaml`, = `hjw7znbz`) | 1,986,332,928 | 2.1212 |
| B1 | 2,012,088,576 | 2.1212 |
| B2 | 1,986,595,328 | 2.1228 |
| B3 | 1,986,365,696 | 2.1212 |
| ST | 2,003,110,144 | 2.2219 (+4.75%) |

**The copy is the reference's recipe.** `h16_base.yaml`, resolved, was compared key by
key with `hjw7znbz`'s own generated config from its cemetery clone, also resolved. They
differ only in:
- names and tags;
- `count_token_updates`;
- `run_exp.py`'s bookkeeping keys;
- the exclude, which both pass at submit;
- the RAM request: 228G, the LEM cluster file's value since 2026-11-07, against the
  reference's 220G. It is host memory and changes nothing in the run.

So the data config (dataset, seed 123, sequence length 2049, batch 512) is the arms'
own. With world size 1 on both sides (W&B config), an arm's `token_counts.pt` describes
the reference's data stream too.

**Time.** `hjw7znbz` took 15 h 06 min at 6.810 s/step. ST, at +4.75% FLOPs, projects to
about 15.8 h. The limit is 24 h.

**Memory.** `hjw7znbz` peaked at 72.17 GiB; the H100 has 93.6 GiB.
- B1's 16 extra columns over 32 banks add 25.8M parameters, about 0.5 GiB of Tier 2 state
  at 18 bytes per parameter.
- ST's gate adds 16.8M parameters (about 0.3 GiB with AdamW) and at most about 3 GiB of
  saved activations at micro-batch 16 (three `[16, 2048, 1024]` bf16 tensors per layer).
- The heaviest arm is therefore at most about 76 GiB.

8B tokens fits both, so there was no reason to stop and ask.

**A test fixed on the way (`c96c31a4`).** `test_every_runnable_config_composes_and_resolves`
failed once `base.yaml` was tracked (`157c3122`). It resolves every config, and
interpolating the mandatory `bank_gates_arm: ???` raises even with missing values allowed.
The suite reported in "Implementation and local checks" ran before the configs were
tracked, so it never saw them. `bank_gates_arm` now defaults to `base`. At `c96c31a4` the
suite passes with every config tracked: 381 passed, 18 skipped. The running h12 jobs are
unaffected, since their arm files set the key.

### Jobs — submitted 2026-11-07 22:39-22:40

All four were submitted with `run_exp.py --config-path configs/bank_gates --config-name
h16_<arm> +infrastructure.slurm.exclude=r16-11`, one at a time, from the worktree at
`c96c31a4` (= `ame_main`). Every cemetery commit has `c96c31a4` as its parent and changes
only `exp.job`, `generated_configs/config_0.yaml` and the two pixi files.

| Arm | `bank_gate` | Job | Cemetery commit | Cemetery clone (under `.../nano/plgmstefaniak/llmrandom_cemetery/`) | W&B run | Node | Parameters, as logged |
|---|---|---|---|---|---|---|---:|
| B1 | confidence | 6264776_0 | `02e1be80` | `bank_gates_h16_b1_2026-11-07_22-39-03` | `h85djd57` | r16-10 | 2,012,088,576 |
| B2 | context | 6264777_0 | `f8434114` | `bank_gates_h16_b2_2026-11-07_22-39-24` | `jwirkswo` | r16-13 | 1,986,595,328 |
| B3 | backoff | 6264778_0 | `92509a63` | `bank_gates_h16_b3_2026-11-07_22-39-46` | `3yc7haqu` | r16-14 | 1,986,365,696 |
| ST | stem | 6264779_0 | `b6dd7a5f` | `bank_gates_h16_stem_2026-11-07_22-40-08` | `ug8g2i2x` | r16-15 | 2,003,110,144 |

Each final checkpoint lands under `checkpoint/<job>/0/step_7629/` in its clone, with
`token_counts.pt` beside it.

### Checks

**Start, 22:39-22:46.** All four were running within about two minutes of submission,
logged the plan's parameter counts, and trained from 22:44. FineWeb opened in about 3
minutes.

| Arm | Step | loss | grad_norm | grad_norm/banks | clip factor | s/step | vs `hjw7znbz`'s 6.810 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 | 34 | 7.06 | 0.75 | 0.00014 | 1.00 | 7.03 | +3.2% |
| B2 | 30 | 6.91 | 0.96 | 0.060 | 1.00 | 6.86 | +0.7% |
| B3 | 26 | 7.09 | 1.07 | 0.099 | 0.94 | 6.81 | 0.0% |
| ST | 4 | 8.75 | 2.12 | 0.018 | 0.47 | 7.01 | +3.0% |

As at h12, the banks are a small share of the norm, and any clipping in the warmup is the
backbone's.

**Step 300, 23:24.** All four passed the step-300 checks. B2 then spiked at steps 345-359,
the first loss spike in this plan.

| Arm | Loss, steps 200-299 | `grad_norm`, 100-199 / 200-299 | `grad_norm/banks`, 200-299 | Clip factor, steps 100-299 | s/step | vs 6.810 |
|---|---|---|---:|---|---:|---:|
| B1 | 4.81/5.05/5.34 | 0.325 / 0.204 | 0.0026 | 1.000 | 7.011 | +3.0% |
| B2 | 4.09/4.28/4.53 | 0.316 / 0.188 | 0.034 | 1.000 | 6.867 | +0.8% |
| B3 | 4.12/4.30/4.50 | 0.354 / 0.233 | 0.041 | 0.96-1.00 | 6.804 | −0.1% |
| ST | 4.12/4.34/4.56 | 0.325 / 0.207 | 0.055 | 1.000 | 7.023 | +3.1% |
| `hjw7znbz` | 4.84/5.10/5.41 | 0.282 / 0.237 | not logged | — | 6.810 | |

- **Up to step 300 the h16 runs look like the h12 runs.**
  - No NaN.
  - s/step within the plan's limits; B1's and ST's overhead matches h12's.
  - `grad_norm` tracks the reference.
  - The banks are a small share of the norm, and the clip has been inactive since about
    step 100.
  - The readouts move: B1's trust mean 0.0506 (q); B2's bias mean 3.08, minimum 2.64,
    weight RMS 0.045; B3's base norms 0.25 (q) and 0.18 (k); ST's gate RMS
    0.027 → 0.055, τ 7.74.
  - The same early lead as at h12: B2, B3 and ST at 4.28-4.34 over steps 200-299 against
    the reference's 5.10, with B1 level with it.
- **B2 spiked at steps 345-359.**
  - The train loss rose from 4.0 at step 335 to 4.28 at 345 and 5.09 at 347, peaked at
    7.97 at step 354, and was back to 6.69 by step 359.
  - `grad_norm` peaked at 5.49 at step 348, and the clip was active (factor 0.18-0.97)
    through the spike.
  - The banks were 0.14-0.58 of that norm. So the spike is mostly the backbone's, and the
    stop rule's condition (the banks dominating the norm after the warmup) does not hold.
  - It is not a crash or a NaN, so nothing is cancelled or resubmitted. Earlier AME runs
    recovered from spikes like it: Stage 2's S2-S at steps 555-581 (to 9.50) and S1-A0
    at 609-635 (to 14.69).
  - **Recovered by 23:32.** The clip was back at 1.0 by step 360, and `grad_norm` had
    settled to 0.15-0.3 by step 380. The 20-step mean loss was 4.59 at step 414: back
    under 4.6, though still above the pre-spike 4.0.

    | Step | 335 | 345 | 350 | 354 | 360 | 370 | 380 | 390 | 400 | 414 |
    |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
    | Train loss | 4.03 | 4.28 | 5.72 | 7.97 | 6.41 | 5.38 | 5.06 | 4.75 | 4.69 | 4.50 |

    The final CE will show whether the detour cost B2 anything lasting at h16.

**Step 500, 23:48.** Every check passes, with a second transient spike, in B3.

| Arm | Loss, steps 400-499 (min/mean/max) | `grad_norm`, 400-499 | `grad_norm/banks` | Clip factor | s/step |
|---|---|---:|---:|---|---:|
| B1 | 4.25/4.38/4.56 | 0.152 | 0.0031 | 1.000 | 7.013 |
| B2 | 4.06/4.28/4.69 | 0.155 | 0.026 | 1.000 | 6.863 |
| B3 | 4.00/4.53/6.56 | 0.481 | 0.070 | 0.30-1.00 (spike) | 6.816 |
| ST | 3.72/3.82/3.97 | 0.126 | 0.032 | 1.000 | 7.023 |
| `hjw7znbz` | 4.19/4.34/4.53 | 0.150 | not logged | — | 6.810 |

- **No NaN.** s/step is steady.
- **B2's recovery held:** 4.28 over steps 400-499, level with the reference.
- **B3 spiked at steps 404-416 and recovered by step 430.**
  - The loss went from 4.28 at step 400 to a peak of 6.56 at step 408, was back to 4.66
    by step 430, and reached 3.92 at step 560, below its pre-spike 4.0-4.2.
  - `grad_norm` peaked at 3.35, and the clip was active (down to 0.30) through steps
    404-416.
  - The banks were a larger share of the norm than in B2's spike, at most 0.95 of 3.35
    (8% of the squared norm). The backbone still dominated, so again the stop rule does
    not apply.
  - At step 500, B3's layers 0 and 1 have nearly collapsed attention (normalized entropy
    0.24 and 0.13, top-8 mass 0.86 and 0.93), and the largest base vectors (norms
    0.45-0.61, against 0.20-0.39 in layers 2-15). The shared vector is strongest where
    attention is sharpest; the bucket analysis looks at it with real activations.
- **The readouts:**
  - B1's trust: mean 0.0510 (q) and 0.0508 (k); the first entries cross 0.5.
  - B2's bias: mean 3.14, minimum 2.76; weight RMS 0.065.
  - B3's base norms: 0.33 (q) and 0.28 (k), layer means.
  - ST's gate RMS: 0.064; τ 7.68.

### Finish — 2026-11-08 13:43-14:16

All four completed (Slurm `COMPLETED`, exit 0) at step 7,629, after 8,000,634,880 tokens.
The watch kept from step 2,100 to the end flagged nothing; it read W&B and Slurm every 5
minutes and scanned the job logs.

| Arm | Job | Elapsed | s/step, median | vs `hjw7znbz` | Peak memory, GiB | Train loss, last 100 steps | In-training `final_ce` (bf16) |
|---|---|---|---:|---:|---:|---:|---:|
| B1 | 6264776_0 | 15:29:20 | 7.011 | +3.0% | 72.77 | 2.911 | 2.8457 |
| B2 | 6264777_0 | 15:10:39 | 6.863 | +0.8% | 72.19 | 2.874 | 2.8203 |
| B3 | 6264778_0 | 15:03:02 | 6.813 | 0.0% | 72.17 | 2.867 | 2.8105 |
| ST | 6264779_0 | 15:35:07 | 7.020 | +3.1% | 73.45 | 2.802 | 2.7520 |
| `hjw7znbz` | 6260570_0 | 15:06:01 | 6.810 | | 72.17 | 2.930 | 2.8594 |

- **Clean after step 450,** over every step of the full W&B histories:
  - no NaN or inf in any arm;
  - two loss spikes, both under Checks above: B2's at steps 346-424 (peak 7.97 at 354)
    and B3's at 402-450 (6.56 at 408). None in B1, ST or `hjw7znbz`, and none in any arm
    after step 450;
  - after the warmup the clip acted only inside those spikes, plus three light steps in
    B3 just after the warmup (77, 88 and 107; 0.88-0.96):
    - B2: 14 steps, 347-361, minimum 0.18;
    - B3: 11 steps, 404-416, minimum 0.30;
    - never in B1 or ST.
- **Memory and time** stayed where the pre-submit estimate put them: peak 72.2-73.5 GiB
  against the H100's 93.6, and 15.0-15.6 h against the 24 h limit.
- **Checkpoints:** each `step_7629/` holds:
  - the model, 4.06-4.26 GB;
  - the banks' sidecar `memory_parallel_rank0.pt`, 19.78 GB (B1's is 20.09 GB, wider by
    the trust columns);
  - the training state;
  - `token_counts.pt`.
- **The four `token_counts.pt` are byte-identical** (sha256 `e8aad098…`): every arm saw the
  same tokens in the same steps.
- **Readouts at step 7,600** (layer means; the range over layers in brackets):

  | Arm | Readouts |
  |---|---|
  | B1 | trust mean 0.056 (q) and 0.057 (k), p90 0.069 and 0.070; 0.011% of (token, head) entries above 0.5; τ 12.5 [10.1, 14.2] |
  | B2 | gate bias mean 2.48 [2.20, 2.83], minimum 1.98; weight RMS 0.079; τ 8.37 |
  | B3 | base norms 0.65 [0.43, 0.94] (q) and 0.88 [0.65, 1.14] (k); τ 8.23 |
  | ST | gate weight RMS 0.092 [0.080, 0.099]; τ 8.20 |

  As at h12, B1's trust hardly left its start and its τ rose furthest. B2's bias fell
  further than at h12 (2.48 against 2.60), and B3's bases grew longer (0.65 and 0.88
  against 0.47 and 0.58).

### Scoring — submitted 2026-11-08 14:34

Five one-GPU jobs were run through `bank_gates_lem_eval.sh`, from the worktree at
**`03317133`** (= `ame_main`, archived at `.../bank_gates/code/03317133`).
- Since the h12 scoring at `8b07c983`, nothing under `documents/scripts/` has changed.
- The only `src/` change is stage 2's `confidence_init`
  ([stage-2 report](ame_bank_gates_stage2_report.md)). Its default leaves every model
  bit-identical, which a test pins. So this is the evaluator that scored h12.

**What each job ran:**
- **Each arm's job:** `parity_numerics.py score` (`ame_ref32`, then `ame_prod`), then
  `bank_gates_eval.py`, on its `step_7629/` checkpoint with its own counts.
- **The reference's job** (`hjw7znbz`, the plan's h16 reference): the same two steps, on
  `h16_base.yaml` with `bank_gates_arm=ref`, using B1's `token_counts.pt`.
  - The counts fit `hjw7znbz`: its data config is the arms' own ("Before submitting":
    the same dataset, seed 123, sequence length 2049, batch 512 and world size 1).
  - All four arms' counts are byte-identical (Finish), so which arm's file it takes does
    not matter.

| Job | Name | Scores | Node | Elapsed | Status |
|---|---|---|---|---|---|
| 6272081 | `bg_eval_h16_b1` | B1 | r16-1 | 17:26 | COMPLETED |
| 6272082 | `bg_eval_h16_b2` | B2 | r16-2 | 17:34 | COMPLETED |
| 6272083 | `bg_eval_h16_b3` | B3 | r16-2 | 17:36 | COMPLETED |
| 6272084 | `bg_eval_h16_stem` | ST | r16-2 | 17:57 | COMPLETED |
| 6272085 | `bg_eval_hjw7znbz` | `hjw7znbz` | r16-2 | 17:28 | COMPLETED |

Outputs (`score_<tag>.json`, `buckets_<tag>.json`) are under
`.../nano/plgmstefaniak/bank_gates/out/<name>_<job>/`.

### Results

**Final CE, common evaluator** (`ame_ref32`; the same 8,192 held-out sequences as h12,
16.8M predictions):

| Rank | Arm | `ame_ref32` | − `hjw7znbz` | `ame_prod` |
|---|---|---:|---:|---:|
| 1 | ST, stem | **2.753039** | −0.1079 | 2.753170 |
| 2 | B3, backoff | 2.811172 | −0.0498 | 2.811307 |
| 3 | B2, context | 2.818295 | −0.0427 | 2.818424 |
| 4 | B1, confidence | 2.845979 | −0.0150 | 2.846129 |
| | `hjw7znbz`, S2b-PS h16 at n = 8 | 2.860982 | | 2.861128 |

The same three consistency checks hold as at h12:
- each bucket pass reproduced its arm's `ame_ref32` to six digits;
- `ame_prod` sits 0.00013-0.00015 above `ame_ref32` in every run;
- the in-training `final_ce` ranks the five runs the same way.

**Pairwise**, row minus column, against the h12 floor ±0.019 as a stated proxy. No noise
floor has been measured at h16 and 8B tokens (Setup of this section).

| | B3 | B2 | B1 | `hjw7znbz` |
|---|---:|---:|---:|---:|
| ST | −0.0581 | −0.0653 | −0.0929 | −0.1079 |
| B3 | | −0.0071, tie | −0.0348 | −0.0498 |
| B2 | | | −0.0277 | −0.0427 |
| B1 | | | | −0.0150, tie |

**Best h16 gate: ST, resolved.**
- **The lead:** 0.058 over B3 and 0.065 over B2, three times the proxy half-width.
  - For it to be a tie, h16's run-to-run spread would have to be about three times
    h12's: an sd of at least 0.018 against 0.00585.
- **What the comparison holds fixed:** tokens, not compute. ST is the costliest arm, at
  +4.75% FLOPs per token and +3.1% s/step.
- **Two ties below it:**
  - B3 and B2 are 0.007 apart;
  - B1 is level with the reference (−0.015).
- **Every gate beats the reference.** Three do so outside the proxy: ST by 0.108, B3 by
  0.050, B2 by 0.043.

**The h16 ranking agrees with h12's.**
- **The order:** ST first and B1 last at both scales. B2 and B3 sit between them, tied at
  both (0.0003 apart at h12, 0.0071 at h16), so their swap means nothing.
- **What scale changed:**
  - ST keeps its whole gain over the reference (−0.106 at h12, −0.108 at h16);
  - B2's and B3's gains roughly halve (−0.090 and −0.089 to −0.043 and −0.050);
  - so ST's lead over them grows from a tie (0.016) to a resolved 0.058-0.065;
  - B1 moves from +0.017 to −0.015, a tie both times.
- **Against the σ = 0.01 confound:** every arm here also starts its banks at σ = 0.01.
  If that init is the gain the arms share (stage 2's question 1), it shrinks with scale,
  while what ST adds does not.

**Replicates for the ties (proposed; not submitted).**
- **Which runs, at the plan's same-seed recipe:**
  - **ST, two more:** they measure h16's own run-to-run spread, which turns the ±0.019
    proxy into a measured floor and tests ST's lead.
  - **B3 and B2, two more each:** they break the B3-B2 tie. It matters for deployment,
    since B3 keeps AME's id-only scores and B2 does not.
  - The B1-reference tie is not worth runs: B1 is the weakest gate at both scales.
- **Cost:** six runs at about 15.3 h each on one LEM H100, about 92 GPU-hours, all in
  parallel in about 15.5 h of wall-clock (ST alone: about 31 GPU-hours).
- **Status:** waiting for Maciej's go.

**Rare-row buckets at h16: what each one holds.** S = 7,630, so the same u range now
picks rarer tokens than at h12. u = 101-1000 means seen in 1.3-13% of steps, against
2.6-26% at h12.

| u | Token ids | Held-out positions | In the attention sample (first 512 sequences) | What dominates |
|---|---:|---:|---:|---|
| 0 | 964 | 5 | 4 | 2 ids, as at h12 |
| 1-10 | 367 | 1,694 | 60 | ' COVID': 1,686 positions (99.5%). Its u is still 2 after 8B tokens |
| 11-100 | 753 | 141 | 7 | 58 ids in 61 sequences; the top 5 sequences hold 37% |
| 101-1000 | 3,685 | 5,819 | 360 | nothing: 1,683 ids in 1,759 sequences, none above 3.3%. Recent topics lead (' cryptocurrencies' 3.3%, ' Ethereum' 2.9%, 'OVID' 1.8%, ' Docker' 1.6%) |
| 1001-7629 | 36,306 | 2,012,491 | | nothing |
| 7630 | 8,229 | 14,757,066 | | nothing |

As at h12, the reading rests on u = 101-1000 (decision 8). It is a third the size of h12's
(5,819 positions against 17,812) and tilted to the held-out tail's recent topics.

**Loss** at positions whose input token is in the bucket:

| u | `hjw7znbz` | ST | B3 | B2 | B1 |
|---|---:|---:|---:|---:|---:|
| 0 (5 positions) | 5.58 | 8.18 | 6.09 | 6.16 | 5.43 |
| 1-10 (' COVID') | 4.17 | 4.99 | 3.70 | 3.63 | 5.33 |
| 11-100 (141 positions) | 4.92 | 4.08 | 4.62 | 4.43 | 4.87 |
| **101-1000** | **3.071** | **2.775** | **2.788** | **2.844** | **3.030** |
| 1001-7629 | 2.222 | 2.105 | 2.150 | 2.161 | 2.202 |
| 7630 | 2.948 | 2.841 | 2.901 | 2.908 | 2.933 |

**Influence as queries and as keys**, at u = 101-1000 / u = 7630:

| | `hjw7znbz` | ST | B3 | B2 | B1 |
|---|---|---|---|---|---|
| Identity spread as queries | 0.95 / 1.10 | 1.24 / 1.13 | 1.31 / 1.22 | 1.07 / 0.89 | 0.43 / 0.91 |
| Attention entropy as queries | 0.534 / 0.500 | 0.540 / 0.503 | 0.437 / 0.490 | 0.550 / 0.569 | 0.550 / 0.521 |
| Mass received as keys ÷ uniform share | 0.945 / 0.793 | 1.043 / 0.733 | 1.078 / 0.701 | 1.124 / 0.822 | 0.889 / 0.744 |
| q row length per head | 7.97 / 8.01 | 0.74 / 1.32 | 0.72 / 1.33 | 0.80 / 1.36 | 0.32 / 0.85 |

**Mechanism readouts by bucket** (as at h12):

| u | B1 trust, q / k | B2 g | ST gate RMS | B3 cos(row, base), q / k |
|---|---|---:|---:|---|
| 0 | 0.0500 / 0.0500 | 0.76 | 1.42 | 0.000 / −0.000 |
| 1-10 | 0.0500 / 0.0500 | 0.76 | 1.20 | 0.007 / −0.003 |
| 11-100 | 0.0504 / 0.0501 | 0.70 | 1.17 | 0.017 / −0.012 |
| 101-1000 | 0.0517 / 0.0509 | 0.66 | 1.06 | 0.008 / −0.025 |
| 1001-7629 | 0.0550 / 0.0555 | 0.63 | 1.07 | 0.016 / −0.002 |
| 7630 | 0.0624 / 0.0676 | 0.57 | 0.90 | 0.078 / 0.017 |

**What the h16 buckets show** (h12's findings, rechecked):
1. **Only B1 makes rare rows less influential, and at h16 that no longer costs it on
   rare tokens.**
   - Its trust still hardly leaves 0.05.
   - Its rare queries have the flattest identity logits (spread 0.43 at u = 101-1000,
     against the reference's 0.95), and its rare keys draw the least mass (0.89 against
     0.95).
   - Unlike at h12 (+0.11), its loss there is now 0.04 below the reference's, and it is
     below the reference's in every bucket from u = 11 up.
2. **ST, B2 and B3 make rare rows more influential and gain most on rare tokens,** as at
   h12.
   - Their spread is 1.07-1.31 and their key mass 1.04-1.12, against 0.95 for both.
   - Their gain over the reference grows as tokens get rarer: −0.04 to −0.11 at u = 7630,
     −0.06 to −0.12 at 1001-7629, and −0.23 to −0.30 at 101-1000.
3. **B2's and ST's gates open wider for rarer query tokens.**
   - B2's g is 0.66 at u = 101-1000 and 0.76 at 1-10, against 0.57 at u = 7630.
   - ST's gate RMS is 1.06 and 1.20 there, against 0.90.
4. **B3 backs rare rows off to its base, and its rare queries attend the most sharply of
   any arm.**
   - The rows stay nearly orthogonal to the base (|cos| ≤ 0.08).
   - Entropy is 0.30-0.44 at u ≤ 1000, against 0.49-0.50 for common tokens, more
     pronounced than at h12.
5. **At ' COVID' the σ = 0.01 arms split, where at h12 all were worse:**
   - B2 and B3 are now better than the reference, by 0.54 and 0.47;
   - ST and B1 are worse, by 0.82 and 1.16;
   - it is one token, so an anecdote again.

**Against the plan's requirement at h16, B1 is the one arm that meets all three
conditions, on the u = 101-1000 reading.** It fails only at ' COVID', which decision 8
sets aside. u = 0 (5 positions) is not read.

| Arm | Rare rows less influential? | Overall CE within the floor, or better | Rare loss not above the reference's |
|---|---|---|---|
| B1 | yes: flat identity logits, s ≈ 0.05, least key mass | yes, −0.015 | yes from u = 11 up (−0.04 at 101-1000); no at ' COVID' (+1.16) |
| B2 | no: sharper logits, most key mass, gate widest on rare tokens | yes, −0.043 | yes in every bucket from u = 1 up |
| B3 | no, by the plan's measures; rare rows back off to a sharp shared vector | yes, −0.050 | yes in every bucket from u = 1 up |
| ST | no: sharp logits, gate widest on rare tokens | yes, −0.108 | yes from u = 11 up; no at ' COVID' (+0.82) |

**The plan's second question at h16, answered:** as at h12, a rarely updated row is less
influential only in B1. At h16 that no longer costs B1 its rare tokens. It is still the
weakest gate overall (2.846, against ST's 2.753). The three gates that do not mute rare
rows gain the most on them.
