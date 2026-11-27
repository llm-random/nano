# Comparing runs

**Status: established (2026-11-27).** The rules every A/B in this repo follows: what fixes a
run's weights and data, which runs pair with which, how a difference is measured, and when it
is a tie. Each rule was set in a task notebook, most of them now in
[`notes/archive/`](../notes/archive/); the evidence stays there, and the rows below link to it.

## What fixes a run

| What | Fixed by | Since | Source |
|---|---|---|---|
| The seed | `trainer.train_dataloader.dataset.seed`, 123 in every config. It seeds the data, and through `torch.manual_seed` the init: one seed moves both | — | [main.py:396](../../main.py#L396); a separate init seed is proposed in [review_upgrades.md](../notes/review_upgrades.md), U13 |
| Each part's initial weights | The seed and the part's name. The token embedding, each block's attention and feed-forward, and the head draw from their own seeds; optional parameters (the KV shift's coefficients, the ST gate) from private generators. A switch moves only what it switches | 2026-11-25 (`cfc4f313`, `acc7f104`) | [src/core/init_rng.py](../../src/core/init_rng.py), [tests/test_init_rng.py](../../tests/test_init_rng.py), [review_issues.md](../notes/archive/review_issues.md) N6 |
| The training rows and their order | The seed and the train loader's `num_workers`: 2 in every FineWeb run, and another count trains on other rows in another order. **Not** the GPU count, **not** `gradient_accumulation_steps`: every GPU reads the whole stream and keeps every world_size-th sequence. A batch that does not split evenly over the GPUs is refused | GPU count: 2026-11-26 (`c5097166`) | [configs/_dataset/fineweb.yaml](../../configs/_dataset/fineweb.yaml), [tests/test_data_order.py](../../tests/test_data_order.py), [review_upgrades.md](../notes/review_upgrades.md) U12 |
| The held-out set | The end of each of FineWeb's 64 equal segments (262,144 rows, disjoint from training). `final_ce` reads its first 8,192 sequences of `sequence_length` tokens, so it compares only between runs at one `sequence_length` | 2026-11-26 (PR #200, `b24c5406`) | [configs/_dataset/fineweb.yaml](../../configs/_dataset/fineweb.yaml) |
| The step count | `trainer.n_steps`, or `trainer.n_tokens_b`, which counts batch × (`sequence_length` − 1) tokens a step and rounds up: 4, 8 and 16B give 3815, 7630 and 15259 steps | 2026-11-25 (`20647031`) | `steps_for_tokens` in [src/definitions.py](../../src/definitions.py) |

## Which runs pair

Two runs *pair* when they see the same batches at the same step from the same initial weights.
Their step-by-step loss difference then measures only the change under test.

| Recipe | Date | What changed | Pairs with the runs before it in |
|---|---|---|---|
| The experiment runs: Stages 1–2c, bank gates, σ grid, init ablations, document mask, settle runs | 2026-11-06 → 11-20 | — | init, data, steps, at equal seed, batch and `sequence_length` |
| Standard v1: [configs/standard/](../../configs/standard/) | 2026-11-22 (`7597f382`) | the six standard configs; `bank_lr_ratio` defaults to 4 | everything: v1 at h12 *is* the recipe of `zdmv3qci` (AME) and `chne4w91` (dense) |
| Standard v2 | 2026-11-25 | each part inits from its own seed (N6); the QK-norm keeps q and k in the compute dtype (N7); `bank_init_std` defaults to 1e-4 under the QK-norm (N1) | data order, steps and schedule, **not init**. v2 ties v1: last-500 paired gap dense +0.0071 (`p15ixjnf`), AME +0.0039 (`4fknyjba`) ([review_upgrades.md](../notes/review_upgrades.md), U11) |
| Standard v3, current for dense | 2026-11-26 (`29a04918`) | the spread held-out set (PR #200) | **nothing**: the training rows shift with the cut, and `final_ce` scores other samples |
| Standard v4, AME only, current | 2026-11-27 | AME's feed-forward widened by dmodel/3: `dff = 32·round(26H/3)`, 3328, 4448 and 5536 at h12, h16 and h20 ([AMEm](ame-model.md#amem--the-matched-variant)). AME then matches dense of the same `H` in active parameters and training FLOPs per token; dense is unchanged | v3 AME in data order, steps, schedule and the init of every part but the feed-forward; v3 dense in data order, steps and schedule |

The config headers in `configs/standard/` say the same. Each version has its own run-name suffix
and tag (`_v3`, `standard_v3`; AME `_v4`, `standard_v4`), so W&B never files two recipes under
one name.

## Measuring a difference

| Measure | What it is | Precision | Source |
|---|---|---|---|
| **Paired gap** (primary) | The mean of (new − reference) `steps/train/loss` over the last 500 steps. Only for runs that pair | Finer than `final_ce`: the per-step rounding errors are independent and average out | `scripts/paired_gap.py`, on branch `bank_sigma_grid` only (not on `ame_main`); it reads W&B project `nano-ame` |
| `final_ce` | The held-out loss after training: a sample-weighted mean of bf16 batch losses | ±0.008 per run, ±0.016 on a difference | `final_eval` in [src/core/trainer.py](../../src/core/trainer.py) |
| The common evaluator | `score` in [parity_numerics.py](../scripts/parity_numerics.py): a checkpoint's CE on the full held-out set, unrounded | Exact; the bank-gates reports used ±0.019 with it | It cannot score FSDP2 runs ([review_upgrades.md](../notes/review_upgrades.md), U5) |
| The step-0 check | The new run's first logged loss must equal the reference's. A switch that changes the init forward (QK-norm, the shift) gives a new step-0 number by design | Exact | [settle_runs_report.md](../notes/archive/settle_runs_report.md), "How each run is judged" |

## When a difference is a tie

- **AME, h12, 4B tokens: under 0.022** in size. That is 3 × the sd of 0.0072 of three runs of
  one config at one seed, so it measures run-to-run nondeterminism, not seed-to-seed variance,
  which no one has measured (U6, U13). All 14 AME runs of the efficiency-parity audit, six
  configurations that differ only in implementation details, ended within 0.015 of each other
  ([ame_efficiency_parity_report.md](../notes/archive/ame_efficiency_parity_report.md);
  [ame_comparison_h12.md](../notes/archive/ame_comparison_h12.md), F12). No band has been
  measured at h16 or h20; the h12 one stands in.
- **Dense: no measured band.** Same-seed dense runs spread 6–24× more than AME's: its six runs
  in that audit spanned 0.27, through early loss-spike episodes (same sources). A dense gap
  over 0.022 is read with its loss curves: a spike episode on either side explains most of
  them.
- **Spikes happen in AME too.** The standard AME h12 recipe spiked in 1 of its 3 runs
  (`zs1bceoy`, steps 330–360, +0.077), and its rerun tied
  ([standard_models_report.md](../notes/archive/standard_models_report.md); U6).

## Across clusters

- **Loss pairs across clusters, within the band.** Entropy replicates of a LEM run paired at
  +0.0157, +0.0002 and +0.0053: level or behind, never ahead. The standard AME rerun on entropy
  tied LEM's reference at +0.0017.
- **Speed does not.** Entropy runs AME h12 about 10% faster than LEM (3.36–3.38 against
  3.75 s/step, the same config). Measure a change's cost on one cluster: across the two, the
  document mask's +12% looked like +1.5%.
- **Paper numbers stay within one cluster.**

Sources: [settle_runs_report.md](../notes/archive/settle_runs_report.md), "How each run is
judged" and Result 4; [review_upgrades.md](../notes/review_upgrades.md), U7.

## Rerunning or resuming an older run

- **An exact rerun needs the code the run used:** its cemetery branch, `<run name>_<timestamp>`,
  which `run_exp.py` pushes at submit. A run submitted by hand (a `.` in its name) has none.
- **On today's code** (2026-11-27), `trainer.bank_lr_ratio=1` and
  `++common.attention.ame.bank_init_std=1.0` restore a pre-merge AME run's optimizer and init
  scale. They do not restore its init draw (v2) or its training rows (v3).
- **To resume a pre-merge AME checkpoint,** set `trainer.bank_lr_ratio=1`: the old optimizer has
  no bank group. Without it the ordinary path fails loudly, and memory parallelism resumes at 4×
  without an error. A weights-only load works at either ratio.

Sources: [review_upgrades.md](../notes/review_upgrades.md), U7;
[review_issues.md](../notes/archive/review_issues.md), N3;
[configs/_trainer/llama.yaml](../../configs/_trainer/llama.yaml).

## Making a paired run

- **Start from the reference's logged W&B config,** not from the config file it names, and
  change only the setting under test. The logged config is the truth.
- **Check the step-0 loss** about 2 minutes after the job starts, then look again only at the
  end of training.
- **Submit in two halves,** so the W&B key never lands in a world-readable cemetery: the local
  `run_exp.py` steps with the secrets unset, then clone and `sbatch` from a shell that read the
  key from stdin ([settle_runs_report.md](../notes/archive/settle_runs_report.md#submission)).

## Reporting a result

- **Every result row carries** the arm, n, seed and tokens; the total, backbone and bank
  parameters; FLOPs per token; the final eval loss; and the commit, config path and W&B run id.
- **Every AME-against-dense comparison states the weight-decay difference.** AME's banks are
  exempt from weight decay; dense `W_Q` and `W_K` are not
  ([ame_review_and_fixes.md](../notes/archive/ame_review_and_fixes.md), 2.6).

Source: [ame_experiments.md](../notes/archive/ame_experiments.md), "Reporting — every stage".
