# simpleP instead of full muP

Line 2, task 2, decided 2026-11-25 (Janek). Line 2 keeps one width-scaling option for
AdamW: simpleP, `configs/_trainer/_optimizer/adamw_simplep.yaml`. Full muP (`adamw_mup`)
was built, swept against simpleP and SP, and dropped. Its code is in the first commit of
PR #202, `455421d2` (`git fetch origin pull/202/head`).

## What the two did

m = `common.dmodel / optimizer.base_dmodel`, base 256 (h4).

| | simpleP (kept) | full muP (dropped) |
| --- | --- | --- |
| hidden and head LR | ÷ m | ÷ m |
| hidden and head weight decay | unchanged | × m, so lr·wd stays fixed |
| head init | unchanged | × m^-1/2 |
| gains and embedding | base LR, no decay | base LR, no decay |

simpleP is muP's LR rule (Tensor Programs V, Table 3, Adam column) without its init and
decay rules. It is the same rule as `origin/ctx_scl-simpleP`, except that branch also
decays gains and embedding.

## Why

**It works as well.** Width sweep: depth 4, 2000 steps at batch 64, 122 runs
(`configs/jm/feature/mup_width_*`, `jupyter_notebooks/mup_lr_plots.ipynb`). Best LR per
width, as 2^-x, with its `final_ce`:

| width | SP | full muP | simpleP |
| --- | --- | --- | --- |
| h2 | 5 (4.477) | 6 (4.474) | 6 (4.504) |
| h4 | 6 (4.205) | 6 (4.201) | 6 (4.211) |
| h8 | 7 (4.011) | 6 (3.985) | 6 (3.998) |
| h12 | 8 (3.921) | 6 (3.909) | 6 (3.892) |
| h16 | 9 (3.868) | 7 (3.857) | 6 (3.838) |

Without width scaling the best LR falls about one power of 2 per width step. With
either option it stays at 2^-6 (full muP's 2^-7 at h16 beats 2^-6 by 0.014, under the
~0.016 bf16 resolution of `final_ce`). From h8 up, simpleP's loss at its best LR is equal
to or slightly below full muP's. The coordinate check (`scripts/mup_coord_check.py`) is
flat for both.

**It is simpler.** Full muP needed the optimizer option to set the model's head init: an
`optimizer.head_init_scale` key read by `_model/h_balanced/base.yaml`, so the optimizer
reached into the model config, and only the H family honoured it. It also needed the
weight decay × m rule. simpleP needs neither. A config picks the option and, optionally,
`optimizer.base_dmodel`; the model config is untouched.

**It carries over unchanged.** Every 2D weight that is not the embedding or head is
`hidden`, so MoE experts and the router get LR ÷ m, each expert like the dense FFN
(`tests/test_optimizers.py`).

## What the sweep does not show

- The runs are 2000 steps, too short for weight decay to matter, so they don't say
  whether full muP's decay scaling helps in long runs. Under simpleP, lr·wd on the big
  matrices falls as 1/m relative to the base width. At h12 with the shifted LR
  (`h12_simplep`) the matrices get exactly the baseline's LR and decay, so the effect
  would only show far above the base width.
- Full muP was never run at h12 for 50k steps. If decay at width becomes a question,
  restore `adamw_mup` from `455421d2` and run it next to `h12_simplep`.

## Using simpleP

- Add `- override ../../_trainer/_optimizer@optimizer: adamw_simplep` after the group
  entries (`configs/jm/feature/` paths), and set `optimizer.base_dmodel` only for a base
  other than 256.
- `trainer.learning_rate` is the base-width LR: gains and embedding get it, the matrices
  get it ÷ m. Tune it at the base width, or shift it as `h12_simplep` does (8.415 =
  10 − log2 3 puts h12's matrices at the tuned 2^-10).
- Use WSD: cosine still rejects param groups with different LRs.
