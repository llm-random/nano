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
decay rules. It is the same rule as `origin/ctx_scl-simpleP`, with two differences: that
branch also decays gains and embedding, and it keeps the MoE router at the base LR. Here
the router is a `hidden` matrix whose fan-in grows with width, so it gets LR ÷ m.

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
to or slightly below full muP's; that edge may come from simpleP's weaker decay (below),
not from dropping the head-init rule. The coordinate check
(`scripts/mup_coord_check.py`) is flat for both.

**It is simpler.** Full muP needed the optimizer option to set the model's head init: an
`optimizer.head_init_scale` key read by `_model/h_balanced/base.yaml`, so the optimizer
reached into the model config, and only the H family honoured it. It also needed the
weight decay × m rule. simpleP needs neither. A config picks the option and, optionally,
`optimizer.base_dmodel`; the model config is untouched.

**It carries over unchanged.** Every 2D weight that is not the embedding or head is
`hidden`, so MoE experts and the router get LR ÷ m, each expert like the dense FFN
(`tests/test_optimizers.py`).

## What the sweep does not show

- **The decay rule.** Decay did matter in these runs: at the best LR, 2^-6, with decay
  0.1, AdamW's decay timescale 1/(lr·wd) is 640 steps, and wd·Σlr over the 2000 steps is
  about 2.7. The two options decayed differently: full muP keeps lr·wd fixed across
  width, while under simpleP it falls as 1/m, so full muP decayed 2× (h8) to 4× (h16)
  harder. Short runs tend to favour weaker decay, so the sweep shows that the best LR
  holds across width under both options, but it does not settle the decay rule (see
  Later). At h12 with the shifted LR (`h12_simplep`) the matrices get exactly the
  baseline's LR and decay, so this matters only further from the base width.
- **Width only.** simpleP scales with width by design, not depth. The sweep held depth at
  4 blocks and ran 2000 steps; in the H family depth grows with h, so its 2^-6 is not
  h12's LR (`h12_simplep` starts from the tuned 2^-10 instead).
- **One h12 run.** At a single width simpleP only fixes the LR ratio between roles (at
  h12, gains and embedding get 3× the matrices' LR), so `h12_simplep` tests that ratio,
  not width transfer. Its fair baseline is `h12_param_groups` (same groups and decay, one
  LR), not `h12_chinchilla` (one group, decay on everything).
- Full muP was never run at h12 for 50k steps. If decay at width becomes a question,
  restore `adamw_mup` from `455421d2` and run it next to `h12_simplep`.

## Using simpleP

- Add `- override ../../_trainer/_optimizer@optimizer: adamw_simplep` after the group
  entries (`configs/jm/feature/` paths), and set `optimizer.base_dmodel` only for a base
  other than 256.
- `trainer.learning_rate` is the base-width LR: gains and embedding get it, the matrices
  get it ÷ m. It transfers across width at the same depth and run length. For an
  H-family size, start from an LR tuned at that size and shift it as `h12_simplep` does
  (8.415 = 10 − log2 3 puts h12's matrices at the tuned 2^-10).
- Cosine and WSD both work: each param group decays to its own final fraction.
  `validate()` rejects a `base_dmodel` ≤ 0.
- At startup, rank 0 logs each param group once: role, LR, decay and its parameters.
  `train/lr` logs group 0 (gains, the base LR); the matrices' LR is that ÷ m.

## Later

A width rule for weight decay, on its own branch (Janek, 2026-11-25). The literature
leans towards scaling decay with width, which this sweep could not test:
- Wang & Aitchison, "How to set AdamW's weight decay as you scale model and dataset
  size" (arXiv 2405.13698): keep the timescale 1/(lr·wd) fixed, i.e. decay × m under
  muP's LR, which is full muP's rule; with fixed decay, muP's LR transfer breaks down.
- Kosson et al., "Weight Decay may matter more than µP for Learning Rate Transfer in
  Practice" (arXiv 2510.19093): independent weight decay, not muP's update scaling, is
  what drives LR transfer; muP acts mostly as an implicit warmup.
- "Robust Layerwise Scaling Rules by Proper Weight Decay Tuning" (arXiv 2510.15262):
  decay ∝ √d alongside muP's LR rule, transferring both LR and decay across width.
