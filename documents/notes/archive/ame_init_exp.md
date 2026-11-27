# AME init experiment: Xavier embedding, head and banks, pkmf's Linear init (plan and log)

**Archived 2026-11-27 (historical).** Done on 2026-11-09: the new init bundle gains nothing,
and it was rejected. Its code stays on the unmerged branch `ame_initialization`. **Today:**
[standard_model_improvements.md](../standard_model_improvements.md) ("Other rejected
changes") and
[ame_implementation.md](../ame_implementation.md#weight-initialization-dense-and-ame).

**Status:** DONE, 2026-11-09. All four runs finished; the answer is no (see Log,
Analysis). Planned 2026-11-08 (Maciej). Follow-up, planned:
[ame_init_embed_ablation.md](ame_init_embed_ablation.md), the new init with the old
embedding at h12, then n ± 1.
Background: [pkmf_init_comparison.md](../pkmf_init_comparison.md) and
[ame_implementation.md](../ame_implementation.md) (weight initialization).

**Question:** do these four init changes, made the default for every model, lower final CE
for AME and dense at h12/4B and h16/8B?

## What changes

Branch `ame_initialization` from `ame_main`, in a new worktree `~/code/nano-ame-init`. It
is not merged into `ame_main`. The new defaults apply to every model, since every config
builds from [configs/_model/h_balanced/base.yaml](../../../configs/_model/h_balanced/base.yaml).

| Weights | Now (`ame_main`) | New default | σ at h12 / h16 |
|---|---|---|---|
| Token embedding `[V, d]` | N(0, 1) cut at ±2 (std 0.88); weight decay 0.1 | Xavier normal, σ = √(2/(V + d)), V = 50,304; **no weight decay** | 0.00626 / 0.00624 |
| Head `[V, d]` | truncated normal, σ = 1/√d, cut at ±2σ | Xavier normal, σ = √(2/(V + d)); weight decay stays 0.1 | 0.00626 / 0.00624 |
| AME banks `q_bank`, `k_bank` `[V, d]` | N(0, 1) (`bank_init_std: 1.0`); no weight decay | Xavier normal, σ = √(2/(V + d)); still no weight decay | 0.00626 / 0.00624 |
| Attention and FF `Linear` weights | truncated normal, σ = 1/√fan_in, cut at ±2σ | pkmf's σ = √(0.1/fan_in), cut kept at ±2σ | 0.0114 / 0.0099 at fan_in = d; 0.0057 / 0.0049 for FF out |

Everything else stays as it is: norm gains, AME scalars, ALiBi, sink, KV shift, schedule,
and weight decay 0.1 on all other weights.

**Where:**
- [base.yaml](../../../configs/_model/h_balanced/base.yaml):
  - embedding `init_fn` → `torch.nn.init.xavier_normal_`;
  - head `partial_init_fn` → a Xavier helper in [model.py](../../../src/core/model.py);
  - FF `scale` and attention `init_scale` → √0.1, so `llm_random_weight_init` gives
    σ = √(0.1/fan_in), cut at ±2σ.
- [model.py](../../../src/core/model.py): `TransformerEmbedding` names its `weight` in
  `no_weight_decay_parameters`; `parameter_groups` already reads that attribute.
- [ame.py](../../../src/core/ame.py): the banks default to Xavier. An explicit
  `bank_init_std` number still draws N(0, σ²), so the stage 1-2 configs keep their meaning.
  Update the typed schema.
- Xavier takes its fans from the weight's shape, so `xavier_normal_` gives √(2/(V + d)) for
  all three `[V, d]` tables.
- **Memory-parallel banks (Tier 2):** check that the fp32 masters hold the Xavier draw.

## Local CPU tests

- The full pytest suite passes. Update any test that pins the old init. Baseline on
  `ame_main`: 393 passed, 18 skipped.
- A new init test, on tiny dense and AME models:
  - the embedding, head and banks have std ≈ √(2/(V + d)), untruncated;
  - each `Linear` has std ≈ 0.88·√(0.1/fan_in), with max |w| ≤ 2σ;
  - the embedding and banks sit in the weight-decay-0 group, and the head and `Linear`
    weights at 0.1;
  - the memory-parallel masters equal the module's draw.
- A CPU smoke of a few steps, on a tiny AME model (KV shift, ALiBi) and a tiny dense one:
  - the loss stays finite;
  - the first loss is ≈ ln V = 10.83, because the logits start at std 0.17-0.20.
- Compose the four run configs and print their init values.

## Runs on LEM

- **Configs:** `configs/ame_init/<run>.yaml`. Each copies its reference's config and
  changes only the names; the init comes from the branch defaults.
- **W&B:** project `nano-ame`, run name `ame_init_<run>`.
- **Seed:** 123, the references' data order.

| Run | Recipe: config copied | Tokens, steps | LR n | Limit | Reference, old init: W&B, final_ce, s/step |
|---|---|---|---|---|---|
| `ame_h12` | S2-PS, AME with KV shift and ALiBi: [bank_gates/base.yaml](../../../configs/bank_gates/base.yaml) | 4B, 3,815 | 8 | 7 h | `doi36p8k`, 3.1250, 3.746 |
| `ame_h16` | S2b-PS h16: [bank_gates/h16_base.yaml](../../../configs/bank_gates/h16_base.yaml) | 8B, 7,630 | 8 | 24 h | `hjw7znbz`, 2.8594, 6.810 |
| `dense_h12` | S1-D, GQA with RoPE: [stage_1/dense_h12_lr_grid.yaml](../../../configs/stage_1/dense_h12_lr_grid.yaml), the grid replaced by `learning_rate: 8` | 4B, 3,815 | 8 | 5 h | `x0xlqroj`, 2.9824, 3.272 |
| `dense_h16` | S1b-D: [stage_1b/dense_h16_lr_grid.yaml](../../../configs/stage_1b/dense_h16_lr_grid.yaml), `learning_rate: 9` | 8B, 7,630 | 9 | 24 h | `eh2sjsh7`, 2.7441, 5.964 |

**Submitting:**
- Commit everything on `ame_initialization` first: run_exp snapshots the tree, untracked
  files included.
- Submit from `~/code/nano-ame-init`, one `run_exp.py` at a time:

```
run_exp.py --config-path configs/ame_init --config-name <run> +infrastructure.slurm.exclude=r16-11 < /dev/null
```

- **Expected durations,** as the references took: about 4.3 h, 15 h, 3.5 h and 12.6 h.

**Checks:**
- **About 2 minutes after each start:** the job is running, the first loss is ≈ 10.83, and
  s/step is within 10% of the reference's.
- **At step 500 and mid-run:**
  - no NaN;
  - loss spikes and grad norm. The Xavier embedding sits in the small-embedding range
    where Takase et al. saw spikes;
  - train loss against the reference's at the same steps.
- **Never cancel a job without asking.**

## How to analyze

- **Each run against its reference:**
  - final_ce;
  - the paired train-loss gap (new − reference at the same step), for each part of
    training and for the last 500 steps, as in
    [ame_improve_runs_report.md](ame_improve_runs_report.md).
- **The paired gap decides,** since final_ce is bf16-quantized (±0.008 per run).
  - AME ties inside ±0.022.
  - Dense has no replicate floor, so report its gap without calling small ones.
- **The AME − dense gap under the new init,** against the old: 0.143 at h12
  (3.1250 − 2.9824) and 0.115 at h16 (2.8594 − 2.7441).
- **Health:** spikes, the grad-norm max after step 100, s/step, peak memory.

## Log

### Implementation

2026-11-08, branch `ame_initialization` from `ame_main` `33b2f059`, worktree
`~/code/nano-ame-init` (`data` symlinked from the main checkout; gitignored). The
code, tests, run configs and script are commit `79108114`.

- [base.yaml](../../../configs/_model/h_balanced/base.yaml):
  - embedding `init_fn` is `torch.nn.init.xavier_normal_`;
  - head `partial_init_fn` is the new `src.core.model.xavier_normal_weight_init`,
    which returns `xavier_normal_` (torch reads both fans off the weight's shape);
  - attention `init_scale` and both FF `scale`s are `${eval:'0.1 ** 0.5'}`.
- [model.py](../../../src/core/model.py): `TransformerEmbedding.no_weight_decay_parameters
  = ("weight",)`.
- [ame.py](../../../src/core/ame.py):
  - `bank_init_std` defaults to `None`, which draws `xavier_normal_` on `[V, H·d]`;
  - a number still draws N(0, σ²), and a value ≤ 0 is refused;
  - header note 3 is rewritten. It said rows must start at σ ≈ 1. With QK-norm, σ
    only sets how fast a row turns; without QK-norm, pass `bank_init_std: 1.0`.
- **Typed schema:** no change needed. `common.attention.ame` is untyped (`Any`) in
  [definitions.py](../../../src/definitions.py), and `build_attention` checks the option
  names; `bank_init_std` is already one, and `null` passes.
- **Memory-parallel masters:** no change needed. `ShardedBank.load_from` copies the
  drawn bank into the fp32 master, so the master holds the Xavier draw (tested).
- **Run configs:** [configs/ame_init/](../../../configs/ame_init/). The AME ones copy
  `bank_gates/base.yaml` and `h16_base.yaml`. The dense ones copy the grid files with
  `learning_rate` fixed at 8 and 9, and stay without `/base_schema` like their sources;
  `run_exp.py` schema-checks them anyway. Each reference's logged W&B config matches
  its copy in seed (123), n, steps, warmup, micro-batch, attention and banks.
- **Analysis script:** [scripts/paired_gap.py](../../../scripts/paired_gap.py), on the
  branch. Checked on a known pair: `y5ogvt3k` against `doi36p8k` gives Δ final_ce
  −0.0312 and a last-500 gap of −0.0342, as
  [ame_improve_runs_report.md](ame_improve_runs_report.md) has.
- **Corrections to the plan:**
  - The plan says an explicit `bank_init_std` keeps the stage 1-2 configs' meaning.
    It doesn't: only `bank_gates/*` set it. On this branch the stage 1-2 AME configs
    get Xavier banks too, which is the "default for every model" asked for;
    `ame_main` is untouched.
  - The code-level `init_scale=1` defaults in `build_attention` and `AMEAttention`
    stay. Every model config passes `init_scale` from `base.yaml`; only layers built
    directly in unit tests see the code default.
- **Pairing holds:** the new draws use the global torch RNG differently, but the
  data order comes only from the dataset's own `random.Random(seed)` and HF's seeded
  shuffle. So each step sees the same batch as in the reference.
- **Not resumable:** old dense checkpoints can't resume their optimizer state on this
  branch, since one group became two. No run here resumes.

### Local tests

- **Full suite** (`-m "not gpu"`, the repo's default; this machine has no CUDA):
  **every test passes, 428 of 428**, with the 13 `gpu` tests deselected.
  - The full run: 406 passed, 22 skipped. The skipped ones need `data_eval/`, which
    the worktree lacked.
  - With `data_eval` symlinked from the main checkout (gitignored), those 22 pass
    too: the two files that hold them gave 47 passed, 0 skipped.
  - The plan's `ame_main` baseline (393 passed, 18 skipped) ran in the main
    checkout and on an earlier `ame_main`, so its counts don't line up one to one
    with these. What matters holds: nothing fails.

  A first run without `-m` also ran the 13 `gpu` tests. They call
  `.cuda()`, so they fail on any machine without CUDA ("No CUDA GPUs are
  available"), on any branch. That run's other failures were caused by the change
  and are fixed in `79108114`:
  - **Two `test_ame` grouping tests pinned the old rule:** the embedding decays, and
    a dense model gets one optimizer group. They now check the new rule: the
    embedding alone is undecayed in a dense model, and the head still decays. The
    one-group guarantee is now tested on a bare stack of `Linear`s.
  - **`test_memory_parallel`:** the loss trajectories at `mem_parallel_size` 1 and 2
    split by 5e-11 relative at step 2, against a 1e-12 tolerance. The toy's AME now
    gets Xavier banks (σ = 0.079 at V = 256, d = 64), so Adam turns a row by about
    LR/σ = 0.13 rad per step instead of 0.01, and float64 rounding grows faster.
    Sharding is still exact: the 1e-14 gradient check passes under Xavier. The toy
    now pins `bank_init_std: 1.0`, the regime the tolerance was set in, and the file
    passes (28 tests).
- **New test,** [tests/test_init_defaults.py](../../../tests/test_init_defaults.py): 12 pass,
  on tiny h4 models (V = 4,096) composed from `base.yaml`, dense and AME. It checks:
  - the embedding, head and banks have std √(2/(V + d)) to 1%, uncut;
  - every attention and FF `Linear` has std 0.88·√(0.1/fan_in), with max |w| ≤ 2σ;
  - the embedding and banks are in the weight-decay-0 group, the head and `Linear`s
    at 0.1;
  - an explicit `bank_init_std: 1.0` still draws N(0, 1), and 0 or a negative is
    refused;
  - the Tier 2 fp32 masters equal the module's draw, and the bf16 copy is its cast;
  - the composed h12 config carries the new targets and scales.
- **CPU smoke through `main.py`** (`bank_gates/smoke_local`, h4, V = 50,304, 4 steps,
  `bank_init_std=null`):

  | Model | Loss, step 0 | Loss, step 3 | final_ce | End of run |
  |---|---:|---:|---:|---|
  | AME, KV shift + ALiBi, Tier 2 banks | 10.821 | 10.053 | 9.879 | checkpoint, bank state and `token_counts.pt` written |
  | Dense GQA + RoPE | 10.827 | 10.060 | 9.789 | checkpoint written |

  ln V = 10.826. At h4 the logits start at std ≈ 0.10.
- **Logits at init, real sizes** (CPU forward, random ids): std 0.173 at h12 and 0.200
  at h16. Cross-entropy against random targets is 10.841 and 10.850.
- **The four configs, composed as `run_exp.py` does** (missing keys, grid,
  `check_point`) and built on meta. All of them get Xavier embedding and head, and
  scale 0.316228 on the attention and FF `Linear`s.

  | Run | Parameters | Weight decay 0.1 | Weight decay 0 | n, steps, warmup | Micro-batch |
  |---|---:|---:|---:|---|---|
  | `ame_h12` | 1,103,599,056 (as stage 2) | 137,743,104 | 965,855,952 (embedding + banks + scalars) | 8, 3,815, 38 | 32 × 16 |
  | `ame_h16` | 1,986,332,928 | 286,426,112 | 1,699,906,816 | 8, 7,630, 76 | 16 × 32 |
  | `dense_h12` | 183,454,464 | 144,820,992 | 38,633,472 (= V·d, the embedding) | 8, 3,815, 38 | 32 × 16 |
  | `dense_h16` | 354,714,624 | 303,203,328 | 51,511,296 (= V·d) | 9, 7,630, 76 | 16 × 32 |

### Runs

Submitted with the plan's command from `~/code/nano-ame-init` at `950c74e0` (code
`79108114` plus this doc), one `run_exp.py` at a time. Each clone is under
`/lustre/pd03/plgrid/plgllmefficont3/nano/plgmstefaniak/llmrandom_cemetery/`.
Pre-flight: no Lustre quota limit for the user or group, and 502 TB free on pd03,
so the final checkpoints (about 23 GB for `ame_h16`) fit.

| Run | Job | Clone | W&B | Node | Submitted | Finished | State |
|---|---|---|---|---|---|---|---|
| `ame_h12` | 6275639 | `ame_init_ame_h12_2026-11-08_18-56-20` | `ycwqa7k9` | r14-9 | 18:56 | 11-09 00:06 | COMPLETED, exit 0, 4 h 14 min |
| `ame_h16` | 6275518 | `ame_init_ame_h16_2026-11-08_18-51-20` | `3lgj6lxg` | r18-6 | 18:51 | 11-09 10:41 | COMPLETED, exit 0, 15 h 02 min |
| `dense_h12` | 6275650 | `ame_init_dense_h12_2026-11-08_18-56-58` | `355fr6rn` | r16-1 | 18:57 | 22:40 | COMPLETED, exit 0, 3 h 42 min |
| `dense_h16` | 6275560 | `ame_init_dense_h16_2026-11-08_18-52-04` | `3vpbp9oj` | r18-13 | 18:52 | 11-09 08:56 | COMPLETED, exit 0, 13 h 15 min |

### Checks

Gaps are paired: new − reference at the same step, as a mean over the window.

**`dense_h12`, start check, 19:02 (step 47):**
- First loss 10.8125, which is ln V = 10.826 in bf16. The reference started at
  11.1875: its old head gave logits of std ≈ 0.88.
- 3.269 s/step against the reference's 3.274.
- No NaN. The grad norm is 2.95 at step 0 and 5.40 at step 1, against 2.38 and 2.92.
- **Early on, the new init is behind:** it plateaus at 7.6-7.7 (plausibly the
  unigram level; not measured) from step 10 to step 47, while the reference has
  reached 6.81 by step 47.
  - Paired gap: +0.48 over steps 1-9, +0.52 over steps 10-49.
  - A small init has to grow its weights first. The step-500 check shows whether the
    gap closes.

**`dense_h12`, step-500 check, 19:30 (step 524). The new init stalls; the gap grows.**
- No NaN and no spike. The grad-norm max after step 100 is 2.20, against the
  reference's 2.58, and s/step is 3.280 against 3.276.
- **The loss stagnates:** 6.50 at step 250, 6.28 at step 500 (reference 5.06 and
  4.25). The paired gap grows:

  | Steps | 1-9 | 10-49 | 50-99 | 100-199 | 200-499 | 500-524 |
  |---|---:|---:|---:|---:|---:|---:|
  | Gap | +0.48 | +0.53 | +1.00 | +1.10 | +1.72 | +2.15 |

- **Why, from the per-block norms W&B logs** (mean per-token L2 norm of the
  residual and of each sublayer's update):

  | | New, step 0 | New, step 100 | New, step 500 | Reference, step 500 |
  |---|---:|---:|---:|---:|
  | Embedding (block 0's residual) | 0.17 | 0.84 | 3.15 | 20.4 |
  | Block 0 attention update | 0.22 | 1,471 | 3,849 | 6.0 |
  | Block 0 FF update | 0.27 | 27,355 | 9,946 | 256 |
  | Residual into block 11 | 6.3 | 108,000 | 23,000 | 776 |

  - The embedding is drowned: the first attention output is about 1,000 times its
    norm. From block 0's FF on, the stream carries the attention's mix over the
    context, not the current token.
  - In the reference the embedding dominates about 3:1, so each layer sees the
    token. Here the model must recover which token it is at.
- **Likely cause:**
  - The Xavier embedding (std 0.006, norm 0.17) feeds a pre-LN stack without the √d
    multiplier or embedding LayerNorm that Takase et al. pair small embeddings with.
    Here it shows up as stagnation, not spikes.
  - The LR (n = 8, tuned for the old init) makes it worse. Adam moves each weight by
    about lr per step whatever its size, and lr/std is now about 0.6 for the
    embedding and head, against 0.004 and 0.12 before.
- **The AME runs likely stall the same way:** they share the embedding and the
  stack, and AME's id-indexed Q/K don't reach the values or the FF.

**`ame_h16`, start check, 19:43 (step 8):**
- First loss 10.8125, which is ln V in bf16; the reference started at 11.25.
- 6.767 s/step against the reference's 6.786.
- No NaN. The grad norm is 5.53 at step 0 and 6.59 at step 1, against 2.95 and 4.65.
- The first 8 steps track the reference: a paired gap of +0.22 over steps 1-8, and
  7.875 against 8.06 at step 8.

**`dense_h16`, start check, 19:54 (step 99; it started at 19:40, before the queue
watcher's first look):**
- First loss 10.8125; the reference started at 11.25.
- 5.946 s/step against the reference's 5.959. No NaN.
- **The same early stall as `dense_h12`:** a paired gap of +0.54 over steps 1-9,
  +0.61 over 10-49 and +1.06 over 50-99 (`dense_h12`: +0.48, +0.53, +1.00). It is at
  7.66 at step 50 and 6.94 at step 99, against 6.53 and 5.84.

**`ame_h12`, start check, 19:56 (step 18):**
- First loss 10.875, the bf16 value next to ln V (bf16 steps are 0.0625 here); the
  reference started at 11.25.
- 3.771 s/step against the reference's 3.743 (+0.7%). No NaN.
- **A one-step grad-norm jump:** 12.69 at step 1 (the reference had 3.20), then back
  to 2.4 by step 2.
- A paired gap of +0.55 over steps 1-9. At step 18 it is at 7.69 against 7.34.

**Step-500 checks, 20:35. AME doesn't stall; dense h16 stalls like dense h12, but
less.**

| Run, at the check | Loss at step 500 (reference) | Gap 100-199 | Gap 200-499 | Gap 500+ | Grad-norm max after step 100 (reference) | s/step (reference) |
|---|---|---:|---:|---:|---:|---:|
| `ame_h12` (step 622) | 4.28 (4.22) | −0.017 | +0.016 | +0.077 (500-622) | 0.70 (0.65) | 3.782 (3.747) |
| `dense_h16` (step 505) | 5.22 (3.70) | +1.44 | +1.68 | +1.54 (500-505) | 1.42 (0.89) | 5.954 (5.970) |
| `ame_h16` (step 515, 20:43) | 4.25 (4.19) | −0.213 | −0.070 | +0.043 (500-515) | 0.81 (0.71) | 6.800 (6.810) |

- **`ame_h12`** catches up by step 100 and stays level through step 500. Its gap
  creeps up to +0.077 over steps 500-622; the mid-run check shows whether it keeps
  growing. No NaN or spike.
- **`ame_h16`** is ahead of its reference from step 50 to step 400 (−0.21 over
  100-199) and level by step 500. No NaN or spike.
- **`dense_h16`** peaks at +1.69 at step 300 and narrows to +1.52 at step 500. Its
  loss keeps falling (6.38 at step 200, 5.22 at step 500), unlike `dense_h12`'s flat
  6.3-6.5.
- **Why AME escapes, from its block norms:**
  - AME's embedding is drowned too: 0.44 → 1.1 at block 0, against an attention
    update of 847 → 327 over steps 100-500, about 300:1. Its reference keeps the
    embedding at about 20 against an update of 7-10.
  - The difference is plausibly where attention gets token identity. AME's Q/K come
    from id-indexed banks, so every layer's attention pattern sees the token ids
    directly, and a head can attend to the current position from step 0 and copy
    its value.
  - Dense Q/K are projections of the drowned stream through small weights. They
    start near-uniform and must learn to find the current token first.
  - AME's stream also stays far smaller: 4,300 into block 11 at step 100, against
    108,000 for dense h12.

**`dense_h12`, mid-run check, 20:51 (step 1,939 of 3,815). Narrowing too slowly to
close.**

| Step | 500 | 750 | 1,000 | 1,250 | 1,500 | 1,750 | 1,900 |
|---|---:|---:|---:|---:|---:|---:|---:|
| New | 6.28 | 6.00 | 5.59 | 5.34 | 5.09 | 4.94 | 4.84 |
| Reference | 4.25 | 3.86 | 3.63 | 3.48 | 3.41 | 3.36 | 3.30 |
| Difference | +2.03 | +2.14 | +1.97 | +1.86 | +1.69 | +1.58 | +1.55 |

- Paired gap: +1.75 over the first quarter, +1.79 over the second, +1.66 over steps
  1,435-1,934. Eval loss is 4.75 at step 1,943.
- **Health:** no NaN.
  - Spikes: one cluster of 7 at steps 513-523; the reference had its own cluster of
    11 at steps 194-203.
  - Grad-norm max after step 100: 3.14 at step 794 (reference 2.58 at step 198).
  - 3.280 s/step against 3.272.
  - `max_memory_gib` is logged only at the end of a run.

**`ame_h12`, mid-run check, 22:01 (step 1,934 of 3,815). Behind by +0.08 at its
widest, narrowing steadily since.**

| Steps | 0-249 | 250-499 | 500-749 | 750-999 | 1,000-1,249 | 1,250-1,499 | 1,500-1,749 | 1,750-1,942 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Paired gap | +0.078 | +0.025 | +0.080 | +0.070 | +0.054 | +0.040 | +0.029 | +0.023 |

- The gap opens again after step 500, peaks at +0.080 over steps 500-749, then
  shrinks by 0.010-0.016 per 250 steps. Eval loss at step 1,900 is 3.344 against
  3.313 (+0.031; eval loss is bf16 too).
- **Health:** no NaN, and no spike in either run. The grad-norm max after step 100
  is 0.70 at step 156 (reference 0.65 at step 130). 3.781 s/step against 3.746
  (+0.9%).
- **Block norms, step 1,900:**
  - From block 5 on, the stream is the reference's size: 145 against 153 into
    block 5, and 393 against 411 into block 11. At step 1,000 it was 264 and 579,
    above the reference's 210 and 470.
  - Block 0 still drowns the embedding: 2.1 against an attention update of 74,
    35:1. In the reference the embedding is 13.1 against 5.7.
  - For comparison, `dense_h12` at step 1,900: embedding 4.2 against 1,485 (350:1),
    and 5,009 into block 11 against its reference's 344.

**`dense_h12`, end of run, 22:40 (COMPLETED, exit 0, 3 h 42 min on r16-1). It never
recovers: final_ce 4.2188 against 2.9824.**

| Steps | 0-499 | 500-999 | 1,000-1,499 | 1,500-1,999 | 2,000-2,499 | 2,500-2,999 | 3,000-3,499 | 3,500-3,814 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Paired gap | +1.40 | +2.13 | +1.87 | +1.63 | +1.46 | +1.36 | +1.30 | +1.26 |

- The gap peaks over steps 500-999. It then narrows ever more slowly: by 0.26
  per 500 steps just after the peak, and by 0.03 over the last 315 steps. The last
  500 steps average +1.271. Eval loss at step 3,800 is 4.22 against 2.98.
- **Health:** no NaN.
  - Spikes: the one cluster of 7 at steps 513-523; the reference's 11 are at steps
    194-203.
  - Grad-norm max after step 100: 3.14 at step 794 (reference 2.58 at step 198).
  - 3.281 s/step against 3.272. Peak memory 46.99 GiB, the same as the reference.
- **Checkpoint:** the step-3,814 model and training state, 2.1 GB, in the clone's
  `checkpoint/6275650/0/step_3814/`.
- **Block norms, step 3,800:**
  - Block 0's attention update still drowns the embedding: 461 against 4.3
    (110:1). The reference has 1.6 against 10.9.
  - The stream into block 11 shrinks through training but stays 7.6 times the
    reference's: 108,000 at step 100, 12,461 at step 1,000, 4,444 at step 1,900 and
    1,594 at step 3,800, against 210.

**`ame_h12`, end of run, 11-09 00:06 (COMPLETED, exit 0, 4 h 14 min on r14-9). A
tie, on the behind side: final_ce 3.1504 against 3.1250.**

| Steps | 0-499 | 500-999 | 1,000-1,499 | 1,500-1,999 | 2,000-2,499 | 2,500-2,999 | 3,000-3,499 | 3,500-3,814 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Paired gap | +0.051 | +0.075 | +0.047 | +0.026 | +0.016 | +0.012 | +0.014 | +0.015 |

- The gap narrows until step 2,500, then holds at +0.012 to +0.015. The last 500
  steps average +0.0143, inside AME's ±0.022, so by the plan's rule this is a tie.
  But it is behind in every window, from start to end.
- final_ce is +0.0254, and eval loss at step 3,800 is 3.156 against 3.125. Both are
  bf16-quantized, so the paired gap decides.
- **Health:** no NaN, and no spike in either run.
  - Grad-norm max after step 100: 0.70 at step 156 (reference 0.65 at step 130).
  - 3.780 s/step against 3.746 (+0.9%). Peak memory 70.40 GiB, the same as the
    reference.
- **Checkpoint:** in the clone's `checkpoint/6275639/0/step_3814/`, 13 GB: the
  model, the training state, the memory-parallel bank state and `token_counts.pt`.
- **Block norms, step 3,800:**
  - Block 0 still drowns the embedding, though less: 2.4 against an attention
    update of 34, 14:1. The reference has 10.7 against 4.1.
  - The stream into block 11 is now smaller than the reference's: 242 against 315.

**`dense_h16`, mid-run check, 11-09 02:18 (step 3,805 of 7,630). Recovering, far
faster than `dense_h12`, but still +0.25.**

| Steps | 0-499 | 500-999 | 1,000-1,499 | 1,500-1,999 | 2,000-2,499 | 2,500-2,999 | 3,000-3,499 | 3,500-3,808 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Paired gap | +1.46 | +1.17 | +0.67 | +0.46 | +0.36 | +0.30 | +0.27 | +0.25 |

- The gap has shrunk in every window since its +1.69 peak at step 300. The
  shrinking slows: by 0.49 from the 500-999 window to the next, and now by 0.03
  per 500 steps. The last 500 logged steps average +0.252. Eval loss at step 3,800
  is 3.109 against 2.875 (+0.234).
- **`dense_h12` at the same steps (3,500-3,814) was still +1.26.** Its LR was
  twice this run's (n = 8 against 9), a plausible reason, but these two runs can't
  separate LR from model size.
- **Health:** no NaN, and no spike after step 100 in either run.
  - Grad-norm max after step 100: 1.42 at step 488 (reference 0.89 at step 102).
  - 5.960 s/step against 5.964.
- **Block norms:**
  - The stream shrinks toward the reference's. Into block 11 it is 2,411 at step
    500, 1,460 at step 1,000, 560 at step 2,000 and 221 at step 3,800, against 191,
    175, 151 and 120.
  - Block 0 still drowns the embedding at step 3,800: 2.05 against an attention
    update of 73.5, 36:1. The reference has 15.1 against 0.97.

**`ame_h16`, mid-run check, 11-09 03:11 (step 3,815 of 7,630). A loss spike at steps
788-834 that the reference never had; recovered, now +0.039.**

| Steps | 0-99 | 100-499 | 500-699 | 700-787 | 788-834 | 835-999 | 1,000-1,499 | 1,500-1,999 | 2,000-2,499 | 2,500-2,999 | 3,000-3,499 | 3,500-3,815 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Paired gap | +0.052 | −0.106 | +0.048 | +0.046 | +1.786 | +0.542 | +0.176 | +0.119 | +0.083 | +0.060 | +0.047 | +0.039 |

- **The spike:** from 3.84 at step 785, the loss climbs to 9.69 at step 815
  (21:18), near ln V. All 44 spike steps fall in 788-834. The grad norm goes from
  about 0.1 to 3.04 at step 807, and the reference's stays at 0.06-0.13 over the
  same steps.
  - It fell between the step-500 check and this one. The watchers stop only on NaN,
    so nothing flagged it at the time; no action would have followed anyway, since
    the run recovered on its own.
- **Recovery:** 5.75 at step 830 and 4.19 at step 900. The paired gap returns to
  its pre-spike level (+0.046 over steps 700-787) from step 3,231 on. So the spike
  cost the run about 2,400 steps, and it now sits where it was before.
- **What blew up, from the block norms** (logged every 100 steps):

  | | Step 700 | Step 800 | Step 900 | Step 3,800 | Reference, step 3,800 |
  |---|---:|---:|---:|---:|---:|
  | Embedding (block 0's residual) | 1.27 | 1.35 | 1.85 | 2.75 | 8.49 |
  | Block 0 attention update | 278 | 306 | 4,273 | 73 | 4.6 |
  | Stream into block 15 | 1,662 | 1,891 | 7,159 | | |
  | Stream into block 11 | 1,079 | 1,250 | 5,193 | 354 | 438 |

  - Before the spike the stream was already growing, with the embedding drowned
    about 230:1. Over steps 800-900 block 0's attention update grows 14-fold and
    the whole stream about 4-fold.
  - This is the instability the plan watched for: a small embedding (Takase et al.).
    `ame_h12` and both dense runs had none like it.
  - By step 3,800 the stream into block 11 is smaller than the reference's (354
    against 438). Block 0 still drowns the embedding 26:1, against 0.5:1 in the
    reference.
- **Other health:** no NaN, and no spike outside 788-834. 6.792 s/step against
  6.810. Eval loss at step 3,800 is 3.094 against 3.031 (+0.063).

**`dense_h16`, end of run, 11-09 08:56 (COMPLETED, exit 0, 13 h 15 min on r18-13).
Recovers from its stall but levels off 0.21 behind: final_ce 2.9365 against 2.7441.**

| Steps | 0-999 | 1,000-1,999 | 2,000-2,999 | 3,000-3,999 | 4,000-4,999 | 5,000-5,999 | 6,000-6,999 | 7,000-7,629 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Paired gap | +1.31 | +0.57 | +0.33 | +0.26 | +0.23 | +0.22 | +0.21 | +0.21 |

- The gap stops closing at about step 5,000: from there it changes by less than
  0.004 per 1,000 steps. The last 500 steps average +0.213. Eval loss at step
  7,600 is 2.938 against 2.734 (+0.203).
- **Health:** no NaN, and no spike after step 100 in either run.
  - Grad-norm max after step 100: 1.42 at step 488 (reference 0.89 at step 102).
  - 5.966 s/step against 5.964. Peak memory 37.75 GiB, the same as the reference.
- **Checkpoint:** the step-7,629 model and training state, 4.0 GB, in the clone's
  `checkpoint/6275560/0/step_7629/`.
- **Block norms, step 7,600:** the stream into block 11 is 109 against 80. Block 0
  still drowns the embedding: 2.1 against an attention update of 30, 15:1. The
  reference has 12.2 against 0.75.

**`ame_h16`, end of run, 11-09 10:41 (COMPLETED, exit 0, 15 h 02 min on r18-6).
Slightly worse: final_ce 2.9004 against 2.8594.**

| Steps | 0-999 | 1,000-1,999 | 2,000-2,999 | 3,000-3,999 | 4,000-4,999 | 5,000-5,999 | 6,000-6,999 | 7,000-7,629 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Paired gap | +0.150 | +0.148 | +0.072 | +0.042 | +0.030 | +0.028 | +0.030 | +0.032 |

- The gap stops closing at about step 4,000. The last 500 steps average +0.0316,
  outside AME's ±0.022. Eval loss at step 7,600 is 2.891 against 2.859 (+0.031).
- **The spike at steps 788-834 stays the only one:** 44 spike steps, none after
  step 834.
  - One run can't tell whether the spike caused the final deficit. Before it the
    gap was +0.046 (steps 700-787), and `ame_h12`, with no spike, closed from +0.08
    to +0.014.
- **Health:**
  - Grad-norm max after step 100: 3.04 at step 807, in the spike (reference 0.71
    at step 121).
  - 6.793 s/step against 6.810. Peak memory 72.17 GiB, the same as the reference.
- **Checkpoint:** in the clone's `checkpoint/6275518/0/step_7629/`, 23 GB: the model,
  the training state, the memory-parallel bank state and `token_counts.pt`.
- **Block norms, step 7,600:** the stream into block 11 is 174 against 253. Block 0
  still drowns the embedding: 3.2 against an attention update of 23, 7:1. The
  reference has 6.0 against 3.0.

### Results

| Run | final_ce | Reference | Δ final_ce | Paired train-loss gap, last 500 steps | Verdict |
|---|---:|---:|---:|---:|---|
| `ame_h12` | 3.1504 | 3.1250 | +0.025 | +0.014 | **Tie** (inside ±0.022), on the behind side. Behind in every 500-step window; the gap narrows to about +0.013 by step 2,500 and holds there. |
| `ame_h16` | 2.9004 | 2.8594 | +0.041 | +0.032 | **Slightly worse** (outside ±0.022). One loss spike at steps 788-834, up to 9.69; the gap levels off at +0.03 from step 4,000. |
| `dense_h12` | 4.2188 | 2.9824 | +1.236 | +1.271 | **Much worse.** It stalls from the start; the gap peaks at +2.13 over steps 500-999 and is still +1.26 at the end. |
| `dense_h16` | 2.9365 | 2.7441 | +0.192 | +0.213 | **Worse.** It stalls early like `dense_h12` (+1.69 at step 300), recovers far faster, then levels off at +0.21 from step 5,000. |

AME − dense gap: h12 old 0.143, new −1.068 (3.1504 − 4.2188); h16 old 0.115, new
−0.036 (2.9004 − 2.9365).

### Analysis

**Answer: no.** At the references' learning rates, the new init lowers final CE in
none of the four runs. AME is unchanged at h12 (a tie) and slightly worse at h16.
Dense is much worse at h12 and worse at h16. The AME − dense gap turns negative only
because dense loses more, not because AME gains.

- **The embedding is the likely cause.**
  - Xavier gives each token a vector of norm 0.17, against 24 before. That is 140×
    smaller, 18× below pkmf's own embedding (std 0.11; pkmf uses Xavier only for its
    memory tables), and 3× below the std of 0.02 that spiked for Takase et al.
  - In every run, block 0's attention output dwarfs it. At h12 the ratio is 1,800:1
    (dense) and 2,000:1 (AME) by step 100, against 0.3:1 in the references.
  - At the end the ratio is still 7:1 to 111:1, against 0.06:1 to 0.5:1. Every later
    layer sees the current token only as a small part of the stream.
- **Dense stalls; AME doesn't** (h12 block 0 metrics).
  - Dense block 0's attention output becomes nearly the same size for every token,
    as a context average would be: a per-token std/mean of 0.10-0.22, against about
    1.0 in its reference.
  - Its loss sits at 7.6-7.7 over steps 10-47 and at 6.3-6.5 until step 500.
  - AME's output keeps its reference's spread: 0.21-0.32 against 0.28-0.43.
  - **Two candidate reasons**, which these runs can't separate:
    - AME's Q/K come from id-indexed banks, so every layer's attention sees the
      token ids.
    - ALiBi makes attention local from step 0, so block 0 mixes the last few tokens,
      the current one included. Dense RoPE attention starts near-uniform under small
      Q/K weights, so it averages the context.
- **The LR confounds the comparison.**
  - The references' LRs were tuned for the old init. At n = 8, Adam's per-step move
    as a fraction of the init std is:
    - 0.62 for the embedding and the head (0.004 and 0.12 before);
    - 0.39 for the attention `Linear`s (0.12);
    - 0.78 for the FF output (0.25).
  - So the first few steps at peak LR overwrite the init.
  - `dense_h16`, at half the LR (n = 9), recovered far faster than `dense_h12`: +0.25
    against +1.26 over steps 3,500-3,808. The size differs too.
- **The four changes were tested together,** so these runs can't apportion the
  effect. The embedding is the prime suspect: the failure sits where the embedding
  meets block 0.
- **Not run; possible follow-ups:**
  - At dense h12, which fails loudest (about 3.7 h on one GPU): the new init with the
    old embedding; Xavier × √d (Scaled Embed); the new init at n = 9 and 10.
  - A dense ALiBi run under the new init, to test why AME is immune.

### Decisions and issues

- **18:51-18:56, the queue.** Both h16 jobs pend with reason `Priority`. Slurm
  forecasts a start at 03:05 for `ame_h16` and 05:06 for `dense_h16`; `sbatch
  --test-only` for the same shape says 2026-11-11.
  - **Ahead of them:** our account's two 4-GPU Stage 1c/2c jobs (priority 533 and
    531, moved from Helios today) and five 1-GPU jobs from another user (530). Ours
    are at 481. Every other pending GPU job is held by a dependency or a budget
    limit.
  - **Limits are not the cause:** the grant QoS is at 10 of its 128 GPUs. The
    partition's GPUs are mostly taken by the hidden `lem-gpu-short` and
    `lem-gpu-normal` partitions, which share the nodes.
  - **Open question:** four r16 nodes sit fully idle (r16-4, -5, -7, -15) and none of
    these jobs starts there, not even the 4-GPU one that fits. r16-9 failed to power
    up at 16:54 ("ResumeTimeout reached"), so the r16 rack may have a problem the
    node states don't show.
  - **Plan:** forecasts assume every running job uses its full limit, so they run
    pessimistic. The jobs stay queued, and I submit the two h12 runs too.
- **18:58:** `dense_h12` started on r16-1 within a minute of its submit, when a
  stage-2 bank-gates job there finished. The other three stay pending.
- **19:35, decision after `dense_h12`'s step-500 check (see Checks):** nothing is
  cancelled; the plan's rule is to ask first. The three pending runs stay queued, and
  `dense_h12` runs to the end. Maciej is asked to choose:
  - **Run all four as planned:** the measurement stands as "the new init at the old
    LRs".
  - **Revise first:** cancel the three pending runs, change the embedding init (a √d
    multiplier, i.e. Scaled Embed, or back to std ≈ 1), and resubmit.
- **No answer came,** so all four ran to the end as planned. The results measure
  the new init at the references' LRs.
- **11-09 09:33:** the `ame_h16` end watcher died on a DNS failure during its W&B
  login; the job was not affected. Restarted with W&B errors tolerated. The session
  restarted before the job ended, so its end was read at 10:52.
