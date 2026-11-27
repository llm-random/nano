# Standard AME and dense models

**Status:** decided by Maciej on 2026-11-21, and merged into `ame_main` on 2026-11-22.
Nothing is open. The [implementation plan](#implementation-plan) at the end gathered the
approved features on branch `standard_models`.

**Implementation (2026-11-22): merged** as `7597f382`. Steps 7–9 pass: the CPU and GPU
tests, 60-step smokes of all six configs, and both h12 reproductions. The first AME
reproduction spiked (+0.077 against the 0.022 band); its rerun on entropy ties `zdmv3qci`
(+0.002). Log: [standard_models_report.md](archive/standard_models_report.md).

## Decisions

Each effect is the change in loss against the reference run, at h12 with one seed unless the
row says otherwise. Negative is better. For AME, a change under 0.022 is a tie
([how the numbers are measured](#how-the-numbers-are-measured)).

### AME

| Change | Decision | Effect on loss | Cost |
|---|---|---|---|
| ALiBi + KV shift + sink (S2-PS), LR 2⁻⁸ | **Keep** (already the default) | Without the shift +0.12, without both +1.14 | — |
| Bank init σ = 1e-4 | **Apply,** and the code default under the QK-norm since 2026-11-25 (it was 1.0, and stays 1.0 without the QK-norm; [review_issues.md](archive/review_issues.md), I17 and N1) | −0.09 vs σ = 1; at h16, σ = 0.01 gave −0.08 | None |
| Banks at 4× the learning rate | **Apply** (Maciej, 2026-11-18), as the default (2026-11-21) | −0.008 on top of σ = 1e-4, a tie (−0.034 at σ = 1) | None |
| Document mask | **Rejected** (Maciej, 2026-11-21): a small gain, 12% slower, and unmasked packed rows are a valid way to train | −0.016 to −0.018 | +12% s/step |
| ST (STEM) bank gate | **Rejected** (Maciej, 2026-11-21): it breaks AME's precomputed score table | A tie at h12; −0.048 beyond the small bank init at h16 | +3.1–3.8% s/step |
| Bank gates B1, B2, B3 | **Rejected**: no gain | B1 +0.07 to +0.11; B2 and B3 ties | — |
| New init bundle (Xavier embedding and head, pkmf Linears, no embedding weight decay) | **Rejected**: no gain | +0.014 (h16: +0.032) | — |
| Feed-forward widened by dmodel/3: `dff = 32·round(26H/3)`, 3328, 4448 and 5536 at h12, h16 and h20 ([AMEm](../concepts/ame-model.md#amem--the-matched-variant)) | **Apply** (Maciej, 2026-11-27), recipe v4: AME then matches dense of the same H in active parameters and training FLOPs per token, so the dense baseline stays as it is. Without it AME has 3.8%, 4.7% and 5.3% fewer active parameters | Not measured. Chinchilla's parameter term predicts about −0.008 at h12 | +4.0%, +4.9% and +5.1% training FLOPs per token |

### Dense

| Change | Decision | Effect on loss | Cost |
|---|---|---|---|
| RoPE; LR 2⁻⁸ at h12, 2⁻⁹ at h16 and h20 | **Keep** (already standard) | Without RoPE +0.28; 2⁻⁸ at h16 is unstable | — |
| KV shift | **Apply** | −0.022 alone, −0.040 on top of QK-norm; no loss spikes | +2.7% s/step, 2.3 GiB |
| QK-norm | **Apply**, for stability (Maciej, 2026-11-20) | A tie: +0.013 alone, −0.006 on top of the shift | +1.8–2.4% s/step, 4 GiB |
| Document mask | **Rejected** (Maciej, 2026-11-21), as for AME | −0.020 to −0.022 | +12% s/step |
| New init bundle | **Rejected**: no gain | +1.27 (the Xavier embedding alone +1.00); h16 +0.21 | — |

### The AME − dense gap

`final_ce`, one run each:

| Runs | AME | Dense | AME − dense |
|---|---|---|---:|
| h12, before | 3.1250 `doi36p8k` | 2.9824 `x0xlqroj` | +0.143 |
| h12, the standard configs | 3.0312 `zdmv3qci` | 2.9561 `chne4w91` | **+0.075** |
| h12, reproduced from `standard_models` (LEM) | 3.1035 `zs1bceoy`, a spike at steps 330–360 | 2.9629 `q2yrtd9t` | +0.141 |
| h12, the AME rerun (entropy; dense from LEM) | 3.0312 `6yx46ng1` | 2.9629 `q2yrtd9t` | +0.068 |
| h16, before | 2.8594 `hjw7znbz` | 2.7441 `eh2sjsh7` | +0.115 |
| h16, AME σ = 0.01 only | 2.7959 `eeaz5uy1` | 2.7441 `eh2sjsh7` | **+0.052** |
| h20, before | 2.5596 `tg4h4g66` | 2.5059 `1x82rzby` | +0.054 |

- At h12, nearly all of AME's gain is the bank init: −0.085 of its −0.094 (`7te9jv9u`, σ =
  1e-4 at ratio 1, scored 3.0400).
- The reproductions' last-500 paired gaps against the references: dense `q2yrtd9t` +0.0075;
  AME `zs1bceoy` +0.0771 (the spike) and its rerun `6yx46ng1` +0.0017. The AME recipe has
  spiked in 1 of its 3 runs.
- No standard config has run at h16 or h20 yet, apart from the 60-step smokes.

## Standard configs

The files are in `configs/standard/` on `ame_main`: AME and dense at h12, h16 and h20.
`zdmv3qci` (AME) and `chne4w91` (dense) are these configs at h12 as recipe v1. On
2026-11-25 they became recipe v2: each part of the model initializes from its own seed, and
the QK-norm keeps q and k in the compute dtype ([review_issues.md](archive/review_issues.md), N6-N8);
v2 runs pair with v1 in data order, steps and schedule, not in init. **Since 2026-11-26 they
are recipe v3:** the held-out set is the end of each of FineWeb's 64 segments (PR #200), the
training rows shift with it, and v3 runs pair with no earlier run, in `final_ce` or in train
batches (`29a04918`). **AME is recipe v4 since 2026-11-27:** its feed-forward is dense's plus
dmodel/3 ([AMEm](../concepts/ame-model.md#amem--the-matched-variant)), so it matches dense of
the same H in active parameters and training FLOPs; dense stays v3. Every version and what it
pairs with:
[comparing-runs.md](../concepts/comparing-runs.md#which-runs-pair). Neither config has a
`document_mask` key: it exists only on the rejected branch `doc_mask`.

AME:

```yaml
common:
  dff: ${eval:'32 * round(26 * ${common.h} / 3)'}  # dense's dff + dmodel/3 (recipe v4)
  attention:
    type: ame
    ame:
      shift: true            # default; the ALiBi ladder and the sink are defaults too
      bank_init_std: 1.0e-4  # also the default under the QK-norm since 2026-11-25
      bank_gate: none        # default
trainer:
  learning_rate: 8           # LR 2^-8 at h12, h16 and h20
  bank_lr_ratio: 4           # also the default
```

Dense:

```yaml
common:
  attention:
    type: gqa                # RoPE on by default
    qk_norm: true            # default: off for gqa
    shift: true              # default: off for gqa
trainer:
  learning_rate: 8           # h12; 9 at h16 and h20
```

## Evidence

### How the numbers are measured

The rules for any comparison, with the recipe versions:
[comparing-runs.md](../concepts/comparing-runs.md). For the numbers here:

- **Sizes:** h12, h16 and h20 are the [H-balanced model](../concepts/h-balanced-model.md) at
  H = 12, 16 and 20, trained on 4B, 8B and 16B tokens. Batch 512 × 2,048 tokens, seed 123,
  learning rate 2⁻ⁿ. h12 and h16 run on one LEM H100, h20 on four.
- **Paired gap:** the mean of (new − reference) training loss over the last 500 steps. Both
  runs see the same batches, so it is finer than `final_ce`.
- **`final_ce`:** the held-out loss at the end of training, rounded to bf16 (±0.008).
- **Tie:** for AME, under 0.022, which is 3 × the same-seed sd of 0.0072
  ([ame_comparison_h12.md](archive/ame_comparison_h12.md)). The bank-gates reports use ±0.019 on a
  common evaluator. Dense varies 6–24× more between same-seed runs, so a single dense
  difference is weaker evidence.

### Baselines (Stages 1–2c)

Best run per model and size, `final_ce` ([stage_runs.md](archive/stage_runs.md)):

| | h12 | h16 | h20 |
|---|---|---|---|
| Dense, RoPE | 2⁻⁸: 2.9824 `x0xlqroj` | 2⁻⁹: 2.7441 `eh2sjsh7` | 2⁻⁹: 2.5059 `1x82rzby` |
| AME, S2-PS | 2⁻⁸: 3.1250 `doi36p8k` | 2⁻⁸: 2.8594 `hjw7znbz` (grid edge) | 2⁻⁸: 2.5596 `tg4h4g66` |

- AME at h12: ALiBi without the shift 3.2480; the shift without ALiBi 3.9346; neither 4.2656
  ([stage_2_report.md](archive/stage_2_report.md)).
- Dense without RoPE: 3.3027 against 3.0244, both at 2⁻⁹ (`b93tal8j`): +0.28.
- Dense h16 at 2⁻⁸ is unstable: 102 spike steps, `final_ce` 3.4346 (`ezzyvpn8`). So 2⁻⁹ is
  its optimum. AME h16 at 2⁻⁷ has never run.
- On 2026-11-06 Maciej set 2⁻⁹ at h12 for stability
  ([stage_1_report.md](archive/stage_1_report.md), decision 9). 2⁻⁸ measured best in both models,
  and every later h12 test ran at 2⁻⁸.

### Bank init σ (AME)

Key `common.attention.ame.bank_init_std`, on `ame_main`; the default is 1e-4 under the QK-norm since
2026-11-25, 1.0 before and without the QK-norm ([review_issues.md](archive/review_issues.md), I17 and
N1). Reports:
[ame_bank_gates_stage2_report.md](archive/ame_bank_gates_stage2_report.md),
[ame_bank_sigma_grid.md](archive/ame_bank_sigma_grid.md).

- **h12:** σ = 0.01 (`kf4q4t4s`) scores 3.0358 against 3.1252 for σ = 1 (`doi36p8k`), on
  the common evaluator: −0.0895.
- **h16:** σ = 0.01 (`eeaz5uy1`) against σ = 1 (`hjw7znbz`): paired −0.080, `final_ce`
  2.7959 against 2.8594.
- **The σ grid:** 14 runs at h12, each paired against σ = 0.01.

  | σ | Result |
  |---|---|
  | 1e-5 to 0.0178 | Ties, −0.011 to +0.009 (1e-4: −0.005) |
  | 0.0316 and up | Worse: +0.070 at 0.0316, +0.080 at 0.1, +0.097 at 1, with loss spikes |
  | 1e-7 and down | Worse: +0.064 at 1e-7, +0.183 at 1e-8, +0.442 at 1e-10 |

  At tiny σ, a token's first gradient scales as 1/‖row‖ and dominates the global norm. The
  clip then shrinks every step, for about 600 steps.
- **Why 1e-4:** it sits in the middle of the plateau, two decades below the upper cliff.
  0.01 is only a quarter-decade below the cliff, which may move with the learning rate
  (untested).

### Banks at 4× the learning rate (AME)

Key `trainer.bank_lr_ratio`, from branch `ame_improve` (on `ame_main` since 2026-11-22): the
Q/K banks' learning rate as a multiple of the base rate. Report:
[ame_improve_runs_report.md](archive/ame_improve_runs_report.md).

| | Ratio 4 | Ratio 1 | `final_ce` Δ | Paired gap |
|---|---|---|---:|---:|
| h12, σ = 1 | 3.0938 `y5ogvt3k` | 3.1250 `doi36p8k` | −0.031 | −0.034 |
| h16, σ = 1 | 2.8311 `hn1qal9z` | 2.8594 `hjw7znbz` | −0.028 | −0.033 |
| h12, σ = 1e-4 | 3.0312 `zdmv3qci` | 3.0400 `7te9jv9u` | −0.009 | −0.008 |

On top of the small σ it is a tie. Both levers let Adam turn the bank rows faster, and under
QK-norm only a row's direction counts. It costs nothing (the same s/step and memory, no
spikes), so it stays, and 4 is the default (Maciej, 2026-11-21). Runs and checkpoints from
before the merge need `bank_lr_ratio: 1` to resume; an exact rerun needs the run's own
code, since the init changed on 2026-11-25 ([review_issues.md](archive/review_issues.md), N6).

### KV shift and QK-norm (dense)

Keys from branch `ame_improve` (on `ame_main` since 2026-11-22):
- `common.attention.qk_norm`: ℓ2 per head, with a learned per-head τ initialized at
  √d_head and exempt from weight decay.
- `common.attention.shift`: the KV shift of Xu et al.
  ([arXiv 2411.19574](https://arxiv.org/abs/2411.19574)), after QK-norm and before RoPE. It
  learns one (c1, c2) pair per KV head for the keys and one for the values.

h12, paired gap against `x0xlqroj`, with `final_ce` in brackets
([settle_runs_report.md](archive/settle_runs_report.md), item 5):

| | No shift | Shift |
|---|---:|---:|
| **No QK-norm** | `x0xlqroj`: 0 (2.9824) | `aak55dd1`: −0.022 (2.9629) |
| **QK-norm** | `8lqp6pk3`: +0.013 (2.9941) | `chne4w91`: −0.027 (2.9561) |

- The shift carries the gain. Neither run with the shift had a loss spike; both runs
  without it had an early spike episode.
- QK-norm is a tie on the loss. It stays for stability, which matters at larger models and
  higher learning rates. h12 at 2⁻⁸ does not test that; dense h16 at 2⁻⁸ was unstable
  without it.
- Cost: the shift +2.7% s/step and 2.3 GiB; QK-norm +1.8% (+2.4% on top of the shift) and
  4.0 GiB; both together +5.3% and 6.3 GiB.

### The document mask (both models)

**Rejected by Maciej on 2026-11-21:** the gain is small, the mask makes every step 12%
slower, and packed rows without a mask, where attention crosses documents, are a valid
way to train.

Key `common.attention.document_mask`, on the local branch `doc_mask` only (feature
`18175ed7`, compiled-eval fix `3974addd`). Attention stays
inside each document of a packed row, and AME's shift stops at document starts. Reports:
[doc_mask_report.md](archive/doc_mask_report.md), [settle_runs_report.md](archive/settle_runs_report.md)
item 4.

| | Masked runs | Unmasked | Paired gap |
|---|---|---|---:|
| AME | `j6wxx6h2` (entropy), `h53srjtk` (LEM) | `doi36p8k` | −0.016, −0.018 |
| Dense | `k3xx6w71` (entropy), `igfkre78` (LEM) | `x0xlqroj` | −0.020, −0.022 |

- The gain is real. The replicates agree within 0.002, and for AME it holds in 8 of 8 pairs
  against four unmasked runs. It leaves the AME − dense gap unchanged: +0.143 without the
  mask, +0.145 with it.
- The cost, on LEM: AME 4.200 against 3.746 s/step, dense 3.681 against 3.272. The mask is
  a flex `score_mod`, so it skips no work.

### The ST (STEM) bank gate (AME)

**Rejected by Maciej on 2026-11-21:** with the gate, a score depends on the hidden state, so
AME loses its precomputed score table. The rule holds for any AME change: a score must stay
a function of token ids and distance.

Key `common.attention.ame.bank_gate: stem`, on `ame_main`, off by default. It is named after
STEM ([Sadhukhan et al., ICLR 2026](https://arxiv.org/abs/2601.10639)) and gates the query
per channel: `q_i = silu(W_g x_i) ⊙ norm(E_Q[t_i])`. Report:
[ame_bank_gates_report.md](archive/ame_bank_gates_report.md).

- h12: a tie, 3.0191 (`l54pyyqp`) against 3.0358 for no gate at σ = 0.01 (`kf4q4t4s`).
- h16: −0.048 beyond the σ = 0.01 init (`ug8g2i2x` against `eeaz5uy1`, paired).
- Cost: +4.75% FLOPs per token, +3.1–3.8% s/step.

### Other rejected changes

| Change | Result | Code |
|---|---|---|
| B1, the confidence gate | 0.07–0.11 worse than no gate at the same σ | `ame_main`, `bank_gate: confidence` |
| B2 (context) and B3 (backoff) gates | Ties beyond the σ = 0.01 init: B2 −0.0003, B3 +0.0001 at h12; B3 +0.017 at h16 | `ame_main`, `bank_gate: context` and `backoff` |
| New init bundle: Xavier embedding, head and banks; pkmf's √(0.1/fan_in) Linears; no embedding weight decay | AME +0.014 (h16 +0.032), dense +1.271 (h16 +0.213) ([ame_init_exp.md](archive/ame_init_exp.md)) | `ame_initialization` `79108114` |
| Its Xavier embedding alone | Dense +0.999, AME +0.003 ([ame_init_embed_ablation.md](archive/ame_init_embed_ablation.md)) | `ame_initialization` |
| Its undecayed embedding alone | AME +0.012, dense +0.039 | `ame_initialization` |
| Its pkmf Linears and head alone | AME +0.099, dense +0.101 ([ame_init_pkmf_linears.md](archive/ame_init_pkmf_linears.md)) | `ame_initialization` `4457ba17` |

The bundle's only real gain was its small bank σ: Xavier banks start near 0.006. The
current init stays:
- the embedding at N(0, 1), cut at ±2, with weight decay;
- the Linears and the head at 1/√fan_in, cut at ±2σ;
- the banks at σ = 1e-4.

## Implementation plan

**Done on 2026-11-22:** merged as `7597f382`. The plan, written for the agent that built
branch `standard_models`, is archived with its log since 2026-11-27:
[standard_models_report.md](archive/standard_models_report.md#appendix-the-implementation-plan).
The runs it left for later are [review_upgrades.md](review_upgrades.md), U9.
