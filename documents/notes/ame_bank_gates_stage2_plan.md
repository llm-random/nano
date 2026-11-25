# AME bank gates, stage 2: B1, B3 and the init ablation (plan)

**Status:** PLAN, 2026-11-08 (Maciej). Started on Maciej's go, `b1open_std001` in: the
four runs were submitted on LEM 2026-11-08 14:32-14:35, from `03317133`. Done 19:20:
trained, scored and analyzed; the init is all of stage 1's gain, and the recommendation is
σ = 0.01 with no gate (replicates proposed, not submitted). The record is
[ame_bank_gates_stage2_report.md](ame_bank_gates_stage2_report.md). Stage 1
([plan](ame_bank_gates_plan.md), [report](ame_bank_gates_report.md)) stays unchanged.

**Why this stage:**
- **B1 and B3 are the only gates that keep AME's inference.** Both fold into the banks at
  export, so scores still depend only on token ids and distance:
  - B1: `E′ = σ(c)·norm(E)`;
  - B3: `E″ = norm(base + E)`.
- **Stage 1 can't say what caused its gain.** Every stage-1 run changed two things at once
  against production: its gate, and the bank init (σ = 0.01 instead of 1).

**Questions:**
1. Does σ = 0.01 alone, with no gate, lower final CE against production?
2. Does B3 add anything beyond the init?
3. Can B1 help if its trust doesn't start at 0.05? Stage 1's B1 kept its trust near 0.05,
   so its scores ran at about 0.3% of their full size.

## What to run

Four new runs, in bold. The other cells already exist.
- **Recipe:** h12, 4B tokens, n = 8 (LR 2⁻⁸), seed 123, on LEM. This is stage 1's
  [base.yaml](../../configs/bank_gates/base.yaml), unchanged.
- **Naming:** run `X` is the file `configs/bank_gates/stage2_X.yaml` and the W&B run
  `bank_gates_stage2_X`.

| σ | No gate | B1, trust starts at 0.05 | B1, trust starts at 0.95 | B3 |
|---|---|---|---|---|
| 1 (production's) | `doi36p8k`: 3.125227 | **`b1_std1`** | — | **`b3_std1`** |
| 0.01 (stage 1's) | **`none_std001`** | stage-1 B1: 3.141958 | **`b1open_std001`** | stage-1 B3: 3.035837 |

Each file is `defaults: [base, _self_]`, sets `bank_gates_arm: stage2_X`, and sets these
options:

| Run | `common.attention.ame` |
|---|---|
| `none_std001` | `bank_init_std: 0.01`, `bank_gate: none` |
| `b1_std1` | `bank_init_std: 1.0`, `bank_gate: confidence` |
| `b3_std1` | `bank_init_std: 1.0`, `bank_gate: backoff` |
| `b1open_std001` | `bank_init_std: 0.01`, `bank_gate: confidence`, `confidence_init: 0.95` |

**One code change, needed only by `b1open_std001`:**
- Make B1's starting trust an option, `confidence_init`. Its default of 0.05 keeps stage
  1's B1 unchanged.
- `_with_trust_columns` in [ame.py](../../src/core/ame.py) uses it, and it is added to
  `_AME_OPTIONS` in [attention.py](../../src/core/attention.py).
- 0.95 mirrors stage 1's 0.05: scores start at 0.90 of their full size instead of 0.0025.

## How to submit

Keep stage 1's worktree and branch: `~/code/nano-bank-gates` on `bank_gates_lem`. Don't
create a new worktree or branch. The rest is stage 1's workflow (stage-1 plan, Where):
- commit the four configs and the `confidence_init` change there;
- run `git -C ~/code/nano-bank-gates rebase ame_main`, then
  `git -C ~/code/nano merge --ff-only bank_gates_lem`;
- submit from `~/code/nano-bank-gates` only when `bank_gates_lem` equals `ame_main`.

Submit all four in one wave, one `run_exp.py` at a time, each with `< /dev/null`:

```
run_exp.py --config-path configs/bank_gates --config-name stage2_<X> +infrastructure.slurm.exclude=r16-11 < /dev/null
```

- **Expected time:** 4:13-4:21 per run, within the 7 h limit.
- **Checks:** look at each job about 2 minutes after it starts and again at step 500:
  - no NaN;
  - s/step within 10% of stage 1's: 3.746 with no gate, 3.771 for B3, 3.880 for B1;
  - B1's trust and B3's base norms moving.
- **Never cancel a job without asking.**

## How to analyze

Score the four new runs with the common evaluator and the bucket analysis
(`bank_gates_lem_eval.sh`), as in stage 1. Then compute:

| Contrast | Formula | Answers |
|---|---|---|
| Init alone | `none_std001` − `doi36p8k` | Q1 |
| Init under B3 | stage-1 B3 − `b3_std1` | Q1 |
| Init under B1 | stage-1 B1 − `b1_std1` | Q1 |
| B3 beyond the init | stage-1 B3 − `none_std001` | Q2 |
| B3 at σ = 1 | `b3_std1` − `doi36p8k` | Q2 |
| B1 started open | `b1open_std001` − `none_std001` | Q3 |
| B1 at σ = 1 | `b1_std1` − `doi36p8k` | Q3 |

A gap inside ±0.019, stage 1's noise floor, is a tie.

**Readings:**
- **`none_std001` resolved below `doi36p8k`:** the init has an effect of its own. Its share
  of B3's gain is (`none_std001` − `doi36p8k`) ÷ (stage-1 B3 − `doi36p8k`).
- **Stage-1 B3 tied with `none_std001`:** B3's gate adds nothing. The recommendation is
  then σ = 0.01 with no gate.
- **`b3_std1` should tie production:** rows 8 long leave the base nothing to do.
- **`b1_std1`:** check its trust and `grad_norm/banks` before reading its CE. At σ = 1 its
  bank gradients may be too small to train the banks at all.

**Recommendation rule:**
- The lowest final CE among the seven cells wins.
- On a tie, the cell with fewer changes from production wins: σ = 1 before σ = 0.01; then
  no gate, then B3, then B1.
- If the choice rests on a tie, propose replicates, but don't submit them.

**Buckets:** read them at u = 101-1000, as in stage 1. The key question: does
`none_std001` reproduce stage 1's rare-token gains (production 3.305 there; ST, B2 and B3
2.94-2.96)?

**Report:** `documents/notes/ame_bank_gates_stage2_report.md`, kept the way stage 1's is.

**Default unless Maciej says otherwise:** `b1open_std001` is in. Without it, this stage
needs no code.
