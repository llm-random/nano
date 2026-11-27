# AME bank gates: four ways to trust a bank row, one h12 run each (plan)

**Archived 2026-11-27 (historical).** Implemented, trained and scored on 2026-11-07/08, then
followed by [stage 2](ame_bank_gates_stage2_plan.md). Every gate was rejected: B1 costs
0.07–0.11, B2 and B3 tie once the bank init is small, and ST, a tie at h12, was rejected on
2026-11-21 because it breaks AME's precomputed score table. **Still the spec of live code:**
`bank_gate` and `configs/bank_gates/` stay on `ame_main`, off by default (Maciej,
2026-11-25), and cite this plan. **Today:**
[standard_model_improvements.md](../standard_model_improvements.md) ("The ST (STEM) bank
gate", "Other rejected changes"); if a gate is revived,
[review_upgrades.md](../review_upgrades.md), U2 and U8.

**Status:**
- **Implemented** in `157c3122` and `9d7ed99d` on `ame_main`.
- **Submitted** 2026-11-07 22:16-22:18: four runs on LEM, jobs 6264723, 6264725,
  6264726, 6264733.
- **Scored** 2026-11-08: ST, B2 and B3 tie for best at the noise floor, and B1 is last.
  Replicates are proposed.
- **Extended** the same evening with an h16 wave of four more runs at 8B tokens
  ([h16 extension](#h16-extension-maciej-2026-11-07)), jobs 6264776-6264779.
- **h16 scored** 2026-11-08: ST is the best gate, resolved against the proxy floor, and
  the ranking agrees with h12's. Replicates are proposed for the ties.
- **Followed** by [stage 2](ame_bank_gates_stage2_plan.md), which separates the σ = 0.01
  init from the gates.

[ame_bank_gates_report.md](ame_bank_gates_report.md) records every job, check, decision
and result. The plan was written 2026-11-07 and revised the same day with Maciej: four
runs, LEM as the cluster, the code on `ame_main`, the work in a separate worktree.

Background: [pkmf_init_comparison.md](../pkmf_init_comparison.md), sections "Memory-bank init
for sparsely updated rows: a proposal" and "Implementation plans for the three bank-init
strategies" (Plan B defines B1, B2 and B3). The STEM gate is added from the four-way
comparison of 2026-11-07, which the mechanism table below restates at h12.

## Question

AME's Q/K bank rows are updated only in steps where their token occurs. In a 4B-token h12
run a token present in every batch gets 3,815 updates and a rare token a handful, yet under
the hard QK-norm both rows act at full strength. Each arm changes how much a row's identity
reaches the attention scores:

| Arm | Mechanism | Idea |
|---|---|---|
| B1 | per-row confidence | each token's row carries a learned trust scalar that starts at 0.05 |
| B2 | context gate (Engram-style, simplified) | the query's hidden state sets one trust scalar per head |
| B3 | backoff (Gemma 3n PLE-style) | every row is mixed with one shared, densely trained vector |
| ST | STEM gate | the query's hidden state scales each channel of the row, as STEM's SwiGLU gate scales its table row |

1. Which mechanism gives the lowest final loss at h12 and 4B tokens?
2. Which one makes rarely updated rows least influential without costing loss?

## Design

Four runs, one per mechanism, all with banks drawn at σ = 0.01 and otherwise identical.

| Run | W&B name | Mechanism | `attention.ame` options |
|---|---|---|---|
| B1 | `bank_gates_b1` | per-row confidence | `bank_init_std: 0.01`, `bank_gate: confidence` |
| B2 | `bank_gates_b2` | context gate | `bank_init_std: 0.01`, `bank_gate: context` |
| B3 | `bank_gates_b3` | shared backoff vector | `bank_init_std: 0.01`, `bank_gate: backoff` |
| ST | `bank_gates_stem` | STEM gate | `bank_init_std: 0.01`, `bank_gate: stem` |
| Production reference, LEM (done) | `s2_ps_h12_lr8.0`, W&B run `doi36p8k` | none; banks at σ = 1 | none; common evaluator 3.125227 |
| AME-B ×3, entropy (done) | `parity_full_ame_b{1,2,3}` (W&B `nano-ame-parity`) | as production, three times | none; 3.131973 ± 0.00719, the source of the noise floor |

- **Every arm uses σ = 0.01.** Plan B specifies B2 and B3 with a small init: B2 "pair it
  with a small init"; B3 "E drawn small", without which the shared vector cannot dominate a
  rare row. B1 and ST work at any σ. Holding σ equal leaves the mechanism as the only
  difference between arms.
- **No control run** (Maciej, 2026-11-07). The arms compare cleanly with each other, since
  only the mechanism differs. Against production each arm differs in two things, the
  mechanism and the σ = 0.01 init, so a gain over production cannot be split between them.

### Order of work

1. Code and tests in a separate worktree, fast-forwarded into `ame_main` (B1, B2, B3, ST and
   the shared changes).
2. Configs and a local smoke run of each arm.
3. The full wave on LEM: four runs of 4B tokens, checked closely over their first 300 steps.
   The evaluator's validation runs alongside.
4. Scoring with the common evaluator, the bucket analysis, and a results doc,
   `ame_bank_gates_report.md`.

There is no separate GPU pre-flight: a short run would show nothing that the full runs' first
300 steps do not, and it would add a queue wait and a startup before them.

Each step gates the next.

## What stays fixed

S2-PS at n = 8, exactly as W&B `doi36p8k` ran it on LEM:
[configs/stage_2/ame_h12_pe_shift.yaml](../../../configs/stage_2/ame_h12_pe_shift.yaml) with
`trainer.learning_rate=8`, which that run passed at submit. The model code has not changed
since: between its commit `f9680709` and today's `ame_main`, the only code change is the gqa
NoPE option (`c4cb1ef6`), which leaves AME untouched.

| | |
|---|---|
| Model | AME h12: 12 layers, 12 heads × 64, dmodel 768, ALiBi ladder, KV shift, hard QK-norm, τ starting at 8, learned sink. 1,103,599,056 parameters, of which 927,203,328 are the 24 banks |
| Data | FineWeb-edu, GPT-NeoX tokenizer (vocab 50,304), seed 123: one token stream for every run |
| Length | 3,815 steps × 1,048,576 tokens = 4.00B tokens |
| Optimizer | AdamW at 2⁻⁸ (n = 8, S2-PS's optimum), warmup 38 steps, cosine to 10%. Weight decay 0.1, except the banks and AME's scalars. Joint gradient clip 1.0 over backbone and banks |
| Execution | micro-batch 32 × 16 accumulation; Tier 2 banks (`mem_parallel_size: 1`: fp32 master, bf16 forward copy); no FSDP2, so AME's readouts are logged; autocast bf16; `compile_blocks_full` |
| Hardware | LEM, one H100 SXM5 94 GB per run, node r16-11 excluded (bad GPU), a 7 h limit: as every Stage 1-2 run |

## The four mechanisms at h12

H = 12 heads, d = 64 per head, dmodel = 768, V = 50,304, 12 layers × 2 banks. `normalize`
is today's `F.normalize` on each head's row
([ame.py:445-448](../../../src/core/ame.py#L445-L448)), unchanged in every arm.

| | B1 confidence | B2 context gate | B3 backoff | ST STEM gate |
|---|---|---|---|---|
| Query | `sigmoid(c_Q[t]) · normalize(E_Q[t])` | `g_i · normalize(E_Q[t])`, `g_i = sigmoid(W_g·x_i + b)` | `normalize((base_Q + E_Q[t]) / √2)` | `silu(W_g·x_i) ⊙ normalize(E_Q[t])` |
| Key | `sigmoid(c_K[t]) · normalize(E_K[t])` | unchanged | `normalize((base_K + E_K[t]) / √2)` | unchanged |
| Shape | one scalar per token, head and bank | one scalar per query position and head | one vector per bank and head, `[12, 64]` | one value per query position and channel (768) |
| Reads | token id | hidden state | token id | hidden state |
| At init | s = 0.05 everywhere: every score × 0.0025, so attention starts as ALiBi + sink alone | `W_g = 0`, `b = 3`: g = 0.953 everywhere | `base = 0`: the same forward as an ungated run (the hard norm hides σ) | `silu(W_g·x)` has mean 0.17 and RMS 0.52: logits start at about half today's spread |
| Extra parameters | 14,487,552 (12 columns × V × 24 banks) | 110,736 | 18,432 | 7,077,888 |
| Extra compute | negligible | a `768 → 12` matmul per layer, 1.6% of a 768² projection | an add | a `768 → 768` matmul per layer, about +4% FLOPs per token |
| Scores depend only on token ids and distance | yes | no, query side | yes | no |
| Precomputable score table | kept: `s_q[a]·s_k[b]·T[a, b]` | kept, × one scalar per query | kept: base folds into the bank | lost: a per-channel weight cannot be pulled out of the dot product |
| Parameter count at startup | 1,118,086,608 | 1,103,709,792 | 1,103,617,488 | 1,110,676,944 |

- `x_i` is the block's pre-norm output: `Residual.forward` already passes `norm(x)` to the
  attention ([model.py:67-72](../../../src/core/model.py#L67-L72)). Neither gate needs an
  RMSNorm of its own, which Plan B's B2 formula wrote out.
- ST's +4% is 42.5 of about 1,050 MFLOP per token, and ST's mean and RMS assume `x` at
  unit RMS. Both are my estimates.

What to expect (analysis from pkmf_init_comparison.md, not results):
- **B1 is the only arm whose trust is earned by the row's own updates.** `c[t]` moves only in
  steps where t occurs, by about one learning rate per update. Reaching s = 0.5 therefore
  takes at least 754 of the token's own updates at the peak LR (2.944 / 2⁻⁸). A token
  present in every step can get there by about step 800; a rare one cannot.
  *Correction (review, 2026-11-25): under the dense AdamW the banks use, one occurrence
  moves an element about 2.5 lr at once and 25 lr in all at step 1,000, so this estimate
  holds only for a token present in every step ([review_upgrades.md](../review_upgrades.md),
  U1).*
- **B2 and ST never see the row,** so they cannot tell a rare row from a frequent one. They
  learn where identity matching misleads. ST against B2 isolates the gate's shape: a
  per-channel unbounded vector against one bounded scalar per head, in the same place
  (after the norm, before τ).
- **B3 turns a rarely updated token into an average token.** Its row becomes about
  `normalize(base)`: full strength, with no token-specific signal.

## Implementation

### Where

- **The code lands on `ame_main`** (Maciej, 2026-11-07); there is no long-lived feature
  branch.
- **The work happens in a separate worktree,** so that other sessions' uncommitted edits in
  `~/code/nano` never reach the code, the tests or the jobs:
  - create it with
    `git -C ~/code/nano worktree add -b bank_gates_lem ~/code/nano-bank-gates ame_main`,
    following the Stage worktrees' naming (`~/code/nano-stage2` on `stage2_lem`);
  - implement, test and commit there. Run Python from the worktree with the main env's
    interpreter, e.g. `cd ~/code/nano-bank-gates && ~/code/nano/.pixi/envs/default/bin/python
    -m pytest …`. The repo is not installed into the env, so this imports the worktree's
    code;
  - the session's working directory stays `~/code/nano`; reach the worktree by absolute path
    and `git -C`.
- **Each finished step goes to `ame_main`:** `git -C ~/code/nano-bank-gates rebase ame_main`,
  then `git -C ~/code/nano merge --ff-only bank_gates_lem`.
  - The fast-forward keeps other sessions' uncommitted edits in place, and refuses, safely,
    if one of them touches a file the commit changes. Then stop and ask; never force it, and
    never stash, revert or commit someone else's edit.
  - Never move `ame_main` with `git update-ref` or a push to `.`: `~/code/nano`'s working
    tree would keep the old files and show the new code as reverted.
- **Jobs are submitted from the worktree, when `bank_gates_lem` and `ame_main` point at the
  same commit,** so every job runs an `ame_main` commit; record it per job.
  - `run_exp.py` needs a named branch; it refuses a detached HEAD.
  - Its `version_code` runs `git add -A` in whatever tree it runs in. The clean worktree adds
    nothing. From `~/code/nano` it would ship every other session's uncommitted files in the
    job, and an interrupted submit would leave a stray commit on `ame_main`.
- **Docs** (this plan, the results doc) are edited and committed in `~/code/nano` on
  `ame_main`, staging only your own hunks; they reach `bank_gates_lem` with the next rebase.

### One option with five values

- `AMEAttention(..., bank_gate="none")`, taking `"none" | "confidence" | "context" |
  "backoff" | "stem"`. Add `bank_gate` to `_AME_OPTIONS`
  ([attention.py:206](../../../src/core/attention.py#L206)). An unknown value raises at
  construction and lists the valid set.
- `"none"` must reproduce today's model bit for bit: the same state dict and the same
  forward.
- One enum rather than four booleans makes "one mechanism per run" structural.

### B1, `confidence`

1. Constructor: build both banks as `nn.Embedding(V, H·d + H)`, width 780. Draw the first
   768 columns with `normal_(0, bank_init_std)` as today
   ([ame.py:280](../../../src/core/ame.py#L280)), and set the last 12 to logit(0.05) = −2.9444.
2. `_mixed_qkv`, after the gather (either path):
   - split each row into `[..., :768]` and `[..., 768:]`;
   - normalize the first part per head, as today;
   - multiply by `sigmoid(c)` per head, computed in fp32 and then cast;
   - then the shift mix and τ, as today. The KV shift then mixes trust-scaled keys.
3. Memory parallelism needs no change:
   - `ShardedBank` takes `embedding_dim` from the module
     ([memory_parallel.py:417](../../../src/core/memory_parallel.py#L417));
   - the fused pair only requires equal widths
     ([memory_parallel.py:290](../../../src/core/memory_parallel.py#L290)), which both banks
     have;
   - at `mem_parallel_size` 4 the width still divides (780 = 4 × 195), though it would not
     at 8.
4. Weight decay: nothing to do. The columns are bank weight, which is already exempt.
5. Instrumentation, parameter-only and cheap: per bank, the mean of `s` over all rows, its
   10th, 50th and 90th percentiles, and the fraction of rows above 0.5.

### B2, `context`

1. Constructor: `self.gate = nn.Linear(768, 12, bias=True)`, with weight zeros and bias 3.
   That gives g = 0.953, the strongest of Highway networks' carry settings: a transform-gate
   bias of −1 to −3 (Srivastava et al. 2015). It is not tuned.
2. `_mixed_qkv`: compute `g = sigmoid(self.gate(x))`, shape `[B, n, H]`, and multiply it into
   q after the norm and before τ. Keys are untouched.
3. Weight decay: add `gate.bias` to `no_weight_decay_parameters`
   ([ame.py:213](../../../src/core/ame.py#L213)), because decay would pull every gate toward
   0.5. The weight decays like any `Linear` weight.

### B3, `backoff`

1. Constructor: `self.q_base` and `self.k_base`, each `nn.Parameter(torch.zeros(12, 64))`.
2. `_mixed_qkv`: before the norm, `q = (q + q_base) / √2`, and the same for k. Under the hard
   norm the `/√2` changes nothing, since the norm is scale-invariant. It stays so the code
   reads as Gemma's formula.
3. Weight decay: add `q_base` and `k_base` to `no_weight_decay_parameters`.
4. FLOP accounting: add both to `non_matmul_parameters`. `model_flops_per_token` counts every
   2-D parameter outside an `nn.Embedding` as a matmul
   ([mfu.py:54-88](../../../src/core/mfu.py#L54-L88)), and these are `[12, 64]`.
5. Instrumentation: `‖base_Q‖` and `‖base_K‖` per head.

### ST, `stem`

1. Constructor: `self.gate = Linear(768, 768)` without bias, initialized with
   `llm_random_weight_init(scale=init_scale)`, the rule our SwiGLU gate uses.
2. `_mixed_qkv`: after the norm and before τ, `q = silu(self.gate(x)).view(B, n, H, d) * q`.
   Keys are untouched.
3. Weight decay: the weight decays, as an FFN gate does. Its FLOPs are counted
   automatically, since it is a 2-D `Linear` weight.

### Shared changes

1. **The diagnostics must not feed `x = 0` to a gate that reads x.**
   - `score_matrix` and `attention_diagnostics`
     ([ame.py:727-797](../../../src/core/ame.py#L727-L797)) call `_mixed_qkv` with
     `x = zeros`. That is exact only while the scores ignore x.
   - Under B2 every query's gate would read as `sigmoid(b)`, whatever the hidden state.
   - Under ST, `silu(0) = 0` would zero every query, so the readout would show the ALiBi
     prior alone. Both errors would be silent.
   - For `context` and `stem`, `attention_diagnostics` skips the entropy and top-k readouts
     and `score_matrix` raises. Their attention statistics come from the offline
     evaluation, which uses real activations.
2. **Token update counts**, from a new trainer option `count_token_updates`.
   - Every optimizer step, take a `torch.bincount` of the input ids and accumulate
     `occurrences` and `steps_seen`. `steps_seen` is the number of steps in which a token
     occurs, which is the number of updates its bank rows get.
     *Correction (review, 2026-11-25): it is the number of steps in which its bank rows
     have a gradient; AdamW's momentum keeps moving a row after that
     ([review_upgrades.md](../review_upgrades.md), U1).*
   - Save `token_counts.pt` next to the final checkpoint.
   - Cost: one bincount over 65,536 ids per micro-batch.
   - Every run writes the file. Identical files across the four runs confirm that they
     shared one data stream.
3. **The banks' share of the gradient.** Log `grad_norm/banks`, the square root of
   `bank_norm_sq`, which `Trainer.clip_gradient` already computes
   ([trainer.py:936](../../../src/core/trainer.py#L936)). Also log the clip factor,
   `min(1, 1.0 / total_norm)`.
4. **The evaluator.** Bring the common evaluator to `ame_main`, so the runs are scored
   exactly as the production references were: `documents/scripts/parity_numerics.py` (its
   `final_ce` step) and `parity_score_runs.sh`, from `parity_audit`. It uses the fp32
   reference path and precise CE on the full held-out set.
   - `parity_audit` also differs from `ame_main` by about 120 lines in `main.py`, `ame.py`
     and `model.py`; bring only what the scoring needs.
   - It must build models with `bank_gate` set.
   - Validate it while the four runs train: re-score `doi36p8k`'s final checkpoint (job
     6256577, under `checkpoint/6256577/` in its LEM cemetery clone
     `s2_ps_h12_lr8.0_2026-11-06_13-52-40`) and match 3.125227. Scoring waits for this.
5. **The bucket analysis**, a new `documents/scripts/bank_gates_eval.py`. It loads a final
   checkpoint and its `token_counts.pt`, runs the held-out set with hooks on every AME layer,
   and reports the per-bucket readings listed under Evaluation.

### Tests

In `tests/test_ame.py`, on CPU unless marked:

- `bank_gate: none` reproduces today's state dict and forward exactly, and an unknown value
  is rejected.
- **B1:**
  - after `main.py`'s build path, not only after the constructor, every token and head
    starts at s = 0.05;
  - scores are exactly s_q·s_k times the ungated model's on the same banks;
  - c receives gradient only for ids in the batch;
  - the fused memory-parallel path equals the plain path at width 780.
- **B2:** g = sigmoid(3) at init; each score is exactly g_i times the ungated model's; keys
  are unchanged; `gate.bias` is not decayed.
- **B3:** at init the forward equals the ungated model's on the same banks; base receives
  gradient; base is neither decayed nor in the FLOP count.
- **ST:** a literal transcription of `silu(W_g·x) ⊙ normalize(E_Q[t])`, and keys are
  unchanged.
- **Invariants that two arms break on purpose:**
  - `test_a_score_depends_only_on_the_token_pair_and_the_distance` and
    `test_the_values_carry_context_but_the_weights_do_not` must still pass for `none`,
    `confidence` and `backoff`;
  - for `context` and `stem` a new test asserts the break (scores change with x);
  - `test_scores_start_at_std_one_with_the_prescribed_init` stays on the default, and each
    arm gets its own init test.
- **The diagnostics guard:** under `context` and `stem`, no entropy readout is computed from
  `x = 0`.
- **Every arm:**
  - the reference and flex backends agree;
  - bf16 gradients are finite;
  - the trainer's instrumentation call succeeds;
  - `gpu`-marked: the compiled flex path trains under `compile_blocks_full`.

### Configs

- `configs/bank_gates/base.yaml` is a copy of
  [configs/stage_2/ame_h12_pe_shift.yaml](../../../configs/stage_2/ame_h12_pe_shift.yaml)
  (S2-PS, already on LEM: `gres: gpu:hopper:1`, Tier 2 banks, no FSDP2, autocast bf16, the
  final checkpoint written into the job's cemetery clone on Lustre). It is copied rather than
  inherited, so that edits to the Stage 2 file cannot change these runs. Changes:
  - `learning_rate: 8`;
  - `scheduler.warmup_steps: 38` written out, as the parity configs do, so it stays put if
    `n_steps` is ever overridden (the default is 1% of `n_steps`);
  - `time: "0-07:00:00"`;
  - W&B project `nano-ame`, name `bank_gates_<arm>`, and a matching job name and tags;
  - `count_token_updates: true`.
- Four arm files of two lines each.
- At submit, pass `+infrastructure.slurm.exclude=r16-11`, as Stage 1-2 did; no config sets
  it ([stage_1_report.md](stage_1_report.md), decision 3).
- Run names carry no learning rate, so no dot reaches `run_exp.py`'s tmux session name
  ([stage_2_report.md](stage_2_report.md), decision 2).
- Checkpoints are 13-20 GB each, so 50-80 GB across the four clones on Lustre.

## Local checks

The tests above, black, and a CPU smoke run of each arm on the H4 local config. Each smoke
run goes to completion with checkpoint saving on, so it exercises the end-of-run path: the
final eval, the checkpoint save and the new `token_counts.pt` write. That path is the one a
4-hour run can still die on after training has finished.

## Full runs

- Submit all four at once from the worktree (Where), with
  `+infrastructure.slurm.exclude=r16-11`.
- LEM's queue: Stage 2's runs started about 50 minutes after submission. Read a pending job's
  reason rather than predicting its start.
- Check each job about 2 minutes after it starts, then over its first 300 steps:
  - the startup parameter counts in the mechanism table, and no NaN;
  - s/step within 10% of `doi36p8k`'s 3.746 (ST: within 15%);
  - `grad_norm` against `doi36p8k`'s first 300 steps, with the new `grad_norm/banks` and
    clip factor beside it:
    - σ = 0.01 makes the bank gradients about 100× larger at init, because a normalized
      row's gradient scales as 1/‖row‖, and a row is 0.08 long per head instead of 8;
    - under Adam a roughly constant clip factor is nearly harmless, but one that swings
      over the first few hundred steps is not;
  - the mechanism readouts move: B1's s, B3's base norms.
- **Stop rule.** If the clip factor stays far below 1 after the warmup, meaning the banks
  dominate the norm, ask Maciej before cancelling. The fix would be a separate bank clip for
  all four runs, a code change in `Trainer.clip_gradient`. A loss at step 300 is not a
  quality signal: production's 220-step losses spread with an sd of 0.13.
- Check again at step 500.
- Time: an AME h12 run takes 4.25-4.45 h on LEM (3.75 s/step plus startup; ST about 4% more),
  so about 4.5 h of wall clock once all four have started.

## Evaluation

**Primary: the final CE on the held-out set, scored with the common evaluator.**

- **Noise floor.** At 4B tokens, AME's run-to-run sd is 0.00585, pooled over four replicated
  groups with 8 degrees of freedom, measured on entropy
  ([ame_efficiency_parity_report.md](ame_efficiency_parity_report.md)). The plan assumes it
  holds on LEM: `doi36p8k`'s 3.125227 falls inside the range of all 14 entropy AME runs
  (3.1245-3.1399). One run against another then resolves 2.306 × 0.00585 × √2 ≈ **0.019**
  (95%).
- **Comparisons**, all on LEM:
  - the arms against each other: the mechanisms, since everything else is shared;
  - each arm − production (`doi36p8k`, 3.125227): the mechanism and the σ = 0.01 init
    together.
- **Ties.** A difference inside its half-width is a tie. If the best arm ties the runner-up,
  the report says so and names the replicates that would separate them: two more runs of
  each, with the same seed, since nondeterminism alone spreads AME runs, as it did AME-B's.
  They are submitted only on Maciej's go.

**Secondary: is a rarely updated row less influential?**

Bucket tokens by `steps_seen` u from `token_counts.pt`: u = 0, 1-10, 11-100, 101-1,000,
1,001-3,814, and 3,815 (every step). For each bucket, on the held-out set, measure:
- the loss at positions whose input token falls in the bucket;
- as queries: the attention entropy and the spread of the identity part of the logits;
- as keys: the attention mass they receive, relative to their share of the visible keys;
- the row length, and each mechanism's own readout: B1's s, B2's g, B3's cos(row, base),
  ST's gate RMS.

An arm meets the requirement if three things hold:
- the rare buckets' influence falls: less mass on rare keys, flatter identity logits, or for
  B1 a small s;
- the overall CE does not rise beyond the noise floor;
- the rare buckets' loss does not rise.

## Risks

- **The joint clip at σ = 0.01** (Full runs, stop rule). It affects all four runs alike, so the arms
  stay comparable with each other even if the small init costs them all against production.
- **Idle rows keep moving.** AdamW steps every row through its decaying momentum, so a rare
  row drifts after its last real update. That holds in every arm and in production. Lazy
  updates are not part of this plan.
- **B1 starts blind to token identity.** Every score starts at 0.0025× its size, so
  attention is ALiBi + sink until frequent tokens earn trust, which takes at least 754
  updates at the peak LR. Its early loss will lag; the question is the final loss.
- **B2's gate learns slowly at first.** The sigmoid's slope at b = 3 is 0.045, a fifth of
  its slope at 0.
- **ST starts at half the logit spread**, and can recover it only through the gate or τ.
- **LEM startup and nodes.** A FineWeb open takes 10-21 minutes when many jobs start
  together, against a median of 1.8, and node r16-11 has a bad GPU (excluded). LEM logs no
  MFU, because `src/core/mfu.py`'s peak table does not match its `NVIDIA H100` name, so
  compare s/step instead.
- **A fast-forward into `ame_main` can be refused** while `~/code/nano` holds someone's
  uncommitted edit in a file the commit changes. That is the safe failure: stop and ask.

## h16 extension (Maciej, 2026-11-07)

Four more runs, one per gate, at h16 on 8B tokens, on LEM. They train alongside the h12 wave
and do not change it; the h12 scoring and report go on as above. From here on they are part
of this plan: the checks, the scoring, the bucket analysis, the report, and "no job from this
plan is still running" all apply to them.

### What is trained

| Run | W&B name | Config | `attention.ame` options |
|---|---|---|---|
| B1 | `bank_gates_h16_b1` | `configs/bank_gates/h16_b1.yaml` | `bank_init_std: 0.01`, `bank_gate: confidence` |
| B2 | `bank_gates_h16_b2` | `configs/bank_gates/h16_b2.yaml` | `bank_init_std: 0.01`, `bank_gate: context` |
| B3 | `bank_gates_h16_b3` | `configs/bank_gates/h16_b3.yaml` | `bank_init_std: 0.01`, `bank_gate: backoff` |
| ST | `bank_gates_h16_stem` | `configs/bank_gates/h16_stem.yaml` | `bank_init_std: 0.01`, `bank_gate: stem` |
| h16 production reference (done) | `s2b_ps_h16_lr8.0`, W&B `hjw7znbz`, job 6260570 | `configs/stage_2b/ame_h16_pe_shift.yaml`, `trainer.learning_rate=8` | none; banks at σ = 1. Trainer `final_ce` 2.8594; scored here by the common evaluator |

### The configs

`configs/bank_gates/h16_base.yaml` is a copy, not an inheritance, of the S2b-PS h16 config
that Stage 2b ran on LEM ([configs/stage_2b/ame_h16_pe_shift.yaml](../../../configs/stage_2b/ame_h16_pe_shift.yaml),
run `hjw7znbz` at n = 8; [stage_2b_report.md](stage_2b_report.md)). Its recipe stays as it
is: one H100, micro-batch 16 × 32 accumulation, Tier 2 banks, no FSDP2, autocast bf16,
`compile_blocks_full`, and the 24 h limit. Only these change:
- `learning_rate: 8`, which `hjw7znbz` passed at submit;
- `n_steps: 7630` (8B tokens at 1,048,576 tokens per step) with `warmup_steps: 76`
  written out, the value the default (1% of `n_steps`) gave the reference;
- `count_token_updates: true`;
- W&B project `nano-ame`; run and job names `bank_gates_h16_<arm>`, with no dots.

Each arm file, `h16_b1`, `h16_b2`, `h16_b3` and `h16_stem`, sets `bank_init_std: 0.01` and
its `bank_gate`, as at h12.

### The mechanisms at h16

H = 16, d = 64, dmodel = 1024, V = 50,304, 16 layers × 2 banks. The reference has
1,986,332,928 parameters, of which 1,648,361,472 are the 32 banks, and 2.121 GFLOP/token.

| | B1 | B2 | B3 | ST |
|---|---:|---:|---:|---:|
| Extra parameters | 25,755,648 (16 columns × V × 32 banks) | 262,400 | 32,768 | 16,777,216 |
| Parameter count at startup | 2,012,088,576 | 1,986,595,328 | 1,986,365,696 | 2,003,110,144 |
| Extra FLOPs per token | none | a `1024 → 16` matmul per layer (+0.07%) | none | a `1024 → 1024` matmul per layer, about +4.7% |

The same formulas give the h12 table above exactly. Composing the configs checks these
counts before submitting.

### Before submitting

- Compose each config as `run_exp.py` does, pass `check_point`, build it on the meta
  device, and record its parameter count and GFLOP/token.
- **Time.** `hjw7znbz` took 15 h 06 min at 6.810 s/step. ST's extra 4.7% FLOPs projects to
  about 16 h, inside the 24 h limit.
- **Memory.** `hjw7znbz` peaked at 72.17 GiB on a 94 GB H100. B1's wider banks add about
  0.5 GiB of Tier 2 state (18 bytes per parameter), and ST's gate about 0.3 GiB of weights
  and optimizer state plus 2-3 GiB of activations at micro-batch 16. So the heaviest arm
  is near 75 GiB.
- If 8B tokens fits neither the time limit nor the memory, stop and ask.

### Submitting and checks

As at h12:
- the configs reach `ame_main` by rebase and `git -C ~/code/nano merge --ff-only
  bank_gates_lem`, and jobs are submitted from the worktree only when `bank_gates_lem`
  equals `ame_main`, with `+infrastructure.slurm.exclude=r16-11`;
- `run_exp.py` runs one at a time, with `< /dev/null`, and is never killed mid-flight.

Each job is checked about 2 minutes after it starts, then at step 300 and at step 500. The
comparison is `hjw7znbz`'s same steps:
- loss and NaN;
- s/step within 10% of 6.810 (ST within 15%);
- `grad_norm`, `grad_norm/banks` and the clip factor;
- the mechanism readouts.

The stop rule is the h12 one: ask Maciej before cancelling anything.

### Evaluation

As at h12, on LEM through `bank_gates_lem_eval.sh`:
- every arm and `hjw7znbz`'s final checkpoint (job 6260570, clone
  `s2b_ps_h16_lr8.0_2026-11-07_03-36-02`) scored with the common evaluator
  (`parity_numerics.py score`, `ame_ref32`);
- the bucket analysis on all five. The reference uses an arm's `token_counts.pt`, after
  checking that its data config (dataset, seed, sequence length, batch size, world size)
  matches the arms'.

The report gives the best h16 gate, each arm against the others and against the h16
reference, the bucket results, and whether the h16 ranking agrees with h12.

**The h12 noise floor does not transfer.** ±0.019 is the 95% half-width of one run against
another at h12 and 4B tokens, from same-seed replicates there. No replicate exists at h16
and 8B: Stage 2b's noise-floor run was cancelled. h16 gaps are therefore reported beside
the h12 floor as a stated proxy, with that caveat, as Stage 2b did. A gap inside it is
treated as a tie, and the report proposes the replicates that would settle it. They are
submitted only on Maciej's go.

## Decisions for Maciej

The defaults below are taken unless you say otherwise.

1. **ST gates the query only, after the norm,** so that ST and B2 differ only in the gate's
   shape. Alternatives:
   - also gate the keys: twice the cost, and keys read context;
   - gate before the norm: logits stay capped at ±τ, but the gate can then only reweight
     channels and cannot mute a query.
2. **B3 uses the shared vector, not Gemma's projection of the main embedding.** The
   projection costs about 14M parameters and a 768² matmul per bank. Under our embedding
   init it also gives a rare token a full-strength, token-specific fallback, since `emb(t)`
   is updated on exactly the steps its bank row is.
3. **B2 starts 95% open** (b = 3, W_g = 0), the strongest Highway setting, not tuned.
4. **One learning rate for all four, n = 8,** S2-PS's optimum. The new parameters (c, the
   gates, base) share it, and there is no per-arm learning-rate check.

Settled by Maciej on 2026-11-07: four runs with no control, LEM as the cluster, the code on
`ame_main`, and the work in a separate worktree (Where).

## Not in this plan

A σ = 0.01 control without a gate (Plan C), Plan A (the soft QK-norm), Gemma's projection
variant of B3, Engram's full gate (which scores x against a key computed from the row), lazy
or selective bank updates, and runs beyond h12 at 4B and h16 at 8B tokens (the
h16 extension).

## Sources

- [pkmf_init_comparison.md](../pkmf_init_comparison.md): the requirement, Plan B (B1-B3) and
  Plan C, and the published-systems table.
- [stage_2_report.md](stage_2_report.md): S2-PS, its learning-rate curve (n = 8 best,
  3.1250 on LEM) and the `run_exp.py` dot issue.
- [stage_1_report.md](stage_1_report.md), decisions 3-4: node r16-11 and the 7 h limit on LEM.
- [ame_efficiency_parity_report.md](ame_efficiency_parity_report.md): AME-B ×3 on entropy,
  the common evaluator and the pooled sd of 0.00585.
- [STEM](https://arxiv.org/abs/2601.10639) (Sadhukhan et al., ICLR 2026): the FFN's up
  projection replaced by a token-indexed table under the SwiGLU gate.
- [Engram](https://arxiv.org/abs/2601.07372), §2.3: the context gate B2 simplifies.
- [Gemma 3n in Google's JAX library](https://github.com/google-deepmind/gemma),
  `gemma/gm/nn/gemma3n/`: the per-layer-embedding mix B3 follows.
- [Srivastava, Greff & Schmidhuber 2015](https://arxiv.org/abs/1505.00387), "Highway
  Networks": a transform-gate bias of −1 to −3 so the network starts biased toward carry,
  the precedent for B2's b = 3.
