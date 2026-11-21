# Standard AME and dense models

**Status:** decided by Maciej on 2026-11-21. Nothing is open. The
[implementation plan](#implementation-plan) at the end gathers the approved features on one
branch and merges it into `ame_main`.

## Decisions

Each effect is the change in loss against the reference run, at h12 with one seed unless the
row says otherwise. Negative is better. For AME, a change under 0.022 is a tie
([how the numbers are measured](#how-the-numbers-are-measured)).

### AME

| Change | Decision | Effect on loss | Cost |
|---|---|---|---|
| ALiBi + KV shift + sink (S2-PS), LR 2⁻⁸ | **Keep** (already the default) | Without the shift +0.12, without both +1.14 | — |
| Bank init σ = 1e-4 | **Apply.** The config must set it: the default is still 1.0 | −0.09 vs σ = 1; at h16, σ = 0.01 gave −0.08 | None |
| Banks at 4× the learning rate | **Apply** (Maciej, 2026-11-18) | −0.008 on top of σ = 1e-4, a tie (−0.034 at σ = 1) | None |
| Document mask | **Rejected** (Maciej, 2026-11-21): a small gain, 12% slower, and unmasked packed rows are a valid way to train | −0.016 to −0.018 | +12% s/step |
| ST (STEM) bank gate | **Rejected** (Maciej, 2026-11-21): it breaks AME's precomputed score table | A tie at h12; −0.048 beyond the small bank init at h16 | +3.1–3.8% s/step |
| Bank gates B1, B2, B3 | **Rejected**: no gain | B1 +0.07 to +0.11; B2 and B3 ties | — |
| New init bundle (Xavier embedding and head, pkmf Linears, no embedding weight decay) | **Rejected**: no gain | +0.014 (h16: +0.032) | — |

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
| h16, before | 2.8594 `hjw7znbz` | 2.7441 `eh2sjsh7` | +0.115 |
| h16, AME σ = 0.01 only | 2.7959 `eeaz5uy1` | 2.7441 `eh2sjsh7` | **+0.052** |
| h20, before | 2.5596 `tg4h4g66` | 2.5059 `1x82rzby` | +0.054 |

- At h12, nearly all of AME's gain is the bank init: −0.085 of its −0.094 (`7te9jv9u`, σ =
  1e-4 at ratio 1, scored 3.0400).
- No standard config has run at h16 or h20 yet.

## Standard configs

`zdmv3qci` (AME) and `chne4w91` (dense) are these configs at h12. The keys `bank_lr_ratio`,
`qk_norm` and `shift` exist only on branch `ame_improve` until the plan merges it. Neither
config has a `document_mask` key: it exists only on the rejected branch `doc_mask`.

AME:

```yaml
common:
  attention:
    type: ame
    ame:
      shift: true            # default; the ALiBi ladder and the sink are defaults too
      bank_init_std: 1.0e-4  # the default is 1.0
      bank_gate: none        # default
trainer:
  learning_rate: 8           # LR 2^-8 at h12, h16 and h20
  bank_lr_ratio: 4           # branch ame_improve
```

Dense:

```yaml
common:
  attention:
    type: gqa                # RoPE on by default
    qk_norm: true            # branch ame_improve
    shift: true              # branch ame_improve
trainer:
  learning_rate: 8           # h12; 9 at h16 and h20
```

## Evidence

### How the numbers are measured

- **Sizes:** h12, h16 and h20 are the [H-balanced model](../concepts/h-balanced-model.md) at
  H = 12, 16 and 20, trained on 4B, 8B and 16B tokens. Batch 512 × 2,048 tokens, seed 123,
  learning rate 2⁻ⁿ. h12 and h16 run on one LEM H100, h20 on four.
- **Paired gap:** the mean of (new − reference) training loss over the last 500 steps. Both
  runs see the same batches, so it is finer than `final_ce`.
- **`final_ce`:** the held-out loss at the end of training, rounded to bf16 (±0.008).
- **Tie:** for AME, under 0.022, which is 3 × the same-seed sd of 0.0072
  ([ame_comparison_h12.md](ame_comparison_h12.md)). The bank-gates reports use ±0.019 on a
  common evaluator. Dense varies 6–24× more between same-seed runs, so a single dense
  difference is weaker evidence.

### Baselines (Stages 1–2c)

Best run per model and size, `final_ce` ([stage_runs.md](stage_runs.md)):

| | h12 | h16 | h20 |
|---|---|---|---|
| Dense, RoPE | 2⁻⁸: 2.9824 `x0xlqroj` | 2⁻⁹: 2.7441 `eh2sjsh7` | 2⁻⁹: 2.5059 `1x82rzby` |
| AME, S2-PS | 2⁻⁸: 3.1250 `doi36p8k` | 2⁻⁸: 2.8594 `hjw7znbz` (grid edge) | 2⁻⁸: 2.5596 `tg4h4g66` |

- AME at h12: ALiBi without the shift 3.2480; the shift without ALiBi 3.9346; neither 4.2656
  ([stage_2_report.md](stage_2_report.md)).
- Dense without RoPE: 3.3027 against 3.0244, both at 2⁻⁹ (`b93tal8j`): +0.28.
- Dense h16 at 2⁻⁸ is unstable: 102 spike steps, `final_ce` 3.4346 (`ezzyvpn8`). So 2⁻⁹ is
  its optimum. AME h16 at 2⁻⁷ has never run.
- On 2026-11-06 Maciej set 2⁻⁹ at h12 for stability
  ([stage_1_report.md](stage_1_report.md), decision 9). 2⁻⁸ measured best in both models,
  and every later h12 test ran at 2⁻⁸.

### Bank init σ (AME)

Key `common.attention.ame.bank_init_std`, on `ame_main`, default 1.0. Reports:
[ame_bank_gates_stage2_report.md](ame_bank_gates_stage2_report.md),
[ame_bank_sigma_grid.md](ame_bank_sigma_grid.md).

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

Key `trainer.bank_lr_ratio`, on branch `ame_improve`: the Q/K banks' learning rate as a
multiple of the base rate. Report: [ame_improve_runs_report.md](ame_improve_runs_report.md).

| | Ratio 4 | Ratio 1 | `final_ce` Δ | Paired gap |
|---|---|---|---:|---:|
| h12, σ = 1 | 3.0938 `y5ogvt3k` | 3.1250 `doi36p8k` | −0.031 | −0.034 |
| h16, σ = 1 | 2.8311 `hn1qal9z` | 2.8594 `hjw7znbz` | −0.028 | −0.033 |
| h12, σ = 1e-4 | 3.0312 `zdmv3qci` | 3.0400 `7te9jv9u` | −0.009 | −0.008 |

On top of the small σ it is a tie. Both levers let Adam turn the bank rows faster, and under
QK-norm only a row's direction counts. It costs nothing (the same s/step and memory, no
spikes), so it stays.

### KV shift and QK-norm (dense)

Keys on branch `ame_improve`:
- `common.attention.qk_norm`: ℓ2 per head, with a learned per-head τ initialized at
  √d_head and exempt from weight decay.
- `common.attention.shift`: the KV shift of Xu et al.
  ([arXiv 2411.19574](https://arxiv.org/abs/2411.19574)), after QK-norm and before RoPE. It
  learns one (c1, c2) pair per KV head for the keys and one for the values.

h12, paired gap against `x0xlqroj`, with `final_ce` in brackets
([settle_runs_report.md](settle_runs_report.md), item 5):

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
[doc_mask_report.md](doc_mask_report.md), [settle_runs_report.md](settle_runs_report.md)
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
[ame_bank_gates_report.md](ame_bank_gates_report.md).

- h12: a tie, 3.0191 (`l54pyyqp`) against 3.0358 for no gate at σ = 0.01 (`kf4q4t4s`).
- h16: −0.048 beyond the σ = 0.01 init (`ug8g2i2x` against `eeaz5uy1`, paired).
- Cost: +4.75% FLOPs per token, +3.1–3.8% s/step.

### Other rejected changes

| Change | Result | Code |
|---|---|---|
| B1, the confidence gate | 0.07–0.11 worse than no gate at the same σ | `ame_main`, `bank_gate: confidence` |
| B2 (context) and B3 (backoff) gates | Ties beyond the σ = 0.01 init: B2 −0.0003, B3 +0.0001 at h12; B3 +0.017 at h16 | `ame_main`, `bank_gate: context` and `backoff` |
| New init bundle: Xavier embedding, head and banks; pkmf's √(0.1/fan_in) Linears; no embedding weight decay | AME +0.014 (h16 +0.032), dense +1.271 (h16 +0.213) ([ame_init_exp.md](ame_init_exp.md)) | `ame_initialization` `79108114` |
| Its Xavier embedding alone | Dense +0.999, AME +0.003 ([ame_init_embed_ablation.md](ame_init_embed_ablation.md)) | `ame_initialization` |
| Its undecayed embedding alone | AME +0.012, dense +0.039 | `ame_initialization` |
| Its pkmf Linears and head alone | AME +0.099, dense +0.101 ([ame_init_pkmf_linears.md](ame_init_pkmf_linears.md)) | `ame_initialization` `4457ba17` |

The bundle's only real gain was its small bank σ: Xavier banks start near 0.006. The
current init stays:
- the embedding at N(0, 1), cut at ±2, with weight decay;
- the Linears and the head at 1/√fan_in, cut at ±2σ;
- the banks at σ = 1e-4.

## Implementation plan

**Goal:** one branch, `standard_models`, holds every approved feature and the standard
configs. It passes the tests and reproduces the two h12 standard runs, then merges into
`ame_main`. This section is written for the agent that executes it.

**Done when:**
- `standard_models` holds:
  - the four `ame_improve` commits and the `run_exp.py` fix;
  - the readout fix;
  - six standard configs: AME and dense at h12, h16 and h20.
- The CPU suite passes, and the GPU tests pass on one cluster GPU.
- The h12 reproduction runs match step 0 and tie with `zdmv3qci` and `chne4w91`.
- Maciej approved, and `standard_models` is merged into `ame_main`. Push only if he asks.

**Rules:**
- Never change the session's working directory. Reach the worktree by absolute path and
  `git -C`.
- The W&B key goes on stdin only, never into argv or a file.
- Never delete a branch; remove your own worktree when done.
- LEM jobs exclude node r16-11.
- If a GPU test fails, run it on `ame_main` too before you blame the merge.

### What to gather

| Feature | Config key | First implemented | Tests |
|---|---|---|---|
| AME S2-PS: ALiBi ladder, KV shift, sink | `common.attention.ame.shift`, `sink`, `slopes` | `ame_main`: the defaults in `src/core/ame.py` | On `ame_main` already |
| AME bank init σ | `common.attention.ame.bank_init_std` | `ame_main`, default 1.0 | On `ame_main` already |
| AME banks at 4× | `trainer.bank_lr_ratio` | `ame_improve` `e64a466c`: `main.py`, `src/core/schedulers.py`, `src/core/trainer.py`, `src/definitions.py`, `configs/_trainer/llama.yaml` | `tests/test_bank_lr_ratio.py`, `tests/test_memory_parallel.py` |
| Dense QK-norm | `common.attention.qk_norm` | `ame_improve` `7762da98`: `src/core/attention.py`, `src/core/model.py`, `src/definitions.py`, `configs/_model/h_balanced/base.yaml` | `tests/test_qk_norm.py` |
| Dense KV shift | `common.attention.shift` | `ame_improve` `7345eebc`: the same four files | `tests/test_dense_kv_shift.py` |
| Training length in tokens (comes with the merge) | `trainer.n_tokens_b` | `ame_improve` `62a59d2f`; its `run_exp.py` fix `83ebaa85` is on `ame_improve_runs` | `tests/test_training_tokens.py` |
| The two h12 standard configs | — | `settle_runs_improve` `08b75753`: `configs/ame_improve/ame_h12_ps_bank4_std1e4.yaml` and `dense_h12_qknorm_shift.yaml` | — |

**Where the rejected work stays:**
- The rejected code stays put: the branches `doc_mask` and `ame_initialization`, and the
  bank gates, which are on `ame_main` and off by default.
- The tool for paired gaps, `scripts/paired_gap.py`, is on branch `bank_sigma_grid`.

### Steps

1. **Branch.** `git -C ~/code/nano worktree add ~/code/nano-standard -b standard_models ame_main`.
2. **Merge `ame_improve`.** `git -C ~/code/nano-standard merge --no-ff ame_improve`.
   - A trial merge on 2026-11-21 had one conflict, in `src/core/trainer.py`.
   - Both branches add `Trainer` fields after `mem_parallel_size`. Keep all three:
     `count_token_updates` from `ame_main`, and `n_tokens_b` and `bank_lr_ratio` from
     `ame_improve`.
3. **Take the run_exp fix.** `git -C ~/code/nano-standard cherry-pick 83ebaa85`. Without it,
   `run_exp.py` cannot submit a config that gives its length as `trainer.n_tokens_b`.
4. **Set `bank_lr_ratio`'s default to 1.0.** This is recommended; Maciej may keep 4.0 instead.
   - On `ame_improve` the default is 4.0, set in three places:
     - `DEFAULT_BANK_LR_RATIO` in `src/definitions.py`;
     - the `Trainer` field;
     - `configs/_trainer/llama.yaml`.
   - With 4.0, every older AME config would silently train at 4×, and resuming an old
     checkpoint would fail on the extra param group.
   - With 1.0, every logged run keeps its meaning, and the standard AME configs set 4.0
     explicitly.
   - Update the tests that assume 4.0 (`tests/test_bank_lr_ratio.py`).
   - `bank_init_std` keeps its default of 1.0, for the same reason.
5. **Fix the AME readouts under FSDP2.**
   - **The bug:** under FSDP2, α/β/τ, the horizons and attention entropy are never logged
     (0 of 38 FSDP2 AME runs logged `alpha_2` or `tau`; [stage_2c_report.md](stage_2c_report.md),
     "Readouts"). `AMEAttention.instrumentation` runs `attention_diagnostics(token_ids)`,
     which embeds outside FSDP2's hooks on sharded parameters and raises. The trainer
     (`_log_layer_instrumentation`) then drops the layer's whole readout for the rest of
     the run.
   - **The fix:** in `instrumentation` (`src/core/ame.py`), call `attention_diagnostics`
     only when no parameter of the layer is sharded, that is, has no `full_tensor`
     attribute. The parameter half already gathers sharded tensors with `_full`, a
     collective that every rank runs, so it keeps working.
   - **The test:** add one to `tests/test_instrumentation_is_not_fatal.py` or
     `tests/test_distributed.py`. It runs the readout on an FSDP2-wrapped AME layer and
     gets back `alpha_2/mean` and `tau/mean`.
6. **Write the six standard configs** in a new folder, `configs/standard/`.
   - Start each file from the source below and change only what is listed.
   - Keep `n_steps` as the source has it, so every run pairs with its baseline.
   - In every file: no `document_mask`, `bank_gate` left at its default (none), and a
     `standard` tag.

   | File | Start from | Change |
   |---|---|---|
   | `ame_h12.yaml` | `settle_runs_improve:configs/ame_improve/ame_h12_ps_bank4_std1e4.yaml` (`zdmv3qci`) | Run name, tags, comments |
   | `dense_h12.yaml` | `settle_runs_improve:configs/ame_improve/dense_h12_qknorm_shift.yaml` (`chne4w91`) | Run name, tags, comments |
   | `ame_h16.yaml` | `configs/stage_2b/ame_h16_pe_shift.yaml` | `learning_rate: 8` (the file says 9, but the best AME h16 run is at 2⁻⁸); add `bank_init_std: 1.0e-4` and `bank_lr_ratio: 4.0` |
   | `dense_h16.yaml` | `configs/stage_1b/dense_h16_lr_grid.yaml` | A single `learning_rate: 9` instead of the grid; `/base_schema` in the defaults; add `qk_norm: true` and `shift: true` |
   | `ame_h20.yaml` | `configs/stage_2c/ame_h20_pe_shift.yaml` | `learning_rate: 8` (the file leaves it null); add `bank_init_std: 1.0e-4` and `bank_lr_ratio: 4.0` |
   | `dense_h20.yaml` | `configs/stage_1c/dense_h20_lr_grid.yaml` | A single `learning_rate: 9`; `/base_schema`; add `qk_norm: true` and `shift: true` |
7. **Run the tests.**
   - In the worktree, run the CI gate from `.github/workflows/tests.yml`:
     `pytest -m "not gpu and not slow and not network"`.
   - Then run `pytest -m gpu` on one cluster GPU. LEM's env has no pytest, so put it on
     `PYTHONPATH` from a copy ([doc_mask_report.md](doc_mask_report.md), log 09:16).
8. **Smoke-run all six configs:** 60 steps each (`trainer.n_steps: 60`) on LEM, h20 on 4
   GPUs.
   - These catch config errors and running out of memory. Dense h16 and h20 have never run
     with QK-norm and the shift, which add 6.3 GiB at h12.
   - Check that the h20 AME smoke now logs `alpha_2/mean` and `tau/mean` (step 5).
9. **Reproduce at h12:** run `ame_h12.yaml` and `dense_h12.yaml` from `standard_models`,
   about 4 GPU-h each on LEM.
   - **Submit** as in [settle_runs_report.md](settle_runs_report.md#submission). Check each
     job about 2 minutes after it starts, then only at the end.
   - **Step 0:** its loss must equal the reference's (`zdmv3qci`, `chne4w91`).
   - **Last 500 steps:** run `python scripts/paired_gap.py NEW_ID zdmv3qci`, and the same
     for dense with `chne4w91`. Copy the script from branch `bank_sigma_grid`.
   - **AME passes** under 0.022.
   - **Dense has no measured tie band.** A gap under 0.022 passes. For a larger one, look
     at where it opens: a spike episode on either side explains most of them.
10. **Merge into `ame_main`,** after Maciej approves.
    - Other sessions work in `~/code/nano`. First check that no file the merge touches has
      uncommitted changes there: compare `git -C ~/code/nano status --short` with
      `git -C ~/code/nano diff --name-only ame_main...standard_models`.
    - Then run `git -C ~/code/nano merge --no-ff standard_models`, and run the CPU suite
      again.
    - Remove the worktree (`git -C ~/code/nano worktree remove ~/code/nano-standard`) and
      keep the branch.
11. **Update this file:**
    - mark the status merged;
    - add the reproduction runs and their gaps to the AME − dense gap table;
    - drop "only on branch `ame_improve`" from [Standard configs](#standard-configs).

### Later, each only with Maciej's OK

Each of these costs GPU budget:
- Dense h16 with QK-norm and the shift, about 13 GPU-h per run:
  - at 2⁻⁹, to see whether the gain holds;
  - at 2⁻⁸, to see whether QK-norm makes it stable.
- AME h16 at 2⁻⁷. Its best learning rate is at the edge of the grid.
- The standard configs at h20, about 14 h on 4 GPUs each.
- σ = 1e-4 at other learning rates, to see whether the cliff moves.
