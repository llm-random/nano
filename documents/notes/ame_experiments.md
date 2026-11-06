# AME — experimental program

**Status: the live plan since 2026-11-05. Stages 1b and 2b were added on 2026-11-06.**
- Stage 0 is under way on its own branch.
- Stage 1 is under way, and [stage_1_report.md](stage_1_report.md) is its record. Stage 2
  has not started. Their configs were pre-flighted locally on 2026-11-05, in
  [configs/stage_1/](../../configs/stage_1/) and [configs/stage_2/](../../configs/stage_2/).
- **Every Stage 1–2 run is on LEM.** All six configs were switched from entropy on
  2026-11-05, before the first submit, so that every run shares one hardware. The s/step
  figures in this plan are entropy's; the LEM smoke runs set LEM's baseline, which
  [stage_1_report.md](stage_1_report.md) records.
- **Stages 1b and 2b repeat Stages 1 and 2 at h16 and 8B tokens**, also on LEM with 1 GPU
  per run. Their configs are not written yet.
- Stage 3 will be written once Stages 1 and 2 have results.

This plan was rewritten from the ground up on 2026-11-05. The program it replaces is kept
at the end, unchanged, as [the previous plan](#appendix-the-previous-plan-superseded-2026-11-05).
It is there for reference only; nothing in it is scheduled.

**The order of work:**
1. Train the simplest AME against a dense baseline (Stage 1).
2. Add AME's positional encoding and its KV shift, one at a time (Stage 2).
3. Repeat both at h16 and 8B tokens (Stages 1b and 2b). Stage 1b needs nothing from
   Stage 1, so it runs alongside Stages 1 and 2.
4. Implementation and efficiency work (Stage 0) runs alongside all of them, and never in
   front.

## The stages

| Stage | What | Priority | Branch · worktree | Progress goes to |
|---|---|---|---|---|
| **0** | Implementation: reading about other methods and baselines; implementing and improving AME; the MFU of AME and of the baselines | Below Stages 1 and 2; continuous and asynchronous | `ame_optimize` · `~/code/nano-ame` | [ame_optimization_report.md](ame_optimization_report.md), which keeps its name |
| **1** | First models: a dense h12 baseline and the simplest AME h12, 4B tokens each, with a learning-rate grid for both | **Next** | `ame_main` · `~/code/nano` | `stage_1_report.md` |
| **2** | AME with positional encoding, with the shift, and with both; which is best | After Stage 1 | `ame_main` · `~/code/nano` | `stage_2_report.md` |
| **1b** | Stage 1 at h16 and 8B tokens: a dense h16 baseline and the simplest AME h16, with a learning-rate grid for both | Alongside Stages 1 and 2 | `ame_main` · `~/code/nano` | `stage_1b_report.md` |
| **2b** | Stage 2 at h16 and 8B tokens | After Stage 1b | `ame_main` · `~/code/nano` | `stage_2b_report.md` |
| **3** | Not written yet. It is decided from the results of Stages 1 and 2 | — | — | — |

Rules for every stage:
- **No experiment stage waits for Stage 0.** A Stage 0 improvement reaches `ame_main`
  between comparisons, never during one.
- **One commit per comparison.**
  - Every run of one learning-rate grid, or of one 2 × 2 (Stage 2 or 2b), uses the same
    commit and the same recipe.
  - The stage report records the commit.
  - If a Stage 0 change can alter the numerics, not only the speed, re-run the reference
    arm before comparing across it.
- **Each stage reports in its own file** in `documents/notes/`, created when the stage
  starts.
  - The Stage 0 agent writes Stage 0's report,
    [ame_optimization_report.md](ame_optimization_report.md).

## Stage 0 — implementation (asynchronous)

**Scope, as set on 2026-11-05:**
- reading about other methods and baselines;
- implementing and improving AME;
- raising the MFU of AME and of the baselines.

More efficient code is always welcome, but it is not the priority. The Stage 0 agent
owns this stage and its report. [ame_implementation.md](ame_implementation.md) lists the
implementation items found so far, under "Plans and open items" and "Positional encoding,
and what a dense comparison must control".

Two Stage 0 items touch Stages 1 and 2. Neither one blocks them:

| Item | What it affects | Without it |
|---|---|---|
| **AME readouts under FSDP2** (details below) | Only runs that use FSDP2. The Stage 1–2 configs train on 1 GPU without it, so their readouts are logged | Nothing changes for the current configs |
| **Research on positional encoding for AME** (ALiBi vs RoPE, and alternatives) | What "AME with positional encoding" means in Stage 2 | The implemented ALiBi ladder |

**The readout problem:**
- `instrumentation()` hits sharded parameters.
- The trainer then drops it for the rest of the run
  ([src/core/trainer.py:996](../../src/core/trainer.py#L996)).
- So under FSDP2, α₂/β₂, τ, the horizons and the attention entropy are never logged.
- Every `configs/ame/` file runs FSDP2 except `h20_1gpu.yaml`. The Stage 1–2 configs set
  `distributed: null`, so they are not affected.

## Stage 1 — first models

**Question:** with a tuned learning rate for each, how far is the simplest AME from a dense
baseline at h12 and 4B tokens?

### Arms

| Arm | Attention | Positional signal | KV shift | Config on top of h_balanced h12 |
|---|---|---|---|---|
| **S1-D**, dense baseline | `gqa` → `RoPEAttention` | RoPE, base 10000 | none | `common.attention.type: gqa`, the h_balanced default |
| **S1-A0**, simplest AME | `ame`: Q/K from id-indexed banks, QK-norm, learned τ, learned sink | **none**: every ALiBi slope is 0 | **off**: α and β pinned to (1, 0) | `common.attention: {type: ame, ame: {shift: false, slopes: [0.0, …]}}`, with 12 zeros |

S1-A0 can be set from config today, with no code change:
- `slopes` takes one value per head and does not check the sign
  ([src/core/ame.py:319-325](../../src/core/ame.py#L319-L325)). Twelve zeros remove the
  distance bias.
- `shift: false` pins α and β as buffers
  ([src/core/ame.py:303-304](../../src/core/ame.py#L303-L304)).
- The horizon readout τ/m then reads ∞. That is expected; check that the logger accepts it.

What S1-A0 can and cannot do *(analysis, not measured)*:
- Every score is `τ·T[t_i, t_j]`, a function of two token ids.
- The causal mask still lets the residual stream carry position, as in transformers
  without positional encoding (Haviv et al. 2022).
- But no head can attend by position, because AME scores never read the residual stream.
- So S1-A0 is expected to be the weakest member of the AME family. It is Stage 2's
  reference cell, not a candidate.

### Recipe — the same for every run in Stages 1 and 2

| Setting | Value | Source |
|---|---|---|
| Model | h_balanced h12: 12 blocks, 12 heads (6 KV in dense), d_model 768, d_head 64, SwiGLU d_ff 3072, vocab 50,304 | [configs/_model/h_balanced/](../../configs/_model/h_balanced/) |
| Data | FineWeb-Edu with seed 123 in every run except S2-N (124), so the runs see the same data in the same order. The seed also sets the initialisation: `main.py` seeds torch from it | [configs/_dataset/fineweb.yaml](../../configs/_dataset/fineweb.yaml), [main.py:370](../../main.py#L370) |
| Context, batch | 2,048 tokens seen (`sequence_length: 2049`); 512 sequences = 1,048,576 tokens per step | [configs/ame/h12_1gpu.yaml](../../configs/ame/h12_1gpu.yaml) |
| Tokens | **D = 4B, which is 3,815 steps** | 4·10⁹ / 1,048,576 |
| Optimizer, schedule | `llama` trainer: AdamW; weight decay 0.1, which the AME banks are exempt from; cosine schedule; 1% warmup (38 steps); final LR 10% of peak | [configs/_trainer/llama.yaml](../../configs/_trainer/llama.yaml) |
| Eval | Every 100 steps on 10 batches. After training, `final_ce` on the fixed held-out set of 8192 samples × 2048 tokens, which is the same in every run | `llama` trainer and `fineweb` dataset defaults |
| Checkpoints | The final checkpoint only, written to the job's cemetery clone under `checkpoint/<job id>/<array task id>/`. About 15–20 GB per AME run, banks and their optimizer state included; about 2 GB per dense run | `trainer.checkpoint.save.path: checkpoint` |
| Hardware, precision | 1 H100 per run, on LEM (`gres: gpu:hopper:1`), micro-batch 32 × 16 accumulation steps. **FSDP2 is off in every arm, dense included** (`distributed: null`, `autocast_bf16: true`), so AME's readouts are logged and every run shares one precision path. Without FSDP2 the throughput at h12 is unmeasured; at h16 it cost −2.5% MFU and at h20 it gained +4.2% | [configs/ame/h12_1gpu.yaml](../../configs/ame/h12_1gpu.yaml); [ame_optimization_report.md](ame_optimization_report.md) §1.16 for dense |

**Model sizes.** These come from the configs and
[src/core/ame.py:265-288](../../src/core/ame.py#L265-L288) by arithmetic; check them against
the logged parameter count. The bank figure matches the report's 16.7 GB of bank state at
18 B/param ([ame_optimization_report.md:343](ame_optimization_report.md#L343)).

| | S1-D, dense | AME arms |
|---|---:|---:|
| Backbone, non-embedding | 106.2M | 99.1M: no W_Q or W_K; W_V is 12 heads wide |
| Q/K banks | — | 927.2M: 2 banks × 50,304 × 768 × 12 layers |
| Embedding + LM head, untied | 77.3M | 77.3M |
| **Total** | **183.5M** | **1.10B** |

4B tokens is 21.8 tokens per dense parameter. That is about compute-optimal for the
baseline by the Chinchilla rule of roughly 20 (Hoffmann et al. 2022).

What the comparison matches:

| Matched | Not matched |
|---|---|
| tokens | total parameters: AME has about 6×, almost all of it banks |
| data order | FLOPs per token: AME skips the W_Q and W_K matmuls |
| recipe | positional signal: RoPE vs none |

Every result table states this.

### Configs

| Arm | Config | One submit |
|---|---|---|
| S1-D | [configs/stage_1/dense_h12_lr_grid.yaml](../../configs/stage_1/dense_h12_lr_grid.yaml) | 4 runs, one per n ∈ {10, 11, 12, 13}, as one sbatch array |
| S1-A0 | [configs/stage_1/ame_h12_simplest_lr_grid.yaml](../../configs/stage_1/ame_h12_simplest_lr_grid.yaml) | 4 runs, the same way |

- **To submit:**
  `./.pixi/envs/default/bin/python run_exp.py --config-path configs/stage_1 --config-name <file name without .yaml>`.
  - Submit from a clean worktree on a named branch, because `run_exp.py` commits every
    untracked file into its cemetery snapshot.
  - The runs go to the W&B project `nano-ame`, named `s1_d_h12_lr<n>` and
    `s1_a0_h12_lr<n>`.
- **The learning-rate grid is a `^learning_rate` list.** So, by the repo's rule, these
  two files leave `/base_schema` out of their defaults, and `run_exp.py` schema-checks
  every expanded grid point instead.
- **A `^` key cannot be overridden from the command line.** To run fewer points, edit
  the list.

**Pre-flighted locally on 2026-11-05; nothing was submitted.**
- All six Stage 1–2 files compose.
- Every grid point passes `run_exp.py`'s schema check.
- The training recipe is identical across all six files.
- The models build at 183.5M parameters (dense) and 1,103.6M (AME), of which 927.2M are
  banks. That matches the arithmetic above.

**Before the grid:**
1. **Smoke runs**, with `trainer.n_steps=60 trainer.eval_interval=20`:
   - **dense:** the S1-D file with `infrastructure.max_concurrent_jobs=1`, which runs
     4 short runs one at a time;
   - **AME:** `configs/stage_2/ame_h12_simplest_seed2.yaml` (the S1-A0 code path) and
     `configs/stage_2/ame_h12_pe_shift.yaml` (ALiBi and the shift, with the α/β
     readouts), each with `trainer.learning_rate=12`.

   Check that:
   - micro-batch 32 fits without FSDP2 (if not, set `gradient_accumulation_steps: 32`
     in all six files);
   - the s/step is close to the cost table below;
   - the AME readouts appear in W&B (the horizon reads ∞ where the slopes are 0);
   - `final_ce` is logged;
   - the checkpoint is written.

   Delete the smoke checkpoints afterwards. On 1 GPU without FSDP2, these runs also
   cover the paused readiness review's eval item.
2. **There are no mid-run checkpoints.** A run that dies starts again from step 0,
   losing at most about 3.5 h.
3. **On entropy, probe preemption before each submit.** A grid file starts 4 jobs at once
   unless `infrastructure.max_concurrent_jobs` is set. *(Since the switch to LEM on
   2026-11-05, no Stage 1–2 run is submitted to entropy.)*

### S1.1 — the learning-rate grid

| | |
|---|---|
| Grid | `learning_rate: n`, LR = 2⁻ⁿ, n ∈ {10, 11, 12, 13}, for both arms: **8 runs** |
| Why these points | n = 12 is the configs' placeholder, never tuned for AME ([configs/ame/h12_4gpu.yaml:45](../../configs/ame/h12_4gpu.yaml#L45)). Dense models of this size usually train near n ≈ 10–11: GPT-3's 125M model used 6·10⁻⁴ (Brown et al. 2020, Table 2.1) |
| Selection | The lowest final eval loss. If the best n is at an edge of the grid, add the next n beyond it, until the best lies inside the grid |
| Steps | Integer n only, i.e. factors of 2, as for every method in this repo |

**Cost.** These figures come from 30-step throughput benchmarks on entropy H100s, run with
FSDP2 on and eval excluded. Each AME benchmark ran with the shift and ALiBi on. The
smoke runs measure the real speed of these configs, on LEM, where every Stage 1–2 run
goes.

| Arm | s/step, 1 GPU | One run, 1 GPU | s/step, 4 GPUs | One run, 4 GPUs | Jobs |
|---|---:|---:|---:|---:|---|
| S1-A0 | 3.240 | 3.4 h | 0.930 | 1.0 h (3.9 GPU-h) | 32832, 33076 |
| S1-D | 2.879 | 3.1 h | 0.738 | 0.8 h (3.1 GPU-h) | 33119, 33116 |

- The grid costs about 26 GPU-h.
- Run as eight 1-GPU jobs, it is one wave of about 3.5 h, and it fits within entropy's
  8-GPU cap.
- 1-GPU runs are also more efficient: AME's MFU is 34.5% on 1 GPU against 30.0% on 4.

**Stage 1 is done when** both arms have their best learning rate inside the grid, and
`stage_1_report.md` records:
- the chosen n for each arm;
- the final eval loss at every grid point;
- the loss curves;
- the gap between AME and dense;
- τ and attention entropy, if logged;
- the commit, configs and W&B run ids.

## Stage 2 — positional encoding and the shift

**Question:** which AME variant is best at h12 and 4B tokens: with positional encoding,
with the KV shift, or with both? And how much does each one add?

Stage 2 completes a 2 × 2 design, and Stage 1 supplies one of its cells:

| | no shift | KV shift |
|---|---|---|
| **no positional encoding** | S1-A0, from Stage 1 | **S2-S**: `shift: true`, slopes 0 |
| **positional encoding** | **S2-P**: `shift: false`, default slopes | **S2-PS**: `shift: true`, default slopes |

The run order is as set: S2-P, then S2-S, then S2-PS. The three runs do not depend on
each other, so they can share one wave of GPUs.

| Arm | Config |
|---|---|
| S2-P | [configs/stage_2/ame_h12_pe.yaml](../../configs/stage_2/ame_h12_pe.yaml) |
| S2-S | [configs/stage_2/ame_h12_shift.yaml](../../configs/stage_2/ame_h12_shift.yaml) |
| S2-PS | [configs/stage_2/ame_h12_pe_shift.yaml](../../configs/stage_2/ame_h12_pe_shift.yaml) |
| S2-N | [configs/stage_2/ame_h12_simplest_seed2.yaml](../../configs/stage_2/ame_h12_simplest_seed2.yaml) |

- Each file is one run, submitted with
  `run_exp.py --config-path configs/stage_2 --config-name <file name without .yaml>`.
- `learning_rate` is `null` in all four until Stage 1 finds n\*, and `run_exp.py`'s
  schema check refuses a null learning rate. Set n\* in the four files and commit before
  submitting.
- The learning-rate check below uses the winning arm's file with
  `trainer.learning_rate=<n*−1>`, then `<n*+1>`.
- The W&B names read `s2_<arm>_h12_lr<n>.0`, because the schema types the learning rate
  as a float.

| Component | What it is in these runs |
|---|---|
| Positional encoding | The implemented ALiBi: `s_ij = τ·T[t_i, t_j] − m_h·(i − j)`. Each head has one fixed slope, `m = τ_init / horizon`, with horizons from 16 to 4 × the context (8,192 at 2,048) ([src/core/ame.py:150-164](../../src/core/ame.py#L150-L164)). Under QK-norm the content logit is at most τ, so each head's horizon is a hard limit on its reach. If the Stage 0 research recommends another form before Stage 2 starts, that form replaces this one, and `stage_2_report.md` says why |
| KV shift | `K̂ = α₁K + α₂·Shift(K)` on the normalized bank rows, and `V̂ = β₁V + β₂·Shift(V)`. These are four learned scalars per head per layer, initialized α₁ ~ U(0, 1) and α₂ = 1 − α₁ ([ame_implementation.md, "KV shifting in AME"](ame_implementation.md#kv-shifting-in-ame)). The shift is itself a fixed one-step positional signal, so S2-S is not free of position |

For orientation:
- **S2-PS** is the method as previously specified, called A1 in the previous plan.
- **S2-P** is its floor without the shift, called A4 there.

**Learning rate:**
1. All three arms train at S1-A0's best n, written n\*.
2. Then the best arm also trains at n\* ± 1, which is 2 runs.
3. If its best n moves, re-run the other two arms at the new n before naming a winner.

**Noise floor.** S1-A0 trains again at n\* with a second seed, in the same wave; this run
is S2-N. Differences smaller than the gap between S1-A0's two seeds are reported as ties.

**Readouts.** They are logged because every config runs without FSDP2
(`distributed: null`).
- α₂ and β₂ per head and layer over training. If they fall to 0, the model rejected the
  shift.
- τ per head, and the horizons τ/m, which drift as τ is learned.
- Attention entropy per head.

**Analysis**, in `stage_2_report.md`:
- the final eval loss of all four cells;
- the effect of positional encoding, averaged over the shift on and off;
- the effect of the shift, averaged over positional encoding on and off;
- the interaction between the two;
- every cell against S1-D;
- **the best AME variant at h12 and 4B tokens**, which is the input to Stage 3.

**Cost:**
- Wave 1 is S2-P, S2-S, S2-PS and S2-N: four 1-GPU runs, about 14 GPU-h and 3.5 h.
- Wave 2 is the learning-rate check: 2 runs, about 7 GPU-h.

**Known caveats, deliberately not controlled in Stages 1 and 2:**
- Any gap between AME and dense mixes three differences:
  - where Q and K come from;
  - the positional signal: RoPE, against none or ALiBi;
  - the shift, which the dense baseline does not have.
- So a shift arm that beats dense may owe part of the gain to the shift itself, which
  Xu et al. showed helps dense attention too.
- The controls that separate these are in
  [ame_implementation.md](ame_implementation.md#positional-encoding-and-what-a-dense-comparison-must-control).
  Whether to run them is for Stage 3 to decide.

## Stage 1b — Stage 1 at h16 and 8B tokens

**Question:** Stage 1's question one size up. With a tuned learning rate for each, how far
is the simplest AME from a dense baseline at h16 and 8B tokens? And does the gap shrink
from Stage 1's?

Stage 1b needs nothing from Stage 1, because it tunes its own learning rate. It can start
at once and run alongside Stages 1 and 2. Section numbers (§) in Stages 1b and 2b refer to
[ame_optimization_report.md](ame_optimization_report.md).

**Arms.** S1b-D and S1b-A0 are S1-D and S1-A0 on h_balanced h16. S1b-A0 needs 16 zero
slopes, one per head; AME refuses a list of any other length
([src/core/ame.py:319-325](../../src/core/ame.py#L319-L325)).

### What changes from Stage 1

Everything not in this table is Stage 1's recipe: LEM, 1 GPU per run, no FSDP2, autocast
bf16, seed 123, the same eval and held-out set, and the final checkpoint only.

| Setting | Stage 1 | Stage 1b |
|---|---|---|
| Model | h_balanced h12 | **h_balanced h16**: 16 blocks, 16 heads (8 KV in dense), d_model 1,024, d_head 64, SwiGLU d_ff 4,096 |
| Tokens | 4B = 3,815 steps | **8B = 7,630 steps**: 8·10⁹ / 1,048,576, rounded up |
| Warmup, 1% | 38 steps | 76 steps |
| Micro-batch × accumulation | 32 × 16 | **16 × 32, in both arms** |
| Learning-rate grid | n ∈ {10, 11, 12, 13} | **n ∈ {9, 10, 11}** |
| Time limit | 7 h ([stage_1_report.md](stage_1_report.md), decision 4) | 24 h, for runs of about 14–16 h |
| Final checkpoint | 13 GB AME, 2.1 GB dense, as measured | about 23 GB AME and 4 GB dense, scaled by parameter count |

**Why micro-batch 16:**
- AME h16 runs out of memory at micro-batch 32 on an 80 GB H100 (§1.15). At micro-batch 16
  it peaks at 68.35 GiB, about 33 of it weights and optimizer state. Doubling the
  micro-batch doubles the other ~35 GiB, to over 100 GiB in all, more than LEM's 93.6.
- Dense h16 fits micro-batch 32. It stays at 16 so that both arms share one setting, at a
  cost of about 3.6% in dense speed (43.31% against 44.91% MFU on entropy, §1.16).

**Model sizes**, by arithmetic as in Stage 1; check them against the logged count.

| | S1b-D, dense | AME h16 arms |
|---|---:|---:|
| Backbone, non-embedding | 251.7M | 234.9M: no W_Q or W_K; W_V is 16 heads wide |
| Q/K banks | — | 1,648.4M: 2 banks × 50,304 × 1,024 × 16 layers, 29.7 GB of state (§1.12) |
| Embedding + LM head, untied | 103.0M | 103.0M |
| **Total** | **354.7M** | **1.99B** |
| Training FLOPs per token, seq 2048 | 2.222 G | 2.121 G |

- 8B tokens is 22.6 tokens per dense parameter, against 21.8 in Stage 1. So both stages
  sit near the compute-optimal ratio of about 20.
- Dense training compute is 4.1× Stage 1's: 2.03× the FLOPs per token, over twice the
  tokens.
- AME has 5.6× the dense parameter count (6.0× at h12), and does 0.95× its FLOPs per token.

### S1b.1 — the learning-rate grid

| | |
|---|---|
| Grid | n ∈ {9, 10, 11}, LR = 2⁻ⁿ, for both arms: **6 runs** |
| Why these points | Maciej, 2026-11-06: n = 10 is optimal at this model size. The grid brackets it one step each way. At h12, Stage 1 found n = 9 best in both arms, with n = 8 still training, so the low edge may fire here too |
| Selection | As in Stage 1: the lowest `final_ce`, extending past an edge until the best lies inside. Starting the next n early, once the edge leads mid-run, saves a wave of about 15 h ([stage_1_report.md](stage_1_report.md), decisions 6–7) |

**Before the grid**, smoke runs as in Stage 1 (`trainer.n_steps=60
trainer.eval_interval=20`) for S1b-D, S1b-A0 and S2b-PS. S2b-PS is the heaviest arm in
memory: at h12, S2-PS peaked 4.5 GiB above S1-A0. Check that micro-batch 16 fits without
FSDP2, the s/step against the table below, the readouts, `final_ce` and the checkpoint.
Then delete the smoke checkpoints.

**Cost — estimated; h16 has not run on LEM.** These scale entropy's 1-GPU, micro-batch-16
benchmarks (FSDP2 on; §1.12 for AME, §1.16 for dense) by how much slower LEM ran Stage 1's
configs than entropy's h12 benchmarks: ×1.14 dense, ×1.11 S1-A0, ×1.16 S2-PS
([stage_1_report.md](stage_1_report.md)). The smoke runs replace them.

| Arm | Entropy s/step | LEM s/step, estimated | 7,630 steps | Whole run, with evals and startup |
|---|---:|---:|---:|---:|
| S1b-D | 5.44 | 6.2 | 13.1 h | about 14 h |
| S1b-A0 | 5.94 | 6.6 | 14.0 h | about 15 h |
| S2b-PS, for Stage 2b | 5.94 | 6.9 | 14.6 h | about 16 h |

- The grid is about 90 GPU-h: one wave of about 15 h on 6 GPUs. Each extension past an
  edge adds 2 runs and about 30 GPU-h.
- 1-GPU jobs schedule sooner than 4-GPU ones (Maciej, 2026-11-06). At h16 they are also
  more efficient: AME's MFU is 37.96% on 1 GPU against 33.30% on 4 (§1.12).
- **There are no mid-run checkpoints**, as in Stage 1. A run that dies starts again from
  step 0 and loses up to about 16 h.
- Check that the partition allows a 24 h limit before the first submit.

**Configs — not written yet.** `configs/stage_1b/dense_h16_lr_grid.yaml` and
`configs/stage_1b/ame_h16_simplest_lr_grid.yaml` are to be the Stage 1 files with the
changes above: `_model/h_balanced` h16, `n_steps: 7630`, `gradient_accumulation_steps: 32`,
`^learning_rate: [9, 10, 11]`, 16 zero slopes and `time: "1-00:00:00"`. They are named
`s1b_d_h16_lr<n>` and `s1b_a0_h16_lr<n>` and tagged `stage_1b`. Submit them as Stage 1's
are, from a clean worktree and excluding the node Stage 1 found faulty
([stage_1_report.md](stage_1_report.md), decision 3).

**Stage 1b is done when** both arms have their best n inside the grid, and
`stage_1b_report.md` records what Stage 1's report records, with the AME–dense gap set
against Stage 1's.

## Stage 2b — positional encoding and the shift at h16

**Question:** Stage 2's question at h16 and 8B tokens: which AME variant is best, and how
much does each component add? And do the answers match Stage 2's?

Stage 2b is Stage 2 on Stage 1b's model, recipe and learning rate. It starts once Stage 1b
has its n\*, and it does not wait for Stage 2. Everything in Stage 2 holds, with these
changes:

| Stage 2 | Stage 2b |
|---|---|
| S1-A0, the reference cell | S1b-A0, from Stage 1b |
| S2-P, S2-S, S2-PS | **S2b-P, S2b-S, S2b-PS**, with the same attention settings at h16. S2b-S needs 16 zero slopes. The ALiBi ladder spans the same horizons, 16 to 8,192 tokens, over 16 heads instead of 12 ([src/core/ame.py:148-163](../../src/core/ame.py#L148-L163)) |
| S2-N: S1-A0 again at n\* with seed 124 | **S2b-N**: S1b-A0 again at n\* with seed 124 |
| n\*: S1-A0's best n | n\*: S1b-A0's best n |
| Positional encoding: the implemented ALiBi, unless the Stage 0 research replaces it first | Whatever Stage 2 uses, so the two scales compare like with like |

**Learning rate**, as in Stage 2:
1. S2b-P, S2b-S and S2b-PS train at n\*, and S2b-N runs in the same wave.
2. The best arm also trains at n\* ± 1.
3. If its best n moves, the other two arms re-run at the new n before a winner is named.

**Analysis**, in `stage_2b_report.md`: Stage 2's list at h16, ending with **the best AME
variant at h16 and 8B tokens**. Then, against Stage 2:
- whether the four cells rank in the same order at both scales;
- how each cell's gap to dense changes from h12 at 4B tokens to h16 at 8B.

Model size and tokens change together between the two scales, at nearly the same tokens
per dense parameter (21.8 and 22.6). So a difference between the scales is a scaling
result, not the effect of either change alone.

**Cost**, estimated as for Stage 1b. S2b-P and S2b-S fall between S1b-A0 and S2b-PS.
- Wave 1 is S2b-P, S2b-S, S2b-PS and S2b-N: four 1-GPU runs, about 62 GPU-h and 16 h.
- Wave 2 is the learning-rate check: 2 runs, about 31 GPU-h and 16 h.

**Configs — not written yet.** `configs/stage_2b/ame_h16_pe.yaml`, `ame_h16_shift.yaml`,
`ame_h16_pe_shift.yaml` and `ame_h16_simplest_seed2.yaml` are to be the Stage 2 files with
Stage 1b's changes. Their `learning_rate` stays `null` until Stage 1b has n\*. They are
named `s2b_<arm>_h16_lr<n>.0` and tagged `stage_2b`.

## Reporting — every stage

- **Every result row carries:**
  - arm, n, seed, tokens;
  - total, backbone and bank parameters;
  - FLOPs per token;
  - final eval loss;
  - commit, config path and W&B run id.
- **The logged W&B config is the truth**, not the config file's name.
- **Every comparison between AME and dense states the weight-decay difference.** AME's
  banks are exempt from weight decay, while the dense W_Q and W_K are not
  ([archive/ame_review_and_fixes.md](archive/ame_review_and_fixes.md) 2.6).

---

## Appendix: the previous plan (superseded 2026-11-05)

> **Superseded on 2026-11-05 by the plan above.** It is kept for reference, unchanged
> except for two edits: its status block is replaced by this note, and its headings
> are moved down one level. Nothing in it is scheduled.
>
> Its stage names do not mean what they mean in the new plan:
> - its **Stage 0** is a synthetic copying test;
> - its **Stage 1** is a study of a 100–300M-parameter language model against baselines;
> - its **Stage 2** is the decode system.
>
> Code comments and tests that cite this file by section number, such as
> `ame_experiments.md 7` in `src/core/ame.py` and `tests/test_ame.py`, point into this
> appendix.

The empirical companion to the design record and the baseline standard. Both live
**outside this repo**, in `~/code/research/ame/` — `ame.md` and `baselines.md`. Those two
say what the method *is* and what a fair table looks like. This file says **what we are
actually trying to find out, in what order, with what controls, and what result would
make us stop.**

Written to be read before any compute is spent. If you read only two sections, read
**§2 (the claims, as falsifiable statements)** and **§3 (Week 0 — the measurements that
need no model)**.

Every number quoted is an estimate from `ame.md`, carried here so that a measurement can
contradict it.

**Contents**
0. [Compressed method reference](#0-compressed-method-reference)
1. [The shape of the program](#1-the-shape-of-the-program)
2. [The claims under test](#2-the-claims-under-test)
3. [Week 0 — what needs no model](#3-week-0--what-needs-no-model)
4. [Stage 0 — does the mechanism exist](#4-stage-0--does-the-mechanism-exist)
5. [Stage 1 — does it scale as a language model](#5-stage-1--does-it-scale-as-a-language-model)
6. [Stage 2 — does the deployment claim survive](#6-stage-2--does-the-deployment-claim-survive)
7. [Instrumentation: log these from the first run](#7-instrumentation-log-these-from-the-first-run)
8. [Silent-corruption traps](#8-silent-corruption-traps)
9. [Risk register](#9-risk-register)
10. [Decision tree](#10-decision-tree)
11. [Reporting standard](#11-reporting-standard)
12. [Open questions, mapped to stages](#12-open-questions-mapped-to-stages)

---

### 0. Compressed method reference

Enough to make this file standalone. The authoritative version is `ame.md` §0 and §7.

Per layer ℓ, per head h: two banks `E_Q, E_K ∈ R^{|V|×d_head}` indexed by **token id**,
replacing the Q and K projections. Values stay an ordinary projection.

```python
q = E_Q[t]; k = E_K[t]            # gathers, NOT x @ W_Q / x @ W_K
v = (x @ W_V).view(B,n,H,D)       # unchanged from MHA   ("Option B" — the method)
q = l2_normalize(q, -1)           # QK-norm on the BANK ROWS, before the mix
k = l2_normalize(k, -1)
k = a1*k + a2*shift1(k)           # KV shifting: 4 learnable scalars per head
v = b1*v + b2*shift1(v)
s = einsum('bihd,bjhd->bhij', q, k) * tau   # tau [H], init sqrt(D) ≈ 11.3
s = s - m * dist                  # ALiBi, per-head slope — the ONLY positional signal
# ... causal mask, softmax, Σ w·v, W_O — all standard
```

Score form: `s_ij = τ·T[t_i,t_j] − m·(i−j)` with `T[a,b] = Ê_Q[a]·Ê_K[b]`. Scores depend
on **ids and distance only** — that is the whole design, and the source of both the
deployment win and every weakness.

Reference config for all numbers: 1.3B backbone · d_att = d_model = 2048 · L = 24 ·
16×128 heads · |V| = 50k · fp16 · 60 GB/s desktop DDR5 · 20–30 ms/token budget.
Bank parameters: **4.9B / 9.8 GB fp16, ≈ 4× the backbone.**

Settled, not under test here: Option B over Option A · KV shifting on, as learnable
per-head scalars · the ALiBi bias is non-optional · QK-norm on rows · training is
unmodified O(n²) attention. Option A (bank values + gate) appears below only as a
deliberate ablation (§5, E1.3).

---

### 1. The shape of the program

Three stages plus a week of model-free measurement. Each stage exists to kill the method
cheaply if it is going to die.

| | question | cost | kill criterion |
|---|---|---|---|
| **Week 0** | is the deployment claim arithmetically possible on real hardware and a real corpus? | days, no GPU | row-cache hit rate so low that the score path cannot fit any budget (§3) |
| **Stage 0** | does KV shifting give AME in-context copying **at all**? | hours, 1 GPU | pinned `α₂=β₂=0` scores the same as learned — the mechanism is inert |
| **Stage 1** | does it hold up as a language model against matched baselines? | the real expense | gap does not shrink with scale, or rare-type rows stay at init |
| **Stage 2** | does the decode system survive memory bandwidth? | engineering | score path cannot be brought under budget by any of §6's routes |

**Stages 0 and 1 need none of the decode machinery** — no table, no row cache, no
inverted index, no CPU path. Training is plain O(n²) attention with two gathers. Do not
build the decode system first; `ame.md` §0 is emphatic about this and it is the single
most likely way to waste a month.

Week 0 is listed first because it is cheap, needs no model, and several of its outcomes
change what Stage 1 is for. It can run concurrently with Stage 0.

---

### 2. The claims under test

The method rests on five load-bearing claims. Stated as falsifiable predictions, each
with the cheapest experiment that could break it. **This table is the spine of the
program**; everything in §4–§6 is an elaboration of one row.

| # | claim | prediction if true | breaks if | tested by |
|---|---|---|---|---|
| **C1** | KV shifting restores in-context copying to an id-scored attention layer | a 2–4 layer AME solves resampled-bigram induction; a pinned-coefficient twin does not | both arms score alike, or α₂ reliably decays to 0 | **E0.1** (§4) |
| **C2** | contextual values (B) buy something static values (A) cannot | B beats A on sense/topic tasks at matched scale, independent of the shift | A matches B everywhere except copying | **E1.3** (§5) |
| **C3** | bank parameters are nearly free at training and entirely free at inference FLOPs | at matched training FLOPs + tokens, AME beats dense; the gap to a matched-*parameter* dense model is the honest cost | the FLOP-matched win is smaller than the step-time regression it cost | **E1.1** (§5) |
| **C4** | per-token-type capacity is adequate at realistic vocab and data scale | loss stratified by token frequency shows no cliff in the tail; tail rows move from init | tail rows stay at init, or tail loss is flat vs a dense twin | **E1.2** (§5) |
| **C5** | attention can be removed from the accelerator's critical path within a 20–30 ms budget | measured batch-1 decode at 100k lands at 11–18 ms with the row cache in place | the score path cannot be brought under budget (§6) | **E0.3** (§3), **E2.2** (§6) |

A sixth statement is not a claim but a **standing threat to the entire design**:

> **C0 — the softmax may not be earning its cost.** The ALiBi decay factorizes, so the
> only thing forcing a growing cache is the kernel `exp(τ·T)`. If that matrix has low
> numerical rank, AME collapses to `S_i = e^{−m}S_{i−1} + ψ(t_i)v_iᵀ` — RetNet/GLA with
> id-indexed keys, constant state, no V-cache, no top-k, no CPU offload (`ame.md` §5.2).
> The whole of §6 would then be machinery preserving a kernel that does not need
> preserving. **One SVD on a Stage-1 checkpoint answers it** (E1.5). A reviewer asks this
> in the first five minutes; have the number.

Everything else in `ame.md` §8 is downstream of these six.

---

### 3. Week 0 — what needs no model

Six measurements, no training, no checkpoint. Together they decide whether C5 is
arithmetically reachable before a single GPU-hour is spent. Each quotes the estimate from
`ame.md` so that the measurement can contradict it.

**E0.1 — Vectorized `exp` throughput.** A decode step needs `k·H·L` = 2048·16·24 =
**786,432** exponentials. Estimate: scalar `libm` on one core ≈ 16 ms (the entire
budget); AVX-512 across 8 cores ≈ 0.05 ms. *Measure both, and check which math library
you actually linked.* Pass: < 0.5 ms. This is the cheapest possible measurement and it
guards the one cost nobody looks for.

**E0.2 — Random-gather bandwidth.** The 200 MB of top-k value reads are **786k
independent 256-byte gathers**, not a stream. Estimate: 50–70% of peak → 5–7 ms on
60 GB/s desktop DDR5, not the 3.4 ms a stream would give. Measure the real derate on the
target machine, at the real record size, from a ≥10 GB region. Repeat at batch 2, 4, 8 —
B's reads scale **linearly**, and the estimate says batch 8 is 26.8 ms of gathers alone.

**E0.3 — Row-cache hit rate under the real token distribution.** *The highest-leverage
measurement in Week 0*, because the single-digit-ms claim rests entirely on it. Tokenize
a corpus sample with the production tokenizer, take the sequence of query-token ids, and
simulate an LRU over pruned rows at cache sizes 1000 / 3000 / 10000 types. Estimate:
~75% hit at 3000, ~86% at 10k; amortized cost ≈ 3.4 + 0.25·32 ≈ **11 ms**, not 3.4.
A miss costs the full ~2.0 GB of bank reads (≈ 33 ms desktop). Report the hit-rate curve,
and from it the amortized ms/token at each cache size — alongside the storage: 1000 types
at k = 2048 is **3.1 GB**, 3000 is **9.4 GB**, 10k is 31.5 GB, the full vocabulary
**157 GB**.

**E0.4 — `|U|(n)` and distinct-bigram counts on the real tokenizer.** Both dominant cost
terms are driven by these and both are currently Zipf estimates. Measure, as a function
of context length n, (a) distinct token types `|U|` and (b) distinct bigrams, on real
text with the production tokenizer. Estimates to beat: `|U|` ~0.6–1k at n = 2k, ~5–10k at
32k, ~10–20k at 100k; distinct bigrams ~50–80k at n = 100k, growing roughly linearly.
`|U|` sets the score-path cost (`|U|·d_att·L·2B`); the bigram count sets the inverted
index, which under the key shift is **per head-layer and unshareable across heads** —
estimated 12.6M scored bigrams/token at 32k and 38.4M at 100k.

**E0.5 — PCIe and NVMe.** Sustained GPU→RAM bandwidth (estimate ≈ 25 GB/s, about 80% of
the 31.5 GB/s spec; prefill must ship 3.2 GB at 32k context ≈ 130 ms) and random 8 KB
NVMe read latency at queue depth ~384 (estimate 0.5–1 ms for the whole-vocabulary row
cache, which would make the 157 GB option real rather than absurd).

**E0.6 — Score scale at initialization.** One forward pass of an untrained model; no
optimizer. Confirm that per-element σ ≈ 1 on the bank rows gives scores of std ≈ 1, and
that the habitual embedding-table init of σ = 0.02 gives std ≈ 4e−4 — uniform attention
that may never break symmetry. With QK-norm the sensitivity should mostly vanish; verify
that too, because it is the second reason to keep QK-norm.

> **Deliverable:** a one-page table of six measured numbers against six estimates, and a
> revised batch-1 decode budget. If E0.3 comes back far below estimate, say so loudly —
> it moves the score path from "engineering risk" to "design problem", and Stage 1 should
> then be framed as a quality result rather than a deployment one.

---

### 4. Stage 0 — does the mechanism exist

**Scale:** 2–4 layers, vocab ~1k, context 256. Hours on one GPU.

**Purpose:** C1 only. Not a quality result, not a baseline comparison. None of
`baselines.md`'s baselines apply at this scale and none should be run here — if the
mechanism is inert, the Pareto frontier never comes up (`baselines.md` §0).

#### E0 task design

Synthetic in-context copying: repeated random bigrams plus needle retrieval.

> **The trap that makes this experiment worthless if you miss it.** Resample the bigram
> mapping **per sequence**. A model whose scores are a trained function of token-id pairs
> is a machine for memorizing bigrams — that is exactly what `T` is — so with a mapping
> fixed across the dataset *both* arms pass and the experiment measures nothing.
> In-context copying means the mapping is new in every sequence.

Readouts, two not one:
1. **Task metric** — next-token accuracy at the second occurrence of a bigram's first
   element, and exact-match on needle retrieval.
2. **The learned coefficients themselves** — trajectories of α₂, β₂ per head over
   training. If they drift to zero the model is *declining* the mechanism, which is a
   publishable result and a different finding from "the mechanism doesn't help". Keep
   these scalars in fp32.

#### E0.1 — the four-arm ablation

Two arms are not enough. α (key-side mix) and β (value-side mix) are **different
mechanisms, and only one of them is irreplaceable**: a AME head with a steep enough ALiBi
slope *is* a previous-token head, so after one layer `x_j` already contains `t_{j−1}` and
a layer-2 head reading `v_j` gets the β shift for free. What no amount of depth supplies
is α, because the score `T[t_i,t_j]` still cannot read `x_j`. Pinning both together
confounds them.

| arm | α | β | what it isolates |
|---|---|---|---|
| **A1** | free | free | the method as specified |
| **A2** | free | pinned 0 | key-side mix alone — the irreplaceable half |
| **A3** | pinned 0 | free | value-side mix alone — partly reachable without the shift |
| **A4** | pinned 0 | pinned 0 | the fixed-kernel floor |
| **P1** | — | — | **positive control**: vanilla 2-layer transformer, no shift. Must solve the task via the standard previous-token + induction circuit. If it does not, the task or the harness is broken, not AME. |
| **P2** | free | free | **harness control**: *standard* attention, one layer, with KV shifting. Xu et al.'s Theorem 2 says this represents induction exactly. If it fails, fix the harness before reading anything else. |

**Control the ALiBi slope ladder across all arms** — hold it identical, or the β result
is contaminated by whether a steep-slope head happened to be available. If α₂ collapses
to zero in A1/A2, run the disambiguating 2×2: {shift on, off} × {steep heads present,
absent}. A collapse that disappears when steep heads are removed means the model was
getting the shift elsewhere; a collapse that persists means it genuinely declined.

**Kill criterion:** A4 scores the same as A1. The mechanism is inert in AME and the
fixed-kernel ceiling is real — go to §10's fallback branch.

**What a pass buys:** C1 holds, and AME has an answer to the fixed-kernel ceiling. It
does *not* buy content-addressed retrieval (see §9, R1) — it buys the induction/copy
primitive, which is narrower.

#### E0.2 — key-window width (cheap, do it here)

p-tap key mix, p = 1, 2, 3. Costs nothing to store, so run it — but **expect nothing**.
A p-tap mix is an *additive* mix of p copies of one bigram table, not a p-gram table: the
logit carries no joint term, only a separable rank-1 product in weight space. An earlier
draft of `ame.md` called this a trigram table and built a hypothesis on it; that
motivation does not survive, so Xu et al.'s negative result on wider windows is now the
**expected** outcome rather than one that might not transfer. The interesting version of
this question — a genuinely *multiplicative* mix that can score `(t_j, t_{j−1})` jointly
— is not cheap and is out of scope here.

---

### 5. Stage 1 — does it scale as a language model

**Scale:** 100–300M backbone, 10–30B tokens. The real expense. Baselines per
`baselines.md`; reporting standard per §11 below.

#### E1.1 — the headline comparison (C3)

Matched on **training FLOPs and tokens**, with one hostile matched-**total-parameter**
row (a ~4× larger dense model, or a PKM/Engram model with a comparable bank). Both are
necessary: the claim under test is that bank parameters are nearly free at training and
free at inference FLOPs, and the matched-parameter row is what a reviewer will ask for.
**If AME wins at matched FLOPs and loses at matched parameters, say so — that is a
result**, and it is the PKM framing.

Report FLOPs *and* seconds. AME is ~15% cheaper per token in FLOPs than its dense twin
(deleting W_Q/W_K removes 0.2B params ≈ 0.4 of ~2.6 GFLOPs in a 1.3B forward) but the
backward scatters gradients into 4.9B bank parameters. **Expect fewer FLOPs and worse
wall-clock**; a step-time regression needs stating out loud, not hiding inside a
matched-FLOP table.

#### E1.2 — the Zipf tail (C4) — *the most likely place this method fails scientifically*

4.9B bank parameters over 50k types is ~**98,304 parameters per token type**, spread over
**768 rows** of width d_head (a q-row and a k-row in each of 384 head-layers). A type seen
10³ times in training gets ~100 parameters per occurrence; the tail is far worse and those
rows will be noise.

Three measurements, all from a single Stage-1 run:
1. **Loss stratified by training-frequency decile of the target token.** Compare against
   the dense twin's same stratification. A cliff in the tail that the dense model does not
   have is C4 failing.
2. **Row movement from init**, bucketed by frequency: `‖E[w] − E_init[w]‖ / ‖E_init[w]‖`.
   Rare-type rows that have not moved are dead capacity.
3. **Bank utilization**: fraction of rows whose norm moved beyond a threshold.

Interventions to test if the tail bites (each is also a scaling-law question in its own
right): tie or cluster rare types · factorize rows to rank r < d_head · share banks across
layers · tie a bank to the input embedding · give banks their own d_bank < d_head ·
regularize rows toward a frequency-dependent prior. **The joint scaling law — quality as a
function of bank parameters and tokens-seen-per-type — is the cleanest scientific
contribution in the project** and the one closest to existing work on PKM scaling.

#### E1.3 — Option B vs Option A (C2)

At matched scale, on tasks needing **in-context sense** rather than copying:
word-sense disambiguation, topic/register tracking, "how was this token being used
earlier". Not copying tasks — those are E0.1's job, and A cannot do them by construction.

Two scoping notes that make this experiment interpretable:
- **B's advantage is exactly zero at layer 1.** `v_j = W_V·E[t_j]` is a pure function of
  `t_j`, so layer 1 of B *is* Option A with `E_V := W_V·E_emb`. The advantage grows only
  with depth; a 2-layer probe is the minimum where the claim is even true.
- **Evaluate A without the key shift**, or re-derive its cost columns. Under a key shift
  A's summands become distinct *bigrams* rather than types, its saturation disappears, and
  its decayed counts must be kept per bigram — O(n) state, which removes the entire reason
  A was attractive.

#### E1.4 — length generalization

You will train at 2k–8k and deploy at 100k. ALiBi extrapolates, but the decayed-count
statistic (A) and top-k selection (B) are long-context machinery that **never ran during
training**. Validate at ~10× the training length before trusting any deployment number.
Specifically check that the stretched slope ladder (2⁻¹ … ≈2⁻¹⁴, horizons ~20 to ~185k
tokens) behaves as designed — the standard ALiBi ladder for H = 16 (2⁻⁰·⁵ … 2⁻⁸) puts the
longest-range head at ~3k tokens, which would cap the method far below its own pitch.

Watch for the **attention-sink eviction** failure: AME will likely learn a sink (an `E_K`
row scoring high against everything, almost certainly BOS), and `−m·(i−1)` kills it at long
context. That is precisely StreamingLLM's documented failure mode, on a method that cites
StreamingLLM as the floor it must beat. A position-independent learned sink key is the
cheap insurance — the same key the position-1 note in `ame.md` §7 asks for.

#### E1.5 — the rank of `exp(τ·T)` (C0) — *run this the day the first checkpoint lands*

One SVD on a sampled submatrix of `exp(τ·T)` per head-layer. No new training run.
Report the numerical rank spectrum, per layer and per head.

Why it matters: `T` is rank-d_head by construction, but elementwise exponentiation raises
rank combinatorially (`exp(τT) = Σ_p (τT)^{∘p}/p!`, and the p-th Hadamard power has rank up
to `C(d_head+p−1, p)`), so at τ ≈ 11 the effective rank may be too large to help. But the
break-even is generous: a rank-r recurrent state costs `r·d_head·H·L·2B` ≈ **98 MB/token at
r = 1000** (~1.6 ms desktop), ~0.8 GB at r = 8000, against the ~2.0 GB/token the exact path
spends on scores alone.

If the rank is a few thousand, **AME has an exact recurrent form with constant state** and
the project should pivot (§10). AME is unusually easy to linearize post hoc, because its
key set is finite and enumerable — `exp(τ·T)` is one fixed `|V|×|V|` matrix per head-layer,
an offline matrix problem rather than the open-ended function approximation that makes
LoLCATs-style linearization hard.

---

### 6. Stage 2 — does the deployment claim survive

Built **against a finished Stage-1 checkpoint**, not before. This is where the table, row
cache, inverted index and CPU path get written.

#### E2.1 — the score path, which is the binding cost

> The cost model in an earlier version of `ame.md` priced only the value reads. Obtaining
> the row `T[t_i,·]` costs `|U|·d_att·L·2B` ≈ **2.0 GB/token at |U| = 20k ≈ 33 ms on
> desktop DDR5** — ten times the 200 MB of top-k value reads, and over the whole
> 20–30 ms budget by itself. It is a matvec at 1 FLOP/byte, so bandwidth-bound; more cores
> do not help. **The same cost falls on both variants** — it is a score-path cost, not a
> value-path one.

Three routes, none measured. Measure all three and report the Pareto among them:
1. **ANN/IVF over the key bank.** "Top-k of a 50k-row table under a query vector" is a
   textbook nearest-neighbour problem, the bank is static, and the index builds once
   offline. Estimate at 256 centroids, nprobe 4: ~266 KB per head-layer ≈ **102 MB/token
   ≈ 1.7 ms** — 20× cheaper, at the price of approximate selection. Report recall@k
   against exact, on the trained bank geometry (an untrained bank tells you nothing about
   recall).
2. **LRU pruned-row cache** keyed by query-token id. A hit reads `k·4 B = 8 KB` per
   head-layer (~3 MB/token, free); a miss costs the full ~2.0 GB. Storage from E0.3.
3. **Candidate-set capping**: always include a local window, cap `|U|` by considering only
   types seen in the last W tokens. Measure the quality cost, not just the speed.

Also here: per-head vs per-layer tables, and **quantization of rows** — entries pass
through `exp`, so sensitivity is unknown and must be measured, not assumed.

#### E2.2 — end-to-end decode latency (C5)

ms/token at batch 1, on named hardware, at a stated context length, reported next to bytes
of per-sequence state. Estimate to beat: **~11–18 ms desktop batch-1 at 100k** — inside the
20–30 ms budget, but not comfortably.

Four things that will show up in the measurement and not in any FLOP model:
- **Streaming softmax needs a running max.** The CPU accumulates raw `e^{s_ij}` into
  (N, Z) and the GPU merges a self-term. Unnormalized exponentials overflow: carry a
  running max per head, ship **(N, Z, max)**, rescale on merge as FlashAttention does.
  Not optional at fp16/fp32, and the most likely first bug.
- **The per-layer handoff defeats CUDA-graph decode capture.** `(N, Z, max)` ships once per
  layer → **24 host↔device synchronizations per sequence per token**, 0.12–0.5 ms of pure
  overhead at 5–20 µs each, *and* the loss of graph capture, which is standard at this size
  and worth 2–5×. Budget for losing it, or batch the handoff across layers — which costs
  the staleness that §4.3 of `ame.md` otherwise guarantees is zero.
- **The pipeline deadline is thin.** "Async" is a *data-dependency* claim, not a timing
  one: `t_i` is known only when the GPU finishes step i−1. The real structure is a software
  pipeline over layers, `CPU_ℓ ≲ GPU_{ℓ−1}` ≈ 0.2 ms against ≈ 0.14 ms of CPU work. **A
  single row-cache miss erases that margin.** What is unconditionally true is that no
  quantity the CPU needs comes from the current step's hidden states — zero staleness,
  always exact values.
- **RAM capacity caps the batch story harder than CPU throughput does.** V-caches do not
  share: 8 sequences at 100k context need **~78 GB** of RAM. Long context and many
  sequences are alternatives on consumer hardware, not a combination.

#### E2.3 — the dense→sparse deployment shift

Training uses the full causal softmax; decode uses top-k. **That distribution shift is
exactly what NSA and DSA avoid by training *with* their sparsity.** Measure it explicitly —
quality at k ∈ {256, 512, 1024, 2048, dense}, and the attention mass dropped — or adopt
sparse training. Do not assume it is free because the indexer is exact. Also decide and
report: k per head or per layer (the 200 MB figure assumes per head), and what score
pruned-out pairs receive, −∞ or 0, which changes the softmax denominator between training
and decode.

#### E2.4 — the bigram index under the key shift

Top-k by token type stops working once the score depends on `(t_{j−1}, t_j)`. Index
distinct bigrams instead; it **cannot be shared across heads**, because α and the slope m
are both per-head — 384 independent selections per decode step. Use standard IR machinery
rather than a scan: per-bigram postings in recency order plus a **WAND/MaxScore** merge,
`O(k log·)` after an `O(n)` heap seed. Note that this structure *grows with context* where
the unigram version saturated, so measure at 32k and 100k, not just at 4k.

One interaction to get right: if the cached row is pruned to the top-k columns of
`T[t_i,·]`, then `T[t_i,t_{j−1}]` may be absent, because a position can enter the true
top-k through *either* slot. **The candidate set must be the union over both slots.** The
degenerate settings `α = (1,0)` and `α = (0,1)` are both safe against a unigram-pruned row;
only the mixture breaks.

---

### 7. Instrumentation: log these from the first run

Cheap to add at the start, impossible to recover afterwards.

**Mechanism diagnostics**
- `α₁, α₂, β₁, β₂` per head per layer, every N steps. The headline readout of E0.1, and the
  thing that tells you whether the model accepted the mechanism at Stage 1 scale too.
- Learned temperature `τ` per head; ALiBi slopes if the learnable-slope ablation is on.
- **Effective horizon per head**, `τ/m`: with QK-norm the scores live in ±τ ≈ ±11, so a
  token at distance Δ competes only while `m·Δ ≲ τ`. Log the horizon ladder; it is how you
  catch a slope ladder that silently caps context.

**Health**
- Attention entropy per head, and max softmax probability. Catches saturation and dead
  heads, the documented consequence of skipping QK-norm.
- Score std at init and after warmup (E0.6's quantity, tracked).
- Fraction of attention mass on the top-k positions, at several k — this is E2.3's shift,
  measurable during *training* at no cost, long before Stage 2.
- Whether a sink row emerges in `E_K` (one row with high score against everything), and
  what happens to its mass as context grows.

**Capacity**
- Per-frequency-bucket row norms and movement-from-init (E1.2).
- Bank gradient density per step — the estimate is 82% of rows touched at Zipf s = 1.0, 67%
  at s = 1.1, 50% at s = 1.2 for a 0.5M-token batch. Worth confirming on the real corpus,
  because it is the reason dense optimizer state is unavoidable.

**Cost**
- Step time and FLOPs separately (E1.1 needs both).
- Peak memory split: backbone / banks / optimizer state / activations. Banks alone are
  **79 GB** at 14 B/param (fp16 weights + fp32 master + Adam m,v + fp16 grads), **~88 GB**
  with fp32 gradient accumulation, before the backbone.

---

### 8. Silent-corruption traps

Each of these produces a *plausible but wrong* result rather than a crash. They are the
reason a null result here should be re-checked before it is believed.

1. **`roll` wraps, and document packing turns that into a correctness bug.** Zeroing "the
   first position" is correct for one sequence per row. Under packing — near-universal in
   pretraining and required by varlen FlashAttention — token 1 of document 2 mixes in the
   last token of document 1. **Zero the first position of every document, from
   `cu_seqlens`.** This surfaces only as "the shift didn't help", which is exactly the
   conclusion E0.1 is trying to reach honestly.
2. **A fixed bigram mapping in Stage 0.** Both arms pass; the experiment measures nothing.
   Resample per sequence (§4).
3. **Embedding-habit initialization.** σ = 0.02 gives scores of std 4e−4 — attention
   uniform at init, symmetry possibly never broken. Use σ ≈ 1 without QK-norm; with QK-norm
   the sensitivity mostly disappears.
4. **QK-norm in the wrong place.** Normalize the **bank rows, before the mix** — not the
   mixed key `k̂`. Normalizing `k̂` makes its norm a function of the bigram `(t_{j−1}, t_j)`,
   so the score stops being a linear combination of two entries of one row and the zero-cost
   key mix is lost. `ame.md` §7 calls this "the single highest-cost ambiguity in the design".
5. **Inverting the `v̂` recursion.** The V-cache holds `v̂_j`, so raw `v_{j−1}` is not in it.
   Do **not** recover it from `v̂_{j−1} = β₁v_{j−1} + β₂v_{j−2}` — recursive to j = 1,
   unstable, and undefined at β₁ → 0, which is exactly the fully-shifted head you want
   reachable. Keep a one-step raw-`v` register per layer per head on the GPU:
   `L·d_att·2B` ≈ **98 KB** total, free, since the GPU just computed it.
6. **Position 1 under the mixture.** `k̂_1 = α₁k_1` is finite, but a head that learns
   `α₁ → 0` gets logit 0 — neutral and finite, and able to dominate a short prefix. Add an
   explicit learned null/sink key. (Implement the *mixture*, never the hard shift: under
   `Σ_{j≤i−1}` the hard shift gives an empty softmax at i = 1, i.e. NaN.)
7. **Scalar `exp` at decode.** 16 ms hiding inside a 20–30 ms budget, in the one place
   nobody profiles (E0.1).
8. **Missing running max in the streaming softmax.** Overflow, not a wrong answer — but it
   can look like instability attributable to the architecture.
9. **Option A's decayed counts overflow their scale factor early.** The O(1) update keeps
   counts in a common scale growing like `e^{m·i}` — fp32 overflow at **i ≈ 125** for the
   steepest head with geometric slopes, i ≈ 23k for the shallowest. The rescale fires inside
   the first few hundred tokens, not at the end of a long document. Rescale periodically, or
   keep counts in the log domain.
10. **Kernel choice is settled before you start.** The attention kernel must accept an
    additive distance bias. FlexAttention's `score_mod` is the default (no n×n tensor
    materialized); `flash-attn`'s `alibi_slopes` (v2.4+) is the fast path but takes a
    **non-differentiable** tensor, so a learnable-slope ablation needs FlexAttention. Plain
    FA2 with neither is not an option.

---

### 9. Risk register

Ordered by how much it would cost to discover late.

| | risk | when known | cheapest probe | if it fires |
|---|---|---|---|---|
| **R1** | **Fixed-kernel ceiling is binding.** `out_i = Σ_j w(t_i, t_j, i−j)·v_j` — weights never read a hidden state, so retrieval is *never content-addressed*, in any variant. KV shifting widens the id pair to a short id *window*; it does not escape ids. Exact multi-token copying, long-range needle retrieval and variable binding remain the expected failure modes, and this is where reviewers will probe. | Stage 0 for copying; Stage 1 for the rest | E0.1, then RULER/needle at Stage 1 | the hybrid fallback (§10) |
| **R2** | **Score path cannot be brought under budget.** ~2.0 GB/token, 10× the value reads. The row cache is **load-bearing, not an optimization**. | Week 0 (E0.3), confirmed Stage 2 | E0.3 — no model needed | reframe as a quality result, or pivot to the recurrent form |
| **R3** | **Zipf tail is dead capacity.** ~98k params/type over 768 rows; a type seen 10³ times gets ~100 params/occurrence. | Stage 1 (E1.2) | frequency-stratified loss on the first checkpoint | bank-size levers (§5, E1.2); this also *is* the scaling-law study |
| **R4** | **The softmax isn't earning its cost** (C0). | Stage 1 checkpoint (E1.5) | one SVD | pivot (§10) |
| **R5** | **Matched-parameter comparison is unflattering.** Banks are ~4× the backbone; W_Q/W_K deletion returns only 0.2B. | Stage 1 (E1.1) | run the hostile row deliberately | report it as the PKM framing — FLOP-free parameters, not free quality |
| **R6** | **Training-cost asymmetry.** AME trains O(n²); GDN-2/KDA/Mamba-3 train chunkwise O(n). ~15% fewer FLOPs but probably worse wall-clock from 4.9B scattered gradients. | Stage 1 | measure step time from run 1 | report honestly; it will be raised |
| **R7** | **Sharding is a prerequisite, not a tuning step.** 79–88 GB of bank optimizer state. Shard the way Megatron shards a vocab-parallel embedding (rows split, gather locally, all-reduce) rather than letting ZeRO-3 all-gather a 204 MB bank per layer per step. 8-bit or CPU-offloaded optimizer state is the other lever. | before Stage 1 starts | capacity arithmetic | plan it into the Stage-1 setup |
| **R8** | **The gain is Xu et al.'s, not AME's.** KV shifting is validated prior art to 19B (+2.1 avg, +3.6 MMLU at 2.9B/500B). | Stage 1 | the matched-mechanism baseline | mandatory, see §11 |
| **R9** | ~~**Name collision.**~~ **Resolved 2026-11-03**: renamed MAE → **AME** (Attention Memory Embedding), matching the `ame` key the attention registry already used. "MAE" collided with Masked Autoencoders (~39k citations) and "mean absolute error"; "AME" collides with neither. | — | — | done |

---

### 10. Decision tree

```
Week 0 ──► E0.3 row-cache hit rate
             ├─ good (≥ ~75% at ≤ 10 GB) ──► deployment claim stays the headline
             └─ poor ─────────────────────► reframe: AME is a quality/parameter-efficiency
                                            result; promote ScoutAttention + SparDA to
                                            primary baselines and evaluate at fixed GPU
                                            memory rather than fixed FLOPs. Decide EARLY —
                                            it changes what Stage 1 measures.

Stage 0 ──► E0.1 four-arm ablation
             ├─ A1 > A4, α₂ survives ─────► C1 holds. Proceed to Stage 1 with the shift on
             │                               and the matched-mechanism baseline mandatory.
             ├─ A1 ≈ A4 ──────────────────► mechanism inert in AME. The fixed-kernel ceiling
             │                               is real → the hybrid fallback becomes the plan:
             │                               interleave AME layers with standard-attention
             │                               layers (the Engram argument — static lookup
             │                               relieves early layers, freeing attention for
             │                               context). Re-scope the whole program around it.
             └─ A1 > A4 but α₂ → 0 ───────► run the steep-head 2×2. If the collapse vanishes
                                            without steep heads, β was being supplied
                                            elsewhere and only β is doing work — report that;
                                            it weakens the "only route to copying" claim.

Stage 1 ──► E1.5 rank of exp(τ·T)
             ├─ rank large ───────────────► the softmax is earning its cost; build Stage 2.
             └─ rank ≲ few thousand ──────► PIVOT. AME has an exact constant-state recurrent
                                            form: no V-cache, no top-k, no inverted index,
                                            no CPU offload, true O(1) decode. The method then
                                            lands INSIDE the GDN-2/KDA/Mamba-3 family rather
                                            than beside it — which changes the positioning
                                            from "new deployment regime" to "id-indexed keys
                                            as a better linear-attention parameterization".
                                            Most of §6 becomes unnecessary.
```

---

### 11. Reporting standard

Fixed in `baselines.md` §0; repeated here because every table in this program must obey it.

- **Every row carries active parameters, total parameters, training FLOPs, and tokens**,
  MoE-style. There is no single honest baseline when the banks are ~4× the backbone, and
  no single number.
- **Headline table matched on training FLOPs and tokens.** Plus **one matched-total-parameter
  row** as the hostile comparison.
- **Cost axis defined once:** ms per token at batch 1, on named hardware, at a stated
  context length, reported next to **bytes of per-sequence state**. Not FLOPs — every method
  in this comparison is bandwidth-bound, and for AME the binding cost is the score path, not
  the value path.
- **Quality means RULER and needle-in-a-haystack at the deployment length**, not perplexity.
- **Headline figure is a Pareto plot** — recall-heavy score vs per-token decode cost, every
  baseline on it — not a win/loss table. The target region (above StreamingLLM, cheaper than
  Quest, competitive with GDN-2 while offloading to CPU) is currently empty.

**The baselines**, in full in `baselines.md`:

| | baseline | role |
|---|---|---|
| 1 | **GDN-2 (2605.22791) · KDA (2510.26692) · Mamba-3 (2603.15569)** | make-or-break: matched deployment cost, and they train *cheaper* |
| 2 | **FlashAttention (2205.14135)** | reference ceiling, in every table; omitting it looks evasive |
| 3 | **Quest (2406.10774) + StreamingLLM (2309.17453)**, on our own checkpoint | the two ends of the frontier — AME must land strictly inside, or a *training-free* method dominates it |
| 4 | **LoLCATs (2410.10254)** | closest philosophical competitor |
| 5 | **KV-shifting transformer (2411.19574)** | **matched-mechanism control — mandatory** |

On (5): a 2-tap causal depthwise mix is the **short convolution** that most linear-attention
baselines already carry — `att-05` says it "generalizes the shift SSM", `att-06` (GDN)
ablates it as **crucial**, `att-07` (KDA) calls it widely adopted, `att-08` (Mamba-3) is
notable for *dropping* it. So **GDN-2/KDA are already matched on this axis**; only the
full-attention rows (2, 3) need the shift added to stay honest.

Conditional: if the story becomes "bigger contexts / smaller GPUs via CPU async" rather than
quality-per-FLOP, promote **ScoutAttention (2603.27138)** and **SparDA (2606.04511)** to
primary baselines and evaluate at fixed GPU memory. That is the Week-0 branch in §10.

One omission to pre-empt: `ame.md` §5.2 shows Option A *is* a linear-attention model
(RetNet/GLA with a one-hot |V|-dimensional key basis, a static value bank and a
query-dependent elementwise readout mask). **A's correct baseline is therefore GLA/RetNet at
matched state size, not a transformer** — and if E1.5 comes back low-rank, that becomes true
of the whole method.

---

### 12. Open questions, mapped to stages

`ame.md` §8 in priority order, with the stage that can answer each. Two are deliberately
out of priority order, marked ⚑.

| # | question | stage | experiment |
|---|---|---|---|
| 1 | does KV shifting buy AME induction? | 0 | E0.1 |
| 2 | what does contextual payload buy (B vs A)? | 1 | E1.3 |
| 3 | quality gap vs the baselines frontier | 1 | E1.1 |
| 13 ⚑ | what is the softmax actually worth? *cheap, early, could redirect the project* | 1 checkpoint | E1.5 |
| 6 ⚑ | the score path at decode — *top engineering risk*, placed late only because it needs Stage 2 | Wk 0 + 2 | E0.3, E2.1 |
| 4 | distance-bias design: per-head slopes, learned per-type decay, how selection interacts with the bias (top-k over `T − m·Δ` favors recency — verify distant high-score tokens still get retrieved) | 1 + 2 | E1.4, E2.1 |
| 5 | table fine-tuning axis — pruned tables only: 0.2 GB/head-layer at k = 2048, but **78.6 GB over all 384 head-layers, ~157 GB with key indices**, plus optimizer state. The practical version is "a few head-layers". | 2+ | — |
| 7 | top-k policy and the dense→sparse shift | 1 (diagnostic) + 2 | E2.3 |
| 8 | key-window width p | 0 | E0.2 |
| 9 | bank-size levers and the Zipf tail | 1 | E1.2 |
| 10 | hybrid layouts: every-second-layer AME, A-heads + B-heads in a layer, AME early layers only | 1 | — |
| 11 | tokenizer dependence — everything is id-indexed, so BPE granularity shapes what `T` can express *and* what a shifted head can copy (`v_{j−1}` is one BPE piece, not one word) | Wk 0 + 1 | E0.4 |
| 12 | length generalization, 2k → 100k | 1 | E1.4 |

---

#### Numbers worth keeping at hand

Reference config: 1.3B backbone · d_att = 2048 · L = 24 · 16×128 heads · |V| = 50k · fp16 ·
60 GB/s desktop DDR5, 300–400 GB/s server · budget 20–30 ms/token.

| quantity | value |
|---|---|
| bank parameters (E_Q + E_K) | **4.915B / 9.83 GB fp16** — ≈ 4× the backbone |
| …with Option A's E_V | 7.37B |
| returned by deleting W_Q, W_K | 0.201B (≈ 0.4 of ~2.6 GFLOPs/forward → ~15% fewer training FLOPs) |
| rows per token type | **768** (2 banks × 384 head-layers) = 98,304 params/type |
| bank optimizer state | 69 GB at 14 B/param → **79 GB** with fp16 grads, **~88 GB** with fp32 accumulation |
| score path per token | `\|U\|·d_att·L·2B` ≈ **2.0 GB @ \|U\|=20k** → ~33 ms desktop / ~5.6 ms server |
| top-k value reads | `k·d_att·L·2B` ≈ **200 MB** → 3.4 ms streamed, **5–7 ms as 786k×256 B gathers** |
| V-cache | `n·d_att·L·2B` = 3.2 GB @ 32k · **9.8 GB @ 100k** · 12.9 GB @ 128k |
| V-cache write | ~100 KB/token at decode; **3.2 GB in bulk at prefill** (~130 ms over PCIe 4.0 ×16 at ≈25 GB/s) |
| raw-`v` register | `L·d_att·2B` ≈ 98 KB total |
| CPU→GPU per layer | (N, Z, max) ≈ 4 KB fp16 / 8 KB fp32 |
| exponentials per token | `k·H·L` = **786,432** |
| row cache, k = 2048, 4 B/entry | 3.1 GB @ 1000 types · 9.4 GB @ 3000 (~75% hit) · 31.5 GB @ 10k (~86%) · **157 GB** full vocab |
| per-layer pipeline deadline | `CPU_ℓ ≲ GPU_{ℓ−1}` ≈ 0.2 ms, against ≈ 0.14 ms of CPU work |
| decode sync residue (Option B) | ≈ 4·d_att ≈ 8k FLOPs + 4 KB — sync share `1/(n+1)` ≈ 0.05% @ n = 2048 |
| realistic desktop batch-1 @ 100k | **~11–18 ms** (estimate; E2.2 measures it) |
| ALiBi ladder | stretch to 2⁻¹ … ≈2⁻¹⁴ (horizons ~20 … ~185k); the H=16 default 2⁻⁰·⁵ … 2⁻⁸ caps at ~3k |
