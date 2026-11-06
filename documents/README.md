# documents/

Written reference for the pretraining framework in this repo — for human researchers,
for people seeing the framework for the first time, and for AI agents that need to look up a
concept instead of re-deriving it from the code.

If you are about to explain a concept in a chat, a PR description or a paper comment, check whether
it belongs here first. If it is a concept the team keeps re-explaining, write it here.

## Layout

| Folder | Content | Lifetime |
|---|---|---|
| `concepts/` | Definitions: what a term means, how a thing is constructed, what it implies. Should stay true for months. | long-lived |
| `notes/` | Task notebooks: a concrete investigation, its plan, its results, its open items. Dated, append-only. | per-task |
| `notes/archive/` | Task notebooks that are **closed**: the work finished, or a later doc supersedes them. Kept for the evidence and the reasoning, not consulted for current state. | retired |
| `scripts/` | Small runnable companions to the docs: parameter/FLOP calculators, figure notebooks. Importable, no side effects on import. | long-lived |

### concepts/

| Doc | Topic |
|---|---|
| [glossary.md](concepts/glossary.md) | Core vocabulary: HP / HPD / PC / PCD, compression-ratio convention, model dimensions, token budgets, LR convention |
| [h-balanced-model.md](concepts/h-balanced-model.md) | The *H-balanced* transformer family: one knob `H` fixes depth, width, heads and `dff` |
| [attention-switching.md](concepts/attention-switching.md) | The attention-type registry: gqa/mha/mqa/ame real, mla/mamba2/gated_deltanet as interfaces, per-type options, and the per-layer mixing patterns |
| [sam-model.md](concepts/sam-model.md) | *SAM* (stateless memory attention): the same backbone without the Q and K projections and with no attention cost, plus *SAMm*, its parameter-matched variant (`dff = 4.5·dmodel`) |

### scripts/

| Script | What it does |
|---|---|
| [h_balanced_params.py](scripts/h_balanced_params.py) | Dimensions and parameter counts of the H-balanced family; regenerates the tables in `h-balanced-model.md` |
| [h_balanced_flops.ipynb](scripts/h_balanced_flops.ipynb) | Inference FLOPs of an H-balanced model (exact causal attention) + plotly figures |
| [stage_results.py](scripts/stage_results.py) | AME Stage 1–2 results from W&B: the run tables, eval curves, readouts and loss-curve figures in `stage_1_report.md` and `stage_2_report.md` |

### notes/

**AME** — the id-indexed Q/K attention method. Three live docs: what it is, what we
will test, how fast it runs. Everything else is in `notes/archive/`.

| Doc | Topic |
|---|---|
| [ame_implementation.md](notes/ame_implementation.md) | **What AME is**: the layer, the plumbing it needed, the spec choices settled to land it, what is verified, what is open |
| [ame_experiments.md](notes/ame_experiments.md) | **The scientific program — unrun, and the next thing to do.** The claims as falsifiable statements, Stage 0/1/2, the decision tree, the reporting standard |
| [ame_optimization_report.md](notes/ame_optimization_report.md) | **The MFU record (2026-11-04/05)**: every improvement tried and its effect, what is still worth trying, all ~66 experiments, the decision history, best config per model size. 26.84% → 35.02% at 4 GPUs, 37.96% at h16/1 GPU |
| [ame_microbatch_sweep.md](notes/ame_microbatch_sweep.md) | **Micro-batch sweep, AME and dense (2026-11-05)**: all 48 runs, h12/h16/h20 × 1/4 GPUs — MFU, step, tokens/s, peak memory, fwd/bwd time per run |
| [stage_1_report.md](notes/stage_1_report.md) | **Stage 1 of the program, on LEM (from 2026-11-06)**: the learning-rate grids for dense h12 and the simplest AME h12 at 4B tokens — every job, result and decision |
| [stage_2_report.md](notes/stage_2_report.md) | **Stage 2, on LEM (from 2026-11-06)**: AME h12 with ALiBi, with the KV shift, and with both, at 4B tokens — the 2 × 2, the noise floor, the learning-rate check |
| [stage_1b_report.md](notes/stage_1b_report.md) | **Stage 1b, on LEM (from 2026-11-06)**: Stage 1 at h16 and 8B tokens — the learning-rate grids for dense h16 and the simplest AME h16 |

**AME configs** (layout since 2026-11-05): `configs/ame/` holds six configs, the
best measured MFU per model size × GPU count. Every other AME config — benchmarks,
A/B twins, ablations, the smoke run — moved to `configs/ame_tests/` under the same
name. See [configs/ame/README.md](../configs/ame/README.md).

**Efficiency**

| Doc | Topic |
|---|---|
| [training_efficiency.md](notes/training_efficiency.md) | **The single-GPU efficiency record, chronological**: baseline audit and MFU tables, the nanoGPT gap analysis, the six-variant ablation, kernel-level attribution, the compile work, final results (46.4% MFU at h20), the measurement protocol. Merges five earlier notes |
| [multigpu_efficiency.md](notes/multigpu_efficiency.md) | The multi-GPU counterpart: same conventions, same protocol, FSDP2 and DDP at 4×H100 |
| [multigpu_compile_scope_plan.md](notes/multigpu_compile_scope_plan.md) | The plan that closed the compile-scope gap under FSDP2 — `compile_blocks_full`, 33.7% → 44.3% |
| [long-context-attention.md](notes/long-context-attention.md) | How deployed 200k+ models avoid all-to-all attention: 8 published methods, their quality evidence, and the attention-cost retention `c(L)` each implies for our FLOP charts |

**Repo and branch work**

| Doc | Topic |
|---|---|
| [main_baseline_audit.md](notes/main_baseline_audit.md) | The audit that made `main` a clean branch point: test tiers, trainer contracts, config fixes |
| [dead_code_cleanup_goal.md](notes/dead_code_cleanup_goal.md) | Dead code on the pretraining branch, verified by whole-repo reference scan + `ruff F401,F811,F841` |
| [typed_config_goal.md](notes/typed_config_goal.md) | Wiring `src/definitions.py` back in: Pylance typing and submit-time config validation. Run after the cleanup goal |
| [pretraining_reset_log.md](notes/pretraining_reset_log.md) | Running log of the pretraining-branch reset: what was removed, kept, put aside |
| [pretraining_stage1_removal_plan.md](notes/pretraining_stage1_removal_plan.md) | Stage-1 removal plan (compression/distillation/legacy), phases and gates |
| [pretraining_stage2_plan.md](notes/pretraining_stage2_plan.md) | Stage-2 plan: H-balanced lineage, NeoX tokenizer, attention registry, pattern mixing |

**Environment**

| Doc | Topic |
|---|---|
| [cluster_fixes_lem_helios.md](notes/cluster_fixes_lem_helios.md) | lem and helios blockers: `$HOME` quota, HF cache permissions, `PIXI_HOME` — diagnosis and fix plan |
| [claude-code-machine-setup.md](notes/claude-code-machine-setup.md) | Replicating the Claude Code setup (settings, skills, agent memory) on a new machine via the `claude-sync` repo |

### notes/archive/

Closed notebooks. Each opens with what superseded it. Two are still the **only** source
for something, and say so at the top — don't assume archived means redundant.

| Doc | Topic | Closed |
|---|---|---|
| [ame_optimization_plan.md](notes/archive/ame_optimization_plan.md) | The three-tier plan. Superseded by the report, **except §3** — the experiment ledger, the only map from a result back to its wandb run and config — and §4, the decision log | 2026-11-05 |
| [ame_optimization.md](notes/archive/ame_optimization.md) | The pkmf static-memory survey: the evidence behind each O-number, which the report's §1.0 inventory only names. Also O22–O26, which are schedule/LR items rather than MFU work | 2026-11-05 |
| [interactive_ame.md](notes/archive/interactive_ame.md) | The hour-by-hour working log of the AME MFU session. The report was written from it and supersedes it | 2026-11-05 |
| [ame_review_and_fixes.md](notes/archive/ame_review_and_fixes.md) | The AME code review and its remediation, merged: 8 findings (F1–F8) with the evidence, and the four phases that fixed them. F1 — FlexAttention running uncompiled — is the ancestor of all the efficiency work | 2026-11-03 |

## Conventions for writing here

- **One concept per file**, named after the concept in kebab-case.
- **Mark status at the top**: `Status: established` (it is what the code does), `Status: proposed`
  (a design nobody has run yet), or `Status: historical` (superseded, kept for context). A newcomer
  must never have to guess whether a doc describes reality or a plan.
- **Numbers carry their source**: a config path, a source file and line, a wandb run id, or the
  script that computed them. A table without a source is a rumour.
- **Date anything that can go stale**, with absolute dates (`2026-10-20`, not "last week").
- Facts that belong to the code (what a function does today) stay in the code. This folder is for
  what the code does *not* say: why a construction is the way it is, which convention is in force,
  and what the alternatives were.
- **Retire a note rather than deleting it.** When a notebook's work is finished, or a later doc
  supersedes it, `git mv` it into `notes/archive/` and replace its status block with one that says
  so and names the successor. A closed note that still *claims* to be current is worse than no note:
  it is read as reality and acted on. Two tests for whether a note is ready to retire — nobody will
  add to it again, and nothing should be looked up in it to answer "what is true today". Check for
  inbound links from code and configs before moving it; comments cite these files by name.
- **Merge a matched pair when neither half stands alone.** A findings doc and its fix plan, a
  proposal and its outcome. Preserve the section identifiers the source tree cites.
