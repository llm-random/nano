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
| `scripts/` | Small runnable companions to the docs: parameter/FLOP calculators, figure notebooks, the common evaluator and its wrappers. Importable; no side effects on import, except that `parity_numerics.py` edits `sys.path` and sets four rank variables (`WORLD_SIZE`, `LOCAL_WORLD_SIZE`, `RANK`, `LOCAL_RANK`), and `bank_gates_eval.py`, which imports it, edits `sys.path`. | long-lived |

### concepts/

| Doc | Topic |
|---|---|
| [glossary.md](concepts/glossary.md) | Core vocabulary: HP / HPD / PC / PCD, compression-ratio convention, model dimensions, token budgets, LR convention |
| [h-balanced-model.md](concepts/h-balanced-model.md) | The *H-balanced* transformer family: one knob `H` fixes depth, width, heads and `dff` |
| [attention-switching.md](concepts/attention-switching.md) | The attention-type registry: gqa/mha/mqa/ame real, mla/mamba2/gated_deltanet as interfaces, per-type options, and the per-layer mixing patterns |
| [ame-model.md](concepts/ame-model.md) | *AME* (Attention Memory Embedding) for counting: Q/K banks gathered by token id instead of Q and K projections, a full-width V; a prefill is dense attention; a decode step reads its score row from a precomputed vocab × vocab table and pays a top-k value sum and the bigram selection. Plus *AMEm*, matched to the vanilla weights (`dff ≈ 4.33·dmodel`) |
| [comparing-runs.md](concepts/comparing-runs.md) | How two runs are compared: what fixes a run's init and data, which runs pair (the standard recipes v1 → v3), the paired gap, `final_ce`'s precision, the tie bands, rerunning an older run, and what a result must report |

### scripts/

| Script | What it does |
|---|---|
| [h_balanced_params.py](scripts/h_balanced_params.py) | Dimensions and parameter counts of the H-balanced family; regenerates the tables in `h-balanced-model.md` |
| [h_balanced_flops.ipynb](scripts/h_balanced_flops.ipynb) | Inference FLOPs of an H-balanced model (exact causal attention) + plotly figures |
| [h_balanced_flops_quotes.py](scripts/h_balanced_flops_quotes.py) | Re-derives every number the notebook's prose quotes from the notebook's own functions and names any that are stale; run it after changing a knob or the cost model |
| [stage_results.py](scripts/stage_results.py) | AME Stage 1–2 results from W&B: the run tables, eval curves, readouts and loss-curve figures in `archive/stage_1_report.md` and `archive/stage_2_report.md` |
| [ame_comparison_h12.py](scripts/ame_comparison_h12.py) | The best AME and dense h12 runs side by side from W&B: every table and figure in `archive/ame_comparison_h12.md`; the figures go to the folder it is given |
| [parity_numerics.py](scripts/parity_numerics.py) | The common evaluator: a checkpoint's precise final CE on the full held-out set (`score`, under `ame_ref32` and `ame_prod`), and the `dump_data` that wrote the set. Ported from `parity_audit` for the bank gates; it cannot score FSDP2 runs (its docstring lists the limits) |
| [bank_gates_eval.py](scripts/bank_gates_eval.py) | The bank gates' secondary evaluation: loss and attention statistics per `steps_seen` bucket, from a run's `token_counts.pt` |
| [bank_gates_lem_eval.sh](scripts/bank_gates_lem_eval.sh) | Runs evaluation scripts on one LEM H100 from a committed state of the repo; the job exits 1 if any of them failed |
| [ame_ckpt_readouts.py](scripts/ame_ckpt_readouts.py) | AME readouts (τ, horizons, α₂, β₂) per layer from a final checkpoint, for the runs that logged none under FSDP2 (Stage 2c) |

### notes/

The live notes: what is true today, and what is still open. Closed notebooks are in
`notes/archive/`, below. On 2026-11-27, 35 notebooks moved there; what they still said about
today was carried into the live notes and into [comparing-runs.md](concepts/comparing-runs.md).

**AME** — the id-indexed Q/K attention method.

| Doc | Topic |
|---|---|
| [ame_implementation.md](notes/ame_implementation.md) | **What AME is**: the layer, the plumbing it needed, the spec choices settled to land it, what is verified, what is open |
| [standard_model_improvements.md](notes/standard_model_improvements.md) | **The standard AME and dense models, and why** (decided 2026-11-21, merged 2026-11-22, recipe v3 since 2026-11-26): every change tested from 2026-11-04 to 11-20 with its result and verdict, the standard configs, the AME − dense gap, and the evidence behind each choice. It summarizes the results of every archived AME notebook |
| [review_upgrades.md](notes/review_upgrades.md) | **The AME backlog** (from the 2026-11-25 review): the analyses and ablations proposed for later, each needing Maciej's OK, and the runs already planned (U9) |
| [ame_optimization_report.md](notes/ame_optimization_report.md) | **The AME MFU record (2026-11-04/05)**: every improvement tried and its effect, what is still worth trying, all ~66 experiments, the decision history, best config per model size, and the 48-run micro-batch sweep (appendix). 26.84% → 35.02% at 4 GPUs, 37.96% at h16/1 GPU |
| [pkmf_init_comparison.md](notes/pkmf_init_comparison.md) | **Weight init, ours against pkmf (2026-11-07)**: how each framework initializes every weight, read from both codebases |

**Configs.** The standard AME and dense models are `configs/standard/`: six files, recipe v3.
`configs/ame/` holds six configs, the best measured MFU per model size × GPU count (layout
since 2026-11-05); every other AME config — benchmarks, A/B twins, ablations, the smoke run —
is in `configs/ame_tests/` under the same name. See
[configs/ame/README.md](../configs/ame/README.md). The experiment configs (`configs/stage_*`,
`configs/bank_gates/`) cite their notebooks in `notes/archive/`.

**Efficiency**

| Doc | Topic |
|---|---|
| [training_efficiency.md](notes/training_efficiency.md) | **The single-GPU efficiency record, chronological**: baseline audit and MFU tables, the nanoGPT gap analysis, the six-variant ablation, kernel-level attribution, the compile work, final results (46.4% MFU at h20), the measurement protocol. Merges five earlier notes |
| [multigpu_efficiency.md](notes/multigpu_efficiency.md) | The multi-GPU counterpart: same conventions, same protocol, FSDP2 and DDP at 4×H100. `compile_blocks_full` took h20 from 33.7% to 44.3% (§7.7) |
| [long-context-attention.md](notes/long-context-attention.md) | How deployed 200k+ models avoid all-to-all attention: 8 published methods, their quality evidence, and the attention-cost retention `c(L)` each implies for our FLOP charts |

**Environment**

| Doc | Topic |
|---|---|
| [lem_file_quota_request.md](notes/lem_file_quota_request.md) | Reply to LEM's admins about project 649's file quota: a permanent 2M, with 1M as the fallback. In Polish; drafted 2026-11-20, revised 2026-11-26, not sent yet |
| [claude-code-machine-setup.md](notes/claude-code-machine-setup.md) | Replicating the Claude Code setup (settings, skills, agent memory) on a new machine via the `claude-sync` repo |

### notes/archive/

Closed notebooks: the work finished, or a later doc supersedes them. Each opens with a block
that says when and how it closed, where today's state is, and what is still **only** there —
don't assume archived means redundant. Code, configs and tests cite many of them by name and
section, so they are kept whole.

**AME: the experimental program, Stages 1–2c**

| Doc | Topic | Closed |
|---|---|---|
| [ame_experiments.md](notes/archive/ame_experiments.md) | The program: the stages, their recipes and configs, the reporting rules, and the previous plan (the claims, the synthetic Stage 0 test) as an appendix | 2026-11-10 |
| [stage_1_report.md](notes/archive/stage_1_report.md) | Stage 1 on LEM: the learning-rate grids for dense h12 and the simplest AME h12 at 4B tokens — every job, result and decision | 2026-11-06 |
| [stage_2_report.md](notes/archive/stage_2_report.md) | Stage 2 on LEM: AME h12 with ALiBi, with the KV shift, and with both — the 2 × 2, the noise floor, the learning-rate check | 2026-11-07 |
| [stage_1b_report.md](notes/archive/stage_1b_report.md) | Stage 1 at h16 and 8B tokens | 2026-11-06 |
| [stage_2b_report.md](notes/archive/stage_2b_report.md) | Stage 2 at h16 and 8B tokens, with the learning-rate check and the re-run at its n | 2026-11-07 |
| [stage_1c_report.md](notes/archive/stage_1c_report.md) | Dense h20 at 16B tokens on 4 GPUs. **Still the only write-up** of the Helios compile fix (aarch64, torch 2.7.1) and of the two-half submission | 2026-11-09 |
| [stage_2c_report.md](notes/archive/stage_2c_report.md) | The best AME variant (ALiBi with the KV shift) at h20 and 16B tokens, and its gap to dense against h12 and h16 | 2026-11-10 |
| [stage_runs.md](notes/archive/stage_runs.md) | Every run of the six stages: 48 training runs, 18 smokes, 3 cancelled jobs, with configs, commits, jobs and W&B ids. **The only full ledger** | 2026-11-10 |
| [ame_comparison_h12.md](notes/archive/ame_comparison_h12.md) | The best AME against the best dense at h12 (S2-PS against S1-D): the training logs compared, the hypotheses, and F12's replicates, which set AME's tie band. Its next angles never ran | 2026-11-07 |

**AME: implementation, MFU and the efficiency-parity audit**

| Doc | Topic | Closed |
|---|---|---|
| [ame_review_and_fixes.md](notes/archive/ame_review_and_fixes.md) | The AME code review and its remediation, merged: 8 findings (F1–F8) with the evidence, and the four phases that fixed them. F1 — FlexAttention running uncompiled — is the ancestor of all the efficiency work | 2026-11-03 |
| [ame_optimization_plan.md](notes/archive/ame_optimization_plan.md) | The three-tier plan. Superseded by the report, **except §3** — the experiment ledger, the only map from a result back to its wandb run and config — and §4, the decision log | 2026-11-05 |
| [ame_optimization.md](notes/archive/ame_optimization.md) | The pkmf static-memory survey: the evidence behind each O-number, which the report's §1.0 inventory only names. Also O22–O26, which are schedule/LR items rather than MFU work | 2026-11-05 |
| [interactive_ame.md](notes/archive/interactive_ame.md) | The hour-by-hour working log of the AME MFU session. The report was written from it and supersedes it | 2026-11-05 |
| [ame_efficiency_parity_plan.md](notes/archive/ame_efficiency_parity_plan.md) | Does AME's MFU work cost it quality? The audit's plan: the inventory of AME-only efficiency changes, offline numerics, 300-step training gates, and the `/goal` that ran it | 2026-11-07 |
| [ame_efficiency_parity_report.md](notes/archive/ame_efficiency_parity_report.md) | The audit's record: no AME efficiency change is harmful. The numerics, the training gates, the 20 full-length runs, and the origin of the common evaluator | 2026-11-07 |
| [ame_bf16_scalars_plan.md](notes/archive/ame_bf16_scalars_plan.md) | The audit's follow-up plan: full-length A/Bs on entropy for bf16 per-head scalars and the parity runs | 2026-11-07 |

**AME: the changes tested for the standard models**

| Doc | Topic | Closed |
|---|---|---|
| [ame_bank_gates_plan.md](notes/archive/ame_bank_gates_plan.md) | Bank gates, plan: four ways to trust a bank row (B1 confidence, B2 context, B3 backoff, ST stem), one h12 run each, and the h16 extension. **The spec behind `bank_gate`**, which stays on `ame_main`, off by default | 2026-11-08 |
| [ame_bank_gates_report.md](notes/archive/ame_bank_gates_report.md) | Bank gates, report: at h12, ST, B2 and B3 tie for best; the twelve W&B runs of both stages | 2026-11-08 |
| [ame_bank_gates_stage2_plan.md](notes/archive/ame_bank_gates_stage2_plan.md) | Bank gates, stage 2 plan: separates the σ = 0.01 init from the gates. The spec behind `confidence_init` and the four `stage2_*` configs | 2026-11-08 |
| [ame_bank_gates_stage2_report.md](notes/archive/ame_bank_gates_stage2_report.md) | Bank gates, stage 2 report: the bank init is all of stage 1's gain. Every gate was rejected (ST on 2026-11-21: it breaks the score table) | 2026-11-08 |
| [ame_improve_runs_report.md](notes/archive/ame_improve_runs_report.md) | Banks at 4× the learning rate: ratio 4 beats 1 at h12 and h16 | 2026-11-08 |
| [ame_init_exp.md](notes/archive/ame_init_exp.md) | Init experiment: Xavier embedding, head and banks with pkmf's `Linear` init. No gain | 2026-11-09 |
| [ame_init_embed_ablation.md](notes/archive/ame_init_embed_ablation.md) | Init follow-up: the new init with the old embedding, then n ± 1. The Xavier embedding was AME's whole problem | 2026-11-10 |
| [ame_init_pkmf_linears.md](notes/archive/ame_init_pkmf_linears.md) | Init follow-up: pkmf-scale `Linear`s and head under the old embedding cost about +0.10 in both models | 2026-11-10 |
| [ame_bank_sigma_grid.md](notes/archive/ame_bank_sigma_grid.md) | Bank-init σ grid at h12: a plateau from 1e-5 to 0.0178 with a failure edge on each side; why σ = 1e-4. **Round 6 (σ = 1e-6) ran on 2026-11-19 and was never scored** | 2026-11-18 |
| [settle_runs_report.md](notes/archive/settle_runs_report.md) | Settle runs: the six open items of the standard-model digest, measured on LEM, with the judging rules and the submission in detail | 2026-11-20 |
| [doc_mask_report.md](notes/archive/doc_mask_report.md) | Document masking for packed rows: implementation and h12 runs. Rejected by Maciej: about −0.02 for 12% more time per step | 2026-11-21 |
| [standard_models_report.md](notes/archive/standard_models_report.md) | Standard models: the branch, the tests, the h12 reproductions and the merge into `ame_main`, with the implementation plan as an appendix | 2026-11-22 |

**The 2026-11-25 review of the unpushed `ame_main` commits**

| Doc | Topic | Closed |
|---|---|---|
| [rev1.md](notes/archive/rev1.md), [rev2.md](notes/archive/rev2.md) | The two reviews of the nine older commits, with Maciej's responses and rev2's reply | 2026-11-25 |
| [review_issues.md](notes/archive/review_issues.md) | Every issue the review found (I1–I27, N1–N9, A1–A4), the solution chosen and the commit that applied it. Code and tests cite its numbers | 2026-11-26 |

**Framework and environment**

| Doc | Topic | Closed |
|---|---|---|
| [pretraining_reset_log.md](notes/archive/pretraining_reset_log.md) | Running log of the pretraining-branch reset: what was removed, kept, put aside. **Still the only list** of the work put aside (eval harness, MoE, `pixi.toml` pruning) and of the decisions parked for Maciej | 2026-11-03 |
| [pretraining_stage1_removal_plan.md](notes/archive/pretraining_stage1_removal_plan.md) | Stage-1 removal plan (compression/distillation/legacy), phases and gates | 2026-10-30 |
| [pretraining_stage2_plan.md](notes/archive/pretraining_stage2_plan.md) | Stage-2 plan: H-balanced lineage, NeoX tokenizer, attention registry, pattern mixing | 2026-10-30 |
| [dead_code_cleanup_goal.md](notes/archive/dead_code_cleanup_goal.md) | Dead code on the pretraining branch, verified by whole-repo reference scan + `ruff F401,F811,F841` | 2026-11-03 |
| [typed_config_goal.md](notes/archive/typed_config_goal.md) | Wiring `src/definitions.py` back in: Pylance typing and submit-time config validation | 2026-11-03 |
| [main_baseline_audit.md](notes/archive/main_baseline_audit.md) | The audit that made `main` a clean branch point: test tiers, trainer contracts, config fixes (PR #199) | 2026-10-30 |
| [multigpu_compile_scope_plan.md](notes/archive/multigpu_compile_scope_plan.md) | The plan that closed the compile-scope gap under FSDP2 — `compile_blocks_full`, 33.7% → 44.3% | 2026-11-03 |
| [cluster_fixes_lem_helios.md](notes/archive/cluster_fixes_lem_helios.md) | lem and helios blockers: `$HOME` quota, HF cache permissions, `PIXI_HOME` — diagnosis and fixes | by 2026-11-27 |

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
