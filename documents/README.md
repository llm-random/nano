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
| `scripts/` | Small runnable companions to the docs: parameter/FLOP calculators, figure notebooks. Importable, no side effects on import. | long-lived |

### concepts/

| Doc | Topic |
|---|---|
| [glossary.md](concepts/glossary.md) | Core vocabulary: HP / HPD / PC / PCD, compression-ratio convention, model dimensions, token budgets, LR convention |
| [h-balanced-model.md](concepts/h-balanced-model.md) | The *H-balanced* transformer family: one knob `H` fixes depth, width, heads and `dff` |
| [sam-model.md](concepts/sam-model.md) | *SAM* (stateless memory attention): the same backbone without the Q and K projections and with no attention cost, plus *SAMm*, its parameter-matched variant (`dff = 4.5·dmodel`) |

### scripts/

| Script | What it does |
|---|---|
| [h_balanced_params.py](scripts/h_balanced_params.py) | Dimensions and parameter counts of the H-balanced family; regenerates the tables in `h-balanced-model.md` |
| [h_balanced_flops.ipynb](scripts/h_balanced_flops.ipynb) | Inference FLOPs of an H-balanced model (exact causal attention) + plotly figures |

### notes/

| Doc | Topic |
|---|---|
| [long-context-attention.md](notes/long-context-attention.md) | How deployed 200k+ models avoid all-to-all attention: 8 published methods, their quality evidence, and the attention-cost retention `c(L)` each implies for our FLOP charts |
| [claude-code-machine-setup.md](notes/claude-code-machine-setup.md) | Replicating the Claude Code setup (settings, skills, agent memory) on a new machine via the `claude-sync` repo |
| [cluster_fixes_lem_helios.md](notes/cluster_fixes_lem_helios.md) | lem and helios blockers: $HOME quota, HF cache permissions, PIXI_HOME -- diagnosis and fix plan |
| [main_baseline_audit.md](notes/main_baseline_audit.md) | The audit that made main a clean branch point: test tiers, trainer contracts, config fixes |
| [pretraining_reset_log.md](notes/pretraining_reset_log.md) | Running log of the pretraining-branch reset: what was removed, kept, put aside |
| [pretraining_stage1_removal_plan.md](notes/pretraining_stage1_removal_plan.md) | Stage-1 removal plan (compression/distillation/legacy), phases and gates |

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
