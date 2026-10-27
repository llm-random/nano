# SliceGPT smoke suite

A short end-to-end check that the SliceGPT baseline still works after a change — to the
port, to the model code, or to the configs it composes from.

## Why the configs are generated, not committed

A composed Hydra config is a *snapshot*: it bakes in the cluster paths, model dimensions
and defaults that were active when it was written. Committing one means that the moment
anything it composed from changes, the smoke test silently stops testing the current
code — which is precisely what it exists to catch.

So `generate.py` rebuilds the configs from whatever is in the repo now. It also expands
`^`-prefixed grid keys the same way `run_exp.py` does, because `main.py` cannot consume
them directly (see `scripts/expand_config.py`).

## Running it

```bash
# 1. activate your environment the way you normally do for this cluster
# 2. generate, pointing at your own paths
python smoke/generate.py --work-dir /scratch/$USER/slicegpt_smoke \
    --source-model /path/to/llama1b_in_nano.pt \
    --importances  /path/to/minitron_dimensions_importances.pt

# 3. run a scenario
bash smoke/run_smoke.sh s1_invariance
SMOKE_GPUS=2 bash smoke/run_smoke.sh s4_rft
```

`run_smoke.sh` with no argument lists what has been generated.

Nothing here hardcodes a user or a cluster.

**Without `--source-model` / `--importances` the generated configs inherit the paths the
source config names**, which are one researcher's. Pass them (or set `SMOKE_SOURCE_MODEL`
/ `SMOKE_IMPORTANCES`) and the output contains no account name at all. Other paths — data
sets on a different cluster — go through `--override KEY=VALUE`, repeatable.

The cluster itself comes from the composed config's defaults list and cannot be overridden
on the command line: those entries are relative (`../_cluster@_here_`) and Hydra's override
lexer rejects both `..` and `@_here_`. To run elsewhere, retarget the paths or add a
cluster variant of the source config.

Paths come from these arguments and variables:

| variable | default | meaning |
|---|---|---|
| `SMOKE_WORK_DIR` | `./smoke_work` | checkpoints the suite produces and consumes |
| `SMOKE_CONFIG_DIR` | `smoke/generated` | generated configs |
| `SMOKE_RESULTS` | `./smoke_results` | logs |
| `SMOKE_GPUS` | 1, or 2 for `s4_*`/`s5_*` | processes for `torchrun` |
| `SMOKE_SOURCE_MODEL` | from the config | nano checkpoint to slice |
| `SMOKE_IMPORTANCES` | from the config | dimension-importances `.pt` |

## Scenarios, and what each would catch

| scenario | GPUs | checks |
|---|---|---|
| `s1_invariance` | 1 | slicing at `d = D` with no dff pruning. The model is mathematically unchanged, so the loss must be too. **The strongest single check** that fusion and rotation are correct — it fails loudly if either breaks. |
| `s2_slice_r50` | 1 | real hybrid r50 slicing with a small calibration set; saves the checkpoint the later scenarios load. Catches breakage in the slicing path and in save. |
| `s4_rft` | 2 | 30-step recovery training from that checkpoint. Catches breakage in the `sliced_nano` load path and in FSDP2 with a sliced model. |
| `s4_rft_distil` | 2 | the same with distillation. |
| `s5_lr<n>` | 2 | 100-step LR sweep, `learning_rate = n`. Not a correctness check — use it when picking an LR. |

Run `s1` first: if it fails, the rest will too, and `s1` says why most directly.

## Order

`s2` writes the checkpoint that `s4_*` and `s5_*` read, so run it before them. `s1` is
independent.

Background on the port itself: `documents/notes/slicegpt.md`.
