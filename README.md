# Nano

## Setup

### Local
Firstly install pixi, then run:
```bash
pixi install
```
### Remote
Run to create / update pixi on cluster:
```bash
python update_pixi.py --config-path configs --config-name tiny_remote
```

**Requirements:**
- Config file must specify `infrastructure.server` (target cluster)
- `infrastructure.slurm.script` must contain `export PIXI_HOME=...` line
- Only affects the cluster specified in the config file

**What it does:**
1. Copies local `pixi.toml` and `pixi.lock` to remote cluster
2. Runs `pixi install` on compute node via SLURM (GPU params auto-removed)
3. Archives old pixi files before installing new environment

## Running Experiments

### Local
```bash
pixi shell
python main.py --config-path configs --config-name tiny_local
```

### Remote
```bash
python run_exp.py --config-path configs --config-name tiny_remote
```

**Note:** `run_exp.py` does not copy pixi files (`pixi.toml`, `pixi.lock`) to the cluster to avoid inflating memory and file count in `$HOME`. Use `update_pixi.py` (see Setup > Remote) to update the pixi environment on the cluster first.

## Hydra
Uses [Hydra](https://hydra.cc/) for configuration management. Classes are instantiated via `_target_`:

```yaml
trainer:
  train_dataloader:
    _target_: src.core.datasets.get_dataloader
    dataset_path: /path/to/data
    sequence_length: 2048
```

# FAQ
1. Why state of model, optim, scheduler is separated from other state parameters?
- We want to start metric_logger ASAP, loading model's distributed checkpoint forces us to create model before loading weights.

2. How to load llama weights?
Set following fields in a config:
```yaml
trainer:
  checkpoint:
   load:
    type: huggingface
    path: "meta-llama/Llama-3.2-1B"
  n_steps: 0
```