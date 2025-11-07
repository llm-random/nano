# Nano

## Running Experiments

### Local
```bash
source venv/bin/activate
python main.py --config-path configs --config-name tiny_local
```

### Remote
```bash
python run_exp.py --config-path configs --config-name tiny_local
```

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