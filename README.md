# Nano

# FAQ
1. Why state of model, optim, scheduler is separated from other state parameters?
- We want to start metric_logger ASAP, loading model's distributed checkpoint forces us to create model before loading weights.

2. How to save llama weights?
Set following fields in a config:
```trainer:
  checkpoint:
   load:
    type: huggingface
    path: "meta-llama/Llama-3.2-1B"
  n_steps: 0
```