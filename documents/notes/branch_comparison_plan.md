# Branch comparison: pretraining-jm vs ctx_scl, and the eval harness

2026-11-05. Step 1 is running; this note covers steps 2 and 3. Step 2 revised
2026-11-06 after reading ctx_scl's code.

## Goal

Check that `pretraining-jm` trains as well as `ctx_scl` (the reference) on one
identical job. Two measurements:

- **Eval loss** — the primary comparison. It is sensitive to small differences.
- **Downstream benchmarks** — a gross-breakage check. At this scale they only
  separate runs that differ by 1.5–4 points (see [Noise](#benchmark-noise)).

## Step 1 (running): the reference job on pretraining-jm

`configs/jm/h16_chinchilla.yaml`, W&B tag `branch_compare`. Everything step 2 must
match is in this table.

| Setting | Value |
|---|---|
| Model | h16 h-balanced: 354.6M params, dmodel 1024, dff 4096, 16 blocks, 16 q heads, 8 kv heads (GQA), head dim 64 |
| Blocks | pre-norm RMSNorm (eps 1e-5), SwiGLU feed-forward, RoPE base 10000 without llama frequency scaling |
| Vocab / tokenizer | 50304 (padded) / GPT-NeoX (`EleutherAI/gpt-neox-20b`) |
| Init | linears: truncated normal, std = 1/sqrt(fan_in), cut at ±2 std (`llm_random_weight_init`, scale 1); embedding: `trunc_normal_` |
| Data | FineWeb-Edu, `/storage_nvme_4/llm-random/datasets/fineweb/train` |
| Sequence × batch | 1024 × 256 = 262k tokens/step |
| Steps / tokens | 30,000 / 7.86B (~22 tokens/param, Chinchilla-optimal) |
| Optimizer | AdamW, lr 2^-10 ≈ 9.8e-4, weight decay 0.1, grad clip 1.0 |
| Schedule | cosine to 10% of peak, 1% warmup (300 steps) |
| Precision | FSDP2 with bf16 params, fp32 gradient reduce |
| Hardware | entropy, 4× H100 |
| Checkpoint | final only, `/storage_nvme_4/nano/pretraining/j321m/checkpoints/` |

## Step 2: the same job on ctx_scl

### What "the same job" means

Every row in the table above must match. What may differ is the code that
implements it — compile, kernels, dataloader internals, sample order. Those
differences are what this comparison tests.

### Code: one function

**The GPT-NeoX tokenizer does not exist on ctx_scl.** It only has gpt2, llama and
smollm tokenize functions. Copy `gpt_neox_tokenize_fn` from `src/core/datasets.py`
here into ctx_scl's `src/core/datasets.py`, the module its configs reference.
(`src/context_scaling/dataset.py` holds a duplicate set of tokenize functions that no
config points at.) Without this, nothing is comparable: per-token loss depends on the
tokenizer, and the benchmarks would tokenize differently.

Everything else is config.

### Config

One file, `configs/context_scaling/branch_compare/h16_chinchilla.yaml`, composed like
ctx_scl's `context_scaling/workshop_icml/dense_mha.yaml`:

```yaml
# @package _global_
defaults:
  - /_cluster@_here_: entropy
  - /_model/context_scaling@_here_: template
  - /_trainer@_here_: llama
  - /_dataset@_here_: fineweb
  - /_checkpoints@_here_: none
  - /_infrastructure@_here_: default
  - /_downstream_eval@_here_: joint
  - /_feedforward: swiglu
  - _self_
```

No `_attention` group (it sets kv_heads to 16 or 1). No `_parameterization`: simpleP
parameter groups are not in step 1, and the cosine scheduler asserts a single group.

**Model.** `_model/context_scaling/template.yaml` is the stack ctx_scl's own
experiments train. It builds `src.core.model.*`, the same classes as this branch
(Residual, TransformerBlock, Linear, RoPEAttention, SwiGLU ≡ GatedFeedForward), and
its `trunc_normal_init` is `llm_random_weight_init` under another name. It also
defines `common.tokenizer`, which `joint.yaml` reads. Not `_model/llama/base_model.yaml`:
that builds the projected-compression stack (`src.projected_compression.model`) with
vocab 128256 and llama RoPE.

With `n_blocks: 16` the template already gives dmodel = datt = 1024, q_heads 16 and
vocab 50304. Override the rest:

| Key | Template | Set to |
|---|---|---|
| `common.n_blocks` | `???` | 16 |
| `common.dff` | 160 × n_blocks = 2560 | 4096 |
| `common.kv_heads` | unset (comes from `_attention`) | 8 |
| `common.init_scale` | **0.1** | 1 |
| `rope_base` / `rope_scale_freqs` | 500000 / true | 10000 / false |
| `attention_fn.pre_attn_fn` (QK-norm) | on | `null` |
| `common.tokenizer` | gpt2 | `EleutherAI/gpt-neox-20b` |
| train and eval `tokenize_fn` | `gpt2_tokenize_fn` | `gpt_neox_tokenize_fn` |
| `common.data_seed` | 123 | 123 (the workshop configs change it; don't) |

`init_scale: 0.1` is the dangerous default: a 10× smaller init that fails silently.

**Trainer.** Use ctx_scl's `_trainer/llama.yaml`, not `_trainer/context_scaling.yaml`.
It already matches step 1: cosine to 10%, 1% warmup, eval every 100 steps × 10
batches, `fixed_eval: false`, and `n_steps: ${trainer.n_steps}` inside the scheduler
block (ctx_scl's `main.py` passes the scheduler only `optimizer=`). It does carry two
stale keys, `downstream_eval_interval` and `final_downstream_eval`. They stopped being
Trainer fields when the evaluator took over its own cadence, and attrs rejects unknown
keys, so delete both on the new branch. (`context_scaling.yaml` brings WSD, eval every
1000 steps and `fixed_eval: true`, and its scheduler's `decay_steps` key can't be
removed by an override.)

Set as in step 1: `sequence_length: 1024`, `batch_size: 256`, `n_steps: 30_000`,
`learning_rate: 10` (same 2^-n convention), `gradient_accumulation_steps: 1`.

**Rest.**
- `heavy_metrics_calculation_interval: 100`. It is `???` in ctx_scl's
  `_infrastructure/default.yaml`, and 100 here.
- `trainer.checkpoint.save.path: ${oc.env:PROJECT_HOME_PATH,<fallback>}/checkpoints`,
  which is `/storage_nvme_2/nano/<user>/checkpoints` in the job. `_checkpoints/none`
  leaves the path `null`, and then nothing is saved. The fallback is needed because
  ctx_scl's `run_exp.py` resolves the config on the laptop for its summary. Don't copy
  the path from `_checkpoints/context_scaling/shared_entropy.yaml`: nothing expands its
  `$USER`.
- `slurm.gres: gpu:4`, W&B tag `branch_compare`.

### Checks before launching (cheap, local)

- [ ] Parameter count is 354.6M, and every tensor shape matches step 1's model
      (dump both `state_dict` shapes and diff them). The embedding key differs:
      `embedding.embedding.weight` there, `embedding.weight` here. Any other
      difference needs explaining.
- [ ] Initial weights are bit-identical. Both branches call
      `torch.manual_seed(train_dataloader.dataset.seed)` right before
      `instantiate(cfg.model)`, build modules in the same order with the same init
      function, and run torch 2.7.1, so with seed 123 on both sides they may match
      exactly. If they don't, fall back to each tensor's std at init matching within
      sampling noise.
- [ ] The learning rate at steps 0, 300, 15,000 and 29,999 matches.
- [ ] Both tokenizers give the same ids for a sample string (transformers is 4.55.0
      there, 4.56.2 here).
- [ ] Same dataset path, and the first decoded batch is FineWeb-Edu text.
- [x] AdamW is the same (checked in code 2026-11-06): default betas (0.9, 0.999) and
      eps 1e-8, weight decay 0.1 on all parameters including norms and embeddings.
      The only difference is `fused=True` here.
- [x] Same FSDP2 mixed-precision policy: `src/core/distributed_training.py` is
      byte-identical on both branches (bf16 params, fp32 reduce).

### Smoke run (does it run, not an experiment)

A short job on entropy, 4× H100, to catch crashes before the long run. Its loss curve
is not compared to anything, so a shortened schedule is fine. The overrides go in
`h16_chinchilla_smoke.yaml` next to the main config: ~100 steps, `eval_interval: 50`,
downstream `limit: 20`, a scratch save path, a 30-minute limit, and a `smoke` tag
instead of `branch_compare`. It passes when:

- [ ] the model builds at 354.6M params and trains on 4 GPUs without OOM;
- [ ] train and eval loss are logged;
- [ ] the final checkpoint is written;
- [ ] the end-of-run downstream eval runs all nine tasks and logs
      `eval/<task>/<metric>` to W&B.

Also read off the step time; the time limit below depends on it.

### Launch

- New branch off `ctx_scl` (e.g. `jm_ctx_scl_branch_compare`), with the tokenizer
  function, the two configs and the `_trainer/llama.yaml` fix.
- entropy, 4× H100, same partition. ctx_scl's pixi env lives under
  `/storage_nvme_2/nano/$USER/pixi`, separate from this branch's, so the two don't
  collide.
- Downstream eval: `_downstream_eval/joint.yaml` already runs once at the end
  (`eval_interval: 0`, `final_eval: true`). It takes `tokenizer` from
  `common.tokenizer`, and `max_length` follows `common.sequence_length` (1024).
- Time: about 1.6e19 training FLOPs. This branch does that in ~5.3 h at 21% MFU, with
  torch.compile. ctx_scl runs eager (no compile in its trainer, `compile: false` in
  its model configs), so expect it to be slower. 16 h is enough if the smoke run shows
  ≤ 1.7 s/step (≈ 8% MFU, leaving room for evals); otherwise raise the limit.

### Comparing the results

- **Eval loss.** Both branches log periodic eval loss. The eval sets are different
  samples from the same distribution: ctx_scl samples its eval batches from the
  training directory (seed 22), while this branch uses a held-out tail of rows.
  Training covers well under 1% of the 1.27B-row corpus, so overlap with training
  data is negligible. If the curves differ by more than looks like noise, rerun one
  side with a different seed to measure the noise before drawing conclusions.
- **Benchmarks.** Score both with the same task list and `max_length`. Read any
  difference against the noise table below.

## Step 3: the eval harness on pretraining-jm

### What it is

An lm-eval wrapper (`NanoLM`) around the live nano model, ported from
`ctx_scl:src/core/eval.py`. It calls `model(input_ids)` and gets logits, so it needs
no HuggingFace export. (The old harness on `main` did need one, which is why it was
deleted in `efe41424`.)

It runs two ways, through one code path:

- **In the training loop:** every `eval_interval` steps, plus once at the end.
- **From a checkpoint:** load the weights, skip training, score.

### Pieces

1. **`src/core/eval.py`: `NanoLM` and `Evaluator`, scoring logic copied unchanged.**
   Both branches' scores then come from identical code. Improvements — sorting
   requests by length, a KV cache for generation — come later, and if they land,
   both models get re-scored.
2. **Config group `configs/_downstream_eval/`, with ctx_scl's field names:**
   `tokenizer`, `tasks`, `limit`, `max_length`, `max_gen_toks`, `batch_size`,
   `eval_interval`, `final_eval`. Configs then translate between branches 1:1. The
   attrs fields get no defaults.
3. **`main.py`:** instantiate the evaluator with `metric_logger` and `model` and pass
   it to the trainer.
4. **`trainer.py`, in-loop hook:** at `eval_interval` and after training, put the
   model in eval mode and score under `torch.compiler.set_stance("force_eager")`.
   Running eager means variable-length eval batches cause no recompiles, and eval
   forwards are short, so eager costs little. Switch back to train mode afterwards.
5. **Multi-GPU safety.**
   - Under FSDP2, every forward is a collective, so all ranks must run the same
     number of forwards.
   - lm-eval pads loglikelihood requests so every rank gets the same count
     (`lm_eval/evaluator.py:576`), so those tasks are safe.
   - `generate_until` stops each request at its own length, so generative tasks can
     deadlock. Refuse them at startup when `world_size > 1`.
6. **From-checkpoint mode.**
   - A top-level `eval_only: true` (read with `cfg.get`, like
     `compile_blocks_full`): load the weights (`load.only_weights: true`), skip
     `train()`, and run `final_eval()` plus the downstream evaluator.
   - `final_eval()` gives `final_ce` on the fixed held-out set for free.
   - Keep FSDP2 on, even on one GPU, so the existing sharded load path reads the
     checkpoint. DCP reshards across world sizes, but confirm this in testing.
   - The checkpoints are on `/storage_nvme_4`, which only the h100 nodes mount, so
     eval-only jobs must run on partition `h100`.
7. **W&B.**
   - In the loop: the training run, keys `eval/<task>/<metric>`, at the training step.
   - From a checkpoint: a new run, tagged with the source run's name, logging at the
     checkpoint's step.

### Tests (on `tiny_local.yaml`, overrides baked into a config file)

- [ ] **In the loop:** tiny model, small `eval_interval`, piqa and hellaswag with
      `limit: 20`. Scores get logged, and training continues normally afterwards.
- [ ] **From a checkpoint:** save the tiny model, run eval-only on it. The scores
      must be identical to that model's in-loop final eval: same weights and same
      requests give the same numbers. This is the check that the load path is right.
- [ ] **Guard:** a generative task with `world_size > 1` fails at startup with a
      clear message.
- [ ] *(Optional)* `NanoLM` on gpt2 vs lm-eval's own HF backend, on piqa with a
      small `limit`. The numbers should be close, which validates the scoring
      itself.

### Task list

Use ctx_scl's `joint.yaml` list unchanged on both branches. All of these are
loglikelihood tasks, so they are safe on multiple GPUs: arc_easy, arc_challenge,
hellaswag, piqa, winogrande, openbookqa, sciq, social_iqa, lambada_standard. That's
~77k requests, a few GPU-minutes per model.

At this scale, hellaswag, piqa, arc_easy, sciq and lambada carry signal. The rest
will sit near chance.

### Benchmark noise

One standard error, from test-set size. Two runs differ meaningfully only beyond
about 2.8× this.

| Task | ±1σ | Needed to call two runs different |
|---|---|---|
| hellaswag | 0.5 pt | ~1.3 pt |
| lambada | 0.7 pt | ~1.9 pt |
| arc_easy | 1.0 pt | ~2.8 pt |
| piqa | 1.1 pt | ~3.1 pt |
| sciq | 1.4 pt | ~3.9 pt |
| winogrande | 1.4 pt | ~4.0 pt |

## Order

1. Step 2 first. It's the long run, and its task list has to be fixed before launch.
2. Build step 3 while both runs train.
3. Score this branch's final checkpoint with the from-checkpoint mode, then compare
   with ctx_scl's in-run scores and both eval-loss curves.
