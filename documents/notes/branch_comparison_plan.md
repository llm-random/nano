# Branch comparison: pretraining-jm vs ctx_scl, and the eval harness

2026-11-05. Step 1 is running; this note covers steps 2 and 3. Step 2 revised
2026-11-06 after reading ctx_scl's code; step 3 revised the same day after a review
against this branch's trainer and lm-eval 0.4.11.

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

**Trainer.** Written in full in the config, with no `_trainer` default, so the
default configs stay untouched. Neither existing file fits:

- `_trainer/llama.yaml` has step 1's settings, but it carries two stale keys,
  `downstream_eval_interval` and `final_downstream_eval`. They stopped being Trainer
  fields when the evaluator took over its own cadence, and attrs rejects unknown keys.
- `_trainer/context_scaling.yaml` brings WSD, eval every 1000 steps and
  `fixed_eval: true`, and its scheduler's `decay_steps` key can't be removed by an
  override.

The block matches step 1: cosine to 10%, 1% warmup, eval every 100 steps × 10 batches,
`fixed_eval: false`. It also puts `n_steps: ${trainer.n_steps}` inside the scheduler
block, because ctx_scl's `main.py` passes the scheduler only `optimizer=`.

Set as in step 1: `sequence_length: 1024`, `batch_size: 256`, `n_steps: 30_000`,
`learning_rate: 10` (same 2^-n convention), `gradient_accumulation_steps: 1`.

**Rest.**
- `heavy_metrics_calculation_interval: 100`. It is `???` in ctx_scl's
  `_infrastructure/default.yaml`, and 100 here.
- `trainer.checkpoint.save.path: ${oc.env:PROJECT_HOME_PATH,<fallback>}/checkpoints`,
  which is `/storage_nvme_4/nano/pretraining/<user>/ctx_scl/checkpoints` in the job. `_checkpoints/none`
  leaves the path `null`, and then nothing is saved. The fallback is needed because
  ctx_scl's `run_exp.py` resolves the config on the laptop for its summary. Don't copy
  the path from `_checkpoints/context_scaling/shared_entropy.yaml`: nothing expands its
  `$USER`.
- `slurm.gres: gpu:4`, `slurm.time: "16:00:00"`, W&B tag `branch_compare`.
- Leave `max_concurrent_jobs` unset. Step 1's `-1` would become `--array=0-0%-1` in
  ctx_scl's sbatch builder.
- `infrastructure.script` is replaced in full (a list can't be patched). This moves
  ctx_scl from `/storage_nvme_2/nano/$USER` to storage_nvme_4, next to this branch:
  - `PROJECT_HOME_PATH=/storage_nvme_4/nano/pretraining/$USER/ctx_scl`.
  - Its own pixi env at `$PROJECT_HOME_PATH/pixi`. This branch's is at `$USER/pixi`,
    so the two don't collide.
  - This branch's HF cache, `/storage_nvme_4/nano/pretraining/hf_cache`, now shared
    by both.
  - It keeps `HF_DATASETS_TRUST_REMOTE_CODE=1`. This branch's entropy config now sets
    it too.

### Checks before launching (cheap, local)

Done 2026-11-06, by building each branch's model, scheduler and tokenizer from its
own config, the way its `main.py` does, in one env (torch 2.7.1, CPU).

- [x] Parameter count matches: 354,714,624 on both sides (the table's 354.6M is
      slightly off). All 147 `state_dict` tensors have the same shapes. The only key
      difference is the embedding: `embedding.embedding.weight` there,
      `embedding.weight` here.
- [x] Initial weights are bit-identical, all 147 tensors. Both branches seed with
      `torch.manual_seed(train_dataloader.dataset.seed)` (123) right before
      `instantiate(cfg.model)` and build modules in the same order.
- [x] The learning rate matches at steps 0, 1, 150, 299, 300, 301, 15,000, 29,998 and
      29,999 (9.77e-5 at step 0, peak 9.77e-4 from step 300).
- [x] Same tokenizer ids for a sample string, under transformers 4.56.2 (here) and
      4.55.0 (ctx_scl's lock).
- [x] Same dataset path in the resolved configs.
- [ ] The first decoded batch is FineWeb-Edu text. The data is cluster-only, and
      ctx_scl logs no batch text (only eval-batch fingerprints), so the smoke run
      can't show this without new code. The path and tokenizer above are already
      verified; a cluster-side spot check is optional.
- [x] The config's trainer and evaluator keys match ctx_scl's `Trainer` and
      `Evaluator` attrs fields exactly: no unknown keys and nothing required missing.
- [x] AdamW is the same (checked in code 2026-11-06): default betas (0.9, 0.999) and
      eps 1e-8, weight decay 0.1 on all parameters including norms and embeddings.
      The only difference is `fused=True` here.
- [x] Same FSDP2 mixed-precision policy: `src/core/distributed_training.py` is
      byte-identical on both branches (bf16 params, fp32 reduce).

### Smoke run (does it run, not an experiment)

A short job on entropy, 4× H100, to catch crashes before the long run. Its loss curve
is not compared to anything, so a shortened schedule is fine.
`configs/context_scaling/branch_compare/h16_chinchilla_smoke.yaml` inherits the main
config and overrides: 100 steps, `eval_interval: 50`, downstream `limit: 20`, saving
under `checkpoints/smoke`, a 30-minute limit, and a `smoke` tag instead of
`branch_compare`. Its dry run composes cleanly (26M tokens, final checkpoint at step
99). Launch with
`run_exp.py --config-path configs --config-name context_scaling/branch_compare/h16_chinchilla_smoke`.
It passes when:

- [ ] the model builds at 354,714,624 params (logged at startup) and trains on 4 GPUs
      without OOM;
- [ ] train and eval loss are logged;
- [ ] the final checkpoint is written;
- [ ] the end-of-run downstream eval runs all nine tasks and logs
      `eval/<task>/<metric>` to W&B.

Also read off the step time; the time limit below depends on it.

ctx_scl runs the downstream eval *before* the final save, so in the long run an
lm-eval crash at the end loses the checkpoint. The smoke run is the guard against
that. It runs the same code path, just with `limit: 20`. The full-size difference
that could matter is fine: ctx_scl passes no `bootstrap_iters`, so rank 0 runs
lm-eval's default 100k-draw bootstrap for lambada's perplexity while the other ranks
wait in the save's collectives. That took ~16 s on 8 laptop cores, against a 10-min
NCCL timeout.

The smoke run also downloads the nine datasets into the shared HF cache
(`/storage_nvme_4/nano/pretraining/hf_cache`; `limit` doesn't shrink the download).
That warms the cache for the long run and for step 3's eval-only job. Run it first.
On a cold cache its four ranks load at once; if that breaks, rerun it.

### Launch

- Branch `jm_ctx_scl_branch_compare` off `ctx_scl` (worktree
  `/home/janek/Documents/IDEAS/ctx_scl_compare`), with the tokenizer function and
  `configs/context_scaling/branch_compare/h16_chinchilla.yaml`. Launch with
  `run_exp.py --config-path configs --config-name context_scaling/branch_compare/h16_chinchilla`.
  The config is `# @package _global_` with absolute group refs, like ctx_scl's
  workshop configs.
- entropy, 4× H100, same partition. ctx_scl's pixi env is new, at
  `/storage_nvme_4/nano/pretraining/$USER/ctx_scl/pixi`. Install it once before the
  smoke run with
  `update_pixi.py --config-path configs --config-name context_scaling/branch_compare/h16_chinchilla`,
  which reads `PIXI_HOME` from the config's script.
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

## Step 3: benchmark scoring on pretraining-jm

### What NanoLM is

lm-eval (EleutherAI's benchmark harness) talks to a model through a backend class
with three methods: `loglikelihood`, `loglikelihood_rolling` and `generate_until`.
Its built-in backends (`hf`, `vllm`) need a model in HuggingFace format.

`NanoLM` is such a backend for nano models. It tokenizes with the HF tokenizer, calls
`model(input_ids)` to get logits, and sums the log-probs of the answer tokens. So it
needs no HF export. ctx_scl has it in `src/core/eval.py` (~200 lines), together with
`Evaluator`, which runs lm-eval with it and logs the scores to W&B. The old harness
on `main` needed an HF export, which is why it was deleted in `efe41424`.

### Now: score a checkpoint

Step 1 is already running without any harness, so its model can only be scored from
its checkpoint. That is all the comparison needs, so it is all that gets built now.

1. **`src/core/eval.py`:** `NanoLM` and `Evaluator` copied from ctx_scl, with the
   scoring unchanged, so both branches' scores come from identical code. Changes:
   - Skip non-numeric entries when logging (lm-eval's `alias` string).
   - Drop the `eval_results.json` dump (it goes to the working directory).
   - Pass `bootstrap_iters=1000` to `simple_evaluate`. Scoring is spread over all
     GPUs, but rank 0 alone computes the stderrs afterwards, and lambada's
     perplexity stderr defaults to a 100k-draw bootstrap that keeps the other
     ranks waiting. Scores don't change. Not 0, which drops every stderr.
2. **Config group `configs/_downstream_eval/`**, for this branch's models only:
   `tokenizer`, `tasks`, `limit`, `max_length`, `max_gen_toks`, `batch_size`. The
   attrs fields get no defaults. ctx_scl scores its own model in its own run, so
   the configs don't need to translate between branches. What must match is the
   scoring logic (item 1), the task list and `max_length`.
3. **Eval-only mode in `main.py`:** a top-level `eval_only: true` (read with
   `cfg.get`, like `compile_blocks_full`). It builds the model, loads the weights
   (`load.only_weights: true`), skips `train()`, and runs `final_eval()` and then
   the evaluator.
   - Assert `load.path` is set. `load_checkpoint_from_file` returns silently on
     `null`, which would score a random init.
   - **`final_ce` must match the source run's logged `final_ce`** (to ~1e-3). This
     is the load check. Settings outside the state_dict (`rope_base`,
     `rope_scale_freqs`, norm eps) still load cleanly when they're wrong, and only
     a loss check catches that.
   - Keep FSDP2 on, even on one GPU, so the existing sharded load path reads the
     checkpoint. DCP reshards across world sizes, but confirm this in testing.
   - Run on partition `h100`: the checkpoints are on `/storage_nvme_4`, which only
     the h100 nodes mount.
   - W&B: a new run (`new_wandb_job: true`, or the logger resumes the source run,
     whose id the checkpoint carries), tagged with the source run's name, logging
     `eval/<task>/<metric>` at the checkpoint's step.
4. **Multi-GPU safety.** Under FSDP2 every forward is a collective, so all ranks
   must run the same number of forwards. lm-eval pads loglikelihood requests so
   every rank gets the same count (`lm_eval/evaluator.py:576`), so the nine tasks
   are safe. `generate_until` stops each request at its own length and can
   deadlock, so refuse generative tasks at startup when `world_size > 1`.

**Tests** (on `tiny_local.yaml`, overrides baked into a config file):

- [ ] **Load path:** train the tiny model and save it, then run eval-only on it. Its
      `final_ce` matches the training run's.
- [ ] **Scoring:** piqa and hellaswag with `limit: 20` log `eval/<task>/<metric>`.
- [ ] **Guard:** a generative task with `world_size > 1` fails at startup with a
      clear message.
- [ ] *(Optional)* `NanoLM` on gpt2 vs lm-eval's own HF backend, on piqa with a
      small `limit`. The numbers should be close, which validates the scoring
      itself. Required if absolute scores are ever quoted: `NanoLM` tokenizes
      context and continuation separately, while the HF backend tokenizes the joined
      string and splits it.

### Later: not needed for this comparison

- **In-loop eval:** score every `eval_interval` steps and at the end of training,
  for future runs. When it's built:
  - Add `eval_interval` and `final_eval` to the config group.
  - Score under `torch.compiler.set_stance("force_eager")`: no recompiles on
    variable-length batches, and eval forwards are short.
  - Save and restore the RNG state (torch CPU and CUDA, Python `random`, numpy).
    `simple_evaluate` reseeds all three, which would reset the training RNG mid-run.
  - Switch back to train mode afterwards.
  - Reset `_prev_log_time` afterwards. Otherwise the next step logs minutes of step
    time and near-zero MFU.
  - At the end of the run: `final_eval()` → `save_final_checkpoint()` → downstream
    eval, so an lm-eval crash can't cost the checkpoint.
  - Test: train loss after an eval is bit-identical to the same run without one.
- **Faster scoring:** sort requests by length, KV cache for generation. If these
  land, re-score both models.
- **Generative tasks on several GPUs:** every rank keeps stepping until all ranks
  are done.
- **HF export, for SFT in coding_distill:** recover the converter removed in
  `6b2b5421` and check that the HF model's logits match nano's. Then
  from-checkpoint eval can also use lm-eval's `vllm` backend, which splits work
  across GPUs, sorts by length, has a KV cache and handles generative tasks. Its
  numbers differ from NanoLM's, so re-score both sides of any comparison the same
  way.

### Task list

Use ctx_scl's `joint.yaml` list unchanged on both branches. All of these are
loglikelihood tasks, so they are safe on multiple GPUs: arc_easy, arc_challenge,
hellaswag, piqa, winogrande, openbookqa, sciq, social_iqa, lambada_standard. That's
~77k requests, a few GPU-minutes per model.

Download the nine datasets into a shared HF cache once, before the long runs.
Otherwise every rank calls `load_dataset` on the cache at the same time.

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
2. Build step 3's "Now" part while both runs train.
3. Score this branch's final checkpoint with eval-only mode, then compare with
   ctx_scl's in-run scores and both eval-loss curves.
4. Step 3's "Later" part, when a future run needs it.
