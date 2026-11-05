# AME optimization — what pkmf's static-memory stack does that `ame_optimize` does not

**Status: historical — a closed survey, kept for the evidence.** Written 2026-11-04
against branch `ame_optimize` (HEAD `f302ae12`) and the `pkmf/` drop at `nano/pkmf`
(gitignored, read-only reference). A survey of a frozen external codebase: nothing will
be added to it. Archived 2026-11-05.

**"Nothing here is implemented" was true on 2026-11-04 and is not true now.** O1–O9, O11
and O33 were implemented and measured on 2026-11-05. **§1.0 of
[ame_optimization_report.md](../ame_optimization_report.md) is the current index** — it
names every O-number with its state and measured effect, and adds O32–O37, which are not
in this document.

**What is still only here:** the *evidence* behind each item — pkmf's call sites, why it
works, its parameters — which the report's one-line inventory does not carry. That is
what you need to implement one of the untried items. Also **O22–O26**, which the report
omits because they are not MFU work at all: GPU-count-invariant schedule sizing, variable
gradient accumulation, granular micro-batching, token-space WSD and Step-Law LR. Those
matter for the training program in [ame_experiments.md](../ame_experiments.md), not for
throughput.

---

## 0. Scope, and what the `pkmf/` drop actually contains

`pkmf/` is the **training framework** (`src/poirot/`, imported as `pkmf.*`) plus its
Hydra configs, run notebooks and benchmark scripts. What it does **not** contain:

| Missing | Why it matters |
|---|---|
| `pkmf.memory.*` — `MemoryTree`, `NdEmbedding`, routers, `ultramem_fused` CUDA/Triton kernels | The fused gather/weighted-sum kernel and the vertical-parallel lookup itself live there. We can read its *contract* (call sites, docstrings, tests, bench scripts) but not its source. |
| `pkmf.models.jmsl_moe.initialization` | Only `get_init_weight`; irrelevant. |

So: every **framework-level** item below is verifiable line by line; the two
**kernel-level** items (O14, O15) are reconstructed from call sites and must be
re-derived or re-implemented, not copied.

Sources read: `pkmf/running/running_static_Hx.ipynb`,
`pkmf/running/ablation_static_gpus_microbatch.ipynb`, `pkmf/scripts/conf/**`,
`pkmf/src/poirot/{memory_parallel,memory_parallel_groups,trainer,callbacks,batch_schedule,variable_ga,granular_batch_sampler,lr_scaling}.py`,
`pkmf/src/poirot/models/{qwen3_stem,qwen3_memory}/**`.

---

## 1. How pkmf actually runs a STEM model (not the obvious way)

There are **two** STEM implementations in the drop and the notebooks use the second one:

1. `models/qwen3_stem/` — the clean standalone port. One `NdEmbedding((vocab,), (memory_size,))`
   per stem layer, gathered by `input_ids`. **Not** what the real runs use.
2. `models/qwen3_memory/` + `routing=token_id` — the production path. Same math, but it
   reuses the whole PKM stack: `StaticRouter(bshape=[vocab_size], depth=1)` feeds the
   same `MemoryTree` / `ultramem_fused` kernel, the same `MemoryParallelController`,
   the same vertical sharding, the same checkpoint format.

Every run command in `running_static_Hx.ipynb` is of the form:

```
pixi run python scripts/pretrain_memory.py -m \
    config=memory_sl/pkm_depth/flatten_h20 routing=token_id \
    accelerate=zero0 hydra.launcher.g=4 mem_parallel_size=4 \
    train.batch_rampup.enabled=true \
    train.batch_rampup.microbatch_per_gpu=8 train.batch_rampup.reference_num_gpus=8 \
    train.total_batch_size=null train.gradient_accumulation_steps=null ...
```

**The lesson for us:** they did not build a separate optimized path for the
router-less model. They made the static model *a routing overlay on the optimized
memory path*, so it inherits every efficiency mechanism for free. That is the shape
the AME work should take too — AME's banks should enter whatever bank-handling
machinery we build as one more "static, zero-parameter router" case, not as a
bespoke AME code path.

`pkmf/scripts/conf/routing/token_id.yaml` is the whole overlay: it clears `bshape`,
`depth`, `router_kwargs`, pins `value_expand_time: 1`, `memory_init_std: null`
(forces Xavier), `memory_size: null` (full FFN width), and sets placement
`memory_interval: 2`, `memory_offset: 1`.

---

## 2. Memory banks: STEM vs AME

| | pkmf STEM (token_id) | AME (`src/core/ame.py`) |
|---|---|---|
| Indexed by | token id `t_j` | token id `t_j` |
| Learnable router | none | none |
| Tables per layer | **1** (`memory_tree.embeddings.0.weight`) | **2** (`q_bank.weight`, `k_bank.weight`) |
| Table shape | `[vocab, memory_size]`, `memory_size = intermediate_size = 4·d_model` | `[vocab, q_heads·d_head] = [50304, d_model]` each |
| Layer placement | every 2nd layer, excluding first and last: `interval=2, offset=1` → `[2,4,…,L-2]` (9 of 20 at H20) | **every** layer (`attention.type: ame`), 20 of 20 at H20 |
| Rows read per token | 1 row (depth-1, top-1 by construction) | 2 rows (one per bank) |
| Where the row enters | `down_proj(act(gate_proj(x)) ⊙ mem)` — multiplicative FFN gate | `q = E_Q[t]`, `k = E_K[t]` — attention operands |
| Row normalization | optional `memory_norm: rmsnorm` (default `none`) | ℓ2 QK-norm on rows, before the KV-shift mix (load-bearing) |
| Init | Xavier normal over `[vocab, memory_size]` (`std = sqrt(2/(V+M))`) | per-element `σ = 1.0` (`bank_init_std`) |
| Bank LR | **4× backbone** (`mem_value_lr_times`) | 1× backbone |
| Bank weight decay | **0** | 0 (`no_weight_decay_parameters`, decided 2026-11-03) |
| Gradient path | hand-managed fp32 `main_grad`, hidden from the framework | ordinary dense `nn.Embedding` backward inside FSDP2 |
| Sharding | vertical (column/`vdim`) across `mem_parallel_size` ranks | FSDP2 flat-shards everything uniformly |
| Bank params at H20 | 9 × ~50k × 5120 ≈ **2.3B** (NeoX-20b vocab, `ff_mult=4`) | 2 × 20 × 50304 × 1280 = **2.576B** on a 587M backbone |

**Two structural differences that change what transfers:**

- AME has **2 banks per layer in every layer**, STEM has **1 table in half the layers**.
  Same order of magnitude, but AME's gather count per token is 2L vs STEM's L/2 —
  40 gathers/token vs 9 at H20.
- STEM's row is consumed **element-wise** by a gate at FFN width. AME's row is consumed
  as an **attention operand** at `d_head` granularity per head, which is exactly why
  pkmf's vertical (column) split maps onto AME cleanly: a column slice of
  `[vocab, H·D]` is a *head subset*, and heads are independent until `o_proj`.

---

## 3. The feature list

IDs (`O1`…) are for referencing in later work. "Present in AME" = what
`ame_optimize` has today.

### A. Bank gradient + optimizer path — the main event

**O1. Take the banks out of the framework's gradient path entirely.**
`pkmf/src/poirot/memory_parallel.py:696` `MemoryParallelController`. For every table param:
set `p.requires_grad = False` so DeepSpeed/DDP never creates a master copy, never
reduces it and never steps it; attach a **persistent fp32 `p.main_grad`** that the
fused kernel `atomicAdd`s into; keep a **private fp32 master** that a dedicated AdamW
steps; copy master → bf16 model param after each step.
*Present in AME:* no. `nn.Embedding` backward allocates a dense `[50304, 1280]` grad per
bank per layer — 40 of them, ~10.3 GB — on **every** micro-batch.
*Why it wins:* no per-step `zeros_like` of a 2.6B-param grad buffer, and the accumulator
accumulates across grad-accum micro-steps for free (relevant: AME runs GA=128).
*Measured by pkmf:* **1.43×** end-to-end on 1 GPU, 1M-row bank, A100 (33 → 47 k tok/s).

**O2. Manual all-reduce at the optimizer-step boundary only.**
`memory_parallel.py:1024` `step()`. One `all_reduce(SUM)` of `main_grad` per optimizer
step, then `div_(world_size)` to keep DeepSpeed-style mean semantics.
*Present in AME:* no — and worse, see O9.

**O3. fp32 master / bf16 forward-copy split for the banks.**
The kernel reads the bf16 param; AdamW steps an fp32 master; `p.data.copy_(master)`
after the step. Mixed-precision without the framework's machinery.
*Present in AME:* FSDP2's `MixedPrecisionPolicy(bf16, fp32)` does the equivalent, but
for *all* 3.16B params uniformly and inside the sharding hooks.

**O4. Gradient clipping over the masters, separately.**
`memory_parallel.py` clips `self._masters` with its own `max_grad_norm` because the
tables are invisible to the trainer's clip.
*Present in AME:* one global clip over everything (`src/core/trainer.py:649`), which is
the *correct* baseline semantics — note that splitting the clip **changes the math** and
must be a deliberate choice, not a side effect.

**O5. Driven by a callback at exact hook points.**
`pkmf/src/poirot/callbacks.py:3347` `MemoryParallelCallback`:
`on_step_begin` → `refresh_shadow()` (re-home master/`main_grad` onto the param's device),
`on_pre_optimizer_step` → `controller.step(global_step, scheduled_base_lr=…)`,
`on_log` → surface `mem_value_lr`.
*Present in AME:* nano's `Trainer` has no callback system; this would be explicit calls
in `calculate_loss` / the step loop.

### B. Vertical memory parallelism (the multi-GPU item)

**O6. Two named process groups, built once.**
`pkmf/src/poirot/memory_parallel_groups.py:43` `init_memory_parallel_group(mem_parallel_size)`:
- **memory-parallel group** — contiguous blocks of `mp` ranks. Every table's `vdim` is
  **column-sliced** across them: each rank owns `[rows, vdim // mp]`.
- **memory-DP subgroup** — strided, size `world // mp`. Table `main_grad` is all-reduced
  **only within this subgroup**, cutting bank comm volume by `mp`.

```
world=8, mp=4:  mp-groups [0,1,2,3] [4,5,6,7]   dp-groups [0,4] [1,5] [2,6] [3,7]
```
`mem_parallel_size == 1` ⇒ every helper is identity and every accessor returns `None`.

**O7. The vertical-parallel forward: all-gather indices, local lookup, all-to-all outputs.**
Each rank all-gathers the routed indices from its mp-group, runs the local `FusedLookup`
on its `vdim` slice for *all* mp ranks' tokens, then all-to-alls the embedding output so
each rank ends with its own tokens at full `vdim`.
Autograd wrappers, all in `memory_parallel_groups.py`:
`_GatherFromMemoryParallelRegion` (`:192`, all-gather fwd / reduce-scatter bwd),
`AllReduceInMemGroup` (`:209`, all-reduce on both passes),
`EmbeddingAll2AllSingle` (`:229`, all-to-all-single with recorded splits, swapped on bwd).
*Effect:* per-rank bank VRAM **÷ mp**, bank comm **÷ mp**.
*Present in AME:* no. FSDP2 flat-shards the banks but then **all-gathers each full bank
per layer per step** — which `ame.md §7` explicitly warned against ("shard them the way
Megatron shards a vocab-parallel embedding … rather than letting ZeRO-3 all-gather a
whole bank per layer per step"). Note pkmf chose the **column** split, `ame.md` suggested
the **row** split; for AME the column split is a head split and is the cheaper one
because no rank then needs a row it does not own.

**O8. The grad-divisor subtlety (easy to get silently wrong).**
Under VS the all-reduce runs in the mem-DP subgroup (size `world/mp`) but the divisor
stays the **global** `world_size`, because the forward's index all-gather already summed
`mp` ranks' tokens into each slice's local gradient. `mp × (world/mp) = world`. Get this
wrong and the update magnitude silently differs from the 1-GPU baseline.
Documented at `memory_parallel.py:1024` docstring.

### C. Framework-level wins AME is missing regardless of banks

**O9. Suppress gradient sync during accumulation.**
`src/core/trainer.py:549` `calculate_loss` loops `loss.backward()` over
`gradient_accumulation_steps` chunks with **no** `set_requires_gradient_sync(False)`.
At `configs/ame_tests/h20_smoke.yaml`'s `gradient_accumulation_steps: 128` that is **128 full
FSDP2 reduce-scatters of 3.16B parameters per optimizer step**, each with a bf16→fp32
cast. Already named as candidate 3 in `ame_implementation.md`.
*This is the single cheapest fix in this document and it is framework-wide, not
AME-specific.* Correct shape: sync off for micro-steps `0..GA-2`, on for the last.

**O10. Activation checkpointing.**
`pkmf/scripts/conf/pretrain_dense.yaml`: `gradient_checkpointing: true`,
`gradient_checkpointing_kwargs: {use_reentrant: false}` — on by default for every pkmf run.
*Present in AME:* **nano has none at all** (no `torch.utils.checkpoint` anywhere in
`src/core/`). This is the direct lever on AME's stated blocker — micro-batch 4 vs the
baseline's 16, "~57 GiB … leaves ~23 GiB, which will not hold 16". Costs ~30% recompute,
buys the micro-batch. Given AME is **15% cheaper in FLOPs/token** than its dense twin and
is memory-bound, this trade is probably favourable; it is measurable in one run.

**O11. Liger fused-linear-cross-entropy.**
`use_liger_kernel: true` in `pretrain_dense.yaml`;
`pkmf/src/poirot/models/qwen3_stem/utils.py:6` applies `rope`, `rms_norm`,
`fused_linear_cross_entropy=True` and deliberately **not** `swiglu` (custom MLP).
FLCE never materializes the `[B, L, V]` logit tensor.
*Present in AME:* no. `src/core/trainer.py:531` `_cross_entropy` takes full logits —
at micro-batch 4 × 2048 × 50304 that is ~0.8 GB bf16 plus the fp32 softmax, per
micro-step. `compile_ce` exists but was **measured harmful**
(`multigpu_efficiency.md 7.9`), which is not the same mechanism as FLCE.

**O12. Fused SiLU-mul on the memory product.**
`modeling_qwen3_memory.py:598` uses `LigerSiLUMulFunction.apply(gate_preact, memories)`
and keeps the gate as a pre-activation until something actually needs it.
*Present in AME:* n/a (no gated MLP in the attention layer), but the pattern —
*don't materialize an intermediate the fused op can take raw* — applies to
`_mixed_qkv`'s normalize → shift → scale chain, which currently materializes
`q`, `k`, `_shift1(k)`, `_shift1(v)` and the scaled `q` separately.

**O13. `overlap_comm` is a trap with vertical sharding.**
`pretrain_memory.yaml:136` — "keep overlap_comm off (accelerate=zero0, the default)
with VS: the profiled grid shows overlap halves VS throughput (215k → 123k on 1M)".
*Relevant to AME* because the instinct with FSDP2 is to turn on more overlap; measure it.

### D. Kernel-level (reconstruct, don't copy — source not in the drop)

**O14. One fused gather + weighted-reduce kernel that scatters straight into `main_grad`.**
`mix_method: ultramem_fused` (default). Options enumerated at
`pretrain_memory.yaml:46`: `ultramem_fused` (CUDA, dv-tiled) | `triton` | `fp32`
(`F.embedding_bag`) | `manual_seq` (checkpointed K-loop) | `manual_vec` | `compiled` |
`ultramem`. The backward writes into the persistent fp32 `main_grad` with `atomicAdd`
and **never zeros it** — that is what makes O1 free across grad-accum.
`pkmf/scripts/bench_fused_lookup_tiled.py` and `scripts/test_fused_lookup_tiled.py`
show the dispatch: `dv=768` → original WMMA kernel, `dv ∈ {3072, 6144}` → dv-tiled kernel.
Tree widths are forced to multiples of 8 (`align=8`) because the kernel asserts
`each_core_dim % 8 == 0`.
*Present in AME:* no custom kernel. AME's gather is a plain `nn.Embedding` forward; the
backward is `embedding_dense_backward`. A fused "gather 2 rows, normalize, mix with the
shifted row, scale by τ" op that scatters into a persistent fp32 accumulator is the AME
analogue and would subsume O1's accumulator.

**O15. Deduplicate the gather by unique token id.**
`modeling_qwen3_stem.py:108` `get_memories`: when the table is CPU-offloaded, take
`input_ids.unique(return_inverse=True)`, gather only the unique rows across the slow
link, then expand on-device. Only used on the offload path in pkmf, but the arithmetic
is general: a 4×2048 micro-batch over a 50k vocab has ~6k unique ids, so the dedup is
~1.4× fewer row reads even on-device, and more at larger micro-batch.
*Present in AME:* no. AME gathers `B·n` rows per bank per layer — 8192 rows × 40 banks
per micro-step at micro-batch 4.

### E. Optimizer variants for very large banks

All in `memory_parallel.py`, selected by `memory_optimizer:` (default `chunked_adamw`).

**O16. `ChunkedAdamW` (`:248`).** Same AdamW state layout and elementwise math, but walks
each parameter as flat chunks bounded by `max_chunk_bytes` (default **128 MiB**,
`memory_optimizer_chunk_size_mb`). Forces `foreach=False, fused=False`.
*Why:* PyTorch's default foreach CUDA AdamW materializes a denominator tensor for every
param in the foreach group at once — **this OOMed on A100 at `mem_parallel_size=2`
before the first step**.
*Present in AME:* `main.py:208` builds `torch.optim.AdamW(..., fused=torch.cuda.is_available())`
over all params including the 2.576B of banks. Different failure mode from pkmf's
(fused ≠ foreach) but the same class of risk at this parameter count.

**O17. `OffloadedChunkedAdamW` (`:379`).** Keeps the two fp32 moments on **CPU**; per
bounded chunk, uploads moments, applies the identical pointwise update, downloads.
Removes two full-table fp32 tensors from persistent GPU memory.
Sub-knobs: `memory_optimizer_pin_cpu_state` (pinned host memory),
`memory_optimizer_gpu_first_moment` (keep `exp_avg` on GPU, offload only `exp_avg_sq` —
halves transfer, costs one full-table fp32 GPU buffer),
`memory_optimizer_async_offload` (two CUDA copy streams double-buffering transfers
against the update; the local sweep favoured **16 MiB** chunks for this path).
Restricted to `benchmark_no_save=true` in pkmf — sharded checkpoint I/O for the CPU
states is not validated.
*Present in AME:* no. Directly relevant: AME's ~57 GiB budget is dominated by exactly
these two fp32 moments over 3.16B params (~25 GiB).

**O18. `SelectiveAdamW` (`:146`) — correctness, not speed.** Masks on `grad != 0` so a row
that saw no gradient this step has **both moments and the weight frozen**, instead of its
`exp_avg` decaying by `(1-β₁)` every step. Without it, a row revisited after N idle steps
has lost its momentum to `β₁^N`.
`mem_selective_updates: false` by default; pkmf is explicit that it is **slightly slower**
than plain AdamW (full-table bool mask + extra pointwise ops).
*Relevant to AME:* AME's banks have exactly this structure, and `ame.md §7` predicted the
coverage numbers (50–82% of rows per 0.5M-token batch at realistic Zipf). Worth an A/B on
*loss*, not on speed.

### F. Per-group LR and weight decay

**O19. Pattern-matched param groups with per-group `lr_mult` and `weight_decay`.**
`pkmf/src/poirot/trainer.py:833` `build_param_groups` + `conf/param_groups/*.yaml`.
Five fixed slots (`param_memory`, `param_router`, `param_group_1..3`), each
`{name, name_patterns, lr_mult, weight_decay}`, matched by **substring on qualified
parameter name**, slot order = priority, unmatched → implicit `default` group.
HF's bias/norm wd-exclusion is layered *inside* each group, so `specs=None` is
byte-identical to stock.
The STEM spec (`conf/param_groups/qwen3_stem.yaml`) is exactly:
`param_memory: {name_patterns: [".embedding.weight"], lr_mult: 4.0, weight_decay: 0.0}`,
`param_router: null`.

**O20. Memory values train at 4× the backbone LR, with wd=0.**
`mem_value_lr_times: 4.0` (`mem_top_value_lr_times: 1.0` for hierarchical tables, inert
at depth 1). Citations given in the config: Lample 2019 used 4× and reported 4–10× stable;
UltraMem/V2 keep 4×; Engram uses 5× with wd=0 ("memorization is the goal, decay fights it").
Under `memory_parallel=true` the multiplier is applied to the **trainer's currently
scheduled** base LR (`memory_parallel.py:907` `_scheduled_lr`), so the bank LR tracks the
WSD/cosine envelope exactly: `memory_lr(t) = scheduled_model_lr(t) × times`.
Optional `mem_value_lr_max_steps_rate` decays the *multiplier itself* 4 → 1 over
`max_steps × rate` (null = fixed multiplier; upstream's default is 1.0).
*Present in AME:* `src/core/model.py:436` `parameter_groups` splits **only** on weight
decay (two groups: `wd` and `0.0`). There is no LR multiplier mechanism at all, and
`src/core/schedulers.py:70` currently asserts a single LR across groups.
AME's banks run at **1×**. If the 4× result transfers, this is a pure quality win
costing nothing in throughput — and it is a prerequisite for a fair AME-vs-dense
comparison, since AME's banks replace `W_Q`/`W_K`, which *are* trained at 1×.
**Open question worth stating in the paper:** STEM's 4× applies to a multiplicative FFN
gate; AME's rows are attention operands under ℓ2 QK-norm, where row *length* does not
reach the score. The 4× may not transfer unchanged.

### G. Batch / schedule machinery (wall-clock efficiency, not MFU)

**O21. Continuous token-driven batch ramp-up.** `pkmf/src/poirot/batch_schedule.py`.
`B(D) = coeff · D^exponent` tokens, clipped to `[align, B_max]`, snapped up to the grid;
defaults `coeff: 1.35, exponent: 0.571`. Small batches early (sample-efficient), large
batches later (hardware-efficient). `BatchSchedule` memoizes a lazy prefix-sum table
because the naive form is O(steps²) (~140M inner iterations on a 16B-token run).

**O22. GPU-count-invariant schedule sizing.** `align = seq_len × reference_num_gpus ×
microbatch_per_gpu` is the *reference* grid; `physical_align = seq_len × WORLD_SIZE ×
microbatch_per_gpu` is what the dataloader actually does; `GA = align / physical_align`.
Hold `reference_num_gpus × microbatch_per_gpu` constant and **the identical global-batch
token trace runs on any GPU count** — which is exactly how the throughput ablation
isolates the physical split (§5).

**O23. Variable gradient accumulation under HF Trainer.**
`pkmf/src/poirot/variable_ga.py` recompiles `_inner_training_loop` from source with
guarded substitutions, because HF assumes a *constant* GA in three coupled places
(sync predicate, chunking arithmetic, resume skip). Each substitution asserts its anchor
matched exactly once, so a transformers bump fails loudly instead of training with a
broken sync boundary.
*For nano:* nano owns its loop, so this is ~10 lines rather than 400 — but the three
hazards are the same, especially **the resume offset**: converting optimizer steps to
micro-batches with a constant multiplier is wrong under a ramp (measured 122,078 vs
134,928 — a ~13k-batch data offset on a 16B run).

**O24. Granular micro-batching.** `pkmf/src/poirot/granular_batch_sampler.py`. Each
optimizer step = full local micro-batches **plus at most one smaller one**, with
proportional loss weighting, so the schedule grid can be `seq_len × reference_num_gpus`
instead of a whole micro-batch. Removes the round-up waste in O21.

**O25. Token-space WSD.** `conf/lr/wsd.yaml` + `warmup_tokens`. Under a ramp, "20% of
steps" ≠ "20% of tokens": the step-space reading put the decay phase **9.6B instead of
12.8B** on a 16B run. Warmup and the decay boundary are resolved in token space;
explicit `warmup_steps` is rejected outright.

**O26. Step-Law LR.** `pkmf/src/poirot/lr_scaling.py`:
`η(N, D) = 1.79 · N^-0.713 · D^0.307`, `D` snapped to the current batch plateau and
floored at 2B tokens; the WSD scheduler still supplies the warmup/stable/decay envelope.
Opt-in (`sl_lr: true`, requires `learning_rate: null`).

**O27. Branch-decay checkpoints.** One open-ended stable-phase parent drops retained
branch points; each seeds a pure-decay child, so **one run yields decayed models at
several token budgets**. `branch_child_total_every`, `stop_after_tokens`, `branch_from`,
`auto_decay_children: interleaved`, with an on-disk dispatcher
(`decide_next_phase`) that is preemption-safe by re-reading `branch_points.json` /
`.decay_done` / `checkpoint-N/`. Costs ~13 GB per branch point.
*Directly relevant to the AME scaling-law arm* — it is how a 1BT/2BT/4BT/6BT family
gets produced from one submission instead of four.

### H. Logging and checkpointing cost at this parameter count

**O28. Master switch + throttle for diagnostics.** `memory_diagnostics: true` (master),
`heavy_memory_logs: false`, `heavy_level0_logs: false`, `sparse_memory_logs: 500`
(compute+log only on `step % N == 0` — "N=10 costs ~10× less"), plus
`CropeLevel0MetricThrottle` (`callbacks.py:3932`), which flips a model-side flag so the
`topk`/`bincount` inside the forward runs ~1/period.
Also: at `aux_loss_coef == 0` the balance loss is computed under `torch.no_grad()`
specifically because a grad-carrying subgraph multiplied by zero is still
**retained and recomputed under gradient checkpointing — ~2× slowdown**.
*Present in AME:* `_log_layer_instrumentation` is already gated on
`_should_log_heavy_metrics` (`src/core/trainer.py:694`), which is the right shape. The
pkmf-specific gap is the *in-forward* throttle: AME's `attention_diagnostics` and the
`_last_*` detached stashes are inside the layer and pay on every step.

**O29. Periodic `gc.collect()` + `torch.cuda.empty_cache()`.**
`callbacks.py:741` `GarbageCollectionCallback(period)`.
*Present in AME:* no.

**O30. Rank-local optimizer-state checkpoints.**
`memory_parallel_checkpoint_format: local_shard` (vs `full`) writes one same-rank shard
per process under `mem_optimizer_sharded/`, avoiding a large all-gather at save time;
`full` stays reshardable across `mem_parallel_size`.
Plus `final_checkpoint_save_only_model: true` for the terminal checkpoint, with the
memory tensors patched from the fp32 masters so the artifact is rank-complete for eval.
`flatten_h20.yaml` additionally sets `preemption_checkpoint: false` because a complete
H20 checkpoint takes ~4 minutes against a ~30 s Kubernetes grace window.
*Relevant to AME:* 3.16B params is the same regime; a full-gather checkpoint of the banks
is minutes, not seconds.

**O31. Per-component and per-callback CUDA-event timing.**
`callbacks_profiling.py` hooks attention / router / memory_tree / down_proj / MLP with
CUDA event pairs over a fixed window (steps 5–55); `callback_timing.py` wraps each
callback hook and can A/B-disable callbacks by class name from Hydra.
*Present in AME:* `LoopTimers` + the kernel profiler already cover most of this; the
missing piece is per-submodule fwd/bwd attribution, which is what would actually
localize AME's unattributed 2.5× gap.

---

## 4. The numbers pkmf reports

| Change | Shape | Result |
|---|---|---|
| Hand-managed table path (O1–O3) vs DeepSpeed zero0 | flat/pkm_depth H12, 1M-row bank, 1× A100 | **1.43×** (33 → 47 k tok/s) |
| Same, 4 GPUs, **without** vertical sharding | as above | **~1.00×** — the dense all-reduce of the 1.6B table (~6.4 GB/step) dominates |
| `memory_parallel` + `vertical_parallel` (O1+O6+O7) | profiled 4-GPU / 1M-row grid | **~2.2×** (215k vs 97k dense tok/s), and matches small-bank speed |
| `overlap_comm=true` **with** VS | same grid | **0.57×** (215k → 123k) — a regression |
| SparseAdam over a row-sparse `main_grad` | production shape | 0.97× (1 GPU) / 1.02× (4 GPU) — **rejected** |

For reference, AME today (`ame_implementation.md`, entropy job 32360, h20, 1×H100):
47,391 tok/s, **18.1% MFU**, 22.1 s/step at micro-batch 4 — against the
`mfu_audit/h20_mgpu.yaml` dense baseline's 109,361 tok/s / 45.57% at micro-batch 16.

---

## 5. How they measured it — the ablation protocol worth copying

`pkmf/running/ablation_static_gpus_microbatch.ipynb`. Grid over
`(gpus, microbatch_per_gpu, reference_num_gpus)` with the invariant
**`reference_num_gpus × microbatch_per_gpu == 64`** held in every cell, so `align` is
constant and every run trains the *identical global-batch token schedule* — only the
physical split changes. 13 valid combinations per model size (the trainer requires
`reference_num_gpus % WORLD_SIZE == 0`), 5 sizes, 65 runs.

Benchmark controls, all of them: `benchmark_no_save=true`, `auto_decay_children=false`,
`train.num_tokens=0.05` (50M tokens), `preemption_checkpoint=false`,
`hydra/launcher=runai` (single phase).

Memory-parallel geometry is varied *with* the GPU count:
`g=1 → vertical_parallel=false, mem_parallel_size=1`; `g∈{2,4,8} → mem_parallel_size=g`.
So fixed-`g` comparisons isolate micro-batch/GA cleanly, and cross-`g` comparisons
measure the real intended distributed configuration including its sharding degree.

**For our 4×H100 target** that grid reduces to `g=4` with
`microbatch_per_gpu ∈ {4, 8, 16}` and `reference_num_gpus ∈ {16, 8, 4}` → GA 4/2/1.

---

## 6. What pkmf tried and explicitly rejected

Recording these so we do not spend the same time.

- **SparseAdam over a row-sparse view of `main_grad`.** Implemented and benchmarked on
  A100 (2026-07). Slower, not faster. Two reasons: at production shapes
  `nnz(main_grad) ≈ bank size` within a few steps (so it iterates ~all rows anyway)
  *plus* the per-step `to_sparse(sparse_dim=1)` scan over 1.6B fp32 elements; and even at
  1.6% coverage PyTorch's row-wise SparseAdam kernel is not well optimized. A win only at
  artificially small batches (1.14× at bs=4/seq=512, gone by bs=4/seq=1024).
  **Stated lesson: row-count sparsity does not translate to optimizer-step speed.**
  `ame.md §7` reached the same conclusion independently for AME — 50–82% row coverage per
  0.5M-token batch at realistic Zipf, union over a few steps ≈ every row, and decoupled
  weight decay touches every row regardless.
  *So: if the user's memory of "pkmf has a sparse optimizer" meant a speed win, it is the
  opposite — the sparse path is `SelectiveAdamW` (O18) and it is a **quality** mechanism.*
- **Upstream Memory+'s dense `selective_updates` mask as a speed mechanism.** Same
  reasoning — it is a correctness flag (momentum drift), still O(table).
- **`overlap_comm` with vertical sharding** (O13).

---

## 7. Parameter reference

Every pkmf knob that bears on this, with its default and the AME equivalent.

| Knob | Default | AME equivalent today |
|---|---|---|
| `memory_parallel` | `true` | — (no such path) |
| `vertical_parallel` | `true` | — |
| `mem_parallel_size` | `4` | — |
| `vp_assume_equal_token_counts` | `false` | — (skips the count all-gather when every rank has the same routed-entry count; safe for fixed-shape sync training, not for eval) |
| `mem_value_lr_times` | `4.0` | 1× (no mechanism) |
| `mem_top_value_lr_times` | `1.0` | n/a (AME has no hierarchy) |
| `mem_value_lr_max_steps_rate` | `null` (fixed multiplier) | n/a |
| `memory_optimizer` | `chunked_adamw` | `torch.optim.AdamW(fused=True)` over everything |
| `memory_optimizer_chunk_size_mb` | `128` (16 for async) | n/a |
| `memory_optimizer_pin_cpu_state` | `false` | n/a |
| `memory_optimizer_gpu_first_moment` | `false` | n/a |
| `memory_optimizer_async_offload` | `false` | n/a |
| `mem_selective_updates` | `false` | n/a |
| `mix_method` | `ultramem_fused` | plain `nn.Embedding` |
| `value_expand_time` | `1` (forced for token_id) | n/a |
| `memory_size` | `null` → `intermediate_size` | `q_heads · d_head` |
| `memory_interval` / `memory_offset` | `2` / `1` → `[2,4,…,L-2]` | `attention.layers: all`; `resolve_layer_types` already supports `every_2nd`, `every_4th` or explicit indices (`src/core/attention.py:212`) |
| `memory_init_std` | `null` → Xavier for token_id | `bank_init_std: 1.0` |
| `memory_norm` | `none` | ℓ2 QK-norm (different mechanism, not optional) |
| `memory_parallel_checkpoint_format` | `local_shard` for memory runs | `full` equivalent (DCP) |
| `use_liger_kernel` | `true` | — |
| `gradient_checkpointing` | `true` (`use_reentrant: false`) | — |
| `attn_implementation` | `flash_attention_2` | compiled `flex_attention` (required — differentiable ALiBi) |
| `accelerate` | `zero0` (not `zero1/2/3`) | FSDP2 |
| `train.batch_rampup.*` | `enabled: false`, `coeff 1.35`, `exponent 0.571` | — |
| `sparse_memory_logs` | `500` | `_should_log_heavy_metrics` gate only |

---

## 8. Suggested order, by payoff per unit of risk

Expected-value ordering, not a commitment. The first three are framework fixes that
cost little and are measurable in one run each.

| # | Item | Why first |
|---|---|---|
| 1 | **O9** — no grad sync during accumulation | ~10 lines; removes 127 of 128 reduce-scatters of 3.16B params per step. Benefits every large model in nano, not just AME. |
| 2 | **O10** — activation checkpointing | The direct lever on micro-batch 4 → 16, which is candidate 1 in `ame_implementation.md`'s unattributed 2.5× gap. AME is FLOP-cheap and memory-bound, so the recompute trade is likely favourable. |
| 3 | **O11** — fused linear cross-entropy | Removes the single largest activation in the step. Independent of everything else. |
| 4 | **O31** — per-submodule fwd/bwd CUDA-event attribution | Before building anything large, *localize* the remaining gap. Cheap, and it decides whether 5–7 are worth it. |
| 5 | **O1–O5** — hand-managed bank gradient + optimizer path | pkmf's measured 1.43× at 1 GPU. The biggest single-GPU item, and a prerequisite for 6. |
| 6 | **O6–O8** — vertical bank sharding across the 4 GPUs | pkmf's measured ~2.2× on the 4-GPU grid. This is the multi-GPU item, and it is the one that needs the most care (the divisor in O8, and the AME-specific question of splitting by head). |
| 7 | **O16/O17** — chunked, optionally CPU-offloaded bank AdamW | Attacks the ~25 GiB of fp32 Adam moments directly; composes with 6. |
| 8 | **O19/O20** — per-group LR with a bank multiplier | Quality, not speed, and free. Needs `src/core/schedulers.py` to stop assuming one LR across groups. |
| 9 | **O14/O15** — fused/deduplicated AME gather | Highest ceiling, highest cost, and must be written from scratch (source not in the drop). |
| 10 | **O21–O27** — batch ramp, token-space WSD, branch-decay | Wall-clock and experiment-throughput wins for the scaling-law arm; orthogonal to MFU. |

**Before any of this:** fix the measurement baseline.

> **Corrected 2026-11-04.** An earlier version of this line said no 4-GPU baseline
> existed. It does: `mfu_audit/h20_mgpu.yaml` carries measured 1/2/4-GPU numbers
> (109,361 / 214,499 / **425,054** tok/s at 45.57 / 44.69 / **44.28%** MFU). The real gap
> is narrower but still real — that table is the **GPT-2-arch** h20 at 4.122 GF/token
> (it needs the `GeluFeedForward` / `n_blocks=25` / `kv_heads=q_heads` overrides), not
> the **H-balanced** h20 at 3.97 GF/token that `configs/ame_tests/h20_smoke.yaml` builds on,
> which has never been measured multi-GPU. AME itself is 3.77 GF/token and has only a
> 1-GPU number. Three denominators: MFU percentages compare across them, raw tokens/s
> do not.

What is actually missing is an **AME 4-GPU** number and a **same-architecture** dense
4-GPU comparator. Both are the first items in
[ame_optimization_plan.md](ame_optimization_plan.md) §2.3.
