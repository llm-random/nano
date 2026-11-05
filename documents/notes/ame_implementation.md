# AME in this repo: the layer, the plumbing, the review, and what is left

**Status: established.** The layer, the plumbing and all eight review findings are done
and tested, on CPU and on an H100, and the code is committed. **No *scientific* experiment
has been run — Stage 0 has not started.** The cluster jobs run since are throughput
benchmarks, not quality measurements; their record is
[ame_optimization_report.md](ame_optimization_report.md).
Written 2026-11-03, last updated 2026-11-05.

AME — *Attention Memory Embedding* — replaces the Q and K projections of a standard
attention head with two memory banks indexed by **token id**. The design record lives
outside this repo, in `research/ame/` (`ame.md` for the method, `ame_experiments.md` for
the staged program, `baselines.md` for the comparison standard). This note is the repo
side: what landed, which open choices were settled to land it, what is verified, and what
is still outstanding. Paired reading:
[attention-switching.md](../concepts/attention-switching.md) for the registry `ame` sits
in, and [archive/ame_review_and_fixes.md](archive/ame_review_and_fixes.md) for the
evidence behind each finding and the remediation detail.

**The code is committed on `ame_main` and `ame_optimize`** as of 2026-11-05. It was
developed in the `/home/maciej/code/nano-small-changes` worktree, which was removed
2026-11-04; passages below that refer to it describe where things stood before then.

> **Name.** Called MAE in discussions before 2026-11-03; renamed to AME, the name the
> registry already used, because MAE collides with Masked Autoencoders and with "mean
> absolute error". Same method.

## Where things stand

| | state |
|---|---|
| The training-time layer | **implemented**, 60 CPU tests + 9 GPU tests |
| Repo plumbing (ids, MFU, config, optimizer, instrumentation) | **implemented** |
| CPU suite | **262 passed, 0 skipped, 0 failed** |
| GPU correctness | **verified** — 9/9 on an H100, and a 3.16B run that trains |
| GPU *performance* | **18.1% MFU** at h20 when this was written. **Optimized since** to 35.02% at 4 GPUs and 37.96% at h16/1 GPU — [ame_optimization_report.md](ame_optimization_report.md) |
| Code review (2026-11-03) | 8 findings, F1–F8, **all fixed** |
| Fix plan | **all four phases applied** |
| Experiments | **none run.** Stage 0 has not started. |
| Decode system | **not built, deliberately** |

## The method, as implemented

```python
q = E_Q[t]                      # [B,n,H,D]  gather by token id -- NOT x @ W_Q
k = E_K[t]                      # [B,n,H,D]  gather              -- NOT x @ W_K
v = (x @ W_V).view(B,n,H,D)     # ordinary projection, unchanged from MHA

q = l2_normalize(q, dim=-1)     # QK-norm on the BANK ROWS, before the mix
k = l2_normalize(k, dim=-1)

k = a1*k + a2*shift1(k)         # KV shifting -- a1,a2,b1,b2 are [H] learnable
v = b1*v + b2*shift1(v)         # shift1(z)[:,j] = z[:,j-1], position 0 zeroed

s = einsum('bihd,bjhd->bhij', q, k) * tau   # tau [H], init sqrt(d_head)
s = s - m * dist                            # ALiBi, per-head slope -- the ONLY
s = s.masked_fill(causal, -inf)             #   positional signal; no RoPE anywhere
out = softmax(s) @ v
out = out.reshape(B,n,H*D) @ W_O
```

Scores are therefore `s_ij = tau * T[t_i,t_j] - m_h*(i-j)` with `T[a,b] =
norm(E_Q[a]) . norm(E_K[b])` — **a function of token ids and distance only**. That is the
whole design: after training the scores collapse into a vocab² table, so decode needs no
Q/K matmuls and no K-cache. It is also the whole weakness: the weights never read a hidden
state, so retrieval is never content-addressed, in any variant.

Training is **unmodified O(n²) attention with two gathers**. There is no table, no row
cache, no V-cache, no CPU offload, and none of it should be built before an experiment
says the mechanism works.

KV shifting (`alpha`, `beta`) is **prior art**: Xu, Cheng, Wang & Chen, arXiv 2411.19574.
It is the only mechanism that gives an id-scored layer the induction/copy primitive, which
is why it is on by default — and why any dense baseline AME is measured against must have
it too, or AME collects a gain that is not its own.

## What landed

| file | what |
|---|---|
| [src/core/ame.py](../../src/core/ame.py) | `AMEAttention` — the layer, both kernel paths, the slope ladder, `instrumentation()` |
| [src/core/attention.py](../../src/core/attention.py) | `ame` is a real registry entry; `_TYPE_OPTIONS` governs per-type option blocks |
| [src/core/model.py](../../src/core/model.py) | token ids threaded `LLM → encoder → block → Residual`; `RoPEAttention` states its own score FLOPs; `parameter_groups` |
| [src/core/mfu.py](../../src/core/mfu.py) | lookups excluded by module type, not by name; score FLOPs collected per module |
| [src/core/trainer.py](../../src/core/trainer.py) | `_log_layer_instrumentation` — per-layer diagnostics, duck-typed |
| [src/core/schedulers.py](../../src/core/schedulers.py) | the cosine scheduler takes grouped optimizers (one *lr*, not one group) |
| [main.py](../../main.py) | both optimizer sites build param groups |
| [src/definitions.py](../../src/definitions.py) | `AttentionSchema.ame` — the options block, typed as `Any` |
| [configs/_model/h_balanced/base.yaml](../../configs/_model/h_balanced/base.yaml) | `vocab_size` reaches `build_attention` |
| [configs/ame/h20_smoke.yaml](../../configs/ame/h20_smoke.yaml) | the first GPU run's config |
| [tests/test_ame.py](../../tests/test_ame.py) | 69 tests: the math, the ablation arms, the plumbing, weight decay, instrumentation, the fused kernel |

### Token ids travel with the activations

Nothing below `LLM.forward` used to see the ids. AME needs them at **every** layer, so
`forward` gained an optional `token_ids` that passes down the stack. A layer opts in by
declaring `wants_token_ids = True`; `Residual` resolves that flag once at construction, so
the branch is a Python constant dynamo folds away rather than a per-step lookup. Every
existing type is untouched and never receives the ids. Passing `None` to an AME layer
raises a `ValueError` naming the cause.

### The config surface

```yaml
common:
  attention:
    type: ame
    ame:                  # options, applied wherever ame is built (base or mix)
      shift: true         # KV shifting on
      pin_alpha: false    # pin the key-side mix to (1,0)   -- ablation arms
      pin_beta: false     # pin the value-side mix to (1,0)
      sink: true          # learned position-independent sink key
      qk_norm: true
      bank_init_std: 1.0
      horizon_factor: 4.0 # flattest head reaches 4x the training context
      backend: auto       # auto | flex | reference
```

Only a type listed in `_TYPE_OPTIONS` takes a block — today that is `ame` alone. A block
named after any other registered type is a build error, not a silent no-op. Unknown keys
inside the block fail at build time with the valid set printed.

## Choices the spec left open, and how they were settled

`ame.md` §0 lists what an implementer must decide. These are the answers this code
commits to; each is one config key away from being revisited.

| question | decided | why |
|---|---|---|
| RMS vs ℓ2 QK-norm | **ℓ2 + learned per-head `tau`**, init `sqrt(d_head)` | §7's default. The two conventions must not be mixed: RMS *with* a temperature lands at score std `1/d_head`, the exact failure §7 warns about. |
| ALiBi slopes per head? | **yes**, geometric, steepest first | assumed by every cost argument in the record |
| the slope ladder itself | **derived from the context**: horizons `16 … 4·seq_len`, converted by `m = tau/horizon` | the standard ALiBi ladder caps the flattest head near 3k tokens *whatever the context is*, silently capping a long-context method far below its own pitch. Specifying horizons makes that failure visible. |
| what position 1 does | **a learned, position-independent sink key**, on by default | a head that learns `alpha_1 -> 0` otherwise gets logit 0 at position 0 — finite, neutral, able to dominate a short prefix. Also insurance against the distance bias evicting the BOS sink (StreamingLLM's failure mode). **Tension:** §0's canonical block has no sink, §7 asks for one. `sink: false` gives the block verbatim. |
| AME everywhere or alternating | **everywhere by default**; `mix` interleaves | the hybrid is a fallback, not the plan |
| bank tied to the input embedding | **no** | an untested lever (`ame.md` §8.9), not a default |
| Option A's gate | **not implemented** | B is the method; A is a later ablation (E1.3) |
| `kv_heads` | **ignored**; AME is MHA-shaped | `mha` sets the precedent. Every head has its own bank slice and value head. |
| `cu_seqlens` for the roll | **not needed here** — see below | |
| weight decay on the scales | **off** for `tau`, `alpha`, `beta`, `slopes`, sinks | see the review, F2 |
| weight decay on the **banks** | **off**, decided 2026-11-03 | With QK-norm a row's length never reaches the score, so decay changes no output — it only shrinks `‖row‖`, inflating the effective LR on direction and breaking Stage 1's "rare-type rows stay at init" check. Costs matched regularization against the dense baseline's `W_Q`/`W_K`; say so when reporting (`ame_fix_plan.md` 2.6). |

## Repo-specific findings

**The `roll` trap is milder here than the record assumes.** `ame.md` §8.1 warns that
document packing makes `torch.roll` a silent correctness bug. This repo's packer
([datasets.py](../../src/core/datasets.py)) concatenates documents with `eos` and **no**
`cu_seqlens` or varlen masking, so attention already crosses document boundaries. The
shift only has to avoid wrapping the sequence itself, which `_shift1` does by padding and
slicing — wrap-free by construction. Zeroing position 0 of every document would make AME
*differ* from the baseline it is measured against, so it is deliberately not done. If
varlen attention is ever adopted, it comes back.

**MFU would have lied in both directions.** The old accounting charged `6*P` for every 2-D
weight not named `*embedding*` and collected score FLOPs only from `RoPEAttention`. AME's
banks are ~4× the backbone and cost zero matmul FLOPs; its score path costs exactly what
standard attention's does. Both rules were replaced with something a type states for
itself: `nn.Embedding` parameters are lookups, `non_matmul_parameters` names 2-D
parameters that are not weights, and `attention_score_flops_per_token` is how a type
declares its score cost. Counted FLOPs of every existing config are unchanged.

**The banks are large even at H4.** One bank is `vocab_size × d_model` *per layer*:

| H | backbone (total) | non-embedding | `E_Q + E_K` | vs non-emb |
|---:|---:|---:|---:|---:|
| 4 | 29.7M | 3.9M | 103.0M | 26.2× |
| 12 | 183.4M | 106.2M | 927.2M | 8.7× |
| 16 | 354.7M | 251.7M | 1648.4M | 6.5× |
| 24 | 1003.9M | 849.3M | 3708.8M | 4.4× |

There is **no CPU smoke config at the production vocab** — tests override
`common.vocab_size` down. Before a real run the bank-size levers (`d_bank < d_head`,
tying `E_Q`/`E_K`, sharing banks across layers, low-rank rows) stop being ablations and
become what decides whether the run fits. Optimizer state binds, not the weights: at
14 B/param an H16 bank set is ~23 GB before the backbone.

**Two kernel paths.** `reference` materializes the `n×n` scores; `flex` uses
FlexAttention's `score_mod`, which applies the distance bias without materializing
anything and keeps the slopes differentiable (flash's `alibi_slopes` takes a
non-differentiable tensor, so the learnable-slope ablation needs flex). They agree
exactly in fp32 and are both checked against a literal transcription of the spec.

**`backend: auto` is device-aware** (flex on CUDA, reference on CPU) as a consequence, not
a preference: a compiled `flex_attention` whose inputs require grad **cannot be lowered on
CPU at all** — it fails for a bare call with no score_mod and no block_mask, so it is the
backend, not this layer.

## What is verified

**Environments.** Everything ran on the pinned torch **2.7.1**
(`.pixi/envs/default/bin/python`, what the cluster uses) and on torch **2.9.0** (conda
`nanollmr`, the default `python3` on this box — *not* the pinned environment).

**CPU — full suite: 249 passed, 0 skipped, 0 failed.** The zero matters: the end-to-end
tests through `main.py` actually ran. See the harness trap below.

- **60 CPU tests** on the layer: the two kernel paths against each other and against the
  spec transcription, the ids-and-distance-only property, QK-norm placement, the no-wrap
  shift at every rank, each ablation arm's pinning, causality, init scale, bf16 autocast,
  weight-decay grouping, instrumentation, and the entropy / top-k diagnostics.
- **The MFU closed forms**, including AME's; existing configs' counted FLOPs unchanged.
- **End-to-end through `main.py`** for both gqa (regression on the plumbing) and AME. The
  AME run builds **132,452,688 parameters — exactly the closed form**, reports finite loss
  every step, and writes a resumable checkpoint.
- **FSDP2 on a gloo mesh** under the production `MixedPrecisionPolicy(bf16, fp32)`:
  forward and backward for `qk_norm` on/off (the mixed-`requires_grad` case), `shift=False`
  and `pin_alpha`, and the weight-decay grouping survives sharding.
- **Optimization health**: a 2-layer AME on fixed-bigram next-token prediction goes
  4.30 → 0.002 in 200 steps, matching a gqa twin. Read narrowly — that task is solvable
  from the current token alone, so it exercises the optimizer and plumbing, **not** the
  attention path.

**GPU — entropy `h100a`, H100 80GB, torch 2.7.1+cu128.** `pytest -m gpu tests/test_ame.py`:
**9 passed** — `auto` selecting flex and matching the reference path, the **compiled** flex
kernel forward and backward, bf16 autocast against fp32, the shift/sink/pinned variants
under compilation, and four added with Phase 1: that the default path does not materialize
the score matrix, that it is correct *without the test compiling it*, that it reaches the
compiled wrapper, and that a whole `TransformerBlock` compiled by `compile_blocks_full`
still receives the token ids.

**GPU training runs at h20, 1×H100.** Both built **3,163,200,720 parameters** — the
closed form exactly — FSDP2 sharded with no OOM at micro-batch 4.

- **Job 32315** (unfused kernel): loss **11.25 → 7.5 at step 19**, monotone, killed by the
  1-hour wall at 165 s/step.
- **Job 32360** (fused kernel, `compile_blocks_full`): all **30 steps in ~13 minutes**,
  loss **11.25 → 7.28**, zero errors, 22.12 s/step throughout.

**Correctness at 3.16B on real data is established**, and the two runs agree on the loss
trajectory, which is the check that the kernel change altered speed and not arithmetic.

### Step time, after the kernel was fixed

Phase 1 landed on 2026-11-03: `src/core/ame.py` now compiles `flex_attention` itself, so
the fused Triton kernel no longer depends on a config flag. Re-measured as entropy job
**32360**, `configs/ame/h20_smoke.yaml`, 1xH100.

| h20, 1xH100, 512 x 2048 tok/step | params | micro-batch | tokens/s | MFU | step time |
|---|---:|---:|---:|---:|---:|
| gqa, `mfu_audit/h20_mgpu.yaml` (FSDP2 + compile_blocks_full) | 620.2M | 16 | 109,361 | 45.57% | 9.58 s |
| **ame, fused kernel** | **3.163B** | **4** | **47,391** | **18.1%** | **22.1 s** |
| ~~ame, unfused kernel (job 32315, superseded)~~ | 3.163B | 4 | 6,295 | 2.4% | 165 s |

**Fixing the kernel was worth 7.5x**: 165 s -> 22.1 s per step, MFU 2.4% -> 18.1%. Steady,
not warmup — steps 4-8 measured 22.124 / 22.119 / 22.113 / 22.115 s. Nothing but the
kernel changed between the two AME rows; the micro-batch, the model and the tokens per
step are identical, which is why `h20_smoke.yaml` keeps `gradient_accumulation_steps: 128`
rather than matching the baseline's 32.

**Which baseline.** Three h20 configs now exist and they are not interchangeable:
`mfu_audit/h20.yaml` (single-GPU, `distributed: null`, 46.4%), `mfu_audit/h20_mgpu.yaml`
(FSDP2 + `compile_blocks_full`, 45.57% at 1 GPU) and the stage-3 audit default that h12
and h16 still carry (30.8%). AME is read against **h20_mgpu**, because it needs FSDP2's
sharding and bf16 parameters and cannot take `distributed: null` at 3.16B parameters.

**The remaining 2.5x gap is not yet attributed**, and two of its candidates are still the
ones Phase 1 did not touch:

1. **Micro-batch 4 against the baseline's 16.** Smaller micro-batches amortise the
   per-step overheads over fewer tokens. Not free to fix: ~57 GiB of parameters, master
   weights, gradients and Adam moments leaves ~23 GiB, which will not hold 16.
2. **Dense bank gradients.** `nn.Embedding` backward produces a dense `[50304, 1280]`
   gradient per bank per layer — 40 of them, ~10.3 GB — on *every* micro-batch regardless
   of its size. `ame.md` 7 predicted this and noted that row-sparse gradients do not help,
   because a 0.5M-token batch touches 50-82% of rows anyway.
3. **FSDP2 reducing gradients on every micro-batch.** `src/core/trainer.py` has no
   `set_requires_gradient_sync(False)` around the accumulation loop, so all 3.16B
   gradients are reduced 128 times per step with a bf16->fp32 cast each time. A
   **framework** behaviour AME only makes visible, because the cost scales with parameter
   count while the compute does not; it would equally affect any large-bank model.

AME remains ~15% *cheaper* in FLOPs/token than its dense twin (3.77 G; the banks correctly
contribute zero), which is R6 in the risk register: fewer FLOPs, more seconds.

**Two bugs found on the way, both worth remembering.**

*bf16 autocast crashed the layer.* Autocast casts `nn.Linear` but **not** an `nn.Embedding`
gather, so `v` arrived bf16 while the banks handed back fp32, and flex rejects mismatched
q/k/v outright. The reference path did not crash, which is worse — it silently ran half the
layer in a precision nobody chose. Fixed by casting q and k to `v`'s dtype at the end of
`_mixed_qkv`. **This would have crashed the first real training run**, and no fp32 test
could have found it.

*A bug in the GPU tests, not the layer.* FlexAttention's compiled kernel requires
`head_dim >= 16`, and the shared `DIMS` fixture gave head_dim **8**. Every GPU test now
builds at `GPU_DIMS`, head_dim **64**, the H family's real width, with an assert pinning
it. A GPU test running an unrepresentative shape is worse than no GPU test.

**The test-harness trap.** `tests/test_training_smoke.py` is marked `slow` *and* skips when
`data/` and `data_eval/` are absent — both gitignored, so **a git worktree silently skips
the entire end-to-end suite**. With a `-m "not slow"` run that is two independent reasons
for the most valuable test in the repo to report nothing. Symlink them:

```bash
ln -sfn /home/maciej/code/nano/data data && ln -sfn /home/maciej/code/nano/data_eval data_eval
```

That suite has now caught two bugs nothing else did (F8 below, and the plumbing
regression). The same file asserts a fresh model's first loss is within **0.15** of
`ln(50304)`; one step of `tiny_local` is 30 tokens, over which gqa wanders 10.92–11.45, so
that assertion passes on a single lucky batch. The AME test uses a band of 1.0.

## The code review, and the fix plan

A second pass over the implementation on 2026-11-03 found **8 issues**. Evidence for each
is in [archive/ame_review_and_fixes.md](archive/ame_review_and_fixes.md) Part A; the
remediation detail and verification record are in Part B of the same document.

| # | severity | finding | state |
|---|---|---|---|
| **F1** | high — perf | GPU path ran FlexAttention **uncompiled** → the unfused fallback that materializes the full score matrix | **fixed** — 7.5x faster |
| **F2** | high — science | weight decay 0.1 hit `tau`, `alpha`, `beta`, sinks, banks — confounding both headline readouts | **fixed** |
| F3 | medium | `_shift1` silently wrong for non-4-D input | fixed |
| F4 | medium | `base.yaml` carried an unrelated hunk removing a mandatory-field guard | reverted |
| F5 | low | `token_ids=None` gave a cryptic `AttributeError` | fixed |
| F6 | low | option blocks for option-less types silently discarded | fixed |
| F7 | low | two factual errors in this note | corrected |
| **F8** | medium | the cosine scheduler asserted one param group, so F2's fix killed every run at startup | **fixed** |

### F1 — the kernel was never fused, and what it cost

`backend="auto"` picks flex on CUDA, but nothing compiled it: `compile_model` and
`compile_blocks` default to `False` and no config set either. Uncompiled, `flex_attention`
dispatches to `sdpa_dense` → `_math_attention_inner`, which materializes
`[4, 20, 2048, 2049]` = 336M elements per layer, applies `score_mod` elementwise under
`vmap`, prunes nothing from the block mask, and re-materializes in the backward. On CPU it
measured **1.9–3.4× slower than this layer's own naive reference path** — worse than no
kernel at all. torch **2.9** warns about this; the pinned **2.7.1** does **not**, which is
why an hour of H100 time went by in silence.

The GPU tests missed it because every one of them wrapped the layer in `torch.compile`
*inside the test*: they proved the compiled kernel works, and nothing asserted that the
shipping config reaches it.

**Fixed** by compiling `flex_attention` in `src/core/ame.py` itself, so the fast path no
longer depends on a config flag. The wrapper steps aside in two cases: when the caller is
already tracing (`compile_blocks_full`, `compile_model`) the surrounding graph lowers the
call, and on CPU the bare function is used because a compiled `flex_attention` whose
inputs require grad cannot be lowered there at all. Measured effect: **7.5×** (see the
step-time table above).

`_block_mask` is deliberately **not** marked `@torch._dynamo.disable`, and that is a
measurement rather than a preference. Skipping it does buy one graph break — on an H100
the layer takes **3** with the mask traced and **2** with it skipped, 2.1 ms against
2.0 ms per forward+backward — but it also makes the BlockMask a graph *input*, and its
sequence length then arrives as a SymInt once dynamo marks that dimension dynamic.
Inductor cannot lower `flex_attention` with it (`args[8]: (s4,)`), and two GPU tests go
red. Reproduced twice with a control, jobs 32368 and 32372; a standalone probe does
*not* reproduce it, so check this against `pytest -m gpu` rather than a script. The mask
is cached per `(seq_len, offset, device)`, so after the first call it is a dict lookup
anyway.

> **The test that was supposed to catch this did not, at first.** The argument for
> asserting on memory rather than on which callable ran was right — an identity check
> passes against a wrapper that still falls back — but the first version measured seq 512
> against 1024 without a warmup and passed on the unfused path. Two reasons, both
> measured on an H100: the first call compiles the kernel and the compiler's workspace
> swamps the reading (cold, seq 512 measured *larger* than 1024, 79.4 vs 38.1 MiB), and
> below 1024 the quadratic term does not dominate, so the unfused path also grows ~2.0×.
> The test now warms up and measures 1024 → 2048, where fused is **1.99×** and unfused is
> **3.75×** (60.6 vs 1097.4 MiB). Verified both ways: it passes on the fixed tree and
> fails on a tree with the fix reverted.

### F2 — why weight decay was confounding the experiment

AdamW's decay is *decoupled*: `p ← p·(1 − lr·wd)` runs whether or not there is a gradient.
Measured with gradients forced to exactly zero, 2000 steps at lr 2⁻¹²: `alpha[0]`
0.8038 → 0.7655, `tau[0]` 8.0000 → 7.6184. Cosine-corrected, ×0.51 by 50k steps.

Two readouts broke. `ame_experiments.md` §8.1's kill criterion is that `α₂, β₂ → 0` means
the model is declining KV shifting — decay drives them there by itself, hardest in the head
whose gradient is weakest, which is the head indifferent to the shift. And `tau` shrinks
while `slopes` stays frozen at `tau_init`, so every head's horizon `tau/m` contracts
silently (the flattest h20 head 8192 → ~4190 by 50k steps).

Fixed with `no_weight_decay_parameters` on the module and `parameter_groups` in
`src/core/model.py`, used at **both** optimizer sites in `main.py` — including the
`only_weights` resume path, which rebuilds the optimizer and would otherwise restore decay
on resume. A model declaring nothing gets exactly one group holding element-for-element
`model.parameters()`, so **every gqa baseline stays bit-identical**.

### Fix-pass progress

| phase | state |
|---|---|
| **1 — the kernel (F1)** | **done.** Verified on an H100 (jobs 32359/32362/32363/32364) and re-measured at h20 (job 32360). |
| 2 — weight decay (F2) | **done**, instrumentation included and now complete. 2.6 (banks) is an open decision. |
| 3 — the revert (F4) | **done.** Verified by composing all 11 runnable configs with and without the hunk: resolved output identical for every one. |
| 4 — cleanup (F3, F5, F6, F7) | **done.** |

**Every new behaviour was mutation-checked** — the fix reverted, the test confirmed
failing, then restored. 11 mutations, all FAIL→PASS: `_shift1`'s pad tuple, the `None`
guard, `_TYPE_OPTIONS`, the `no_weight_decay_parameters` declaration, the `parameter_groups`
split, `instrumentation()`, the scheduler guard, the trainer wiring, the heavy-metric gate,
the duck-type filter, and `isinstance` → `type(p) is nn.Parameter` (which breaks *only* the
sharded test, which is why that test exists).

**Not verifiable on CPU:** anything in Phase 1; the `full_tensor()` collective in
`instrumentation()` at world size > 1 (uniform because `log_metrics` runs on every rank
with the rank gate inside the logger — reasoning, not measurement); and whether the decay
exemption changes what `α₂` converges to, which is a Stage-0 question.

### The baseline moved under this work

On 2026-11-03 another agent rewrote `configs/mfu_audit/h20.yaml`: it is no longer the
stage-3 audit config but "the measured-best single-GPU configuration" at **46.4% MFU**,
with `compile_model: true`, `compile_loss: true`, `autocast_bf16: true`,
`distributed: null` and grad-accum 32. A new **`configs/mfu_audit/h20_mgpu.yaml`** carries
the FSDP2 + `compile_blocks_full` recipe. Three h20 baselines now exist — say which one
you mean.

For Phase 1 this means: **`h20_mgpu.yaml` is the better template**, because AME needs
FSDP2's sharding and bf16 parameters and cannot simply take `distributed: null` at 3.16B
parameters. And `compile_blocks_full` compiles the whole `TransformerBlock`, which puts
`_attend_flex` inside a compiled region and **may make the Phase 1.1 fix redundant at
h20** — check that before building both.

## Deliberately not built

The decode system: the vocab² table, the Zipf row cache, the inverted index, the V-cache,
the CPU-async path. `ame.md` §0 is emphatic that building it first is the single most
likely way to waste a month, and nothing in Stages 0–1 needs it. Also absent: Option A
(bank values + gate), the p-tap key window, and any experiment harness.

## Open items, in the order they matter

1. **Stage 0 — does the mechanism exist?** The six-arm synthetic induction ablation
   (A1–A4 free/pinned, plus the two positive controls), with the bigram mapping
   **resampled per sequence** — a fixed mapping makes both arms pass and measures nothing.
   Agreed form: a standalone script. The arms are reachable from config today
   (`pin_alpha` / `pin_beta`). Its two prerequisites — the weight-decay fix and the α/β
   instrumentation — are both done, so nothing blocks it.
2. **Report the regularization asymmetry.** 2.6 is decided — the banks are exempt from
   weight decay (`archive/ame_review_and_fixes.md` 2.6) — which means AME's banks and the dense baseline's
   `W_Q`/`W_K` are no longer treated alike. `baselines.md` asks for matched treatment, so
   this has to be said out loud in any comparison rather than left in the config.
3. ~~**Attribute the remaining 2.5× step-time gap**~~ — **done 2026-11-05.** This became
   the MFU program; the attribution, and what came of it, are in
   [ame_optimization_report.md](ame_optimization_report.md). Original wording:
   — micro-batch 4 vs the baseline's 16,
   dense bank gradients, and FSDP2's per-micro-batch reduction. One short run with
   `trainer.profile_timers=true` separates the last two; the first needs either more
   memory headroom or an explicit decision to compare at equal micro-batch.
4. **Bank-size levers**, before any run above H4 (see the table above).
5. **`d_bank < d_head`** is accepted as a constructor argument (`dhead=`) but untested.
