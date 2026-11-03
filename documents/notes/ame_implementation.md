# AME in this repo: the layer, the plumbing, and what was decided along the way

**Status:** the training-time layer is implemented and tested (2026-11-03). No experiment
has been run. The decode-time system does not exist and is not needed yet.

AME — *Attention Memory Embedding* — replaces the Q and K projections of a standard
attention head with two memory banks indexed by **token id**. The design record lives
outside this repo, in `research/ame/` (`ame.md` for the method, `ame_experiments.md` for
the staged program, `baselines.md` for the comparison standard). This note is the repo
side: what landed, which open choices were settled to land it, and what is deliberately
absent. It is the paired reading for
[attention-switching.md](../concepts/attention-switching.md), which covers the registry
`ame` sits in.

> **Name.** Called MAE in discussions before 2026-11-03; renamed to AME, the name the
> registry already used, because MAE collides with Masked Autoencoders and with "mean
> absolute error". Same method.

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

Training is **unmodified O(n²) attention with two gathers**. That is the point of this
stage — there is no table, no row cache, no V-cache, no CPU offload, and none of it should
be built before an experiment says the mechanism works.

KV shifting (`alpha`, `beta`) is **prior art**: Xu, Cheng, Wang & Chen, arXiv 2411.19574.
It is the only mechanism that gives an id-scored layer the induction/copy primitive, which
is why it is on by default — and it is why any dense baseline AME is measured against must
have it too, or AME collects a gain that is not its own.

## What landed

| file | what |
|---|---|
| [src/core/ame.py](../../src/core/ame.py) | `AMEAttention` — the layer, both kernel paths, the slope ladder, instrumentation hooks |
| [src/core/attention.py](../../src/core/attention.py) | `ame` is a real registry entry; per-type options blocks; builders tolerate type-specific kwargs |
| [src/core/model.py](../../src/core/model.py) | token ids threaded `LLM → encoder → block → Residual`; `RoPEAttention` states its own score FLOPs |
| [src/core/mfu.py](../../src/core/mfu.py) | lookups excluded by module type, not by name; score FLOPs collected per module |
| [src/definitions.py](../../src/definitions.py) | `AttentionSchema.ame` — the options block, typed as `Any` |
| [configs/_model/h_balanced/base.yaml](../../configs/_model/h_balanced/base.yaml) | `vocab_size` reaches `build_attention` |
| [tests/test_ame.py](../../tests/test_ame.py) | 30 tests: the math, the ablation arms, the plumbing |

### Token ids now travel with the activations

Nothing below `LLM.forward` used to see the ids. AME needs them at **every** layer, so
`forward` gained an optional `token_ids` that passes down the stack. A layer opts in by
declaring `wants_token_ids = True`; `Residual` resolves that flag once at construction, so
the branch is a Python constant that dynamo folds away rather than a per-step lookup. Every
existing type is untouched and never receives the ids.

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

A per-type options block is named after its type, so a future `mla:` block composes the
same way. Unknown keys fail at build time with the valid set printed.

## Choices the spec left open, and how they were settled

`ame.md` §0 lists what an implementer must decide. These are the answers this code
commits to; each is one config key away from being revisited.

| question | decided | why |
|---|---|---|
| RMS vs ℓ2 QK-norm | **ℓ2 + learned per-head `tau`**, init `sqrt(d_head)` | §7's default. The two conventions must not be mixed: RMS *with* a temperature lands at score std `1/d_head`, the exact failure §7 warns about. |
| ALiBi slopes per head? | **yes**, geometric, steepest first | assumed by every cost argument in the record |
| the slope ladder itself | **derived from the context**: horizons `16 … 4·seq_len`, converted to slopes by `m = tau/horizon` | the standard ALiBi ladder caps the flattest head near 3k tokens *whatever the context is*, which would silently cap a long-context method far below its own pitch. Specifying horizons instead of slopes makes that failure visible. |
| what position 1 does | **a learned, position-independent sink key**, on by default | a head that learns `alpha_1 -> 0` otherwise gets logit 0 at position 0 — finite, neutral, and able to dominate a short prefix. The same key is the insurance against the distance bias evicting the sink the model will otherwise learn on BOS (StreamingLLM's documented failure mode). **Note the tension:** §0's canonical block has no sink and §7 asks for one. Set `sink: false` to get the block verbatim. |
| AME everywhere or alternating | **everywhere by default**; `mix` already interleaves | the hybrid is a fallback, not the plan; the registry makes it free either way |
| bank tied to the input embedding | **no** | an untested lever (`ame.md` §8.9), not a default |
| Option A's gate | **not implemented** | B is the method; A is a later ablation (E1.3) |
| `kv_heads` | **ignored**; AME is MHA-shaped | `mha` already sets this precedent. Every head has its own bank slice and value head. |
| `cu_seqlens` for the roll | **not needed here** — see below | |

## Repo-specific findings

**The `roll` trap is milder in this repo than the record assumes.** `ame.md` §8.1 warns
that document packing makes `torch.roll` a silent correctness bug. This repo's packer
([datasets.py](../../src/core/datasets.py)) concatenates documents with `eos` and **no**
`cu_seqlens` or varlen masking, so attention already crosses document boundaries. The
shift only has to avoid wrapping the sequence itself, which `_shift1` does by padding and
slicing rather than rolling — wrap-free by construction, not by a guard that can be
forgotten. If varlen attention is ever adopted, the document-boundary zeroing comes back.

**MFU would have lied in both directions.** The old accounting charged `6*P` for every 2-D
weight not named `*embedding*` and collected score FLOPs only from `RoPEAttention`. AME's
banks are ~4× the backbone and cost zero matmul FLOPs; its score path costs exactly what
standard attention's does. Both rules were replaced with something a type states for
itself: `nn.Embedding` parameters are lookups, `non_matmul_parameters` names 2-D
parameters that are not weights, and `attention_score_flops_per_token` is how a type
declares its score cost. The counted FLOPs of every existing config are unchanged —
`test_h4_flops_match_the_closed_form` still holds exactly.

**The banks are large even at H4.** One bank is `vocab_size × d_model` *per layer*:

| H | backbone (total) | non-embedding | `E_Q + E_K` | vs non-emb |
|---:|---:|---:|---:|---:|
| 4 | 29.7M | 3.9M | 103.0M | 26.2× |
| 12 | 183.4M | 106.2M | 927.2M | 8.7× |
| 16 | 354.7M | 251.7M | 1648.4M | 6.5× |
| 24 | 1003.9M | 849.3M | 3708.8M | 4.4× |

Consequences: there is **no CPU smoke config for AME** at the production vocab — the tests
instantiate on the meta device or override `common.vocab_size` down. Before a real run,
the bank-size levers (`d_bank < d_head`, tying `E_Q`/`E_K`, sharing banks across layers,
low-rank rows) stop being ablations and become the thing that decides whether the run
fits. Optimizer state is the binding cost, not the weights: at 14 B/param an H16 bank set
is ~23 GB before the backbone.

**Two kernel paths, both tested.** `reference` materializes the `n×n` scores; `flex` uses
FlexAttention's `score_mod`, which applies the distance bias without materializing
anything and keeps the slopes differentiable (flash's `alibi_slopes` takes a
non-differentiable tensor, so the learnable-slope ablation needs flex). They are checked
against each other and against a literal transcription of the spec. FlexAttention runs on
CPU here, so the equivalence test needs no GPU — but **no part of this has run on a GPU
yet**, and flex without `torch.compile` falls back to materializing scores.
Compiling it is a step-time question for the first real run, not a correctness one.

> **Corrected 2026-11-03.** That last sentence was wrong about who finds out. torch **2.9**
> warns (`flex_attention called without torch.compile() - this will use an unfused
> implementation that materializes the full scores matrix`); the pinned **2.7.1** that the
> cluster runs does **not**, so the first real run went out on the unfused path in silence.
> It is also not merely a step-time question — see the corrected performance section below
> and [`ame_code_review.md`](ame_code_review.md) F1.

## What has actually been verified

**Environments.** Everything below ran on both torch **2.9.0** (conda `nanollmr`, the
default interpreter on this box) and torch **2.7.1** (`.pixi/envs/default`, the version
the repo actually pins for linux-64). The full suite is green on both. Worth knowing that
the default `python3` here is *not* the pinned environment.

**Covered:**

- **38 unit tests** on the layer's math: the two kernel paths against each other and
  against a literal transcription of the spec, the ids-and-distance-only property, QK-norm
  placement, the no-wrap shift, each ablation arm's pinning, causality, init scale.
- **bf16 autocast**, both backends, forward and backward — see the bug below.
- **The MFU closed forms**, including AME's; existing configs' counted FLOPs unchanged.
- **The repo's end-to-end smoke path**, `main.py --config-name tiny_local`, for both gqa
  (regression on the plumbing change) and AME. The AME run builds **132,452,688
  parameters — exactly the closed form** (103.5M banks + 29.0M backbone), reports a finite
  positive loss every step, and writes a resumable checkpoint.
  `test_an_ame_stack_trains_end_to_end_through_main` makes it permanent.
- **Optimization health**: a 2-layer AME on fixed-bigram next-token prediction goes
  4.30 → 0.002 in 200 steps, matching a gqa twin step for step. Read narrowly — that task
  is solvable from the current token alone, so it exercises the optimizer and the
  plumbing, **not** the attention path.

**One real bug, found and fixed.** Under `trainer.autocast_bf16=true` the layer **crashed**:
autocast casts `nn.Linear` but **not** an `nn.Embedding` gather, so `v` arrived bf16 while
the banks handed back fp32, and flex rejects mismatched q/k/v outright. The reference path
did not crash, which is worse — it silently ran half the layer in a precision nobody chose.
Fixed by casting q and k to `v`'s dtype at the end of `_mixed_qkv`, after the gather, the
norm and the mix have had the banks' own precision. `test_bf16_autocast_does_not_mix_dtypes`
is the regression. **This would have crashed the first real training run**, and no fp32
test could have found it.

**`backend: auto` is now device-aware** (flex on CUDA, reference on CPU), which is a
consequence rather than a preference: in torch 2.9 a compiled `flex_attention` whose inputs
require grad **cannot be lowered on CPU at all** — it fails for a bare call with no
score_mod and no block_mask, so it is the backend and not this layer. Before the change the
CPU default was the one path that cannot be compiled for training.

**GPU — run on entropy (`h100a`, NVIDIA H100 80GB, torch 2.7.1+cu128), 2026-11-03.**
`pytest -m gpu tests/test_ame.py`: **5 passed**. That covers `auto` selecting flex and
matching the reference path, the **compiled** flex kernel forward and backward, bf16
autocast against fp32, and the shift/sink/pinned variants under compilation. The flex
backend is no longer unverified.

Getting there found a **bug in the GPU tests, not the layer**: FlexAttention's compiled
kernel requires `head_dim >= 16` ("NYI: embedding dimension ... must be at least 16"), and
the shared `DIMS` fixture is `dmodel=32, q_heads=4` — head_dim **8**. Fine eagerly,
impossible to compile. Every GPU test now builds at `GPU_DIMS`, head_dim **64**, the H
family's real width, with an assert pinning it. A GPU test running an unrepresentative
shape would have been worse than no GPU test.

**First GPU training run — entropy job 32315, 1xH100, 2026-11-03.** h20 with
`attention.type: ame`: **3,163,200,720 parameters** (the closed form exactly), FSDP2
sharded, no OOM at micro-batch 4. Loss falls **11.25 -> 10.44 -> 9.75 -> 9.50 -> 9.31**
over the first five steps. Config: `configs/ame/h20_smoke.yaml`, shaped to sit beside
`configs/mfu_audit/h20.yaml`.

**It is ~12x slower per token than the gqa backbone** — but the comparison is not sound,
see the correction under the table.

> **VOID, 2026-11-03.** These numbers compare a **fused** kernel against an **unfused**
> one and must not be quoted. AME ran FlexAttention uncompiled, i.e. the fallback that
> materializes the full `n×n` scores; the gqa baseline runs fused FlashAttention through
> SDPA, which needs no compile. Re-measure after `ame_fix_plan.md` Phase 1.

| h20, 1xH100, 512 x 2048 tok/step | params | tokens/s | MFU | step time |
|---|---:|---:|---:|---:|
| gqa baseline (stage 3) | 620.2M | 76,858 | 30.8% | 13.6 s |
| **ame** (unfused kernel — void) | **3.163B** | **6,295** | **2.4%** | **165 s** |

Steady, not warmup: steps 1-4 measured 166.5 / 165.6 / 164.6 / 168.2 s. Per token the gap
is the same 12x whether you count per micro-step or per step, so the smaller micro-batch
(4 here against the baseline's 8, chosen because the banks need ~53 GiB) does not by
itself explain it.

This is **R6 in the risk register, now measured rather than predicted** — `ame.md` §7 said
"expect fewer FLOPs and worse wall-clock... the backward scatters gradients into 4.9B bank
parameters, and that traffic plus the optimizer state is what will actually set your step
time". AME is ~15% *cheaper* in FLOPs/token (3.77 G; the banks correctly contribute zero)
and 12x more expensive in seconds.

Three candidate causes. The first was found by the 2026-11-03 code review and is
plausibly dominant; the other two are separable with `trainer.profile_timers=true`.

1. **The kernel was never fused** (`ame_code_review.md` F1). `backend="auto"` picks flex
   on CUDA, but nothing compiles it — `compile_model` and `compile_blocks` default to
   `False` and no config sets either, this one included. Uncompiled, `flex_attention`
   dispatches to `sdpa_dense` → `_math_attention_inner`, which materializes
   `[4, 20, 2048, 2049]` = 336M elements per layer, applies `score_mod` elementwise under
   `vmap`, prunes nothing from the block mask, and re-materializes in the backward. On CPU
   it measures **1.9-3.4x slower than this layer's own naive reference path**, so it is
   worse than no kernel at all. The GPU tests missed it because every one of them wraps
   the layer in `torch.compile` *inside the test*: they prove the compiled kernel works,
   nothing asserts that the shipping config reaches it.
2. **Dense bank gradients.** `nn.Embedding` backward produces a dense `[50304, 1280]`
   gradient per bank per layer — 40 of them, ~10.3 GB — on *every* micro-batch regardless
   of micro-batch size. §7 predicted exactly this and noted that row-sparse gradients do
   not help, because a 0.5M-token batch touches 50-82% of rows anyway.
3. **FSDP2 reducing gradients on every micro-batch.** `src/core/trainer.py` has no
   `set_requires_gradient_sync(False)` / `no_sync` around the accumulation loop, so all
   3.16B gradients are reduced 128 times per step, with `reduce_dtype=torch.float32`
   forcing a bf16->fp32 cast each time. This is a **framework** behaviour, not an AME one
   — AME only makes it visible, because the cost scales with parameter count while the
   compute does not. It would equally affect any large-bank model (PKM, Engram) and is
   worth checking against the multi-GPU grad-sync work.

Until these are separated, **do not quote AME's step time as a property of the method**.

**Not covered, and nameable:**
- **`torch.compile` on CPU is flaky for this module**: it intermittently aborts the
  interpreter from inside an Inductor-generated kernel, including for the reference backend,
  which uses nothing exotic. It is not reliably reproducible in either direction and cannot
  be written around at this level. The compile test is therefore GPU-marked rather than
  skipped quietly.
- **No FSDP2 run.** The banks are `nn.Embedding` with a vocab-sized first dimension and
  should shard the way a vocab-parallel embedding does, but that is reasoning, not a
  measurement.
- **No experiment of any kind has been run.**

**Two things about the test harness, found on the way.**

`tests/test_training_smoke.py` is marked `slow` *and* skips when `data/` and `data_eval/`
are absent — and both are gitignored, so **a git worktree silently skips the entire
end-to-end suite**. Combined with a `-m "not slow"` run that is two independent reasons
for the most valuable test in the repo to report nothing, and neither is visible unless
you read the skip reasons. Symlinking the two directories from the main tree is enough:

```bash
ln -s ../nano/data data && ln -s ../nano/data_eval data_eval   # both gitignored
```

The same file asserts that a fresh model's first loss is within **0.15** of
`ln(50304) = 10.826`, as a guard on init, vocab and head wiring. One step of `tiny_local`
is batch 2 x seq 16 = **30 tokens**, and over six steps gqa wanders 10.92-11.45 while AME
wanders 10.55-11.34 — so that assertion passes for gqa on the strength of a single lucky
batch, and a 0.5 deviation means nothing. (It briefly looked like AME had an init bug for
exactly this reason; a 3-seed, 512-token comparison puts gqa at 11.20 and AME at 11.24,
with identical logit std.) The AME smoke test uses a band of 1.0 instead. Tightening the
gqa one properly would mean averaging the first loss over more tokens — not done here, it
is not this note's change to make.

## Deliberately not built

The decode system: the vocab² table, the Zipf row cache, the inverted index, the V-cache,
the CPU-async path. `ame.md` §0 is emphatic that building it first is the single most
likely way to waste a month, and nothing in Stages 0–1 needs it. Also absent: Option A
(bank values + gate), the p-tap key window, and any experiment harness.

## Open items, in the order they matter

1. **Stage 0 — does the mechanism exist?** The six-arm synthetic induction ablation
   (A1–A4 free/pinned, plus the two positive controls), with the bigram mapping
   **resampled per sequence**; a fixed mapping makes both arms pass and measures nothing.
   Agreed form: a standalone script, since the dataset stack here streams a real corpus.
   The arms are reachable from config today (`pin_alpha` / `pin_beta`).
2. **Instrumentation — partly done (2026-11-03).** `AMEAttention.instrumentation()` now
   reports per-layer `alpha_2`/`beta_2` (mean, min, max and **absmean**, so heads that
   cancel do not read as a collapse), `tau`, and the horizon ladder; `Trainer`
   collects it by duck typing from any module that offers one, on the existing
   heavy-metric interval. Still missing from §7's list: **attention entropy and top-k
   mass**, for which `score_matrix()` exists and nothing calls it.
3. **The kernel has to be fixed before any step time is quoted** — see cause 1 above and
   `ame_fix_plan.md` Phase 1. A GPU has run this code (entropy job 32315, 20 steps, loss
   11.25 → 7.5) so correctness at 3.16B is established; only the performance picture is
   void.
4. **Bank-size levers**, before any run above H4 (see the table above).
5. **`d_bank < d_head`** is accepted as a constructor argument (`dhead=`) but untested as
   a lever.
