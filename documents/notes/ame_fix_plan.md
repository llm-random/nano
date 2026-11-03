# AME fix plan

**Status: Phases 2–4 applied 2026-11-03; Phase 1 outstanding (needs a GPU).**
Written 2026-11-03.

| phase | state |
|---|---|
| 1 — the kernel (F1) | **not started.** Needs a GPU and, for 1.5, new cluster time. |
| 2 — weight decay (F2) | **done**, including 2.5 instrumentation. 2.6 left at the documented default (banks stay decayed) — still your call. |
| 3 — the revert (F4) | **done**, verified by composing all 11 configs before and after: byte-identical. |
| 4 — cleanup (F3, F5, F6, F7) | **done.** |
| — | **F8 found while fixing F2**: the cosine scheduler asserted a single param group. See Phase 2.3. |

The remediation plan for the issues in
[`ame_code_review.md`](ame_code_review.md), which carries the evidence for each (F1-F7
from the review, plus F8 found while applying Phase 2). Code
lives in `/home/maciej/code/nano-small-changes` (branch `small-changes`, uncommitted);
`ame_implementation.md` is the record of what was built.

Four phases. **Phases 1 and 2 change results**; 3 and 4 are hygiene. Each is
independently committable and leaves the suite green. Phase 1 is the only one that needs
a GPU or new cluster time.

| phase | fixes | blocks | needs GPU |
|---|---|---|---|
| 1 — the kernel | F1 | every performance claim | yes |
| 2 — weight decay | F2 | Stage 0 | no |
| 3 — the revert | F4 | nothing | no |
| 4 — cleanup | F3, F5, F6, F7 | nothing | no |

Phases 1 and 2 are independent of each other. **If Stage 0 should move before a GPU frees
up, run Phase 2 first** — Stage 0 *is* the α/β readout that F2 confounds.

---

## Phase 1 — the kernel (F1)

**What is wrong.** `backend="auto"` picks flex on CUDA ([ame.py:307](../../src/core/ame.py#L307)),
but nothing compiles it: `compile_model` and `compile_blocks` default to `False`
([main.py:151,168](../../main.py#L151)) and no config in the repo sets either. Uncompiled,
`flex_attention` runs the unfused fallback that materializes the full `[B,H,Q,KV]` score
tensor (`torch/_higher_order_ops/flex_attention.py:168`), applies `score_mod` elementwise
under `vmap`, prunes nothing from the block mask, and re-materializes in the backward
(line 826). Measured on CPU, it is **1.9–3.4× slower than this layer's own naive reference
path**. The gqa baseline it was compared against uses fused FlashAttention via
`F.scaled_dot_product_attention` ([model.py:415](../../src/core/model.py#L415)), which is
fast without compile — so the h20 comparison was fused-vs-unfused.

**Why first.** Every performance statement downstream is waiting on a number that is
currently an artifact. The 2.4% MFU / 165 s table in `ame_implementation.md` should be
treated as void until this lands.

### 1.1 Compile flex inside the layer

`src/core/ame.py`, module scope:

```python
_FLEX = None

def _flex():
    """flex_attention, compiled once. Uncompiled it is the unfused fallback that
    materializes the n x n scores (torch _higher_order_ops/flex_attention.py:168)
    -- measured slower than this layer's own reference path, and quadratic in
    memory. torch 2.9 warns about this; the pinned 2.7.1 does not."""
    global _FLEX
    if _FLEX is None:
        from torch.nn.attention.flex_attention import flex_attention
        _FLEX = torch.compile(flex_attention, dynamic=False)
    return _FLEX
```

`_attend_flex` ([ame.py:380](../../src/core/ame.py#L380)) calls `_flex()(q, k, v, ...)`
instead of importing and calling `flex_attention` directly.

Compiling here rather than switching on `compile_model` keeps the fast path independent
of a config flag — the failure mode was precisely that the tests compiled and the config
did not. CPU is unaffected: `auto` still picks the reference path there.

### 1.2 Keep the block mask out of the traced region

Decorate `_block_mask` ([ame.py:357](../../src/core/ame.py#L357)) with
`@torch._dynamo.disable`. `create_block_mask` compiles internally, so tracing it inside
another graph is fragile; the dict cache makes it a guard anyway.

### 1.3 A test that fails on the real failure

GPU-marked, in `tests/test_ame.py`:

- **Memory test.** Run the `auto` path at seq 512 and 1024, assert peak CUDA memory grows
  **sub-quadratically**. Asserting on the callable's identity would pass against a
  compiled wrapper that still fell back; memory tests the property that matters.
- **Equivalence under 1.1.** Every existing equivalence test wraps the layer in
  `torch.compile` itself ([test_ame.py:569-614](../../tests/test_ame.py#L569)), so none of
  them would notice 1.1 breaking the uncompiled call. Add one that compares the plain
  `auto` path against `_attend_reference`.

### 1.4 Verify on GPU

`pytest -m gpu tests/test_ame.py` on an entropy node. Two things cannot be checked from a
CPU box and must be checked there:

- the interaction with FSDP2;
- whether `compile_model: true` on top of 1.1 helps or double-compiles.

### 1.5 Re-run the h20 smoke

Then replace the step-time table in `ame_implementation.md` and mark the old one
superseded. **Needs explicit go-ahead** — it is new cluster time.

> **The baseline moved on 2026-11-03** (commit `492645f0`, another agent, after this
> review was written). `configs/mfu_audit/h20.yaml` is no longer the stage-3 audit
> config: it is now "the measured-best single-GPU configuration" at **46.4% MFU**
> (job 32023), with `compile_model: true`, `compile_loss: true`, `autocast_bf16: true`,
> `distributed: null` and grad-accum 32 (micro-batch 16). The 30.8% / 13.6 s figure this
> plan and the review quote is the *old* h20; h12 and h16 still carry it.
>
> Three consequences for Phase 1:
>
> - **`compile_model: true` is now the baseline's own setting**, which is the knob 1.1
>   works around. Doing both is still right — 1.1 makes AME's fast path independent of
>   config, 1.5 should mirror the new baseline — but it means the comparison config has
>   to be rebuilt against the new h20, not the one `ame/h20_smoke.yaml` was shaped from.
> - **`distributed: null` will not transfer unchanged.** Without FSDP2 the parameters
>   stay fp32, and AME's 3.16B is ~12.6 GB of weights plus gradients plus Adam moments
>   before activations. Decide that deliberately rather than by copying the header.
> - Whichever baseline number is quoted, say which h20 it is. Two now exist.
>
> Also note the worktree sits at `b9189d1c`, three commits behind `pretraining`. A 3-way
> merge of `main.py` is clean (their change is the wandb config block at line 212, these
> fixes are the optimizers at 179/251), and `_should_log_heavy_metrics` — which
> `Trainer._log_layer_instrumentation` reads — survives their `metric_loggers.py` change.

**Exit criterion.** A step time measured against a fused kernel. Only then are the two
remaining candidate causes — dense bank gradients, and FSDP2 reducing every micro-batch —
worth separating with `trainer.profile_timers=true`.

---

## Phase 2 — weight decay (F2)

**What is wrong.** `main.py:179` builds one AdamW over `model.parameters()` with
`weight_decay=0.1` (`configs/_trainer/llama.yaml`), no groups. AdamW's decay is
*decoupled*: `p ← p·(1 − lr·wd)` runs whether or not there is a gradient. Measured with
gradients forced to exactly zero, 2000 steps at lr 2⁻¹²: `alpha[0]` 0.8038 → 0.7655,
`tau[0]` 8.0000 → 7.6184. Cosine-corrected for this repo's schedule, ×0.51 by 50k steps.

Two readouts break:

- `ame_experiments.md` §8.1's kill criterion is that `α₂, β₂ → 0` means the model is
  declining the mechanism. Decay drives them there by itself, hardest in the head whose
  gradient is weakest — the one indifferent to the shift.
- `tau` shrinks but `slopes` is frozen at `tau_init`
  ([ame.py:194-214](../../src/core/ame.py#L194)). A head's horizon is `tau/m`, so every
  head's reach contracts silently while the ladder stays put: the flattest h20 head falls
  8192 → ~4190 tokens by 50k steps.

**Why before Stage 0.** Stage 0 *is* the α/β readout. Running it with decay on measures
the optimizer.

### 2.1 Declare the exempt parameters

`src/core/ame.py`, beside the existing `non_matmul_parameters`:

```python
no_weight_decay_parameters = ("tau", "alpha", "beta", "slopes", "k_sink", "v_sink")
```

Kept separate from `non_matmul_parameters` on purpose. That one answers *"is this a matmul
weight"* for MFU accounting; this one answers *"should decay touch it"*. They overlap
today but are not the same question, and conflating them makes the next model that needs
one and not the other quietly wrong.

### 2.2 Build parameter groups

A `parameter_groups(model, weight_decay)` helper in `src/core/model.py` (or a small
`src/core/optim.py`):

```python
def parameter_groups(model, weight_decay):
    """Two groups: declared scales at wd=0, everything else at wd. A model that
    declares nothing yields one group, bit-identical to a plain parameters() call."""
    undecayed = set()
    for module in model.modules():
        for name in getattr(module, "no_weight_decay_parameters", ()):
            parameter = getattr(module, name, None)
            if isinstance(parameter, nn.Parameter):
                undecayed.add(id(parameter))
    if not undecayed:
        return [{"params": list(model.parameters()), "weight_decay": weight_decay}]
    return [
        {"params": [p for p in model.parameters() if id(p) not in undecayed],
         "weight_decay": weight_decay},
        {"params": [p for p in model.parameters() if id(p) in undecayed],
         "weight_decay": 0.0},
    ]
```

A model declaring nothing returns a single group, so **every gqa baseline stays
bit-identical** and no committed number moves.

### 2.3 Use it at both optimizer sites

`main.py:179` **and** `main.py:251` — the latter is the
`cfg.trainer.checkpoint.load.only_weights` resume path, which rebuilds the optimizer from
scratch and would otherwise silently re-enable decay on resume.

> **F8, found doing this.** `get_cosine_scheduler_with_warmup`
> (`src/core/schedulers.py`) asserted `len(optimizer.param_groups) == 1` — *"Cosine
> scheduler only supports one param group"* — so the grouped optimizer made every run
> die at startup. Nothing in the unit tests noticed; the end-to-end smoke test through
> `main.py` caught it, which is the second time that test has earned its keep.
>
> The guard's real precondition is **one learning rate**, not one group: `eta_min` is a
> single scalar handed to every group, so groups at different lrs would be floored at the
> wrong fraction, while groups differing only in weight decay are fine. The assertion now
> checks `{group["lr"] for group in optimizer.param_groups}` and names the lrs it found.
> Both halves are tested — the grouped optimizer is accepted and steps with a shared lr,
> and two groups at different lrs are still rejected.

### 2.4 Tests

In `tests/test_ame.py` and `tests/test_mfu.py`:

- an AME model yields two groups, with exactly the declared parameters in the `wd=0` one;
- a gqa model yields one group whose parameter list is exactly `model.parameters()`;
- the regression that found this: step a layer with **zero** gradients, assert `alpha`,
  `beta` and `tau` do not move.

### 2.5 Wire the instrumentation

§7 of `ame_experiments.md` asks for α/β trajectories, τ, the horizon ladder, attention
entropy and top-k mass. `horizons()` and `score_matrix()` exist and **nothing logs them**.
At minimum: per-head α₂/β₂, τ, and the horizon ladder to the metric logger.

Without this, Phase 2's fix is unverifiable on a real run, and the τ/slope decoupling is
invisible by construction.

### 2.6 Decide the banks

Not a code change — a call to make.

- **For `wd=0` on the banks:** with QK-norm the rows are scale-invariant, so decay changes
  nothing functionally. It only shrinks `‖row‖`, which inflates the effective learning rate
  on row direction and destroys Stage 1's "rare-type rows stay at init" diagnostic — they
  do not stay, they shrink toward zero keeping their random direction.
- **Against:** the dense baseline decays its `W_Q`/`W_K`, and `baselines.md` cares about
  matched treatment.

**Recommendation:** exempt τ/α/β/sinks now (unambiguous), leave the banks decayed, and
settle the banks explicitly before Stage 1 rather than discovering it inside Stage 1.

---

## Phase 3 — the revert (F4)

`configs/_model/h_balanced/base.yaml` in the worktree carries a hunk unrelated to AME:

```diff
-  sequence_length: ???
+  sequence_length: 2048
+  batch_size: 512
```

The main tree has no diff for this file, so it is not the other agent's work. Nothing
needs it: all eleven configs using `h_balanced` set both values themselves,
`configs/ame/h20_smoke.yaml` included. `???` is OmegaConf's mandatory-value marker, so
removing it turns a composition-time failure into a silent default of 2048 for every
h-family config — and `batch_size` is interpolated by
`configs/_dataset/default.yaml` as `total_batch_size`.

**3.1** Restore `sequence_length: ???`, drop the added `batch_size: 512` and their comment
block. Keep the two AME edits: `vocab_size: ${common.vocab_size}` under `attention_fn`,
and the options-block comment.

**3.2** Confirm nothing regressed — loop `python main.py --cfg job -cn <name>` over
`configs/*.yaml` and `configs/*/*.yaml` and check every one still composes.

Standalone and independent of everything else.

---

## Phase 4 — cleanup (F3, F5, F6, F7)

### 4.1 `_shift1` rank-safe (F3)

The docstring promises `[B, n, ...]` but the six-entry pad tuple reaches dim 1 only at
4-D. A `(2,4,3)` input returns `(3,3,3)` with no exception — it pads the batch dimension.
Not live (the only caller passes `[B,n,H,D]`), but §4.1's planned one-step raw-`v`
register is exactly the next caller.

```python
pad = [0, 0] * (z.dim() - 2) + [1, 0]   # always dim 1, whatever the rank
return F.pad(z, pad)[:, :-1]
```

Plus tests at 3-D and 5-D asserting shape preservation and position-0 zeroing.

### 4.2 A real error for missing ids (F5)

`token_ids=None` currently dies with `AttributeError: 'NoneType' object has no attribute
'shape'` inside `_mixed_qkv`. Raise a `ValueError` at the top of `AMEAttention.forward`
naming the cause instead.

### 4.3 Reject option blocks for option-less types (F6)

`src/core/attention.py:274` subtracts `set(ATTENTION_TYPES)`, so
`{type: gqa, gqa: {...}}` or `{type: mla, mla: {...}}` composes, validates, and is
silently discarded by `_rope_kwargs`. Replace with a `_TYPE_OPTIONS` mapping so a block
for a type that declares no options is named as an error.

### 4.4 Correct `ame_implementation.md` (F7)

- "the baseline's 16" — the gqa baseline's micro-batch is `512/64 = 8`
  (`configs/mfu_audit/h20.yaml`); AME's is 4. The ratio is 2×, not 4×.
- "flex without `torch.compile` warns" — true on torch **2.9**, false on the **2.7.1**
  pinned in pixi that the cluster runs. As written it implies job 32315 would have told
  us, and it could not have.
- Add F1 as the first of the step-time causes; mark the table void pending Phase 1.5.

---

## Not in the plan, deliberately

- **Document boundaries in the shift.** `ame.md` §7 wants position 0 of every document
  zeroed from `cu_seqlens`. This repo does no document masking anywhere — `is_causal=True`
  and nothing else — so AME is consistent with the baseline it is measured against. Fixing
  it in AME alone would make the two *differ*. It belongs to a repo-wide packing change,
  if one ever happens; until then it is a line in the method's limitations.
- **Gradient-norm split under global clipping.** `gradient_clipping: 1.0` is one norm over
  3.16B parameters, 82% of them banks, so the banks set the clip scale for the backbone.
  Worth one logged number during Phase 1.5 rather than a change now — it may be nothing,
  and it has not been measured.
- **`src/core/trainer.py`'s missing `no_sync`.** A framework issue AME only exposes. It
  would move every multi-GPU number in the repo and belongs with the multi-GPU work.
