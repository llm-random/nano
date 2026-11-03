# AME implementation review — 2026-11-03

**Status: established** — these are findings about the code as it stands on
2026-11-03; none of them have been fixed. The plan is
[`ame_fix_plan.md`](ame_fix_plan.md).

A second pass over the AME code in `/home/maciej/code/nano-small-changes`, after the h20
smoke run (entropy job 32315). Everything below was checked against
[`ame.md`](../../../research/ame/ame.md) §0/§7 and verified on CPU where verification was
possible; nothing was changed. Proposed fixes are written out but **not applied** — they
are decisions, not typos.

Companion: [`ame_implementation.md`](ame_implementation.md), which this review corrects in
three places (F7).

## Summary

| # | severity | finding |
|---|---|---|
| **F1** | **high — performance** | The GPU path ran FlexAttention **uncompiled**, i.e. the unfused fallback that materializes the full score matrix. The 12× step-time gap was substantially an artifact. **Fixed 2026-11-03: 7.5× faster, 2.4% → 18.1% MFU.** |
| **F2** | **high — scientific validity** | Weight decay 0.1 hits `tau`, `alpha`, `beta`, the sinks and the banks. It confounds both of the method's headline readouts. |
| **F3** | medium — latent | `_shift1` silently returns a wrong shape for any input that is not 4-D. |
| **F4** | medium — scope | `h_balanced/base.yaml` carries a hunk unrelated to AME that removes a mandatory-field guard. |
| **F5** | low | `token_ids=None` into an AME layer raises `AttributeError: 'NoneType' object has no attribute 'shape'`. |
| **F6** | low | Per-type option blocks are accepted and silently discarded for every type except `ame`. |
| **F7** | low — docs | Two factual errors and one missing cause in `ame_implementation.md`. |
| **F8** | medium — found later | `get_cosine_scheduler_with_warmup` asserted a single optimizer param group, so F2's fix made every run die at startup. Found by the end-to-end smoke test while applying Phase 2, not by this review. Fixed with the rest; details in [`ame_fix_plan.md`](ame_fix_plan.md) 2.3. |

Nothing in the review found an error in the **attention mathematics**. The score path,
the shift, the bias, the sink and the two backends all hold up; see *What is verified
good* at the end.

---

## F1 — the production GPU path never ran the fused kernel

**What happens.** `backend="auto"` resolves to `flex` on CUDA
([ame.py:307](../../src/core/ame.py#L307)), and `_attend_flex` calls `flex_attention(...)`
directly. Nothing compiles it: `compile_model` and `compile_blocks` both default to
`False` ([main.py:151,168](../../main.py#L151)), and **no config in the repo sets either**
— including `configs/ame/h20_smoke.yaml`.

Uncompiled, `flex_attention` dispatches to the `CompositeExplicitAutograd` implementation
`sdpa_dense` → `math_attention` → `_math_attention_inner`, which is:

```python
# torch/_higher_order_ops/flex_attention.py:168
scores = (query @ key.transpose(-2, -1)).to(dtype=working_precision)
```

— the full `[B, H, Q, KV]` tensor, with `score_mod` applied elementwise under `vmap` and
the mask applied as a `torch.where` (line 187), so **the block mask prunes nothing**:
block skipping only exists inside the compiled Triton kernel. The backward
(`sdpa_dense_backward`, line 763) re-runs `_math_attention_inner` at line 826 and
materializes several more `n×n` tensors.

PyTorch says so itself, on torch 2.9:

> `flex_attention called without torch.compile() - this will use an unfused implementation that materializes the full scores matrix instead of …`

**torch 2.7.1 — the pinned pixi/cluster version — does not emit that warning.** That is
why job 32315 ran for an hour with no signal.

**Measured.** On CPU, eager flex against the layer's own materializing reference path,
forward + backward:

| seq | reference | eager flex | ratio |
|---:|---:|---:|---:|
| 256 | 29.9 ms | 100.5 ms | 3.37× |
| 512 | 81.1 ms | 155.4 ms | 1.92× |
| 1024 | 142.7 ms | 274.5 ms | 1.92× |

Eager flex is **slower than a naive materialized implementation** — it buys no kernel and
pays `vmap` overhead on top.

**Why this changes the h20 conclusion.** At h20 the materialized tensor is
`4 × 20 × 2048 × 2049` = 336M elements **per layer, per materialization**, with no causal
skip. The baseline it was compared against is not in the same class: `RoPEAttention` calls
`F.scaled_dot_product_attention` under `sdpa_kernel([FLASH_ATTENTION, …])`
([model.py:415-431](../../src/core/model.py#L415)) — fused, O(n) memory, causal-skipped,
and it needs no `torch.compile` to be fast.

So `ame_implementation.md`'s two candidate causes (dense bank gradients; FSDP2 reducing
every micro-batch) are missing a third that is plausibly dominant, and it is a
configuration bug rather than a property of the method. The note had even flagged the risk
— *"flex without `torch.compile` … falls back to materializing scores. Compiling it is a
step-time question for the first real run"* — and then the first real run went out
uncompiled.

The GPU tests did not catch it because every one of them wraps the layer in
`torch.compile` inside the test
([test_ame.py:569-614](../../tests/test_ame.py#L569)). They prove the compiled kernel
works; nothing asserts that the shipping path reaches it.

### Proposed fix

Compile flex inside `ame.py`, so correctness of the fast path does not depend on a config
flag:

```python
_FLEX = None

def _flex():
    """flex_attention, compiled once. Uncompiled it is the unfused fallback that
    materializes the n x n scores -- slower than this layer's own reference path
    (measured) and quadratic in memory. torch 2.9 warns about this; 2.7 does not."""
    global _FLEX
    if _FLEX is None:
        from torch.nn.attention.flex_attention import flex_attention
        _FLEX = torch.compile(flex_attention, dynamic=False)
    return _FLEX
```

and mark the block-mask builder so it is not traced when the outer model is also compiled
(`create_block_mask` compiles internally; tracing it inside another graph is fragile):

```python
@torch._dynamo.disable
def _block_mask(self, seq_len, offset, device):
    ...
```

Two caveats, stated rather than assumed:

- **This needs a GPU to verify.** Compiling flex is the documented path, but the
  interaction with FSDP2 and with `compile_model: true` is not something I can check here.
- CPU is unaffected — `auto` keeps the reference path there.

**And a test that would have caught it**, GPU-marked: run the `auto` path at seq 512 and
1024 and assert peak CUDA memory grows sub-quadratically. Asserting on the kernel's
identity would pass against a compiled wrapper that still fell back; asserting on memory
tests the thing that matters.

**Then re-run the h20 smoke** before any step-time number is quoted anywhere. The current
2.4% MFU / 165 s figure should be treated as void, not as a measurement of AME.

---

## F2 — weight decay confounds both headline readouts

**What happens.** `main.py:179` builds a single AdamW over `model.parameters()` with
`weight_decay=0.1` (`configs/_trainer/llama.yaml`), with no parameter groups. Every AME
parameter is therefore decayed:

```
tau (4,)   alpha (4,2)   beta (4,2)   k_sink (4,64)   v_sink (4,64)
q_bank.weight   k_bank.weight   v_proj.weight   o_proj.weight
```

AdamW's decay is **decoupled**: `p ← p·(1 − lr·wd)` runs whether or not there is a
gradient. Measured, 2000 steps at lr 2⁻¹² with gradients forced to **exactly zero**:

```
alpha[0]  [0.1962, 0.8038]  ->  [0.1868, 0.7655]
tau[0]    8.0000            ->  7.6184
```

Closed form `(1 − lr·wd)^N` at lr 2⁻¹², and the same corrected for this repo's cosine
schedule (peak → 0.1·peak, so the integrated lr is ≈ 0.55 of peak):

| steps | constant lr | cosine-corrected |
|---:|---:|---:|
| 10,000 | ×0.78 | ×0.87 |
| 50,000 | ×0.30 | ×0.51 |
| 100,000 | ×0.09 | ×0.26 |

**Readout 1 is broken.** `ame_experiments.md` §8.1's kill criterion is that `α₂, β₂`
collapsing to zero means the model is declining the mechanism. Weight decay drives them to
zero by itself, hardest in exactly the head where the gradient is weakest — the head that
is indifferent to the shift. A collapse is then not evidence of anything. This is the
project's **first experiment**, so the confound lands on the first result.

**Readout 2 is broken, and silently.** `tau` shrinks, but `slopes` is computed **once**
from `tau_init` and frozen ([ame.py:194-214](../../src/core/ame.py#L194)). A head's horizon
is `tau/m`, so every head's reach contracts with τ while the ladder stays put. At h20
(τ_init = 8, flattest slope 9.77e-4, horizon 8192) the cosine-corrected trajectory gives
~4190 tokens at 50k steps, and the horizon falls below the 2048 training context at
~100k steps. That is a progressive loss of precisely the long-range capability the method
is pitched on, and the `horizons()` readout that would catch it is **not wired to the
metric logger**.

**Third-order, worth knowing.** With QK-norm the bank rows are scale-invariant, so decay
on the banks does not change the function at all — it only shrinks `‖row‖`, which inflates
the effective learning rate on row *direction*. It also breaks Stage 1's "rare-type rows
stay at init" criterion: they do not stay, they shrink toward zero while keeping their
random direction.

**Context, so this is not mis-filed as an AME bug.** The repo decays everything, including
`RMSNorm` weights (torch's `RMSNorm`, 1-D affine, in `parameters()`). This is a
pre-existing repo-wide choice. AME is where it does scientific damage, because AME is the
model whose research question is literally *"does this coefficient go to zero?"*.

### Proposed fix

A second class-level declaration beside the existing `non_matmul_parameters`, and
parameter groups built from it:

```python
class AMEAttention(nn.Module):
    non_matmul_parameters = ("alpha", "beta", "k_sink", "v_sink")
    # Scales and per-head scalars, not weights. Decoupled decay moves these with
    # no gradient, which would drive alpha_2/beta_2 to zero on its own -- the exact
    # readout ame_experiments.md 8.1 reads as "the mechanism is inert".
    no_weight_decay_parameters = ("tau", "alpha", "beta", "slopes", "k_sink", "v_sink")
```

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
    decayed = [p for p in model.parameters() if id(p) not in undecayed]
    return [
        {"params": decayed, "weight_decay": weight_decay},
        {"params": [p for p in model.parameters() if id(p) in undecayed],
         "weight_decay": 0.0},
    ]
```

Non-AME models declare nothing, so **gqa baselines stay bit-identical** and no existing
number moves.

**One decision left to you: the banks.** The argument for `wd=0` on them is the
scale-invariance above — decay buys no regularization through a normalized dot product and
costs the "rows stay at init" diagnostic. The argument against is that it is a real
deviation from the dense baseline's treatment of its `W_Q`/`W_K`, which `baselines.md`
cares about. I would exclude τ/α/β/sinks now (they are unambiguous) and settle the banks
before Stage 1 rather than inside it.

**Related, not confirmed — worth measuring before Stage 1.** `gradient_clipping: 1.0` is a
single global norm over all 3.16B parameters, of which the banks are 82%. The banks
therefore set the clip scale, and the backbone's updates get scaled by whatever the banks
are doing. One logged grad-norm split (banks vs backbone) would say whether this matters;
it is cheap and it rhymes with the per-block clipping work in
`project_pc_gradient_clipping`.

---

## F3 — `_shift1` is silently wrong off its 4-D path

The docstring promises `[B, n, ...]`, but the pad tuple is hard-coded to six entries, which
reaches dim 1 only when the input is exactly 4-D. Measured:

```
3-D (2,4,3) -> (3,3,3)     # pads the BATCH dim, slices n. No exception.
4-D (2,4,3,2) -> (2,4,3,2) # correct: position 0 zeroed, no wrap
```

Not live — the only caller passes `[B, n, H, D]`. It is a trap for the next caller, and
§4.1's "one-step raw-`v` register" is exactly that caller.

### Proposed fix

```python
def _shift1(z):
    pad = [0, 0] * (z.dim() - 2) + [1, 0]   # always dim 1, whatever the rank
    return F.pad(z, pad)[:, :-1]
```

plus a test at 3-D and 5-D.

---

## F4 — an unrelated hunk rides along in `base.yaml`

`configs/_model/h_balanced/base.yaml` in the worktree contains, besides the two AME lines:

```diff
-  sequence_length: ???
+  sequence_length: 2048
+  batch_size: 512
```

This is **not** the other agent's change — the main tree has no diff for this file at all,
so it is mine. Nothing needs it: all eleven configs that use `h_balanced` already set both
values themselves, `configs/ame/h20_smoke.yaml` included.

What it costs: `???` is OmegaConf's mandatory-value marker. Removing it means a config that
forgets `sequence_length` now silently trains at 2048 instead of failing at composition.
Same for `batch_size`, which `configs/_dataset/default.yaml` interpolates as
`total_batch_size`. That is a loud-to-silent regression for every h-family config, traded
for nothing — and it cuts against the submit-time config validation this repo is otherwise
moving toward.

### Proposed fix

Revert that hunk. Keep only the two AME-relevant edits: `vocab_size: ${common.vocab_size}`
under `attention_fn`, and the options-block comment.

---

## F5 — a cryptic failure when the ids do not arrive

An AME layer called with `token_ids=None` — reachable by calling `TransformerEncoder` or a
block directly, since the parameter defaults to `None` all the way down — dies with
`AttributeError: 'NoneType' object has no attribute 'shape'` inside `_mixed_qkv`.

### Proposed fix

```python
def forward(self, x, token_ids):
    if token_ids is None:
        raise ValueError(
            "AME attention needs token ids: its Q/K banks are indexed by token id. "
            "The caller must thread input_ids through (see LLM.forward)."
        )
```

---

## F6 — option blocks for other types are accepted and dropped

`build_attention`'s unknown-key check allows a key named after **any** registered type:

```python
unknown = set(attention) - {"type", "mix"} - set(ATTENTION_TYPES)
```

but only `_build_ame` reads `options`. So `attention: {type: gqa, gqa: {...}}` or
`{type: mla, mla: {...}}` composes, validates, and silently discards the block — the
failure mode a sweep hits at 3 a.m.

### Proposed fix

Validate against the types that actually declare options:

```python
_TYPE_OPTIONS = {"ame": _AME_OPTIONS}
unknown = set(attention) - {"type", "mix"} - set(_TYPE_OPTIONS)
```

so a block for an option-less type is named as an error.

---

## F7 — corrections to `ame_implementation.md`

1. **"the baseline's 16"** — the gqa baseline's micro-batch is `512 / 64 = 8`, not 16
   (`configs/mfu_audit/h20.yaml`). AME's is 4. The ratio is 2×, not 4×.
2. **"flex without `torch.compile` warns"** — torch **2.9** warns; torch **2.7.1**, the
   pinned pixi version the cluster runs, does **not**. Stated as written it implies the
   h20 run would have told us, and it could not have.
3. **The two-candidate-cause list needs F1 as a third**, listed first, and the step-time
   table should be marked void pending a compiled re-run.

---

## What is verified good

Recorded so it does not get re-reviewed.

- **Full CPU suite green** (exit 0), including the end-to-end training smoke tests, which
  run only because the `data` / `data_eval` symlinks are in place.
- **The attention mathematics matches §0.** Shift direction (position *j* mixes in *j−1*),
  ALiBi sign (distant keys more negative), QK-norm on the rows before the mix, τ folded
  into `q`, the sink unbiased and visible from position 0, causal `j ≤ i`. The spec-oracle
  test is a literal transcription that shares no code with the implementation.
- **The two backends agree exactly** — max abs difference **0.0** in fp32 on CPU, across
  shift and sink variants. (Eager flex does run on CPU; `auto` avoids it there for
  compile reasons, not correctness ones.)
- **`learnable_slopes` gradients flow through both backends** and agree to 1e-5. The
  in-place `bias[:, :, :offset] = 0.0` on an autograd-tracked tensor is safe.
- **FSDP2 holds**, tested on a gloo mesh with the production
  `MixedPrecisionPolicy(bf16, fp32)`: forward and backward succeed and every parameter
  gets a gradient for `qk_norm=True`, `qk_norm=False` (the **mixed `requires_grad`** case —
  frozen τ — which is the one I expected to fail), `shift=False`, and `pin_alpha=True`.
- **`torch.compile(fullgraph=True)` traces the reference path.** The generator return in
  `_mixed_qkv` is not a graph break.
- **Config plumbing holds under struct-mode OmegaConf**: `attention.get("gqa")` returns
  `None` rather than raising, structured and plain-dict configs both build, a YAML list of
  slopes survives, a typo'd option name raises with the valid set printed, and a missing
  `vocab_size` raises a message that names the fix. `vocab_size` is threaded in the only
  config that defines `attention_fn`.
- **The shift crosses document boundaries**, which deviates from `ame.md` §7's `cu_seqlens`
  requirement — but this repo does no document masking anywhere (`is_causal=True` and
  nothing else), so AME is consistent with the baseline it is measured against. It is a
  known open item, not a differential bug, and it should not be fixed in AME alone.

## Fix plan

The remediation plan lives in its own doc: [`ame_fix_plan.md`](ame_fix_plan.md) — four
phases, what each one changes, and what it blocks. Short version: **Phase 1** (compile
flex) blocks every performance claim; **Phase 2** (weight-decay groups) blocks Stage 0;
Phase 3 is a revert and Phase 4 is cleanup. Phases 1 and 2 are independent, so if Stage 0
should move before a GPU frees up, Phase 2 goes first.
