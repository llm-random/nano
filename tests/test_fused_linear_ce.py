"""O11, fused-linear-CE: the section 1.4 gate.

The claim this path makes is that chunking the head projection and the
cross-entropy changes *memory* and nothing else. These tests hold it to that,
in the order section 1.4 demands -- equivalence first, throughput never quoted
until it passes.

Everything here runs on CPU in float64. The claim is **the same function, in a
different summation order** -- not bit-identity, which chunking cannot give and
should not be asked for: a chunked sum reassociates, and the measured deviation
is one ulp on the loss (8.88e-16 at a value of 4.18) and ~1e-17 on the
gradients. So the tolerance here is a few ulps of float64, and every test that
uses one carries a **negative control** showing that a real defect -- the wrong
divisor, a split vocabulary axis -- misses by orders of magnitude more. A
tolerance without that control would be the thing worth distrusting.
"""

# A few ulps of float64. The observed reassociation is 8.9e-16 on the loss and
# 4.2e-17 on the gradients; the negative controls below miss by 1e-2 and up.
ULPS = 1e-13

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.model import TransformerHead


VOCAB = 71
DMODEL = 16


def _head(dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    head = TransformerHead(
        linear_fn=lambda: nn.Linear(DMODEL, VOCAB, bias=False),
        norm_fn=lambda: nn.LayerNorm(DMODEL),
    )
    return head.to(dtype)


def _batch(batch=3, seq=8, dtype=torch.float64, seed=1, ignore=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, seq, DMODEL, dtype=dtype, generator=generator)
    targets = torch.randint(0, VOCAB, (batch, seq), generator=generator)
    if ignore:
        targets[0, 0] = -100
        targets[-1, -1] = -100
    return x, targets


def _reference_loss(head, x, targets):
    """The path O11 replaces: build every logit, then one cross-entropy."""
    logits = head(x)
    return F.cross_entropy(
        logits.reshape(-1, VOCAB), targets.reshape(-1), ignore_index=-100
    )


@pytest.mark.parametrize("chunk", [1, 5, 7, 24, 10_000])
def test_the_loss_is_identical_at_every_chunk_size(chunk):
    """Chunking is a summation order, not a different objective.

    Every chunk size must give the same number as the unfused path, including
    sizes that do not divide the row count (7 and 5 against 24 rows) and one
    larger than the whole batch.
    """
    head = _head()
    x, targets = _batch()

    reference = _reference_loss(head, x, targets)
    fused = head(x, targets=targets, chunk_size=chunk)

    delta = (fused - reference).abs().item()
    assert delta < ULPS, (
        f"chunk_size={chunk}: fused {fused.item()!r} != reference "
        f"{reference.item()!r} (delta {delta:.3e})"
    )


@pytest.mark.parametrize("chunk", [1, 5, 24])
def test_every_gradient_is_identical_at_every_chunk_size(chunk):
    """The gradient is what a wrong implementation gets wrong silently.

    The loss can be right while `dL/dW` is scaled or truncated -- a chunked
    reduction that averaged per chunk instead of summing would land within a
    few percent on the loss and be badly wrong here. So both the input and the
    projection weight are checked, exactly.
    """
    head_ref = _head()
    head_fused = _head()  # same seed, so same initial weights

    x_ref, targets = _batch()
    x_fused = x_ref.clone()
    x_ref.requires_grad_(True)
    x_fused.requires_grad_(True)

    _reference_loss(head_ref, x_ref, targets).backward()
    head_fused(x_fused, targets=targets, chunk_size=chunk).backward()

    delta = (x_fused.grad - x_ref.grad).abs().max().item()
    assert (
        delta < ULPS
    ), f"chunk_size={chunk}: input gradient differs, max |delta| {delta:.3e}"
    for (name, fused_p), (_, ref_p) in zip(
        head_fused.named_parameters(), head_ref.named_parameters()
    ):
        delta = (fused_p.grad - ref_p.grad).abs().max().item()
        assert delta < ULPS, (
            f"chunk_size={chunk}: gradient of {name} differs, max |delta| "
            f"{delta:.3e}"
        )

    # Negative control: a chunked reduction that averaged per chunk instead of
    # summing would scale dW by roughly the chunk count. That is what the
    # tolerance above must not admit.
    scaled = head_fused.linear.weight.grad * (len(range(0, 24, chunk)))
    if chunk != 24:
        assert (scaled - head_ref.linear.weight.grad).abs().max().item() > 1e-6


def test_ignored_targets_are_excluded_from_the_mean_not_just_the_sum():
    """`reduction="mean"` divides by the number of VALID targets.

    The chunked path sums per chunk and divides once at the end, so it has to
    count the ignored entries itself. Dividing by the row count instead would
    be a quiet, small, always-low loss -- the kind of bug that survives review.
    """
    head = _head()
    x, targets = _batch(ignore=True)
    assert (targets == -100).sum() == 2, "fixture should ignore exactly two"

    reference = _reference_loss(head, x, targets)
    fused = head(x, targets=targets, chunk_size=5)

    assert (fused - reference).abs().item() < ULPS

    # Negative control: dividing by all rows instead of valid ones. It misses
    # by ~1e-2 -- eleven orders of magnitude above the tolerance, which is the
    # margin that makes the tolerance safe.
    rows = targets.numel()
    wrong = fused * (rows - 2) / rows
    assert (wrong - reference).abs().item() > 1e-3, "negative control too close"


def test_a_chunk_boundary_cannot_fall_inside_a_row():
    """Chunking splits rows, never the vocabulary.

    A chunk that cut the vocab axis would renormalise each piece's softmax
    separately and produce a smaller loss. This pins it by making the row count
    coprime with every chunk size tried.
    """
    head = _head()
    x, targets = _batch(batch=1, seq=23)
    reference = _reference_loss(head, x, targets)
    for chunk in (2, 3, 4, 11, 23):
        fused = head(x, targets=targets, chunk_size=chunk)
        delta = (fused - reference).abs().item()
        assert delta < ULPS, f"chunk_size={chunk}, delta {delta:.3e}"


def test_targets_none_is_the_original_path_untouched():
    """Every committed number was produced by this branch; it must not move."""
    head = _head()
    x, _ = _batch()
    logits = head(x)
    assert logits.shape == (3, 8, VOCAB)
    manual = head.linear(head.norm(x))
    assert torch.equal(logits, manual)


def test_the_logits_tensor_is_never_materialised():
    """The point of O11, asserted rather than assumed.

    A chunked implementation that still built the full projection would pass
    every numerical test above and buy nothing. This watches the linear's
    output sizes: with a chunk of 5 rows, nothing wider than 5 rows may appear.
    """
    head = _head()
    x, targets = _batch(batch=3, seq=8)  # 24 rows

    # Probes BOTH the module and the functional form. The default path (v1)
    # calls `self.linear(...)`, which a module hook sees; `_ChunkedLinearCE`
    # calls `F.linear` directly, which it does not. Watching only one of them
    # would pass vacuously the moment the implementation switched -- which is
    # exactly what happened once already during this work.
    import src.core.model as model_module

    real_linear = model_module.F.linear
    seen = []

    def spy(hidden, weight, bias=None):
        out = real_linear(hidden, weight, bias)
        seen.append(out.shape[0])
        return out

    handle = head.linear.register_forward_hook(
        lambda module, args, output: seen.append(output.shape[0])
    )
    model_module.F.linear = spy
    try:
        head(x, targets=targets, chunk_size=5)
    finally:
        model_module.F.linear = real_linear
        handle.remove()

    assert seen, "the projection was never called -- the probe missed it"
    assert max(seen) <= 5, f"a projection of {max(seen)} rows was built, chunk was 5"


def test_float32_matches_within_reassociation_noise():
    """Outside float64 the claim weakens honestly: same function, different
    summation order, so the difference is reassociation noise rather than zero.
    """
    head = _head(dtype=torch.float32)
    x, targets = _batch(dtype=torch.float32)
    reference = _reference_loss(head, x, targets)
    fused = head(x, targets=targets, chunk_size=5)
    assert torch.allclose(
        fused, reference, rtol=0, atol=1e-6
    ), f"fused {fused.item()!r} vs reference {reference.item()!r}"


def test_the_schema_rejects_fused_ce_together_with_compile_ce():
    """`compile_ce` compiles a cross-entropy over logits that `fused_linear_ce`
    never builds. Both set is a configuration error, not a silent precedence."""
    from types import SimpleNamespace

    from src.definitions import validate

    def config(**trainer):
        base = dict(
            fused_linear_ce=False,
            compile_ce=False,
            fused_ce_chunk_size=None,
            mem_parallel_size=None,
            # The rest of what `validate` reads off cfg.trainer. Present and
            # valid so the only problems it can report are the ones under test.
            learning_rate=0.001,
            n_steps=10,
            n_eval_steps=1,
            eval_interval=0,
            gradient_accumulation_steps=1,
            gradient_clipping=1.0,
            weight_decay=0.1,
            scheduler=SimpleNamespace(warmup_steps=1),
            checkpoint=SimpleNamespace(save=SimpleNamespace(interval=-1)),
            save=None,
        )
        base.update(trainer)
        return SimpleNamespace(
            common=SimpleNamespace(
                h=4,
                n_blocks=4,
                dmodel=256,
                dff=1024,
                dhead=64,
                q_heads=4,
                kv_heads=2,
                vocab_size=VOCAB,
                sequence_length=8,
                batch_size=2,
            ),
            trainer=SimpleNamespace(**base),
        )

    # `validate` raises on any problem rather than returning a list.
    with pytest.raises(ValueError, match="compile_ce"):
        validate(config(fused_linear_ce=True, compile_ce=True))

    # And the combination that is legitimate must pass cleanly, or the check
    # above would be rejecting the feature rather than the conflict.
    validate(config(fused_linear_ce=True, fused_ce_chunk_size=2048))

    with pytest.raises(ValueError, match="fused_ce_chunk_size"):
        validate(config(fused_linear_ce=True, fused_ce_chunk_size=0))
