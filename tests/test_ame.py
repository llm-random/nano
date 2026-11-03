"""AME's math: the properties the method depends on, not just its shapes.

Most of these guard a *silent* failure -- a layer that trains, produces finite
loss and is quietly the wrong model. The spec they check against is
`documents/notes/ame_implementation.md` (and the research record behind it); the
canonical forward pass is transcribed literally into `_spec_forward` below, so a
refactor of src/core/ame.py has to keep agreeing with the specification rather
than with itself.

The expensive ones to lose are:

- scores must be a function of (token ids, distance) **only** -- that property is
  the entire method, and nothing else in the suite would notice its loss;
- QK-norm must land on the bank rows *before* the shift mix, so a score stays a
  linear combination of two entries of one row of T;
- the shift must not wrap across the sequence boundary.
"""

import math
import os

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import resolver  # noqa: F401  (registers the eval resolver the configs use)
from src.core.ame import (
    AMEAttention,
    _diagnostic_rows,
    _shift1,
    horizon_ladder,
)
from src.core.model import RoPEAttention, parameter_groups

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VOCAB = 64
DIMS = dict(dmodel=32, q_heads=4, kv_heads=2, seq_len=10)


def _layer(seed=0, **options):
    torch.manual_seed(seed)
    options.setdefault("backend", "reference")
    return AMEAttention(**DIMS, vocab_size=VOCAB, **options)


def _batch(layer, batch=2, seed=1):
    generator = torch.Generator().manual_seed(seed)
    token_ids = torch.randint(0, VOCAB, (batch, DIMS["seq_len"]), generator=generator)
    x = torch.randn(batch, DIMS["seq_len"], DIMS["dmodel"], generator=generator)
    return x, token_ids


def _spec_forward(layer, x, token_ids):
    """The canonical forward pass, transcribed line by line from the spec.

    Deliberately naive -- einsums, a materialized mask, an explicit zero-fill
    shift. It exists to disagree with src/core/ame.py if the implementation
    drifts, so it must not share code with it.
    """
    batch, seq_len = token_ids.shape
    heads, dhead = layer.q_heads, layer.dhead

    q = layer.q_bank.weight[token_ids].view(batch, seq_len, heads, dhead)
    k = layer.k_bank.weight[token_ids].view(batch, seq_len, heads, dhead)
    v = (x @ layer.v_proj.weight.T).view(batch, seq_len, heads, dhead)

    q = q / q.norm(dim=-1, keepdim=True)  # QK-norm on the bank rows ...
    k = k / k.norm(dim=-1, keepdim=True)

    def shift1(z):  # ... before the mix
        shifted = torch.zeros_like(z)
        shifted[:, 1:] = z[:, :-1]
        return shifted

    alpha, beta = layer.alpha, layer.beta
    k = alpha[:, 0].view(1, 1, -1, 1) * k + alpha[:, 1].view(1, 1, -1, 1) * shift1(k)
    v = beta[:, 0].view(1, 1, -1, 1) * v + beta[:, 1].view(1, 1, -1, 1) * shift1(v)

    scores = torch.einsum("bihd,bjhd->bhij", q, k) * layer.tau.view(1, -1, 1, 1)
    i = torch.arange(seq_len).view(-1, 1)
    j = torch.arange(seq_len).view(1, -1)
    scores = scores - layer.slopes.view(1, -1, 1, 1) * (i - j).float().view(
        1, 1, seq_len, seq_len
    )
    scores = scores.masked_fill((j > i).view(1, 1, seq_len, seq_len), float("-inf"))
    weights = scores.softmax(dim=-1)
    out = torch.einsum("bhij,bjhd->bihd", weights, v)
    return out.reshape(batch, seq_len, heads * dhead) @ layer.o_proj.weight.T


# ---------------------------------------------------------------------------
# The forward pass is the forward pass in the spec.


def test_the_forward_matches_a_literal_transcription_of_the_spec():
    layer = _layer(sink=False)
    x, token_ids = _batch(layer)
    torch.testing.assert_close(
        layer(x, token_ids), _spec_forward(layer, x, token_ids), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("sink", [False, True])
@pytest.mark.parametrize("shift", [False, True])
def test_the_two_backends_compute_the_same_function(sink, shift):
    """flex applies the distance bias inside the kernel, reference materializes
    it. They are the same model or one of them is wrong."""
    layer = _layer(sink=sink, shift=shift)
    x, token_ids = _batch(layer)
    layer.backend = "reference"
    reference = layer(x, token_ids)
    layer.backend = "flex"
    flex = layer(x, token_ids)
    torch.testing.assert_close(reference, flex, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Scores are a function of (ids, distance) and nothing else. This IS the method.


def test_a_score_depends_only_on_the_token_pair_and_the_distance():
    """Two positions holding the same token must score identically against the
    same query, at the same distance. If a hidden state ever reaches the scores
    this fails -- and nothing else in the suite would catch it."""
    layer = _layer(shift=False, sink=False, slopes=[0.0] * DIMS["q_heads"])
    token_ids = torch.tensor([[5, 9, 5, 9, 5, 9, 5, 9, 5, 9]])
    scores = layer.score_matrix(token_ids)[0]  # [H, n, n]
    # every (query=5, key=9) pair, at any distance, is the same number
    for head in range(layer.q_heads):
        pairs = [scores[head, i, j] for i in (4, 6, 8) for j in (1, 3, 5) if j <= i]
        assert len(pairs) > 1
        for value in pairs[1:]:
            torch.testing.assert_close(value, pairs[0], rtol=0, atol=1e-6)


def test_the_values_carry_context_but_the_weights_do_not():
    """Option B's whole claim: change x, keep the ids, and the attention
    weights must not move (only the payload does)."""
    layer = _layer()
    x, token_ids = _batch(layer)
    first = layer.score_matrix(token_ids)
    other_x, _ = _batch(layer, seed=99)
    second = layer.score_matrix(token_ids)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.allclose(layer(x, token_ids), layer(other_x, token_ids))


# ---------------------------------------------------------------------------
# QK-norm goes on the rows, before the mix.


def test_rescaling_a_bank_row_does_not_move_any_score():
    """ell2 on the rows makes a row's norm irrelevant. If the norm were applied
    to the mixed key instead, scaling one row would move the mix's norm and
    every score with it."""
    layer = _layer()
    token_ids = torch.randint(0, VOCAB, (2, DIMS["seq_len"]))
    before = layer.score_matrix(token_ids)
    with torch.no_grad():
        layer.k_bank.weight[int(token_ids[0, 0])] *= 7.5
        layer.q_bank.weight[int(token_ids[0, 1])] *= 0.3
    torch.testing.assert_close(
        before, layer.score_matrix(token_ids), rtol=1e-5, atol=1e-6
    )


def test_a_score_is_a_linear_combination_of_two_entries_of_one_row():
    """The decode-time property, checked at training time: with QK-norm on the
    rows, s_ij = tau * (a1*T[t_i,t_j] + a2*T[t_i,t_{j-1}]). Normalizing the
    mixed key instead would add a per-bigram norm and break this identity --
    the single highest-cost mistake available in this design."""
    layer = _layer(slopes=[0.0] * DIMS["q_heads"], sink=False)
    token_ids = torch.randint(0, VOCAB, (1, DIMS["seq_len"]))
    scores = layer.score_matrix(token_ids)[0]

    rows_q = torch.nn.functional.normalize(
        layer.q_bank.weight[token_ids[0]].view(-1, layer.q_heads, layer.dhead), dim=-1
    )
    rows_k = torch.nn.functional.normalize(
        layer.k_bank.weight[token_ids[0]].view(-1, layer.q_heads, layer.dhead), dim=-1
    )
    table = torch.einsum("ihd,jhd->hij", rows_q, rows_k)  # T[t_i, t_j] per head

    a1 = layer.alpha[:, 0].view(-1, 1, 1)
    a2 = layer.alpha[:, 1].view(-1, 1, 1)
    shifted = torch.zeros_like(table)
    shifted[:, :, 1:] = table[:, :, :-1]
    expected = layer.tau.view(-1, 1, 1) * (a1 * table + a2 * shifted)

    causal = torch.arange(DIMS["seq_len"]).view(-1, 1) >= torch.arange(
        DIMS["seq_len"]
    ).view(1, -1)
    torch.testing.assert_close(
        scores[:, causal], expected[:, causal], rtol=1e-5, atol=1e-5
    )


# ---------------------------------------------------------------------------
# The shift: what it reads, and that it never wraps.


def test_a_fully_shifted_head_reads_its_predecessors_key():
    layer = _layer(sink=False, slopes=[0.0] * DIMS["q_heads"])
    with torch.no_grad():
        layer.alpha.copy_(torch.tensor([[0.0, 1.0]] * DIMS["q_heads"]))
    token_ids = torch.randint(0, VOCAB, (1, DIMS["seq_len"]))
    scores = layer.score_matrix(token_ids)[0]

    unshifted = _layer(sink=False, slopes=[0.0] * DIMS["q_heads"])
    unshifted.load_state_dict(layer.state_dict())
    with torch.no_grad():
        unshifted.alpha.copy_(torch.tensor([[1.0, 0.0]] * DIMS["q_heads"]))
    plain = unshifted.score_matrix(token_ids)[0]

    # The shifted head's score at key j is the unshifted score at key j-1,
    # wherever both are inside the causal mask (j <= i, so j-1 <= i-1).
    for query in range(1, DIMS["seq_len"]):
        torch.testing.assert_close(
            scores[:, query, 1 : query + 1],
            plain[:, query, :query],
            rtol=1e-5,
            atol=1e-5,
        )


def test_the_shift_zeroes_position_zero_instead_of_wrapping():
    """A roll would pull the last token of the row into position 0. Under this
    repo's document packing that is a silent corruption, so the shifted
    component at position 0 must be exactly zero."""
    layer = _layer(sink=False, slopes=[0.0] * DIMS["q_heads"])
    with torch.no_grad():
        layer.alpha.copy_(torch.tensor([[0.0, 1.0]] * DIMS["q_heads"]))
    ids = torch.randint(0, VOCAB, (1, DIMS["seq_len"]))
    scores_a = layer.score_matrix(ids)[0, :, :, 0]
    changed = ids.clone()
    changed[0, -1] = (int(ids[0, -1]) + 1) % VOCAB  # a different LAST token
    scores_b = layer.score_matrix(changed)[0, :, :, 0]
    # column 0's key is the (zero) shifted component either way: score exactly 0
    torch.testing.assert_close(scores_a, torch.zeros_like(scores_a), rtol=0, atol=1e-6)
    torch.testing.assert_close(scores_a, scores_b, rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------
# Ablation arms: the four-arm design needs alpha and beta pinned separately.


@pytest.mark.parametrize(
    "options, alpha_pinned, beta_pinned",
    [
        ({}, False, False),  # A1: the method as specified
        ({"pin_beta": True}, False, True),  # A2: key-side mix alone
        ({"pin_alpha": True}, True, False),  # A3: value-side mix alone
        ({"pin_alpha": True, "pin_beta": True}, True, True),  # A4: fixed-kernel floor
        ({"shift": False}, True, True),  # shift off == A4
    ],
)
def test_each_arm_pins_exactly_what_it_claims(options, alpha_pinned, beta_pinned):
    layer = _layer(**options)
    assert layer.alpha_is_pinned is alpha_pinned
    assert layer.beta_is_pinned is beta_pinned
    names = dict(layer.named_parameters())
    assert ("alpha" in names) is not alpha_pinned
    assert ("beta" in names) is not beta_pinned
    for name, pinned in (("alpha", alpha_pinned), ("beta", beta_pinned)):
        if pinned:
            torch.testing.assert_close(
                getattr(layer, name),
                torch.tensor([[1.0, 0.0]] * DIMS["q_heads"]),
                rtol=0,
                atol=0,
            )


def test_a_pinned_coefficient_cannot_drift():
    """A4 must stay A4 for the whole run: a buffer takes no gradient."""
    layer = _layer(pin_alpha=True, pin_beta=True)
    x, token_ids = _batch(layer)
    layer(x, token_ids).sum().backward()
    assert layer.alpha.grad is None and layer.beta.grad is None
    torch.testing.assert_close(
        layer.alpha, torch.tensor([[1.0, 0.0]] * DIMS["q_heads"]), rtol=0, atol=0
    )


def test_pinning_both_arms_is_the_same_model_as_turning_the_shift_off():
    pinned = _layer(pin_alpha=True, pin_beta=True)
    off = _layer(shift=False)
    off.load_state_dict(pinned.state_dict())
    x, token_ids = _batch(pinned)
    torch.testing.assert_close(pinned(x, token_ids), off(x, token_ids), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Causality, initialization scale, the slope ladder, the sink.


def test_an_output_never_sees_a_later_token():
    layer = _layer()
    x, token_ids = _batch(layer, batch=1)
    before = layer(x, token_ids)
    changed_ids = token_ids.clone()
    changed_ids[0, -1] = (int(token_ids[0, -1]) + 1) % VOCAB
    changed_x = x.clone()
    changed_x[0, -1] += 3.0
    after = layer(changed_x, changed_ids)
    torch.testing.assert_close(before[:, :-1], after[:, :-1], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(before[:, -1], after[:, -1])


def test_scores_start_at_std_one_with_the_prescribed_init():
    """E0.6. sigma ~ 1 on the rows, QK-norm on, tau = sqrt(dhead): scores of
    std ~1. The embedding habit (sigma = 0.02, no QK-norm) gives ~4e-4 --
    attention uniform at init, symmetry possibly never broken."""
    big = AMEAttention(
        dmodel=256,
        q_heads=4,
        kv_heads=4,
        seq_len=64,
        vocab_size=512,
        sink=False,
        slopes=[0.0] * 4,
        backend="reference",
    )
    ids = torch.randint(0, 512, (4, 64))
    prescribed = big.score_matrix(ids)
    assert 0.5 < prescribed[torch.isfinite(prescribed)].std().item() < 2.0

    habit = AMEAttention(
        dmodel=256,
        q_heads=4,
        kv_heads=4,
        seq_len=64,
        vocab_size=512,
        sink=False,
        slopes=[0.0] * 4,
        qk_norm=False,
        bank_init_std=0.02,
        backend="reference",
    )
    uniform = habit.score_matrix(ids)
    assert uniform[torch.isfinite(uniform)].std().item() < 1e-2


def test_the_slope_ladder_spans_from_a_local_head_to_beyond_the_context():
    """A head competes out to ~tau/m tokens. The standard ALiBi ladder caps the
    flattest head near 3k whatever the context is; this one is derived from the
    context, so it must actually reach past it."""
    heads, seq_len, tau = 8, 2048, math.sqrt(64)
    slopes = horizon_ladder(heads, tau, seq_len, 16.0, 4.0)
    horizons = tau / slopes
    assert horizons[0] == pytest.approx(16.0)
    assert horizons[-1] == pytest.approx(4 * seq_len)
    assert torch.all(slopes[:-1] > slopes[1:]), "steepest head first, monotone"

    layer = AMEAttention(
        dmodel=512, q_heads=heads, kv_heads=heads, seq_len=seq_len, vocab_size=32
    )
    torch.testing.assert_close(layer.horizons()[0], torch.tensor(16.0))
    torch.testing.assert_close(layer.horizons()[-1], torch.tensor(4.0 * seq_len))


def test_the_sink_is_unbiased_and_visible_from_the_first_position():
    """A head that learns alpha_1 -> 0 has no finite key at position 0 and needs
    somewhere to put its mass; the sink must also escape the distance bias, or
    it is evicted at long context exactly as StreamingLLM documents."""
    layer = _layer(sink=True)
    ids = torch.randint(0, VOCAB, (1, DIMS["seq_len"]))
    scores = layer.score_matrix(ids)
    assert scores.shape[-1] == DIMS["seq_len"] + 1, "sink occupies kv column 0"
    assert torch.isfinite(scores[0, :, 0, 0]).all(), "visible to query 0"
    # unbiased: a zero-init sink key scores 0 against every query, at any distance
    torch.testing.assert_close(
        scores[0, :, :, 0],
        torch.zeros(layer.q_heads, DIMS["seq_len"]),
        atol=1e-6,
        rtol=0,
    )
    assert not layer.use_sink or "k_sink" in dict(layer.named_parameters())


def test_without_a_sink_there_is_no_extra_column():
    layer = _layer(sink=False)
    ids = torch.randint(0, VOCAB, (1, DIMS["seq_len"]))
    assert layer.score_matrix(ids).shape[-1] == DIMS["seq_len"]


# ---------------------------------------------------------------------------
# The contract with the rest of the framework.


def test_ame_asks_for_token_ids_and_a_projection_type_does_not():
    assert AMEAttention.wants_token_ids is True
    assert getattr(RoPEAttention, "wants_token_ids", False) is False


def test_the_banks_are_lookups_not_matmul_weights():
    """They are nn.Embeddings, which is how src.core.mfu knows not to charge
    6*P of matmul FLOPs for ~4x the backbone in parameters."""
    layer = _layer()
    assert isinstance(layer.q_bank, torch.nn.Embedding)
    assert isinstance(layer.k_bank, torch.nn.Embedding)
    assert layer.q_bank.weight.shape == (VOCAB, DIMS["q_heads"] * layer.dhead)


def test_a_bad_backend_name_is_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown ame backend"):
        _layer(backend="flashier")


def test_gradients_reach_every_part_of_the_layer():
    layer = _layer()
    x, token_ids = _batch(layer)
    layer(x, token_ids).sum().backward()
    missing = [name for name, p in layer.named_parameters() if p.grad is None]
    assert missing == []
    touched = token_ids.unique()
    rows = layer.q_bank.weight.grad.abs().sum(dim=-1)
    assert (rows[touched] > 0).all(), "every id in the batch moves its bank row"
    untouched = [i for i in range(VOCAB) if i not in set(touched.tolist())]
    assert rows[untouched].abs().max() == 0, "and no id outside the batch does"


# ---------------------------------------------------------------------------
# End to end: the config path, the token-id plumbing, and a backward pass.


def _h4_ame(overrides=(), vocab=96, device="cpu"):
    with initialize_config_dir(
        version_base=None,
        config_dir=os.path.join(REPO_ROOT, "configs", "_model", "h_balanced"),
    ):
        cfg = compose(
            config_name="h4",
            overrides=[
                "common.sequence_length=16",
                f"common.vocab_size={vocab}",
                "common.attention.type=ame",
                *overrides,
            ],
        )
    with torch.device(device):
        return instantiate(cfg.model, _convert_="all")


def test_an_ame_stack_trains_end_to_end_on_cpu():
    """Token ids have to reach every layer, not just the embedding -- this is
    the one test that fails if the plumbing in src/core/model.py regresses."""
    torch.manual_seed(0)
    model = _h4_ame()
    ids = torch.randint(0, 96, (2, 16))
    loss = torch.nn.functional.cross_entropy(model(ids).flatten(0, -2), ids.reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in model.parameters())
    bank = model.encoder.blocks[-1].attention_layer.layer.q_bank.weight
    assert bank.grad.abs().sum() > 0, "the deepest layer's bank saw the ids"


def test_options_from_the_config_reach_the_layer():
    model = _h4_ame(["+common.attention.ame={pin_alpha: true, sink: false}"])
    for block in model.encoder.blocks:
        layer = block.attention_layer.layer
        assert layer.alpha_is_pinned and not layer.beta_is_pinned
        assert layer.use_sink is False


def test_ame_can_be_mixed_with_gqa_in_one_stack():
    """The hybrid fallback has to be reachable: id-indexed layers interleaved
    with ordinary attention, both fed from the same forward."""
    torch.manual_seed(0)
    model = _h4_ame(["+common.attention.mix={type: gqa, layers: every_2nd}"])
    kinds = [type(b.attention_layer.layer).__name__ for b in model.encoder.blocks]
    assert kinds == ["AMEAttention", "RoPEAttention"] * 2
    ids = torch.randint(0, 96, (2, 16))
    model(ids).sum().backward()
    assert all(p.grad is not None for p in model.parameters())


# ---------------------------------------------------------------------------
# Precision and compilation: the production config runs under bf16 autocast and
# torch.compile, and neither was exercised by anything above.


@pytest.mark.parametrize("backend", ["reference", "flex"])
def test_bf16_autocast_does_not_mix_dtypes(backend):
    """Regression. Autocast casts nn.Linear but NOT an nn.Embedding gather, so
    `v` arrived bf16 while the banks handed back fp32: flex rejected the
    mismatch outright and the reference path silently ran half the layer in the
    wrong precision. This is a crash on the default path under the production
    trainer setting (`trainer.autocast_bf16=true`), not a cosmetic issue."""
    layer = _layer(backend=backend)
    x, token_ids = _batch(layer)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = layer(x, token_ids)
    assert out.dtype is torch.bfloat16
    assert torch.isfinite(out).all()


def test_the_backends_still_agree_under_bf16():
    """bf16 costs resolution, but it must cost the two paths the *same*
    resolution -- the distance bias is computed in fp32 inside the kernel for
    exactly this reason."""
    layer = _layer()
    x, token_ids = _batch(layer)
    layer.backend = "reference"
    reference = layer(x, token_ids)
    outputs = {}
    for backend in ("reference", "flex"):
        layer.backend = backend
        with torch.autocast("cpu", dtype=torch.bfloat16):
            outputs[backend] = layer(x, token_ids).float()
    scale = reference.abs().max()
    gap = (outputs["reference"] - outputs["flex"]).abs().max() / scale
    cost = (reference - outputs["reference"]).abs().max() / scale
    assert gap <= 4 * cost, "the backends disagree by more than bf16 itself costs"


def test_bf16_gradients_are_finite():
    layer = _layer()
    x, token_ids = _batch(layer)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        layer(x, token_ids).float().sum().backward()
    for name, parameter in layer.named_parameters():
        assert torch.isfinite(parameter.grad).all(), f"{name} has a non-finite grad"


def test_auto_picks_the_path_that_can_actually_train_on_this_device():
    """In torch 2.9 a compiled flex_attention whose inputs require grad fails to
    lower on CPU -- for a bare call with no score_mod and no block_mask, so it
    is the backend and not this layer. `auto` must therefore be device-aware, or
    the default config cannot be compiled on a CPU box."""
    layer = _layer(backend="auto")
    x, token_ids = _batch(layer)
    assert not x.is_cuda
    torch.testing.assert_close(
        layer(x, token_ids),
        layer._attend_reference(*layer._mixed_qkv(x, token_ids))
        .transpose(1, 2)
        .reshape(x.shape[0], x.shape[1], -1)
        @ layer.o_proj.weight.T,
        rtol=1e-6,
        atol=1e-7,
    )


# ---------------------------------------------------------------------------
# Weight decay. AdamW's decay is decoupled -- it moves a parameter whose gradient
# is exactly zero -- so a flat parameters() list hands the method's own first
# question (ame_experiments.md 8.1: do alpha_2 and beta_2 collapse to zero?) a
# prewritten answer. These guard the grouping that stops it.


def _declared_scales(model):
    """Every parameter the stack asked to keep out of weight decay.

    Resolves dotted names the same way src.core.model.parameter_groups does --
    AME names its banks as `q_bank.weight`, since the attribute itself is a
    module.
    """

    def resolve(module, name):
        for part in name.split("."):
            module = getattr(module, part, None)
            if module is None:
                return None
        return module

    found = set()
    for _, module in model.named_modules():
        for name in getattr(module, "no_weight_decay_parameters", ()):
            parameter = resolve(module, name)
            if isinstance(parameter, torch.nn.Parameter):
                found.add(id(parameter))
    return found


def test_a_model_that_declares_no_scales_gets_exactly_one_parameter_group():
    """The compatibility guarantee: every gqa/mha/mqa config keeps the optimizer
    it had, so no committed baseline number moves. Checked by identity and order
    rather than by count -- a second, empty group would also pass a count test
    and is not the same optimizer."""
    model = _h4_ame(["common.attention.type=gqa"])
    assert _declared_scales(model) == set(), "fixture is not scale-free"

    groups = parameter_groups(model, 0.1)
    expected = list(model.parameters())
    assert len(groups) == 1
    assert groups[0]["weight_decay"] == 0.1
    assert len(groups[0]["params"]) == len(expected)
    assert all(a is b for a, b in zip(groups[0]["params"], expected))


def test_an_ame_model_puts_its_scales_in_an_undecayed_group():
    model = _h4_ame()
    groups = parameter_groups(model, 0.1)

    assert [group["weight_decay"] for group in groups] == [0.1, 0.0]
    undecayed = {id(p) for p in groups[1]["params"]}
    decayed = {id(p) for p in groups[0]["params"]}
    assert undecayed == _declared_scales(model)
    assert undecayed.isdisjoint(decayed), "a parameter is in both groups"
    # Nothing may be dropped on the way: an optimizer that silently stops
    # updating a parameter is the worst outcome of a grouping bug.
    assert len(undecayed) + len(decayed) == len(list(model.parameters()))


def test_a_pinned_arm_contributes_no_undecayed_parameters():
    """A pinned alpha is a *buffer*, not a parameter. getattr answers for both,
    so a grouping that forgot to check would hand the optimizer a buffer."""
    model = _h4_ame(["+common.attention.ame={pin_alpha: true, pin_beta: true}"])
    groups = parameter_groups(model, 0.1)
    optimizer_params = {id(p) for group in groups for p in group["params"]}
    for block in model.encoder.blocks:
        layer = block.attention_layer.layer
        assert layer.alpha_is_pinned and layer.beta_is_pinned
        assert id(layer.alpha) not in optimizer_params
        assert id(layer.beta) not in optimizer_params


def test_the_banks_are_not_decayed():
    """Decided 2026-11-03 (ame_fix_plan.md 2.6): the banks are exempt.

    With QK-norm a row's length never reaches the score -- only its direction
    does -- so decay on the banks changes no output. It only shrinks ||row||
    every step, which inflates the effective learning rate on direction and
    means a row that is never gradient-updated does not stay at init: it decays
    toward zero carrying its random direction, which is exactly what Stage 1's
    "rare-type rows stay at init" check is supposed to detect.
    """
    model = _h4_ame()
    groups = parameter_groups(model, 0.1)
    decayed = {id(p) for p in groups[0]["params"]}
    undecayed = {id(p) for p in groups[1]["params"]}

    for block in model.encoder.blocks:
        layer = block.attention_layer.layer
        assert id(layer.q_bank.weight) in undecayed
        assert id(layer.k_bank.weight) in undecayed
        assert id(layer.q_bank.weight) not in decayed

    # The token embedding is a different table and keeps the repo's default.
    assert (
        id(model.embedding.weight) in decayed
    ), "only AME's banks were exempted; the input embedding is unchanged"
    # The projections are ordinary matmul weights and still decay.
    assert id(model.encoder.blocks[0].attention_layer.layer.v_proj.weight) in decayed


def test_decay_cannot_move_a_bank_row():
    """The property the exemption exists for, end to end on the optimizer."""
    torch.manual_seed(0)
    layer = _layer()
    optimizer = torch.optim.AdamW(parameter_groups(layer, 0.1), lr=2**-6)
    before = layer.q_bank.weight.detach().clone()
    for parameter in layer.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    assert (layer.q_bank.weight - before).abs().max().item() == 0.0


def test_decay_moves_a_scale_without_the_groups_and_cannot_with_them():
    """The regression, written as the difference the fix makes rather than as
    its result -- so it fails if parameter_groups is reverted, not only if it
    is broken. Gradients are exactly zero, so decay is the only thing that can
    move anything."""

    def drift(build_params):
        torch.manual_seed(0)
        layer = _layer()
        optimizer = torch.optim.AdamW(build_params(layer), lr=2**-6, weight_decay=0.1)
        before = layer.tau.detach().clone(), layer.alpha.detach().clone()
        for parameter in layer.parameters():
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        return (
            (layer.tau - before[0]).abs().max().item(),
            (layer.alpha - before[1]).abs().max().item(),
        )

    flat = drift(lambda layer: layer.parameters())
    assert min(flat) > 0, "nothing to protect against -- the premise is wrong"

    grouped = drift(lambda layer: parameter_groups(layer, 0.1))
    assert grouped == (0.0, 0.0), f"decay still reaches tau/alpha: {grouped}"


def test_the_lr_scheduler_accepts_the_grouped_optimizer():
    """The grouping is useless if the scheduler rejects it, and it does not fail
    until a whole run starts: `get_cosine_scheduler_with_warmup` asserted on the
    *group count*, which the end-to-end smoke test caught. The precondition it
    actually needs is one shared learning rate."""
    from src.core.schedulers import get_cosine_scheduler_with_warmup

    layer = _layer()
    optimizer = torch.optim.AdamW(parameter_groups(layer, 0.1), lr=2**-12)
    assert len(optimizer.param_groups) == 2, "fixture is not exercising the grouping"

    scheduler = get_cosine_scheduler_with_warmup(
        optimizer, warmup_steps=2, n_steps=10, final_lr_fraction=0.1
    )
    for _ in range(4):
        optimizer.step()
        scheduler.step()
    # Both groups must stay on the same schedule -- a per-group lr would mean the
    # scales train at a different rate than the weights, which nobody asked for.
    assert len({group["lr"] for group in optimizer.param_groups}) == 1
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.1, 0.0]


def test_the_scheduler_still_rejects_groups_at_different_learning_rates():
    """Relaxing the guard must not delete it: differing lrs are the case eta_min
    genuinely cannot express."""
    from src.core.schedulers import get_cosine_scheduler_with_warmup

    layer = _layer()
    optimizer = torch.optim.AdamW(
        [
            {"params": [layer.tau], "lr": 1e-3},
            {"params": [layer.alpha], "lr": 1e-4},
        ]
    )
    with pytest.raises(AssertionError, match="one learning rate"):
        get_cosine_scheduler_with_warmup(
            optimizer, warmup_steps=2, n_steps=10, final_lr_fraction=0.1
        )


# ---------------------------------------------------------------------------
# Instrumentation. horizons() and the alpha/beta trajectories existed but nothing
# read them, which makes both of the method's headline readouts unobservable on a
# real run (ame_experiments.md 7).

_READOUT_KEYS = {
    "alpha_2/mean",
    "alpha_2/min",
    "alpha_2/max",
    "alpha_2/absmean",
    "beta_2/mean",
    "beta_2/min",
    "beta_2/max",
    "beta_2/absmean",
    "tau/mean",
    "tau/min",
    "horizon/min",
    "horizon/max",
}


def test_the_instrumentation_reports_the_collapse_readout_and_the_horizons():
    layer = _layer()
    readout = layer.instrumentation()

    assert set(readout) == _READOUT_KEYS
    assert all(
        isinstance(value, float) for value in readout.values()
    ), "the metric logger takes scalars; a tensor here would log a repr"
    assert readout["alpha_2/mean"] == pytest.approx(layer.alpha[:, 1].mean().item())
    assert readout["beta_2/max"] == pytest.approx(layer.beta[:, 1].max().item())
    horizons = layer.horizons()
    assert readout["horizon/min"] == pytest.approx(horizons.min().item())
    assert readout["horizon/max"] == pytest.approx(horizons.max().item())


def test_the_collapse_readout_survives_heads_that_cancel():
    """Signed means cancel: a stack split into +c and -c heads would read as
    zero and be mistaken for a collapse. absmean is the one that must not."""
    layer = _layer()
    with torch.no_grad():
        layer.alpha[:, 1] = torch.tensor([0.5, -0.5, 0.5, -0.5])
    readout = layer.instrumentation()
    assert readout["alpha_2/mean"] == pytest.approx(0.0)
    assert readout["alpha_2/absmean"] == pytest.approx(0.5)


def test_the_instrumentation_reads_a_pinned_arm_as_collapsed():
    """The pinned arm IS the 'mechanism off' control, so the readout has to
    report it as zero -- otherwise the control and a real collapse look the
    same in the logs, which is the comparison Stage 0 rests on."""
    layer = _layer(pin_alpha=True)
    readout = layer.instrumentation()
    assert readout["alpha_2/absmean"] == 0.0
    assert readout["beta_2/absmean"] > 0.0


def test_only_an_id_indexed_layer_offers_instrumentation():
    """The trainer collects this by duck typing, so a projection layer must
    offer nothing at all -- not an empty dict, which would still cost a call."""
    model = _h4_ame(["+common.attention.mix={type: gqa, layers: every_2nd}"])
    offering = [
        name
        for name, module in model.named_modules()
        if hasattr(module, "instrumentation")
    ]
    assert len(offering) == 2, "expected exactly the two ame layers"
    assert all(isinstance(model.get_submodule(name), AMEAttention) for name in offering)
    assert not hasattr(RoPEAttention, "instrumentation")


def _instrumentation_stub(model, heavy=True, token_ids=None):
    """Exercise Trainer._log_layer_instrumentation without building a Trainer.

    Same idiom as tests/test_trainer_save_types.py: a Trainer needs dataloaders,
    an optimizer and a live metric logger, while this method reads only
    `self.model` and `self.metric_logger`.
    """
    from src.core.trainer import Trainer

    class Logger:
        _should_log_heavy_metrics = heavy

        def __init__(self):
            self.logged = {}

        def log(self, name, value):
            self.logged[name] = value

    class Stub:
        _log_layer_instrumentation = Trainer._log_layer_instrumentation

    stub = Stub()
    stub.model = model
    stub.metric_logger = Logger()
    stub._last_input_ids = token_ids
    return stub


def test_the_trainer_logs_the_readout_for_every_ame_layer_and_no_other():
    stub = _instrumentation_stub(
        _h4_ame(["+common.attention.mix={type: gqa, layers: every_2nd}"])
    )
    stub._log_layer_instrumentation()

    logged = stub.metric_logger.logged
    assert logged, "nothing was logged at all"
    assert {name.rsplit("/", 2)[0] for name in logged} == {
        "encoder.blocks.0.attention_layer.layer",
        "encoder.blocks.2.attention_layer.layer",
    }
    assert len(logged) == 2 * len(_READOUT_KEYS)
    assert all(isinstance(value, float) for value in logged.values())


def test_a_stack_with_no_ame_layers_logs_no_instrumentation():
    stub = _instrumentation_stub(_h4_ame(["common.attention.type=gqa"]))
    stub._log_layer_instrumentation()
    assert stub.metric_logger.logged == {}


def test_the_readout_obeys_the_heavy_metric_interval():
    """Every scalar costs an `.item()` sync, and under FSDP2 reading a sharded
    parameter costs a collective. It is heavy by the definition already in the
    metric logger, so it has to answer to the same gate rather than run every
    step at h20's forty layers."""
    stub = _instrumentation_stub(_h4_ame(), heavy=False)
    stub._log_layer_instrumentation()
    assert stub.metric_logger.logged == {}


# ---------------------------------------------------------------------------
# The shift helper, off its one live call site.


@pytest.mark.parametrize("shape", [(2, 5), (2, 5, 3), (2, 5, 4, 3), (2, 5, 4, 3, 2)])
def test_the_shift_preserves_shape_at_every_rank(shape):
    """Regression: the pad tuple was hard-coded for 4-D, so a 3-D input padded
    the *batch* dimension instead of the sequence -- (2, 4, 3) came back as
    (3, 3, 3), wrong, with no exception. Only 4-D is passed today; ame.md 4.1's
    raw-v register is the next caller."""
    z = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    shifted = _shift1(z)

    assert shifted.shape == z.shape
    assert (shifted[:, 0] == 0).all(), "position 0 is not zeroed"
    torch.testing.assert_close(shifted[:, 1:], z[:, :-1])


def test_a_missing_token_id_tensor_is_named_as_the_error():
    """token_ids defaults to None all the way down so that layers which do not
    want it need no signature change; the cost is that forgetting to thread it
    used to surface as AttributeError on None.shape, four frames in."""
    layer = _layer()
    x, _ = _batch(layer)
    with pytest.raises(ValueError, match="indexed by token id"):
        layer(x, None)


# ---------------------------------------------------------------------------
# GPU. FlexAttention's compiled kernel requires head_dim >= 16 ("NYI: embedding
# dimension ... must be at least 16"), and DIMS above gives head_dim 8 -- fine
# for the eager reference path, impossible to compile. Every GPU test therefore
# builds at the H family's real head_dim of 64. A GPU test that silently ran an
# unrepresentative shape would be worse than no GPU test at all.

GPU_DIMS = dict(dmodel=256, q_heads=4, kv_heads=2, seq_len=64)


def _gpu_layer(**options):
    torch.manual_seed(0)
    options.setdefault("backend", "auto")
    layer = AMEAttention(**GPU_DIMS, vocab_size=VOCAB, **options).cuda()
    assert layer.dhead == 64, "GPU tests must run at the production head width"
    generator = torch.Generator().manual_seed(1)
    token_ids = torch.randint(
        0, VOCAB, (2, GPU_DIMS["seq_len"]), generator=generator
    ).cuda()
    x = torch.randn(
        2, GPU_DIMS["seq_len"], GPU_DIMS["dmodel"], generator=generator
    ).cuda()
    return layer, x, token_ids


@pytest.mark.gpu
def test_the_default_path_compiles_and_trains():
    """CUDA is where this question matters: it is where the production path
    (flex) compiles, and compiled-flex-with-backward cannot even be lowered on
    CPU, so there is no way to anticipate the answer without a GPU."""
    layer, x, token_ids = _gpu_layer()
    eager = layer(x, token_ids)
    compiled = torch.compile(layer, fullgraph=True)(x, token_ids)
    torch.testing.assert_close(compiled, eager, rtol=1e-4, atol=1e-5)
    compiled.sum().backward()
    assert all(p.grad is not None for p in layer.parameters())


@pytest.mark.gpu
def test_on_cuda_auto_selects_flex_and_matches_the_reference_path():
    layer, x, token_ids = _gpu_layer()
    assert x.is_cuda
    auto = layer(x, token_ids)
    layer.backend = "reference"
    reference = layer(x, token_ids)
    torch.testing.assert_close(auto, reference, rtol=1e-4, atol=1e-5)


@pytest.mark.gpu
def test_on_cuda_the_flex_path_compiles_and_trains():
    """flex's compiled Triton kernel is the production path and is a genuinely
    different code path from the eager fallback CPU exercises."""
    layer, x, token_ids = _gpu_layer(backend="flex")
    eager = layer(x, token_ids)
    compiled = torch.compile(layer)(x, token_ids)
    torch.testing.assert_close(compiled, eager, rtol=1e-4, atol=1e-5)
    compiled.sum().backward()
    assert all(p.grad is not None for p in layer.parameters())


@pytest.mark.gpu
def test_on_cuda_bf16_autocast_matches_fp32_within_bf16_resolution():
    layer, x, token_ids = _gpu_layer(backend="flex")
    fp32 = layer(x, token_ids)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        bf16 = layer(x, token_ids).float()
    relative = ((fp32 - bf16).abs().max() / fp32.abs().max()).item()
    assert relative < 0.05, f"bf16 diverges from fp32 by {relative:.2e}"


@pytest.mark.gpu
def test_on_cuda_the_shift_and_sink_survive_the_compiled_kernel():
    """The pieces the score_mod and the extra kv column touch, at production
    width, compiled -- the combination the eager CPU tests cannot reach."""
    for options in ({"shift": False}, {"sink": False}, {"pin_alpha": True}):
        layer, x, token_ids = _gpu_layer(**options)
        eager = layer(x, token_ids)
        compiled = torch.compile(layer)(x, token_ids)
        torch.testing.assert_close(compiled, eager, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# The shipping path. Every GPU test above wraps the layer in torch.compile
# *inside the test*, which is exactly how the unfused kernel shipped: they prove
# the compiled kernel works, and none of them asserts that a plain forward --
# what every config actually runs -- reaches it.


def _peak_mib_for_attention(seq_len):
    """Peak CUDA memory of one forward+backward of the default path, in MiB.

    Warmed up first, and that is not a detail: the very first call compiles the
    kernel, and the compiler's own workspace swamps the measurement. Measured
    cold, seq 512 came out *larger* than seq 1024 -- which is how the first
    version of this test passed against the unfused path it was written to
    catch.
    """
    torch.manual_seed(0)
    layer = AMEAttention(
        dmodel=256, q_heads=4, kv_heads=2, seq_len=seq_len, vocab_size=VOCAB
    ).cuda()
    assert layer.dhead == 64
    generator = torch.Generator().manual_seed(1)
    token_ids = torch.randint(0, VOCAB, (2, seq_len), generator=generator).cuda()
    x = torch.randn(2, seq_len, 256, generator=generator).cuda()

    layer(x, token_ids).sum().backward()
    layer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    layer(x, token_ids).sum().backward()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - before) / (1024**2)


@pytest.mark.gpu
def test_the_default_path_does_not_materialize_the_score_matrix():
    """The regression for the whole of ame_code_review.md F1.

    An uncompiled flex_attention is the `sdpa_dense` fallback: it materializes
    `[B, H, Q, KV]`, so doubling the sequence costs ~4x the memory. The fused
    Triton kernel never forms that tensor, so the cost is dominated by the
    linear-sized tensors and roughly doubles instead.

    Measured on an H100: 1.99x with the compiled kernel, 3.75x without it, so
    the gate sits between them. The sizes are 1024 and 2048 rather than 512 and
    1024 because below 1024 the quadratic term does not yet dominate -- the
    unfused path also measures ~2x there, which is the second reason the first
    version of this test could not tell the two apart.

    Asserted on memory rather than on which callable ran: a wrapper that is
    compiled but still falls back would satisfy an identity check, and that is
    the precise failure this exists to catch.
    """
    small = _peak_mib_for_attention(1024)
    large = _peak_mib_for_attention(2048)
    ratio = large / small
    assert ratio < 3.0, (
        f"peak memory grew {ratio:.2f}x when the sequence doubled "
        f"({small:.1f} -> {large:.1f} MiB); quadratic growth means the default "
        "path is running the unfused flex_attention fallback"
    )


@pytest.mark.gpu
def test_the_default_path_is_right_without_the_test_compiling_it():
    """The plain forward -- no torch.compile in sight -- end to end.

    `auto` selects flex on CUDA, so this is the path a config actually runs, and
    before Phase 1 nothing checked it: every other GPU test wraps the layer in
    `torch.compile` itself, which is exactly how the unfused kernel shipped.

    "End to end" is both halves, so both are asserted here: the output matches
    the reference, **and** the call went through the compiled wrapper rather
    than the materialising fallback. Either half alone leaves a hole -- F1 was
    a path that computed the right answer the wrong way.
    """
    import src.core.ame as ame_module

    ame_module._COMPILED_FLEX = None
    layer, x, token_ids = _gpu_layer()
    shipped = layer(x, token_ids)

    assert ame_module._COMPILED_FLEX is not None, (
        "the shipping path never reached the compiled flex wrapper: it is "
        "running the unfused fallback that materializes the score matrix"
    )

    layer.backend = "reference"
    reference = layer(x, token_ids)
    torch.testing.assert_close(shipped, reference, rtol=1e-4, atol=1e-5)


@pytest.mark.gpu
def test_the_default_path_reaches_the_compiled_kernel():
    """Complements the memory test from the other side: cheap, and it localises
    a failure to the wiring rather than to the kernel."""
    import src.core.ame as ame_module

    ame_module._COMPILED_FLEX = None
    layer, x, token_ids = _gpu_layer()
    layer(x, token_ids)
    assert (
        ame_module._COMPILED_FLEX is not None
    ), "a plain CUDA forward did not go through the compiled flex wrapper"


@pytest.mark.gpu
def test_a_whole_compiled_block_still_receives_the_token_ids():
    """`compile_blocks_full` (configs/mfu_audit/h20_mgpu.yaml) compiles each
    whole TransformerBlock, and AME made `TransformerBlock.forward` take a
    second argument. Nothing had exercised that pairing, and it is the
    configuration the AME h20 run uses -- a failure here is a wasted job.
    """
    model = _h4_ame(vocab=96, device="cuda")
    ids = torch.randint(0, 96, (2, 16)).cuda()
    eager = model(ids)

    for block in model.encoder.blocks:
        block.compile()
    compiled = model(ids)

    torch.testing.assert_close(compiled, eager, rtol=1e-3, atol=1e-4)
    compiled.sum().backward()
    banks = model.encoder.blocks[-1].attention_layer.layer.q_bank.weight
    assert banks.grad is not None and banks.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Attention entropy and top-k mass -- the other half of ame_experiments.md 7.
# Functions of the attention weights, which the fused kernel never forms, so
# they are recomputed for a sample of query rows.

_DIAGNOSTIC_KEYS = {
    "attention_entropy/nats",
    "attention_entropy/normalized",
    "top8_mass/mean",
}


def test_the_diagnostics_report_entropy_and_top_k_mass():
    layer = _layer()
    _, token_ids = _batch(layer)
    readout = layer.attention_diagnostics(token_ids)

    assert set(readout) == _DIAGNOSTIC_KEYS
    assert all(isinstance(value, float) for value in readout.values())
    assert 0.0 <= readout["attention_entropy/normalized"] <= 1.0
    assert 0.0 < readout["top8_mass/mean"] <= 1.0
    # At init the banks are random and tau is modest, so attention should be
    # closer to uniform than to collapsed. This is the init sanity ame.md 7
    # warns about from the other direction.
    assert readout["attention_entropy/normalized"] > 0.5


def test_a_collapsed_head_reads_as_collapsed():
    """The readout has to move the right way, or it cannot detect anything.
    Driving tau up sharpens every softmax onto its argmax."""
    layer = _layer()
    _, token_ids = _batch(layer)
    before = layer.attention_diagnostics(token_ids)
    with torch.no_grad():
        layer.tau.mul_(500.0)
    after = layer.attention_diagnostics(token_ids)

    assert after["attention_entropy/normalized"] < 0.1, after
    assert after["top8_mass/mean"] > before["top8_mass/mean"]


def test_the_diagnostics_need_no_activations():
    """AME's scores are a function of ids and distance only, which is why this
    is affordable: no forward pass, no batch of hidden states, and the answer
    must not depend on the residual stream."""
    layer = _layer()
    _, token_ids = _batch(layer)
    first = layer.attention_diagnostics(token_ids)
    with torch.no_grad():
        layer.v_proj.weight.mul_(7.0)  # changes every activation, no score
    assert layer.attention_diagnostics(token_ids) == first


@pytest.mark.parametrize("seq_len,max_rows", [(2048, 128), (2048, 1), (64, 128)])
def test_the_diagnostics_never_look_at_more_rows_than_asked(seq_len, max_rows):
    """The cap is what keeps a quadratic diagnostic off the logging path. At h20
    the full matrix is [B, H, 2048, 2049]; without the cap this would materialize
    it every heavy-metric step, which is the cost `score_matrix()` was never
    wired up to pay."""
    rows = _diagnostic_rows(seq_len, max_rows)

    assert len(rows) <= max_rows, f"{len(rows)} rows for a cap of {max_rows}"
    assert len(rows) == min(max_rows, seq_len)
    assert rows.min() >= 0 and rows.max() < seq_len
    assert torch.equal(rows, torch.unique(rows)), "rows must be sorted and distinct"
    # Spread across the whole sequence, not clustered at the start. Query row i
    # can only see i + 1 keys, so a sample bunched at the beginning would
    # measure the position regime rather than the heads -- the exact artifact
    # the normalized entropy exists to remove.
    if len(rows) > 1:
        assert (
            rows[0] == 0 and rows[-1] == seq_len - 1
        ), f"sample spans only {int(rows[0])}..{int(rows[-1])} of {seq_len} rows"
    if max_rows >= seq_len:
        assert torch.equal(rows, torch.arange(seq_len))


def test_the_diagnostics_subsample_the_query_rows():
    """The cap reaches `attention_diagnostics`, not just the helper."""
    layer = _layer()
    _, token_ids = _batch(layer)
    assert layer.score_matrix(token_ids).shape[2] == DIMS["seq_len"]

    assert set(layer.attention_diagnostics(token_ids, max_rows=3)) == _DIAGNOSTIC_KEYS
    # a cap above the sequence length is simply every row, and must still work
    assert layer.attention_diagnostics(token_ids, max_rows=10_000)
    # and the cap changes the answer, so it is genuinely being applied
    one_row = layer.attention_diagnostics(token_ids, max_rows=1)
    every_row = layer.attention_diagnostics(token_ids, max_rows=10_000)
    assert one_row != every_row


def test_the_trainer_includes_the_diagnostics_when_it_has_a_batch():
    model = _h4_ame()
    ids = torch.randint(0, 96, (2, 16))
    stub = _instrumentation_stub(model, token_ids=ids)
    stub._log_layer_instrumentation()

    logged = stub.metric_logger.logged
    suffixes = {name.split("/", 1)[1] for name in logged}
    assert _DIAGNOSTIC_KEYS <= suffixes, sorted(suffixes)
    assert len(logged) == 4 * (len(_READOUT_KEYS) + len(_DIAGNOSTIC_KEYS))


def test_the_trainer_logs_the_parameter_readout_even_with_no_batch():
    """Before the first micro-batch there are no ids; the parameter half must
    still report rather than crash."""
    stub = _instrumentation_stub(_h4_ame(), token_ids=None)
    stub._log_layer_instrumentation()
    suffixes = {name.split("/", 1)[1] for name in stub.metric_logger.logged}
    assert suffixes == _READOUT_KEYS
