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
    """Decided 2026-11-03 (archive/ame_review_and_fixes.md 2.6): the banks are exempt.

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


def test_the_scheduler_floors_each_group_at_its_own_learning_rate():
    """Groups at different lrs used to be rejected, because torch's eta_min is
    one value for all groups and a second lr would anneal to the wrong floor.
    The banks' bank_lr_ratio needs exactly that case, so the cosine now floors
    every group at final_lr_fraction of its OWN lr (CosineAnnealingToFraction):
    the ratio holds at every step and each group ends on its own floor."""
    from src.core.schedulers import get_cosine_scheduler_with_warmup

    layer = _layer()
    optimizer = torch.optim.AdamW(
        [
            {"params": [layer.tau], "lr": 1e-3},
            {"params": [layer.alpha], "lr": 1e-4},
        ]
    )
    scheduler = get_cosine_scheduler_with_warmup(
        optimizer, warmup_steps=2, n_steps=10, final_lr_fraction=0.1
    )
    for _ in range(10):
        high, low = (group["lr"] for group in optimizer.param_groups)
        assert high == pytest.approx(10 * low, rel=1e-12)
        optimizer.step()
        scheduler.step()
    # Step 10 is the cosine's T_max: every group sits exactly on its floor.
    high, low = (group["lr"] for group in optimizer.param_groups)
    assert high == pytest.approx(1e-4, rel=1e-12)
    assert low == pytest.approx(1e-5, rel=1e-12)


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
    # What __attrs_post_init__ sets and _log_layer_instrumentation reads: the
    # set of modules whose readout raised, warned about once and skipped after.
    stub._instrumentation_failures = set()
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
    """The regression for the whole of archive/ame_review_and_fixes.md F1.

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


# ---------------------------------------------------------------------------
# Bank gates (documents/notes/ame_bank_gates_plan.md): four ways to trust a bank
# row. `none` is the model every test above checks; each gate must change
# exactly what the plan says and nothing else.

GATES = ("confidence", "context", "backoff", "stem")
# Gates whose scores still depend only on token ids and distance.
ID_ONLY_GATES = ("none", "confidence", "backoff")
# Gates that read the query's hidden state, and so break that on purpose.
X_READING_GATES = ("context", "stem")


def _perturb_gate(layer, seed=7):
    """Move a gate's parameters off their init, so a test cannot pass because
    every trust is 0.05, every B2 gate sigmoid(3) or every base zero."""
    generator = torch.Generator().manual_seed(seed)
    datt = layer.q_heads * layer.dhead
    with torch.no_grad():
        if layer.bank_gate == "confidence":
            for bank in (layer.q_bank, layer.k_bank):
                noise = torch.randn(bank.weight[:, datt:].shape, generator=generator)
                bank.weight[:, datt:] = 2.0 * noise
        elif layer.bank_gate == "context":
            layer.gate.weight.copy_(
                0.3 * torch.randn(layer.gate.weight.shape, generator=generator)
            )
            layer.gate.bias.copy_(
                torch.randn(layer.gate.bias.shape, generator=generator)
            )
        elif layer.bank_gate == "backoff":
            for base in (layer.q_base, layer.k_base):
                base.copy_(torch.randn(base.shape, generator=generator))
        elif layer.bank_gate == "stem":
            layer.gate.weight.mul_(3.0)
    return layer


def _scores_with_x(layer, x, token_ids):
    """`[B, H, n, n (+1)]` scores from real activations. score_matrix feeds
    x = 0 and refuses the gates that read x; this is the general path."""
    q, k, v = layer._mixed_qkv(x, token_ids)
    _, scores = layer._attend_reference(q, k, v, return_scores=True)
    return scores


def _spec_gated_forward(layer, x, token_ids):
    """The forward of every gate, transcribed from the plan's mechanism table.
    Naive on purpose, like `_spec_forward`, and sink-free like it."""
    batch, seq_len = token_ids.shape
    heads, dhead = layer.q_heads, layer.dhead
    datt = heads * dhead
    gate = layer.bank_gate

    q_rows = layer.q_bank.weight[token_ids]
    k_rows = layer.k_bank.weight[token_ids]
    q = q_rows[..., :datt].reshape(batch, seq_len, heads, dhead)
    k = k_rows[..., :datt].reshape(batch, seq_len, heads, dhead)
    if gate == "backoff":  # normalize((base + E[t]) / sqrt 2)
        q = (layer.q_base + q) / math.sqrt(2)
        k = (layer.k_base + k) / math.sqrt(2)
    q = q / q.norm(dim=-1, keepdim=True)
    k = k / k.norm(dim=-1, keepdim=True)
    if gate == "confidence":  # sigmoid(c[t]) * normalize(E[t]), both sides
        q = torch.sigmoid(q_rows[..., datt:]).unsqueeze(-1) * q
        k = torch.sigmoid(k_rows[..., datt:]).unsqueeze(-1) * k
    elif gate == "context":  # sigmoid(W_g x + b) * normalize(E_Q[t])
        g = torch.sigmoid(x @ layer.gate.weight.T + layer.gate.bias)
        q = g.unsqueeze(-1) * q
    elif gate == "stem":  # silu(W_g x) (.) normalize(E_Q[t])
        g = torch.nn.functional.silu(x @ layer.gate.weight.T)
        q = g.reshape(batch, seq_len, heads, dhead) * q

    def shift1(z):
        shifted = torch.zeros_like(z)
        shifted[:, 1:] = z[:, :-1]
        return shifted

    v = (x @ layer.v_proj.weight.T).view(batch, seq_len, heads, dhead)
    alpha, beta = layer.alpha, layer.beta
    k = alpha[:, 0].view(1, 1, -1, 1) * k + alpha[:, 1].view(1, 1, -1, 1) * shift1(k)
    v = beta[:, 0].view(1, 1, -1, 1) * v + beta[:, 1].view(1, 1, -1, 1) * shift1(v)
    scores = torch.einsum("bihd,bjhd->bhij", q, k) * layer.tau.view(1, -1, 1, 1)
    i = torch.arange(seq_len).view(-1, 1)
    j = torch.arange(seq_len).view(1, -1)
    scores = scores - layer.slopes.view(1, -1, 1, 1) * (i - j).float()
    scores = scores.masked_fill((j > i).view(1, 1, seq_len, seq_len), float("-inf"))
    out = torch.einsum("bhij,bjhd->bihd", scores.softmax(dim=-1), v)
    return out.reshape(batch, seq_len, datt) @ layer.o_proj.weight.T


def test_bank_gate_none_is_the_default_and_adds_nothing():
    """`none` is today's layer: no extra parameter, the same state dict, the
    same forward."""
    default, explicit = _layer(seed=3), _layer(seed=3, bank_gate="none")
    assert list(default.state_dict()) == list(explicit.state_dict())
    for name, tensor in default.state_dict().items():
        assert torch.equal(tensor, explicit.state_dict()[name]), name
    assert default.q_bank.weight.shape == (VOCAB, DIMS["dmodel"])
    x, token_ids = _batch(default)
    assert torch.equal(default(x, token_ids), explicit(x, token_ids))


def test_an_unknown_bank_gate_is_rejected_with_the_valid_set():
    with pytest.raises(ValueError, match="confidence, context, backoff, stem"):
        _layer(bank_gate="confidense")


def test_the_config_path_reaches_the_gate_and_rejects_a_typo():
    model = _h4_ame(["+common.attention.ame={bank_gate: backoff, bank_init_std: 0.01}"])
    for block in model.encoder.blocks:
        assert block.attention_layer.layer.bank_gate == "backoff"
    with pytest.raises(Exception, match="bank_gate"):
        _h4_ame(["+common.attention.ame={bank_gate: bogus}"])


@pytest.mark.parametrize("bank_gate", GATES)
def test_every_gate_starts_every_shared_weight_where_none_does(bank_gate):
    """The arms must differ in their mechanism only. A gate that drew its init
    from the global RNG would shift every later parameter -- this layer's
    values, the next blocks -- and the arms would differ by a seed as well."""
    torch.manual_seed(11)
    ungated = AMEAttention(**DIMS, vocab_size=VOCAB, bank_init_std=0.01)
    after_ungated = torch.random.get_rng_state()
    torch.manual_seed(11)
    gated = AMEAttention(
        **DIMS, vocab_size=VOCAB, bank_init_std=0.01, bank_gate=bank_gate
    )
    assert torch.equal(torch.random.get_rng_state(), after_ungated)
    datt = DIMS["dmodel"]
    for name, tensor in ungated.state_dict().items():
        mine = gated.state_dict()[name]
        if name.endswith("_bank.weight"):
            mine = mine[:, :datt]
        assert torch.equal(mine, tensor), name


@pytest.mark.parametrize("bank_gate", GATES)
def test_every_gate_matches_a_literal_transcription_of_its_formula(bank_gate):
    layer = _perturb_gate(_layer(sink=False, bank_gate=bank_gate))
    x, token_ids = _batch(layer)
    torch.testing.assert_close(
        layer(x, token_ids),
        _spec_gated_forward(layer, x, token_ids),
        rtol=1e-5,
        atol=1e-6,
    )


# B1, per-row confidence.


@pytest.mark.parametrize(
    "option, trust_init",
    [("", 0.05), (", confidence_init: 0.95", 0.95)],
    ids=["default_005", "stage2_open_095"],
)
def test_b1_every_trust_starts_at_its_init_after_mains_build_path(option, trust_init):
    """Not only after the constructor: main.py hands the banks to the
    memory-parallel controller, which copies them into fresh parameters and an
    fp32 master. The trust columns must survive that copy, and `confidence_init`
    must reach the layer from the config's `attention.ame` block."""
    import main
    from src.core.memory_parallel import ShardedBank, get_controller

    with initialize_config_dir(
        version_base=None, config_dir=os.path.join(REPO_ROOT, "configs")
    ):
        cfg = compose(
            config_name="tiny_local",
            overrides=[
                "common.vocab_size=96",
                "common.attention.type=ame",
                "+common.attention.ame="
                f"{{bank_gate: confidence, bank_init_std: 0.01{option}}}",
                "trainer.mem_parallel_size=1",
            ],
        )
    torch.manual_seed(0)
    model = instantiate(cfg.model, _convert_="all")
    model, _, _ = main.get_model_optimizer_scheduler(cfg, model, 1e-3)
    assert get_controller(model) is not None
    layers = [m for m in model.modules() if isinstance(m, AMEAttention)]
    assert len(layers) == 4
    for layer in layers:
        for which in ("q", "k"):
            bank = getattr(layer, f"{which}_bank")
            assert isinstance(bank, ShardedBank)
            assert bank.width == 4 * 64 + 4
            trust = torch.sigmoid(layer._trust_logits(which))
            assert trust.shape == (96, 4)
            torch.testing.assert_close(
                trust, torch.full_like(trust, trust_init), rtol=0, atol=1e-6
            )
            # and the forward reads the same columns from the compute copy
            torch.testing.assert_close(
                torch.sigmoid(bank.weight[:, 256:].float()), trust, rtol=0, atol=1e-6
            )


def test_b1_scores_are_the_ungated_scores_times_both_trusts():
    options = dict(shift=False, sink=False, slopes=[0.0] * DIMS["q_heads"])
    ungated = _layer(seed=5, **options)
    gated = _perturb_gate(_layer(seed=5, bank_gate="confidence", **options))
    _, token_ids = _batch(gated)
    datt = DIMS["dmodel"]
    s_q = torch.sigmoid(gated.q_bank.weight[:, datt:])[token_ids]  # [B, n, H]
    s_k = torch.sigmoid(gated.k_bank.weight[:, datt:])[token_ids]
    factor = s_q.permute(0, 2, 1).unsqueeze(-1) * s_k.permute(0, 2, 1).unsqueeze(-2)
    expected = ungated.score_matrix(token_ids)
    finite = torch.isfinite(expected)
    actual = gated.score_matrix(token_ids)
    torch.testing.assert_close(
        actual[finite], (factor * expected)[finite], rtol=1e-5, atol=1e-6
    )
    assert torch.equal(torch.isfinite(actual), finite)


@pytest.mark.parametrize(
    "init, factor",
    [({}, 0.0025), ({"confidence_init": 0.95}, 0.9025)],
    ids=["default_005", "stage2_open_095"],
)
def test_b1_at_init_every_score_is_trust_squared_of_the_ungated_one(init, factor):
    """0.05 * 0.05 = 0.0025 by default (stage 1's B1); 0.95 * 0.95 = 0.9025 for
    stage 2's b1open_std001."""
    options = dict(shift=False, sink=False, slopes=[0.0] * DIMS["q_heads"])
    ungated = _layer(seed=5, bank_init_std=0.01, **options)
    gated = _layer(
        seed=5, bank_init_std=0.01, bank_gate="confidence", **init, **options
    )
    _, token_ids = _batch(gated)
    expected = ungated.score_matrix(token_ids)
    finite = torch.isfinite(expected)
    torch.testing.assert_close(
        gated.score_matrix(token_ids)[finite],
        factor * expected[finite],
        rtol=1e-4,
        atol=1e-7,
    )


def test_b1_the_default_confidence_init_is_stage_1s_b1_bit_for_bit():
    """Making the init an option must not move stage 1's B1: the default and an
    explicit 0.05 build the same state dict, and the trust columns hold the
    fp32 value the old module constant, logit(0.05), wrote."""
    default = _layer(seed=5, bank_init_std=0.01, bank_gate="confidence")
    explicit = _layer(
        seed=5, bank_init_std=0.01, bank_gate="confidence", confidence_init=0.05
    )
    assert list(default.state_dict()) == list(explicit.state_dict())
    for name, tensor in default.state_dict().items():
        assert torch.equal(explicit.state_dict()[name], tensor), name
    datt = DIMS["dmodel"]
    for bank in (default.q_bank, default.k_bank):
        trust_logits = bank.weight[:, datt:]
        assert torch.equal(
            trust_logits, torch.full_like(trust_logits, -2.9444389791664403)
        )


def test_b1_open_trust_reads_out_at_095():
    layer = _layer(bank_gate="confidence", confidence_init=0.95)
    readout = layer.instrumentation()
    assert readout["trust_q/mean"] == pytest.approx(0.95, abs=1e-6)
    assert readout["trust_k/p10"] == pytest.approx(0.95, abs=1e-6)
    assert readout["trust_q/above_half"] == 1.0
    assert "confidence_init=0.95" in repr(layer)


@pytest.mark.parametrize("value", (0.0, 1.0, -0.1, 1.5))
def test_a_confidence_init_outside_0_1_is_rejected(value):
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        _layer(bank_gate="confidence", confidence_init=value)


@pytest.mark.parametrize("bank_gate", ("none", "context", "backoff", "stem"))
def test_confidence_init_on_another_gate_is_rejected_not_ignored(bank_gate):
    with pytest.raises(ValueError, match="applies only to bank_gate=confidence"):
        _layer(bank_gate=bank_gate, confidence_init=0.95)
    # Stated at its default value it changes nothing, so it is accepted.
    _layer(bank_gate=bank_gate, confidence_init=0.05)
    with pytest.raises(Exception, match="confidence_init"):
        _h4_ame(["+common.attention.ame={bank_gate: backoff, confidence_init: 0.95}"])


def test_b1_trust_gets_gradient_only_for_ids_in_the_batch():
    layer = _layer(bank_gate="confidence")
    x, token_ids = _batch(layer)
    layer(x, token_ids).sum().backward()
    datt = DIMS["dmodel"]
    for bank in (layer.q_bank, layer.k_bank):
        rows = bank.weight.grad[:, datt:].abs().sum(dim=-1)
        touched = token_ids.unique()
        assert (rows[touched] > 0).all(), "every id in the batch moves its trust"
        untouched = torch.ones(VOCAB, dtype=torch.bool)
        untouched[touched] = False
        assert rows[untouched].abs().max() == 0, "and no id outside it does"


def test_b1_the_fused_memory_parallel_path_equals_the_plain_path():
    """At width H*d + H the controller's fused gather and its scatter into
    main_grad must give the plain nn.Embedding path's forward and gradient."""
    import copy

    from src.core.memory_parallel import MemoryParallelController

    plain = _perturb_gate(_layer(bank_gate="confidence"))
    fused = copy.deepcopy(plain)
    controller = MemoryParallelController(
        fused, mem_parallel_size=1, compute_dtype=torch.float32
    )
    assert fused.qk_bank is not None, "the pair is fused"
    assert all(
        bank.width == DIMS["dmodel"] + DIMS["q_heads"] for bank in controller.banks
    )
    controller.materialize()
    x, token_ids = _batch(plain)
    out_plain, out_fused = plain(x, token_ids), fused(x, token_ids)
    torch.testing.assert_close(out_fused, out_plain, rtol=0, atol=0)
    out_plain.square().sum().backward()
    out_fused.square().sum().backward()
    for bank, name in zip(controller.banks, ("q_bank", "k_bank")):
        torch.testing.assert_close(
            bank.main_grad, getattr(plain, name).weight.grad, rtol=1e-6, atol=1e-7
        )


# B2, the context gate.


def test_b2_every_gate_starts_at_sigmoid_3_whatever_the_hidden_state():
    layer = _layer(bank_gate="context")
    x, _ = _batch(layer)
    gate = layer._query_gate(x * 50.0, x)
    torch.testing.assert_close(
        gate, torch.full_like(gate, 1 / (1 + math.exp(-3.0))), rtol=0, atol=1e-7
    )


def test_b2_scores_are_the_ungated_scores_times_the_query_gate():
    options = dict(shift=False, sink=False, slopes=[0.0] * DIMS["q_heads"])
    ungated = _layer(seed=5, **options)
    gated = _perturb_gate(_layer(seed=5, bank_gate="context", **options))
    x, token_ids = _batch(gated)
    g = torch.sigmoid(x @ gated.gate.weight.T + gated.gate.bias)  # [B, n, H]
    expected = _scores_with_x(ungated, x, token_ids)
    finite = torch.isfinite(expected)
    actual = _scores_with_x(gated, x, token_ids)
    factor = g.permute(0, 2, 1).unsqueeze(-1)
    torch.testing.assert_close(
        actual[finite], (factor * expected)[finite], rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("bank_gate", X_READING_GATES)
def test_b2_and_st_leave_the_keys_unchanged(bank_gate):
    ungated = _layer(seed=5)
    gated = _perturb_gate(_layer(seed=5, bank_gate=bank_gate))
    x, token_ids = _batch(gated)
    _, k_ungated, v_ungated = ungated._mixed_qkv(x, token_ids)
    q_gated, k_gated, v_gated = gated._mixed_qkv(x, token_ids)
    assert torch.equal(k_gated, k_ungated)
    assert torch.equal(v_gated, v_ungated)
    q_ungated, _, _ = ungated._mixed_qkv(x, token_ids)
    assert not torch.allclose(q_gated, q_ungated), "the query is what is gated"


def test_b2_the_gate_bias_is_not_decayed_and_the_weight_is():
    layer = _layer(bank_gate="context")
    decayed, undecayed = parameter_groups(layer, weight_decay=0.1)
    assert decayed["weight_decay"] == 0.1 and undecayed["weight_decay"] == 0.0
    ids = lambda group: {id(p) for p in group["params"]}  # noqa: E731
    assert id(layer.gate.bias) in ids(undecayed)
    assert id(layer.gate.weight) in ids(decayed)


def test_b2_at_init_every_score_is_sigmoid_3_of_the_ungated_one():
    options = dict(shift=False, sink=False, slopes=[0.0] * DIMS["q_heads"])
    ungated = _layer(seed=5, **options)
    gated = _layer(seed=5, bank_gate="context", **options)
    x, token_ids = _batch(gated)
    expected = _scores_with_x(ungated, x, token_ids)
    finite = torch.isfinite(expected)
    torch.testing.assert_close(
        _scores_with_x(gated, x, token_ids)[finite],
        (1 / (1 + math.exp(-3.0))) * expected[finite],
        rtol=1e-5,
        atol=1e-6,
    )


# B3, the shared backoff vector.


def test_b3_at_init_the_forward_is_the_ungated_forward():
    """base = 0 at init, and the hard norm hides the 1/sqrt(2): an ungated run."""
    ungated = _layer(seed=5, bank_init_std=0.01)
    gated = _layer(seed=5, bank_init_std=0.01, bank_gate="backoff")
    x, token_ids = _batch(gated)
    torch.testing.assert_close(
        gated(x, token_ids), ungated(x, token_ids), rtol=1e-5, atol=1e-6
    )


def test_b3_the_base_vectors_receive_gradient():
    layer = _layer(bank_gate="backoff")
    x, token_ids = _batch(layer)
    layer(x, token_ids).sum().backward()
    for base in (layer.q_base, layer.k_base):
        assert base.grad is not None and base.grad.abs().sum() > 0


def test_b3_the_base_is_neither_decayed_nor_counted_as_matmul_flops():
    from src.core.mfu import model_flops_per_token

    layer = _layer(bank_gate="backoff")
    _, undecayed = parameter_groups(layer, weight_decay=0.1)
    undecayed_ids = {id(p) for p in undecayed["params"]}
    assert id(layer.q_base) in undecayed_ids and id(layer.k_base) in undecayed_ids
    seq = DIMS["seq_len"]
    assert model_flops_per_token(layer, seq) == model_flops_per_token(_layer(), seq)


# ST, the STEM gate.


def test_st_starts_at_about_half_the_ungated_logit_spread():
    """silu(W_g x) at unit-RMS x has RMS ~0.5, so the identity logits start at
    about half today's spread (the plan's estimate), not at zero and not at 1."""
    dims = dict(dmodel=256, q_heads=4, kv_heads=4, seq_len=64, vocab_size=512)
    options = dict(sink=False, shift=False, slopes=[0.0] * 4, backend="reference")
    torch.manual_seed(0)
    ungated = AMEAttention(**dims, **options)
    torch.manual_seed(0)
    gated = AMEAttention(**dims, bank_gate="stem", **options)
    generator = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 512, (4, 64), generator=generator)
    x = torch.randn(4, 64, 256, generator=generator)
    spread = {}
    for name, layer in (("ungated", ungated), ("stem", gated)):
        scores = _scores_with_x(layer, x, ids)
        spread[name] = scores[torch.isfinite(scores)].std().item()
    assert 0.3 < spread["stem"] / spread["ungated"] < 0.8, spread


# Invariants two arms break on purpose.


@pytest.mark.parametrize("bank_gate", ID_ONLY_GATES)
def test_with_an_id_only_gate_a_score_depends_only_on_the_pair_and_the_distance(
    bank_gate,
):
    layer = _perturb_gate(
        _layer(
            shift=False,
            sink=False,
            slopes=[0.0] * DIMS["q_heads"],
            bank_gate=bank_gate,
        )
    )
    token_ids = torch.tensor([[5, 9, 5, 9, 5, 9, 5, 9, 5, 9]])
    scores = layer.score_matrix(token_ids)[0]
    for head in range(layer.q_heads):
        pairs = [scores[head, i, j] for i in (4, 6, 8) for j in (1, 3, 5) if j <= i]
        for value in pairs[1:]:
            torch.testing.assert_close(value, pairs[0], rtol=0, atol=1e-6)


@pytest.mark.parametrize("bank_gate", ID_ONLY_GATES)
def test_with_an_id_only_gate_the_weights_ignore_the_hidden_state(bank_gate):
    layer = _perturb_gate(_layer(bank_gate=bank_gate))
    x, token_ids = _batch(layer)
    other_x, _ = _batch(layer, seed=99)
    torch.testing.assert_close(
        _scores_with_x(layer, x, token_ids),
        _scores_with_x(layer, other_x, token_ids),
        rtol=0,
        atol=0,
    )
    assert not torch.allclose(layer(x, token_ids), layer(other_x, token_ids))


@pytest.mark.parametrize("bank_gate", X_READING_GATES)
def test_a_gate_that_reads_x_makes_the_scores_depend_on_x(bank_gate):
    """The break B2 and ST make on purpose: the same ids, another hidden state,
    other scores."""
    layer = _perturb_gate(_layer(bank_gate=bank_gate))
    x, token_ids = _batch(layer)
    other_x, _ = _batch(layer, seed=99)
    first = _scores_with_x(layer, x, token_ids)
    second = _scores_with_x(layer, other_x, token_ids)
    finite = torch.isfinite(first)
    assert not torch.allclose(first[finite], second[finite])


@pytest.mark.parametrize("bank_gate", X_READING_GATES)
def test_a_gate_that_reads_x_gets_no_readout_computed_from_x_zero(bank_gate):
    """score_matrix and attention_diagnostics feed x = 0, which would read B2's
    gate as sigmoid(b) everywhere and erase ST's queries -- silently."""
    layer = _layer(bank_gate=bank_gate)
    _, token_ids = _batch(layer)
    with pytest.raises(ValueError, match="hidden state"):
        layer.score_matrix(token_ids)
    assert layer.attention_diagnostics(token_ids) == {}
    readout = layer.instrumentation(token_ids)
    assert not (_DIAGNOSTIC_KEYS & set(readout)), sorted(readout)


@pytest.mark.parametrize("bank_gate", ("confidence", "backoff"))
def test_an_id_only_gate_keeps_its_attention_diagnostics(bank_gate):
    layer = _layer(bank_gate=bank_gate)
    _, token_ids = _batch(layer)
    assert set(layer.attention_diagnostics(token_ids)) == _DIAGNOSTIC_KEYS


# Every arm.


@pytest.mark.parametrize("bank_gate", GATES)
def test_under_every_gate_the_two_backends_compute_the_same_function(bank_gate):
    layer = _perturb_gate(_layer(bank_gate=bank_gate))
    x, token_ids = _batch(layer)
    layer.backend = "reference"
    reference = layer(x, token_ids)
    layer.backend = "flex"
    torch.testing.assert_close(layer(x, token_ids), reference, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("bank_gate", GATES)
def test_under_every_gate_bf16_gradients_are_finite(bank_gate):
    layer = _perturb_gate(_layer(bank_gate=bank_gate))
    x, token_ids = _batch(layer)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = layer(x, token_ids)
    assert out.dtype is torch.bfloat16
    out.float().sum().backward()
    for name, parameter in layer.named_parameters():
        assert parameter.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} has a non-finite grad"


_GATE_READOUT_KEYS = {
    "none": set(),
    "confidence": {
        f"trust_{which}/{stat}"
        for which in ("q", "k")
        for stat in ("mean", "p10", "p50", "p90", "above_half")
    },
    "context": {"gate_weight/rms", "gate_bias/mean", "gate_bias/min", "gate_bias/max"},
    "backoff": {
        f"base_{which}_norm/{stat}"
        for which in ("q", "k")
        for stat in ("mean", "h0", "h1", "h2", "h3")
    },
    "stem": {"gate_weight/rms"},
}


@pytest.mark.parametrize("bank_gate", ("none",) + GATES)
def test_the_trainer_logs_every_gates_readout(bank_gate):
    model = _h4_ame([f"+common.attention.ame={{bank_gate: {bank_gate}}}"])
    ids = torch.randint(0, 96, (2, 16))
    stub = _instrumentation_stub(model, token_ids=ids)
    stub._log_layer_instrumentation()
    assert stub._instrumentation_failures == set(), "a readout raised"
    suffixes = {name.split("/", 1)[1] for name in stub.metric_logger.logged}
    diagnostics = set() if bank_gate in X_READING_GATES else _DIAGNOSTIC_KEYS
    assert suffixes == _READOUT_KEYS | diagnostics | _GATE_READOUT_KEYS[bank_gate]
    assert all(isinstance(v, float) for v in stub.metric_logger.logged.values())


def test_the_trust_readout_reports_the_init_and_moves_with_the_logits():
    layer = _layer(bank_gate="confidence")
    readout = layer.instrumentation()
    assert readout["trust_q/mean"] == pytest.approx(0.05, abs=1e-6)
    assert readout["trust_k/p90"] == pytest.approx(0.05, abs=1e-6)
    assert readout["trust_q/above_half"] == 0.0
    with torch.no_grad():
        layer.q_bank.weight[: VOCAB // 2, DIMS["dmodel"] :] = 4.0
    readout = layer.instrumentation()
    assert readout["trust_q/above_half"] == pytest.approx(0.5)
    assert readout["trust_k/above_half"] == 0.0


@pytest.mark.gpu
@pytest.mark.parametrize("bank_gate", GATES)
def test_on_cuda_every_gate_trains_under_compile_blocks_full(bank_gate):
    """The production path: whole compiled blocks, flex, bf16 autocast."""
    model = _h4_ame(
        [f"+common.attention.ame={{bank_gate: {bank_gate}, bank_init_std: 0.01}}"],
        vocab=96,
        device="cuda",
    )
    ids = torch.randint(0, 96, (2, 16)).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        eager = model(ids).float()
    for block in model.encoder.blocks:
        block.compile()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        compiled = model(ids).float()
    torch.testing.assert_close(compiled, eager, rtol=5e-2, atol=5e-2)
    loss = torch.nn.functional.cross_entropy(compiled.flatten(0, -2), ids.reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


# ---------------------------------------------------------------------------
# The trainer side of the bank-gates plan: token update counts, the banks'
# share of the gradient norm, and the clip factor.


def _counting_stub(vocab=50, world_start_step=0):
    from src.core.trainer import Trainer

    class Stub:
        _vocab_size = Trainer._vocab_size
        _count_tokens = Trainer._count_tokens
        _fold_token_counts = Trainer._fold_token_counts
        _save_token_counts = Trainer._save_token_counts

    stub = Stub()
    stub.model = torch.nn.Sequential(torch.nn.Embedding(vocab, 4))
    stub.start_step = world_start_step
    stub.step = -1
    stub._step_token_counts = None
    stub._token_occurrences = None
    stub._token_steps_seen = None
    stub._token_count_steps = 0
    return stub


def test_token_counts_count_occurrences_and_the_steps_a_row_has_a_gradient_in(
    tmp_path,
):
    """steps_seen is the number of optimizer steps a token occurs in -- the
    steps in which its bank rows have a gradient -- not the number of
    micro-batches or tokens."""
    stub = _counting_stub()
    steps = [
        [torch.tensor([[1, 1, 2]]), torch.tensor([[2, 3, 3]])],  # step 0
        [torch.tensor([[3, 4, 4]])],  # step 1
        [torch.tensor([[1, 0, 0]]), torch.tensor([[0, 0, 0]])],  # step 2
    ]
    for step, micro_batches in enumerate(steps):
        stub.step = step
        for ids in micro_batches:
            stub._count_tokens(ids)
        stub._fold_token_counts()
    occurrences = stub._token_occurrences
    steps_seen = stub._token_steps_seen
    assert occurrences[:5].tolist() == [5, 3, 2, 3, 2]
    assert steps_seen[:5].tolist() == [1, 2, 1, 2, 1]
    assert occurrences[5:].abs().sum() == 0 and steps_seen[5:].abs().sum() == 0
    assert int(occurrences.sum()) == sum(ids.numel() for s in steps for ids in s)

    stub._save_token_counts(str(tmp_path))
    saved = torch.load(tmp_path / "token_counts.pt")
    assert torch.equal(saved["occurrences"], occurrences.cpu())
    assert torch.equal(saved["steps_seen"], steps_seen.cpu())
    assert saved["steps"] == 3 and saved["first_step"] == 0 and saved["last_step"] == 2


def test_without_counting_no_token_counts_file_is_written(tmp_path):
    stub = _counting_stub()
    stub._save_token_counts(str(tmp_path))
    assert not (tmp_path / "token_counts.pt").exists()


def test_the_clip_reports_the_banks_share_and_the_factor_it_applied():
    """grad_norm/banks is sqrt of the bank term in the joint norm, and the clip
    factor is the coefficient both the backbone and the banks were scaled by."""
    from types import SimpleNamespace

    from src.core.memory_parallel import MemoryParallelController
    from src.core.trainer import Trainer

    layer = _perturb_gate(_layer(bank_gate="confidence"))
    controller = MemoryParallelController(
        layer, mem_parallel_size=1, compute_dtype=torch.float32
    )
    controller.materialize()
    x, token_ids = _batch(layer)
    (100.0 * layer(x, token_ids)).square().sum().backward()
    bank_norm = math.sqrt(
        sum(bank.main_grad.double().square().sum().item() for bank in controller.banks)
    )
    backbone = [p.grad.double() for p in layer.parameters() if p.grad is not None]
    total = math.sqrt(bank_norm**2 + sum(g.square().sum().item() for g in backbone))

    stub = SimpleNamespace(model=layer, gradient_clipping=1.0, _banks=controller)
    returned = Trainer.clip_gradient(stub)
    assert returned.item() == pytest.approx(total, rel=1e-5)
    assert stub._last_bank_grad_norm.item() == pytest.approx(bank_norm, rel=1e-5)
    assert stub._last_clip_factor.item() == pytest.approx(1.0 / total, rel=1e-5)
    # and the banks really were scaled by it
    after = math.sqrt(
        sum(bank.main_grad.double().square().sum().item() for bank in controller.banks)
    )
    assert after == pytest.approx(bank_norm / total, rel=1e-4)
