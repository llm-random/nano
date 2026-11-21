"""KV shifting on the dense types: the matched control for AME's shift.

Xu et al. 2024 (arXiv 2411.19574), the operator src.core.ame applies to its banks:
k_j <- a1*k_j + a2*k_{j-1} and v_j <- b1*v_j + b2*v_{j-1}, per kv head, BEFORE the
rotary embedding, queries untouched, init a1 ~ U(0, 1), a2 = 1 - a1. Without it any
AME-vs-dense gain is confounded with Xu's published one
(documents/notes/ame_implementation.md, KV shifting, open item 2). `attention.shift`
sets it on every layer of every type; `gqa` also takes it as an option.
"""

import os
import warnings

import pytest
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import resolver  # noqa: F401
from src.core.ame import _shift1
from src.core.attention import build_attention
from src.core.mfu import model_flops_per_token
from src.core.model import parameter_groups
from src.definitions import check_point

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DIMS = dict(dmodel=64, q_heads=4, kv_heads=2, seq_len=16, vocab_size=128)
DHEAD = DIMS["dmodel"] // DIMS["q_heads"]
DENSE = ("mha", "mqa", "gqa")


def _build(attention, seed=0):
    torch.manual_seed(seed)
    return build_attention(0, attention=attention, n_blocks=1, **DIMS)


def _x(seed=7, batch=2, n=DIMS["seq_len"]):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, n, DIMS["dmodel"], generator=generator)


def _set_mix(layer, alpha, beta):
    with torch.no_grad():
        layer.alpha.copy_(torch.as_tensor(alpha, dtype=torch.float32))
        layer.beta.copy_(torch.as_tensor(beta, dtype=torch.float32))


def _reference(layer, x, order="shift_then_rope"):
    """Xu et al.'s forward, written out: project, (normalize), shift, rope."""
    batch, n = x.shape[:2]
    heads, kv = layer.q_heads, layer.kv_heads
    q = layer.q_proj(x).view(batch, n, heads, DHEAD)
    k = layer.k_proj(x).view(batch, n, kv, DHEAD)
    v = layer.v_proj(x).view(batch, n, kv, DHEAD)
    if layer.qk_norm:
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    a = layer.alpha.view(1, 1, kv, 2)
    b = layer.beta.view(1, 1, kv, 2)
    q, v = q.transpose(1, 2), (b[..., :1] * v + b[..., 1:] * _shift1(v)).transpose(1, 2)
    if order == "shift_then_rope":
        k = (a[..., :1] * k + a[..., 1:] * _shift1(k)).transpose(1, 2)
        k = layer.rope(k)
    else:  # the wrong order: each half rotated at its own position
        k = layer.rope(k.transpose(1, 2))
        shifted = F.pad(k, (0, 0, 1, 0))[:, :, :-1]
        k = a.transpose(1, 2)[..., :1] * k + a.transpose(1, 2)[..., 1:] * shifted
    q = layer.rope(q)
    k = k.repeat_interleave(heads // kv, dim=1)
    v = v.repeat_interleave(heads // kv, dim=1)
    if layer.qk_norm:
        scores = layer.tau.view(1, heads, 1, 1) * (q @ k.transpose(-1, -2))
    else:
        scores = (q @ k.transpose(-1, -2)) / DHEAD**0.5
    future = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    weights = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
    return layer.o_proj((weights @ v).transpose(1, 2).flatten(-2))


@pytest.mark.parametrize("name", DENSE)
def test_off_is_the_dense_default_bit_for_bit(name):
    default = _build({"type": name})
    off = _build({"type": name, "shift": False})
    assert not default.shift and not off.shift
    assert not hasattr(default, "alpha") and not hasattr(default, "beta")
    for (key, p), (_, q) in zip(
        default.named_parameters(), off.named_parameters(), strict=True
    ):
        assert torch.equal(p, q), key
    torch.testing.assert_close(default(_x()), off(_x()), rtol=0, atol=0)


@pytest.mark.parametrize("name", DENSE)
def test_on_adds_one_mix_per_kv_head_and_nothing_else(name):
    off = _build({"type": name})
    on = _build({"type": name, "shift": True})
    for mix in (on.alpha, on.beta):
        assert mix.shape == (on.kv_heads, 2)
        assert torch.all((mix[:, 0] >= 0) & (mix[:, 0] <= 1))
        torch.testing.assert_close(mix.sum(dim=-1), torch.ones(on.kv_heads))
    on_params = dict(on.named_parameters())
    assert set(on_params) - set(dict(off.named_parameters())) == {"alpha", "beta"}
    # Drawn after the projections: the same seed builds the same projections.
    for key, p in off.named_parameters():
        assert torch.equal(p, on_params[key]), key


@pytest.mark.parametrize("qk_norm", [False, True])
def test_the_forward_is_xu_et_al_with_the_shift_before_rope(qk_norm):
    """gqa (4 query heads over 2 kv heads), a different mix per kv head."""
    layer = _build({"type": "gqa", "shift": True, "qk_norm": qk_norm}, seed=1)
    _set_mix(layer, [[0.7, 0.3], [0.2, 0.8]], [[0.4, 0.6], [0.9, 0.1]])
    x = _x()
    with torch.no_grad():
        expected = _reference(layer, x)
        wrong_order = _reference(layer, x, order="rope_then_shift")
        actual = layer(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    # The order is observable, so the test above would catch a swap.
    assert not torch.allclose(actual, wrong_order, rtol=1e-3, atol=1e-4)


def test_pinned_at_one_zero_it_is_the_plain_layer():
    plain = _build({"type": "gqa"})
    shifted = _build({"type": "gqa", "shift": True})
    _set_mix(shifted, [[1.0, 0.0]] * 2, [[1.0, 0.0]] * 2)
    torch.testing.assert_close(shifted(_x()), plain(_x()), rtol=0, atol=0)


def test_a_fully_shifted_head_sees_nothing_at_position_zero():
    """Position 0 has no predecessor: its shifted key and value are zeros (no
    wrap from the end of the row), so it attends to a zero value and outputs
    exactly zero -- the projections carry no bias."""
    layer = _build({"type": "gqa", "shift": True})
    _set_mix(layer, [[0.0, 1.0]] * 2, [[0.0, 1.0]] * 2)
    with torch.no_grad():
        out = layer(_x())
    assert torch.equal(out[:, 0], torch.zeros_like(out[:, 0]))
    assert out[:, 1:].abs().sum() > 0


def test_the_mix_learns_and_costs_no_decay_and_no_flops():
    layer = _build({"type": "gqa", "shift": True})
    layer(_x()).square().mean().backward()
    assert torch.all(layer.alpha.grad != 0) and torch.all(layer.beta.grad != 0)

    decayed, undecayed = parameter_groups(layer, weight_decay=0.1)
    assert {id(p) for p in undecayed["params"]} == {id(layer.alpha), id(layer.beta)}
    assert all(p is not layer.alpha and p is not layer.beta for p in decayed["params"])

    # [kv_heads, 2] is 2-D, but not a matmul weight.
    plain = _build({"type": "gqa"})
    assert model_flops_per_token(layer, 16) == model_flops_per_token(plain, 16)


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize("name", [*DENSE, "ame"])
def test_the_stack_wide_switch_reaches_every_real_type(name, value):
    layer = _build({"type": name, "shift": value})
    if name == "ame":  # AME's shift off pins both mixes to (1, 0) buffers
        assert layer.alpha_is_pinned is (not value)
        assert layer.beta_is_pinned is (not value)
    else:
        assert layer.shift is value


def test_ame_keeps_its_shift_when_nothing_is_said():
    assert not _build({"type": "ame"}).alpha_is_pinned
    assert not _build({"type": "ame", "shift": None}).alpha_is_pinned
    assert _build({"type": "ame", "ame": {"shift": False}}).alpha_is_pinned


def test_the_gqa_option_alone_turns_it_on():
    layer = _build({"type": "gqa", "gqa": {"shift": True, "rope": False}})
    assert layer.shift and layer.rope is None
    assert layer(_x()).shape == _x().shape


@pytest.mark.parametrize(
    "attention",
    [
        {"type": "gqa", "shift": True, "gqa": {"shift": False}},
        {"type": "ame", "shift": False, "ame": {"shift": True}},
        {"type": "gqa", "shift": False, "ame": {"shift": True}},  # unused block too
    ],
)
def test_a_contradiction_is_a_build_error(attention):
    with pytest.raises(ValueError, match="contradicts"):
        _build(attention)


def _h4_model(overrides, device="cpu"):
    with initialize_config_dir(
        version_base=None,
        config_dir=os.path.join(REPO_ROOT, "configs", "_model", "h_balanced"),
    ):
        cfg = compose(
            config_name="h4", overrides=["common.sequence_length=16", *overrides]
        )
    with torch.device(device):
        return instantiate(cfg.model, _convert_="all")


def test_a_mixed_stack_matching_ame_on_both_switches_trains_on_cpu():
    """The control the AME comparison needs, in one model: dense layers with
    AME's QK-norm and KV shifting beside AME layers."""
    torch.manual_seed(0)
    model = _h4_model(
        [
            "+common.attention.qk_norm=true",
            "+common.attention.shift=true",
            "+common.attention.mix={type: ame, layers: every_2nd}",
        ]
    )
    layers = [b.attention_layer.layer for b in model.encoder.blocks]
    dense = [m for m in layers if type(m).__name__ == "RoPEAttention"]
    assert len(dense) == 2 and all(m.shift and m.qk_norm for m in dense)
    x = torch.randint(0, 50304, (2, 16))
    loss = F.cross_entropy(model(x).flatten(0, -2), x.reshape(-1))
    loss.backward()
    for m in dense:
        assert all(p.grad is not None for p in (m.alpha, m.beta, m.tau))


def test_the_schema_admits_the_switch():
    warnings.filterwarnings("ignore")
    with initialize_config_dir(
        version_base=None, config_dir=os.path.join(REPO_ROOT, "configs")
    ):
        cfg = compose(
            config_name="tiny_local", overrides=["common.attention.shift=true"]
        )
    assert cfg.common.attention.shift is True
    check_point(cfg)
