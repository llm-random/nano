"""QK-norm on every attention type: AME's operator, now selectable for the dense ones.

q and k l2-normalized per head, the score scaled by a learned per-head temperature
`tau` (init sqrt(dhead)) instead of 1/sqrt(dhead). `attention.qk_norm` sets it on
every layer of every type; `gqa` and `ame` also take it as an option, which may
repeat the stack-wide value but not contradict it. Off -- the dense default -- must
not move a bit, and AME keeps its own default (on) when nothing is said.
"""

import math
import os
import warnings

import pytest
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from hydra.utils import instantiate

import resolver  # noqa: F401
from src.core.attention import build_attention
from src.core.model import parameter_groups
from src.definitions import check_point

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DIMS = dict(dmodel=64, q_heads=4, kv_heads=2, seq_len=16, vocab_size=128)
DHEAD = DIMS["dmodel"] // DIMS["q_heads"]
DENSE = ("mha", "mqa", "gqa")


def _build(attention, seed=0):
    torch.manual_seed(seed)
    return build_attention(0, attention=attention, n_blocks=1, **DIMS)


def _x(seed=7, batch=2):
    return torch.randn(
        batch,
        DIMS["seq_len"],
        DIMS["dmodel"],
        generator=torch.Generator().manual_seed(seed),
    )


@pytest.mark.parametrize("name", DENSE)
def test_off_is_the_dense_default_bit_for_bit(name):
    default = _build({"type": name})
    off = _build({"type": name, "qk_norm": False})
    assert not default.qk_norm and not off.qk_norm
    assert "tau" not in dict(default.named_parameters())
    for (key, p), (_, q) in zip(
        default.named_parameters(), off.named_parameters(), strict=True
    ):
        assert torch.equal(p, q), key
    torch.testing.assert_close(default(_x()), off(_x()), rtol=0, atol=0)


def test_the_gqa_option_off_is_the_default_bit_for_bit():
    default = _build({"type": "gqa"})
    off = _build({"type": "gqa", "gqa": {"qk_norm": False}})
    torch.testing.assert_close(default(_x()), off(_x()), rtol=0, atol=0)


@pytest.mark.parametrize("name", DENSE)
def test_on_adds_one_temperature_per_query_head_and_nothing_else(name):
    off = _build({"type": name})
    on = _build({"type": name, "qk_norm": True})
    assert on.qk_norm
    assert on.tau.shape == (DIMS["q_heads"],)
    assert torch.all(on.tau == math.sqrt(DHEAD))
    # Same seed, same projections: tau is built without touching the RNG.
    on_params = dict(on.named_parameters())
    assert set(on_params) - set(dict(off.named_parameters())) == {"tau"}
    for key, p in off.named_parameters():
        assert torch.equal(p, on_params[key]), key


def test_the_forward_matches_a_reference():
    """gqa (4 query heads over 2 kv heads) with RoPE, a different tau per head."""
    layer = _build({"type": "gqa", "qk_norm": True}, seed=1)
    with torch.no_grad():
        layer.tau.copy_(torch.tensor([1.0, 2.0, 4.0, 8.0]))
    x = _x()
    batch, n = x.shape[:2]
    heads, kv = DIMS["q_heads"], DIMS["kv_heads"]

    with torch.no_grad():
        q = layer.q_proj(x).view(batch, n, heads, DHEAD).transpose(1, 2)
        k = layer.k_proj(x).view(batch, n, kv, DHEAD).transpose(1, 2)
        v = layer.v_proj(x).view(batch, n, kv, DHEAD).transpose(1, 2)
        q = layer.rope(F.normalize(q, dim=-1))
        k = layer.rope(F.normalize(k, dim=-1))
        k = k.repeat_interleave(heads // kv, dim=1)
        v = v.repeat_interleave(heads // kv, dim=1)
        scores = layer.tau.view(1, heads, 1, 1) * (q @ k.transpose(-1, -2))
        future = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        weights = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
        expected = layer.o_proj((weights @ v).transpose(1, 2).flatten(-2))
        torch.testing.assert_close(layer(x), expected, rtol=1e-5, atol=1e-5)


def test_scores_see_only_the_direction_of_q_and_k():
    """Rescaling W_Q and W_K changes nothing with QK-norm and everything without."""
    x = _x()
    for qk_norm in (True, False):
        layer = _build({"type": "gqa", "qk_norm": qk_norm})
        with torch.no_grad():
            before = layer(x)
            layer.q_proj.weight.mul_(7.0)
            layer.k_proj.weight.mul_(0.3)
            after = layer(x)
        if qk_norm:
            torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)
        else:
            assert not torch.allclose(before, after, rtol=1e-3, atol=1e-4)


def test_tau_learns_and_stays_out_of_weight_decay():
    layer = _build({"type": "gqa", "qk_norm": True})
    layer(_x()).square().mean().backward()
    assert layer.tau.grad is not None and torch.all(layer.tau.grad != 0)

    decayed, undecayed = parameter_groups(layer, weight_decay=0.1)
    assert undecayed["weight_decay"] == 0.0
    assert [p is layer.tau for p in undecayed["params"]] == [True]
    assert all(p is not layer.tau for p in decayed["params"])


def test_without_qk_norm_the_optimizer_stays_one_group():
    """The class declares `tau` as no-decay; a layer without one must not split
    the optimizer, or every existing dense run would change its param groups."""
    groups = parameter_groups(_build({"type": "gqa"}), weight_decay=0.1)
    assert len(groups) == 1 and groups[0]["weight_decay"] == 0.1


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize("name", [*DENSE, "ame"])
def test_the_stack_wide_switch_reaches_every_real_type(name, value):
    assert _build({"type": name, "qk_norm": value}).qk_norm is value


def test_ame_keeps_its_default_when_nothing_is_said():
    default = _build({"type": "ame"})
    unset = _build({"type": "ame", "qk_norm": None})
    assert default.qk_norm is True and unset.qk_norm is True
    for (key, p), (_, q) in zip(
        default.named_parameters(), unset.named_parameters(), strict=True
    ):
        assert torch.equal(p, q), key
    # The ame option still works on its own.
    assert _build({"type": "ame", "ame": {"qk_norm": False}}).qk_norm is False


def test_the_gqa_option_alone_turns_it_on_and_composes_with_nope():
    assert _build({"type": "gqa", "gqa": {"qk_norm": True}}).qk_norm is True
    nope = _build({"type": "gqa", "gqa": {"rope": False, "qk_norm": True}})
    assert nope.rope is None and nope.qk_norm is True
    assert nope(_x()).shape == _x().shape


@pytest.mark.parametrize(
    "attention",
    [
        {"type": "gqa", "qk_norm": True, "gqa": {"qk_norm": False}},
        {"type": "ame", "qk_norm": False, "ame": {"qk_norm": True}},
        # A contradicting block fails even where no layer is of its type.
        {"type": "gqa", "qk_norm": False, "ame": {"qk_norm": True}},
    ],
)
def test_a_contradiction_is_a_build_error(attention):
    with pytest.raises(ValueError, match="contradicts"):
        _build(attention)


def test_agreeing_settings_build():
    attention = {"type": "gqa", "qk_norm": True, "gqa": {"qk_norm": True}}
    assert _build(attention).qk_norm is True


def test_planned_types_construct_with_the_switch_set():
    _build({"type": "mla", "qk_norm": True})


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


def test_a_mixed_stack_with_qk_norm_everywhere_trains_on_cpu():
    torch.manual_seed(0)
    model = _h4_model(
        [
            "+common.attention.qk_norm=true",
            "+common.attention.mix={type: ame, layers: every_2nd}",
        ]
    )
    layers = [b.attention_layer.layer for b in model.encoder.blocks]
    assert [type(m).__name__ for m in layers] == ["RoPEAttention", "AMEAttention"] * 2
    assert all(m.qk_norm for m in layers)
    x = torch.randint(0, 50304, (2, 16))
    loss = F.cross_entropy(model(x).flatten(0, -2), x.reshape(-1))
    loss.backward()
    assert all(m.tau.grad is not None for m in layers)


def _compose_tiny(overrides):
    warnings.filterwarnings("ignore")
    with initialize_config_dir(
        version_base=None, config_dir=os.path.join(REPO_ROOT, "configs")
    ):
        return compose(config_name="tiny_local", overrides=list(overrides))


def test_the_schema_admits_the_switch_and_the_gqa_option():
    cfg = _compose_tiny(["common.attention.qk_norm=true"])
    assert cfg.common.attention.qk_norm is True
    check_point(cfg)
    cfg = _compose_tiny(["common.attention.gqa={qk_norm: true}"])
    check_point(cfg)


def test_a_non_boolean_switch_fails_composition():
    with pytest.raises(ConfigCompositionException):
        _compose_tiny(["common.attention.qk_norm=sometimes"])
