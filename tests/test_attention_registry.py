"""The attention registry: named types, one shared contract, stubs that construct.

mha/mqa/gqa are the same RoPEAttention with different kv head counts, so mha must be
bit-equal to gqa-with-kv_heads=q_heads -- not approximately, bitwise. `ame` is real
too and lives in src.core.ame (its own math is tested in tests/test_ame.py); here it
only has to behave like a registry citizen. The planned types (mla, mamba2,
gated_deltanet) must construct from a config today and fail only at forward, so the
per-layer mixing can be wired and tested before their implementations land.
"""

import os

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import resolver  # noqa: F401
from src.core.attention import ATTENTION_TYPES, build_attention, resolve_layer_types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REAL = ("mha", "mqa", "gqa", "ame")
PLANNED = ("mla", "mamba2", "gated_deltanet")

DIMS = dict(dmodel=64, q_heads=4, kv_heads=2, seq_len=16, vocab_size=128)


def _build(attention_type, seed=0, attention=None, **overrides):
    torch.manual_seed(seed)
    kwargs = {**DIMS, **overrides}
    return build_attention(
        0, attention=attention or {"type": attention_type}, n_blocks=1, **kwargs
    )


def test_the_registry_holds_exactly_the_planned_set():
    assert set(ATTENTION_TYPES) == set(REAL) | set(PLANNED)


def test_mha_is_gqa_with_a_kv_head_per_query_head():
    mha = _build("mha")
    gqa = _build("gqa", kv_heads=DIMS["q_heads"])
    for (name, p), (_, q) in zip(
        mha.named_parameters(), gqa.named_parameters(), strict=True
    ):
        assert torch.equal(p, q), f"{name} differs between mha and gqa(kv=q)"
    x = torch.randn(
        2, DIMS["seq_len"], DIMS["dmodel"], generator=torch.Generator().manual_seed(7)
    )
    torch.testing.assert_close(mha(x), gqa(x), rtol=0, atol=0)


def test_mqa_keeps_exactly_one_kv_head():
    mqa = _build("mqa")
    dhead = DIMS["dmodel"] // DIMS["q_heads"]
    assert mqa.kv_heads == 1
    assert mqa.k_proj.weight.shape == (dhead, DIMS["dmodel"])
    assert mqa.v_proj.weight.shape == (dhead, DIMS["dmodel"])
    x = torch.randn(2, DIMS["seq_len"], DIMS["dmodel"])
    assert mqa(x).shape == x.shape


def test_gqa_rejects_a_head_count_that_does_not_divide():
    with pytest.raises(ValueError, match="q_heads % kv_heads"):
        _build("gqa", q_heads=4, kv_heads=3)


@pytest.mark.parametrize("name", PLANNED)
def test_planned_types_construct_but_refuse_to_run(name):
    module = _build(name)
    x = torch.randn(2, DIMS["seq_len"], DIMS["dmodel"])
    with pytest.raises(NotImplementedError, match="registered interface"):
        module(x)


def test_an_unknown_type_is_named_in_the_error():
    with pytest.raises(ValueError, match="flash_mamba_9000"):
        _build("flash_mamba_9000")


def test_an_unknown_config_key_is_a_build_error():
    with pytest.raises(ValueError, match="unknown attention config keys"):
        build_attention(0, attention={"type": "gqa", "typo_key": 1}, n_blocks=1, **DIMS)


def test_a_block_id_outside_the_stack_is_a_build_error():
    with pytest.raises(ValueError, match="block_id"):
        build_attention(5, attention={"type": "gqa"}, n_blocks=4, **DIMS)


@pytest.mark.parametrize("name", ["mha", "mqa", "mla", "ame"])
def test_the_h4_config_builds_with_any_registered_type(name):
    """The config surface: overriding common.attention.type swaps every layer."""
    with initialize_config_dir(
        version_base=None,
        config_dir=os.path.join(REPO_ROOT, "configs", "_model", "h_balanced"),
    ):
        cfg = compose(
            config_name="h4",
            overrides=["common.sequence_length=16", f"common.attention.type={name}"],
        )
    with torch.device("meta"):
        model = instantiate(cfg.model, _convert_="all")
    layer = model.encoder.blocks[0].attention_layer.layer
    expected = {
        "mha": "RoPEAttention",
        "mqa": "RoPEAttention",
        "mla": "MLAttention",
        "ame": "AMEAttention",
    }[name]
    assert type(layer).__name__ == expected
    if name == "mha":
        assert layer.kv_heads == layer.q_heads
    if name == "mqa":
        assert layer.kv_heads == 1


# ---------------------------------------------------------------------------
# Layer-pattern mixing: resolve_layer_types and the config surface.


def test_every_2nd_replaces_the_odd_indices():
    types = resolve_layer_types(4, "gqa", {"type": "mqa", "layers": "every_2nd"})
    assert types == ["gqa", "mqa", "gqa", "mqa"]


def test_every_4th_never_touches_layer_zero():
    types = resolve_layer_types(16, "gqa", {"type": "ame", "layers": "every_4th"})
    assert [i for i, t in enumerate(types) if t == "ame"] == [3, 7, 11, 15]
    assert types[0] == "gqa"


def test_all_makes_the_mix_type_the_whole_stack():
    assert (
        resolve_layer_types(4, "gqa", {"type": "mha", "layers": "all"}) == ["mha"] * 4
    )


def test_an_explicit_index_list_replaces_exactly_those_layers():
    types = resolve_layer_types(6, "gqa", {"type": "mla", "layers": [0, 2]})
    assert types == ["mla", "gqa", "mla", "gqa", "gqa", "gqa"]


def test_no_mix_keeps_a_uniform_stack():
    assert resolve_layer_types(3, "gqa") == ["gqa"] * 3


@pytest.mark.parametrize(
    "mix, message",
    [
        ({"type": "mqa", "layers": [7]}, "outside"),
        ({"type": "mqa", "layers": [1, 1]}, "duplicate"),
        ({"type": "mqa", "layers": "every_3rd"}, "unknown layer pattern"),
        ({"type": "warpdrive", "layers": "all"}, "warpdrive"),
        ({"type": "mqa", "layers": "all", "offset": 1}, "unknown mix keys"),
    ],
)
def test_a_bad_mix_is_a_named_config_error(mix, message):
    with pytest.raises(ValueError, match=message):
        resolve_layer_types(4, "gqa", mix)


def _h4_model(overrides, device="meta"):
    with initialize_config_dir(
        version_base=None,
        config_dir=os.path.join(REPO_ROOT, "configs", "_model", "h_balanced"),
    ):
        cfg = compose(
            config_name="h4", overrides=["common.sequence_length=16", *overrides]
        )
    with torch.device(device):
        return instantiate(cfg.model, _convert_="all")


def test_a_mixed_stack_builds_and_trains_on_cpu():
    """Two real types in one model: gqa everywhere, mqa on layers 1 and 3."""
    torch.manual_seed(0)
    model = _h4_model(
        ["+common.attention.mix={type: mqa, layers: every_2nd}"], device="cpu"
    )
    kv = [b.attention_layer.layer.kv_heads for b in model.encoder.blocks]
    assert kv == [2, 1, 2, 1]
    x = torch.randint(0, 50304, (2, 16))
    loss = torch.nn.functional.cross_entropy(model(x).flatten(0, -2), x.reshape(-1))
    loss.backward()
    grads = [p.grad is not None for p in model.parameters()]
    assert all(grads), "a mixed stack must train end to end"


def test_a_stubbed_type_mixes_in_but_fails_only_at_forward():
    """The wiring is ready for a type before the type exists: the config
    composes, the model instantiates, and the failure happens exactly at the
    stub's forward. (AME took this path until it landed; mla still does.)"""
    model = _h4_model(
        ["+common.attention.mix={type: mla, layers: every_4th}"], device="cpu"
    )
    names = [type(b.attention_layer.layer).__name__ for b in model.encoder.blocks]
    assert names == ["RoPEAttention"] * 3 + ["MLAttention"]
    with pytest.raises(NotImplementedError, match="registered interface"):
        model(torch.randint(0, 50304, (1, 16)))


def test_an_unknown_per_type_options_key_is_a_build_error():
    with pytest.raises(ValueError, match="unknown attention.ame options"):
        _build("ame", attention={"type": "ame", "ame": {"shfit": True}})


def test_ame_without_a_vocab_size_says_why():
    with pytest.raises(ValueError, match="indexed by token id"):
        build_attention(
            0,
            attention={"type": "ame"},
            n_blocks=1,
            **{k: v for k, v in DIMS.items() if k != "vocab_size"},
        )


@pytest.mark.parametrize("name", ["mha", "mla"])
def test_an_options_block_for_a_type_that_takes_none_is_rejected(name):
    """Only ame and gqa read an options block. The key check used to allow a
    block named after *any* registered type, so `{type: gqa, mha: {...}}`
    composed, validated, and was then dropped by the builder's kwarg filter -- a
    sweep knob that does nothing and says nothing."""
    with pytest.raises(ValueError, match="unknown attention config keys"):
        _build("gqa", attention={"type": "gqa", name: {"anything": 1}})


def test_an_unknown_gqa_option_is_rejected():
    with pytest.raises(ValueError, match="unknown attention.gqa options"):
        _build("gqa", attention={"type": "gqa", "gqa": {"anything": 1}})


def test_gqa_rope_true_is_the_default_bit_for_bit():
    """The NoPE knob must not move a single default dense run: `rope: true` and
    no options block build the same layer and compute the same function."""
    default = _build("gqa")
    explicit = _build("gqa", attention={"type": "gqa", "gqa": {"rope": True}})
    x = torch.randn(2, DIMS["seq_len"], DIMS["dmodel"])
    torch.testing.assert_close(default(x), explicit(x), rtol=0, atol=0)


def test_gqa_rope_false_is_nope():
    """Without RoPE a causal head sees the tokens before it as a set: swapping two
    earlier tokens leaves a later token's output unchanged. With RoPE it does not."""
    rope = _build("gqa", seed=3)
    nope = _build("gqa", seed=3, attention={"type": "gqa", "gqa": {"rope": False}})
    assert nope.rope is None and rope.rope is not None
    for (_, p), (_, q) in zip(
        rope.named_parameters(), nope.named_parameters(), strict=True
    ):
        assert torch.equal(p, q)  # the knob changes no parameter
    x = torch.randn(1, 6, DIMS["dmodel"])
    swapped = x[:, [1, 0, 2, 3, 4, 5]]
    torch.testing.assert_close(nope(x)[:, 2:], nope(swapped)[:, 2:])
    assert not torch.allclose(rope(x)[:, 2:], rope(swapped)[:, 2:])


def test_the_error_names_the_types_that_do_take_options():
    """A rejection has to point somewhere: the message lists the types with an
    options block, not every registered type."""
    with pytest.raises(ValueError) as caught:
        _build("gqa", attention={"type": "gqa", "nonsense": {}})
    assert "'ame'" in str(caught.value)
    assert "mamba2" not in str(caught.value)
