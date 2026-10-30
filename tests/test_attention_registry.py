"""The attention registry: named types, one shared contract, stubs that construct.

mha/mqa/gqa are the same RoPEAttention with different kv head counts, so mha must be
bit-equal to gqa-with-kv_heads=q_heads -- not approximately, bitwise. The planned
types (mla, mamba2, gated_deltanet, ame) must construct from a config today and fail
only at forward, so the per-layer mixing can be wired and tested before their
implementations land.
"""

import os

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import resolver  # noqa: F401
from src.core.attention import ATTENTION_TYPES, build_attention

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REAL = ("mha", "mqa", "gqa")
PLANNED = ("mla", "mamba2", "gated_deltanet", "ame")

DIMS = dict(dmodel=64, q_heads=4, kv_heads=2, seq_len=16)


def _build(attention_type, seed=0, **overrides):
    torch.manual_seed(seed)
    kwargs = {**DIMS, **overrides}
    return build_attention(0, attention={"type": attention_type}, n_blocks=1, **kwargs)


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
