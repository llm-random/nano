"""The H-balanced contract: one integer knob fixes every dimension.

See documents/concepts/h-balanced-model.md. A model is named by its H, and depth,
width, head count, kv heads and FF width all follow. These tests instantiate the
shipped instance configs on the meta device (no memory is allocated, so h24's 1B
parameters cost nothing) and check both the derived dimensions and the closed-form
parameter count exactly -- not approximately, so a drifted interpolation cannot
hide inside a tolerance.
"""

import os

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import resolver  # noqa: F401  (registers the eval resolver the configs use)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB = 50304  # GPT-NeoX's 50277 tokens padded to a multiple of 128


def _h_model(h):
    with initialize_config_dir(
        version_base=None,
        config_dir=os.path.join(REPO_ROOT, "configs", "_model", "h_balanced"),
    ):
        cfg = compose(config_name=f"h{h}", overrides=["common.sequence_length=128"])
    with torch.device("meta"):
        return instantiate(cfg.model, _convert_="all")


@pytest.mark.parametrize("h", [4, 16, 24])
def test_every_dimension_follows_the_knob(h):
    model = _h_model(h)
    dmodel = 64 * h

    assert model.embedding.weight.shape == (VOCAB, dmodel)
    assert len(model.encoder.blocks) == h

    block = model.encoder.blocks[0]
    attention = block.attention_layer.layer
    assert attention.q_heads == h
    assert attention.kv_heads == h // 2
    assert attention.q_proj.weight.shape == (64 * h, dmodel)
    assert attention.k_proj.weight.shape == (64 * (h // 2), dmodel)
    assert attention.v_proj.weight.shape == (64 * (h // 2), dmodel)
    assert attention.o_proj.weight.shape == (dmodel, 64 * h)
    assert attention.dhead == 64, "head_dim must be 64 at every size"

    ff = block.ff_layer.layer
    assert ff.ff_pre_act.weight.shape == (256 * h, dmodel)
    assert ff.gate.weight.shape == (256 * h, dmodel)
    assert ff.ff_post_act.weight.shape == (dmodel, 256 * h)

    head = model.head
    assert head.linear.weight.shape == (VOCAB, dmodel), "embedding and head are untied"


@pytest.mark.parametrize("h", [4, 16, 24])
def test_the_closed_form_parameter_count_is_exact(h):
    """non-embedding = 15*d^2 per layer = 61440*H^3; emb+head = 2*V*64H; plus the
    RMSNorm weights (two per block and one in the head, each dmodel wide), which
    the concept doc's closed form leaves out as negligible but an exact test must
    not."""
    model = _h_model(h)
    total = sum(p.numel() for p in model.parameters())
    expected = 61440 * h**3 + 2 * VOCAB * 64 * h + (2 * h + 1) * 64 * h
    assert total == expected, f"H={h}: {total} != {expected} ({total - expected:+d})"
