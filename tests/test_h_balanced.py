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


@pytest.mark.parametrize("h", [4, 12, 16, 20, 24])
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
    dff = 3 * dmodel
    assert ff.ff_pre_act.weight.shape == (dff, dmodel)
    assert ff.gate.weight.shape == (dff, dmodel)
    assert ff.ff_post_act.weight.shape == (dmodel, dff)

    head = model.head
    assert head.linear.weight.shape == (VOCAB, dmodel), "embedding and head are untied"


@pytest.mark.parametrize("h", [4, 12, 16, 20, 24])
def test_the_closed_form_parameter_count_is_exact(h):
    """Per layer: attention 3*d^2 (q, o are d x d; k, v are d x d/2) plus gated FF
    3*d*dff with dff = 3d, so 12*d^2 = 49152*H^2; emb+head = 2*V*64H; plus the
    RMSNorm weights (two per block and one in the head, each dmodel wide) and the
    QK-norm gains (two per block, each dhead = 64 wide), which the closed form
    leaves out as negligible but an exact test must not."""
    model = _h_model(h)
    total = sum(p.numel() for p in model.parameters())
    d = 64 * h
    dff = 3 * d
    per_layer = 3 * d**2 + 3 * d * dff
    norms = (2 * h + 1) * 64 * h + 2 * 64 * h
    expected = h * per_layer + 2 * VOCAB * 64 * h + norms
    assert total == expected, f"H={h}: {total} != {expected} ({total - expected:+d})"
