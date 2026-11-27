"""Model initialization that no config switch can perturb (src/core/init_rng.py).

Each component -- the token embedding, every block's attention and feed-forward,
the head -- is built from its own seed (the run's seed and the component's name),
and an optional parameter inside one draws from a private generator. So a switch
changes only the parameters it adds or changes, never another one's init, and the
global stream after the model is built does not depend on the model
(documents/notes/archive/review_issues.md, N6). Before, the dense KV shift alone changed
25 of a small model's 39 shared parameters.
"""

import os

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import resolver  # noqa: F401  (registers the repo's custom OmegaConf resolvers)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 1234
DENSE = ["common.attention.type=gqa"]
AME = ["common.attention.type=ame"]


def _build(overrides, seed=SEED):
    """An h4 model built as main.py builds one: torch.manual_seed, then
    instantiate. Returns it with the global stream's state right after."""
    with initialize_config_dir(
        version_base=None,
        config_dir=os.path.join(REPO_ROOT, "configs", "_model", "h_balanced"),
    ):
        cfg = compose(
            config_name="h4",
            overrides=["common.sequence_length=16", "common.vocab_size=96", *overrides],
        )
    torch.manual_seed(seed)
    model = instantiate(cfg.model, _convert_="all")
    return model, torch.random.get_rng_state()


def _changed(a, b):
    """The shared parameters (same name and shape) whose values differ, and how
    many are shared."""
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    shared = [n for n in pa if n in pb and pa[n].shape == pb[n].shape]
    return [n for n in shared if not torch.equal(pa[n], pb[n])], len(shared)


# (switch, the "off" overrides, the "on" overrides, name substrings the switch
# is allowed to change among the shared parameters)
SWITCHES = [
    ("dense KV shift", DENSE, DENSE + ["++common.attention.shift=true"], ()),
    ("dense QK-norm", DENSE, DENSE + ["++common.attention.qk_norm=true"], ()),
    (
        "dense QK-norm and shift",
        DENSE,
        DENSE + ["++common.attention.qk_norm=true", "++common.attention.shift=true"],
        (),
    ),
    ("AME shift", AME, AME + ["++common.attention.ame={shift: false}"], ()),
    ("AME pin_alpha", AME, AME + ["++common.attention.ame={pin_alpha: true}"], ()),
    ("AME pin_beta", AME, AME + ["++common.attention.ame={pin_beta: true}"], ()),
    ("AME sink", AME, AME + ["++common.attention.ame={sink: false}"], ()),
    (
        "AME learnable slopes",
        AME,
        AME + ["++common.attention.ame={learnable_slopes: true}"],
        (),
    ),
    (
        "AME QK-norm, same sigma",
        AME + ["++common.attention.ame={bank_init_std: 1.0}"],
        AME + ["++common.attention.ame={qk_norm: false, bank_init_std: 1.0}"],
        (".tau",),
    ),
    (
        "AME bank sigma",
        AME,
        AME + ["++common.attention.ame={bank_init_std: 0.01}"],
        ("_bank",),
    ),
    *[
        (
            f"AME gate {gate}",
            AME,
            AME + [f"++common.attention.ame={{bank_gate: {gate}}}"],
            (),
        )
        for gate in ("confidence", "context", "backoff", "stem")
    ],
    ("dense NoPE", DENSE, DENSE + ["++common.attention.gqa={rope: false}"], ()),
    ("attention type", DENSE, AME, ("attention_layer.layer",)),
    (
        "mha against gqa",
        DENSE,
        ["common.attention.type=mha"],
        ("attention_layer.layer",),
    ),
    (
        "an AME layer in a dense stack",
        DENSE,
        DENSE + ["++common.attention.mix={type: ame, layers: every_2nd}"],
        ("attention_layer.layer",),
    ),
]


@pytest.mark.parametrize(
    "off, on, allowed", [s[1:] for s in SWITCHES], ids=[s[0] for s in SWITCHES]
)
def test_a_switch_changes_only_what_it_switches(off, on, allowed):
    model_off, stream_off = _build(off)
    model_on, stream_on = _build(on)
    changed, shared = _changed(model_off, model_on)
    assert shared > 10, "the comparison found almost nothing in common"
    unexpected = [n for n in changed if not any(a in n for a in allowed)]
    assert (
        not unexpected
    ), f"the switch moved {len(unexpected)} other inits: {unexpected}"
    # And everything drawn after the model is built (the data loaders' base
    # seeds among it) sees the same global stream.
    assert torch.equal(stream_off, stream_on)


def test_building_the_model_leaves_the_global_stream_as_seeded():
    torch.manual_seed(SEED)
    seeded = torch.random.get_rng_state()
    _, after = _build(AME)
    assert torch.equal(after, seeded)


def test_every_component_draws_from_its_own_seed():
    model, _ = _build(DENSE)
    blocks = model.encoder.blocks
    assert not torch.equal(
        blocks[0].ff_layer.layer.ff_pre_act.weight,
        blocks[1].ff_layer.layer.ff_pre_act.weight,
    )
    assert not torch.equal(
        blocks[0].attention_layer.layer.q_proj.weight,
        blocks[1].attention_layer.layer.q_proj.weight,
    )
    # One draw per optional parameter: alpha and beta are not the same numbers.
    shifted, _ = _build(DENSE + ["++common.attention.shift=true"])
    layer = shifted.encoder.blocks[0].attention_layer.layer
    assert not torch.equal(layer.alpha, layer.beta)


def test_the_seed_still_decides_the_init():
    a, _ = _build(AME, seed=SEED)
    b, _ = _build(AME, seed=SEED)
    c, _ = _build(AME, seed=SEED + 1)
    assert _changed(a, b)[0] == []
    changed, shared = _changed(a, c)
    assert len(changed) == shared - _deterministic(a)


def _deterministic(model):
    """Parameters every seed builds the same: norms, tau, sinks, zero-init."""
    return sum(
        1
        for name, p in model.named_parameters()
        if p.numel() == 0 or torch.all(p == p.flatten()[0])
    )
