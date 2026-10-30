"""The attention-type registry: configs name a type, this module builds the layer.

Every type shares one constructor contract so they are interchangeable per layer:

    (dmodel, q_heads, kv_heads, seq_len, rope_base, rope_scale_freqs, init_scale)

`mha`, `mqa` and `gqa` are real today and are all the same `RoPEAttention` -- they
differ only in how many kv heads they keep (mha: q_heads, mqa: 1, gqa: the config's
kv_heads) and in the k/v projection widths that follow. `mla`, `mamba2`,
`gated_deltanet` and `ame` are registered interfaces: they construct (so a config
naming them composes and instantiates today) and raise on forward until their real
implementations land. Adding a type = one builder function + one registry line.

The config surface lives in `common.attention` (see configs/_model/h_balanced/
base.yaml); `build_attention` is the `attention_fn` the transformer block calls
with its block id, which is what makes per-layer switching possible.
"""

from functools import partial

import torch.nn as nn

from src.core.model import Linear, RoPEAttention, llm_random_weight_init


def _linear_fn(in_features, out_features, init_scale):
    return partial(
        Linear,
        in_features,
        out_features,
        partial_init_fn=partial(llm_random_weight_init, scale=init_scale),
    )


def _rope_attention(
    dmodel, q_heads, kv_heads, seq_len, rope_base, rope_scale_freqs, init_scale
):
    dhead = dmodel // q_heads
    return RoPEAttention(
        q_proj_fn=_linear_fn(dmodel, dhead * q_heads, init_scale),
        k_proj_fn=_linear_fn(dmodel, dhead * kv_heads, init_scale),
        v_proj_fn=_linear_fn(dmodel, dhead * kv_heads, init_scale),
        o_proj_fn=_linear_fn(dhead * q_heads, dmodel, init_scale),
        dmodel=dmodel,
        q_heads=q_heads,
        kv_heads=kv_heads,
        seq_len=seq_len,
        rope_base=rope_base,
        rope_scale_freqs=rope_scale_freqs,
    )


def _build_mha(dmodel, q_heads, kv_heads, **kwargs):
    """Multi-head attention: every query head has its own kv head."""
    return _rope_attention(dmodel, q_heads, q_heads, **kwargs)


def _build_mqa(dmodel, q_heads, kv_heads, **kwargs):
    """Multi-query attention: one kv head shared by every query head."""
    return _rope_attention(dmodel, q_heads, 1, **kwargs)


def _build_gqa(dmodel, q_heads, kv_heads, **kwargs):
    """Grouped-query attention: the config's kv_heads (H/2 in the H family)."""
    if q_heads % kv_heads:
        raise ValueError(f"gqa needs q_heads % kv_heads == 0, got {q_heads}/{kv_heads}")
    return _rope_attention(dmodel, q_heads, kv_heads, **kwargs)


class _PlannedAttention(nn.Module):
    """A registered attention type whose implementation has not landed yet.

    It constructs with the shared contract -- so configs naming it compose,
    instantiate and can be pattern-mixed today -- and fails loudly at forward,
    which is the first moment a real run would need the missing math or kernel.
    """

    PLANNED = "planned for stage 3"

    def __init__(
        self,
        dmodel,
        q_heads,
        kv_heads,
        seq_len,
        rope_base,
        rope_scale_freqs,
        init_scale=1,
    ):
        super().__init__()
        self.dmodel = dmodel
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.seq_len = seq_len

    def forward(self, x):
        raise NotImplementedError(
            f"{type(self).__name__} is a registered interface without an "
            f"implementation yet ({self.PLANNED})."
        )


class MLAttention(_PlannedAttention):
    """Multi-head latent attention (DeepSeek-style kv compression)."""


class Mamba2Mixer(_PlannedAttention):
    """Mamba-2 state-space mixer; the real one brings its own scan kernel."""


class GatedDeltaNet(_PlannedAttention):
    """Gated DeltaNet linear-attention mixer; needs its chunked-scan kernel."""


class AMEAttention(_PlannedAttention):
    """AME (attention memory embedding) -- this project's own method."""


def _planned(cls):
    def build(dmodel, q_heads, kv_heads, **kwargs):
        return cls(dmodel, q_heads, kv_heads, **kwargs)

    return build


ATTENTION_TYPES = {
    "mha": _build_mha,
    "mqa": _build_mqa,
    "gqa": _build_gqa,
    "mla": _planned(MLAttention),
    "mamba2": _planned(Mamba2Mixer),
    "gated_deltanet": _planned(GatedDeltaNet),
    "ame": _planned(AMEAttention),
}


_NTH_PATTERNS = {"every_2nd": 2, "every_4th": 4}


def _check_type(name):
    if name not in ATTENTION_TYPES:
        raise ValueError(
            f"unknown attention type {name!r}; registered: {sorted(ATTENTION_TYPES)}"
        )
    return name


def resolve_layer_types(n_blocks, base_type, mix=None):
    """Which attention type each layer runs, as a list of n_blocks names.

    `mix` replaces the base type on a subset of layers:
        {"type": <name>, "layers": "every_2nd" | "every_4th" | "all" | [indices]}

    Convention: `every_Nth` means the 0-based indices i with (i + 1) % N == 0 --
    layers 3, 7, 11, ... for N=4 -- so layer 0 is never silently replaced.
    Explicit indices are 0-based; out-of-range or duplicate indices are config
    errors named at build time.
    """
    types = [_check_type(base_type)] * n_blocks
    if mix is None:
        return types
    unknown = set(mix) - {"type", "layers"}
    if unknown:
        raise ValueError(f"unknown mix keys {sorted(unknown)}; expected type, layers")
    name = _check_type(mix["type"])
    layers = mix["layers"]
    if isinstance(layers, str):
        if layers == "all":
            return [name] * n_blocks
        if layers not in _NTH_PATTERNS:
            raise ValueError(
                f"unknown layer pattern {layers!r}; "
                f"expected {sorted(_NTH_PATTERNS)}, 'all' or a list of indices"
            )
        step = _NTH_PATTERNS[layers]
        indices = [i for i in range(n_blocks) if (i + 1) % step == 0]
    else:
        indices = [int(i) for i in layers]
        out_of_range = [i for i in indices if not 0 <= i < n_blocks]
        if out_of_range:
            raise ValueError(
                f"mix layer indices {out_of_range} outside 0..{n_blocks - 1}"
            )
        if len(set(indices)) != len(indices):
            raise ValueError(f"duplicate mix layer indices in {indices}")
    for i in indices:
        types[i] = name
    return types


def build_attention(
    block_id,
    *,
    attention,
    n_blocks,
    dmodel,
    q_heads,
    kv_heads,
    seq_len,
    rope_base=10000,
    rope_scale_freqs=False,
    init_scale=1,
):
    """The `attention_fn` a TransformerBlock calls with its block id.

    `attention` is the config mapping: {"type": <name>} for a uniform stack, plus
    an optional "mix" (see resolve_layer_types) to swap the type on some layers.
    """
    unknown = set(attention) - {"type", "mix"}
    if unknown:
        raise ValueError(
            f"unknown attention config keys {sorted(unknown)}; expected type, mix"
        )
    if not 0 <= block_id < n_blocks:
        raise ValueError(f"block_id {block_id} outside 0..{n_blocks - 1}")
    name = resolve_layer_types(n_blocks, attention["type"], attention.get("mix"))[
        block_id
    ]
    return ATTENTION_TYPES[name](
        dmodel,
        q_heads,
        kv_heads,
        seq_len=seq_len,
        rope_base=rope_base,
        rope_scale_freqs=rope_scale_freqs,
        init_scale=init_scale,
    )
