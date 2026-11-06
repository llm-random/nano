"""The attention-type registry: configs name a type, this module builds the layer.

Every type shares one constructor contract so they are interchangeable per layer:

    (dmodel, q_heads, kv_heads, seq_len, rope_base, rope_scale_freqs, init_scale)

`mha`, `mqa` and `gqa` are real today and are all the same `RoPEAttention` -- they
differ only in how many kv heads they keep (mha: q_heads, mqa: 1, gqa: the config's
kv_heads) and in the k/v projection widths that follow. `ame` is real and lives in
`src.core.ame`: it takes two extra pieces of context the others do not -- the vocab
size (its banks are indexed by token id) and the token ids themselves at forward,
which is why it declares `wants_token_ids`. `mla`, `mamba2` and `gated_deltanet` are
registered interfaces: they construct (so a config naming them composes and
instantiates today) and raise on forward until their real implementations land.
Adding a type = one builder function + one registry line.

The config surface lives in `common.attention` (see configs/_model/h_balanced/
base.yaml); `build_attention` is the `attention_fn` the transformer block calls
with its block id, which is what makes per-layer switching possible. A type may
take its own options block there -- `attention: {type: ame, ame: {...}}` -- which
applies wherever that type appears, base or mix.

One of AME's mechanisms is a switch every real type honors, with AME's own
operator, so a dense baseline can match it:

- `qk_norm`: q and k l2-normalized per head, the score scaled by a learned
  per-head temperature (`tau`, init sqrt(dhead)) instead of 1/sqrt(dhead).

`attention.qk_norm` sets it on every layer of every type; left null, each type
keeps its default (mha/mqa/gqa off, ame on). `gqa` and `ame` also take it as an
option, which may repeat the stack-wide value but not contradict it -- that is a
build error, not a silent win for either side. The planned types ignore it
until they have a forward.
"""

from functools import partial

import torch.nn as nn

from src.core.ame import AMEAttention
from src.core.model import Linear, RoPEAttention, llm_random_weight_init


def _linear_fn(in_features, out_features, init_scale):
    return partial(
        Linear,
        in_features,
        out_features,
        partial_init_fn=partial(llm_random_weight_init, scale=init_scale),
    )


def _rope_attention(
    dmodel,
    q_heads,
    kv_heads,
    seq_len,
    rope_base,
    rope_scale_freqs,
    init_scale,
    rope=True,
    qk_norm=False,
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
        rope=rope,
        qk_norm=qk_norm,
    )


# Every builder is called with the same keyword set, so a type that needs extra
# context (ame: vocab_size, its options block) does not change the call site. A
# builder keeps what it uses and drops the rest through this filter.
_ROPE_KEYS = ("seq_len", "rope_base", "rope_scale_freqs", "init_scale")


def _rope_kwargs(kwargs):
    return {key: kwargs[key] for key in _ROPE_KEYS if key in kwargs}


def _build_mha(dmodel, q_heads, kv_heads, *, qk_norm=None, **kwargs):
    """Multi-head attention: every query head has its own kv head."""
    return _rope_attention(
        dmodel,
        q_heads,
        q_heads,
        **_rope_kwargs(kwargs),
        qk_norm=bool(qk_norm),
    )


def _build_mqa(dmodel, q_heads, kv_heads, *, qk_norm=None, **kwargs):
    """Multi-query attention: one kv head shared by every query head."""
    return _rope_attention(
        dmodel,
        q_heads,
        1,
        **_rope_kwargs(kwargs),
        qk_norm=bool(qk_norm),
    )


def _build_gqa(dmodel, q_heads, kv_heads, *, options=None, qk_norm=None, **kwargs):
    """Grouped-query attention: the config's kv_heads (H/2 in the H family).

    `options` is the config's `attention.gqa` block. `rope` (default true)
    drops the rotary encoding when false: the NoPE dense baseline,
    `attention: {type: gqa, gqa: {rope: false}}`. `qk_norm` (default false)
    arrives already merged with the stack-wide switch by build_attention, so
    the argument is what this layer runs with, not the option.
    """
    if q_heads % kv_heads:
        raise ValueError(f"gqa needs q_heads % kv_heads == 0, got {q_heads}/{kv_heads}")
    options = dict(options or {})
    unknown = set(options) - set(_GQA_OPTIONS)
    if unknown:
        raise ValueError(
            f"unknown attention.gqa options {sorted(unknown)}; "
            f"expected {sorted(_GQA_OPTIONS)}"
        )
    return _rope_attention(
        dmodel,
        q_heads,
        kv_heads,
        **_rope_kwargs(kwargs),
        rope=bool(options.get("rope", True)),
        qk_norm=bool(qk_norm),
    )


def _build_ame(
    dmodel,
    q_heads,
    kv_heads,
    *,
    vocab_size=None,
    options=None,
    qk_norm=None,
    **kwargs,
):
    """AME: id-indexed Q/K banks (src.core.ame).

    MHA-shaped, so the config's `kv_heads` does not apply and is ignored, the
    same way `mha` ignores it. `vocab_size` is required -- the banks have one row
    per token type -- and `options` is the config's `attention.ame` block.
    `qk_norm` is the layer's merged setting (see build_attention); None leaves
    AMEAttention's own default, which is on.
    """
    if vocab_size is None:
        raise ValueError(
            "ame needs common.vocab_size: its Q/K banks are indexed by token id. "
            "Pass vocab_size to build_attention (the model config does this via "
            "`vocab_size: ${common.vocab_size}` under attention_fn)."
        )
    options = dict(options or {})
    unknown = set(options) - set(_AME_OPTIONS)
    if unknown:
        raise ValueError(
            f"unknown attention.ame options {sorted(unknown)}; "
            f"expected {sorted(_AME_OPTIONS)}"
        )
    # The merged value replaces the option: build_attention has already
    # checked that the two agree whenever both are set.
    options.pop("qk_norm", None)
    if qk_norm is not None:
        options["qk_norm"] = bool(qk_norm)
    return AMEAttention(
        dmodel,
        q_heads,
        kv_heads,
        vocab_size=vocab_size,
        **_rope_kwargs(kwargs),
        **options,
    )


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


def _planned(cls):
    def build(dmodel, q_heads, kv_heads, **kwargs):
        return cls(dmodel, q_heads, kv_heads, **_rope_kwargs(kwargs))

    return build


ATTENTION_TYPES = {
    "mha": _build_mha,
    "mqa": _build_mqa,
    "gqa": _build_gqa,
    "mla": _planned(MLAttention),
    "mamba2": _planned(Mamba2Mixer),
    "gated_deltanet": _planned(GatedDeltaNet),
    "ame": _build_ame,
}

# The options `attention.ame` accepts, i.e. AMEAttention's keyword-only knobs
# minus vocab_size (which comes from common). Named here so a typo in a sweep
# fails at build time with the valid set printed, instead of reaching the model.
_AME_OPTIONS = (
    "dhead",
    "qk_norm",
    "bank_init_std",
    "shift",
    "pin_alpha",
    "pin_beta",
    "sink",
    "min_horizon",
    "horizon_factor",
    "slopes",
    "learnable_slopes",
    "backend",
)

# The options `attention.gqa` accepts. `rope: false` is NoPE; `qk_norm: true`
# adds AME's QK-norm (see _build_gqa).
_GQA_OPTIONS = ("rope", "qk_norm")

# The stack-wide switches: `attention.<name>` applies to every layer of every
# real type, and must agree with any per-type option of the same name.
_SWITCHES = ("qk_norm",)

# Which types accept an options block, and which options. Checked instead of
# `ATTENTION_TYPES` so that a block for a type that has no options -- `{type:
# mha, mha: {...}}`, or one aimed at a still-planned type -- is an error rather
# than silently dropped on the floor by the builder's kwarg filter.
_TYPE_OPTIONS = {"ame": _AME_OPTIONS, "gqa": _GQA_OPTIONS}


_NTH_PATTERNS = {"every_2nd": 2, "every_4th": 4}


def _check_type(name):
    if name not in ATTENTION_TYPES:
        raise ValueError(
            f"unknown attention type {name!r}; registered: {sorted(ATTENTION_TYPES)}"
        )
    return name


def _layer_switch(name, attention, switch):
    """A switch (`qk_norm`) for a layer of type `name`: True, False,
    or None (the type's default).

    A stack-wide `attention.<switch>` applies to every layer, and every
    per-type option of the same name must agree with it -- checked across all
    blocks, not just this layer's type, so a contradiction fails the build
    even where the contradicting block happens to be unused. Without it, the
    type's own option, if any.
    """
    stack = attention.get(switch)
    if stack is not None:
        for type_name in _TYPE_OPTIONS:
            own = (attention.get(type_name) or {}).get(switch)
            if own is not None and bool(own) != bool(stack):
                raise ValueError(
                    f"attention.{switch}={stack} contradicts attention.{type_name}"
                    f".{switch}={own}: set {switch} in one place, or make them agree"
                )
        return bool(stack)
    own = (attention.get(name) or {}).get(switch)
    return None if own is None else bool(own)


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
    vocab_size=None,
    rope_base=10000,
    rope_scale_freqs=False,
    init_scale=1,
):
    """The `attention_fn` a TransformerBlock calls with its block id.

    `attention` is the config mapping: {"type": <name>} for a uniform stack, plus
    an optional "mix" (see resolve_layer_types) to swap the type on some layers,
    plus the optional stack-wide switch "qk_norm" (see _layer_switch), plus an
    optional options block per registered type name -- `{"type": "ame", "ame":
    {...}}` -- which applies wherever that type is built, base or mix.

    `vocab_size` is only consumed by types whose parameters are indexed by token
    id (ame); the projection-based types ignore it.
    """
    unknown = set(attention) - {"type", "mix", *_SWITCHES} - set(_TYPE_OPTIONS)
    if unknown:
        raise ValueError(
            f"unknown attention config keys {sorted(unknown)}; expected type, mix, "
            f"{', '.join(_SWITCHES)}, or an options block for a type that takes "
            f"options ({sorted(_TYPE_OPTIONS)})"
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
        vocab_size=vocab_size,
        options=attention.get(name),
        **{switch: _layer_switch(name, attention, switch) for switch in _SWITCHES},
    )
