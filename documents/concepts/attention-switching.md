# Attention switching

**Status: established** (stage 2 of the pretraining reset, 2026-10-30). This project
pretrains the same H-balanced backbone with different attention types, alone or mixed
within one model. The registry and pattern logic live in `src/core/attention.py`;
`tests/test_attention_registry.py` pins all of it.

## Naming a type

The model config carries an `attention` mapping in `common`:

```yaml
common:
  attention:
    type: gqa          # the whole stack, unless mixed below
```

Registered types and their state:

| name | state | what it is |
|---|---|---|
| `gqa` | real | grouped-query attention, `kv_heads` from the config (H/2 in the H family) |
| `mha` | real | one kv head per query head (`kv_heads = q_heads`) |
| `mqa` | real | one shared kv head (`kv_heads = 1`) |
| `mla` | interface | multi-head latent attention (DeepSeek-style kv compression) |
| `mamba2` | interface | Mamba-2 state-space mixer; brings its own scan kernel |
| `gated_deltanet` | interface | gated DeltaNet linear attention; chunked-scan kernel |
| `ame` | interface | **attention memory embedding — this project's method** |

`mha`/`mqa`/`gqa` are one `RoPEAttention` differing only in kv head count (`mha` is
bit-equal to `gqa` with `kv_heads = q_heads`). The interface types construct from a
config today — same constructor contract — and raise `NotImplementedError` at
forward, so configs naming them compose, instantiate and pattern-mix before the
implementations land.

## Mixing two types per layer

```yaml
common:
  attention:
    type: gqa
    mix:
      type: ame
      layers: every_4th   # every_2nd | every_4th | all | [3, 7, 11]
```

Convention: `every_Nth` selects the 0-based indices `i` with `(i + 1) % N == 0`
(layers 3, 7, 11, 15 for N=4 on H=16), so **layer 0 is never silently replaced**.
`all` turns the whole stack into `mix.type`; an explicit list is 0-based, and
out-of-range, duplicate or unknown patterns fail at build time with the offender
named.

## Adding a real implementation

1. Write the module with the shared constructor contract:
   `(dmodel, q_heads, kv_heads, seq_len, rope_base, rope_scale_freqs, init_scale)`.
   Input and output are `(batch, seq, dmodel)`.
2. Point the type's entry in `ATTENTION_TYPES` at its builder (for the current
   stubs: replace the `_PlannedAttention` subclass).
3. The registry tests pick the type up automatically; add type-specific tests for
   the math. New kernel dependencies go through the deliberate pixi lock re-solve
   (pixi ≤0.67 keeps the lock at v6 — see the reset log).
