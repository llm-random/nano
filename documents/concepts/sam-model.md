# SAM — stateless memory attention

**Status:** in development (Maciej, 2026-10-20). This doc covers only what the cost model needs.

**What it is, for counting purposes:** the [H-balanced transformer](h-balanced-model.md) of the
same `H`, **without the Q and K projection weights** and **without any attention cost**. With no q
and k vectors there is no `QKᵀ`/`A@V` pair work to charge, so SAM's FLOPs are purely linear in
tokens. How its attention actually works is out of scope here and does not enter these numbers.

A knob is provided for the case where q and k vectors turn out to be produced some other way:
`attention_cost = c` charges `c ×` the vanilla score path (see
[The attention-cost knob](#the-attention-cost-knob)). Everything below uses `c = 0`.

A parameter-matched variant, **SAMm**, is at the [end of this doc](#samm--the-parameter-matched-variant).

## Definition

| Quantity | Rule |
|---|---|
| everything structural | identical to the H-balanced model of the same `H` |
| `q_proj` (`dmodel x datt`) | absent |
| `k_proj` (`dmodel x head_dim·kv_heads`) | absent |
| `v_proj`, `o_proj` | unchanged |
| FF, head, embedding, norms | unchanged |
| attention score path (`QKᵀ`, `A@V`) | not charged (`c = 0`) |

Per layer, attention weights drop from `3·d²` to `1.5·d²` (`d = 64H`; under GQA the k/v projections
are half width, so `q + k + v + o = d² + d²/2 + d²/2 + d² = 3d²`, and removing `q` and `k` leaves
`v + o = 1.5d²`). With the gated FF's `12·d²` that is `13.5·d²` per layer against the vanilla
`15·d²`:

```
SAM non-embedding params     = H · 13.5·d² = 55296·H³
vanilla non-embedding params = H · 15·d²   = 61440·H³
```

## Parameters, against the vanilla H model

Same counting convention as the H-balanced doc (GQA, gated FF, untied embedding and head,
`V = 50257`). Rows are the sizes closest to doubling the parameter count at each step, up to a
64B-class model.

| H | dmodel | layers | q_heads | kv_heads | dff | SAM params | vanilla H params | Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 768 | 12 | 12 | 6 | 3072 | 172.8M | 183.4M | −10.6M (−5.8%) |
| 16 | 1024 | 16 | 16 | 8 | 4096 | 329.5M | 354.6M | −25.2M (−7.1%) |
| 20 | 1280 | 20 | 20 | 10 | 5120 | 571.1M | 620.2M | −49.2M (−7.9%) |
| 24 | 1536 | 24 | 24 | 12 | 6144 | 918.9M | 1003.8M | −84.9M (−8.5%) |
| 32 | 2048 | 32 | 32 | 16 | 8192 | 2017.9M | 2219.3M | −201.3M (−9.1%) |
| 40 | 2560 | 40 | 40 | 20 | 10240 | 3796.5M | 4189.7M | −393.2M (−9.4%) |
| 50 | 3200 | 50 | 50 | 25 | 12800 | 7234.0M | 8002.0M | −768.0M (−9.6%) |
| 64 | 4096 | 64 | 64 | 32 | 16384 | 14.91B | 16.52B | −1.61B (−9.8%) |
| 80 | 5120 | 80 | 80 | 40 | 20480 | 28.83B | 31.97B | −3.15B (−9.8%) |
| 100 | 6400 | 100 | 100 | 50 | 25600 | 55.94B | 62.08B | −6.14B (−9.9%) |
| 126 | 8064 | 126 | 126 | 63 | 32256 | 111.43B | 123.72B | −12.29B (−9.9%) |
| 158 | 10112 | 158 | 158 | 79 | 40448 | 219.12B | 243.36B | −24.23B (−10.0%) |
| 200 | 12800 | 200 | 200 | 100 | 51200 | 443.66B | 492.81B | −49.15B (−10.0%) |
| 252 | 16128 | 252 | 252 | 126 | 64512 | 886.53B | 984.85B | −98.32B (−10.0%) |

The saving grows with `H` because the dropped weights scale as `H³` while the embedding and head,
which SAM keeps in full, scale as `H`. At `H = 12` the two models are within 6% of each other; at
`H = 50` the gap is 768M parameters, at `H = 100` it is 6.1B and at `H = 252` it is 98.3B, which is
the 10% asymptote to two decimals.

## Inference FLOPs, against the vanilla H model

2 FLOPs per active parameter per token, and nothing else — with `c = 0` there is no quadratic term,
so a SAM prefill costs exactly

```
2 · L · (55296·H³ + 64·V·H)      i.e. linear in the context
```

against the vanilla model's `2·L·(61440·H³ + 64·V·H) + 128·H²·L²`.

| H | 4k ctx (SAM / vanilla) | 32k ctx | 1M ctx |
|---:|---|---|---|
| 12 | 1.1 / 1.5 (−27%) | 8.8 / 29.3 (−70%) | 281 / 20,570 (−99%) |
| 16 | 2.3 / 3.0 (−25%) | 18.2 / 55.1 (−67%) | 583 / 36,665 (−98%) |
| 20 | 4.2 / 5.4 (−23%) | 33.2 / 91.4 (−64%) | 1,063 / 57,461 (−98%) |
| 24 | 6.9 / 8.8 (−22%) | 55.2 / 139.9 (−61%) | 1,765 / 83,008 (−98%) |
| 32 | 15.7 / 19.5 (−20%) | 125.5 / 279.4 (−55%) | 4,016 / 148,553 (−97%) |
| 40 | 30.0 / 36.7 (−18%) | 240.4 / 486.0 (−51%) | 7,692 / 233,696 (−97%) |
| 50 | 57.9 / 69.6 (−17%) | 463.5 / 857.5 (−46%) | 14,833 / 368,288 (−96%) |
| 64 | 120.4 / 142.4 (−15%) | 963.5 / 1,632.0 (−41%) | 30,831 / 610,670 (−95%) |
| 80 | 234.0 / 273.6 (−14%) | 1,872.3 / 2,958.1 (−37%) | 59,913 / 967,231 (−94%) |
| 100 | 455.6 / 527.4 (−14%) | 3,645.0 / 5,422.0 (−33%) | 116,639 / 1,536,900 (−92%) |
| 126 | 909.5 / 1,044 (−13%) | 7,276 / 10,263 (−29%) | 232,822 / 2,492,947 (−91%) |
| 158 | 1,791 / 2,043 (−12%) | 14,327 / 19,346 (−26%) | 458,464 / 4,022,661 (−89%) |
| 200 | 3,629 / 4,118 (−12%) | 29,033 / 37,752 (−23%) | 929,062 / 6,661,646 (−86%) |
| 252 | 7,256 / 8,198 (−11%) | 58,046 / 73,218 (−21%) | 1,857,475 / 11,001,074 (−83%) |

TFLOPs per prefill. The vanilla column charges **every** causal pair, which no deployed
long-context model does; against a vanilla model running one of the published sparse-attention
schemes the gap is far smaller, and sometimes reversed - see
[../notes/long-context-attention.md](../notes/long-context-attention.md). Two readings:

1. **At short contexts the win is just the dropped weights** — 14–27%, bigger for small `H` because
   the `H³` attention projections are a larger share of a small model than the `H`-scaling
   embedding and head.
2. **At long contexts the win is the whole quadratic term** — 92–99%, and it is *smaller* for large
   models, because their weight work (`H³`) is a bigger counterweight to the score path (`H²`).
   The vanilla model has no equivalent of this: past `L* = 960H + V/H` it is mostly paying for
   attention, and SAM pays none of it.

## The attention-cost knob

If q and k vectors do get produced somehow, set `attention_cost = c` (a fraction of the vanilla
score path, `0` by default) and the term comes back scaled:

```
SAM score FLOPs = c · H · 4 · datt · L(L+1)/2
crossover       = L*(c) = (864·H + V/H) / c        (no crossover at c = 0)
```

At `H = 16`, TFLOPs per prefill:

| context | c = 0 | c = 10% | c = 25% | c = 50% | c = 100% | vanilla |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 2.3 | 2.3 | 2.4 | 2.6 | 2.8 | 3.0 |
| 32,768 | 18.2 | 21.7 | 27.0 | 35.8 | 53.4 | 55.1 |
| 262,144 | 145.7 | 370.9 | 708.7 | 1,271.6 | 2,397.5 | 2,410.7 |
| 1,048,576 | 582.9 | 4,185.8 | 9,590.1 | 18,597.3 | 36,611.7 | 36,664.5 |

`c` buys context, not throughput: at 4k it barely registers, while at 1M every 10% of the score
path costs about 3,600 TFLOPs — more than six times SAM's entire `c = 0` pass. The crossover moves
in proportion: at `H = 16`, attention owns half the pass past 170k tokens at `c = 10%`, past 68k at
25%, past 17k at full cost.

## SAMm — the parameter-matched variant

`m` is for *matched*: SAM with `dff` raised until the model has the same parameter count as the
vanilla H-balanced model of the same `H`. The parameters freed by dropping `q_proj` and `k_proj`
are spent on a wider feed-forward instead of being given back.

Per layer, the freed weights are `1.5·d²` and each unit of `dff` costs `3·d` (gate + up + down),
so:

```
3·d·Δdff = 1.5·d²   ->   Δdff = d/2 = 32·H
dff(SAMm) = 256·H + 32·H = 288·H = 4.5 · dmodel
```

`288·H = 32 · 9H`, so `dff` is divisible by 32 at every `H` with no rounding, and the match is
exact rather than approximate — the two models agree to the parameter, as the Δ column shows.

| H | dmodel | layers | q_heads | kv_heads | SAMm dff | vanilla dff | SAMm params | vanilla H params | Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 768 | 12 | 12 | 6 | 3456 | 3072 | 183.4M | 183.4M | 0 |
| 16 | 1024 | 16 | 16 | 8 | 4608 | 4096 | 354.6M | 354.6M | 0 |
| 20 | 1280 | 20 | 20 | 10 | 5760 | 5120 | 620.2M | 620.2M | 0 |
| 24 | 1536 | 24 | 24 | 12 | 6912 | 6144 | 1003.8M | 1003.8M | 0 |
| 32 | 2048 | 32 | 32 | 16 | 9216 | 8192 | 2219.3M | 2219.3M | 0 |
| 40 | 2560 | 40 | 40 | 20 | 11520 | 10240 | 4189.7M | 4189.7M | 0 |
| 50 | 3200 | 50 | 50 | 25 | 14400 | 12800 | 8002.0M | 8002.0M | 0 |
| 64 | 4096 | 64 | 64 | 32 | 18432 | 16384 | 16.52B | 16.52B | 0 |
| 80 | 5120 | 80 | 80 | 40 | 23040 | 20480 | 31.97B | 31.97B | 0 |
| 100 | 6400 | 100 | 100 | 50 | 28800 | 25600 | 62.08B | 62.08B | 0 |
| 126 | 8064 | 126 | 126 | 63 | 36288 | 32256 | 123.72B | 123.72B | 0 |
| 158 | 10112 | 158 | 158 | 79 | 45504 | 40448 | 243.36B | 243.36B | 0 |
| 200 | 12800 | 200 | 200 | 100 | 57600 | 51200 | 492.81B | 492.81B | 0 |
| 252 | 16128 | 252 | 252 | 126 | 72576 | 64512 | 984.85B | 984.85B | 0 |

The rule is one constant: `dff = 288·H`, i.e. `4.5 · dmodel` instead of `4 · dmodel`. It does not
depend on the vocab, because the embedding and head are untouched, and it holds at every size.

### What SAMm costs

Same parameters as the vanilla model, so the same weight work per token — but still no score path
at `c = 0`, so its whole advantage is the quadratic term:

| H | 4k ctx (SAMm / vanilla) | 32k ctx | 1M ctx |
|---:|---|---|---|
| 12 | 1.2 / 1.5 (−21%) | 9.5 / 29.3 (−68%) | 304 / 20,570 (−99%) |
| 16 | 2.5 / 3.0 (−18%) | 19.9 / 55.1 (−64%) | 636 / 36,665 (−98%) |
| 20 | 4.6 / 5.4 (−16%) | 36.4 / 91.4 (−60%) | 1,166 / 57,461 (−98%) |
| 24 | 7.6 / 8.8 (−14%) | 60.7 / 139.9 (−57%) | 1,943 / 83,008 (−98%) |
| 32 | 17.3 / 19.5 (−11%) | 138.7 / 279.4 (−50%) | 4,438 / 148,553 (−97%) |
| 40 | 33.3 / 36.7 (−9%) | 266.1 / 486.0 (−45%) | 8,516 / 233,696 (−96%) |
| 50 | 64.2 / 69.6 (−8%) | 513.9 / 857.5 (−40%) | 16,443 / 368,288 (−96%) |
| 64 | 133.6 / 142.4 (−6%) | 1,069.0 / 1,632.0 (−34%) | 34,209 / 610,670 (−94%) |
| 80 | 259.8 / 273.6 (−5%) | 2,078.4 / 2,958.1 (−30%) | 66,510 / 967,231 (−93%) |
| 100 | 506.0 / 527.4 (−4%) | 4,047.6 / 5,422.0 (−25%) | 129,524 / 1,536,900 (−92%) |
| 126 | 1,010 / 1,044 (−3%) | 8,081 / 10,263 (−21%) | 258,596 / 2,492,947 (−90%) |
| 158 | 1,989 / 2,043 (−3%) | 15,915 / 19,346 (−18%) | 509,287 / 4,022,661 (−87%) |
| 200 | 4,032 / 4,118 (−2%) | 32,254 / 37,752 (−15%) | 1,032,141 / 6,661,646 (−85%) |
| 252 | 8,061 / 8,198 (−2%) | 64,490 / 73,218 (−12%) | 2,063,672 / 11,001,074 (−81%) |

Because the parameter counts match exactly, SAMm's weight work per token is the vanilla model's,
and the ratio between them is therefore just the part of a vanilla prefill that is *not* the score
path: `SAMm / vanilla = 1 − attention_share`, which is exactly **50% at the crossover**
`L* = 960H + V/H` (18,501 tokens at `H = 16`, 49,005 at `H = 50`). SAM's curve is the same one
scaled by its parameter ratio — 91.7% of SAMm at `H = 16`, 90.2% at `H = 50`, tending to 90%.

So the three families are one trade: SAM keeps the vanilla shape and returns ~10% of the
parameters; SAMm keeps the parameters and moves that 10% from the attention projections into the
FF; both skip the quadratic term entirely, which is what dominates past a few tens of thousands of
tokens. Per-layer weight split:

| | attention projections | gated FF | per-layer weights |
|---|---:|---:|---:|
| vanilla H | 3.0·d² (20%) | 12.0·d² (80%) | 15.0·d² |
| SAM | 1.5·d² (11%) | 12.0·d² (89%) | 13.5·d² |
| SAMm | 1.5·d² (10%) | 13.5·d² (90%) | 15.0·d² |

Any quality difference between them is an empirical question these numbers cannot answer.

## Where the numbers come from

[`../scripts/h_balanced_flops.ipynb`](../scripts/h_balanced_flops.ipynb) — `inference_flops(...,
variant="sam")` plots SAM and the vanilla H model together over the same `H` values, and
`variant="samm"` joins them in the last two figures: the three families on one axis, and the same
comparison as a share of the vanilla cost. The FLOP model (2 FLOPs per active parameter per token, exact causal pair counting for
the vanilla model, what is excluded) is in the first cell of that notebook; parameter counts come
from [`h_balanced_params.py`](../scripts/h_balanced_params.py).
