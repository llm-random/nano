# AME — the cost model

**Status:** the counting model behind [`../scripts/h_balanced_flops.ipynb`](../scripts/h_balanced_flops.ipynb),
2026-11-07; since 2026-11-13 a decode step reads its score row from a precomputed score table rather
than computing it (see [One generated token](#one-generated-token)). AME (Attention Memory
Embedding, formerly MAE) is the attention method in [`src/core/ame.py`](../../src/core/ame.py); what
it is and why is in
[`../notes/ame_implementation.md`](../notes/ame_implementation.md) and in the design record
`~/code/research/ame/ame.md` (outside this repo). This doc covers only what parameter and FLOP
counts need. Until 2026-11-07 it was `sam-model.md`, after the method's earlier acronym; that version
predates AME's design and charged no attention cost at all, which the costs below replace.

A variant with the FF widened to the vanilla weight count, **AMEm**, is at the
[end of this doc](#amem--the-matched-variant). The standard AME configs use it since 2026-11-27.

## Definition

| Quantity | Rule |
|---|---|
| everything structural | identical to the [H-balanced model](h-balanced-model.md) of the same `H` |
| `q_proj`, `k_proj` | absent: queries and keys are rows of two banks, `E_Q` and `E_K`, gathered by token id |
| Q/K banks | `V x datt` each per layer, so `2·V·datt` per layer and `128·V·H²` in all; gathers, no FLOPs |
| `v_proj` | **full width**, `dmodel x datt`: AME is MHA-shaped, every query head has its own value head (`kv_heads` is ignored) |
| `o_proj`, FF, head, embedding, norms | unchanged |
| sinks and per-head scalars | a key and a value sink of `datt` each, α (2), β (2) and τ (1) per head: `2·datt + 5·H` per layer, no FLOPs |
| prefill score path | ordinary causal attention over the gathered rows: the vanilla `QKᵀ` + `A@V` pair work |
| decode score path | no `QKᵀ`; `A@V` over the top-k selected positions; the query token's score row, read from a precomputed `V x V` table per head and layer (computing it from the banks instead is a knob); with the KV shift, every cached bigram scored for the top-k |

Per layer, the attention weights drop from `3·d²` to `2·d²` (`d = 64H`): GQA's
`q + k + v + o = d² + d²/2 + d²/2 + d²`, and AME keeps `v + o = d² + d²` because its `v_proj` is full
width. With the gated FF's `12·d²` that is `14·d²` per layer against the vanilla `15·d²`:

```
AME backbone, non-embedding = H · 14·d² + 133·H²  = 57344·H³ + 133·H²    (+ sinks and scalars)
vanilla non-embedding       = H · 15·d²           = 61440·H³
AME Q/K banks               = H · 2·V·64H         = 128·V·H²
```

These are the parameters nano trains: at `H = 12` and the padded vocab 50,304 they give
1,103,599,056 for AME, the count [stage_2_report.md](../notes/archive/stage_2_report.md) reports for S2-PS,
and the notebook asserts it.

## Parameters, against the vanilla H model

Same counting convention as the H-balanced doc (GQA for the vanilla model, gated FF, untied embedding
and head), at the configs' `V = 50304`. Rows are the sizes closest to doubling the parameter count at
each step, up to a 1T-class model. "Backbone" is everything but the banks.

| H | dmodel | layers | heads | AME backbone | Q/K banks | AME total | vanilla H | backbone Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 768 | 12 | 12 | 176.4M | 927.2M | 1103.6M | 183.5M | −7.1M (−3.8%) |
| 16 | 1024 | 16 | 16 | 338.0M | 1648.4M | 1986.3M | 354.7M | −16.7M (−4.7%) |
| 20 | 1280 | 20 | 20 | 587.6M | 2575.6M | 3163.2M | 620.4M | −32.7M (−5.3%) |
| 24 | 1536 | 24 | 24 | 947.4M | 3708.8M | 4656.2M | 1004.0M | −56.5M (−5.6%) |
| 32 | 2048 | 32 | 32 | 2085.4M | 6593.4M | 8678.8M | 2219.4M | −134.1M (−6.0%) |
| 40 | 2560 | 40 | 40 | 3928.0M | 10.30B | 14.23B | 4189.9M | −261.9M (−6.3%) |
| 50 | 3200 | 50 | 50 | 7490.6M | 16.10B | 23.59B | 8002.3M | −511.7M (−6.4%) |
| 64 | 4096 | 64 | 64 | 15.45B | 26.37B | 41.82B | 16.52B | −1073.2M (−6.5%) |
| 80 | 5120 | 80 | 80 | 29.88B | 41.21B | 71.09B | 31.97B | −2096.3M (−6.6%) |
| 100 | 6400 | 100 | 100 | 57.99B | 64.39B | 122.38B | 62.09B | −4094.7M (−6.6%) |
| 126 | 8064 | 126 | 126 | 115.53B | 102.22B | 217.75B | 123.72B | −8191.4M (−6.6%) |
| 158 | 10112 | 158 | 158 | 227.21B | 160.74B | 387.95B | 243.36B | −16.15B (−6.6%) |
| 200 | 12800 | 200 | 200 | 460.05B | 257.56B | 717.61B | 492.81B | −32.76B (−6.6%) |
| 252 | 16128 | 252 | 252 | 919.32B | 408.90B | 1328.21B | 984.86B | −65.54B (−6.7%) |

The backbone gap grows with `H` because the dropped weights scale as `H³` while the embedding and
head, which AME keeps in full, scale as `H`; it tends to `1/15 = 6.7%`. The banks run the other way:
they scale as `H²`, so they are 5.3x the backbone at `H = 12`, 1.1x at `H = 100` and 0.44x at
`H = 252`. Any comparison has to say which count it matches; `ame_experiments.md` matches tokens and
reports both.

## Inference FLOPs, against the vanilla H model

2 FLOPs per active parameter per token, the score path counted exactly. The banks and extras cost
nothing; what the score path costs depends on the pass.

### A prefill: dense attention

An AME prefill is ordinary causal attention, so it pays the vanilla score path and saves only the
deleted `W_Q`/`W_K` matmuls, net of the wider `v_proj`:

```
AME prefill = 2·L·(57344·H³ + 64·V·H) + 128·H²·L²      vanilla: 2·L·(61440·H³ + 64·V·H) + 128·H²·L²
```

| H | 4k ctx (AME / vanilla) | 32k ctx | 1M ctx |
|---:|---|---|---|
| 12 | 1.44 / 1.50 (−3.9%) | 28.8 / 29.3 (−1.6%) | 20,555 / 20,570 (−0.1%) |
| 16 | 2.90 / 3.03 (−4.5%) | 54.0 / 55.1 (−2.0%) | 36,629 / 36,665 (−0.1%) |
| 20 | 5.14 / 5.41 (−5.0%) | 89.3 / 91.4 (−2.3%) | 57,392 / 57,461 (−0.1%) |
| 24 | 8.36 / 8.83 (−5.3%) | 136.2 / 139.9 (−2.7%) | 82,889 / 83,008 (−0.1%) |
| 32 | 18.4 / 19.5 (−5.6%) | 270.6 / 279.4 (−3.1%) | 148,272 / 148,554 (−0.2%) |
| 40 | 34.6 / 36.7 (−5.9%) | 468.9 / 486.0 (−3.5%) | 233,147 / 233,697 (−0.2%) |
| 50 | 65.4 / 69.6 (−6.0%) | 823.9 / 857.5 (−3.9%) | 367,214 / 368,288 (−0.3%) |
| 64 | 133.6 / 142.4 (−6.2%) | 1,562 / 1,632 (−4.3%) | 608,419 / 610,670 (−0.4%) |
| 80 | 256.4 / 273.6 (−6.3%) | 2,821 / 2,958 (−4.6%) | 962,834 / 967,232 (−0.5%) |
| 100 | 493.9 / 527.4 (−6.4%) | 5,154 / 5,422 (−5.0%) | 1,528,310 / 1,536,900 (−0.6%) |
| 126 | 977.1 / 1,044 (−6.4%) | 9,726 / 10,263 (−5.2%) | 2,475,765 / 2,492,948 (−0.7%) |
| 158 | 1,911 / 2,043 (−6.5%) | 18,288 / 19,346 (−5.5%) | 3,988,780 / 4,022,662 (−0.8%) |
| 200 | 3,849 / 4,118 (−6.5%) | 35,605 / 37,752 (−5.7%) | 6,592,928 / 6,661,647 (−1.0%) |
| 252 | 7,661 / 8,198 (−6.6%) | 68,922 / 73,218 (−5.9%) | 10,863,610 / 11,001,075 (−1.2%) |

TFLOPs per prefill. The saving is the weights alone - 3.9-6.6% at 4k - and it fades as the shared
score path takes over the pass: 0.1-1.2% at 1M.

### One generated token

A decode step has no `QKᵀ`: the score of key `j` is an entry of the query token's score row
`R = T[t_i, ·]`, its row of `T = Ê_Q·Ê_Kᵀ`, the table of every query type against every key type -
one `V x V` table per head and layer. The counts here take `T` as **precomputed**, so reading a row
costs no FLOPs (since 2026-11-13; until then every row was computed from the banks). What a step
pays (design §4.4; `ame_implementation.md`, "At decode"):

```
vanilla:  2·(61440·H³ + 64·V·H) + 256·H²·L
AME:      2·(57344·H³ + 64·V·H) + 128·H²·(min(L, k) + miss·U(L)) + 3·H²·L
```

- `min(L, k)`: the value sum over the top-k positions, `k = 2048`;
- `miss·U(L)`: computing the row from the banks instead - a `d_head` dot product per head for each of
  the `U(L)` distinct types in the context (`U(L) = 2.55·L^0.76`, capped at `L` and `V`: a fit to the
  design's Zipf estimates, reaching the vocab at ~450k tokens) - on the share `miss` of steps that do.
  **0 by default**, since the table is precomputed. Computing is the design record's own plan (§2.1):
  the full table is `V²` entries per head and layer, 5.06 GB in bf16 and `5.06·H²` GB in all (729 GB
  at `H = 12`, 50.6 TB at `H = 100`), so it stores only the banks and caches the rows of frequent
  query tokens;
- `3·H²·L`: with the KV shift a key's score is `α₁R[t_j] + α₂R[t_{j−1}]`, so the top-k index is keyed by
  bigram and every cached position is scored, 3 FLOPs per head.

| H | 4k cache (AME / vanilla) | 32k cache | 1M cache |
|---:|---|---|---|
| 12 | 0.31 / 0.44 (−28.5%) | 0.33 / 1.50 (−78.1%) | 0.77 / 38.9 (−98.0%) |
| 16 | 0.64 / 0.87 (−26.5%) | 0.67 / 2.75 (−75.8%) | 1.45 / 69.3 (−97.9%) |
| 20 | 1.16 / 1.53 (−24.5%) | 1.19 / 4.47 (−73.4%) | 2.41 / 108.5 (−97.8%) |
| 24 | 1.90 / 2.46 (−22.8%) | 1.95 / 6.69 (−70.9%) | 3.70 / 156.5 (−97.6%) |
| 32 | 4.25 / 5.31 (−20.0%) | 4.33 / 12.8 (−66.2%) | 7.45 / 279.1 (−97.3%) |
| 40 | 8.04 / 9.80 (−18.0%) | 8.17 / 21.5 (−62.1%) | 13.1 / 437.6 (−97.0%) |
| 50 | 15.3 / 18.3 (−16.2%) | 15.6 / 36.7 (−57.6%) | 23.2 / 686.8 (−96.6%) |
| 64 | 31.6 / 36.9 (−14.4%) | 32.0 / 67.0 (−52.3%) | 44.4 / 1,132 (−96.1%) |
| 80 | 61.0 / 70.1 (−13.0%) | 61.5 / 117.1 (−47.5%) | 81.0 / 1,781 (−95.5%) |
| 100 | 118.1 / 134.0 (−11.9%) | 118.9 / 207.4 (−42.7%) | 149.4 / 2,808 (−94.7%) |
| 126 | 234.6 / 263.3 (−10.9%) | 236.0 / 379.8 (−37.9%) | 284.3 / 4,508 (−93.7%) |
| 158 | 460.2 / 511.9 (−10.1%) | 462.4 / 695.1 (−33.5%) | 538.5 / 7,187 (−92.5%) |
| 200 | 929.8 / 1,026 (−9.4%) | 933.2 / 1,320 (−29.3%) | 1,055 / 11,722 (−91.0%) |
| 252 | 1,854 / 2,035 (−8.9%) | 1,860 / 2,501 (−25.6%) | 2,053 / 19,015 (−89.2%) |

GFLOPs per generated token at the defaults. Three readings:

1. **At a short cache the win is the weights and a cheaper score path** - 8.9-28.5% at 4k.
2. **At a long cache the vanilla model pays for every key and AME for `k` of them**: 89.2-98.0%
   cheaper at 1M, less for large models, whose weights (`H³`) are a bigger counterweight to the score
   path (`H²`).
3. **AME's own score cost is not small, even with the table**: at `H = 16` and 1M it is 60% of the
   step, 56 points of it the bigram selection; computing every row as well would take the step from
   1.45 to 3.09 GFLOPs, 53% of it the row. Against a vanilla model that runs a sparse method rather
   than dense attention the comparison is much closer - see
   [../notes/long-context-attention.md](../notes/long-context-attention.md) and the notebook's last two
   figures.

## The AME knobs

| knob | default | what it charges |
|---|---|---|
| `prefill_cost` | 1.0 | share of the dense score path a prefill pays; 1 is AME as designed |
| `top_k` | 2048 | positions whose values a decode step sums, per head (`None` = all) |
| `row_miss` | 0.0 | share of decode steps that compute the score row from the banks instead of reading the precomputed table |
| `shift` | True | the KV shift's bigram selection, `3·H²·L` per decode step |
| `heaps` | (2.55, 0.76) | the distinct-type estimate `U(L) = a·L^b`, used when rows are computed |

At `H = 16`, GFLOPs per generated token:

| cache | no attention cost | table, no shift | table (default) | 25% of rows computed | every row computed | vanilla |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 0.573 | 0.640 | 0.643 | 0.655 | 0.690 | 0.875 |
| 32,768 | 0.573 | 0.640 | 0.665 | 0.722 | 0.891 | 2.754 |
| 262,144 | 0.573 | 0.640 | 0.841 | 1.115 | 1.938 | 17.786 |
| 1,048,576 | 0.573 | 0.640 | 1.445 | 1.857 | 3.094 | 69.326 |

"No attention cost" is every knob off - what this cost model assumed before AME was designed - and it
is flat at the weights. The table is worth most at long contexts: 1.45 GFLOPs at 1M, against 3.09
if every row is computed. With it the shift's selection is the only cost still growing, 0.81 GFLOPs
of the 1.45, and the default is 48x under the vanilla model at 1M (22x if every row is computed).

## AMEm — the matched variant

`m` is for *matched*: AME with `dff` raised until everything except the banks has the vanilla
parameter count of the same `H`. Dropping `q_proj` and `k_proj` frees `1.5·d²` per layer and the
full-width `v_proj` takes `0.5·d²` of it back, which leaves `d²`; each unit of `dff` costs `3·d`
(gate + up + down), so:

```
3·d·Δdff = d²   ->   Δdff = d/3 = 64H/3
dff(AMEm) = 256·H + 64·H/3 ≈ 277.3·H ≈ 4.33·dmodel,   rounded to a multiple of 32
```

`64H/3` is a multiple of 32 only when 3 divides `H`, so the match is exact at `H = 12, 24, 126, 252`
among these rows and within 0.15% elsewhere; with an ungated FF (`2·d` per unit of `dff`)
`Δdff = d/2 = 32·H` and it is exact at every `H`. The banks are left out of the match on purpose: they
are 0.4-5.3x the backbone, so no `dff` could bring AME's *total* to the vanilla one, and they cost no
FLOPs, so matching the rest matches the weight work per token.

| H | dmodel | AMEm dff | dff / dmodel | AMEm, without banks | vanilla H | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 768 | 3328 | 4.3333 | 183.5M | 183.5M | 0 |
| 16 | 1024 | 4448 | 4.3438 | 355.2M | 354.7M | +0.5M (+0.15%) |
| 20 | 1280 | 5536 | 4.3250 | 619.5M | 620.4M | −0.8M (−0.13%) |
| 24 | 1536 | 6656 | 4.3333 | 1004.0M | 1004.0M | 0 |
| 32 | 2048 | 8864 | 4.3281 | 2217.3M | 2219.4M | −2.1M (−0.09%) |
| 40 | 2560 | 11104 | 4.3375 | 4193.2M | 4189.9M | +3.3M (+0.08%) |
| 50 | 3200 | 13856 | 4.3300 | 7997.1M | 8002.3M | −5.1M (−0.06%) |
| 64 | 4096 | 17760 | 4.3359 | 16.53B | 16.52B | +8.4M (+0.05%) |
| 80 | 5120 | 22176 | 4.3312 | 31.96B | 31.97B | −13.1M (−0.04%) |
| 100 | 6400 | 27744 | 4.3350 | 62.11B | 62.09B | +20.5M (+0.03%) |
| 126 | 8064 | 34944 | 4.3333 | 123.72B | 123.72B | 0 |
| 158 | 10112 | 43808 | 4.3323 | 243.31B | 243.36B | −51.1M (−0.02%) |
| 200 | 12800 | 55456 | 4.3325 | 492.73B | 492.81B | −81.9M (−0.02%) |
| 252 | 16128 | 69888 | 4.3333 | 984.86B | 984.86B | 0 |

**The standard AME configs are AMEm since 2026-11-27** (recipe v4,
[`configs/standard/`](../../configs/standard/)): `common.dff: ${eval:'32 * round(26 * ${common.h} / 3)'}`
is the rule above, and the dense configs keep `256·H`. Built from the configs, with each bank
counted at the one row a token reads and dense's QK-norm and shift scalars included, AME's active
parameters match dense's to +0.020% at h12, +0.166% at h16 and −0.115% at h20, and its training
FLOPs per token to +0.010%, +0.150% and −0.116%.

### What AMEm costs

AMEm does the vanilla model's weight work per token, to the rounding, so its prefill *is* the vanilla
prefill - both are dense attention - and its decode step differs from the vanilla one only in
attention: AME's value sum and selection, its score rows read from the table, against the vanilla
`QKᵀ` and `A@V` over every cached key.

| H | 4k cache (AMEm / vanilla) | 32k cache | 1M cache |
|---:|---|---|---|
| 12 | 0.33 / 0.44 (−25.3%) | 0.34 / 1.50 (−77.2%) | 0.78 / 38.9 (−98.0%) |
| 16 | 0.68 / 0.87 (−22.5%) | 0.70 / 2.75 (−74.6%) | 1.48 / 69.3 (−97.9%) |
| 20 | 1.22 / 1.53 (−20.3%) | 1.25 / 4.47 (−71.9%) | 2.47 / 108.5 (−97.7%) |
| 24 | 2.01 / 2.46 (−18.1%) | 2.06 / 6.69 (−69.2%) | 3.82 / 156.5 (−97.6%) |
| 32 | 4.51 / 5.31 (−15.0%) | 4.60 / 12.8 (−64.1%) | 7.72 / 279.1 (−97.2%) |
| 40 | 8.57 / 9.80 (−12.6%) | 8.71 / 21.5 (−59.6%) | 13.6 / 437.6 (−96.9%) |
| 50 | 16.4 / 18.3 (−10.6%) | 16.6 / 36.7 (−54.8%) | 24.2 / 686.8 (−96.5%) |
| 64 | 33.8 / 36.9 (−8.5%) | 34.1 / 67.0 (−49.1%) | 46.6 / 1,132 (−95.9%) |
| 80 | 65.2 / 70.1 (−7.1%) | 65.7 / 117.1 (−43.9%) | 85.2 / 1,781 (−95.2%) |
| 100 | 126.3 / 134.0 (−5.7%) | 127.2 / 207.4 (−38.7%) | 157.6 / 2,808 (−94.4%) |
| 126 | 251.0 / 263.3 (−4.7%) | 252.3 / 379.8 (−33.6%) | 300.7 / 4,508 (−93.3%) |
| 158 | 492.4 / 511.9 (−3.8%) | 494.6 / 695.1 (−28.8%) | 570.7 / 7,187 (−92.1%) |
| 200 | 995.1 / 1,026 (−3.0%) | 998.6 / 1,320 (−24.3%) | 1,120 / 11,722 (−90.4%) |
| 252 | 1,985 / 2,035 (−2.4%) | 1,991 / 2,501 (−20.4%) | 2,184 / 19,015 (−88.5%) |

GFLOPs per generated token at the defaults. AMEm sits a constant weight gap above AME of the same `H`
- the `d²` per layer it puts back - and below the vanilla model by exactly the difference between the
two attentions: 2.4-25.3% at 4k, 88.5-98.0% at 1M.

## Where the numbers come from

[`../scripts/h_balanced_flops.ipynb`](../scripts/h_balanced_flops.ipynb) - `inference_flops(...,
variant="ame")` and `variant="amem"`, with the four AME knobs; the tables above are generated from its
functions. The FLOP model (2 FLOPs per active parameter per token, exact causal pair counting, what is
excluded) is in its first cell. The vanilla counts here use the configs' vocab of 50,304;
[`h_balanced_params.py`](../scripts/h_balanced_params.py) and the H-balanced doc's table still use
50,257, so they differ in the last digit at small `H`.
