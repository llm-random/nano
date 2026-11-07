# AME — the cost model

**Status:** the counting model behind [`../scripts/h_balanced_flops.ipynb`](../scripts/h_balanced_flops.ipynb),
2026-11-07. AME (Attention Memory Embedding, formerly MAE) is the attention method in
[`src/core/ame.py`](../../src/core/ame.py); what it is and why is in
[`../notes/ame_implementation.md`](../notes/ame_implementation.md) and in the design record
`~/code/research/ame/ame.md` (outside this repo). This doc covers only what parameter and FLOP
counts need. Until 2026-11-07 it was `sam-model.md`, after the method's earlier acronym; that version
predates AME's design and charged no attention cost at all, which the costs below replace.

A variant with the FF widened to the vanilla weight count, **AMEm**, is at the
[end of this doc](#amem--the-matched-variant).

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
| decode score path | no `QKᵀ`; `A@V` over the top-k selected positions; the query token's score row, computed from the banks unless cached; with the KV shift, every cached bigram scored for the top-k |

Per layer, the attention weights drop from `3·d²` to `2·d²` (`d = 64H`): GQA's
`q + k + v + o = d² + d²/2 + d²/2 + d²`, and AME keeps `v + o = d² + d²` because its `v_proj` is full
width. With the gated FF's `12·d²` that is `14·d²` per layer against the vanilla `15·d²`:

```
AME backbone, non-embedding = H · 14·d² + 133·H²  = 57344·H³ + 133·H²    (+ sinks and scalars)
vanilla non-embedding       = H · 15·d²           = 61440·H³
AME Q/K banks               = H · 2·V·64H         = 128·V·H²
```

These are the parameters nano trains: at `H = 12` and the padded vocab 50,304 they give
1,103,599,056 for AME, the count [stage_2_report.md](../notes/stage_2_report.md) reports for S2-PS,
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
`R = T[t_i, ·]`. What it pays (design §4.4; `ame_implementation.md`, "At decode"):

```
vanilla:  2·(61440·H³ + 64·V·H) + 256·H²·L
AME:      2·(57344·H³ + 64·V·H) + 128·H²·(min(L, k) + miss·U(L)) + 3·H²·L
```

- `min(L, k)`: the value sum over the top-k positions, `k = 2048`;
- `miss·U(L)`: computing the row from the banks - a `d_head` dot product per head for each of the
  `U(L)` distinct types in the context (`U(L) = 2.55·L^0.76`, capped at `L` and `V`: a fit to the
  design's Zipf estimates, reaching the vocab at ~450k tokens) - on the share `miss` of steps whose row
  is not cached, 1 by default because no row cache is built;
- `3·H²·L`: with the KV shift a key's score is `α₁R[t_j] + α₂R[t_{j−1}]`, so the top-k index is keyed by
  bigram and every cached position is scored, 3 FLOPs per head.

| H | 4k cache (AME / vanilla) | 32k cache | 1M cache |
|---:|---|---|---|
| 12 | 0.34 / 0.44 (−22.6%) | 0.45 / 1.50 (−69.7%) | 1.69 / 38.9 (−95.7%) |
| 16 | 0.69 / 0.87 (−21.2%) | 0.89 / 2.75 (−67.6%) | 3.09 / 69.3 (−95.5%) |
| 20 | 1.23 / 1.53 (−19.8%) | 1.54 / 4.47 (−65.5%) | 4.98 / 108.5 (−95.4%) |
| 24 | 2.00 / 2.46 (−18.5%) | 2.46 / 6.69 (−63.3%) | 7.41 / 156.5 (−95.3%) |
| 32 | 4.43 / 5.31 (−16.5%) | 5.24 / 12.8 (−59.2%) | 14.0 / 279.1 (−95.0%) |
| 40 | 8.33 / 9.80 (−15.0%) | 9.59 / 21.5 (−55.5%) | 23.4 / 437.6 (−94.7%) |
| 50 | 15.8 / 18.3 (−13.7%) | 17.8 / 36.7 (−51.5%) | 39.3 / 686.8 (−94.3%) |
| 64 | 32.3 / 36.9 (−12.4%) | 35.6 / 67.0 (−46.9%) | 70.8 / 1,132 (−93.7%) |
| 80 | 62.2 / 70.1 (−11.4%) | 67.2 / 117.1 (−42.6%) | 122.3 / 1,781 (−93.1%) |
| 100 | 119.9 / 134.0 (−10.5%) | 127.8 / 207.4 (−38.4%) | 213.8 / 2,808 (−92.4%) |
| 126 | 237.5 / 263.3 (−9.8%) | 250.0 / 379.8 (−34.2%) | 386.6 / 4,508 (−91.4%) |
| 158 | 464.8 / 511.9 (−9.2%) | 484.4 / 695.1 (−30.3%) | 699.2 / 7,187 (−90.3%) |
| 200 | 937.0 / 1,026 (−8.7%) | 968.5 / 1,320 (−26.6%) | 1,313 / 11,722 (−88.8%) |
| 252 | 1,866 / 2,035 (−8.3%) | 1,916 / 2,501 (−23.4%) | 2,462 / 19,015 (−87.1%) |

GFLOPs per generated token at the defaults. Three readings:

1. **At a short cache the win is the weights and a cheaper score path** - 8-23% at 4k.
2. **At a long cache the vanilla model pays for every key and AME for `k` of them**: 87-96% cheaper at
   1M, less for large models, whose weights (`H³`) are a bigger counterweight to the score path (`H²`).
3. **AME's own score cost is not small**: at `H = 16` and 1M it is 81% of the step, most of it the
   score row, which is why the design calls the row cache load-bearing. Against a vanilla model that
   runs a sparse method rather than dense attention the comparison is much closer - see
   [../notes/long-context-attention.md](../notes/long-context-attention.md) and the notebook's last two
   figures.

## The AME knobs

| knob | default | what it charges |
|---|---|---|
| `prefill_cost` | 1.0 | share of the dense score path a prefill pays; 1 is AME as designed |
| `top_k` | 2048 | positions whose values a decode step sums, per head (`None` = all) |
| `row_miss` | 1.0 | share of decode steps that compute the score row from the banks |
| `shift` | True | the KV shift's bigram selection, `3·H²·L` per decode step |
| `heaps` | (2.55, 0.76) | the distinct-type estimate `U(L) = a·L^b` |

At `H = 16`, GFLOPs per generated token:

| cache | no attention cost | rows cached, no shift | rows cached | 25% row misses | no row cache (default) | vanilla |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 0.573 | 0.640 | 0.643 | 0.655 | 0.690 | 0.875 |
| 32,768 | 0.573 | 0.640 | 0.665 | 0.722 | 0.891 | 2.754 |
| 262,144 | 0.573 | 0.640 | 0.841 | 1.115 | 1.938 | 17.786 |
| 1,048,576 | 0.573 | 0.640 | 1.445 | 1.857 | 3.094 | 69.326 |

"No attention cost" is every knob off - what this cost model assumed before AME was designed - and it
is flat at the weights. A row cache is worth most at long contexts, 3.09 against 1.45 GFLOPs at 1M;
once the row is cached the shift's selection is the only cost still growing, 0.81 GFLOPs of the 1.45;
and even the default stays 22x under the vanilla model at 1M.

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

### What AMEm costs

AMEm does the vanilla model's weight work per token, to the rounding, so its prefill *is* the vanilla
prefill - both are dense attention - and its decode step differs from the vanilla one only in
attention: AME's value sum, score row and selection against the vanilla `QKᵀ` and `A@V` over every
cached key.

| H | 4k cache (AMEm / vanilla) | 32k cache | 1M cache |
|---:|---|---|---|
| 12 | 0.36 / 0.44 (−19.4%) | 0.47 / 1.50 (−68.7%) | 1.71 / 38.9 (−95.6%) |
| 16 | 0.72 / 0.87 (−17.2%) | 0.93 / 2.75 (−66.4%) | 3.13 / 69.3 (−95.5%) |
| 20 | 1.29 / 1.53 (−15.6%) | 1.61 / 4.47 (−64.0%) | 5.05 / 108.5 (−95.3%) |
| 24 | 2.12 / 2.46 (−13.9%) | 2.57 / 6.69 (−61.6%) | 7.52 / 156.5 (−95.2%) |
| 32 | 4.70 / 5.31 (−11.5%) | 5.50 / 12.8 (−57.1%) | 14.3 / 279.1 (−94.9%) |
| 40 | 8.86 / 9.80 (−9.6%) | 10.1 / 21.5 (−53.0%) | 23.9 / 437.6 (−94.5%) |
| 50 | 16.8 / 18.3 (−8.1%) | 18.8 / 36.7 (−48.8%) | 40.3 / 686.8 (−94.1%) |
| 64 | 34.5 / 36.9 (−6.5%) | 37.7 / 67.0 (−43.7%) | 73.0 / 1,132 (−93.6%) |
| 80 | 66.3 / 70.1 (−5.4%) | 71.4 / 117.1 (−39.1%) | 126.4 / 1,781 (−92.9%) |
| 100 | 128.1 / 134.0 (−4.4%) | 136.0 / 207.4 (−34.4%) | 222.0 / 2,808 (−92.1%) |
| 126 | 253.9 / 263.3 (−3.6%) | 266.3 / 379.8 (−29.9%) | 402.9 / 4,508 (−91.1%) |
| 158 | 497.0 / 511.9 (−2.9%) | 516.6 / 695.1 (−25.7%) | 731.4 / 7,187 (−89.8%) |
| 200 | 1,002 / 1,026 (−2.3%) | 1,034 / 1,320 (−21.7%) | 1,378 / 11,722 (−88.2%) |
| 252 | 1,997 / 2,035 (−1.8%) | 2,047 / 2,501 (−18.1%) | 2,593 / 19,015 (−86.4%) |

GFLOPs per generated token at the defaults. AMEm sits a constant weight gap above AME of the same `H`
- the `d²` per layer it puts back - and below the vanilla model by exactly the difference between the
two attentions: 2-19% at 4k, 86-96% at 1M.

## Where the numbers come from

[`../scripts/h_balanced_flops.ipynb`](../scripts/h_balanced_flops.ipynb) - `inference_flops(...,
variant="ame")` and `variant="amem"`, with the four AME knobs; the tables above are generated from its
functions. The FLOP model (2 FLOPs per active parameter per token, exact causal pair counting, what is
excluded) is in its first cell. The vanilla counts here use the configs' vocab of 50,304;
[`h_balanced_params.py`](../scripts/h_balanced_params.py) and the H-balanced doc's table still use
50,257, so they differ in the last digit at small `H`.
