# H-balanced transformer

**Status:** established convention (llm-random lineage); written down here for the first time on
2026-10-20, from Maciej's definition. Attention is **always GQA** here (Maciej, 2026-10-20); the two
configs in the repo predate that and set `kv_heads = q_heads` (see
[Instances in this repo](#instances-in-this-repo)).

## Definition

An *H-balanced* transformer is a model whose whole architecture is derived from **one integer knob
`H`** (the "model size"), with `H` divisible by 2:

| Quantity | Rule | Note |
|---|---|---|
| number of layers | `n_blocks = H` | depth is tied to width |
| attention heads | `q_heads = H` | |
| kv heads | `kv_heads = H / 2` | GQA, always |
| residual width | `dmodel = 64 · H` | |
| head dim | `dmodel / q_heads = 64` | constant by construction |
| FF hidden | `dff = 4 · dmodel = 256 · H` | |
| attention inner width | `datt = dmodel` | the attention stack is as wide as the residual |
| tokenizer | GPT-NeoX, ≈50k vocab | len 50277 (max id 50276); `vocab_size: 50304` in configs, padded to a multiple of 128. An earlier revision said 50257, which is GPT-2's count — corrected 2026-10-30 after measuring. |

So `H` fixes depth, width, head count and FF width at once. Nothing is tuned per size: a model is
named by its `H`, and every dimension follows. `head_dim = 64` is a consequence of
`dmodel = 64·H` and `q_heads = H`, not a separate choice.

```
H → dmodel = 64H, dff = 256H, n_blocks = H, q_heads = H, kv_heads = H/2
```

## Why it is defined this way

- **One scaling axis.** Depth, width and head count move together, so a sweep over `H` is a clean
  scaling sweep: no "is this width-depth ratio fair?" confound between points.
- **Fixed head dim.** Every size has the same 64-wide heads, so attention kernels, RoPE behaviour
  and per-head statistics are comparable across sizes.

## Derived dimensions and parameter counts

Counting convention: **gated feed-forward** (`gate` + `up` + `down`, three `dmodel x dff`
matrices — SwiGLU-style, assumed for every model in this project), **untied** embedding and head
(`V·dmodel` each), norms without bias, `V = 50257`.

Closed form (`d = 64H`):

```
emb + head          = 2 · V · d
attention per layer = 3·d²            (q, o, and k+v at half width under GQA)
FF per layer        = 3·d·dff = 12·d²  (gated: gate + up + down)
non-embedding total = H · 15·d² = 61440·H³
```

| H | dmodel | layers | q_heads | kv_heads | dff | params | non-embedding |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 768 | 12 | 12 | 6 | 3072 | 183.4M | 106.2M |
| 16 | 1024 | 16 | 16 | 8 | 4096 | 354.6M | 251.7M |
| 20 | 1280 | 20 | 20 | 10 | 5120 | 620.2M | 491.6M |
| 24 | 1536 | 24 | 24 | 12 | 6144 | 1003.8M | 849.4M |
| 32 | 2048 | 32 | 32 | 16 | 8192 | 2219.3M | 2013.4M |
| 40 | 2560 | 40 | 40 | 20 | 10240 | 4189.7M | 3932.4M |
| 50 | 3200 | 50 | 50 | 25 | 12800 | 8002.0M | 7680.3M |
| 64 | 4096 | 64 | 64 | 32 | 16384 | 16.52B | 16.11B |
| 80 | 5120 | 80 | 80 | 40 | 20480 | 31.97B | 31.46B |
| 100 | 6400 | 100 | 100 | 50 | 25600 | 62.08B | 61.44B |
| 126 | 8064 | 126 | 126 | 63 | 32256 | 123.72B | 122.91B |
| 158 | 10112 | 158 | 158 | 79 | 40448 | 243.36B | 242.34B |
| 200 | 12800 | 200 | 200 | 100 | 51200 | 492.81B | 491.53B |
| 252 | 16128 | 252 | 252 | 126 | 64512 | 984.85B | 983.23B |

Totals include the embedding and the untied head. The rows are the sizes closest to doubling the
parameter count at each step, from 183M up to a 1T-class model (`H = 252`, 984.85B). Parameters grow as `H³` — doubling `H` is
~8× the model, so a doubling in size is roughly `H × 1.26`. Reproduce the table with `documents/scripts/h_balanced_params.py` (pass
`gated_ff=False` for the ungated variant).
Inference FLOPs, with the attention score path counted exactly, are in
[`../scripts/h_balanced_flops.ipynb`](../scripts/h_balanced_flops.ipynb). For the variant without
Q/K projections see [sam-model.md](sam-model.md).

## Instances in this repo

The family lives in [configs/_model/h_balanced/](../../configs/_model/h_balanced/):
`base.yaml` derives every dimension from `common.h` by interpolation, and an instance
file is three lines. Shipped sizes:

| Config | H | dmodel | layers | q_heads | kv_heads | dff | params |
|---|---:|---:|---:|---:|---:|---:|---:|
| [h4.yaml](../../configs/_model/h_balanced/h4.yaml) | 4 | 256 | 4 | 4 | 2 | 1024 | 29.7M |
| [h16.yaml](../../configs/_model/h_balanced/h16.yaml) | 16 | 1024 | 16 | 16 | 8 | 4096 | 354.6M |
| [h24.yaml](../../configs/_model/h_balanced/h24.yaml) | 24 | 1536 | 24 | 24 | 12 | 6144 | 1003.8M |

h16 and h24 are the def_300- and def_800-class sizes (the old `llmrandom/def_*` configs,
which predated the GQA and gated-FF rules, were removed with the PC project). Family
defaults, as shipped: gated FF (`GatedFeedForward`), `RMSNorm` (eps 1e-5, the stage-1
stack's norm — the llm-random instances used `LayerNorm`), `RoPEAttention` with
`rope_base: 10000`, `rope_scale_freqs: false`, untied head, `llm_random_weight_init`
scale 1. Attention is selected per layer through the registry — see
[attention-switching.md](attention-switching.md); the default is `gqa` everywhere.
`tests/test_h_balanced.py` pins the dimension rules and the exact parameter count.

## Defining a new one

Pick an even `H` and write a three-line instance file next to the others:

```yaml
defaults:
  - base
  - _self_

common:
  h: 32
```

Everything else is derived. (`tests/test_repo_hygiene.py` flags a size no experiment
references, so wire it into a run config or add it to the known list.)
