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
| tokenizer | GPT-NeoX, ≈50k vocab | `vocab_size: 50257` in this repo's configs |

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

| Config | H | dmodel | layers | heads | dff | vocab |
|---|---:|---:|---:|---:|---:|---:|
| [configs/_model/llmrandom/def_300.yaml](../../configs/_model/llmrandom/def_300.yaml) | 16 | 1024 | 16 | 16 | 4096 | 50257 |
| [configs/_model/llmrandom/def_800.yaml](../../configs/_model/llmrandom/def_800.yaml) | 24 | 1536 | 24 | 24 | 6144 | 50257 |

Two caveats on those two, both predating the current convention: they set `kv_heads = q_heads`
(no GQA) and they instantiate `FeedForward`, the ungated two-matrix
FF — which is where the `def_NNN` names come from (304.3M and 833.9M as written, against 354.6M and
1003.8M for the canonical GQA + gated model). Everything else matches the rule. Quote the table
above for new models, and recompute if you reuse `def_300`/`def_800` unchanged.

Their other settings (shared by both, and the defaults to copy for a new size):

- `src.core.model.LayerNorm`, `bias: false`, eps 1e-5 (the Llama configs use `RMSNorm` instead)
- `RoPEAttention`, `rope_base: 10000`, `rope_scale_freqs: false`
- `sequence_length: 512`, `head_norm: true`, untied head (`head.linear` is a separate parameter)
- `init_type: truncated_normal_fixed`, `init_scale: 0.1`, `llm_random_weight_init`

## Defining a new one

1. Pick `H` (even). Read `dmodel = 64H`, `dff = 256H`, `n_blocks = H`, `q_heads = H`,
   `kv_heads = H/2`, `datt = dmodel`.
2. Copy `configs/_model/llmrandom/def_300.yaml` to `configs/_model/llmrandom/def_<params>.yaml`
   and set the six `common:` fields. Nothing below `common:` needs to change — every dimension in
   the model tree is an interpolation of those fields, including the k/v output width, which is
   computed as `kv_heads · (datt // q_heads)`. Swap `FeedForward` for `LlamaFeedForward` and add
   the `gate_fn` block (copy it from [llama_3_1.yaml](../../configs/_model/llama_3_1.yaml)) so the
   FF is gated.
3. Check the parameter count against the table before launching anything.
