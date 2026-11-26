# Weight init: our framework vs pkmf

*Written 2026-11-07 from both codebases. Methods only: how each framework initializes a
model's weights before pretraining, not the shapes it builds. pkmf is read along the STEM /
static token-ID run in `pkmf/running/running_static_Hx.ipynb`: `scripts/pretrain_memory.py`
with `routing=token_id`, i.e. `Qwen3MemoryForCausalLM` with the static router. Ours is
`src/core` on `ame_main`. The pkmf links work only locally: `pkmf/` is gitignored here.*

## Init rules 

Every weight in both frameworks is set by one of these rules. `fan_in` is the number of
inputs a layer has: a `Linear` matrix's input dimension, e.g. `d_model` for `W_Q` and `d_ff`
for `W_down`. For a lookup table, PyTorch counts its width as `fan_in` and its number of
rows (the vocabulary) as `fan_out`.

| rule | formula | in words | used for |
|---|---|---|---|
| **our fan-in truncated normal** (`llm_random_weight_init`, scale 1) | N(0, σ²) with σ = `scale/√fan_in`, truncated to [−2σ, 2σ] | σ shrinks as the number of inputs grows, so each output starts at about the scale of the input | ours: [L], [H] |
| **pkmf's fan-in truncated normal** (`truncated_normal`, scale 0.1) | N(0, σ²) with σ = `√(scale/fan_in)`, truncated to [−0.2, 0.2], i.e. ±2·scale | the same idea, but the 0.1 scales the variance, so σ is √0.1 ≈ 0.32× ours. The ±0.2 bound is fixed, so it only bites when σ is large, as in the embedding | pkmf: [L], [H], and [E] with `fan_in = 1` |
| **standard truncated normal** (PyTorch's `trunc_normal_` defaults) | N(0, 1), truncated to [−2, 2] | a fixed std, whatever the table's shape | ours: [E] |
| **plain normal** | N(0, σ²) with σ = `bank_init_std` = 1; no truncation | a fixed std, whatever the table's shape | ours: [B] |
| **Xavier (Glorot) normal** | N(0, σ²) with σ = `√(2/(fan_in + fan_out))`; no truncation | σ depends on both dimensions of the table, i.e. on the vocabulary size and the width | pkmf: [B] |
| **ones** | every value set to 1 | a fixed starting value | [N] in both |

*Truncated to [−c, c]* (also written "cut at ±c") means every value lies between −c and c;
none is larger or smaller. The values follow the normal's bell curve inside that range,
rescaled so it still sums to one: the mass the full normal would put beyond ±c is spread
over the inside, which is the same distribution as drawing from the normal and redrawing
anything outside. PyTorch's `trunc_normal_` samples it directly (inverse-CDF sampling over
[−c, c], then a clamp against rounding). Cutting off the tails lowers the std: at ±2σ it
comes out 0.88σ, so the standard truncated normal has std 0.88, not 1.

## Where each weight lives

Both models are pre-norm transformers. The tags in brackets mark each kind of weight; the
legend below the diagrams names each tag's init rule in both frameworks, using the names
from *Init rules* above.

```text
OURS (src/core): the AME model. The dense baseline is the same except inside the attention.

  token ids t
      │
      ▼
  token embedding [E]          looked up once, at the input; gives the starting vector x
      │ x
  ┌───▼── block, repeated n_blocks times ─────────────────────────────
  │ RMSNorm [N]
  │ attention
  │   q = q_bank[t] [B]    k = k_bank[t] [B]      the token ids, looked up again
  │   v = W_V·x [L]
  │   out = W_O·(…) [L]
  │ x = x + out
  │ RMSNorm [N]
  │ FFN: out = W_down·( silu(W_gate·x) ⊙ W_up·x )       all three W [L]
  │ x = x + out
  └───┬───────────────────────────────────────────────────────────────
      ▼
  RMSNorm [N] ─► output head [H] ─► logits

  The dense baseline's attention instead has q = W_Q·x [L], k = W_K·x [L] and RoPE;
  no [B].
```

```text
pkmf (STEM / static token-ID run): Qwen3MemoryForCausalLM

  token ids t
      │
      ▼
  token embedding [E]          looked up once, at the input; gives the starting vector x
      │ x
  ┌───▼── decoder layer, repeated L times ────────────────────────────
  │ RMSNorm [N]
  │ attention
  │   q = W_Q·x [L]    k = W_K·x [L]    v = W_V·x [L]
  │   QK-norm gains [N], RoPE
  │   out = W_O·(…) [L]
  │ x = x + out
  │ RMSNorm [N]
  │ FFN in the dense layers (0, 1, 3, 5, …):
  │   out = W_down·( silu(W_gate·x) ⊙ W_up·x )          all three W [L]
  │ FFN in the memory layers (2, 4, …, L−2):
  │   out = W_down·( silu(W_gate·x) ⊙ M[t] [B] )        W_gate, W_down [L]
  │   M[t]: the token ids, looked up again in this layer's memory table
  │ x = x + out
  └───┬───────────────────────────────────────────────────────────────
      ▼
  RMSNorm [N] ─► output head [H] ─► logits
```

| tag | weights | where | ours | pkmf |
|---|---|---|---|---|
| [E] | token embedding | once, at the input | standard truncated normal: N(0, 1), cut at ±2 | pkmf's fan-in truncated normal with `fan_in = 1`: σ = √0.1 ≈ 0.32, cut at ±0.2, so close to uniform on [−0.2, 0.2] |
| [B] | memory banks | ours: `q_bank` and `k_bank` in every attention layer; pkmf: `M` in every memory layer's FFN | plain normal: N(0, 1), no truncation | **Xavier normal**: σ = `√(2/(fan_in + fan_out))`, no truncation |
| [L] | every `Linear` matrix | attention and FFN | our fan-in truncated normal: σ = `1/√fan_in`, cut at ±2σ | pkmf's fan-in truncated normal: σ = `√(0.1/fan_in)`, cut at ±0.2 |
| [H] | output head | after the last norm; untied from [E] | our fan-in truncated normal, as for [L] | pkmf's fan-in truncated normal, as for [L] |
| [N] | norm gains | before each attention and FFN, and at the end; in pkmf also the QK-norm | ones | ones |

## Differences

Most consequential first. Memory banks are lookup tables indexed by token id, like the
token embedding. The embedding is the model's one input table; the banks are extra tables
that AME and pkmf add inside the layers, and each framework initializes them with a rule of
their own.

| # | what | ours | pkmf | in short |
|---|---|---|---|---|
| 1 | memory banks [B] (our AME Q/K banks, pkmf's STEM memory tables) | plain normal with a fixed std: N(0, σ²), σ = `bank_init_std` = 1; untruncated | **Xavier (Glorot) normal**: std `√(2/(fan_in + fan_out))` of the whole table; untruncated. Forced for static routing: `memory_init_std` must stay null | The main difference. Xavier ties the init to the table's shape; ours is a chosen constant, and our framework uses Xavier nowhere. Under our QK-norm the constant only sets how fast the banks learn. pkmf's table feeds the FFN with no norm after it, so its scale is also the memory layer's starting output. |
| 2 | `Linear` layers [L] and the output head [H] | fan-in (LeCun) truncated normal with the scale on the std: std `scale/√fan_in`, scale 1 | fan-in (LeCun) truncated normal with the scale on the variance: std `√(scale/fan_in)`, scale 0.1 | Same method family, different scale convention: every pkmf `Linear` weight starts √0.1 ≈ 0.32× ours, at any width. |
| 3 | truncation of the normals in [L], [H] and [E] | relative: cut at ±2σ | fixed: cut at ±2·scale = ±0.2, whatever σ is | pkmf's bound never binds on a `Linear` weight but cuts the embedding hard. pkmf's non-default `truncated_normal_fixed` is exactly our fan-in truncated normal. |
| 4 | token embedding [E] | its own rule: a standard normal cut at ±2 (PyTorch's `trunc_normal_` defaults) | pkmf's fan-in truncated normal, the function its `Linear` layers use, with `fan_in = 1` (std √0.1) under the fixed ±0.2 cut | Ours stays a truncated standard normal; pkmf's comes out close to uniform on [−0.2, 0.2]. |
## Same in both

- Every `Linear`, including attention O, FF out and the untied output head, uses the
  framework's fan-in truncated normal (see *Init rules*); neither scales residual-output
  layers by depth.
- Norm gains start at one, pkmf's QK-norm gains included. No layer has a bias.
- One seed is set before the model is built: ours calls `torch.manual_seed` with the
  dataset seed; pkmf's `set_seed` seeds `random`, `numpy` and `torch`.
- Sharded memory tables start from the same numbers as unsharded ones. Ours initializes the
  full table once and each rank slices out its columns; in pkmf each rank draws the full
  table from the same seeded RNG and keeps its column band.

## Sources

Ours: [`llm_random_weight_init`](../../src/core/model.py#L19-L23); the embedding's
`init_fn` in [base.yaml](../../configs/_model/h_balanced/base.yaml#L39-L41); the bank init
at [ame.py:280](../../src/core/ame.py#L280); the seeded CPU build at
[main.py:370-375](../../main.py#L370-L375); the column slicing at
[memory_parallel.py:188-199](../../src/core/memory_parallel.py#L188-L199).

pkmf (local):
[`_qwen3_memory_init_weights`](../../pkmf/src/poirot/models/qwen3_memory/modeling_qwen3_memory.py#L917-L1009)
and its [Xavier std](../../pkmf/src/poirot/models/qwen3_memory/modeling_qwen3_memory.py#L808-L811);
[`get_init_weight`](../../pkmf/src/poirot/models/jmsl_moe/initialization.py#L11-L53)
(`truncated_normal` at L40, `truncated_normal_fixed` at L48);
[config defaults](../../pkmf/src/poirot/models/qwen3_memory/configuration_qwen3_memory.py#L75-L123);
the [token-ID overlay](../../pkmf/scripts/conf/routing/token_id.yaml); and
[`set_seed`](../../pkmf/src/poirot/utils.py#L383-L391).

*Caveat.* The local pkmf snapshot has the package as `src/poirot/` and lacks
`src/pkmf/memory/`, so `NdEmbedding`'s own class is not readable here. The memory-table
values come from `_qwen3_memory_init_weights`, which `post_init()` applies to every module
after construction, and a pkmf test
([test_static_router.py:87](../../pkmf/tests/poirot/test_static_router.py#L87)) pins them
bit-exactly to `torch.nn.init.xavier_normal_`. `Qwen3Attention`, `Qwen3RMSNorm` and the head
are transformers' stock Qwen3 code, checked against 4.56.2.

## Analysis: what the differences mean, and which framework does better

*Researched 2026-11-07 from the papers listed at the end of this section and from both
codebases. It covers only the four differences above. Verdicts, and anything marked
**analysis**, are reasoning, not measurements: nothing here was run. Numbers are at h16
(`d_model` = 1024) and at the learning rate both frameworks use there, 2⁻⁹ ≈ 2·10⁻³,
unless said otherwise.*

| # | difference | how much it matters | better | confidence |
|---|---|---|---|---|
| 1 | memory banks [B]: plain N(0, 1) vs Xavier | a lot for how fast the banks learn; the two tables do different jobs | neither in general; for our banks, σ = 1 sits at the slow end of a knob nobody has tuned | low: no published comparison |
| 2 | `Linear` scale [L], [H]: std `1/√fan_in` vs `√(0.1/fan_in)` | moderate: final quality is insensitive in pre-norm models, stability is not | pkmf, read together with #4 | low to medium |
| 3 | truncation: ±2σ vs a fixed ±0.2 | small by itself; its real effect shows up in #4 | ours | high |
| 4 | token embedding [E]: std 0.88 vs a near-uniform std 0.11 | moderate, through training stability | ours, read together with #2 | medium |

### 1. Memory banks: a fixed std against Xavier

- **The same init means different things in the two layers.** Our bank rows pass through
  QK-norm, so σ never reaches the forward pass and only sets how fast the rows learn.
  pkmf's table enters the FFN with no norm after it, so its σ is also the memory layer's
  starting output.
- **Xavier's reasoning does not carry over to a lookup table.** Glorot & Bengio choose
  Var(W) = 2/(fan_in + fan_out) so that a dense layer, whose every output sums over its
  inputs, keeps activation and gradient variance level from layer to layer. A lookup sums
  nothing: its output is one row. On a vocabulary-sized table the formula just gives
  σ ≈ √(2/V), 0.0061 for pkmf's tables, about 52× below the `up_proj(x)` output (std
  0.32) the table replaces, so pkmf's memory layers start nearly silent. The STEM paper
  states no init and ablates none; pkmf's config says the Xavier init matches STEM's
  released code.
- **Related evidence says the scale costs little in final loss.** UltraMem sets its memory
  values so the layer's output starts like an MLP's, N(0, 1/(2L)). That init change moved
  its loss by −0.003 (151M model, 500B tokens) while preventing variance blow-ups early in
  training.
- **For our banks the init sets the learning speed (analysis).** QK-normed rows are
  scale-invariant, and the rotation of a scale-invariant weight per step falls as its norm
  grows (Kosson et al.), so the initial norm sets the early speed. Under Adam a row turns by
  about `lr/σ` per step: 0.2% at σ = 1. Xavier on our bank shape would give σ ≈ 0.0062 and
  about 31% per step at first, likely too fast, then slowing as the norm grows, since no
  weight decay holds it back. The best σ lies somewhere between, and nothing has measured
  it.
- **Never-seen tokens differ too (analysis).** In pkmf a row that is never updated stays at
  its tiny init, so an unseen token adds almost nothing. In ours QK-norm keeps every row at
  unit length, so an unseen token's row is a random direction whatever σ is.
- **Verdict.** No experiment compares the two, and each init fits its own layer. pkmf's
  tiny init together with its 4× learning rate means a frequent token's row loses its init
  within a few steps at peak learning rate. Ours is the open question: σ = 1 is the slow end
  of the `bank_init_std` knob, and a short sweep (for example 1, 0.1 and 0.03) would
  settle it.
- **Would Xavier help our banks? No.** Neither init is established for token-indexed Q/K
  tables, which are AME's own construction. N(0, 1) is PyTorch's default for an embedding
  table; Xavier on a token table comes from STEM's code and is neither justified nor tested
  in its paper, and pkmf's learned-router tables use UltraMem's formula instead. Under
  QK-norm, Xavier would not change the forward pass at init: both inits give random
  directions, and with the same seed the same ones. It would change only the rows' length
  (σ = 0.0062 on our `[50304, 1024]` bank shape), and with it two things (analysis):
  - **learning speed:** about 31% per step at first instead of 0.2%, then slowing as the
    rows grow, since no weight decay holds them back;
  - **gradient size:** a normalized row's gradient scales as 1/‖row‖ (Kosson et al.), so
    the bank gradients would grow about 160×. Our trainer clips one joint norm over the
    backbone and the banks at 1.0
    ([trainer.py:925-947](../../src/core/trainer.py#L925-L947)), so the banks would
    dominate it and every clip would also shrink the backbone's gradients. How much that
    matters under Adam is untested.

  So keep N(0, 1) until the `bank_init_std` sweep says otherwise, and watch the joint
  gradient norm while running it; a Xavier-sized 0.006 is the extreme end of that range.

### 2. `Linear` layers: scale on the std against scale on the variance

- **The gap.** pkmf's weights are 0.36× ours, after our truncation. Common defaults sit in
  between: 0.02 in GPT-2 and OLMo 2, and SmallInit's `√(2/(5d))` (Nguyen & Salazar, and
  Takase et al.'s default), both about 0.63/√d at `d_model` = 1024. Ours (0.88/√d; nominal
  1/√d, T5's rule) is about 1.4× above that; pkmf's (0.32/√d) is about 2× below.
- **Quality is insensitive in pre-norm models.** Nguyen & Salazar shrank the init by 0.63×
  (SmallInit). With pre-norm, BLEU moved by at most 0.29, which they read as no effect;
  with post-norm it decided whether training converged at all.
- **Stability favours smaller weights.** Takase et al. find that stable pre-norm training
  needs small sub-layer weights as well as a large embedding; see #4.
- **Adam shifts the difference (analysis).** Adam moves each weight by about the learning
  rate per step, so smaller weights change faster relative to their size: about 2.8× faster
  in pkmf at the same learning rate. Each framework tunes its own learning rate, which
  absorbs part of this.
- **The output head** follows the same rule. Its only visible effect is the initial loss,
  about 11.2 for ours and 10.9 for pkmf, gone within a few steps.
- **Verdict.** No evidence separates them on final quality. On stability pkmf's smaller
  weights are the safer side, but they have to be read together with its much smaller
  embedding (#4).

### 3. Truncation: bounds that follow σ against a fixed bound

- **Ours cuts at ±2σ**, the convention of TensorFlow ("re-drawing any samples that are more
  than two standard deviations from the mean") and of JAX ("truncated at 2 standard
  deviations"). JAX also rescales so that the std after the cut matches the nominal one;
  ours does not, which only shrinks it to 0.88× and is harmless.
- **pkmf cuts at ±2·`scale` = ±0.2 whatever σ is**, which looks like a units slip: the bound
  is in units of the scale knob instead of σ. pkmf itself carries `truncated_normal_fixed`,
  whose ±2σ bounds are identical to ours, but does not use it by default.
- **Significance.** On `Linear` layers the slip changes nothing that matters: pkmf's
  weights are effectively untruncated, and no LLM study we found isolates truncation at
  ±2σ against none. Its real effect is on the embedding (#4).
- **Verdict.** Ours, with high confidence: it does what its name says.

### 4. Token embedding: std 0.88 against a near-uniform std 0.11

- **pkmf's embedding init is an accident of #3**, not a chosen design: its usual fan-in rule
  with `fan_in = 1`, under the fixed ±0.2 cut.
- **Evidence.** Takase et al. (pre-LN, 350M to 13B) show that the first layer norm's
  Jacobian scales as 1 over its input's std, so a small embedding amplifies gradients in the
  shallow layers.
  - Their default, with an embedding std of about 0.02, spiked. Giving the embedding a std
    of about 0.63 to 1 (Scaled Embed, Embed LN) removed the spikes and lowered 1.7B
    WikiText perplexity from 22.58 to 21.29.
  - At learning rate 10⁻³ the 1.7B default diverged while Scaled Embed reached 20.95; at 13B
    the default failed at 3·10⁻⁴.
  - Stds of 5 and 50 also avoided spikes but hurt quality, and at a low learning rate
    (10⁻⁴) the default was roughly as good.
  - OLMo 2 tried these remedies, found they slowed convergence, and kept 0.02 embeddings
    with weight decay removed from them instead.
- **Where the two frameworks land.** Ours (0.88) is inside Takase's target of 0.63 to 1.
  pkmf's (0.11) is 5.7× above Takase's spiking default and 5.6× below Scaled Embed. Both
  frameworks train at 2⁻⁹ ≈ 2·10⁻³, the high-learning-rate regime where the remedies
  mattered most.
- **Read with #2 (analysis).** Takase's bound weighs the sub-layer weights' size against the
  embedding's std. On the attention path (`W_V`, `W_O`) the two frameworks come out within
  1% of each other: pkmf's two weights are together 0.13× ours, and so is its embedding. For
  the SwiGLU FFN, with three matrices, pkmf's value is about a third of ours (our extension
  of their two-matrix bound). Neither framework has the depth-scaled output projections
  that Takase's stable runs use.
- **Verdict.** Ours, for the embedding taken alone. Together with #2 the two frameworks
  start out close, and the combination Takase found stable is a mix of both: our std-1
  embedding with smaller `Linear` weights (our `init_scale` ≈ 0.7 gives SmallInit's
  0.63/√d) and depth-scaled output projections. Untested here.
- **Would Xavier help our embedding? No.** It is not the established choice for a token
  embedding: PyTorch's `nn.Embedding` defaults to N(0, 1), and ours is that truncated at
  ±2. T5 uses N(0, 1); the original Transformer and Gemma multiply the embedding by
  √d_model in the forward pass, for an effective std near 1; GPT-2 and OLMo 2 use a fixed
  0.02. Xavier was derived for dense layers.
  - On our `[50304, 1024]` embedding it gives σ = 0.0062, about 140× smaller than ours and
    about 3× below the 0.02 default that spiked in Takase et al.
  - Their "Xavier Init" run, with Xavier for every weight including the embedding, was
    their worst configuration and spiked most often. Its WikiText perplexity was 33.92 at
    350M, against 30.03 for their default and 29.86 for Scaled Embed, and 30.10 at 1.7B,
    against 22.58 and 21.29. Multiplying that Xavier embedding by √d recovered most of the
    gap: 30.50 and 23.16.
  - Xavier also ties the embedding's scale to the vocabulary size, where μP keeps it at a
    fixed O(1) whatever the width.

  So keep the current init.

### Overall

- **Ours does better** on truncation (#3) and on the embedding (#4).
- **pkmf does better** on `Linear` scale, for stability (#2), though it goes about 2× below
  the common defaults.
- **Memory banks (#1) have no winner**, because the two tables do different jobs. For ours,
  σ is a learning-speed knob that nobody has tuned.
- Out of scope: the "Same in both" items, and everything that is not an init (learning
  rates, weight decay, QK-norm).

Sources for this section:
- [Glorot & Bengio 2010](https://proceedings.mlr.press/v9/glorot10a.html), Xavier init.
- [Takase et al. 2025](https://arxiv.org/abs/2312.16903), "Spike No More" (COLM 2025):
  Tables 1, 2, 6 and 7; Appendices B.1 and C.
- [Nguyen & Salazar 2019](https://arxiv.org/abs/1910.05895), "Transformers without Tears":
  SmallInit, Table 2.
- [Kosson et al. 2024](https://arxiv.org/abs/2305.17212), "Rotational Equilibrium" (ICML
  2024): scale-invariant weights, §1 and §5.
- [Huang et al. 2024](https://arxiv.org/abs/2411.12364), UltraMem: §3.2, Appendix A and
  Table 2.
- [Sadhukhan et al. 2026](https://arxiv.org/abs/2601.10639), STEM (ICLR 2026): no init
  stated and no init ablation.
- [OLMo 2](https://arxiv.org/abs/2501.00656): §2.3, §3.2 and §3.4.2.
- [TensorFlow `truncated_normal`](https://www.tensorflow.org/api_docs/python/tf/random/truncated_normal)
  and [JAX `variance_scaling`](https://docs.jax.dev/en/latest/_autosummary/jax.nn.initializers.variance_scaling.html).
- [PyTorch `nn.Embedding`](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html):
  weights initialized from N(0, 1).
- [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762), §3.4: embedding weights
  multiplied by √d_model.
- Gemma in [transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma/modeling_gemma.py):
  the embedding output multiplied by √hidden_size (checked in 4.56.2).
- T5's init in [transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py)
  (`_init_weights`): embedding N(0, 1).
- [Yang et al. 2022](https://arxiv.org/abs/2203.03466), μP (Tensor Programs V), Table 3.

## Memory-bank init for sparsely updated rows: a proposal

*Written 2026-11-07. A design proposal built from the analysis above and from published
memory systems; nothing here was run or implemented. It refines the sweep suggested under
Analysis #1 and the "Would Xavier help our banks?" note. Numbers use the learning rate both
frameworks train at, 2⁻⁹ ≈ 0.002.*

### The requirement

Memory rows are updated only when their token appears. One row may have had 1M updates and
another only 1K, while the rest of the model has had 1M. When the 1K-update row is used next
to weights trained 1000× more, it should have little influence, and its influence should
grow as it is trained.

### How far a row moves (analysis)

- Adam moves each element by about the learning rate per update. On noisy gradients, the
  usual case for a rare row, a row has moved about `lr·√N` per element after N updates:
  about 0.06 after 1K updates and about 2 after 1M.
- **Rule:** start every row well below how far a rare row's updates will move it, so that
  even a rare row reflects its own updates rather than its random start. That means σ of
  about 0.01 or smaller, never 1.

### What published memory systems do

| system | table init | what follows the table | memory learning rate | a rarely trained row at init |
|---|---|---|---|---|
| pkmf, static routing (STEM) | Xavier normal, ≈ 0.006 | nothing: it multiplies the FFN gate | 4×, constant; no weight decay | near zero, so nearly silent |
| STEM, official code | Xavier normal, ≈ 0.004 at a ~128k vocabulary | nothing: it multiplies the FFN gate | no separate rate in the released config | near zero |
| UltraMem v1 | normal with variance `E/(2mHL)`, so the output starts as N(0, 1/(2L)) like an MLP's | a weighted sum of the retrieved values | 10×, decaying linearly to 1× | small |
| UltraMemV2 | normal, std matched to the dense FFN's output variance | a weighted sum of the retrieved values | 4×, decaying to 1× | small |
| Engram | not stated; the demo code leaves PyTorch's N(0, 1) | projections and a context gate that suppresses memory contradicting the context | 5× with Adam; no weight decay | whatever the gate lets through |
| Gemma 3n PLE | normal 0.01, ×16 on lookup (effective 0.16) | mixed with a unit-RMS projection of the main embedding; the branch ends in an RMSNorm | not public | a small perturbation on a well-trained shared path |
| ours (AME banks) | N(0, 1) | QK-norm: every row rescaled to unit length | 1×; no weight decay | a random direction at full strength |

Every system either starts its tables small or trains them at a boosted learning rate, or
both. Ours does neither.

### Unnormalized tables: near-zero init already does it

With no norm after the table, a row's size grows with how much it has been trained, so a
1K-update row contributes little next to 1M-update rows. pkmf's Xavier (≈ 0.006) and
Gemma's 0.01 both qualify. In a multiplicative path like STEM's, even an exactly zero init
works, because the row's gradient does not depend on its own value.

### Our QK-normed banks: init alone cannot do it

- QK-norm rescales every row to unit length, so a 1K-update row acts with the same strength
  as a 1M-update row.
- At σ = 1 a 1K-update row has moved 0.06 against an init scale of 1, so its direction is
  still 99.8% its random start. Today every rare token is a full-strength random query or
  key, giving attention logits of std about 1 (τ = 8 over 64 dimensions).
- No choice of σ fixes this on its own: the norm discards the size that would have carried
  the "how much has this row been trained" information.

### Recommendation, best first

1. **Small init plus a soft QK-norm.** Replace `q / ‖q‖` with `q / √(‖q‖² + ε²)`, with
   ε = 1, fixed, and σ ≈ 0.01.
   - **ε = 1 comes from prior work and is not tuned.** The epsilons of LayerNorm, RMSNorm
     and `F.normalize` (1e-5 to 1e-12) only guard against dividing by zero: with them a row
     of length 0.08 would already be fully normalized, and the soft norm would do nothing.
     Capsule networks use the same "a vector's length is its confidence" idea with the
     constant 1: their squash, `‖s‖²/(1 + ‖s‖²) · s/‖s‖`, ensures "that short vectors get
     shrunk to almost zero length and long vectors get shrunk to a length slightly below 1"
     (Sabour et al. 2017). ISRU, `x/√(1 + αx²)`, is the one-number form of our formula, with
     ε = 1/√α (Carlile et al. 2017). What matters is ε against the row lengths, and the init
     sets those, so at σ = 0.01 the numbers below follow from ε = 1 without tuning. We keep
     our formula rather than the squash, because the squash shrinks short vectors
     quadratically and leaves a tiny row with a tiny gradient.
   - Untrained rows (length about 0.08 per head) stay nearly zero, so their logits are
     about 0. As a key, a rare token neither attracts nor repels attention; as a query, its
     attention falls back on the ALiBi distance prior and the sink.
   - Influence grows with training: a 1K-update row (length about 0.5) acts at about half
     strength, and frequent rows (length well above 3) behave exactly like today's QK-norm,
     with logits still capped at ±τ.
   - Gradients stay bounded, at most 1/ε, so the joint gradient clip stays sane.
   - The precomputable score table survives: the norm still acts on each row, before the
     KV shift.
   - It is a one-line change in `_mixed_qkv`; not made.
2. **Alternatives with the same effect, at the cost of extra parameters or code:**
   - a per-row confidence scalar initialized near 0, `q = s[t]·normalize(E[t])` (analysis);
   - Engram's context gate, which "tends toward zero, effectively suppressing the noise"
     when retrieved memory contradicts the current context; removing it gave some of the
     largest losses in their ablation;
   - Gemma 3n PLE's backoff: mix each row with a projection of the token's main embedding,
     so a rare token falls back on a shared, well-trained path.
3. **Init only:** set `bank_init_std` ≈ 0.01. Rare rows then point where their own few
   updates took them instead of in a random direction, but they still act at full strength.

### What to avoid

- **σ = 1 without a soft norm**, today's setup: rare rows inject random queries and keys.
- **Exactly zero init with the hard norm:** `F.normalize` of a zero vector has a gradient of
  about 1/eps = 10¹². Under the joint clip, every step that uses a token's row for the first
  time would scale down the backbone's gradients.

### Keep idle rows still

- Plain AdamW moves every row on every step through its decaying momentum, so a rare row
  keeps drifting after its last real update. Our bank optimizer steps every row this way.
- Lazy or selective updates step only the rows that received a gradient; pkmf has this as
  `mem_selective_updates`, off by default. With idle rows held still, under-trained rows stay
  near their small init.
- With a soft norm, weight decay on the banks would matter again (analysis): it would let
  long-idle rows fade back toward neutral, where today's hard QK-norm makes decay invisible.

### Testing it

- Sweep `bank_init_std` over 1, 0.1, 0.03 and 0.01 with today's hard norm, watching the
  joint gradient norm.
- Then one run with the soft norm at σ = 0.01 and ε = 1. ε is not swept; see the
  recommendation above.
- Read the effect per token-frequency bucket: row length, attention entropy of queries from
  rare tokens, and loss on rare tokens.

Sources for this section:
- [Engram](https://arxiv.org/abs/2601.07372): §2.3 (the gate), §6.2 (ablation), Table 5.
- [UltraMem](https://arxiv.org/abs/2411.12364): §3.2, Appendix A, §5.4.
- [UltraMemV2](https://arxiv.org/abs/2508.18756): §3.5, §6, §4.2.9.
- [STEM code](https://github.com/Infini-AI-Lab/STEM): `lingua/stem_dist_utils.py`
  (`ParallelEmbedding`, defaulting to `xavier_normal_`).
- [Gemma 3n in Google's JAX library](https://github.com/google-deepmind/gemma):
  `gemma/gm/nn/gemma3n/_modules.py` and `_layers.py`; JAX's
  [`normal`](https://docs.jax.dev/en/latest/_autosummary/jax.nn.initializers.normal.html)
  defaults to std 0.01.
- pkmf: `mem_selective_updates` in
  [pretrain_memory.yaml](../../pkmf/scripts/conf/pretrain_memory.yaml) (local).
- [Sabour, Frosst & Hinton 2017](https://arxiv.org/abs/1710.09829), "Dynamic Routing Between
  Capsules", Eq. 1: the squash, the precedent for ε = 1.
- [Carlile et al. 2017](https://arxiv.org/abs/1710.09967): ISRU, `x/√(1 + αx²)`.

## Implementation plans for the three bank-init strategies

*Written 2026-11-07. Descriptions only: nothing here is implemented. Each plan says what the
strategy is, what changes where, what it interacts with, and how to check it. Code
references are to `ame_main`. The plans follow the recommendation above: A is the soft
QK-norm, B the explicit trust mechanisms, C the init-only change.*

### Shared ground

- **One place to change.** QK-norm happens only in `_mixed_qkv`
  ([ame.py:445-448](../../src/core/ame.py#L445-L448)). `score_matrix` and
  `attention_diagnostics` call `_mixed_qkv`, so a change there reaches the diagnostics too.
- **Bank init** is `bank.weight.normal_(0, bank_init_std)` in the constructor
  ([ame.py:280](../../src/core/ame.py#L280)). `bank_init_std` is already a config option
  (`attention.ame.bank_init_std`).
- **New options** go into `_AME_OPTIONS`
  ([attention.py:206](../../src/core/attention.py#L206)), so a typo fails at build time.
- **Memory parallelism.** The bank code finds the banks by name, `q_bank` and `k_bank`
  ([memory_parallel.py:380](../../src/core/memory_parallel.py#L380)), slices their columns
  across GPUs and fuses the two into one gather (`qk_bank`). Any new token-indexed parameter
  either goes through that code or is packed into the banks as extra columns.
- **Existing tests that encode today's behaviour**, in `tests/test_ame.py`:
  `test_rescaling_a_bank_row_does_not_move_any_score`,
  `test_scores_start_at_std_one_with_the_prescribed_init`,
  `test_the_banks_are_not_decayed` and `test_decay_cannot_move_a_bank_row`.
- **Shared evaluation.** Bucket tokens by training frequency (for example under 1K, 1K to
  100K, over 100K occurrences, counted from the data or online). Per bucket read the mean
  row length, the attention entropy of queries from those tokens, and the loss on those
  tokens; overall read the joint gradient norm and the loss against today's baseline.

### Plan A: small init plus a soft QK-norm (recommended)

**What it is.** Rows are still normalized, but a row shorter than ε stays short:
`q̂ = q / √(‖q‖² + ε²)`. Trained rows (`‖q‖ ≫ ε`) become unit vectors exactly as today.
Untrained rows, drawn small, give near-zero queries and keys, so their logits are about 0,
and their influence grows as they train.

**How to implement.**
1. `AMEAttention.__init__`: a new boolean option `soft_qk_norm`, false by default, which keeps
   today's hard norm. ε is a module constant fixed at 1.0, deliberately not a config option,
   so it cannot drift into a sweep (why 1: the recommendation above). Add `soft_qk_norm` to
   `_AME_OPTIONS`.
2. `_mixed_qkv`: when `soft_qk_norm` is on, replace the two `F.normalize` calls with
   `q * torch.rsqrt(q.pow(2).sum(-1, keepdim=True) + 1.0)`, likewise for `k`. Compute it
   in fp32, since the rows arrive in bf16 from the bank code, and keep it on the rows,
   before the KV-shift mix, so each score is still a combination of two entries of one row
   of the precomputable table.
3. Run configs set `bank_init_std: 0.01` together with the soft norm. Keep the code
   default at 1.0 so existing configs reproduce their numbers.
4. `tau` stays at √d_head: trained rows still cap at ±τ, and the horizon readout (τ/m)
   keeps its meaning for them.
5. Weight decay: the banks stay out of decay by default. Add an option that lets decay act
   on them in this mode; with a soft norm it would shrink long-idle rows back toward neutral.
6. Tests:
   - keep the rescaling test for the hard norm and add a soft-norm counterpart: rescaling a
     long row barely moves its scores, while rescaling a short row moves them in proportion;
   - a soft-norm version of the init test: scores start near 0 at σ = 0.01;
   - finite gradients in bf16 for tiny rows;
   - reference vs flex backend parity with the soft norm on;
   - the two decay tests for the new option.
7. Instrumentation: log the mean row length, and the fraction of rows shorter than ε, per
   frequency bucket.

**Interactions.**
- Gradients through the soft norm are bounded by 1/ε, so the joint gradient clip is safe.
- At ε = 1 and σ = 0.01 every row starts in the near-linear regime; a row needs a length of
  about 3 to be close to fully normalized.
- Stage 1's "rare-type rows stay at init" diagnostic changes meaning: rows now start near
  zero rather than at a random unit direction.

**How to check.** The soft norm at σ = 0.01 with ε of 0.5, 1 and 2, against the hard norm
at σ = 1 and at σ = 0.01, on the same seed and data, read with the shared evaluation.

### Plan B: an explicit "how much to trust this row" mechanism

Each variant makes rare rows non-influential with an extra component instead of the norm.

**B1, a per-row confidence scalar.** `q = s[t] · normalize(E_Q[t])`, and likewise for `k`,
with one `s[t]` per token and head in (0, 1).
- Parameterize `s = sigmoid(c)` with `c` initialized so that `s ≈ 0.05`. The row still
  learns from the start, since its gradient is proportional to `s`, and a rarely used token
  keeps a small `s`.
- Store `c` as H extra columns of each bank (width `H·d_head + H`, 1040 at h16, which stays
  divisible by 1, 2, 4 and 8 GPUs). The memory-parallel gather, slicing and fused path then
  need no new plumbing. Split the gathered row in `_mixed_qkv` and apply `sigmoid(c)` after
  the norm and before the KV-shift mix. The score stays a combination of two entries of one
  row of a precomputable table, now `s_q[a]·s_k[b]·T[a, b]`.
- The constructor initializes the d_head columns with `normal_` and the H columns with a
  constant. Keep `c` out of weight decay, which would pull `s` toward 0.5.
- Cost: V × H scalars per bank, about 0.8M at h16 and about 26M across all 32 banks, against
  1.65B bank parameters.

**B2, a context gate (Engram-style).** A per-query, per-head gate computed from the hidden
state, `q_t ← g_t · q_t` with `g_t = sigmoid(w · RMSNorm(x_t) + b)`. Engram's own gate
compares the hidden state with the retrieved key; this is its simplest form for AME.
- Add a `Linear(dmodel → H)` to `AMEAttention`, compute the gate in `_mixed_qkv` from `x`,
  and multiply it into `q` after the norm and before τ. Initialize `b` so the gate starts
  near 1, i.e. today's behaviour.
- Trade-off: the score becomes `g_i · τ · T[t_i, t_j] − m·(i − j)`, so the query side now
  reads the hidden state. Keys and the precomputed table are untouched, and decode still
  works with one extra scalar per query, but "scores depend on token ids and distance only"
  no longer holds on the query side. When `g_i` is small, attention falls back on the ALiBi
  distance prior and the sink.
- The gate does not know which rows are under-trained; it learns when identity matching
  misleads. Pair it with a small init.

**B3, a backoff path (Gemma 3n PLE-style).** `row(t) = (base(t) + E[t]) / √2` with `E`
drawn small, so that a token with little training falls back on `base(t)`.
- Shared-vector variant: `base` is one learned vector per bank and head, `[H, d_head]`,
  added to the gathered rows before the norm. Every token trains it, so a never-seen token
  behaves like an average token instead of a random one. It is cheap, and the scores stay a
  function of token ids only.
- Projection variant, Gemma's choice: `base(t) = RMSNorm(P · emb(t))`, a projection of the
  token's main embedding, which pools gradient from every layer. It needs the embedding rows
  passed into the attention and costs about one Q/K projection per bank, the cost AME was
  built to avoid; use it only if the shared vector proves too weak. `base(t)` still depends
  on the token alone, so decode can precompute it.
- Under the hard QK-norm a rare token's query or key becomes about `normalize(base)`, an
  average direction; combined with Plan A it becomes near zero instead.

### Plan C: init only, `bank_init_std` ≈ 0.01

**What it is.** No code change: the banks are drawn from N(0, 0.01²) instead of N(0, 1).
Under the hard QK-norm the forward pass at init is identical. Rare rows then point where
their own few updates took them instead of in a random direction, but they still act at
full strength, and every row turns about 100× faster relative to its length.

**How to implement.** Set `attention.ame.bank_init_std: 0.01` in the run config. The
memory-parallel slicing is unaffected, and so are the existing tests, since the forward pass
does not change.

**Interactions.**
- A normalized row's gradient scales as 1/‖row‖, so the bank gradients enter the joint 1.0
  clip about 100× larger and may dominate it. Watch the joint gradient norm and the banks'
  share of it; if they dominate, clip the banks separately (a change in
  `Trainer.clip_gradient`) or lower the banks' learning rate.
- Without weight decay the rows grow quickly, by about `lr·√N` per element, so the banks'
  effective learning rate falls over training.

**How to check.** The sweep σ = 1, 0.1, 0.03 and 0.01 on the same seed and data, read with
the shared evaluation. It also serves as the baseline for Plans A and B.

### Suggested order

1. **Plan C first.** It is config-only and cheap, and it measures how much the banks'
   learning speed matters and whether their gradients dominate the clip.
2. **Plan A next.** A one-line code change plus tests that targets the requirement
   directly.
3. **Plan B only if A falls short.** B1 is the closest in spirit to A, and B3's
   shared-vector variant is the cheapest. B2 changes the method's "token ids and distance
   only" property, so it needs a deliberate decision first.
