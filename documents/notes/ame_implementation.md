# AME in this repo: the layer, the plumbing, the review, and what is left

**Status: established.** The layer, the plumbing and all eight review findings are done
and tested, on CPU and on an H100, and the code is committed. **No *scientific* experiment
has been run — Stage 0 has not started.** The cluster jobs run since are throughput
benchmarks, not quality measurements; their record is
[ame_optimization_report.md](ame_optimization_report.md).
Written 2026-11-03, last updated 2026-11-06.

AME — *Attention Memory Embedding* — replaces the Q and K projections of a standard
attention head with two memory banks indexed by **token id**. The design record lives
outside this repo, in `research/ame/` (`ame.md` for the method, `ame_experiments.md` for
the staged program, `baselines.md` for the comparison standard). This note is the repo
side: what landed, which open choices were settled to land it, what is verified, and what
is still outstanding. Paired reading:
[attention-switching.md](../concepts/attention-switching.md) for the registry `ame` sits
in, and [archive/ame_review_and_fixes.md](archive/ame_review_and_fixes.md) for the
evidence behind each finding and the remediation detail.

**The code is committed on `ame_main` and `ame_optimize`** as of 2026-11-05. It was
developed in the `/home/maciej/code/nano-small-changes` worktree, which was removed
2026-11-04; passages below that refer to it describe where things stood before then.

> **Name.** Called MAE in discussions before 2026-11-03; renamed to AME, the name the
> registry already used, because MAE collides with Masked Autoencoders and with "mean
> absolute error". Same method.

## Where things stand

| | state |
|---|---|
| The training-time layer | **implemented**, 60 CPU tests + 9 GPU tests |
| Repo plumbing (ids, MFU, config, optimizer, instrumentation) | **implemented** |
| CPU suite | **262 passed, 0 skipped, 0 failed** |
| GPU correctness | **verified** — 9/9 on an H100, and a 3.16B run that trains |
| GPU *performance* | **18.1% MFU** at h20 when this was written. **Optimized since** to 35.02% at 4 GPUs and 37.96% at h16/1 GPU — [ame_optimization_report.md](ame_optimization_report.md) |
| Code review (2026-11-03) | 8 findings, F1–F8, **all fixed** |
| Fix plan | **all four phases applied** |
| Experiments | **none run.** Stage 0 has not started. |
| Decode system | **not built, deliberately** |

## The method, as implemented

```python
q = E_Q[t]                      # [B,n,H,D]  gather by token id -- NOT x @ W_Q
k = E_K[t]                      # [B,n,H,D]  gather              -- NOT x @ W_K
v = (x @ W_V).view(B,n,H,D)     # ordinary projection, unchanged from MHA

q = l2_normalize(q, dim=-1)     # QK-norm on the BANK ROWS, before the mix
k = l2_normalize(k, dim=-1)

k = a1*k + a2*shift1(k)         # KV shifting -- a1,a2,b1,b2 are [H] learnable
v = b1*v + b2*shift1(v)         # shift1(z)[:,j] = z[:,j-1], position 0 zeroed

s = einsum('bihd,bjhd->bhij', q, k) * tau   # tau [H], init sqrt(d_head)
s = s - m * dist                            # ALiBi, per-head slope -- the ONLY
s = s.masked_fill(causal, -inf)             #   positional signal; no RoPE anywhere
out = softmax(s) @ v
out = out.reshape(B,n,H*D) @ W_O
```

Scores are therefore `s_ij = tau * T[t_i,t_j] - m_h*(i-j)` with `T[a,b] =
norm(E_Q[a]) . norm(E_K[b])` — **a function of token ids and distance only**. That is the
whole design: after training the scores collapse into a vocab² table, so decode needs no
Q/K matmuls and no K-cache. It is also the whole weakness: the weights never read a hidden
state, so retrieval is never content-addressed, in any variant.

Training is **unmodified O(n²) attention with two gathers**. There is no table, no row
cache, no V-cache, no CPU offload, and none of it should be built before an experiment
says the mechanism works.

KV shifting (`alpha`, `beta`) is **prior art**: Xu, Cheng, Wang & Chen, arXiv 2411.19574.
It is the only mechanism that gives an id-scored layer the induction/copy primitive, which
is why it is on by default — and why any dense baseline AME is measured against must have
it too, or AME collects a gain that is not its own. How it works, why nothing else can do
its job, what it costs at decode, and what is planned: [KV shifting in AME](#kv-shifting-in-ame).

## KV shifting in AME

*Written 2026-11-05 from the method record (`research/ame/ame.md` §1.2, §4.1, §4.3–§4.4),
the code, and the paper itself (arXiv 2411.19574v2). Points marked **analysis** are
reasoning from the code, not measurements.*

### The operator

KV shifting (Xu, Cheng, Wang & Chen, Baichuan, Nov 2024) replaces each position's key and
value with a learned blend of its own and the previous position's, before attention runs.
Queries are not touched.

```python
# Xu et al., Section 2.2 and the Appendix F code
Q = X @ W_Q                        # NOT shifted -- hence "KV" shifting
K = X @ W_K
V = X @ W_V
K = a1*K + a2*shift1(K)            # shift1(z)[:, j] = z[:, j-1]; zeros at position 0, no wrap
V = b1*V + b2*shift1(V)            # a1, a2, b1, b2: learned scalars per (KV) head, per layer
Q, K = rope(Q), rope(K)            # the shift runs BEFORE the rotary embedding
out = softmax(Q @ K.T / sqrt(d) + causal_mask) @ V
```

Init `a1 ~ U(0,1)`, `a2 = 1 − a1`, and the same for `b`; nothing constrains them after
that. It is a 2-tap causal convolution over time on K and V, costing 4 parameters per head
and one shifted copy of K and V.

| position | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| key | α₁k₁ | α₁k₂ + α₂k₁ | α₁k₃ + α₂k₂ | α₁k₄ + α₂k₃ |
| value | β₁v₁ | β₁v₂ + β₂v₁ | β₁v₃ + β₂v₂ | β₁v₄ + β₂v₃ |

- **Score:** `q_i·k̂_j = α₁(q_i·k_j) + α₂(q_i·k_{j−1})` — position j is matched partly on
  what sits at j−1.
- **Readout:** `Σ_j w_ij·(β₁v_j + β₂v_{j−1})` — reading position j also returns part of
  j−1's content.
- **Special cases:** `α = β = (1,0)` is plain attention. `α = (0,1), β = (1,0)` is "find the
  position whose *predecessor* matches the query, return that position's value": an
  induction head in one layer.

What the paper shows, mostly from single runs on private data:

| claim | evidence |
|---|---|
| one layer is enough for induction | Theorem 2: one layer, one head, width d represents the induction-heads operator **exactly**; vanilla attention needs two layers at width 2d and only approximates it (Theorem 1) |
| it helps pretraining | 2.9B / 500B tokens, average of 8 benchmarks **36.45 → 38.57** (+2.1), MMLU 37.26 → 40.88 (+3.6); 19B / 200B tokens 38.06 → 38.83 (+0.8) |
| a wider window does not help | mixing in 2 or 3 previous tokens gave no gain (1.5B, 10B tokens) — the reason AME has no p-tap key window |
| shifting Q too hurts | Appendix J, 19B stopped at 180B tokens: "not as good as shifting KV, and even not as good as the vanilla" — the reason AME never shifts queries |

### Same operator, different input

AME applies the operator unchanged: the two `shift1` lines in the method block above,
[src/core/ame.py:448-455](../../src/core/ame.py#L448-L455). What differs is what it is
applied to.

| | Xu et al. | AME |
|---|---|---|
| shift, init, queries | `k̂_j = α₁k_j + α₂k_{j−1}`, `v̂_j = β₁v_j + β₂v_{j−1}`; α₁ ~ U(0,1), α₂ = 1 − α₁; Q unshifted | **identical** |
| where the mixed key sits | shift before RoPE, so it is rotated as position j | ALiBi distance `i−j` on the mixed score: also position j |
| **key before the shift** | `W_K·x_j`: from the hidden state, already carries context | `Ê_K[t_j]`: a bank row picked by token id, no context |
| **what the shifted-in half brings** | the previous position's whole hidden state | only the previous token's id |
| values | `W_V·x_j` | `W_V·x_j` (Option B): identical |
| key normalization | none | ℓ2 on the bank rows, **before** the mix |
| score scale | fixed `1/√d` | learned `tau` per head |
| positional signal | RoPE, base 100k | ALiBi, a fixed geometric ladder |
| coefficients | per KV head (GQA) | per head (AME is MHA-shaped) |

In a standard transformer the shift is an **upgrade**: two layers can already build
induction, and the shift does it in one, sooner. In AME it is **load-bearing**: it is the
only context a key ever gets. AME may claim *"for an id-scored layer the key shift is
necessary for copying"*; it may not claim the shift.

### Why AME cannot copy without it

The score `s_ij = tau·T[t_i,t_j] − m·(i−j)` sees two token ids and a distance. No AME key
or query ever reads the hidden state. Copying needs the score at the copied position to
know that position's *predecessor*. Take "Mr Kowalski met Mrs Nowak. … Mr → ?":

| position | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| token | Mr | Kowalski | met | Mrs | Nowak | **Mr** ← query |
| plain key | K[Mr] | K[Kowalski] | K[met] | K[Mrs] | K[Nowak] | K[Mr] |
| shifted key, α = (0,1) | 0 | **K[Mr]** | K[Kowalski] | K[met] | K[Mrs] | K[Nowak] |

- **Without the shift**, positions 2 and 5 differ only in their predecessor (Mr vs Mrs),
  which the score cannot see. The best a head can learn is "Mr attends to surnames",
  `T[Mr,Kowalski] ≈ T[Mr,Nowak]`, and the distance term then picks Nowak: wrong.
- **Depth does not help.** In a standard transformer, layer 1 writes "I came after Mr"
  into `x_2` and layer 2's `W_K·x_2` reads it. AME's layer 1 can write the note, but layer
  2's key is `E_K[Kowalski]`, which never looks at `x_2`. The same holds at every layer.
- **With the shift**, position 2 scores `tau·T[Mr,Mr]` (the query meets its own id) and
  position 5 scores `tau·T[Mr,Mrs]`. Position 2 wins, and `v_2 = W_V·x_2` carries Kowalski.
- **Why it is the only mechanism.** Context can reach the score only through the query,
  the key or the bias. A query or key that reads `x` is ordinary attention and loses the
  vocab² table, the point of the method; the bias knows only distance. What is left is
  feeding the key the id at a fixed offset, which is the shift.
- **α versus β.** Depth can stand in for β, because values read the hidden state and any
  local head mixes `t_{j−1}` into `x_j`; only layer 1 has no substitute. Nothing stands in
  for α. Hence four Stage 0 arms, not two.

### At decode: an exact table form

The dot product is linear and the bank rows are normalized *before* the mix, so

```
s_ij = tau · q_i · (α₁k_j + α₂k_{j−1})            − m·(i−j)
     = tau · (α₁·T[t_i, t_j] + α₂·T[t_i, t_{j−1}]) − m·(i−j)
```

— two entries of the **same** row `T[t_i, ·]`, an exact rewrite rather than an
approximation. Normalizing the *mixed* key instead would divide by
`‖α₁Ê_K[t_j] + α₂Ê_K[t_{j−1}]‖`, which depends on a key–key cosine that `T` does not hold,
so it would need a second vocab² table. The code normalizes the rows first
([src/core/ame.py:25-27](../../src/core/ame.py#L25-L27)), and
[test_a_score_is_a_linear_combination_of_two_entries_of_one_row](../../tests/test_ame.py#L169)
pins it.

```python
# One decode step i, Option B, one head-layer: research/ame/ame.md §1.2 with the shift.
# R = T[t_i, :] from the row cache keyed by token id.
score[j] = tau*(a1*R[t[j]] + a2*R[t[j-1]]) - m*(i - j)   # past j: two reads, one row
                                                         # (the j-1 term is 0 at position 0)
# CPU, async: top-k over score, read v_hat[j] from the RAM V-cache, accumulate (N, Z, max)
v_i     = W_V @ x_i                                      # GPU, synchronous from here on
v_hat_i = b1*v_i + b2*v_prev                             # v_prev = RAW v_{i-1}, a register
s_ii    = tau*(a1*R[t[i]] + a2*R[t[i-1]])
out_i   = merge(N, Z, max, exp(s_ii) * v_hat_i)          # online-softmax merge
V_cache.append(v_hat_i); v_prev = v_i                    # the cache holds mixed values
```

| decode step | what changes | cost |
|---|---|---|
| score | two entries of the same cached row | **none**: same row, same bandwidth |
| V-cache | store `v̂_j` instead of `v_j` | **none**: same bytes. Plus a raw-`v` register (~98 KB at the reference config): `v̂_i` needs raw `v_{i−1}`, and inverting `v̂` to recover it is unstable. |
| self-term | `α₁T[t_i,t_i] + α₂T[t_i,t_{i−1}]` | same cost |
| CPU async path | `t_{j−1}` is a past id | **none**: still no dependence on hidden states |
| **top-k index** | the score depends on `(t_{j−1}, t_j)`, so the index is keyed by **bigram**, not type | **the real cost.** One selection per head-layer, since α and m are per head: 12.6M scored bigrams/token at 32k context and 38.4M at 100k (reference config, 384 head-layers). It grows with context where the type count saturates, and needs WAND/MaxScore-style postings. |
| pruned row cache | a top-k-pruned row may lack `T[t_i, t_{j−1}]` | candidates must come from either slot; α = (1,0) or (0,1) is safe, mixtures break |
| Option A (bank values) | its sum over types becomes a sum over bigrams | it loses saturation and constant state, so evaluate A *without* the key shift |

**Verdict.** Exact decode: feasible and essentially free. Top-k decode: feasible, but
bigram selection is a new cost nobody has measured — "not obviously smaller than the
200 MB value gather it would be dismissed against" (`ame.md` §4.1). Nothing is built.

A **query shift** would also be table-computable — four entries from the rows `T[t_i,·]`
and `T[t_{i−1},·]`, the second being the previous step's row — but Xu et al.'s one test
says it hurts, so it is not planned.

### What the shift does not buy

- **One-token triggers.** "Anna said yes. Bob said no. Anna said → ?": both positions
  after "said" score `tau·T[said,said]`, and distance picks "no". A standard transformer
  can match on "Anna said"; AME's query does not see "Anna", and its keys see one step
  back.
- **A conjunction, not a trigram.** `α₁T[a,b] + α₂T[a,c]` is additively separable, so in
  weight space it is a product of two bigram factors. No term knows the pair `(c, b)` as
  a unit.
- **Rare triggers** (analysis, unmeasured). The shifted match needs a high diagonal
  `T[a,a]`: a token matching itself.
  - In standard attention that comes from one shared `W_Qᵀ W_K`, so it holds for every
    token, rare ones included. That is why induction heads copy even random tokens.
  - In AME, `E_Q` and `E_K` are separate tables initialized independently
    ([src/core/ame.py:278-280](../../src/core/ame.py#L278-L280)). So `T[a,a] ≈ 0` at init,
    and each token has to learn its own self-match.
  - AME copying therefore generalizes over **what is copied** (values go through the
    shared `W_V`) but not over **what triggers the copy**. This is risk R3, the Zipf
    tail, showing up inside the copy mechanism.
  - A small-vocabulary synthetic task will not reveal it. Probe it with repeated random
    tokens, split by token frequency.
  - Untested lever: initialize `E_K := E_Q`. That gives `T[a,a] = 1` for every token from
    step 0, at the cost of every head starting out as a same-token head.
- **"A steep slope is a previous-token head" overstates it** (analysis).
  [ame_experiments.md](ame_experiments.md) E0.1 and `ame.md` §4.1 say a steep enough slope
  puts "nearly all mass on j = i−1".
  - Here the causal mask includes `j = i` at zero distance penalty
    ([src/core/ame.py:572](../../src/core/ame.py#L572)), so a steep slope alone
    concentrates on the token itself. A true previous-token head also needs a low
    `T[a,a]`, learned per token.
  - The default slopes are fixed ([src/core/ame.py:326-329](../../src/core/ame.py#L326-L329)),
    and the steepest has a 16-token horizon, where one step of distance costs well under
    one logit.
  - The conclusion survives in a weaker form: local heads blur `t_{j−1}` into `x_j`. Read
    Stage 0's steep-head 2×2 with that in mind.

### Plans and open items

1. **Stage 0** ([ame_experiments.md](ame_experiments.md) E0.1) measures exactly this. The
   arms:
   - **A1:** α and β free.
   - **A2:** α free, β pinned. This is the irreplaceable half.
   - **A3:** α pinned, β free.
   - **A4:** both pinned. This is the fixed-kernel floor.
   - **P1:** a vanilla 2-layer transformer.
   - **P2:** one standard-attention layer with KV shifting, the Theorem 2 harness check.

   The arms are reachable from config (`shift`, `pin_alpha`, `pin_beta`). Hold the slope
   ladder identical across arms. Kill criterion: A4 scores the same as A1. Headline
   readout: the α₂ and β₂ trajectories, where a collapse to 0 means the model declined
   the shift.
2. **The matched dense control does not exist.** `RoPEAttention` has no α/β, so any
   AME-versus-dense win is confounded with Xu et al.'s published gain (risk R8, baseline
   #5, "mandatory"). Matching Xu exactly means shifting K and V *before* the rotary
   embedding. It is about 30 lines (`ame_theory.md`), and should be built before the
   first comparison run. The positional encoding differs as well (ALiBi vs RoPE); all
   three candidate controls are in
   [Positional encoding, and what a dense comparison must control](#positional-encoding-and-what-a-dense-comparison-must-control).
3. **The α₂/β₂ readouts are lost under FSDP2** (found 2026-11-05).
   - `instrumentation()` re-runs the bank gather and `v_proj` outside FSDP2's hooks, which
     raises on DTensor parameters. The trainer then disables that layer's
     instrumentation for the rest of the run
     ([src/core/trainer.py:996](../../src/core/trainer.py#L996)).
   - The parameter-only readouts (α₂, β₂, tau, horizons) work without token ids, but
     they are computed in the same call and lost with it.
   - Only `distributed: null` runs log them today
     ([configs/ame/h20_1gpu.yaml](../../configs/ame/h20_1gpu.yaml)). That covers every
     trainer run under FSDP2, the planned h16 run included.
   - Splitting the parameter readouts from the attention diagnostics would bring them
     back.

## Positional encoding, and what a dense comparison must control

*Written 2026-11-05. AME and the dense baseline use different positional encodings, which
the experiment plan does not yet account for. The section closes with what the literature
says about ALiBi vs RoPE, and what that implies for AME.*

| | AME | dense baseline (`gqa` → `RoPEAttention`) |
|---|---|---|
| positional signal | **ALiBi only**: `s_ij = tau·T[t_i,t_j] − m_h·(i−j)`; no RoPE anywhere | **RoPE only**, on q and k; no ALiBi |
| settings | one slope per head from a geometric ladder of horizons, 16 tokens … 4× the training context, `m = tau_init / horizon`; fixed by default, learnable only as an ablation; applied in fp32 in the flex `score_mod` ([src/core/ame.py:686-694](../../src/core/ame.py#L686-L694)) | base 10000, no frequency scaling ([configs/_model/h_balanced/base.yaml:67-68](../../configs/_model/h_balanced/base.yaml#L67-L68)) |
| other position-dependent parts | the KV shift's fixed one-step offset; the learned sink is position-independent | none |

**Why AME cannot use RoPE.** RoPE rotates q and k by their positions, so the score would
depend on the two ids and the distance *jointly*, `T[t_i, t_j, i−j]`. That breaks both
decode-time properties: the single vocab² table (it would need one per distance), and the
top-k merge, which relies on a per-type score plus a term that only falls with distance
(`ame.md` §2.3). An additive penalty keeps the score as "id part + distance part".

**Why the penalty cannot be dropped.** Without a positional signal the score is a function
of ids alone, and the layer sees an unordered bag of tokens (`ame.md` §3).

**Consequence: AME and the dense model differ in three ways, not one.**

1. Where Q and K come from: token-id tables vs projections of the hidden state. This is
   the idea under test.
2. The KV shift. The planned dense+shift control covers it (risk R8).
3. ALiBi vs RoPE. Nothing in `baselines.md`, `ame_experiments.md` or `ame_theory.md`
   addresses it as of 2026-11-05.

| control | isolates | effort |
|---|---|---|
| dense RoPE + shift, Xu-style (they use RoPE base 100k and shift before RoPE) | the shift | ~30 lines on `RoPEAttention` |
| dense + AME's ALiBi ladder + shift | the shift and the positional encoding | needs a flex `score_mod` path in the dense attention, which today runs on SDPA |
| **`AMEAttention` with `x@W_Q`, `x@W_K` in place of the bank lookups** | everything except AME's idea: same ALiBi ladder, normalization, `tau`, sink and shift | contained in one module: build `W_Q`/`W_K` instead of the banks and use them in `_mixed_qkv` |

**Recommendation.** Keep the plain RoPE dense model as the standard baseline a reviewer
expects. Add the last row as the attribution control: it is the only comparison whose
single difference is id-indexed vs projected keys, and it settles the shift and the
positional encoding in one run. Treat the Xu-style row as a replication of their result.
None of the three is implemented.

### ALiBi vs RoPE in a standard transformer: what the papers say

*Researched 2026-11-05 from the papers' own text. Every number in this subsection is the
paper's; none is ours.*

**In short.** At the training length the two are close. The studies split, most gaps are
under half a percent of the loss, and the newer studies lean RoPE. ALiBi's selling point is
flat perplexity past the training length, but it gets that by acting like a sliding window,
not by using the extra context. On real long-context tasks RoPE is clearly ahead, and
RoPE's context-extension recipes are why the field settled on it.

| | RoPE | ALiBi |
|---|---|---|
| mechanism | rotates q and k by position, so the score depends on content and distance *jointly* | adds `−m_h·(i−j)` to the score; the distance term ignores content |
| what one head can express | any learned distance pattern, e.g. "exactly the previous token" | only "nearer is better", at a fixed rate per head |
| cost | element-wise on Q and K; works with any attention kernel | a bias inside the kernel (FlashAttention-2 `alibi_slopes`, flex `score_mod`); Falcon calls it "significantly cheaper to compute than RoPE" |
| standard settings | base 10000 | slopes `m_k = 2^(−8k/n)`, k = 1…n; at 16 heads, 0.71 … 0.0039 |
| used by | LLaMA, Mistral, Qwen, Falcon 7B/40B/180B: most models since 2023 | BLOOM, MPT, Falcon-RW, BTLM, Jais, Baichuan 13B |

**Head-to-head at the training length**

| study | scale | result | ahead |
|---|---|---|---|
| Press, Smith & Lewis 2022 (the ALiBi paper) | 247M, WikiText-103 | perplexity: ALiBi 19.73 vs rotary 20.07 at L = 512; 18.66 vs 19.33 at L = 1024 | ALiBi |
| Le Scao et al. 2022 (BigScience) | 1.3B, OSCAR, 112B tokens | zero-shot average over 27 tasks: ALiBi 43.70, learned 41.71, rotary 41.46, none 41.23 | ALiBi |
| Almazrouei et al. 2023 (Falcon) | 1B and 3B, the Pile, 30B and 60B tokens | 1B: RoPE +0.8 and +1.1 on the two zero-shot suites, and lower loss. 3B: tied on the main suite, RoPE +0.7 on the small one | RoPE, shrinking with scale |
| Dey et al. 2023 (BTLM) | 111M, the Pile, 2048 tokens | loss: RoPE 2.259, ALiBi 2.267, learned 2.296 | RoPE, barely |
| Li et al. 2024 (FIRE) | 125M, C4, 2048 tokens | log-perplexity: RoPE 3.070, ALiBi 3.216, no positional encoding 3.111 | RoPE, clearly: the largest gap |
| Schröder et al. 2026 | 148M, three seeds | validation loss: RoPE 2.842, ALiBi 2.846 | tie |

The two ALiBi wins are the oldest results, and neither reports a significance test.
BigScience's evaluation put a model with no positional encoding within 0.23 of rotary, so
it is coarse.

**Past the training length**

- **Perplexity.** RoPE without tricks improves for 100–200 extra tokens and then breaks.
  Trained at 512, it reaches 79 at 15.5k tokens, against ALiBi's 18 (Press). ALiBi stays
  flat, but Press's own sliding-window evaluation finds it "cannot improve perplexity when
  looking at longer sequences".
- **Why ALiBi stays flat.**
  - Chi et al. 2023: "Large slopes implicitly enforce a narrow windowed bias".
  - Extrapolation works only while the training length covers a head's empirical
    receptive field. Flatten the slopes past that, and perplexity explodes.
  - Their conclusion names "the models' limitation of not utilizing information beyond
    the training sequence length".
- **The flat curve fades with training.**
  - BTLM chose ALiBi for extrapolation, then found the model learns to "overfit" its
    context length: "ALiBi alone does not provide competitive extrapolation capability".
    Their 8K model holds to about 9.2K.
  - In FIRE's runs ALiBi's perplexity rose past the training length too: 3.216 at 2K,
    3.537 at 8K.
- **Using long context.** Fine-tuned at 8K (FIRE), ALiBi trails RoPE by 5–6 points on the
  SCROLLS average: 15.1 vs 20.4 at 125M, 16.6 vs 22.4 at 350M. On NarrativeQA it gets half
  RoPE's F1, 4.2 vs 8.7. FIRE's general explanation is that such biases "significantly
  bias towards local attention".
- **Counter-evidence, from small or synthetic setups.**
  - Copying strings longer than any seen in training: "the ALiBi and NoPE transformers
    dramatically outperform the RoPE model on longer inputs" (Jelassi et al. 2024).
  - At 148M, ALiBi beat RoPE at in-context passkey retrieval (0.93 vs 0.64) and on the
    needle test (0.68 vs 0.28). Past its training length RoPE scored 0.00 (Schröder et
    al. 2026).
  - On synthetic length-generalization tasks, no positional encoding beats both
    (Kazemnejad et al. 2023).
- **Heads go blind** (Schröder et al. 2026).
  - Once `m·d` reaches about 90, the attention weight underflows to exactly zero. The
    thresholds are about −92 in bf16 and −103 in fp32.
  - Steep heads therefore become hard windows: BLOOM's steepest at about 124 tokens.
  - Clamping the bias, or penalizing `log(d+1)` instead of `d`, restores retrieval past
    the training length: passkey AUC 0.08 → 0.79 with both.
- **ALiBi stretches too** (Al-Khateeb et al. 2023). Scaling every slope by
  `L_train / L_test` at inference, with no training, keeps perplexity flat to about 2× the
  training length. It also roughly doubles ROUGE on 16K-token summarization.

**Why the field chose RoPE.** Position interpolation (Chen et al. 2023) and its successors
gave RoPE long context after a short fine-tune. Falcon cites these recipes as eroding
ALiBi's extrapolation edge, and adopted RoPE "in-line with other popular large-scale
models", with custom kernels for the overhead.

Sources:
- [Press, Smith & Lewis 2022](https://arxiv.org/abs/2108.12409), the ALiBi paper.
- [Le Scao et al. 2022](https://arxiv.org/abs/2210.15424), BigScience's architecture
  ablations.
- [Almazrouei et al. 2023](https://arxiv.org/abs/2311.16867), Falcon, §4.3.2.
- [Dey et al. 2023](https://arxiv.org/abs/2309.11568), BTLM-3B-8K.
- [Li et al. 2024](https://arxiv.org/abs/2310.04418), FIRE.
- [Chi et al. 2023](https://arxiv.org/abs/2212.10356), receptive-field analysis.
- [Kazemnejad et al. 2023](https://arxiv.org/abs/2305.19466), positional encoding and length
  generalization.
- [Jelassi et al. 2024](https://arxiv.org/abs/2402.01032), "Repeat After Me".
- [Schröder et al. 2026](https://arxiv.org/abs/2608.03994), "When Attention Goes Blind".
- [Al-Khateeb et al. 2023](https://arxiv.org/abs/2310.13017), position interpolation for
  ALiBi.

### What the papers mean for AME

*Analysis: reasoning from the papers above, the code and the design record. Nothing here
is measured.*

- **The positional-encoding confound is too large to ignore.**
  - Published gaps at the training length run from a tie to 0.15 nats (FIRE), and
    BigScience saw 2.2 points the other way. That is the order of the effect AME is meant
    to show.
  - The literature cannot subtract it. The `AMEAttention`-with-projections control above
    holds the positional encoding fixed.
- **AME's slopes are not standard ALiBi.** At h16, `tau_init = √64 = 8`
  ([src/core/ame.py:150-164](../../src/core/ame.py#L150-L164),
  [src/core/ame.py:294](../../src/core/ame.py#L294)):

  | | steepest head, m | flattest head, m | flattest head's penalty across a 2048-token context |
  |---|---|---|---|
  | standard ALiBi, 16 heads | 0.71 | 0.0039 | 8 nats |
  | AME default ladder | 0.5 | 0.00098 | 2 nats |

  - AME's scores start at std ≈ 1 by construction, about the scale of a standard head at
    init, so the slopes compare directly. AME's ladder is 1.4× flatter at the steep end and
    4× at the flat end.
  - Its horizons, `tau/m`, run 16 … 8192 tokens. A horizon is the distance at which the
    penalty cancels a perfect match.
  - Inside the training context, the flat heads are close to having no positional
    encoding. The papers put that slightly behind RoPE: FIRE 3.111 vs 3.070, BigScience
    41.23 vs 41.46.
  - So the published ALiBi numbers bracket AME's case but do not pin it down.
- **Underflow does not bind.** With `tau = 8`, the penalty reaches Schröder et al.'s ≈ 90
  only at about 11 horizons. A head is out of reach long before that: at two horizons, a
  perfect match already weighs `e^−tau ≈ 3·10⁻⁴` of a neutral neighbour's.
- **The ladder caps AME's reach far below its 100k pitch.**
  - What was planned: `ame.md` §7 pitches 100k context from 2k–8k training, and asks for
    horizons of about 20 … 185k (slopes 2⁻¹ … 2⁻¹⁴).
  - What the code does: it spans 16 … 4·seq_len, which is 8192 tokens at seq_len 2048
    (`ame_theory.md` T4: "DECIDED in code, UNVALIDATED").
  - What that means: at 32k, the flattest head's best possible match weighs
    `e^−24 ≈ 4·10⁻¹¹` of a neutral neighbour's, and at 100k, `e^−90`. That is the
    receptive-field limit Chi et al. describe. A larger learned `tau` stretches every
    horizon in proportion.
  - The papers make neither obvious fix free:
    - Heads that reach past the training length look like no positional encoding inside
      it, and Chi et al. and FIRE see those degrade when run past it.
    - Heads that fit inside it never use the extra context.
  - The published ways out:
    - rescale the slopes at inference: Al-Khateeb et al., about 2× without training;
    - train at the target length;
    - clamp the bias, or penalize log-distance: Schröder et al.
  - None of these is in the plan docs. `ame.md` §7 already asks for validation at about
    10× the training length before any deployment number is trusted. The papers suggest a
    fixed ladder will either ignore the extra context there or degrade on it.

## Weight initialization, dense and AME

*Written 2026-11-06 from the code; values at h16 (`dmodel` 1024, `dff` 4096). Points marked
**analysis** are reasoning, not measurements.*

**How it runs.**
- Every weight is initialized in its module's constructor. There is no separate init pass
  and no meta-device build.
- `main.py` seeds with `torch.manual_seed(<dataset seed>)`, builds the model on CPU and
  moves it to the GPU ([main.py:370-375](../../main.py#L370-L375)). So one seed fixes both
  the initial weights and the data order; without a code change they cannot be varied
  separately.
- A resumed run loads its weights from the checkpoint instead.
- With `mem_parallel_size: 4`, each GPU slices its columns out of the already-initialized
  bank ([src/core/memory_parallel.py:187-195](../../src/core/memory_parallel.py#L187-L195)),
  so 1-GPU and 4-GPU runs start from identical numbers.

**Shared by both models** ([configs/_model/h_balanced/base.yaml](../../configs/_model/h_balanced/base.yaml#L35-L109)):

| weights | init | h16 std |
|---|---|---|
| every `Linear`: dense Q, K, V, O; AME `v_proj`, `o_proj`; FF in and gate; output head | truncated normal, std `1/√fan_in`, cut at ±2σ ([src/core/model.py:19-23](../../src/core/model.py#L19-L23)) | 0.031 |
| FF out | the same rule; its fan_in is `dff` | 0.0156 |
| token embedding | `trunc_normal_` with PyTorch's defaults: std 1, cut at ±2 | 1 |
| output head | its own `Linear`, not tied to the embedding | 0.031 |
| RMSNorm gains | 1 | |
| biases | none: `Linear` defaults to `bias=False` | |

- The ±2σ cut makes the realized std about 0.88× the nominal one.
- There is no depth scaling. O and FF-out are neither scaled by `1/√(2·n_blocks)` nor
  zero-initialized, unlike GPT-2 or Megatron. Both models share this, so it is not a
  confound.
- The knobs are `init_scale` (the attention `Linear`s), the FF and head `scale`, and
  `bank_init_std`. All are 1 today.

**AME only** ([src/core/ame.py:270-340](../../src/core/ame.py#L270-L340)):

| weights | init | why |
|---|---|---|
| `q_bank`, `k_bank` | N(0, σ²) with σ = `bank_init_std` = 1, not truncated; two independent draws | `ame.md` §1 and §7: σ ≈ 1 is the per-element scale `W_Q·x` has in a standard transformer. Without QK-norm, σ = 0.02 would make attention uniform at init. |
| `tau` | `√d_head` = 8, learned | puts the QK-normed scores at std ≈ 1 |
| α, β | `c₁ ~ U(0,1)`, `c₂ = 1 − c₁` per head | Xu et al.'s init; both `(1,0)` and `(0,1)` are reachable. Pinned arms hold a frozen `(1,0)` buffer. |
| ALiBi slopes | the horizon ladder, a fixed buffer | [Positional encoding](#positional-encoding-and-what-a-dense-comparison-must-control) |
| sink key and value | zeros | logit 0, neutral in a score distribution of std ≈ 1 |

**What follows** (analysis):
- **No token matches itself at init.** The banks are independent draws, so `T[a,a] ≈ 0`.
  [What the shift does not buy](#what-the-shift-does-not-buy) covers this, including the
  untested `E_K := E_Q` lever.
- **With QK-norm, σ sets how fast the banks learn, not their scale.**
  - `tau` sets the score scale, so σ drops out of the forward pass.
  - Adam moves each element by about `lr` per step, so a row turns by about `lr/σ` per
    step: 2.4·10⁻⁴ at the configs' `lr = 2⁻¹²`.
  - A dense projection's weights move by about `lr/0.031 ≈ 8·10⁻³` of their size per
    step, about 30× more. The matrix is shared by every token, so the queries it produces
    can move faster still.
  - The banks are embedding-like tables, and μP gives such tables a larger learning rate
    than hidden matrices for exactly this reason. Here they share the global rate and get
    no weight decay (the bank row in
    [Choices the spec left open](#choices-the-spec-left-open-and-how-they-were-settled)).
    Their norms therefore tend to grow over training, which slows them further.
  - The dense model's token embedding is in the same position. What AME changes is that
    Q and K now live in such tables.
  - Untested. The knobs are `bank_init_std` and a per-group learning rate, and the
    per-group rate does not exist yet.

## What landed

| file | what |
|---|---|
| [src/core/ame.py](../../src/core/ame.py) | `AMEAttention` — the layer, both kernel paths, the slope ladder, `instrumentation()` |
| [src/core/attention.py](../../src/core/attention.py) | `ame` is a real registry entry; `_TYPE_OPTIONS` governs per-type option blocks |
| [src/core/model.py](../../src/core/model.py) | token ids threaded `LLM → encoder → block → Residual`; `RoPEAttention` states its own score FLOPs; `parameter_groups` |
| [src/core/mfu.py](../../src/core/mfu.py) | lookups excluded by module type, not by name; score FLOPs collected per module |
| [src/core/trainer.py](../../src/core/trainer.py) | `_log_layer_instrumentation` — per-layer diagnostics, duck-typed |
| [src/core/schedulers.py](../../src/core/schedulers.py) | the cosine scheduler takes grouped optimizers (one *lr*, not one group) |
| [main.py](../../main.py) | both optimizer sites build param groups |
| [src/definitions.py](../../src/definitions.py) | `AttentionSchema.ame` — the options block, typed as `Any` |
| [configs/_model/h_balanced/base.yaml](../../configs/_model/h_balanced/base.yaml) | `vocab_size` reaches `build_attention` |
| [configs/ame_tests/h20_smoke.yaml](../../configs/ame_tests/h20_smoke.yaml) | the first GPU run's config |
| [configs/ame/](../../configs/ame/README.md) | six configs, the best measured MFU per model size × GPU count; every other AME config moved to [configs/ame_tests/](../../configs/ame_tests/) on 2026-11-05 |
| [tests/test_ame.py](../../tests/test_ame.py) | 69 tests: the math, the ablation arms, the plumbing, weight decay, instrumentation, the fused kernel |

### Token ids travel with the activations

Nothing below `LLM.forward` used to see the ids. AME needs them at **every** layer, so
`forward` gained an optional `token_ids` that passes down the stack. A layer opts in by
declaring `wants_token_ids = True`; `Residual` resolves that flag once at construction, so
the branch is a Python constant dynamo folds away rather than a per-step lookup. Every
existing type is untouched and never receives the ids. Passing `None` to an AME layer
raises a `ValueError` naming the cause.

### The config surface

```yaml
common:
  attention:
    type: ame
    ame:                  # options, applied wherever ame is built (base or mix)
      shift: true         # KV shifting on
      pin_alpha: false    # pin the key-side mix to (1,0)   -- ablation arms
      pin_beta: false     # pin the value-side mix to (1,0)
      sink: true          # learned position-independent sink key
      qk_norm: true
      bank_init_std: 1.0
      horizon_factor: 4.0 # flattest head reaches 4x the training context
      backend: auto       # auto | flex | reference
```

Only a type listed in `_TYPE_OPTIONS` takes a block — today that is `ame` alone. A block
named after any other registered type is a build error, not a silent no-op. Unknown keys
inside the block fail at build time with the valid set printed.

## Choices the spec left open, and how they were settled

`ame.md` §0 lists what an implementer must decide. These are the answers this code
commits to; each is one config key away from being revisited.

| question | decided | why |
|---|---|---|
| RMS vs ℓ2 QK-norm | **ℓ2 + learned per-head `tau`**, init `sqrt(d_head)` | §7's default. The two conventions must not be mixed: RMS *with* a temperature lands at score std `1/d_head`, the exact failure §7 warns about. |
| ALiBi slopes per head? | **yes**, geometric, steepest first | assumed by every cost argument in the record |
| the slope ladder itself | **derived from the context**: horizons `16 … 4·seq_len`, converted by `m = tau/horizon` | the standard ALiBi ladder caps the flattest head near 3k tokens *whatever the context is*, silently capping a long-context method far below its own pitch. Specifying horizons makes that failure visible. |
| what position 1 does | **a learned, position-independent sink key**, on by default | a head that learns `alpha_1 -> 0` otherwise gets logit 0 at position 0 — finite, neutral, able to dominate a short prefix. Also insurance against the distance bias evicting the BOS sink (StreamingLLM's failure mode). **Tension:** §0's canonical block has no sink, §7 asks for one. `sink: false` gives the block verbatim. |
| AME everywhere or alternating | **everywhere by default**; `mix` interleaves | the hybrid is a fallback, not the plan |
| bank tied to the input embedding | **no** | an untested lever (`ame.md` §8.9), not a default |
| Option A's gate | **not implemented** | B is the method; A is a later ablation (E1.3) |
| `kv_heads` | **ignored**; AME is MHA-shaped | `mha` sets the precedent. Every head has its own bank slice and value head. |
| `cu_seqlens` for the roll | **not needed here** — see below | |
| weight decay on the scales | **off** for `tau`, `alpha`, `beta`, `slopes`, sinks | see the review, F2 |
| weight decay on the **banks** | **off**, decided 2026-11-03 | With QK-norm a row's length never reaches the score, so decay changes no output — it only shrinks `‖row‖`, inflating the effective LR on direction and breaking Stage 1's "rare-type rows stay at init" check. Costs matched regularization against the dense baseline's `W_Q`/`W_K`; say so when reporting (`ame_fix_plan.md` 2.6). |

## Repo-specific findings

**The `roll` trap is milder here than the record assumes.** `ame.md` §8.1 warns that
document packing makes `torch.roll` a silent correctness bug. This repo's packer
([datasets.py](../../src/core/datasets.py)) concatenates documents with `eos` and **no**
`cu_seqlens` or varlen masking, so attention already crosses document boundaries. The
shift only has to avoid wrapping the sequence itself, which `_shift1` does by padding and
slicing — wrap-free by construction. Zeroing position 0 of every document would make AME
*differ* from the baseline it is measured against, so it is deliberately not done. If
varlen attention is ever adopted, it comes back.

**MFU would have lied in both directions.** The old accounting charged `6*P` for every 2-D
weight not named `*embedding*` and collected score FLOPs only from `RoPEAttention`. AME's
banks are ~4× the backbone and cost zero matmul FLOPs; its score path costs exactly what
standard attention's does. Both rules were replaced with something a type states for
itself: `nn.Embedding` parameters are lookups, `non_matmul_parameters` names 2-D
parameters that are not weights, and `attention_score_flops_per_token` is how a type
declares its score cost. Counted FLOPs of every existing config are unchanged.

**The banks are large even at H4.** One bank is `vocab_size × d_model` *per layer*:

| H | backbone (total) | non-embedding | `E_Q + E_K` | vs non-emb |
|---:|---:|---:|---:|---:|
| 4 | 29.7M | 3.9M | 103.0M | 26.2× |
| 12 | 183.4M | 106.2M | 927.2M | 8.7× |
| 16 | 354.7M | 251.7M | 1648.4M | 6.5× |
| 24 | 1003.9M | 849.3M | 3708.8M | 4.4× |

There is **no CPU smoke config at the production vocab** — tests override
`common.vocab_size` down. Before a real run the bank-size levers (`d_bank < d_head`,
tying `E_Q`/`E_K`, sharing banks across layers, low-rank rows) stop being ablations and
become what decides whether the run fits. Optimizer state binds, not the weights: at
14 B/param an H16 bank set is ~23 GB before the backbone.

**Two kernel paths.** `reference` materializes the `n×n` scores; `flex` uses
FlexAttention's `score_mod`, which applies the distance bias without materializing
anything and keeps the slopes differentiable (flash's `alibi_slopes` takes a
non-differentiable tensor, so the learnable-slope ablation needs flex). They agree
exactly in fp32 and are both checked against a literal transcription of the spec.

**`backend: auto` is device-aware** (flex on CUDA, reference on CPU) as a consequence, not
a preference: a compiled `flex_attention` whose inputs require grad **cannot be lowered on
CPU at all** — it fails for a bare call with no score_mod and no block_mask, so it is the
backend, not this layer.

## What is verified

**Environments.** Everything ran on the pinned torch **2.7.1**
(`.pixi/envs/default/bin/python`, what the cluster uses) and on torch **2.9.0** (conda
`nanollmr`, the default `python3` on this box — *not* the pinned environment).

**CPU — full suite: 249 passed, 0 skipped, 0 failed.** The zero matters: the end-to-end
tests through `main.py` actually ran. See the harness trap below.

- **60 CPU tests** on the layer: the two kernel paths against each other and against the
  spec transcription, the ids-and-distance-only property, QK-norm placement, the no-wrap
  shift at every rank, each ablation arm's pinning, causality, init scale, bf16 autocast,
  weight-decay grouping, instrumentation, and the entropy / top-k diagnostics.
- **The MFU closed forms**, including AME's; existing configs' counted FLOPs unchanged.
- **End-to-end through `main.py`** for both gqa (regression on the plumbing) and AME. The
  AME run builds **132,452,688 parameters — exactly the closed form**, reports finite loss
  every step, and writes a resumable checkpoint.
- **FSDP2 on a gloo mesh** under the production `MixedPrecisionPolicy(bf16, fp32)`:
  forward and backward for `qk_norm` on/off (the mixed-`requires_grad` case), `shift=False`
  and `pin_alpha`, and the weight-decay grouping survives sharding.
- **Optimization health**: a 2-layer AME on fixed-bigram next-token prediction goes
  4.30 → 0.002 in 200 steps, matching a gqa twin. Read narrowly — that task is solvable
  from the current token alone, so it exercises the optimizer and plumbing, **not** the
  attention path.

**GPU — entropy `h100a`, H100 80GB, torch 2.7.1+cu128.** `pytest -m gpu tests/test_ame.py`:
**9 passed** — `auto` selecting flex and matching the reference path, the **compiled** flex
kernel forward and backward, bf16 autocast against fp32, the shift/sink/pinned variants
under compilation, and four added with Phase 1: that the default path does not materialize
the score matrix, that it is correct *without the test compiling it*, that it reaches the
compiled wrapper, and that a whole `TransformerBlock` compiled by `compile_blocks_full`
still receives the token ids.

**GPU training runs at h20, 1×H100.** Both built **3,163,200,720 parameters** — the
closed form exactly — FSDP2 sharded with no OOM at micro-batch 4.

- **Job 32315** (unfused kernel): loss **11.25 → 7.5 at step 19**, monotone, killed by the
  1-hour wall at 165 s/step.
- **Job 32360** (fused kernel, `compile_blocks_full`): all **30 steps in ~13 minutes**,
  loss **11.25 → 7.28**, zero errors, 22.12 s/step throughout.

**Correctness at 3.16B on real data is established**, and the two runs agree on the loss
trajectory, which is the check that the kernel change altered speed and not arithmetic.

### Step time, after the kernel was fixed

Phase 1 landed on 2026-11-03: `src/core/ame.py` now compiles `flex_attention` itself, so
the fused Triton kernel no longer depends on a config flag. Re-measured as entropy job
**32360**, `configs/ame_tests/h20_smoke.yaml`, 1xH100.

| h20, 1xH100, 512 x 2048 tok/step | params | micro-batch | tokens/s | MFU | step time |
|---|---:|---:|---:|---:|---:|
| gqa, `mfu_audit/h20_mgpu.yaml` (FSDP2 + compile_blocks_full) | 620.2M | 16 | 109,361 | 45.57% | 9.58 s |
| **ame, fused kernel** | **3.163B** | **4** | **47,391** | **18.1%** | **22.1 s** |
| ~~ame, unfused kernel (job 32315, superseded)~~ | 3.163B | 4 | 6,295 | 2.4% | 165 s |

**Fixing the kernel was worth 7.5x**: 165 s -> 22.1 s per step, MFU 2.4% -> 18.1%. Steady,
not warmup — steps 4-8 measured 22.124 / 22.119 / 22.113 / 22.115 s. Nothing but the
kernel changed between the two AME rows; the micro-batch, the model and the tokens per
step are identical, which is why `h20_smoke.yaml` keeps `gradient_accumulation_steps: 128`
rather than matching the baseline's 32.

**Which baseline.** Three h20 configs now exist and they are not interchangeable:
`mfu_audit/h20.yaml` (single-GPU, `distributed: null`, 46.4%), `mfu_audit/h20_mgpu.yaml`
(FSDP2 + `compile_blocks_full`, 45.57% at 1 GPU) and the stage-3 audit default that h12
and h16 still carry (30.8%). AME is read against **h20_mgpu**, because it needs FSDP2's
sharding and bf16 parameters and cannot take `distributed: null` at 3.16B parameters.

**The remaining 2.5x gap is not yet attributed**, and two of its candidates are still the
ones Phase 1 did not touch:

1. **Micro-batch 4 against the baseline's 16.** Smaller micro-batches amortise the
   per-step overheads over fewer tokens. Not free to fix: ~57 GiB of parameters, master
   weights, gradients and Adam moments leaves ~23 GiB, which will not hold 16.
2. **Dense bank gradients.** `nn.Embedding` backward produces a dense `[50304, 1280]`
   gradient per bank per layer — 40 of them, ~10.3 GB — on *every* micro-batch regardless
   of its size. `ame.md` 7 predicted this and noted that row-sparse gradients do not help,
   because a 0.5M-token batch touches 50-82% of rows anyway.
3. **FSDP2 reducing gradients on every micro-batch.** `src/core/trainer.py` has no
   `set_requires_gradient_sync(False)` around the accumulation loop, so all 3.16B
   gradients are reduced 128 times per step with a bf16->fp32 cast each time. A
   **framework** behaviour AME only makes visible, because the cost scales with parameter
   count while the compute does not; it would equally affect any large-bank model.

AME remains ~15% *cheaper* in FLOPs/token than its dense twin (3.77 G; the banks correctly
contribute zero), which is R6 in the risk register: fewer FLOPs, more seconds.

**Two bugs found on the way, both worth remembering.**

*bf16 autocast crashed the layer.* Autocast casts `nn.Linear` but **not** an `nn.Embedding`
gather, so `v` arrived bf16 while the banks handed back fp32, and flex rejects mismatched
q/k/v outright. The reference path did not crash, which is worse — it silently ran half the
layer in a precision nobody chose. Fixed by casting q and k to `v`'s dtype at the end of
`_mixed_qkv`. **This would have crashed the first real training run**, and no fp32 test
could have found it.

*A bug in the GPU tests, not the layer.* FlexAttention's compiled kernel requires
`head_dim >= 16`, and the shared `DIMS` fixture gave head_dim **8**. Every GPU test now
builds at `GPU_DIMS`, head_dim **64**, the H family's real width, with an assert pinning
it. A GPU test running an unrepresentative shape is worse than no GPU test.

**The test-harness trap.** `tests/test_training_smoke.py` is marked `slow` *and* skips when
`data/` and `data_eval/` are absent — both gitignored, so **a git worktree silently skips
the entire end-to-end suite**. With a `-m "not slow"` run that is two independent reasons
for the most valuable test in the repo to report nothing. Symlink them:

```bash
ln -sfn /home/maciej/code/nano/data data && ln -sfn /home/maciej/code/nano/data_eval data_eval
```

That suite has now caught two bugs nothing else did (F8 below, and the plumbing
regression). The same file asserts a fresh model's first loss is within **0.15** of
`ln(50304)`; one step of `tiny_local` is 30 tokens, over which gqa wanders 10.92–11.45, so
that assertion passes on a single lucky batch. The AME test uses a band of 1.0.

## The code review, and the fix plan

A second pass over the implementation on 2026-11-03 found **8 issues**. Evidence for each
is in [archive/ame_review_and_fixes.md](archive/ame_review_and_fixes.md) Part A; the
remediation detail and verification record are in Part B of the same document.

| # | severity | finding | state |
|---|---|---|---|
| **F1** | high — perf | GPU path ran FlexAttention **uncompiled** → the unfused fallback that materializes the full score matrix | **fixed** — 7.5x faster |
| **F2** | high — science | weight decay 0.1 hit `tau`, `alpha`, `beta`, sinks, banks — confounding both headline readouts | **fixed** |
| F3 | medium | `_shift1` silently wrong for non-4-D input | fixed |
| F4 | medium | `base.yaml` carried an unrelated hunk removing a mandatory-field guard | reverted |
| F5 | low | `token_ids=None` gave a cryptic `AttributeError` | fixed |
| F6 | low | option blocks for option-less types silently discarded | fixed |
| F7 | low | two factual errors in this note | corrected |
| **F8** | medium | the cosine scheduler asserted one param group, so F2's fix killed every run at startup | **fixed** |

### F1 — the kernel was never fused, and what it cost

`backend="auto"` picks flex on CUDA, but nothing compiled it: `compile_model` and
`compile_blocks` default to `False` and no config set either. Uncompiled, `flex_attention`
dispatches to `sdpa_dense` → `_math_attention_inner`, which materializes
`[4, 20, 2048, 2049]` = 336M elements per layer, applies `score_mod` elementwise under
`vmap`, prunes nothing from the block mask, and re-materializes in the backward. On CPU it
measured **1.9–3.4× slower than this layer's own naive reference path** — worse than no
kernel at all. torch **2.9** warns about this; the pinned **2.7.1** does **not**, which is
why an hour of H100 time went by in silence.

The GPU tests missed it because every one of them wrapped the layer in `torch.compile`
*inside the test*: they proved the compiled kernel works, and nothing asserted that the
shipping config reaches it.

**Fixed** by compiling `flex_attention` in `src/core/ame.py` itself, so the fast path no
longer depends on a config flag. The wrapper steps aside in two cases: when the caller is
already tracing (`compile_blocks_full`, `compile_model`) the surrounding graph lowers the
call, and on CPU the bare function is used because a compiled `flex_attention` whose
inputs require grad cannot be lowered there at all. Measured effect: **7.5×** (see the
step-time table above).

`_block_mask` is deliberately **not** marked `@torch._dynamo.disable`, and that is a
measurement rather than a preference. Skipping it does buy one graph break — on an H100
the layer takes **3** with the mask traced and **2** with it skipped, 2.1 ms against
2.0 ms per forward+backward — but it also makes the BlockMask a graph *input*, and its
sequence length then arrives as a SymInt once dynamo marks that dimension dynamic.
Inductor cannot lower `flex_attention` with it (`args[8]: (s4,)`), and two GPU tests go
red. Reproduced twice with a control, jobs 32368 and 32372; a standalone probe does
*not* reproduce it, so check this against `pytest -m gpu` rather than a script. The mask
is cached per `(seq_len, offset, device)`, so after the first call it is a dict lookup
anyway.

> **The test that was supposed to catch this did not, at first.** The argument for
> asserting on memory rather than on which callable ran was right — an identity check
> passes against a wrapper that still falls back — but the first version measured seq 512
> against 1024 without a warmup and passed on the unfused path. Two reasons, both
> measured on an H100: the first call compiles the kernel and the compiler's workspace
> swamps the reading (cold, seq 512 measured *larger* than 1024, 79.4 vs 38.1 MiB), and
> below 1024 the quadratic term does not dominate, so the unfused path also grows ~2.0×.
> The test now warms up and measures 1024 → 2048, where fused is **1.99×** and unfused is
> **3.75×** (60.6 vs 1097.4 MiB). Verified both ways: it passes on the fixed tree and
> fails on a tree with the fix reverted.

### F2 — why weight decay was confounding the experiment

AdamW's decay is *decoupled*: `p ← p·(1 − lr·wd)` runs whether or not there is a gradient.
Measured with gradients forced to exactly zero, 2000 steps at lr 2⁻¹²: `alpha[0]`
0.8038 → 0.7655, `tau[0]` 8.0000 → 7.6184. Cosine-corrected, ×0.51 by 50k steps.

Two readouts broke. `ame_experiments.md` §8.1's kill criterion is that `α₂, β₂ → 0` means
the model is declining KV shifting — decay drives them there by itself, hardest in the head
whose gradient is weakest, which is the head indifferent to the shift. And `tau` shrinks
while `slopes` stays frozen at `tau_init`, so every head's horizon `tau/m` contracts
silently (the flattest h20 head 8192 → ~4190 by 50k steps).

Fixed with `no_weight_decay_parameters` on the module and `parameter_groups` in
`src/core/model.py`, used at **both** optimizer sites in `main.py` — including the
`only_weights` resume path, which rebuilds the optimizer and would otherwise restore decay
on resume. A model declaring nothing gets exactly one group holding element-for-element
`model.parameters()`, so **every gqa baseline stays bit-identical**.

### Fix-pass progress

| phase | state |
|---|---|
| **1 — the kernel (F1)** | **done.** Verified on an H100 (jobs 32359/32362/32363/32364) and re-measured at h20 (job 32360). |
| 2 — weight decay (F2) | **done**, instrumentation included and now complete. 2.6 (banks) is an open decision. |
| 3 — the revert (F4) | **done.** Verified by composing all 11 runnable configs with and without the hunk: resolved output identical for every one. |
| 4 — cleanup (F3, F5, F6, F7) | **done.** |

**Every new behaviour was mutation-checked** — the fix reverted, the test confirmed
failing, then restored. 11 mutations, all FAIL→PASS: `_shift1`'s pad tuple, the `None`
guard, `_TYPE_OPTIONS`, the `no_weight_decay_parameters` declaration, the `parameter_groups`
split, `instrumentation()`, the scheduler guard, the trainer wiring, the heavy-metric gate,
the duck-type filter, and `isinstance` → `type(p) is nn.Parameter` (which breaks *only* the
sharded test, which is why that test exists).

**Not verifiable on CPU:** anything in Phase 1; the `full_tensor()` collective in
`instrumentation()` at world size > 1 (uniform because `log_metrics` runs on every rank
with the rank gate inside the logger — reasoning, not measurement); and whether the decay
exemption changes what `α₂` converges to, which is a Stage-0 question.

### The baseline moved under this work

On 2026-11-03 another agent rewrote `configs/mfu_audit/h20.yaml`: it is no longer the
stage-3 audit config but "the measured-best single-GPU configuration" at **46.4% MFU**,
with `compile_model: true`, `compile_loss: true`, `autocast_bf16: true`,
`distributed: null` and grad-accum 32. A new **`configs/mfu_audit/h20_mgpu.yaml`** carries
the FSDP2 + `compile_blocks_full` recipe. Three h20 baselines now exist — say which one
you mean.

For Phase 1 this means: **`h20_mgpu.yaml` is the better template**, because AME needs
FSDP2's sharding and bf16 parameters and cannot simply take `distributed: null` at 3.16B
parameters. And `compile_blocks_full` compiles the whole `TransformerBlock`, which puts
`_attend_flex` inside a compiled region and **may make the Phase 1.1 fix redundant at
h20** — check that before building both.

## Deliberately not built

The decode system: the vocab² table, the Zipf row cache, the inverted index, the V-cache,
the CPU-async path. `ame.md` §0 is emphatic that building it first is the single most
likely way to waste a month, and nothing in Stages 0–1 needs it. Also absent: Option A
(bank values + gate), the p-tap key window, and any experiment harness.

## Open items, in the order they matter

1. **Stage 0 — does the mechanism exist?** The six-arm synthetic induction ablation
   (A1–A4 free/pinned, plus the two positive controls), with the bigram mapping
   **resampled per sequence** — a fixed mapping makes both arms pass and measures nothing.
   Agreed form: a standalone script. The arms are reachable from config today
   (`pin_alpha` / `pin_beta`). Its two prerequisites — the weight-decay fix and the α/β
   instrumentation — are both done, so nothing blocks it.
2. **Report the regularization asymmetry.** 2.6 is decided — the banks are exempt from
   weight decay (`archive/ame_review_and_fixes.md` 2.6) — which means AME's banks and the dense baseline's
   `W_Q`/`W_K` are no longer treated alike. `baselines.md` asks for matched treatment, so
   this has to be said out loud in any comparison rather than left in the config.
3. ~~**Attribute the remaining 2.5× step-time gap**~~ — **done 2026-11-05.** This became
   the MFU program; the attribution, and what came of it, are in
   [ame_optimization_report.md](ame_optimization_report.md). Original wording:
   — micro-batch 4 vs the baseline's 16,
   dense bank gradients, and FSDP2's per-micro-batch reduction. One short run with
   `trainer.profile_timers=true` separates the last two; the first needs either more
   memory headroom or an explicit decision to compare at equal micro-batch.
4. **Bank-size levers**, before any run above H4 (see the table above).
5. **`d_bank < d_head`** is accepted as a constructor argument (`dhead=`) but untested.

## The h20 dense baseline next to AME: architecture and active parameters

*Written 2026-11-05, from the evening micro-batch sweep on entropy. Every run, with job
IDs, is in [ame_microbatch_sweep.md](ame_microbatch_sweep.md).*

**The like-for-like dense model is the H-balanced h20 with `gqa` attention.** It has the
same feed-forward, shape, embeddings and training recipe as AME; only the attention layer
differs. No dense config file exists. The dense runs are the AME configs plus two
overrides: `common.attention.type=gqa`, and `trainer.mem_parallel_size=null`, because
that is the bank knob and the trainer rejects it when there are no banks. The sweep's
bases differ from [configs/ame/h20_1gpu.yaml](../../configs/ame/h20_1gpu.yaml) and
[h20_4gpu.yaml](../../configs/ame/h20_4gpu.yaml) only in run names and tags, so the best
dense points reproduce from those two files:

```bash
./.pixi/envs/default/bin/python run_exp.py --config-path configs/ame --config-name h20_1gpu \
    common.attention.type=gqa trainer.mem_parallel_size=null \
    trainer.gradient_accumulation_steps=32 trainer.profile_timers=true \
    infrastructure.metric_logger.name=dense_h20_1gpu
# 4 GPUs: --config-name h20_4gpu with the same overrides minus gradient_accumulation_steps
# (already 8, i.e. micro-batch 16), named dense_h20_4gpu.
```

| h20 | dense, H-balanced `gqa` (best of the sweep) | AME, `configs/ame/h20_1gpu.yaml` / `h20_4gpu.yaml` |
|---|---|---|
| MFU, 1 GPU | 44.98%, micro-batch 16 (job 33129) | 35.45%, micro-batch 4 (job 33051) |
| MFU, 4 GPUs | 44.47%, micro-batch 16 (job 33107) | 34.91%, micro-batch 16 (job 33067) |
| Attention | GQA: 20 query heads, 10 KV heads, head dim 64; RoPE (base 10000, no Llama scaling); causal; PyTorch SDPA | Q and K gathered from per-layer banks indexed by token id; all 20 heads for K/V (`kv_heads` ignored); ALiBi slopes, no RoPE ([above](#positional-encoding-and-what-a-dense-comparison-must-control)); KV shift and a learned sink; `flex_attention` |
| Feed-forward | SwiGLU (`GatedFeedForward`): 3 bias-free matrices, dff 5120 = 4 × dmodel (Llama would use ~8/3) | identical |
| Shape | 20 blocks, dmodel 1280, vocab 50304, RMSNorm before each sublayer, untied input and output embeddings | identical |
| Total params | 620,350,720 | 3,163,200,720 |
| **Active params per token** | **620.35M** (all of them) | **587.69M** |
| GFLOPs/token (MFU accounting) | 3.965 | 3.768 |

At h20, AME runs at 78.8% (1 GPU) and 78.5% (4 GPUs) of the dense MFU. Total params and
GFLOPs/token come from the `Model LLM created with … trainable parameters` and
`MFU accounting` log lines of jobs 33129 and 33051.

**Active parameters** are counted the way MoE papers count experts: a parameter is active
for a token if that token's forward pass reads it. The banks are lookup tables, so a token
reads one row of the Q bank and one of the K bank per layer (2 × 1,280 values, 51,200
over the 20 layers), and the other 2.58B bank parameters are idle for it. Earlier
tokens' keys reach a query as cached activations, as in dense attention, so each row is
counted once, for the token that reads it. The input embedding is counted in full, as
MoE papers usually do. Counting it as one row too, the same way as the banks, gives
**555.96M dense vs 523.30M AME**.

**Why AME has 32.66M fewer active parameters.** Per layer it drops the Q and K
projections (−2.46M: 1280×1280 + 1280×640), but its V projection is twice as wide, 20
heads instead of 10 (+0.82M). That nets −1.64M per layer and −32.77M over 20 layers, of
which 0.10M of bank rows, temperatures, shift coefficients and sinks win back a little.

**Both totals check exactly against the logs.**

| per layer | dense | AME |
|---|---:|---:|
| Q/K: projections (dense) or banks (AME, 2 × 50304 × 1280) | 2,457,600 | 128,778,240 |
| V, O | 2,457,600 | 3,276,800 |
| feed-forward | 19,660,800 | 19,660,800 |
| norms | 2,560 | 2,560 |
| `tau` (20), shift α/β (`[20, 2]` each, 80), sinks `k_sink`/`v_sink` (2,560) | — | 2,660 |
| **layer total** | **24,578,560** | **151,721,060** |

Both models also hold 128,779,520 outside the blocks: the embedding (64,389,120), and the
head (64,389,120) with its final norm (1,280). So 20 × 24,578,560 + 128,779,520 =
620,350,720, and 20 × 151,721,060 + 128,779,520 = 3,163,200,720.

**The 45.57% `gqa` baseline earlier in this note is a different model.** The step-time
table in [Step time, after the kernel was fixed](#step-time-after-the-kernel-was-fixed),
and the "Which baseline" paragraph after it, quote `mfu_audit/h20_mgpu.yaml` at 45.57%.
That config's measured table was taken with the GPT-2-arch overrides: GELU feed-forward
(2 matrices), MHA (`kv_heads = q_heads`) and 25 blocks, at 4.122 GFLOPs/token. Its own
numbers confirm it: 109,361 tokens/s × 4.122 GFLOPs is 45.6% of 989 TFLOP/s, and
× 3.965 would give 43.8%. The parameter count cannot tell the two apart: 25 such blocks
hold about as much as 20 H-balanced ones, ~620M either way. For AME vs dense, quote the
H-balanced numbers above, or B2
([configs/mfu_audit/h20_hbal_4gpu.yaml](../../configs/mfu_audit/h20_hbal_4gpu.yaml),
44.44% at 4 GPUs), which this sweep reproduces at 44.47%.
