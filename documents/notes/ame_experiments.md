# AME — experimental program

The empirical companion to the design record and the baseline standard. Both live
**outside this repo**, in `~/code/research/ame/` — `ame.md` and `baselines.md`. Those two
say what the method *is* and what a fair table looks like. This file says **what we are
actually trying to find out, in what order, with what controls, and what result would
make us stop.**

Written to be read before any compute is spent. If you read only two sections, read
**§2 (the claims, as falsifiable statements)** and **§3 (Week 0 — the measurements that
need no model)**.

**Status: proposed — still unrun as of 2026-11-05, and still the live plan.** Every
number quoted is an estimate from `ame.md`, carried here so that a measurement can
contradict it.

Nothing in the 2026-11-04/05 efficiency work touched this program: those runs measured
**throughput**, not quality, so "nothing here has been run" is still literally true. Its
two prerequisites — the weight-decay fix and the α/β instrumentation — are both done
(`archive/ame_review_and_fixes.md` Phase 2), so **Stage 0 is unblocked**. This is the
next thing to do with AME.

**Contents**
0. [Compressed method reference](#0-compressed-method-reference)
1. [The shape of the program](#1-the-shape-of-the-program)
2. [The claims under test](#2-the-claims-under-test)
3. [Week 0 — what needs no model](#3-week-0--what-needs-no-model)
4. [Stage 0 — does the mechanism exist](#4-stage-0--does-the-mechanism-exist)
5. [Stage 1 — does it scale as a language model](#5-stage-1--does-it-scale-as-a-language-model)
6. [Stage 2 — does the deployment claim survive](#6-stage-2--does-the-deployment-claim-survive)
7. [Instrumentation: log these from the first run](#7-instrumentation-log-these-from-the-first-run)
8. [Silent-corruption traps](#8-silent-corruption-traps)
9. [Risk register](#9-risk-register)
10. [Decision tree](#10-decision-tree)
11. [Reporting standard](#11-reporting-standard)
12. [Open questions, mapped to stages](#12-open-questions-mapped-to-stages)

---

## 0. Compressed method reference

Enough to make this file standalone. The authoritative version is `ame.md` §0 and §7.

Per layer ℓ, per head h: two banks `E_Q, E_K ∈ R^{|V|×d_head}` indexed by **token id**,
replacing the Q and K projections. Values stay an ordinary projection.

```python
q = E_Q[t]; k = E_K[t]            # gathers, NOT x @ W_Q / x @ W_K
v = (x @ W_V).view(B,n,H,D)       # unchanged from MHA   ("Option B" — the method)
q = l2_normalize(q, -1)           # QK-norm on the BANK ROWS, before the mix
k = l2_normalize(k, -1)
k = a1*k + a2*shift1(k)           # KV shifting: 4 learnable scalars per head
v = b1*v + b2*shift1(v)
s = einsum('bihd,bjhd->bhij', q, k) * tau   # tau [H], init sqrt(D) ≈ 11.3
s = s - m * dist                  # ALiBi, per-head slope — the ONLY positional signal
# ... causal mask, softmax, Σ w·v, W_O — all standard
```

Score form: `s_ij = τ·T[t_i,t_j] − m·(i−j)` with `T[a,b] = Ê_Q[a]·Ê_K[b]`. Scores depend
on **ids and distance only** — that is the whole design, and the source of both the
deployment win and every weakness.

Reference config for all numbers: 1.3B backbone · d_att = d_model = 2048 · L = 24 ·
16×128 heads · |V| = 50k · fp16 · 60 GB/s desktop DDR5 · 20–30 ms/token budget.
Bank parameters: **4.9B / 9.8 GB fp16, ≈ 4× the backbone.**

Settled, not under test here: Option B over Option A · KV shifting on, as learnable
per-head scalars · the ALiBi bias is non-optional · QK-norm on rows · training is
unmodified O(n²) attention. Option A (bank values + gate) appears below only as a
deliberate ablation (§5, E1.3).

---

## 1. The shape of the program

Three stages plus a week of model-free measurement. Each stage exists to kill the method
cheaply if it is going to die.

| | question | cost | kill criterion |
|---|---|---|---|
| **Week 0** | is the deployment claim arithmetically possible on real hardware and a real corpus? | days, no GPU | row-cache hit rate so low that the score path cannot fit any budget (§3) |
| **Stage 0** | does KV shifting give AME in-context copying **at all**? | hours, 1 GPU | pinned `α₂=β₂=0` scores the same as learned — the mechanism is inert |
| **Stage 1** | does it hold up as a language model against matched baselines? | the real expense | gap does not shrink with scale, or rare-type rows stay at init |
| **Stage 2** | does the decode system survive memory bandwidth? | engineering | score path cannot be brought under budget by any of §6's routes |

**Stages 0 and 1 need none of the decode machinery** — no table, no row cache, no
inverted index, no CPU path. Training is plain O(n²) attention with two gathers. Do not
build the decode system first; `ame.md` §0 is emphatic about this and it is the single
most likely way to waste a month.

Week 0 is listed first because it is cheap, needs no model, and several of its outcomes
change what Stage 1 is for. It can run concurrently with Stage 0.

---

## 2. The claims under test

The method rests on five load-bearing claims. Stated as falsifiable predictions, each
with the cheapest experiment that could break it. **This table is the spine of the
program**; everything in §4–§6 is an elaboration of one row.

| # | claim | prediction if true | breaks if | tested by |
|---|---|---|---|---|
| **C1** | KV shifting restores in-context copying to an id-scored attention layer | a 2–4 layer AME solves resampled-bigram induction; a pinned-coefficient twin does not | both arms score alike, or α₂ reliably decays to 0 | **E0.1** (§4) |
| **C2** | contextual values (B) buy something static values (A) cannot | B beats A on sense/topic tasks at matched scale, independent of the shift | A matches B everywhere except copying | **E1.3** (§5) |
| **C3** | bank parameters are nearly free at training and entirely free at inference FLOPs | at matched training FLOPs + tokens, AME beats dense; the gap to a matched-*parameter* dense model is the honest cost | the FLOP-matched win is smaller than the step-time regression it cost | **E1.1** (§5) |
| **C4** | per-token-type capacity is adequate at realistic vocab and data scale | loss stratified by token frequency shows no cliff in the tail; tail rows move from init | tail rows stay at init, or tail loss is flat vs a dense twin | **E1.2** (§5) |
| **C5** | attention can be removed from the accelerator's critical path within a 20–30 ms budget | measured batch-1 decode at 100k lands at 11–18 ms with the row cache in place | the score path cannot be brought under budget (§6) | **E0.3** (§3), **E2.2** (§6) |

A sixth statement is not a claim but a **standing threat to the entire design**:

> **C0 — the softmax may not be earning its cost.** The ALiBi decay factorizes, so the
> only thing forcing a growing cache is the kernel `exp(τ·T)`. If that matrix has low
> numerical rank, AME collapses to `S_i = e^{−m}S_{i−1} + ψ(t_i)v_iᵀ` — RetNet/GLA with
> id-indexed keys, constant state, no V-cache, no top-k, no CPU offload (`ame.md` §5.2).
> The whole of §6 would then be machinery preserving a kernel that does not need
> preserving. **One SVD on a Stage-1 checkpoint answers it** (E1.5). A reviewer asks this
> in the first five minutes; have the number.

Everything else in `ame.md` §8 is downstream of these six.

---

## 3. Week 0 — what needs no model

Six measurements, no training, no checkpoint. Together they decide whether C5 is
arithmetically reachable before a single GPU-hour is spent. Each quotes the estimate from
`ame.md` so that the measurement can contradict it.

**E0.1 — Vectorized `exp` throughput.** A decode step needs `k·H·L` = 2048·16·24 =
**786,432** exponentials. Estimate: scalar `libm` on one core ≈ 16 ms (the entire
budget); AVX-512 across 8 cores ≈ 0.05 ms. *Measure both, and check which math library
you actually linked.* Pass: < 0.5 ms. This is the cheapest possible measurement and it
guards the one cost nobody looks for.

**E0.2 — Random-gather bandwidth.** The 200 MB of top-k value reads are **786k
independent 256-byte gathers**, not a stream. Estimate: 50–70% of peak → 5–7 ms on
60 GB/s desktop DDR5, not the 3.4 ms a stream would give. Measure the real derate on the
target machine, at the real record size, from a ≥10 GB region. Repeat at batch 2, 4, 8 —
B's reads scale **linearly**, and the estimate says batch 8 is 26.8 ms of gathers alone.

**E0.3 — Row-cache hit rate under the real token distribution.** *The highest-leverage
measurement in Week 0*, because the single-digit-ms claim rests entirely on it. Tokenize
a corpus sample with the production tokenizer, take the sequence of query-token ids, and
simulate an LRU over pruned rows at cache sizes 1000 / 3000 / 10000 types. Estimate:
~75% hit at 3000, ~86% at 10k; amortized cost ≈ 3.4 + 0.25·32 ≈ **11 ms**, not 3.4.
A miss costs the full ~2.0 GB of bank reads (≈ 33 ms desktop). Report the hit-rate curve,
and from it the amortized ms/token at each cache size — alongside the storage: 1000 types
at k = 2048 is **3.1 GB**, 3000 is **9.4 GB**, 10k is 31.5 GB, the full vocabulary
**157 GB**.

**E0.4 — `|U|(n)` and distinct-bigram counts on the real tokenizer.** Both dominant cost
terms are driven by these and both are currently Zipf estimates. Measure, as a function
of context length n, (a) distinct token types `|U|` and (b) distinct bigrams, on real
text with the production tokenizer. Estimates to beat: `|U|` ~0.6–1k at n = 2k, ~5–10k at
32k, ~10–20k at 100k; distinct bigrams ~50–80k at n = 100k, growing roughly linearly.
`|U|` sets the score-path cost (`|U|·d_att·L·2B`); the bigram count sets the inverted
index, which under the key shift is **per head-layer and unshareable across heads** —
estimated 12.6M scored bigrams/token at 32k and 38.4M at 100k.

**E0.5 — PCIe and NVMe.** Sustained GPU→RAM bandwidth (estimate ≈ 25 GB/s, about 80% of
the 31.5 GB/s spec; prefill must ship 3.2 GB at 32k context ≈ 130 ms) and random 8 KB
NVMe read latency at queue depth ~384 (estimate 0.5–1 ms for the whole-vocabulary row
cache, which would make the 157 GB option real rather than absurd).

**E0.6 — Score scale at initialization.** One forward pass of an untrained model; no
optimizer. Confirm that per-element σ ≈ 1 on the bank rows gives scores of std ≈ 1, and
that the habitual embedding-table init of σ = 0.02 gives std ≈ 4e−4 — uniform attention
that may never break symmetry. With QK-norm the sensitivity should mostly vanish; verify
that too, because it is the second reason to keep QK-norm.

> **Deliverable:** a one-page table of six measured numbers against six estimates, and a
> revised batch-1 decode budget. If E0.3 comes back far below estimate, say so loudly —
> it moves the score path from "engineering risk" to "design problem", and Stage 1 should
> then be framed as a quality result rather than a deployment one.

---

## 4. Stage 0 — does the mechanism exist

**Scale:** 2–4 layers, vocab ~1k, context 256. Hours on one GPU.

**Purpose:** C1 only. Not a quality result, not a baseline comparison. None of
`baselines.md`'s baselines apply at this scale and none should be run here — if the
mechanism is inert, the Pareto frontier never comes up (`baselines.md` §0).

### E0 task design

Synthetic in-context copying: repeated random bigrams plus needle retrieval.

> **The trap that makes this experiment worthless if you miss it.** Resample the bigram
> mapping **per sequence**. A model whose scores are a trained function of token-id pairs
> is a machine for memorizing bigrams — that is exactly what `T` is — so with a mapping
> fixed across the dataset *both* arms pass and the experiment measures nothing.
> In-context copying means the mapping is new in every sequence.

Readouts, two not one:
1. **Task metric** — next-token accuracy at the second occurrence of a bigram's first
   element, and exact-match on needle retrieval.
2. **The learned coefficients themselves** — trajectories of α₂, β₂ per head over
   training. If they drift to zero the model is *declining* the mechanism, which is a
   publishable result and a different finding from "the mechanism doesn't help". Keep
   these scalars in fp32.

### E0.1 — the four-arm ablation

Two arms are not enough. α (key-side mix) and β (value-side mix) are **different
mechanisms, and only one of them is irreplaceable**: a AME head with a steep enough ALiBi
slope *is* a previous-token head, so after one layer `x_j` already contains `t_{j−1}` and
a layer-2 head reading `v_j` gets the β shift for free. What no amount of depth supplies
is α, because the score `T[t_i,t_j]` still cannot read `x_j`. Pinning both together
confounds them.

| arm | α | β | what it isolates |
|---|---|---|---|
| **A1** | free | free | the method as specified |
| **A2** | free | pinned 0 | key-side mix alone — the irreplaceable half |
| **A3** | pinned 0 | free | value-side mix alone — partly reachable without the shift |
| **A4** | pinned 0 | pinned 0 | the fixed-kernel floor |
| **P1** | — | — | **positive control**: vanilla 2-layer transformer, no shift. Must solve the task via the standard previous-token + induction circuit. If it does not, the task or the harness is broken, not AME. |
| **P2** | free | free | **harness control**: *standard* attention, one layer, with KV shifting. Xu et al.'s Theorem 2 says this represents induction exactly. If it fails, fix the harness before reading anything else. |

**Control the ALiBi slope ladder across all arms** — hold it identical, or the β result
is contaminated by whether a steep-slope head happened to be available. If α₂ collapses
to zero in A1/A2, run the disambiguating 2×2: {shift on, off} × {steep heads present,
absent}. A collapse that disappears when steep heads are removed means the model was
getting the shift elsewhere; a collapse that persists means it genuinely declined.

**Kill criterion:** A4 scores the same as A1. The mechanism is inert in AME and the
fixed-kernel ceiling is real — go to §10's fallback branch.

**What a pass buys:** C1 holds, and AME has an answer to the fixed-kernel ceiling. It
does *not* buy content-addressed retrieval (see §9, R1) — it buys the induction/copy
primitive, which is narrower.

### E0.2 — key-window width (cheap, do it here)

p-tap key mix, p = 1, 2, 3. Costs nothing to store, so run it — but **expect nothing**.
A p-tap mix is an *additive* mix of p copies of one bigram table, not a p-gram table: the
logit carries no joint term, only a separable rank-1 product in weight space. An earlier
draft of `ame.md` called this a trigram table and built a hypothesis on it; that
motivation does not survive, so Xu et al.'s negative result on wider windows is now the
**expected** outcome rather than one that might not transfer. The interesting version of
this question — a genuinely *multiplicative* mix that can score `(t_j, t_{j−1})` jointly
— is not cheap and is out of scope here.

---

## 5. Stage 1 — does it scale as a language model

**Scale:** 100–300M backbone, 10–30B tokens. The real expense. Baselines per
`baselines.md`; reporting standard per §11 below.

### E1.1 — the headline comparison (C3)

Matched on **training FLOPs and tokens**, with one hostile matched-**total-parameter**
row (a ~4× larger dense model, or a PKM/Engram model with a comparable bank). Both are
necessary: the claim under test is that bank parameters are nearly free at training and
free at inference FLOPs, and the matched-parameter row is what a reviewer will ask for.
**If AME wins at matched FLOPs and loses at matched parameters, say so — that is a
result**, and it is the PKM framing.

Report FLOPs *and* seconds. AME is ~15% cheaper per token in FLOPs than its dense twin
(deleting W_Q/W_K removes 0.2B params ≈ 0.4 of ~2.6 GFLOPs in a 1.3B forward) but the
backward scatters gradients into 4.9B bank parameters. **Expect fewer FLOPs and worse
wall-clock**; a step-time regression needs stating out loud, not hiding inside a
matched-FLOP table.

### E1.2 — the Zipf tail (C4) — *the most likely place this method fails scientifically*

4.9B bank parameters over 50k types is ~**98,304 parameters per token type**, spread over
**768 rows** of width d_head (a q-row and a k-row in each of 384 head-layers). A type seen
10³ times in training gets ~100 parameters per occurrence; the tail is far worse and those
rows will be noise.

Three measurements, all from a single Stage-1 run:
1. **Loss stratified by training-frequency decile of the target token.** Compare against
   the dense twin's same stratification. A cliff in the tail that the dense model does not
   have is C4 failing.
2. **Row movement from init**, bucketed by frequency: `‖E[w] − E_init[w]‖ / ‖E_init[w]‖`.
   Rare-type rows that have not moved are dead capacity.
3. **Bank utilization**: fraction of rows whose norm moved beyond a threshold.

Interventions to test if the tail bites (each is also a scaling-law question in its own
right): tie or cluster rare types · factorize rows to rank r < d_head · share banks across
layers · tie a bank to the input embedding · give banks their own d_bank < d_head ·
regularize rows toward a frequency-dependent prior. **The joint scaling law — quality as a
function of bank parameters and tokens-seen-per-type — is the cleanest scientific
contribution in the project** and the one closest to existing work on PKM scaling.

### E1.3 — Option B vs Option A (C2)

At matched scale, on tasks needing **in-context sense** rather than copying:
word-sense disambiguation, topic/register tracking, "how was this token being used
earlier". Not copying tasks — those are E0.1's job, and A cannot do them by construction.

Two scoping notes that make this experiment interpretable:
- **B's advantage is exactly zero at layer 1.** `v_j = W_V·E[t_j]` is a pure function of
  `t_j`, so layer 1 of B *is* Option A with `E_V := W_V·E_emb`. The advantage grows only
  with depth; a 2-layer probe is the minimum where the claim is even true.
- **Evaluate A without the key shift**, or re-derive its cost columns. Under a key shift
  A's summands become distinct *bigrams* rather than types, its saturation disappears, and
  its decayed counts must be kept per bigram — O(n) state, which removes the entire reason
  A was attractive.

### E1.4 — length generalization

You will train at 2k–8k and deploy at 100k. ALiBi extrapolates, but the decayed-count
statistic (A) and top-k selection (B) are long-context machinery that **never ran during
training**. Validate at ~10× the training length before trusting any deployment number.
Specifically check that the stretched slope ladder (2⁻¹ … ≈2⁻¹⁴, horizons ~20 to ~185k
tokens) behaves as designed — the standard ALiBi ladder for H = 16 (2⁻⁰·⁵ … 2⁻⁸) puts the
longest-range head at ~3k tokens, which would cap the method far below its own pitch.

Watch for the **attention-sink eviction** failure: AME will likely learn a sink (an `E_K`
row scoring high against everything, almost certainly BOS), and `−m·(i−1)` kills it at long
context. That is precisely StreamingLLM's documented failure mode, on a method that cites
StreamingLLM as the floor it must beat. A position-independent learned sink key is the
cheap insurance — the same key the position-1 note in `ame.md` §7 asks for.

### E1.5 — the rank of `exp(τ·T)` (C0) — *run this the day the first checkpoint lands*

One SVD on a sampled submatrix of `exp(τ·T)` per head-layer. No new training run.
Report the numerical rank spectrum, per layer and per head.

Why it matters: `T` is rank-d_head by construction, but elementwise exponentiation raises
rank combinatorially (`exp(τT) = Σ_p (τT)^{∘p}/p!`, and the p-th Hadamard power has rank up
to `C(d_head+p−1, p)`), so at τ ≈ 11 the effective rank may be too large to help. But the
break-even is generous: a rank-r recurrent state costs `r·d_head·H·L·2B` ≈ **98 MB/token at
r = 1000** (~1.6 ms desktop), ~0.8 GB at r = 8000, against the ~2.0 GB/token the exact path
spends on scores alone.

If the rank is a few thousand, **AME has an exact recurrent form with constant state** and
the project should pivot (§10). AME is unusually easy to linearize post hoc, because its
key set is finite and enumerable — `exp(τ·T)` is one fixed `|V|×|V|` matrix per head-layer,
an offline matrix problem rather than the open-ended function approximation that makes
LoLCATs-style linearization hard.

---

## 6. Stage 2 — does the deployment claim survive

Built **against a finished Stage-1 checkpoint**, not before. This is where the table, row
cache, inverted index and CPU path get written.

### E2.1 — the score path, which is the binding cost

> The cost model in an earlier version of `ame.md` priced only the value reads. Obtaining
> the row `T[t_i,·]` costs `|U|·d_att·L·2B` ≈ **2.0 GB/token at |U| = 20k ≈ 33 ms on
> desktop DDR5** — ten times the 200 MB of top-k value reads, and over the whole
> 20–30 ms budget by itself. It is a matvec at 1 FLOP/byte, so bandwidth-bound; more cores
> do not help. **The same cost falls on both variants** — it is a score-path cost, not a
> value-path one.

Three routes, none measured. Measure all three and report the Pareto among them:
1. **ANN/IVF over the key bank.** "Top-k of a 50k-row table under a query vector" is a
   textbook nearest-neighbour problem, the bank is static, and the index builds once
   offline. Estimate at 256 centroids, nprobe 4: ~266 KB per head-layer ≈ **102 MB/token
   ≈ 1.7 ms** — 20× cheaper, at the price of approximate selection. Report recall@k
   against exact, on the trained bank geometry (an untrained bank tells you nothing about
   recall).
2. **LRU pruned-row cache** keyed by query-token id. A hit reads `k·4 B = 8 KB` per
   head-layer (~3 MB/token, free); a miss costs the full ~2.0 GB. Storage from E0.3.
3. **Candidate-set capping**: always include a local window, cap `|U|` by considering only
   types seen in the last W tokens. Measure the quality cost, not just the speed.

Also here: per-head vs per-layer tables, and **quantization of rows** — entries pass
through `exp`, so sensitivity is unknown and must be measured, not assumed.

### E2.2 — end-to-end decode latency (C5)

ms/token at batch 1, on named hardware, at a stated context length, reported next to bytes
of per-sequence state. Estimate to beat: **~11–18 ms desktop batch-1 at 100k** — inside the
20–30 ms budget, but not comfortably.

Four things that will show up in the measurement and not in any FLOP model:
- **Streaming softmax needs a running max.** The CPU accumulates raw `e^{s_ij}` into
  (N, Z) and the GPU merges a self-term. Unnormalized exponentials overflow: carry a
  running max per head, ship **(N, Z, max)**, rescale on merge as FlashAttention does.
  Not optional at fp16/fp32, and the most likely first bug.
- **The per-layer handoff defeats CUDA-graph decode capture.** `(N, Z, max)` ships once per
  layer → **24 host↔device synchronizations per sequence per token**, 0.12–0.5 ms of pure
  overhead at 5–20 µs each, *and* the loss of graph capture, which is standard at this size
  and worth 2–5×. Budget for losing it, or batch the handoff across layers — which costs
  the staleness that §4.3 of `ame.md` otherwise guarantees is zero.
- **The pipeline deadline is thin.** "Async" is a *data-dependency* claim, not a timing
  one: `t_i` is known only when the GPU finishes step i−1. The real structure is a software
  pipeline over layers, `CPU_ℓ ≲ GPU_{ℓ−1}` ≈ 0.2 ms against ≈ 0.14 ms of CPU work. **A
  single row-cache miss erases that margin.** What is unconditionally true is that no
  quantity the CPU needs comes from the current step's hidden states — zero staleness,
  always exact values.
- **RAM capacity caps the batch story harder than CPU throughput does.** V-caches do not
  share: 8 sequences at 100k context need **~78 GB** of RAM. Long context and many
  sequences are alternatives on consumer hardware, not a combination.

### E2.3 — the dense→sparse deployment shift

Training uses the full causal softmax; decode uses top-k. **That distribution shift is
exactly what NSA and DSA avoid by training *with* their sparsity.** Measure it explicitly —
quality at k ∈ {256, 512, 1024, 2048, dense}, and the attention mass dropped — or adopt
sparse training. Do not assume it is free because the indexer is exact. Also decide and
report: k per head or per layer (the 200 MB figure assumes per head), and what score
pruned-out pairs receive, −∞ or 0, which changes the softmax denominator between training
and decode.

### E2.4 — the bigram index under the key shift

Top-k by token type stops working once the score depends on `(t_{j−1}, t_j)`. Index
distinct bigrams instead; it **cannot be shared across heads**, because α and the slope m
are both per-head — 384 independent selections per decode step. Use standard IR machinery
rather than a scan: per-bigram postings in recency order plus a **WAND/MaxScore** merge,
`O(k log·)` after an `O(n)` heap seed. Note that this structure *grows with context* where
the unigram version saturated, so measure at 32k and 100k, not just at 4k.

One interaction to get right: if the cached row is pruned to the top-k columns of
`T[t_i,·]`, then `T[t_i,t_{j−1}]` may be absent, because a position can enter the true
top-k through *either* slot. **The candidate set must be the union over both slots.** The
degenerate settings `α = (1,0)` and `α = (0,1)` are both safe against a unigram-pruned row;
only the mixture breaks.

---

## 7. Instrumentation: log these from the first run

Cheap to add at the start, impossible to recover afterwards.

**Mechanism diagnostics**
- `α₁, α₂, β₁, β₂` per head per layer, every N steps. The headline readout of E0.1, and the
  thing that tells you whether the model accepted the mechanism at Stage 1 scale too.
- Learned temperature `τ` per head; ALiBi slopes if the learnable-slope ablation is on.
- **Effective horizon per head**, `τ/m`: with QK-norm the scores live in ±τ ≈ ±11, so a
  token at distance Δ competes only while `m·Δ ≲ τ`. Log the horizon ladder; it is how you
  catch a slope ladder that silently caps context.

**Health**
- Attention entropy per head, and max softmax probability. Catches saturation and dead
  heads, the documented consequence of skipping QK-norm.
- Score std at init and after warmup (E0.6's quantity, tracked).
- Fraction of attention mass on the top-k positions, at several k — this is E2.3's shift,
  measurable during *training* at no cost, long before Stage 2.
- Whether a sink row emerges in `E_K` (one row with high score against everything), and
  what happens to its mass as context grows.

**Capacity**
- Per-frequency-bucket row norms and movement-from-init (E1.2).
- Bank gradient density per step — the estimate is 82% of rows touched at Zipf s = 1.0, 67%
  at s = 1.1, 50% at s = 1.2 for a 0.5M-token batch. Worth confirming on the real corpus,
  because it is the reason dense optimizer state is unavoidable.

**Cost**
- Step time and FLOPs separately (E1.1 needs both).
- Peak memory split: backbone / banks / optimizer state / activations. Banks alone are
  **79 GB** at 14 B/param (fp16 weights + fp32 master + Adam m,v + fp16 grads), **~88 GB**
  with fp32 gradient accumulation, before the backbone.

---

## 8. Silent-corruption traps

Each of these produces a *plausible but wrong* result rather than a crash. They are the
reason a null result here should be re-checked before it is believed.

1. **`roll` wraps, and document packing turns that into a correctness bug.** Zeroing "the
   first position" is correct for one sequence per row. Under packing — near-universal in
   pretraining and required by varlen FlashAttention — token 1 of document 2 mixes in the
   last token of document 1. **Zero the first position of every document, from
   `cu_seqlens`.** This surfaces only as "the shift didn't help", which is exactly the
   conclusion E0.1 is trying to reach honestly.
2. **A fixed bigram mapping in Stage 0.** Both arms pass; the experiment measures nothing.
   Resample per sequence (§4).
3. **Embedding-habit initialization.** σ = 0.02 gives scores of std 4e−4 — attention
   uniform at init, symmetry possibly never broken. Use σ ≈ 1 without QK-norm; with QK-norm
   the sensitivity mostly disappears.
4. **QK-norm in the wrong place.** Normalize the **bank rows, before the mix** — not the
   mixed key `k̂`. Normalizing `k̂` makes its norm a function of the bigram `(t_{j−1}, t_j)`,
   so the score stops being a linear combination of two entries of one row and the zero-cost
   key mix is lost. `ame.md` §7 calls this "the single highest-cost ambiguity in the design".
5. **Inverting the `v̂` recursion.** The V-cache holds `v̂_j`, so raw `v_{j−1}` is not in it.
   Do **not** recover it from `v̂_{j−1} = β₁v_{j−1} + β₂v_{j−2}` — recursive to j = 1,
   unstable, and undefined at β₁ → 0, which is exactly the fully-shifted head you want
   reachable. Keep a one-step raw-`v` register per layer per head on the GPU:
   `L·d_att·2B` ≈ **98 KB** total, free, since the GPU just computed it.
6. **Position 1 under the mixture.** `k̂_1 = α₁k_1` is finite, but a head that learns
   `α₁ → 0` gets logit 0 — neutral and finite, and able to dominate a short prefix. Add an
   explicit learned null/sink key. (Implement the *mixture*, never the hard shift: under
   `Σ_{j≤i−1}` the hard shift gives an empty softmax at i = 1, i.e. NaN.)
7. **Scalar `exp` at decode.** 16 ms hiding inside a 20–30 ms budget, in the one place
   nobody profiles (E0.1).
8. **Missing running max in the streaming softmax.** Overflow, not a wrong answer — but it
   can look like instability attributable to the architecture.
9. **Option A's decayed counts overflow their scale factor early.** The O(1) update keeps
   counts in a common scale growing like `e^{m·i}` — fp32 overflow at **i ≈ 125** for the
   steepest head with geometric slopes, i ≈ 23k for the shallowest. The rescale fires inside
   the first few hundred tokens, not at the end of a long document. Rescale periodically, or
   keep counts in the log domain.
10. **Kernel choice is settled before you start.** The attention kernel must accept an
    additive distance bias. FlexAttention's `score_mod` is the default (no n×n tensor
    materialized); `flash-attn`'s `alibi_slopes` (v2.4+) is the fast path but takes a
    **non-differentiable** tensor, so a learnable-slope ablation needs FlexAttention. Plain
    FA2 with neither is not an option.

---

## 9. Risk register

Ordered by how much it would cost to discover late.

| | risk | when known | cheapest probe | if it fires |
|---|---|---|---|---|
| **R1** | **Fixed-kernel ceiling is binding.** `out_i = Σ_j w(t_i, t_j, i−j)·v_j` — weights never read a hidden state, so retrieval is *never content-addressed*, in any variant. KV shifting widens the id pair to a short id *window*; it does not escape ids. Exact multi-token copying, long-range needle retrieval and variable binding remain the expected failure modes, and this is where reviewers will probe. | Stage 0 for copying; Stage 1 for the rest | E0.1, then RULER/needle at Stage 1 | the hybrid fallback (§10) |
| **R2** | **Score path cannot be brought under budget.** ~2.0 GB/token, 10× the value reads. The row cache is **load-bearing, not an optimization**. | Week 0 (E0.3), confirmed Stage 2 | E0.3 — no model needed | reframe as a quality result, or pivot to the recurrent form |
| **R3** | **Zipf tail is dead capacity.** ~98k params/type over 768 rows; a type seen 10³ times gets ~100 params/occurrence. | Stage 1 (E1.2) | frequency-stratified loss on the first checkpoint | bank-size levers (§5, E1.2); this also *is* the scaling-law study |
| **R4** | **The softmax isn't earning its cost** (C0). | Stage 1 checkpoint (E1.5) | one SVD | pivot (§10) |
| **R5** | **Matched-parameter comparison is unflattering.** Banks are ~4× the backbone; W_Q/W_K deletion returns only 0.2B. | Stage 1 (E1.1) | run the hostile row deliberately | report it as the PKM framing — FLOP-free parameters, not free quality |
| **R6** | **Training-cost asymmetry.** AME trains O(n²); GDN-2/KDA/Mamba-3 train chunkwise O(n). ~15% fewer FLOPs but probably worse wall-clock from 4.9B scattered gradients. | Stage 1 | measure step time from run 1 | report honestly; it will be raised |
| **R7** | **Sharding is a prerequisite, not a tuning step.** 79–88 GB of bank optimizer state. Shard the way Megatron shards a vocab-parallel embedding (rows split, gather locally, all-reduce) rather than letting ZeRO-3 all-gather a 204 MB bank per layer per step. 8-bit or CPU-offloaded optimizer state is the other lever. | before Stage 1 starts | capacity arithmetic | plan it into the Stage-1 setup |
| **R8** | **The gain is Xu et al.'s, not AME's.** KV shifting is validated prior art to 19B (+2.1 avg, +3.6 MMLU at 2.9B/500B). | Stage 1 | the matched-mechanism baseline | mandatory, see §11 |
| **R9** | ~~**Name collision.**~~ **Resolved 2026-11-03**: renamed MAE → **AME** (Attention Memory Embedding), matching the `ame` key the attention registry already used. "MAE" collided with Masked Autoencoders (~39k citations) and "mean absolute error"; "AME" collides with neither. | — | — | done |

---

## 10. Decision tree

```
Week 0 ──► E0.3 row-cache hit rate
             ├─ good (≥ ~75% at ≤ 10 GB) ──► deployment claim stays the headline
             └─ poor ─────────────────────► reframe: AME is a quality/parameter-efficiency
                                            result; promote ScoutAttention + SparDA to
                                            primary baselines and evaluate at fixed GPU
                                            memory rather than fixed FLOPs. Decide EARLY —
                                            it changes what Stage 1 measures.

Stage 0 ──► E0.1 four-arm ablation
             ├─ A1 > A4, α₂ survives ─────► C1 holds. Proceed to Stage 1 with the shift on
             │                               and the matched-mechanism baseline mandatory.
             ├─ A1 ≈ A4 ──────────────────► mechanism inert in AME. The fixed-kernel ceiling
             │                               is real → the hybrid fallback becomes the plan:
             │                               interleave AME layers with standard-attention
             │                               layers (the Engram argument — static lookup
             │                               relieves early layers, freeing attention for
             │                               context). Re-scope the whole program around it.
             └─ A1 > A4 but α₂ → 0 ───────► run the steep-head 2×2. If the collapse vanishes
                                            without steep heads, β was being supplied
                                            elsewhere and only β is doing work — report that;
                                            it weakens the "only route to copying" claim.

Stage 1 ──► E1.5 rank of exp(τ·T)
             ├─ rank large ───────────────► the softmax is earning its cost; build Stage 2.
             └─ rank ≲ few thousand ──────► PIVOT. AME has an exact constant-state recurrent
                                            form: no V-cache, no top-k, no inverted index,
                                            no CPU offload, true O(1) decode. The method then
                                            lands INSIDE the GDN-2/KDA/Mamba-3 family rather
                                            than beside it — which changes the positioning
                                            from "new deployment regime" to "id-indexed keys
                                            as a better linear-attention parameterization".
                                            Most of §6 becomes unnecessary.
```

---

## 11. Reporting standard

Fixed in `baselines.md` §0; repeated here because every table in this program must obey it.

- **Every row carries active parameters, total parameters, training FLOPs, and tokens**,
  MoE-style. There is no single honest baseline when the banks are ~4× the backbone, and
  no single number.
- **Headline table matched on training FLOPs and tokens.** Plus **one matched-total-parameter
  row** as the hostile comparison.
- **Cost axis defined once:** ms per token at batch 1, on named hardware, at a stated
  context length, reported next to **bytes of per-sequence state**. Not FLOPs — every method
  in this comparison is bandwidth-bound, and for AME the binding cost is the score path, not
  the value path.
- **Quality means RULER and needle-in-a-haystack at the deployment length**, not perplexity.
- **Headline figure is a Pareto plot** — recall-heavy score vs per-token decode cost, every
  baseline on it — not a win/loss table. The target region (above StreamingLLM, cheaper than
  Quest, competitive with GDN-2 while offloading to CPU) is currently empty.

**The baselines**, in full in `baselines.md`:

| | baseline | role |
|---|---|---|
| 1 | **GDN-2 (2605.22791) · KDA (2510.26692) · Mamba-3 (2603.15569)** | make-or-break: matched deployment cost, and they train *cheaper* |
| 2 | **FlashAttention (2205.14135)** | reference ceiling, in every table; omitting it looks evasive |
| 3 | **Quest (2406.10774) + StreamingLLM (2309.17453)**, on our own checkpoint | the two ends of the frontier — AME must land strictly inside, or a *training-free* method dominates it |
| 4 | **LoLCATs (2410.10254)** | closest philosophical competitor |
| 5 | **KV-shifting transformer (2411.19574)** | **matched-mechanism control — mandatory** |

On (5): a 2-tap causal depthwise mix is the **short convolution** that most linear-attention
baselines already carry — `att-05` says it "generalizes the shift SSM", `att-06` (GDN)
ablates it as **crucial**, `att-07` (KDA) calls it widely adopted, `att-08` (Mamba-3) is
notable for *dropping* it. So **GDN-2/KDA are already matched on this axis**; only the
full-attention rows (2, 3) need the shift added to stay honest.

Conditional: if the story becomes "bigger contexts / smaller GPUs via CPU async" rather than
quality-per-FLOP, promote **ScoutAttention (2603.27138)** and **SparDA (2606.04511)** to
primary baselines and evaluate at fixed GPU memory. That is the Week-0 branch in §10.

One omission to pre-empt: `ame.md` §5.2 shows Option A *is* a linear-attention model
(RetNet/GLA with a one-hot |V|-dimensional key basis, a static value bank and a
query-dependent elementwise readout mask). **A's correct baseline is therefore GLA/RetNet at
matched state size, not a transformer** — and if E1.5 comes back low-rank, that becomes true
of the whole method.

---

## 12. Open questions, mapped to stages

`ame.md` §8 in priority order, with the stage that can answer each. Two are deliberately
out of priority order, marked ⚑.

| # | question | stage | experiment |
|---|---|---|---|
| 1 | does KV shifting buy AME induction? | 0 | E0.1 |
| 2 | what does contextual payload buy (B vs A)? | 1 | E1.3 |
| 3 | quality gap vs the baselines frontier | 1 | E1.1 |
| 13 ⚑ | what is the softmax actually worth? *cheap, early, could redirect the project* | 1 checkpoint | E1.5 |
| 6 ⚑ | the score path at decode — *top engineering risk*, placed late only because it needs Stage 2 | Wk 0 + 2 | E0.3, E2.1 |
| 4 | distance-bias design: per-head slopes, learned per-type decay, how selection interacts with the bias (top-k over `T − m·Δ` favors recency — verify distant high-score tokens still get retrieved) | 1 + 2 | E1.4, E2.1 |
| 5 | table fine-tuning axis — pruned tables only: 0.2 GB/head-layer at k = 2048, but **78.6 GB over all 384 head-layers, ~157 GB with key indices**, plus optimizer state. The practical version is "a few head-layers". | 2+ | — |
| 7 | top-k policy and the dense→sparse shift | 1 (diagnostic) + 2 | E2.3 |
| 8 | key-window width p | 0 | E0.2 |
| 9 | bank-size levers and the Zipf tail | 1 | E1.2 |
| 10 | hybrid layouts: every-second-layer AME, A-heads + B-heads in a layer, AME early layers only | 1 | — |
| 11 | tokenizer dependence — everything is id-indexed, so BPE granularity shapes what `T` can express *and* what a shifted head can copy (`v_{j−1}` is one BPE piece, not one word) | Wk 0 + 1 | E0.4 |
| 12 | length generalization, 2k → 100k | 1 | E1.4 |

---

### Numbers worth keeping at hand

Reference config: 1.3B backbone · d_att = 2048 · L = 24 · 16×128 heads · |V| = 50k · fp16 ·
60 GB/s desktop DDR5, 300–400 GB/s server · budget 20–30 ms/token.

| quantity | value |
|---|---|
| bank parameters (E_Q + E_K) | **4.915B / 9.83 GB fp16** — ≈ 4× the backbone |
| …with Option A's E_V | 7.37B |
| returned by deleting W_Q, W_K | 0.201B (≈ 0.4 of ~2.6 GFLOPs/forward → ~15% fewer training FLOPs) |
| rows per token type | **768** (2 banks × 384 head-layers) = 98,304 params/type |
| bank optimizer state | 69 GB at 14 B/param → **79 GB** with fp16 grads, **~88 GB** with fp32 accumulation |
| score path per token | `\|U\|·d_att·L·2B` ≈ **2.0 GB @ \|U\|=20k** → ~33 ms desktop / ~5.6 ms server |
| top-k value reads | `k·d_att·L·2B` ≈ **200 MB** → 3.4 ms streamed, **5–7 ms as 786k×256 B gathers** |
| V-cache | `n·d_att·L·2B` = 3.2 GB @ 32k · **9.8 GB @ 100k** · 12.9 GB @ 128k |
| V-cache write | ~100 KB/token at decode; **3.2 GB in bulk at prefill** (~130 ms over PCIe 4.0 ×16 at ≈25 GB/s) |
| raw-`v` register | `L·d_att·2B` ≈ 98 KB total |
| CPU→GPU per layer | (N, Z, max) ≈ 4 KB fp16 / 8 KB fp32 |
| exponentials per token | `k·H·L` = **786,432** |
| row cache, k = 2048, 4 B/entry | 3.1 GB @ 1000 types · 9.4 GB @ 3000 (~75% hit) · 31.5 GB @ 10k (~86%) · **157 GB** full vocab |
| per-layer pipeline deadline | `CPU_ℓ ≲ GPU_{ℓ−1}` ≈ 0.2 ms, against ≈ 0.14 ms of CPU work |
| decode sync residue (Option B) | ≈ 4·d_att ≈ 8k FLOPs + 4 KB — sync share `1/(n+1)` ≈ 0.05% @ n = 2048 |
| realistic desktop batch-1 @ 100k | **~11–18 ms** (estimate; E2.2 measures it) |
| ALiBi ladder | stretch to 2⁻¹ … ≈2⁻¹⁴ (horizons ~20 … ~185k); the H=16 default 2⁻⁰·⁵ … 2⁻⁸ caps at ~3k |
