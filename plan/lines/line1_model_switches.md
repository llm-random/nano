# Line 1: model-code switches

## Human

Line 1 adds seven model-code switches, one task at a time. Now: Task 1/7, QK-norm, done.
Every switch is off by default, and off means identical training to before.

- [x] QK-norm: per-head RMSNorm on q, k before RoPE
- [ ] Output gate: sigmoid gate on attention output
- [ ] NoPE: skip RoPE on chosen layers
- [ ] Document masking: attend only within each document
- [ ] Indexer sparse attention (tentative): attend to top-k keys
- [ ] Wider residual (tentative): several parallel residual streams
- [ ] nGPT: weights and activations on the unit sphere

**Needs Janek:**
- Run tasks 5 and 6 in parallel once 4 lands? Recommended: yes.
- Before task 5: still wanted, and at what sequence length and k?
- Cluster tests inherit h12's 4-LR sweep (4 jobs per switch): keep, or pin one LR?
- Launch the QK-norm cluster test (4 jobs, 2 GPUs each).

**Last change:** 2026-11-16: QK-norm landed, Qwen-style per-head norm; its cluster config is ready.

## AI

Seven switches that edit the same model code, landed one after another on `pretraining-jm`.
The per-switch survey is `documents/notes/pretraining_switches.md`. This file holds only what
the task agents must share.

Work only from `pretraining-jm`, a plain pretraining branch. AME (`origin/ame_*`) is a separate
project, and branch `pretraining` is someone else's base; neither is a target or a reference
design.

### How to use this file

You are a task agent for one row of the table below.

1. Find your task in the status table. Read **Shared decisions**, then the sections of every
   task before yours (including their "what landed"), then your own section.
2. Do your task. Earlier tasks' code is on your branch, so read it rather than this file's
   summary of it when they disagree.
3. When you finish, in the same diff as your code:
   - keep `## Human` current: tick your checkbox (at most ~8 words on what landed), update
     the status line ("Task N/7, <name>, done"), "Needs Janek" (one line per open decision,
     or "nothing") and "Last change" (one line). Plain words, no file paths, ~15 lines
     total;
   - set your row to `done`;
   - fill "what landed": flag name, config keys, default, anything that departs from your
     section;
   - edit your own section if the plan turned out wrong, so the next agent reads the truth.
   Don't edit other tasks' sections. If a later task's plan is now wrong, say so in your
   final report.

#### Decide yourself, or ask Janek

Janek can't review everything, so most calls are yours.

- **Go ahead without asking**, and record each one in your row or your section:
  - fixing a wrong detail, line number or wording in this file;
  - adding a missing step;
  - reordering steps inside your task;
  - choosing between equivalent implementations.
- **Stop and ask Janek** only for:
  - a blocker;
  - a change to a shared decision (config surface, how flags reach attention, the forward
    contract);
  - anything that touches another line's code or contract;
  - anything that changes behaviour with your switch off;
  - a new dependency;
  - a change to what a switch means;
  - cluster cost beyond the inherited h12 config;
  - re-ordering or dropping tasks in the line.
- **When you ask,** lead with the decision you need, then the options, with your
  recommendation first.

### Status

| # | Task | Status | What landed |
| --- | --- | --- | --- |
| 1 | QK-norm | done | `common.attention.qk_norm`, off `false` in `base.yaml`. Pick: per-head RMSNorm over `dhead` (eps 1e-6) on q and k before RoPE, k on kv heads before `repeat_kv`, SDPA's default scale, because 3 of the 4 surveyed models with full QK-norm use it (Qwen3.8, MiniMax-M3.1, Gemma 4; `documents/notes/qk_norm_survey.md`). Gain `(1 + w)`, w zero-init (`ZeroCenteredRMSNorm`, as Qwen3.8 and MiniMax), so weight decay pulls it to 1, not 0. No `scale` argument. Plumbing: `SWITCHES` in `attention.py`, `SWITCHES_OFF` in `tests/test_attention_registry.py`. Configs in `configs/jm/feature/`: `tiny_local_qk_norm`, `tiny_local_line1_all`, `h12_qk_norm`. |
| 2 | Attention output gate | todo | |
| 3 | NoPE (RoPE off switch) | todo | |
| 4 | Document masking | todo | |
| 5 | Indexer sparse attention (tentative) | todo | |
| 6 | Wider residual stream (tentative) | todo | |
| 7 | nGPT | todo | |

Tasks run in order. Proposal, not decided yet: run 5 and 6 in parallel once 4 has landed (they
touch different classes; see task 6).

### Shared decisions

#### What line 1 edits

- **Yours:**
  - `src/core/model.py`: `RoPEAttention`, `attention_mechanism` / `AttentionMechanism`,
    `Residual`, `TransformerBlock`, `TransformerEncoder`, `TransformerHead`, `LLM`, and for
    nGPT `TransformerEmbedding`
  - `src/core/attention.py`
  - `src/core/mfu.py`
  - the `AttentionSchema` / `CommonSchema` fields in `src/definitions.py`
  - `configs/_model/h_balanced/base.yaml`
  - new `configs/jm/feature/tiny_local_<switch>.yaml`, and `configs/jm/feature/h12_<switch>.yaml`
    for the cluster test
  - existing tests your change breaks
  - this file
- **Not yours:**
  - optimizer construction in `main.py` and `src/core/schedulers.py` (line 2)
  - the FFN slot (`ff_layer_fn`, `GatedFeedForward`, a new `moe.py`; line 3)
  - `src/core/trainer.py`
  - `src/core/datasets.py` and `src/core/eval.py`: document masking needs no change there
    (see task 4)
  - The one planned exception is nGPT's post-step hook (task 7).
- **New parameters are plain `nn.Parameter`s** of the natural shape (a gain is 1-D, a projection
  is a `Linear`). Line 2 sorts parameters into optimizer groups by role (hidden matrix,
  vector/gain, embedding, head), so line 1 never touches the optimizer.

#### Config surface

All switches live under `common`, so a sweep or a run's W&B config shows them in one place,
and the strict `common` schema catches a typo at compose time.

- **Attention switches** go in `common.attention`, next to `type` and `mix`:

  ```yaml
  common:
    attention:
      type: gqa
      qk_norm: false            # task 1
      output_gate: false        # task 2
      no_rope_layers: null      # task 3
      document_masking: false   # task 4
      dsa: null                 # task 5, options block for the `dsa` type
  ```

- **Model-wide switches** (tasks 6 and 7) are keys directly under `common`.
- Each task adds its key in three places:
  - a field in `AttentionSchema` or `CommonSchema` (`src/definitions.py`) with no default
    (`= MISSING`, like `type`);
  - the off value in `base.yaml`;
  - the code that reads it.
- **New constructor arguments have no defaults either.** `base.yaml` is where "off" is written
  down.
- **How an attention switch reaches the layer:**
  - `build_attention` today rejects every `common.attention` key except `type` and `mix`. Task 1
    widens that to a list of switch keys, which each task extends.
  - `build_attention` passes every switch as a keyword to whichever builder `ATTENTION_TYPES`
    picks for the layer. `_rope_attention` forwards them to `RoPEAttention`.
  - The `_planned` builder (mla, mamba2, gated_deltanet) accepts every switch and raises
    `ValueError` at build time when one is on, naming the type and the switch.
  - A builder of a real type that can't honour a switch does the same. A combination that
    can't work fails at build time, never silently.
- **Tests that build attention by hand** (`tests/test_attention_registry.py`, `tests/test_mfu.py`
  pass `attention={"type": ...}`) need the switch keys. Task 1 gives them one shared dict of off
  values; later tasks add their key to it.

#### Off means the old code path

With a switch at its `base.yaml` value, the model is bit-identical to the commit before:

- same parameters, buffers and state-dict keys (old checkpoints load);
- the same RNG draws: a new parameter is created after the existing ones, and only when on;
- the same ops on the forward path.

#### Acceptance check (every task)

Two stages, in order: local checks 1–6, then a cluster config (7) once they all pass.

This machine has no GPU, so local runs are on CPU, which is also where bit-identity can be
checked (identical GPU runs are not bit-reproducible).

Every config for a feature test run, local or cluster, goes in `configs/jm/feature/`.

1. **Setup:** copy `data/` and `data_eval/` from `/home/janek/Documents/IDEAS/nano` into your
   worktree (`cp -r`, not a symlink). Always use `pixi run python`.
2. **Baseline, before your first edit:**
   `pixi run python main.py --config-name tiny_local infrastructure.metric_logger.type=stdout`,
   with the output saved to `$CLAUDE_JOB_DIR/tmp/`.
3. **Switch off:** the same command after your change prints identical loss values at every
   logged step.
4. **Switch on:** add `configs/jm/feature/tiny_local_<switch>.yaml`:
   - `defaults: [../../tiny_local@_global_, _self_]`, with the switch and
     `infrastructure.metric_logger.type: stdout` baked in, and a one-line comment saying how to
     run it. `@_global_` is required: without it the parent lands in a relative package and
     interpolations fail;
   - `pixi run python main.py --config-path configs/jm/feature --config-name tiny_local_<switch>`
     trains all steps with no NaN;
   - show the feature is active (parameter-count delta, or the task's own check).
5. **Combined:** `configs/jm/feature/tiny_local_line1_all.yaml` turns on every line-1 switch that works with
   the others. Task 1 creates it, each later task adds its switch, and it must still train.
   Switches that raise in combination (task 7) instead show the build error.
6. **Tests:** the tests under `tests/` that cover what you touched pass:
   - at least `test_attention_registry.py`, `test_mfu.py`, `test_typed_config.py` and
     `test_h_balanced.py`;
   - `pixi run pytest <files>`.
7. **Cluster config, only after 1–6 pass:** write `configs/jm/feature/h12_<switch>.yaml` for
   Janek to launch. Agents never submit cluster jobs.
   - **What goes in it:** follow `configs/jm/h12_param_groups.yaml`'s pattern, one level down:
     `defaults: [../h12_chinchilla@_global_, _self_]`, then turn the switch on and set
     `infrastructure.metric_logger.name: h12_<switch>` and the tags. Nothing else, so the run
     compares directly with the `h12_chinchilla` baseline.
   - **Tags:** the list is replaced, not merged, so restate `h12_chinchilla`'s tags and add the
     switch's name.
   - **Size:** it inherits `h12_chinchilla`'s `^learning_rate` grid, so it launches four jobs of
     2 GPUs each.
   - **Group refs** are relative (`../../` from `configs/jm/feature/`), and any `override`
     entry goes after the group entries.
   - **Check that it composes:**
     `pixi run python scripts/expand_config.py --config-path configs/jm/feature --config-name h12_<switch> --list`
     prints 4 grid points. Then expand point 0 with `-o $CLAUDE_JOB_DIR/tmp/h12_<switch>.yaml`
     and confirm the switch is on in it.
   - **In the report,** give the launch command:
     `pixi run python run_exp.py --config-path configs/jm/feature --config-name h12_<switch>`.

**Throwaway checks** (a script comparing two code paths, a timing) go in `$CLAUDE_JOB_DIR/tmp/`
and into your report, not into the diff. Don't add new test files unless your section asks.

#### The diff

- **No commits.** Janek reads the diff line by line in VS Code and lands it with `/land`.
- **Keep it small:**
  - no refactors beyond what the switch needs;
  - one-line comments;
  - follow the project memory notes (no field defaults, short comments, tiny_local testing,
    pixi).
- **End with a report**, in this order:
  - decisions Janek needs to know about (or "none");
  - what landed in the diff, and which file to read first;
  - the acceptance results (the loss comparison, the on-run, the tests, the h12 config's
    expand check);
  - the launch command;
  - anything a later task must know.

### Tasks

#### 1. QK-norm

- **Key:** `common.attention.qk_norm: false`.
- **Step 1, pick the operator by a short survey, before any code.** Use the QK-norm form most
  common in recent strong open models.
  - **Models:** the latest release of each family as of today, about 10–12 models. At least the
    four Janek named (DeepSeek, Kimi, Qwen, Mistral), plus Gemma, OLMo, GLM, Llama, MiniMax and
    gpt-oss.
  - **Sources:** primary only: the model's `config.json` and modeling code (transformers
    `modeling_*.py` or the official repo), and the tech report where the code is unclear. Fetch raw
    files with WebFetch, or `curl` a codeload tarball into `$CLAUDE_JOB_DIR/tmp`. `git clone`
    outside the worktree is blocked.
  - **Per model, record:**
    - QK-norm or not;
    - the operator (RMSNorm, LayerNorm, ℓ2 + learned scale) and what it normalizes over (per head
      over `dhead`, or the whole q/k projection);
    - the gain shape, where it sits relative to RoPE and the kv-head repeat, and the softmax scale;
    - with no QK-norm, what the model does instead (another attention type, a logit clip or cap)
      and whether that transfers to our GQA `RoPEAttention`.
  - **Pick** the operator most QK-norm models use. Break a tie, or overrule the majority, only on
    MFU grounds (below), and say why.
  - **MFU, without GPU runs.** No GPU timings; reason it out and give an estimate:
    - `mfu.py` counts only 2-D matmul weights and scores, so the norm and its 1-D gains add no
      counted FLOPs. Any step-time cost shows up directly as lower MFU.
    - The norm is memory-bound elementwise work on q and k. Estimate its time against the block's
      matmuls at h12 (sequence length and batch from `configs/jm/h12_chinchilla.yaml`).
    - The cluster runs use `compile_blocks_full`. Say whether the variant fuses with its
      neighbours (projection output, RoPE) or forces a separate kernel or an extra reshape or copy.
    - Under GQA, normalizing k on the kv heads before `repeat_kv` is cheaper than after. A
      whole-projection norm needs the full q or k vector of each token.
    - CPU timings don't reflect GPU cost, so they aren't evidence.
  - **Write-up:** `documents/notes/qk_norm_survey.md` in your diff (under ~60 lines): the table with
    a source link per row, the pick, and the MFU estimate. Put the pick and a one-line reason in
    "what landed".
  - **Ask Janek** only if no operator has a clear majority, or MFU argues against the majority.
- **Step 2, implement the pick.**
  - Place the norm where the surveyed models do. Under GQA, normalize k before `repeat_kv`.
  - Gains are plain 1-D `nn.Parameter`s, created after the existing parameters and only when on.
  - Keep the softmax scale the chosen models use. If it isn't SDPA's default, give
    `attention_mechanism` / `AttentionMechanism` a `scale` argument (`None` means the default), so
    task 4's flex path can reuse it.
  - Throwaway check: on random input, your q/k path matches the chosen model's reference code to
    float tolerance.
  - `origin/ctx_scl` has a `pre_attn_fn` hook on `RoPEAttention` you may borrow from.
- **Plumbing:** this task sets up the switch plumbing every later task reuses:
  - the switch list in `build_attention`;
  - the `_planned` builder raising when a switch is on;
  - the shared off-values dict in the tests;
  - `configs/jm/feature/tiny_local_line1_all.yaml`.
- **As built (2026-11-16):**
  - `SWITCHES` in `src/core/attention.py` lists the keys. `build_attention` passes
    `attention[key]` for each to the builder, and `_planned` raises when any is truthy. A later
    task appends its key there and to `SWITCHES_OFF` in `tests/test_attention_registry.py`, which
    `tests/test_mfu.py` imports.
  - The chosen models keep 1/√dhead, so `attention_mechanism` has no `scale` argument; task 4's
    flex path uses the default scale.

#### 2. Attention output gate

- **Key:** `common.attention.output_gate: false`.
- **When on:**
  - `g_proj = Linear(dmodel → q_heads·dhead)`, built with `_linear_fn` after `o_proj`, so the
    q/k/v/o draws don't move.
  - The attention output, as `[B, S, q_heads·dhead]` just before `o_proj`, is multiplied by
    `sigmoid(g_proj(x))`, where `x` is the attention layer's input.
  - This is Qwen's best variant (elementwise, head-specific, query-dependent, after SDPA): Qiu et
    al. 2025, "Gated Attention for LLMs", arXiv 2505.06708; Qwen3.5 modeling code in transformers.
- **Why a separate `Linear`, not fused into a doubled `q_proj`:**
  - `q_proj`'s shape and state-dict key stay the same;
  - `mfu.py` counts it as a matmul with no change;
  - line 2 sees an ordinary hidden matrix.
  - Fusing is a later speed optimization.
- **Combines with QK-norm:** works together.

#### 3. NoPE (RoPE off switch)

- **Key:** `common.attention.no_rope_layers: null`. It says which layers skip RoPE, with the
  same syntax as `mix.layers`:
  - `all` is NoPE everywhere (Kimi K3);
  - `every_4th` is the SmolLM3 / OLMo / GLM pattern;
  - a list of indices;
  - `null` is RoPE everywhere (today).
- **Pattern parsing:** reuse the parser in `resolve_layer_types` by factoring out its
  `layers → indices` part, rather than writing a second one.
- **What a builder gets:** `build_attention` resolves the pattern into a per-layer `rope` bool
  and passes that, so "on" for a planned type means `rope=False`.
- **The layer:** `RoPEAttention` gets `rope: bool` and builds no `RoPE` module when it's false
  `RoPE` holds no parameters and non-persistent
  buffers, so the state dict doesn't change.
- **Out of scope:** partial RoPE (`rope_fraction`). Mention it in the report if it would be a
  few lines.
- **Combines with QK-norm and the gate:** works together.

#### 4. Document masking

- **Key:** `common.attention.document_masking: false`, plus the EOS id in `base.yaml`'s model
  node: `eos_token_id: 0`. That's GPT-NeoX `<|endoftext|>`, checked on 2026-11-12; note it in a
  one-line comment.
- **Boundaries come from EOS tokens, inside the model.** Every document ends with EOS
  (`gpt_neox_tokenize_fn`), and both packers keep it. So `datasets.py`, the collate, `trainer.py`
  and `eval.py` don't change, and `LLM.forward(input_ids)` keeps its signature.
  - `doc_ids[b, i]` is the number of EOS tokens in `input_ids[b, :i]`. It's exclusive, so an EOS
    belongs to the document it ends.
  - The mask is causal AND same `doc_ids`. Positions are not reset per document; RoPE stays
    global, as in Llama 3 and OLMo.
- **Forward contract**, which tasks 5–7 must keep:
  - `LLM` takes `document_masking` (interpolated from `common.attention.document_masking` in
    `base.yaml`) and `eos_token_id`.
  - When the switch is on, `LLM.forward` builds one FlexAttention `BlockMask` per forward
    (`create_block_mask`, `B=batch`, `H=None`).
  - It passes it as `block_mask` through `TransformerEncoder.forward` → `TransformerBlock.forward`
    → the attention `Residual` → the attention layer's `forward`. The feed-forward `Residual`
    gets nothing.
  - `block_mask=None` means today's SDPA call, untouched.
  - With a mask, `RoPEAttention` calls flex with the same `scale` QK-norm uses.
  - A planned type raises at build when `document_masking` is on.
- **Flex wrapper:** borrow `_flex_attention` from `src/core/ame.py` on `origin/ame_improve`, an
  unrelated project (take the code, not its design). On
  CUDA it compiles flex when the caller isn't already tracing, and on CPU it makes a bare call.
  Its docstring explains why uncompiled flex is a trap. Also read the `_block_mask` docstring
  there: a `BlockMask` crossing a compile boundary once failed to lower when a sequence dimension
  went dynamic.
- **The tiny_local config:** `tiny_local`'s 16-token sequences rarely contain an EOS (C4
  documents are hundreds of tokens), so `tiny_local_document_masking.yaml` needs a longer
  sequence. Set it in `common` and in both dataloaders' `sequence_length`. Report the fraction
  of sequences that contain a boundary.
- **Checks for the report:**
  - (a) On a packed sequence with the switch on, each document's logits equal that document run
    alone. RoPE is relative, so this holds to float tolerance.
  - (b) The same checkpoint scored by `tiny_local_eval`-style downstream eval gives the same
    scores with the switch on and off (see Cross-cutting issues).
- **Not verified locally:** the compiled path (`compile_blocks_full`, FSDP2, a `BlockMask` as a
  block input) only runs on GPU. `h12_document_masking` is its first compiled run, so write
  "compiled path untested until the h12 run" in "what landed".
- **MFU:** see Cross-cutting issues.
- **Combines with tasks 1–3:** works together.

#### 5. Indexer sparse attention (tentative)

- **Go/no-go before the prompt is written:** our runs use 256–1024-token sequences, where
  DeepSeek's top-2048 keeps every key. It pays only at long context or with a much smaller k.
  Janek confirms the target sequence length and k first.
- **The type:** a new registered type `dsa`, usable as `type` or through `mix`, with an options
  block `common.attention.dsa` (`AttentionSchema.dsa`).
- **Indexer:**
  - `I(t,s) = Σ_j w_j(t) · ReLU(q_j(t) · k(s))` with a few small heads. The design rule is
    `H_I ≤ 2·c·H` (`documents/notes/long-context-attention.md` §6).
  - Top-k keys per query, ANDed with causal and, when on, with task 4's mask. Compose
    `block_mask.mask_mod` with a top-k `mask_mod`.
- **Training:**
  - The indexer learns by KL to the dense attention distribution, with the main path detached.
  - Inject that loss with an autograd function, in the style of DeepSeek-V3's
    `AddAuxiliaryLoss`, so `trainer.py` is untouched.
  - The dense-warmup → sparse switch is a config value. Use a checkpoint branch, as in
    `tiny_local_wsd_branch`, if it needs two stages.
- **MFU:** a `dsa` layer must report its own cost. Add a hook that `mfu.py` asks each attention
  module for (`origin/ame_improve` has an `attention_score_flops_per_token` one to borrow), and
  count k keys plus the indexer's dense scores.
- **Combines:** works with QK-norm, the gate and NoPE (the indexer's keys follow the layer's RoPE
  setting), and with document masking via the AND above.

#### 6. Wider residual stream (tentative)

- **Key:** a `common.<name>` count of residual streams. 1 is today's `Residual` path,
  bit-identical.
- **Structure:** expand after the embedding and collapse before the head. Each sublayer reads a
  learned combination of the streams and writes back with learned weights.
- **Variant:** start from mHC (DeepSeek-V4 `inference/model.py`, 4 streams, Sinkhorn-projected
  mixing). Pick one variant and name it in "what landed". Activation memory grows ×n.
- **What it touches:** `Residual`, `TransformerBlock`, `TransformerEncoder`, `LLM`. It must carry
  task 4's `block_mask`.
- **Residual metrics:** keep the metric names. Define what "residual norm" means with n streams
  (for example, the norm of the input the sublayer reads) and record it.
- **MFU:** the mixing parameters are small 2-D tensors that `mfu.py` would count as 6·P matmuls.
  Correct that if it's off by more than ~1%.
- **Combines:** works with every attention switch and document masking. Raises with nGPT.
- **Parallel with 5?** It edits the residual and block classes, which the indexer doesn't
  touch, so it could run beside task 5 once task 4 lands.

#### 7. nGPT

- **Key:** `common.ngpt: false`.
- **What nGPT changes** (Loshchilov et al. 2024, arXiv 2410.01131):
  - Embeddings and every matrix are unit-norm along dmodel and renormalized after each
    optimizer step.
  - The hidden state stays on the sphere:
    `h ← norm(h + α_A ⊙ (norm(attn(h)) − h))`, and the same with α_M for the MLP. The α are
    learned per-dimension.
  - No RMSNorm layers.
  - q and k are ℓ2-normalized with a learned `s_qk`, and the softmax scale is `sqrt(dhead)`.
  - `s_u` and `s_v` scale the SwiGLU branches.
  - A learned logit scale `s_z` sits in the head.
  - No weight decay and no warmup; that goes in the configs, not the code. So
    `h12_ngpt.yaml` sets those beside the switch, the one departure from step 7's "nothing
    else", and its comment says why.
- **The post-step renorm** needs a hook where the optimizer is built, which is line 2's code in
  `main.py` (`optimizer.register_step_post_hook`). Agree with Janek on where it goes. By then
  line 2 has likely landed.
- **FSDP2:** for `o_proj` and `ff_post_act`, dmodel is dim 0, which FSDP2 shards. Their norms
  cross shards; check this.
- **FFN slot:** `s_u` and `s_v` touch line 3's slot. Put them in an nGPT feed-forward class, not
  in `GatedFeedForward`, and raise when `ff_layer_fn` is anything else (for example, MoE).
- **Combines:**
  - Raises with QK-norm (nGPT normalizes q and k itself), the indexer and wider residual.
  - Works with the gate, NoPE and document masking.
- **Size:** this is the largest diff of the line and may be split into two prompts.

### Cross-cutting issues

#### MFU (`src/core/mfu.py`)

It counts 6·P per token over 2-D weights plus dense score FLOPs per `RoPEAttention`, with no
discount for the causal mask.

- **QK-norm** (1-D gains; the norm isn't counted, like RMSNorm today) and **the gate** (a real
  matmul): counted correctly with no change. Task 1 estimates QK-norm's time cost.
- **Document masking:** keep the dense count. The convention already ignores masking, so MFU
  with document masking is dense-equivalent and overstates useful work by up to the score share
  (~10% of FLOPs at h16, seq 1024). Compare throughput with masking on and off in tokens/s. Task
  4 adds one line saying so to the `mfu.py` docstring.
- **Indexer:** reports its own cost (task 5).
- **Wider residual:** check its mixing parameters (task 6).
- **nGPT:** its scales are 1-D, and normalization isn't counted, like RMSNorm today.

#### In-loop downstream eval (`NanoLM` in `src/core/eval.py`)

It calls `model(input_ids)` on right-padded batches whose contexts contain no document
boundaries.

- **Correctness:** with EOS-derived masks, every real token sees the same keys as under plain
  causal attention. The pad id is 0, which is EOS under GPT-NeoX, but pads only change pad
  positions, which are never scored. So no change to `eval.py`.
- **Cost:** eval runs under `force_eager`, so with document masking on it takes flex's unfused
  path, which materializes `[B, H, S, S]` scores. That's ~2 GB fp32 at batch 32, h16, seq 1024.
  It fits, but measure eval time on the first GPU run.
- **Indexer:** same as document masking.
- **Wider residual and nGPT:** no eval-side change.

#### Residual metrics

`Residual` logs residual-stream and update norms per block. Tasks 6 and 7 change what those mean:

- With n streams, there's more than one stream.
- Under nGPT, the stream is unit-norm by construction, so the residual norm is a constant 1 and
  only the update norm says anything.

Keep the metric names and record what each one now measures in "what landed".

#### Compile and GPU

- The CPU acceptance check runs eager. Tasks 4 and 5 add flex paths that only compile on GPU.
  Their `h12_<switch>` run (`compile_blocks_full`, FSDP2 on 2 GPUs) is the first compiled run,
  and they say so in "what landed".
- Because "off" keeps the SDPA call, runs with the switch off need no GPU recheck.

#### Weight decay on new parameters

Line 2's param groups have landed: with `optimizer: adamw_groups`, every 1-D parameter (QK-norm
gains, nGPT's α) goes to the undecayed `vector` group with no change on line 1's side. The default
optimizer, which `h12_chinchilla` uses, still decays everything in one group. nGPT configs set
`trainer.weight_decay: 0`.

#### Other lines and branches

- **Line 3 (MoE)** may touch `TransformerBlock` for aux-loss collection. Keep task 4's change to
  `TransformerBlock.forward` to the extra argument.
