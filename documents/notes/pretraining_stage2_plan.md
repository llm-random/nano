# Stage 2: H-balanced lineage, GPT-NeoX tokenizer, switchable attention

Builds the new project's foundations on the stage-1 skeleton. Spec sources:
`documents/concepts/h-balanced-model.md` (the H-balanced definition) and the stage-2
ingredient list in `pretraining_reset_log.md`.

**Status: PLAN — committed for review. Awaiting green light before execution.**

## Ground rules (carried from stage 1, plus stage-2 specifics)

- Work in `/home/maciej/code/nano`, direct commits on `pretraining`.
- Gates after every commit: `pytest -m "not gpu and not slow and not network"` and
  `black --check` green. A **runnable end-to-end local CPU smoke config exists at all
  times** (today: `tiny_local` + `smoketest_fineweb_final_eval_local`; after Phase B:
  their H4 successors).
- Phases B and C **change behavior on purpose** (new shapes, new token ids). Before
  each, record a fresh seeded baseline and state in the log what must change and what
  must not. Phases D and E **must not change numerics**: their gate is a seeded run
  bit-identical to the Phase-C baseline on the default (GQA-everywhere) config.
- No cluster runs, no real training runs, **no pixi.toml/pixi.lock changes** — the
  plan needs no new dependencies (kernels are deferred behind interfaces).

## Phase A — CI on this branch (parked item 1)

Add `pretraining` to the `push` branch list of `.github/workflows/tests.yml` and
`black.yml` (pull_request stays main-only). Gate: push, then confirm both workflows
run green on GitHub — this also proves the CI pip environment still covers the suite
after the stage-1 deletions.

## Phase B — H-balanced model lineage

1. New config family `configs/_model/h_balanced/`:
   - `base.yaml` — everything the old `llama/base_model.yaml` provided, but every
     dimension derived from one `common.h` knob via Hydra interpolation
     (`dmodel: ${eval:'64 * ${common.h}'}`-style resolver already exists in
     `resolver/`; if not, dimensions are written out per instance file — decided at
     execution, logged):
     `n_blocks=H, q_heads=H, kv_heads=H/2, dmodel=64H, dff=256H`, head_dim 64,
     gated FF (present `GatedFeedForward`), RMSNorm, untied embedding/head.
   - `h4.yaml` (smoke/tests, ~tiny), `h16.yaml` (def_300-class, 354.6M),
     `h24.yaml` (def_800-class, 1003.8M).
   - RoPE: `base=10000, scale_freqs=false` — the llm-random-lineage setting the RoPE
     docstring documents, instead of llama's 500000+scaling. **Flagged for Maciej**;
     one line to change if he prefers otherwise.
2. `vocab_size`: the concept doc says 50257, but that is GPT-2's count. At execution,
   measure `len(AutoTokenizer("EleutherAI/gpt-neox-20b"))`; if it exceeds 50257
   (expected ≈50277), set vocab_size to the next multiple of 128 (≈50304, the Pythia
   convention), update the concept doc accordingly, and log the deviation. The
   embedding must never be smaller than the largest token id.
3. Delete the scaffolding: `configs/_model/tiny.yaml`, `configs/_model/llama/`
   (base_model.yaml). Re-point `tiny_local`, `tiny_remote`, `tiny_remote_wandb`,
   `smoketest_fineweb_final_eval_local`, `exp.yaml` at `h_balanced/h4`.
4. Tests: an H-balanced invariants test — instantiate `h16` and `h24` on the `meta`
   device (no memory cost) and assert every derived dimension and the closed-form
   parameter count `61440·H³ + 2·V·64H` (exact, not approximate).
5. Gate: pytest + black; record the **Phase-B baseline** (seeded tiny-H4 run +
   fineweb smoketest). Loss values will differ from stage-1 baselines — expected,
   logged; what must hold: training runs, loss finite and >0, checkpoint lands.

## Phase C — GPT-NeoX tokenizer on fineweb-edu

1. Add `gpt_neox_tokenize_fn` (`EleutherAI/gpt-neox-20b`, ungated — no HF token
   needed) to `src/core/datasets.py`, mirroring the existing fns.
2. Point `configs/_trainer/llama.yaml`'s two dataloaders at it (this is what every
   fineweb run composes; `local_dummy` keeps `gpt2_tokenize_fn` so the offline CI
   tests stay offline — gpt2's 50257 ids fit the same embedding).
3. Delete `llama_tokenize_fn` (meta-llama is a gated repo needing a token; nothing
   should depend on it in a from-scratch project).
4. Download the tokenizer once locally; any test touching it is marked `network`,
   same convention as today.
5. Gate: pytest + black; fineweb smoketest end-to-end; record the **Phase-C
   baseline** (this is the reference for D and E). Expected change: token ids and
   therefore all fineweb losses. Must not change: the local-dummy (gpt2) path —
   its seeded run must stay bit-identical to Phase B's.

## Phase D — attention-switching abstraction

1. New `src/core/attention.py`:
   - `ATTENTION_TYPES` registry: `mha`, `mqa`, `gqa` (real), `mla`, `mamba2`,
     `gated_deltanet`, `ame` (registered interfaces).
   - `mha/mqa/gqa` all build the existing `RoPEAttention` — they differ only in
     `kv_heads` (mha: `q_heads`, mqa: `1`, gqa: the config's `kv_heads`) and the
     k/v projection widths that follow. No new math, no new class.
   - The other four are classes with the constructor signature every attention type
     shares (`dmodel, q_heads, kv_heads, seq_len, rope/init settings`) whose
     `forward` raises `NotImplementedError("<type> needs its kernel/deps — stage 3")`.
     MLA may get a plain-torch CPU reference if it fits the shared signature
     cleanly; decided at execution, logged (stub is the floor, not the goal).
2. Config surface on the model: `model.attention.type: gqa` replaces the inlined
   `RoPEAttention` block in `h_balanced/base.yaml`; a factory in `attention.py`
   turns `(type, common dims)` into the per-layer module. `TransformerBlock`'s
   `attention_fn` contract gains the block id (it already receives `block_id`), so
   Phase E can vary the type per layer.
3. Tests: registry completeness (every name constructs or raises the documented
   error, none silently missing); mha==gqa-with-kv=q bitwise on a seeded forward;
   mqa shapes; stub types raise on forward but **construct** (so configs naming
   them compose).
4. Gate: pytest + black + **bit-identical** seeded tiny-H4 and fineweb smoketest vs
   the Phase-C baseline — the abstraction must be a pure refactor for the default
   GQA-everywhere config.

## Phase E — layer-pattern mixing

1. `resolve_layer_types(n_blocks, base_type, mix=None) -> list[str]` in
   `attention.py`. `mix = {type: <name>, layers: "every_2nd" | "every_4th" | "all"
   | [explicit 0-based indices]}`. Convention (documented + tested): `every_Nth`
   means 0-based indices `i` with `(i + 1) % N == 0` — layer 3, 7, 11… for N=4 —
   so the first layer is never silently replaced. `"all"` makes `mix.type` the
   whole stack. Out-of-range or duplicate explicit indices are a config error,
   raised at build time with the offending indices named.
2. Config surface: `model.attention: {type: gqa, mix: {type: mqa, layers: every_4th}}`.
   Omitting `mix` keeps a uniform stack.
3. Tests: resolution table for every pattern form on H=4 and H=16; a seeded CPU
   e2e run on H4 mixing `gqa` + `mqa` (two real types) trains for a few steps; a
   config naming a stubbed type (`ame` on every 4th layer) **composes and
   instantiates**, failing only at forward — proving the wiring is ready for the
   real AME.
4. Gate: pytest + black + default (no-mix) config still bit-identical to Phase C.

## Phase F — close-out

Update `documents/concepts/h-balanced-model.md` ("Instances in this repo" section
names the new config family; vocab correction from B.2), add a short
`documents/concepts/attention-switching.md` (the registry, the pattern convention,
how to add a type), README one-liner, final full gates, push, summary with the
parked-decision list.

## Decisions already parked for Maciej (will be appended to during execution)

- RoPE base 10000 / no freq scaling for the H family (B.1).
- vocab_size padding if NeoX measures >50257 (B.2) — includes a concept-doc edit.
- Old parked items from stage 1 remain: glossary pruning, `llm_random_weight_init`
  naming (the H configs will keep the name for now), pixi dep pruning.

## Execution order & gates summary

A (CI) → B (models, new baseline) → C (tokenizer, new baseline) → D (attention,
bit-identical gate) → E (patterns, bit-identical gate) → F (docs + push).
One phase ≈ one commit unless a phase's test additions warrant a second.
