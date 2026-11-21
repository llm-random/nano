# Standard AME and dense models: what is applied, what is not

**Status:** decided by Maciej on 2026-11-21. Nothing is open. This page is the short
version of [standard_model_improvements.md](standard_model_improvements.md), which holds
the runs, branches and reasoning behind each row, and the
[proposed standard configs](standard_model_improvements.md#proposed-standard-configs).

**How to read the tables:**
- The effect is the change in loss against the reference run. Lower loss is better.
- Unless a row says otherwise, it was measured on h12
  ([H-balanced model](../concepts/h-balanced-model.md), H = 12) at 4B tokens, with one
  seed.
- For AME, a difference under 0.022 is a tie. Dense varies more between same-seed runs.
- LR 2⁻ⁿ is the learning rate.

## AME

| Change | Decision | Effect on loss | Cost |
|---|---|---|---|
| ALiBi + KV shift + sink (S2-PS), LR 2⁻⁸ | **Keep** (already the default) | Without the shift +0.12, without both +1.14 | — |
| Bank init σ = 1e-4 | **Apply.** The config must set it: the default is still 1.0 | −0.09 vs σ = 1; at h16, σ = 0.01 gave −0.08 | None |
| Banks at 4× the learning rate | **Apply** (Maciej, 2026-11-18) | −0.008 on top of σ = 1e-4, a tie (−0.034 at σ = 1) | None |
| Document mask | **Rejected** (Maciej, 2026-11-21): a small gain, 12% slower, and unmasked packed rows are a valid way to train | −0.016 to −0.018 | +12% s/step |
| ST (STEM) bank gate | **Rejected** (Maciej, 2026-11-21): it breaks AME's precomputed score table | A tie at h12; −0.048 beyond the small bank init at h16 | +3.1–3.8% s/step |
| Bank gates B1, B2, B3 | **Rejected**: no gain | B1 +0.07 to +0.11; B2 and B3 ties | — |
| New init bundle (Xavier embedding and head, pkmf Linears, no embedding weight decay) | **Rejected**: no gain | +0.014 (h16: +0.032) | — |

## Dense

| Change | Decision | Effect on loss | Cost |
|---|---|---|---|
| RoPE; LR 2⁻⁸ at h12, 2⁻⁹ at h16 and h20 | **Keep** (already standard) | Without RoPE +0.28; 2⁻⁸ at h16 is unstable | — |
| KV shift | **Apply** | −0.022 alone, −0.040 on top of QK-norm; no loss spikes | +2.7% s/step, 2.3 GiB |
| QK-norm | **Apply**, for stability (Maciej, 2026-11-20) | A tie: +0.013 alone, −0.006 on top of the shift | +1.8–2.4% s/step, 4 GiB |
| Document mask | **Rejected** (Maciej, 2026-11-21), as for AME | −0.020 to −0.022 | +12% s/step |
| New init bundle | **Rejected**: no gain | +1.27 (the Xavier embedding alone +1.00); h16 +0.21 | — |

## Net effect and what is left

- Together, the applied changes cut the AME − dense gap at h12 from +0.143 to +0.075.
- Before they become the standard:
  - Merge `ame_improve`, which holds the 4× bank rate, QK-norm and the dense shift.
  - Train one run per model at h12 with the full config, about 4 GPU-h each. No run has
    combined all the changes yet.
