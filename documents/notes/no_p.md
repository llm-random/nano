# Task: "no P" ablation — train only W_b, freeze P1/P2

Branch: `mas/nips` · Started: 2026-10-05

## Goal

Add a **W_b-only** row to Table 2(b) (`tab:combined_3`, "Bias term W_b") of the AAAI 2027 paper:
freeze the projections P1/P2 at their one-hot init and train only the auxiliary weight W_b.
Run at 50 / 70 / 90% compression so panel (b) becomes a complete 2×2 ablation.

Paper source: `pc_paper/6a3e7175abc07c85648514cd/mainAAAI.tex`
(Overleaf clone — fetch only, **never commit/push** there). Panel (b) is at lines ~562-575,
the "Impact of Bias Weight W_b" paragraph at ~615-616.

## Current state of Table 2(b)

Llama 3 1B, 2B tokens, fineweb-edu, cross-entropy loss (lower is better):

| | 50% | 70% | 90% |
|---|---|---|---|
| PCD | 2.676 | **2.801** | **3.131** |
| HPD | **2.674** | 2.820 | 3.245 |
| w/o bias | 2.762 | 2.951 | 3.354 |

## Why this ablation

With P1, P2 frozen at one-hot selection init: `W_c = P1 W P2 + W_b = (pruned submatrix of W) + W_b`.
`∂L/∂W_b = ∂L/∂W_c`, so training only W_b ≡ training the pruned weights directly ≡ **hard pruning**.
HP is therefore a special case of PC, and panel (b) becomes a clean decomposition:

| | P trainable | P frozen |
|---|---|---|
| **W_b trainable** | PCD (full) | W_b only (≈ HP) — **new** |
| **W_b off** | w/o bias | nothing trains (pruned model at init) |

Narrative: neither component alone (P-only, W_b-only) is enough; the combination is what beats
pruning at 70/90%. This also defuses the current weak spot — "w/o bias" is worse than HPD at
every ratio, which on its own reads as "projections alone lose to pruning".

## Assumptions / known differences vs. HPD

W_b-only is *expected* to be ≈ HPD, but not bit-identical:

1. **Weight decay** — AdamW applies `cfg.trainer.weight_decay` to all params
   (`src/projected_compression/initialization.py:76-107`). In W_b-only, decay pulls W_b → 0,
   i.e. W_c → pruned pretrained weights (L2-SP-like). HP decays weights → 0.
   Decide: run with `weight_decay=0` for both, or keep it and state it as the one known difference.
2. **Embedding & head are covered** — they have their own auxiliary weights
   (`auxiliary_embedding_weights`, `mem_eff.py:783`; `head` is a `CompressibleLinear` with
   `auxiliary_weight`), so freezing all projections still leaves every compressed matrix trainable, as in HP.
3. **Numerics** — W_c is materialized each step as frozen product + W_b in bf16
   (`mem_eff.py:640`); small drift vs. HP updating weights directly.
4. If W_b-only ≈ HPD → report as a sanity check (HP is recovered as a special case of PC).

## Concerns with the existing table

- The **"w/o bias" row is likely stale**: its numbers (2.762 / 2.951 / 3.354) are identical to an
  older commented-out version of the table (`mainAAAI.tex:618-633`), while PCD/HPD rows were updated
  (old: PCD 2.678/2.812/3.162, HPD 2.799/2.823/3.190). Probably not re-run under the current setup.
- Row label "w/o bias" is ambiguous — old table said "PCD without bias"; clarify distillation on/off.
- At 50%, HPD (2.674) ≈ PCD (2.676) — bias helps PC there but doesn't make it win.

## Plan / TODO

- [ ] Implement a config flag to freeze projections (no existing one found): `requires_grad=False` on
      `projection_in_weight`, `projection_out_weight`, `Projections.embedding`; exclude them from the
      optimizer (incl. the `separate_block_optimizers` path).
- [ ] Decide weight-decay handling (see assumption 1).
- [ ] Find the configs/wandb runs behind current PCD / HPD rows (wandb `ideas_cv/pmtest_llm-random`)
      and copy their exact setup (seed, 2B tokens, distillation, LR, clipping).
- [ ] Configs: W_b-only at 50 / 70 / 90%.
- [ ] Configs: re-run "w/o bias" at 50 / 70 / 90% under the same setup.
- [ ] Run, collect final losses.
- [ ] Update Table 2(b) + rewrite the "Impact of Bias Weight W_b" paragraph with the 2×2 framing
      (on Overleaf — not committed from the local clone).

## Notes / log

- 2026-10-05: task defined; analysis of panel (b) and code paths above.
- 2026-10-05: branch is wandb-ready. Side issue (deferred, not blocking): `convert_memeff_to_hf.py:87` passes stale `neptune_run_id` → HF conversion crashes; fix if we need HF eval of new checkpoints.
- 2026-10-05: Table 1 token budgets = n_steps {2048,4096,8192,12288} × 1024 seq × 512 batch → 1.07/2.1/4.3/6.4B; paper rounds the last to "6B" (confirmed by Maciej). Candidate configs for Table 1 1B rows: `configs/pc_project_mas/llama_1B_{comp_memeff,comp_memeff_distil,prune,prune_distillation}.yaml` (snapshot from b1034ffc; verifying against wandb run configs).

## Table 1 (1B) provenance — verified in wandb 2026-10-05

wandb project **`ideas_cv/llm-random-test`** (NOT `pmtest_llm-random`, which holds pre-2026-03 Neptune imports).
Metric used in the paper = final **`steps/100/train/loss`**. Every PC/PCD/HP/HPD 1B cell matches a finished run.
Ratio map: paper 50/70/90% = dmodel 1344/960/448 (dff 5376/3840/1728). n_steps 2048/4096/8192/12288 = 1/2/4/"6"B.

| Row | config | runs (2BT col: 50/70/90) | LR (50/70/90) | clipping | wd |
|---|---|---|---|---|---|
| PC  | `llama_1B_comp_memeff.yaml` | `1n3qr5re` / `mvt98h80` / `y7nhl0rc` (2026-04-24) | 13 all | 0.25 + only_target 1.0 (mode A-C), only_compress false | 0.1 |
| PCD | `llama_1B_comp_memeff_distil.yaml` | `ic3zozd8` / `pk6lz5t9` / `k9ndr8i1` (2026-04-24) | 13 all | same as PC | 0.1 |
| HP  | `llama_1B_prune.yaml` | `8y0ayn2t` / `d697o7ip` / `ef584qrn` (04-24) | 12 / 12 / 11 | 1.0 | 0.1 |
| HPD | `llama_1B_prune_distillation.yaml` | `7fwudoud` / `4yuyjuf9` / `pajazura` (04-23) | 11 / 11 / 12 | 1.0 | 0.1 |

- PC/PCD were re-run 2026-05-05 with identical final losses → reproducible.
- PC/PCD: `cast_bfloat16: false`; git remote = cemetery.
- **Current HPD config has `^learning_rate: [9]` — NOT what Table 1 used** (11/11/12). Fix before any HPD rerun.
- **"w/o bias" row (2.762/2.951/3.354): no matching run in llm-random-test** → from the Neptune era / old setup. Confirms it must be re-run.
- New no-P / w/o-bias runs should copy the **PCD** setup (LR 13, clip 0.25 + only_target 1.0, wd 0.1, 4096 steps, distillation).

### Full Table 1 (1B) run-id map (wandb `ideas_cv/llm-random-test`, original runs; tokens 1B/2B/4B/6B)

| Row | 50% | 70% | 90% |
|---|---|---|---|
| PC  | dvfixq8j / 1n3qr5re / kbsy7i3o / aebc2rsj | tu71ks3t / mvt98h80 / re7ji13d / bhq2ysle | a83pvc44 / y7nhl0rc / 0ibnc42n / l6u6t80m |
| PCD | scfmfbqy / ic3zozd8 / 9mjpyltf / c3vqmfv7 | 45yj0mdp / pk6lz5t9 / j66vre7a / qef5mvf0 | xk83ntk1 / k9ndr8i1 / whk9g4c0 / ciw4ajxz |
| HP  | hd8xh1ye / 8y0ayn2t / vxb3iquw / m3pbg31v | 3hs71dfy / d697o7ip / k6cqnpmp / nfn49e2m | mqlk0lvt / ef584qrn / hp2q65qm / o23y1q3a |
| HPD | 904w73pp / 7fwudoud / t40l65yb / f9uv2mjb | xogsbrqa / 4yuyjuf9 / nzq3ghfr / 44wrh68u | hvd1i37o / pajazura / d95wedea / v4v1k0dd |

PC/PCD runs from 2026-04-24, HP 04-23/24, HPD 04-23. PC/PCD duplicates from 2026-05-05 give identical losses.

### Table 2(b) provenance
- PCD row (2.676/2.801/3.131) = Table 1 PCD 2B column: `ic3zozd8` / `pk6lz5t9` / `k9ndr8i1`.
- HPD row (2.674/2.820/3.245) = Table 1 HPD 2B column: `7fwudoud` / `4yuyjuf9` / `pajazura`.
- "w/o bias" row (2.762/2.951/3.354): not in llm-random-test (exact-match search over all llama_1 runs). Partial scan of Neptune imports (`pmtest_llm-random`, interrupted by API timeout; `llm-random_pmtest_llm-random` fully scanned) found only coincidental matches: `o6tgb99w` (PC 50%, 1BT, no distil, helios) = 2.7619, `b9scyrxm` (HP 50% 1BT) = 2.7616, `pe283yw1` (PC 50%, 6144 steps, entropy) = 3.3535 — none is "PCD w/o bias @ 2BT". No credible source → re-run the row under the PCD recipe.

### PCD recipe (from logged wandb config of ic3zozd8 / pk6lz5t9 / k9ndr8i1, 2026-04-24, server lem)
Only dmodel/dff (+ tags) differ between the 3 runs. Matches `configs/pc_project_mas/llama_1B_comp_memeff_distil.yaml`.
- model `mem_eff.MemoryEfficientProjectedCompression`, cast_bfloat16 false, separate_block_optimizers true, init_norms_with_ones false
- source `/lustre/pd01/plgrid/plgllmefficont2/projected_compression/hf_to_nano_converted/llama1b_in_nano.pt`, importances `.../importances/1b/minitron_dimensions_importances.pt` (activation-based / Minitron)
- trainer `trainer_distillation.PCDistillationTrainer`, distillation_alpha 0.5, temperature 1; teacher = frozen source model (`source_model_for_distillation: true`, load type `pc_memeff_base`, Llama-3.2-1B)
- n_steps 4096, seq 1024, batch 512 (grad accum 8), LR 13, cosine w/ 40 warmup, wd 0.1, clip 0.25 + only_target 1.0, only_compress false
- data FineWebEduDataset, seed 123
- Reported metric `steps/100/train/loss` = CE part (distill/total logged separately).

## Entropy resource audit (2026-10-05, verified on compute node h100a via srun)
NOTE: `/storage_nvme_{1,2,3,4}` mounts exist ONLY on GPU nodes (h100a/a100a), NOT on login `asusgpu0` or `common` partition nodes. Must `srun -p h100` to see/use them. Entropy cfg `cluster_switch` + `PROJECT_HOME_PATH=/storage_nvme_4/nano`.

| Resource | Repo path (lem/old) | Entropy status |
|---|---|---|
| **FineWeb-Edu train** | `cluster_switch.train_path_fineweb` = `/storage_nvme_4/llm-random/datasets/fineweb/train` | ✅ PRESENT — 12,990 `.arrow` shards; path matches entropy.yaml exactly |
| C4 | `/storage_nvme_1/llm-random/datasets/c4` | present (per history) |
| **Llama src model (nano)** | lem `.../hf_to_nano_converted/llama1b_in_nano.pt` | ❌ MISSING — no llama `*in_nano*`/`*.pt` anywhere; only `olmo2_1b_in_nano.pt`, `smollm2_1_7b_in_nano.pt` in `/storage_nvme_4/nano/j321m/pc/` |
| **Llama importances** | lem `.../importances/1b/minitron_dimensions_importances.pt` | ❌ MISSING — no `minitron_dimensions_importances.pt` on any nvme; only olmo2/smollm2 importances under j321m |
| hf_cache | — | only OLMo-2-1B, SmolLM2-1.7B cached (no Llama-3.2-1B) |
| old `/storage_nvme_3/mstefaniak/*` | entropy pre-reset | ❌ WIPED — nvme_3 now has kevinxqiu, robotgeneralist only |
| pixi env | — | `/storage_nvme_4/nano/j321m/pixi` + `/storage_nvme_4/coding_distill/j321m/pixi` |

**Conclusion:** FineWeb-Edu is ready and path-compatible. To run our Llama 1B PC/no-P experiments on entropy we must first provide (copy from lem, or regenerate in-repo): (1) `llama1b_in_nano.pt` via HF→nano conversion, (2) Llama `minitron_dimensions_importances.pt` via the importances pipeline. j321m's post-reset layout uses `/storage_nvme_4/nano/$USER/pc/{importances,checkpoints}` — follow that convention.

## Copying Llama assets lem → entropy (2026-10-05, in progress)
Source files exist on lem (ControlMaster alias `lem`, user plgmstefaniak, key ~/.ssh/plgrid):
- `/lustre/pd01/plgrid/plgllmefficont2/projected_compression/hf_to_nano_converted/llama1b_in_nano.pt` (4.94 GB)
- `/lustre/pd01/plgrid/plgllmefficont2/projected_compression/importances/1b/minitron_dimensions_importances.pt` (522 KB)

Route: direct lem→entropy pull via forwarded SSH agent (plgrid added to local agent; `ssh -A entropy_j`, login node asusgpu0 reaches lem:22). `/storage_nvme_4` is compute-node-only, but /home is NFS-shared + writable from h100a, so:
1. scp lem → `/home/j321m/pc_staging/` (login node). importances ✅ done; checkpoint in progress.
2. `srun -p h100` to move both into `/storage_nvme_4/nano/j321m/pc/` (921G free).
3. Point entropy run configs at `/storage_nvme_4/nano/j321m/pc/{llama1b_in_nano.pt,importances/1b/minitron_dimensions_importances.pt}`.

### Copy DONE (2026-10-05)
Both Llama assets now on entropy fast storage (sizes verified = lem):
- `/storage_nvme_4/nano/j321m/pc/llama1b_in_nano.pt` (4,943,324,727 B)
- `/storage_nvme_4/nano/j321m/pc/importances/1b/minitron_dimensions_importances.pt` (534,629 B)
For an entropy run, set in the PC config: `projected_compression.source_model_path` + `path_to_importances` to the two paths above (replacing the lem `/lustre/...` paths). FineWeb-Edu already matches entropy.yaml.

## AUTONOMOUS MONITORING (started 2026-10-05 ~23:42, user asleep ~12h)
GOAL: reproduce PCD 50% Table 2(b) loss **2.676** on entropy. Wake every 2h, check the run, fix+resubmit on failure, keep notes. Stop waking & wait for user only if blocked by something I can't fix.

### The run
- wandb: `ideas_cv/pc_v2` run `uhhl03fy` (name llama1_pcd_50p_2BT_repro)
- entropy SLURM job **23281** on h100a (4× H100 FSDP2), user j321m
- slurm log: `/home/j321m/llmrandom_cemetery/llama1_pcd_50p_2BT_repro_2026-10-05_23-38-53/slurm-23281_0.out`
- config: `configs/pc_project_mas/llama_1B_pcd_50p_entropy_repro.yaml`
- submit cmd: `pixi run --frozen python run_exp.py --config-path configs/pc_project_mas --config-name llama_1B_pcd_50p_entropy_repro`

### Baselines / thresholds
- TARGET final `steps/100/train/loss` = **2.6763** (orig ic3zozd8). SUCCESS if within ~±0.02 (≤~2.70). INVESTIGATE if >2.72; definite bug if >2.80.
- SPEED baseline = 4.05 s/step (16610s/4096). STOP+debug if sustained >12 s/step (3×).
- Run finishes at step 4095 (n_steps 4096); expected wall ~4.6h → ~04:15.

### Check procedure each wake (via `ssh entropy_j`):
1. `squeue -u j321m` + `scontrol show job 23281` → state (RUNNING/COMPLETED/FAILED/absent).
2. wandb pc_v2 run: current `step`, `_runtime`, `steps/100/train/loss`; compute s/step = _runtime/step (minus ~110s startup).
3. tail slurm log (filter httpx/NVML/WARN) for errors/NaN/OOM.
4. Decisions:
   - still running + on-pace → note & reschedule 2h.
   - finished + loss ≤~2.70 → SUCCESS, record, stop loop, summarize for user.
   - finished + loss >2.72 → inspect config/code diff vs orig, fix, resubmit, note.
   - >3× slow → `scancel 23281`, debug (GPU count? cast? dataloader?), fix, resubmit.
   - crashed → read error, fix if possible + resubmit; if unfixable blocker → STOP loop, leave note, wait for user.

### Log
- 23:42 submitted & confirmed RUNNING+training (block timing ~0.31s, 4 ranks, distill alpha0.5/temp1). loss not yet logged (first at step 100). Scheduling first check +2h.
- 00:45 (check 1): RUNNING, step 1035/4096, 3.7 s/step (faster than 4.05 baseline, OK). loss 2.90 descending smoothly, no errors/NaN/OOM. Trajectory matches orig: repro@1001=2.9008 vs ic3zozd8@1001=2.907 (Δ0.006) → on track for 2.676. ETA ~03:50. Reschedule +1h.
- 01:47 (check 2): RUNNING, step 2041/4096 (halfway), 3.7 s/step steady, loss 2.7711, no errors. Matches orig: repro@2001=2.7711 vs ic3zozd8@2001=2.7782 (repro slightly lower). On track for 2.676. ETA ~03:50. Reschedule +1h.
- 02:48 (check 3): RUNNING, step 3031/4096 (~74%), 3.7 s/step steady, loss 2.6973, no errors. On track for 2.676. ~1065 steps left, ETA ~03:54. Reschedule +1h (should catch completion).

## ✅ REPRODUCED (2026-10-06 03:54)
- Run `uhhl03fy` (pc_v2) FINISHED cleanly at step 4095. **Final steps/100/train/loss = 2.6705** vs target 2.6763 (Δ-0.0058, within ±0.02 — marginally better). Runtime 15314s (~4.25h) vs orig 16610s (faster). Checkpoint saved to /storage_nvme_4/nano/j321m/pc/checkpoints/pc_v2_pcd_50p. No errors.
- Conclusion: entropy setup (data + copied llama assets + config) correctly reproduces PCD 50% Table 2(b). The reproduction pipeline (config `llama_1B_pcd_50p_entropy_repro.yaml`, run_exp→entropy_j, 4×h100) is validated.
- Autonomous monitoring loop STOPPED. Trajectory matched orig throughout (step1001 2.9008 vs 2.907; step2001 2.7711 vs 2.7782).
- NEXT (for user): this validates the baseline; the no-P (W_b-only, frozen P) ablation runs can now use the same entropy pipeline. Still TODO: freeze-projections flag + no-P/w-o-bias configs.

## freeze_projections flag IMPLEMENTED (2026-10-06, early AM)
`projected_compression.freeze_projections: true` (default false) in `src/projected_compression/initialization.py`.
Mechanism (keeps EVERYTHING about W_b identical to PCD):
- P1/P2 (blocks + head) + embedding projection STAY in the optimizer exactly as PCD → their grads are still computed (autograd in backward_compressed_weights) and counted in the per-block/head/embedding clip norm, and the single scheduler param group is preserved (scheduler asserts 1 group).
- A `register_step_post_hook` restores each frozen P to its init snapshot after every optimizer.step(), so P never moves — regardless of grad or weight decay.
- W_b (auxiliary_weight), auxiliary_embedding_weights, head auxiliary_weight, and norms train normally with unchanged clip (0.25) + decay (0.1).
- Snapshot taken at optimizer-build time, which is AFTER one-hot init (create_model:371/375) and before any step. No checkpoint.load in our runs.
Unit-tested locally: name classification correct; P held at init (max|ΔP|=0) under grad+wd while W_b trains. Scheduler-safe (no extra param group); earlier lr=0-group approach rejected (cosine eta_min corrupts a 0-lr group + scheduler 1-group assert).

## no-P ablation run LAUNCHED (2026-10-06 09:36)
- wandb `ideas_cv/pc_v2` run **5r7j91f2** (name llama1_pcd_50p_noP_2BT); entropy SLURM job **23312** on h100a (4×H100).
- slurm log: /home/j321m/llmrandom_cemetery/llama1_pcd_50p_noP_2BT_2026-10-06_09-36-16/slurm-23312_0.out
- config: configs/pc_project_mas/llama_1B_pcd_50p_noP_entropy.yaml (freeze_projections: true; else identical to validated repro).
- Startup confirmed: RUNNING, "freeze_projections=True ... held at init via step-post-hook" logged on all ranks, 4 ranks init, no errors.
- This is a NEW measurement (W_b-only @ 50%), NOT reproducing 2.676. Expectations: loss curve should sit ABOVE full-PCD (uhhl03fy) — if it tracks PCD exactly, the freeze is ineffective (bug). Likely final near HP/HPD-ish. Speed ~3.7 s/step.
- Monitoring criteria: runs clean + normal speed (stop+debug if >12 s/step or crash); confirm loss > full-PCD at matched step; collect final loss. No "reproduce target".

### Monitoring log (no-P)
- 09:38: launched & confirmed RUNNING+freeze active (step 43, loss not yet logged, first at 100). Reschedule +1h.

## no-P with HPD learning rates — SUBMITTED (2026-10-06 ~09:51)
Configs `llama_1B_pcd_{90p_noP_lr12,70p_noP_lr11,50p_noP_lr11}_entropy.yaml` (= noP config, freeze_projections true, LR from HPD Table 2(b): 90%→12, 70%→11, 50%→11). wandb pc_v2.
Submitted in order 90 → 70 → 50:
- 90% lr12: SLURM 23326 (llama1_pcd_90p_noP_lr12_2BT)
- 70% lr11: SLURM 23328 (llama1_pcd_70p_noP_lr11_2BT)
- 50% lr11: SLURM 23329 (llama1_pcd_50p_noP_lr11_2BT)
At submit: all PENDING (90% "Resources", first in line); 23312 (50% noP lr13) still RUNNING on 4 of h100a's 8 GPUs; a colleague's `toy-exp.sh` (23327, also user j321m) queued between 70% and 50%.
- 09:57: cancelled 23312 (50% noP lr13, ~20 min in; its wandb run 5r7j91f2 is partial) to free GPUs → 90% lr12 (23326) started RUNNING 09:57:47. Resubmitted 50% noP lr13 as **23337** (queued last). Queue now: 23326 RUNNING; pending 23328 (70% lr11), 23327 (colleague toy-exp), 23329 (50% lr11), 23337 (50% lr13). h100a: 4 GPUs ours (90%), 1 mateuszwys, 3 free (each job needs 4).
- 10:04: per Maciej, 50% noP lr13 is NOT wanted → cancelled 23337. Only the HPD-LR set matters: 90% lr12 (23326 RUNNING), 70% lr11 (23328), 50% lr11 (23329). The partial lr13 run 5r7j91f2 in wandb pc_v2 can be ignored. (llama_1B_pcd_50p_noP_entropy.yaml = lr13 variant, unused.)
- ~11:40: considered moving 50% lr11 (23329) to a100 — REJECTED. a100a has node-local disks only (/storage_nvme_1 472G free, /storage_ssd_1 18T free), no /storage_nvme_4: no FineWeb (6.0 TB on h100a), no llama assets, no pixi env (16G). GPUs = A100 80GB PCIe (~2-3x slower than h100a). Copy 6TB (~2-8h) + ~9-12h run > waiting for h100. 23329 stays queued on h100. 90% run 0z8eaq4v at step ~1966/4096 at 11:30 (2.7 s/step, ETA ~13:05).
- 12:40: notebook `notebooks/no_p_90_loss_compare.ipynb` — plotly express overlay of steps/100/train/loss (CE) for PCD-90 (k9ndr8i1), HPD-90 (pajazura), no-P-90 lr12 (0z8eaq4v). Executed OK (plotly not in project pixi env; ran via `pixi exec --spec plotly --spec pandas --spec wandb --spec nbconvert --spec ipykernel`). At step 3501 no-P-90 = 3.2289 (HPD-90 final 3.2448, PCD-90 final 3.1312).

### RESULT 90% no-P (lr12) — 0z8eaq4v FINISHED
Final steps/100/train/loss (CE) = **3.2214** (runtime 11208s). vs HPD-90 3.2448 (no-P lower by 0.023), vs PCD-90 3.1312 (no-P higher by 0.090).
→ At 90%, freezing P loses most of PCD's gain over HPD (PCD beats HPD by 0.114; W_b-only recovers only 0.023 of that).
Queue at ~13:45: 70% lr11 (23328) RUNNING (~42 min in); toy-exp done; 50% lr11 (23329) PENDING (Resources).
