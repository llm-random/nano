# AME Stages 1 to 2c — every run

**Status: complete through Stage 2c (2026-11-10).** This file lists every run of
Stages 1, 2, 1b, 2b, 1c and 2c of [ame_experiments.md](ame_experiments.md):
- 48 training runs, all finished;
- 18 smoke runs;
- 3 jobs cancelled before they started.

When a stage adds runs, add them here. Each stage's report holds the decisions,
loss curves, readouts and costs:

| Stage | Scale | Question | Report |
|---|---|---|---|
| 1 | h12, 4B tokens | Dense against the simplest AME, with a learning-rate grid for each | [stage_1_report.md](stage_1_report.md) |
| 2 | h12, 4B tokens | AME with ALiBi, with the KV shift, and with both | [stage_2_report.md](stage_2_report.md) |
| 1b | h16, 8B tokens | Stage 1 one size up | [stage_1b_report.md](stage_1b_report.md) |
| 2b | h16, 8B tokens | Stage 2 one size up | [stage_2b_report.md](stage_2b_report.md) |
| 1c | h20, 16B tokens, 4 GPUs | Dense's learning-rate grid | [stage_1c_report.md](stage_1c_report.md) |
| 2c | h20, 16B tokens, 4 GPUs | The best AME variant, ALiBi with the KV shift | [stage_2c_report.md](stage_2c_report.md) |

**Every run shares the following**, unless its row says otherwise:
- **Seed and learning rate.** Seed 123. The learning rate is 2⁻ⁿ.
- **Bank defaults.** Every AME run here trained its banks at the base learning rate and
  initialized them at σ = 1.0, the defaults then. Since the standard-models merge,
  `trainer.bank_lr_ratio` defaults to 4.0, and since 2026-11-25 `bank_init_std` to 1e-4.
  An exact rerun needs the code the run used (its cemetery branch): since 2026-11-25 each
  part of the model initializes from its own seed (`src/core/init_rng.py`), so today's code
  builds other starting weights. On today's code, `trainer.bank_lr_ratio=1` and
  `++common.attention.ame.bank_init_std=1.0` restore the optimizer and the init scale, and
  ratio 1 lets the run's checkpoint resume (no bank group,
  [model.py:753](../../src/core/model.py#L753)).
- **Batch.** 512 × 2,048 = 1,048,576 tokens per step.
- **Run lengths.**
  - h12: 3,815 steps, 4.00B tokens;
  - h16: 7,630 steps, 8.00B tokens;
  - h20: 15,259 steps, 16.00B tokens.
- **Hardware.** H100s on LEM.
  - h12 and h16: 1 GPU, autocast bf16, no FSDP2.
  - h20: 4 GPUs, FSDP2 mixed precision.
- **W&B.** Runs are in `ideas_cv/nano-ame`. The logged config is the truth.
- **`final_ce`** is the held-out loss over 8,192 × 2,048 tokens at the end of the run, as
  logged in W&B. Each batch loss is rounded to bf16, so a value can be off by up to
  ±0.008. The reports compare close runs by paired train loss instead.
- **Job** is the SLURM job on LEM, written `<array>_<task>` when the array has more than one
  task.

## Best result per arm and scale

| Arm | h12, 4B tokens | h16, 8B tokens | h20, 16B tokens |
|---|---|---|---|
| Dense with RoPE: S1-D, S1b-D, S1c-D | n = 8: **2.9824** | n = 9: **2.7441**, at the edge | n = 9: **2.5059** |
| AME with ALiBi and the KV shift: S2-PS, S2b-PS, S2c-PS | n = 8: **3.1250** | n = 8: **2.8594**, at the edge | n = 8: **2.5596** |
| Gap, AME − dense, best against best | +0.143 | +0.115 | **+0.054** |
| AME with ALiBi only: S2-P, S2b-P | n = 8: 3.2480, at the edge | n = 8: 2.9404, at the edge | not run |
| AME with the KV shift only: S2-S, S2b-S | n = 8: 3.9346, at the edge | n = 8: 3.3721, at the edge | not run |
| Simplest AME, with neither: S1-A0, S1b-A0 | n = 6: 4.2656 | n = 9: 3.9746, at the edge | not run |
| Dense without RoPE: S2-D0 | n = 9, the only n tried: 3.3027 | not run | not run |

- **"At the edge"** means the best n is the highest learning rate tried, so the arm's
  optimum may be lower. h16 did not try dense at n = 8 or AME with ALiBi and the shift at
  n = 7.
- **Each cell has one seed.** The only seed pair is the simplest AME at h12 and n = 9:
  4.4316 for seed 123 (S1-A0) and 4.1992 for seed 124 (S2-N), a gap of 0.232. Its seed-123
  run spiked early. No arm with a positional signal has a second seed.
- **What the AME–dense comparison does not match:**
  - AME has 5.1–6.0× dense's parameters, most of them in the banks, and does 0.95–0.96×
    its FLOPs per token.
  - AME's banks are exempt from weight decay, while the dense W_Q and W_K are not.
  - h20 also changes the GPU count, the precision path and the tokens per dense
    parameter: 21.8, 22.6 and 25.8 from h12 to h20.

## Arms

| Arm | Stage | Scale | Attention | Parameters: total / backbone / banks | GFLOP/token | Config | Commit |
|---|---|---|---|---|---:|---|---|
| S1-D | 1 | h12 | Dense, RoPE | 183.45M / 106.19M / — | 1.095 | [stage_1/dense_h12_lr_grid.yaml](../../configs/stage_1/dense_h12_lr_grid.yaml) | `8a240816` |
| S1-A0 | 1 | h12 | AME, no positional signal, no shift | 1,103.60M / 99.05M / 927.28M | 1.053 | [stage_1/ame_h12_simplest_lr_grid.yaml](../../configs/stage_1/ame_h12_simplest_lr_grid.yaml) | `8a240816` |
| S2-N | 2 | h12 | S1-A0 with seed 124 | as S1-A0 | 1.053 | [stage_2/ame_h12_simplest_seed2.yaml](../../configs/stage_2/ame_h12_simplest_seed2.yaml) | `f9680709` |
| S2-P | 2 | h12 | AME, ALiBi | as S1-A0 | 1.053 | [stage_2/ame_h12_pe.yaml](../../configs/stage_2/ame_h12_pe.yaml) | `f9680709` |
| S2-S | 2 | h12 | AME, KV shift | as S1-A0, + 576 (α, β) | 1.053 | [stage_2/ame_h12_shift.yaml](../../configs/stage_2/ame_h12_shift.yaml) | `f9680709` |
| S2-PS | 2 | h12 | AME, ALiBi and the KV shift | as S2-S | 1.053 | [stage_2/ame_h12_pe_shift.yaml](../../configs/stage_2/ame_h12_pe_shift.yaml) | `f9680709` |
| S2-D0 | 2 | h12 | Dense, no RoPE (NoPE) | as S1-D | 1.095 | [stage_2/dense_h12_nope.yaml](../../configs/stage_2/dense_h12_nope.yaml) | `c4cb1ef6` |
| S1b-D | 1b | h16 | Dense, RoPE | 354.71M / 251.7M / — | 2.222 | [stage_1b/dense_h16_lr_grid.yaml](../../configs/stage_1b/dense_h16_lr_grid.yaml) | `31cbc268` |
| S1b-A0 | 1b | h16 | AME, no positional signal, no shift | 1,986.33M / 234.9M / 1,648.36M | 2.121 | [stage_1b/ame_h16_simplest_lr_grid.yaml](../../configs/stage_1b/ame_h16_simplest_lr_grid.yaml) | `31cbc268` |
| S2b-P | 2b | h16 | AME, ALiBi | as S1b-A0 | 2.121 | [stage_2b/ame_h16_pe.yaml](../../configs/stage_2b/ame_h16_pe.yaml) | `89975bbf` |
| S2b-S | 2b | h16 | AME, KV shift | as S1b-A0, + 1,024 (α, β) | 2.121 | [stage_2b/ame_h16_shift.yaml](../../configs/stage_2b/ame_h16_shift.yaml) | `89975bbf` |
| S2b-PS | 2b | h16 | AME, ALiBi and the KV shift | as S2b-S | 2.121 | [stage_2b/ame_h16_pe_shift.yaml](../../configs/stage_2b/ame_h16_pe_shift.yaml) | `89975bbf` |
| S1c-D | 1c | h20 | Dense, RoPE | 620.35M / 491.57M / — | 3.9646 | [stage_1c/dense_h20_lr_grid.yaml](../../configs/stage_1c/dense_h20_lr_grid.yaml) | `a8553b7e` |
| S2c-PS | 2c | h20 | AME, ALiBi and the KV shift | 3,163.20M / 458.86M / 2,575.56M | 3.7683 | [stage_2c/ame_h20_pe_shift.yaml](../../configs/stage_2c/ame_h20_pe_shift.yaml) | `a8553b7e` |

- **Parameters.** Totals include the untied embedding and head: 77.27M at h12, 103.02M
  at h16 and 128.78M at h20. The backbone is everything else except the banks.
- **FLOPs** are the trainer's training FLOPs per token at sequence length 2,048. They
  leave out the bank gathers, which are lookups.
- **Commits.** All are on `ame_main`.
  - Stage 1's runs after its first grid, n ≤ 9, used `8a240816` with only the grid list
    changed.
  - Stage 2's `f9680709` has the same code as `8a240816`.
  - Stage 2b's `89975bbf` leaves AME unchanged since Stage 1b's `31cbc268`.
  - Each report gives the cemetery snapshot of every run.
- **ALiBi** is AME's own ladder of one fixed slope per head, with horizons from 16 tokens
  to 4 × the context. The KV shift adds learned α and β per head and layer.

## Every training run

The best `final_ce` of each arm in each stage is bold. "Grid" is the stage's planned
learning-rate grid. An extension is a point added past its edge. A learning-rate check
(LR check) is Stage 2's or 2b's ±1 around n = 9. "2 × 2" is a cell of Stage 2's or 2b's
comparison.

| Stage | Arm | n | Seed | `final_ce` | W&B run | Job | Role, notes |
|---|---|---:|---:|---:|---|---|---|
| 1 | S1-D | 7 | 123 | 4.2734 | `6ldjhbyd` | 6253087 | extension; stuck near 6.0 from step ~280 to ~1,200, then learned as the LR decayed |
| 1 | S1-D | 8 | 123 | **2.9824** | `x0xlqroj` | 6252398 | extension |
| 1 | S1-D | 9 | 123 | 3.0244 | `ovy19sqp` | 6251812 | extension |
| 1 | S1-D | 10 | 123 | 3.2002 | `y451s5xq` | 6251667_0 | grid |
| 1 | S1-D | 11 | 123 | 3.3027 | `v34fc3g7` | 6251667_1 | grid |
| 1 | S1-D | 12 | 123 | 3.5215 | `4k9jtp9b` | 6251667_2 | grid |
| 1 | S1-D | 13 | 123 | 3.8301 | `l7iznfkf` | 6251667_3 | grid |
| 1 | S1-A0 | 4 | 123 | 4.5215 | `vla8p16s` | 6255094_1 | extension |
| 1 | S1-A0 | 5 | 123 | 4.4512 | `lj0xg6yj` | 6255094_0 | extension |
| 1 | S1-A0 | 6 | 123 | **4.2656** | `lr528luy` | 6253086_1 | extension |
| 1 | S1-A0 | 7 | 123 | 4.3262 | `5bvmqr0l` | 6253086_0 | extension |
| 1 | S1-A0 | 8 | 123 | 4.3711 | `tuvcps1y` | 6252402 | extension; also Stage 2's N cell at n = 8 |
| 1 | S1-A0 | 9 | 123 | 4.4316 | `p5u294hj` | 6251816 | extension; also Stage 2's N cell at n = 9 (seed 123) |
| 1 | S1-A0 | 10 | 123 | 4.5137 | `ncfg11gm` | 6251668_0 | grid |
| 1 | S1-A0 | 11 | 123 | 4.6387 | `0cdw01vh` | 6251668_1 | grid |
| 1 | S1-A0 | 12 | 123 | 4.7715 | `h45qwv5g` | 6251668_2 | grid |
| 1 | S1-A0 | 13 | 123 | 4.9844 | `ssdlz0nk` | 6251668_3 | grid |
| 2 | S2-N | 9 | 124 | **4.1992** | `fqzr01mh` | 6253468 | S1-A0 again with seed 124: the noise floor (gap 0.232 to seed 123) |
| 2 | S2-P | 8 | 123 | **3.2480** | `z7f5bsuo` | 6259007 | 2 × 2 at n = 8 |
| 2 | S2-P | 9 | 123 | 3.2979 | `0rbp1bju` | 6253416 | 2 × 2 at n = 9 |
| 2 | S2-S | 8 | 123 | **3.9346** | `4fvk32q7` | 6259008 | 2 × 2 at n = 8 |
| 2 | S2-S | 9 | 123 | 3.9385 | `r6lxbmtd` | 6253417 | 2 × 2 at n = 9 |
| 2 | S2-PS | 6 | 123 | 3.1348 | `egvqspym` | 6259085 | LR curve past the edge (Maciej's request) |
| 2 | S2-PS | 7 | 123 | 3.2031 | `zdlmmz7n` | 6259084 | LR curve past the edge |
| 2 | S2-PS | 8 | 123 | **3.1250** | `doi36p8k` | 6256577 | LR check; 2 × 2 at n = 8 |
| 2 | S2-PS | 9 | 123 | 3.1865 | `ublx3ion` | 6253467 | 2 × 2 at n = 9 |
| 2 | S2-PS | 10 | 123 | 3.2959 | `p1oslt5b` | 6256578 | LR check |
| 2 | S2-D0 | 9 | 123 | **3.3027** | `b93tal8j` | 6253574 | extra arm: dense without RoPE |
| 1b | S1b-D | 9 | 123 | **2.7441** | `eh2sjsh7` | 6253313_0 | grid; best at the high-LR edge, not extended (decision 3) |
| 1b | S1b-D | 10 | 123 | 2.8438 | `8jm0qkjs` | 6253313_1 | grid |
| 1b | S1b-D | 11 | 123 | 2.9600 | `7rd0xax5` | 6253313_2 | grid |
| 1b | S1b-A0 | 9 | 123 | **3.9746** | `n15qni7a` | 6253317_0 | grid; also Stage 2b's N cell; best at the edge, not extended |
| 1b | S1b-A0 | 10 | 123 | 4.0664 | `v76wb0e1` | 6253317_1 | grid |
| 1b | S1b-A0 | 11 | 123 | 4.2227 | `7ujg1z77` | 6253317_2 | grid |
| 2b | S2b-P | 8 | 123 | **2.9404** | `le2dr3r6` | 6261187 | 2 × 2 at n = 8 |
| 2b | S2b-P | 9 | 123 | 2.9834 | `eedqxi1j` | 6255245 | 2 × 2 at n = 9 |
| 2b | S2b-S | 8 | 123 | **3.3721** | `eeuebwsj` | 6261188 | 2 × 2 at n = 8 |
| 2b | S2b-S | 9 | 123 | 3.4658 | `c2tjvh5w` | 6255246 | 2 × 2 at n = 9 |
| 2b | S2b-PS | 8 | 123 | **2.8594** | `hjw7znbz` | 6260570 | LR check; 2 × 2 at n = 8; n = 7 not tried |
| 2b | S2b-PS | 9 | 123 | 2.8965 | `rv4xqzap` | 6255247 | 2 × 2 at n = 9 |
| 2b | S2b-PS | 10 | 123 | 2.9688 | `kxt7oe48` | 6260571 | LR check |
| 1c | S1c-D | 8 | 123 | 2.8252 | `abtjvllm` | 6275160_0 | grid; stalled near 5.2 for its first 2,000 steps, then recovered |
| 1c | S1c-D | 9 | 123 | **2.5059** | `1x82rzby` | 6275160_1 | grid; n_d\* = 9, by 0.0039 over n = 10 |
| 1c | S1c-D | 10 | 123 | 2.5098 | `mona5up0` | 6275160_2 | grid |
| 2c | S2c-PS | 7 | 123 | 2.5928 | `40cdf2te` | 6285238 | extension past the edge |
| 2c | S2c-PS | 8 | 123 | **2.5596** | `tg4h4g66` | 6284092 | grid |
| 2c | S2c-PS | 9 | 123 | 2.5762 | `g3gw9wru` | 6275194 | grid; started alongside Stage 1c |
| 2c | S2c-PS | 10 | 123 | 2.6318 | `5xclzzfy` | 6277615 | grid |

## Jobs cancelled before they started

| Stage | Arm | n | Job | Why |
|---|---|---:|---|---|
| 2b | S2b-N, seed 124 | 9 | 6255248 | At Maciej's request ([stage_2b_report.md](stage_2b_report.md), decision 3). So Stage 2b has no noise floor of its own |
| 1c | S1c-D | 11 | 6281083 | Queued early, while n = 10 led. n = 9 ended best, so it was not needed ([stage_1c_report.md](stage_1c_report.md), decision 14) |
| 2c | S2c-PS | 11 | 6282280 | Queued early, in case dense's best n came out as 10. It came out as 9 ([stage_2c_report.md](stage_2c_report.md), decision 8) |

## Smoke runs

Each smoke run is 60 steps. It checks that a stage's configs fit in memory, the speed,
that `final_ce` is logged and that the checkpoint is written. The losses below are not
results. Stage 1's report records the h12 smoke runs, Stage 1b's the h16 ones, and
Stage 1c's and 2c's their own.

| Stage | Arm | n | `final_ce` after 60 steps | W&B run | Job | Notes |
|---|---|---:|---:|---|---|---|
| 1 | S1-D | 10 | — | `auyt41sd` | 6251533_0 | failed at 20 s: bad GPU on r16-11 |
| 1 | S1-D | 11 | — | `1peeepdd` | 6251533_1 | failed at 13 s: the same node |
| 1 | S1-D | 10 | 6.7773 | `bm41vdbc` | 6251543_0 | resubmitted with r16-11 excluded |
| 1 | S1-D | 11 | 6.6270 | `ymolkx78` | 6251543_1 | resubmitted with r16-11 excluded |
| 1 | S1-D | 12 | 6.7344 | `rzjcmjgx` | 6251533_2 |  |
| 1 | S1-D | 13 | 7.1328 | `b1vhd7hb` | 6251533_3 |  |
| 2 | S2-N | 12 | 6.7129 | `ns484ak5` | 6251535 | the S1-A0 code path, seed 124 |
| 2 | S2-PS | 12 | 6.7285 | `k3ydhc0h` | 6251536 |  |
| 1b | S1b-D | 9 | 7.0605 | `x8fdvgvx` | 6253231_0 |  |
| 1b | S1b-D | 10 | 6.9102 | `hv43mznh` | 6253231_1 |  |
| 1b | S1b-D | 11 | 6.8066 | `fq0rxyo3` | 6253231_2 |  |
| 2b | S2b-N | 10 | 6.9414 | `p572pg6e` | 6253235 | the S1b-A0 code path, seed 124 |
| 2b | S2b-PS | 10 | 6.9980 | `82b324oh` | 6253237 |  |
| 1c | S2b-PS at h16 | 9 | 7.1602 | `7f1uzqtu` | 25860605 | Helios compile proof: 1 GH200, `--qos=now` |
| 1c | S1c-D | 8 | 7.3164 | `mo7jtzjt` | 6274930_0 |  |
| 1c | S1c-D | 9 | 7.2988 | `iuroj05q` | 6274930_1 |  |
| 1c | S1c-D | 10 | 7.0703 | `s6qlhzjb` | 6274930_2 |  |
| 2c | S2c-PS | 9 | 7.3535 | `yss8c50l` | 6274942 |  |

**Sources.** The tables were built on 2026-11-10 from W&B: every run in `ideas_cv/nano-ame`
tagged `stage_1` to `stage_2c`. They were then checked against the reports. Every training
run finished its full step count. Its `final_ce` to 4 decimals, its run id and its job all
appear in its stage's report.
