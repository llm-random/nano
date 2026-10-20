# Table 1 experiments — hardware & training time

Source: wandb `ideas_cv/llm-random-test` (runs that produced the paper's Table 1 cells, matched by final `steps/100/train/loss`). GPU type/count from wandb system metadata of rank 0.

- **Train time** = wandb `_runtime` (wall clock incl. startup/model load).
- **s/step (steady)** = Δ`_runtime` / Δstep between the first and last logged loss points (excludes startup).
- **s/step (total)** = train time / number of steps.
- Tokens: 1B model = n_steps 2048/4096/8192/12288 × seq 1024 × batch 512; 8B model = n_steps 1024/2048/4096/6144 × seq 2048 × batch 512. Both ≈ 1.07/2.1/4.3/6.4B tokens (paper rounds to 1/2/4/6B).
- Runs were matched to cells by exact final loss + expected run name + expected n_steps (8B). 'Note' column flags ambiguities.
- SVD (8B) was run in a separate framework, not logged to wandb — no hardware/timing data.

| # | Model | Method | Ratio | Tokens | wandb run | GPUs | GPU type | Cluster | Train time (h) | s/step (steady) | s/step (total) | Final loss | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1B | PC | 50% | 1B | dvfixq8j | 4 | H100 | lem | 1.47 | 2.52 | 2.58 | 2.7784 |  |
| 2 | 1B | PC | 50% | 2B | 1n3qr5re | 4 | H100 | lem | 2.90 | 2.52 | 2.55 | 2.7089 |  |
| 3 | 1B | PC | 50% | 4B | kbsy7i3o | 4 | H100 | lem | 5.77 | 2.52 | 2.53 | 2.6438 |  |
| 4 | 1B | PC | 50% | 6B | aebc2rsj | 4 | H100 | lem | 8.61 | 2.52 | 2.52 | 2.6125 |  |
| 5 | 1B | PC | 70% | 1B | tu71ks3t | 4 | H100 | lem | 1.14 | 1.95 | 2.00 | 2.9293 |  |
| 6 | 1B | PC | 70% | 2B | mvt98h80 | 4 | H100 | lem | 2.23 | 1.94 | 1.96 | 2.8380 |  |
| 7 | 1B | PC | 70% | 4B | re7ji13d | 4 | H100 | lem | 4.43 | 1.93 | 1.95 | 2.7606 |  |
| 8 | 1B | PC | 70% | 6B | bhq2ysle | 4 | H100 | lem | 6.60 | 1.92 | 1.93 | 2.7247 |  |
| 9 | 1B | PC | 90% | 1B | a83pvc44 | 4 | H100 | lem | 0.81 | 1.38 | 1.43 | 3.2875 |  |
| 10 | 1B | PC | 90% | 2B | y7nhl0rc | 4 | H100 | lem | 1.60 | 1.38 | 1.41 | 3.1580 |  |
| 11 | 1B | PC | 90% | 4B | 0ibnc42n | 4 | H100 | lem | 3.16 | 1.38 | 1.39 | 3.0618 |  |
| 12 | 1B | PC | 90% | 6B | l6u6t80m | 4 | H100 | lem | 4.71 | 1.37 | 1.38 | 3.0197 |  |
| 13 | 1B | PCD | 50% | 1B | scfmfbqy | 4 | H100 | lem | 2.32 | 4.04 | 4.08 | 2.7338 |  |
| 14 | 1B | PCD | 50% | 2B | ic3zozd8 | 4 | H100 | lem | 4.61 | 4.03 | 4.06 | 2.6763 |  |
| 15 | 1B | PCD | 50% | 4B | 9mjpyltf | 4 | H100 | lem | 9.18 | 4.02 | 4.03 | 2.6251 |  |
| 16 | 1B | PCD | 50% | 6B | c3vqmfv7 | 4 | H100 | lem | 13.80 | 4.03 | 4.04 | 2.6023 |  |
| 17 | 1B | PCD | 70% | 1B | 45yj0mdp | 4 | H100 | lem | 2.00 | 3.46 | 3.51 | 2.8806 |  |
| 18 | 1B | PCD | 70% | 2B | pk6lz5t9 | 4 | H100 | lem | 3.96 | 3.46 | 3.48 | 2.8007 |  |
| 19 | 1B | PCD | 70% | 4B | j66vre7a | 4 | H100 | lem | 7.96 | 3.48 | 3.50 | 2.7363 |  |
| 20 | 1B | PCD | 70% | 6B | qef5mvf0 | 4 | H100 | lem | 12.63 | 3.69 | 3.70 | 2.7085 |  |
| 21 | 1B | PCD | 90% | 1B | xk83ntk1 | 4 | H100 | lem | 1.69 | 2.92 | 2.97 | 3.2527 |  |
| 22 | 1B | PCD | 90% | 2B | k9ndr8i1 | 4 | H100 | lem | 3.33 | 2.90 | 2.93 | 3.1312 |  |
| 23 | 1B | PCD | 90% | 4B | whk9g4c0 | 4 | H100 | lem | 6.65 | 2.91 | 2.92 | 3.0427 |  |
| 24 | 1B | PCD | 90% | 6B | ciw4ajxz | 4 | H100 | lem | 9.93 | 2.90 | 2.91 | 3.0052 |  |
| 25 | 1B | HP | 50% | 1B | hd8xh1ye | 4 | H100 | lem | 1.28 | 2.16 | 2.25 | 2.7683 |  |
| 26 | 1B | HP | 50% | 2B | 8y0ayn2t | 4 | H100 | lem | 2.51 | 2.16 | 2.21 | 2.6998 |  |
| 27 | 1B | HP | 50% | 4B | vxb3iquw | 4 | H100 | lem | 4.97 | 2.16 | 2.19 | 2.6375 |  |
| 28 | 1B | HP | 50% | 6B | m3pbg31v | 4 | H100 | lem | 7.42 | 2.16 | 2.17 | 2.6075 |  |
| 29 | 1B | HP | 70% | 1B | 3hs71dfy | 4 | H100 | lem | 1.00 | 1.66 | 1.76 | 2.9510 |  |
| 30 | 1B | HP | 70% | 2B | d697o7ip | 4 | H100 | lem | 1.94 | 1.66 | 1.70 | 2.8491 |  |
| 31 | 1B | HP | 70% | 4B | k6cqnpmp | 4 | H100 | lem | 3.88 | 1.68 | 1.70 | 2.7674 |  |
| 32 | 1B | HP | 70% | 6B | nfn49e2m | 4 | H100 | lem | 5.72 | 1.66 | 1.67 | 2.7299 |  |
| 33 | 1B | HP | 90% | 1B | mqlk0lvt | 4 | H100 | lem | 0.71 | 1.17 | 1.26 | 3.3882 |  |
| 34 | 1B | HP | 90% | 2B | ef584qrn | 4 | H100 | lem | 1.38 | 1.17 | 1.22 | 3.2207 |  |
| 35 | 1B | HP | 90% | 4B | hp2q65qm | 4 | H100 | lem | 2.72 | 1.17 | 1.20 | 3.0969 |  |
| 36 | 1B | HP | 90% | 6B | o23y1q3a | 4 | H100 | lem | 4.05 | 1.18 | 1.19 | 3.0481 |  |
| 37 | 1B | HPD | 50% | 1B | 904w73pp | 4 | H100 | lem | 2.07 | 3.57 | 3.64 | 2.7706 |  |
| 38 | 1B | HPD | 50% | 2B | 7fwudoud | 4 | H100 | lem | 4.10 | 3.57 | 3.60 | 2.6743 |  |
| 39 | 1B | HPD | 50% | 4B | t40l65yb | 4 | H100 | lem | 8.14 | 3.56 | 3.58 | 2.6340 |  |
| 40 | 1B | HPD | 50% | 6B | f9uv2mjb | 4 | H100 | lem | 12.19 | 3.56 | 3.57 | 2.6060 |  |
| 41 | 1B | HPD | 70% | 1B | xogsbrqa | 4 | H100 | lem | 1.79 | 3.08 | 3.14 | 2.9165 |  |
| 42 | 1B | HPD | 70% | 2B | 4yuyjuf9 | 4 | H100 | lem | 3.54 | 3.08 | 3.11 | 2.8199 |  |
| 43 | 1B | HPD | 70% | 4B | nzq3ghfr | 4 | H100 | lem | 7.06 | 3.08 | 3.10 | 2.7463 |  |
| 44 | 1B | HPD | 70% | 6B | 44wrh68u | 4 | H100 | lem | 10.54 | 3.08 | 3.09 | 2.7190 |  |
| 45 | 1B | HPD | 90% | 1B | hvd1i37o | 4 | H100 | lem | 1.51 | 2.60 | 2.66 | 3.4340 |  |
| 46 | 1B | HPD | 90% | 2B | pajazura | 4 | H100 | lem | 3.00 | 2.60 | 2.64 | 3.2448 |  |
| 47 | 1B | HPD | 90% | 4B | d95wedea | 4 | H100 | lem | 5.97 | 2.61 | 2.62 | 3.1095 |  |
| 48 | 1B | HPD | 90% | 6B | v4v1k0dd | 4 | H100 | lem | 8.89 | 2.60 | 2.61 | 3.0522 |  |
| 49 | 8B | PC | 50% | 1B | 60ix45vb | 4 | H100 | lem | 5.97 | 20.77 | 20.97 | 2.4316 | 3 identical-loss runs, earliest used |
| 50 | 8B | PC | 50% | 2B | qlukfr2z | 4 | H100 | lem | 11.79 | 20.59 | 20.72 | 2.3691 |  |
| 51 | 8B | PC | 50% | 4B | 1qal4vfq | 4 | H100 | lem | 23.53 | 20.62 | 20.68 | 2.3114 |  |
| 52 | 8B | PC | 50% | 6B | 11baikmr | 4 | H100 | lem | 37.55 | 21.96 | 22.00 | 2.2815 |  |
| 53 | 8B | PC | 70% | 1B | itekqp2v | 4 | H100 | lem | 4.73 | 16.37 | 16.63 | 2.6344 |  |
| 54 | 8B | PC | 70% | 2B | 4uj8gy0x | 4 | H100 | lem | 8.87 | 15.47 | 15.60 | 2.5275 |  |
| 55 | 8B | PC | 70% | 4B | o7lvmjqp | 4 | H100 | lem | 17.59 | 15.40 | 15.46 | 2.4427 |  |
| 56 | 8B | PC | 70% | 6B | 49yaqs7q | 4 | H100 | lem | 26.44 | 15.45 | 15.49 | 2.4019 |  |
| 57 | 8B | PC | 90% | 1B | v45fo4fk | 4 | H100 | lem | 3.03 | 10.43 | 10.64 | 3.3323 |  |
| 58 | 8B | PC | 90% | 2B | 4x63r2gg | 4 | H100 | lem | 5.98 | 10.46 | 10.52 | 3.0275 |  |
| 59 | 8B | PC | 90% | 4B | 4e41kuki | 4 | H100 | lem | 12.39 | 10.86 | 10.89 | 2.8526 |  |
| 60 | 8B | PC | 90% | 6B | 602sf8ph | 4 | H100 | lem | 17.84 | 10.43 | 10.45 | 2.7979 |  |
| 61 | 8B | PCD | 50% | 1B | zsxblhqv | 4 | H100 | lem | 9.04 | 31.56 | 31.77 | 2.4095 |  |
| 62 | 8B | PCD | 50% | 2B | d53qw8ka | 4 | H100 | lem | 18.01 | 31.57 | 31.66 | 2.3500 |  |
| 63 | 8B | PCD | 50% | 4B | lkiem9bm | 4 | H100 | lem | 36.01 | 31.57 | 31.65 | 2.2996 |  |
| 64 | 8B | PCD | 50% | 6B | y77e0mj2 | 4 | H100 | lem | 53.98 | 31.59 | 31.63 | 2.2737 |  |
| 65 | 8B | PCD | 70% | 1B | 70pc7s2s | 4 | H100 | lem | 5.60 | 19.47 | 19.70 | 2.6116 |  |
| 66 | 8B | PCD | 70% | 2B | i5saam18 | 4 | H100 | lem | 11.26 | 19.68 | 19.80 | 2.5061 |  |
| 67 | 8B | PCD | 70% | 4B | uo27a1in | 4 | H100 | lem | 22.44 | 19.66 | 19.72 | 2.4253 |  |
| 68 | 8B | PCD | 70% | 6B | 3nwt5gna | 4 | H100 | lem | 33.29 | 19.47 | 19.51 | 2.3884 |  |
| 69 | 8B | PCD | 90% | 1B | m4aghdz0 | 4 | H100 | lem | 6.45 | 22.48 | 22.66 | 3.3434 |  |
| 70 | 8B | PCD | 90% | 2B | o4xghwwp | 4 | H100 | lem | 12.87 | 22.53 | 22.62 | 3.0217 |  |
| 71 | 8B | PCD | 90% | 4B | vsw3bte2 | 4 | H100 | lem | 25.72 | 22.55 | 22.61 | 2.8253 |  |
| 72 | 8B | PCD | 90% | 6B | 2usdkh5e | 4 | H100 | lem | 38.50 | 22.53 | 22.56 | 2.7650 |  |
| 73 | 8B | HP | 50% | 1B | 0rot9xtk | 4 | GH200 120GB | helios | 4.35 | 14.52 | 15.30 | 2.4273 |  |
| 74 | 8B | HP | 50% | 2B | sj2yl60p | 4 | GH200 120GB | helios | 8.52 | 14.54 | 14.98 | 2.3650 | GPU info from SLURM config (gres/partition) |
| 75 | 8B | HP | 50% | 4B | cbzie04m | 4 | GH200 120GB | helios | 17.10 | 14.67 | 15.03 | 2.3068 |  |
| 76 | 8B | HP | 50% | 6B | 9d4is9p1 | 4 | H100 | lem | 25.34 | 14.73 | 14.84 | 2.2762 |  |
| 77 | 8B | HP | 70% | 1B | 55wbh2hf | 4 | GH200 120GB | helios | 3.39 | 11.01 | 11.92 | 2.6574 | 2 identical-loss runs, earliest used |
| 78 | 8B | HP | 70% | 2B | tc231b7c | 4 | GH200 120GB | helios | 6.58 | 11.13 | 11.56 | 2.5430 |  |
| 79 | 8B | HP | 70% | 4B | 4h06bv9s | 4 | GH200 120GB | helios | 12.85 | 11.08 | 11.30 | 2.4521 |  |
| 80 | 8B | HP | 70% | 6B | 4nu9lcaj | 4 | GH200 120GB | helios | 18.58 | 10.73 | 10.89 | 2.4075 |  |
| 81 | 8B | HP | 90% | 1B | 5dhq5nlz | 4 | H100 | lem | 1.71 | 5.73 | 6.00 | 3.4292 |  |
| 82 | 8B | HP | 90% | 2B | rlhzko13 | 4 | H100 | lem | 3.34 | 5.74 | 5.88 | 3.1570 |  |
| 83 | 8B | HP | 90% | 4B | o46hordq | 4 | H100 | lem | 7.30 | 6.35 | 6.42 | 2.9054 |  |
| 84 | 8B | HP | 90% | 6B | zjjh5z4x | 4 | H100 | lem | 9.88 | 5.74 | 5.79 | 2.8733 |  |
| 85 | 8B | HPD | 50% | 1B | ky3icpe9 | 4 | H100 | lem | 6.27 | 21.56 | 22.05 | 2.4079 |  |
| 86 | 8B | HPD | 50% | 2B | yja1644t | 4 | GH200 120GB | helios | 12.35 | 21.10 | 21.71 | 2.3492 | GPU info from SLURM config (gres/partition) |
| 87 | 8B | HPD | 50% | 4B | aisgpc9s | 4 | GH200 120GB | helios | 24.34 | 21.07 | 21.39 | 2.3011 | GPU info from SLURM config (gres/partition) |
| 88 | 8B | HPD | 50% | 6B | 6xuzuh19 | 4 | GH200 120GB | helios | 36.44 | 21.14 | 21.35 | 2.2705 |  |
| 89 | 8B | HPD | 70% | 1B | qh0vqjvy | 4 | GH200 120GB | helios | 5.32 | 17.48 | 18.70 | 2.6495 |  |
| 90 | 8B | HPD | 70% | 2B | dern8aya | 4 | GH200 120GB | helios | 10.44 | 17.75 | 18.35 | 2.5239 |  |
| 91 | 8B | HPD | 70% | 4B | 01w97j5h | 4 | GH200 120GB | helios | 20.16 | 17.41 | 17.72 | 2.4395 |  |
| 92 | 8B | HPD | 70% | 6B | pn9tmb1x | 4 | GH200 120GB | helios | 30.32 | 17.55 | 17.77 | 2.3967 |  |
| 93 | 8B | HPD | 90% | 1B | y2vjj8y1 | 4 | GH200 120GB | helios | 4.32 | 14.05 | 15.20 | 3.3989 |  |
| 94 | 8B | HPD | 90% | 2B | of37oonz | 4 | GH200 120GB | helios | 8.42 | 14.24 | 14.80 | 3.0794 |  |
| 95 | 8B | HPD | 90% | 4B | w3lvqyv5 | 4 | GH200 120GB | helios | 17.32 | 14.95 | 15.22 | 2.9189 |  |
| 96 | 8B | HPD | 90% | 6B | 58ze2apk | 4 | GH200 120GB | helios | 25.52 | 14.77 | 14.95 | 2.8223 |  |
| 97 | 8B | SVD | 50% | 1B | — | — | — | — | — | — | — | 2.5520 | run in separate framework, not in wandb |
| 98 | 8B | SVD | 50% | 2B | — | — | — | — | — | — | — | 2.5330 | run in separate framework, not in wandb |
| 99 | 8B | SVD | 50% | 4B | — | — | — | — | — | — | — | 2.5030 | run in separate framework, not in wandb |
| 100 | 8B | SVD | 50% | 6B | — | — | — | — | — | — | — | 2.3120 | run in separate framework, not in wandb |
| 101 | 8B | SVD | 70% | 1B | — | — | — | — | — | — | — | 2.7560 | run in separate framework, not in wandb |
| 102 | 8B | SVD | 70% | 2B | — | — | — | — | — | — | — | 2.7180 | run in separate framework, not in wandb |
| 103 | 8B | SVD | 70% | 4B | — | — | — | — | — | — | — | 2.6000 | run in separate framework, not in wandb |
| 104 | 8B | SVD | 70% | 6B | — | — | — | — | — | — | — | 2.4120 | run in separate framework, not in wandb |
| 105 | 8B | SVD | 90% | 1B | — | — | — | — | — | — | — | 3.3740 | run in separate framework, not in wandb |
| 106 | 8B | SVD | 90% | 2B | — | — | — | — | — | — | — | 3.2130 | run in separate framework, not in wandb |
| 107 | 8B | SVD | 90% | 4B | — | — | — | — | — | — | — | 3.1070 | run in separate framework, not in wandb |
| 108 | 8B | SVD | 90% | 6B | — | — | — | — | — | — | — | 3.0340 | run in separate framework, not in wandb |

Rows: 108 (48 × 1B, 60 × 8B).
No wandb run: 12 cells (all SVD).

## Time comparison: Projected Compression vs Hard Pruning

Overhead of PC over HP (and PCD over HPD) for the same model / ratio / token budget. `+X%` = PC-family run took X% longer. **Time %** uses total train time (wandb `_runtime`); **s/step %** uses steady-state seconds per step (excludes startup). ⚠ = the two runs used different hardware, so the % mixes method cost with GPU speed.

| Model | Ratio | Tokens | HP (h) | PC (h) | PC vs HP time | PC vs HP s/step | HPD (h) | PCD (h) | PCD vs HPD time | PCD vs HPD s/step | Hardware note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1B | 50% | 1B | 1.28 | 1.47 | +14.8% | +16.6% | 2.07 | 2.32 | +12.2% | +13.3% |  |
| 1B | 50% | 2B | 2.51 | 2.90 | +15.3% | +16.6% | 4.10 | 4.61 | +12.5% | +13.0% |  |
| 1B | 50% | 4B | 4.97 | 5.77 | +16.0% | +16.7% | 8.14 | 9.18 | +12.9% | +13.0% |  |
| 1B | 50% | 6B | 7.42 | 8.61 | +16.1% | +16.6% | 12.19 | 13.80 | +13.2% | +13.3% |  |
| 1B | 70% | 1B | 1.00 | 1.14 | +13.5% | +17.3% | 1.79 | 2.00 | +11.7% | +12.3% |  |
| 1B | 70% | 2B | 1.94 | 2.23 | +15.2% | +16.6% | 3.54 | 3.96 | +12.1% | +12.3% |  |
| 1B | 70% | 4B | 3.88 | 4.43 | +14.3% | +15.3% | 7.06 | 7.96 | +12.8% | +13.0% |  |
| 1B | 70% | 6B | 5.72 | 6.60 | +15.5% | +15.9% | 10.54 | 12.63 | +19.8% | +19.9% |  |
| 1B | 90% | 1B | 0.71 | 0.81 | +13.9% | +17.4% | 1.51 | 1.69 | +11.5% | +12.2% |  |
| 1B | 90% | 2B | 1.38 | 1.60 | +15.7% | +17.6% | 3.00 | 3.33 | +11.2% | +11.6% |  |
| 1B | 90% | 4B | 2.72 | 3.16 | +16.2% | +17.2% | 5.97 | 6.65 | +11.3% | +11.4% |  |
| 1B | 90% | 6B | 4.05 | 4.71 | +16.2% | +16.6% | 8.89 | 9.93 | +11.7% | +11.8% |  |
| 8B | 50% | 1B | 4.35 | 5.97 | +37.1% | +43.0% | 6.27 | 9.04 | +44.0% | +46.4% | ⚠ HP GH200 120GB@helios vs PC H100@lem |
| 8B | 50% | 2B | 8.52 | 11.79 | +38.4% | +41.6% | 12.35 | 18.01 | +45.8% | +49.7% | ⚠ HP GH200 120GB@helios vs PC H100@lem; ⚠ HPD GH200 120GB@helios vs PCD H100@lem |
| 8B | 50% | 4B | 17.10 | 23.53 | +37.6% | +40.6% | 24.34 | 36.01 | +48.0% | +49.8% | ⚠ HP GH200 120GB@helios vs PC H100@lem; ⚠ HPD GH200 120GB@helios vs PCD H100@lem |
| 8B | 50% | 6B | 25.34 | 37.55 | +48.2% | +49.1% | 36.44 | 53.98 | +48.1% | +49.4% | ⚠ HPD GH200 120GB@helios vs PCD H100@lem |
| 8B | 70% | 1B | 3.39 | 4.73 | +39.5% | +48.6% | 5.32 | 5.60 | +5.4% | +11.4% | ⚠ HP GH200 120GB@helios vs PC H100@lem; ⚠ HPD GH200 120GB@helios vs PCD H100@lem |
| 8B | 70% | 2B | 6.58 | 8.87 | +34.9% | +39.1% | 10.44 | 11.26 | +7.9% | +10.9% | ⚠ HP GH200 120GB@helios vs PC H100@lem; ⚠ HPD GH200 120GB@helios vs PCD H100@lem |
| 8B | 70% | 4B | 12.85 | 17.59 | +36.9% | +38.9% | 20.16 | 22.44 | +11.3% | +13.0% | ⚠ HP GH200 120GB@helios vs PC H100@lem; ⚠ HPD GH200 120GB@helios vs PCD H100@lem |
| 8B | 70% | 6B | 18.58 | 26.44 | +42.2% | +44.0% | 30.32 | 33.29 | +9.8% | +10.9% | ⚠ HP GH200 120GB@helios vs PC H100@lem; ⚠ HPD GH200 120GB@helios vs PCD H100@lem |
| 8B | 90% | 1B | 1.71 | 3.03 | +77.3% | +82.0% | 4.32 | 6.45 | +49.1% | +60.0% | ⚠ HPD GH200 120GB@helios vs PCD H100@lem |
| 8B | 90% | 2B | 3.34 | 5.98 | +79.0% | +82.3% | 8.42 | 12.87 | +52.8% | +58.2% | ⚠ HPD GH200 120GB@helios vs PCD H100@lem |
| 8B | 90% | 4B | 7.30 | 12.39 | +69.7% | +71.0% | 17.32 | 25.72 | +48.5% | +50.9% | ⚠ HPD GH200 120GB@helios vs PCD H100@lem |
| 8B | 90% | 6B | 9.88 | 17.84 | +80.6% | +81.7% | 25.52 | 38.50 | +50.9% | +52.5% | ⚠ HPD GH200 120GB@helios vs PCD H100@lem |

### Summary — mean s/step overhead per model × ratio (averaged over the 4 token budgets)

| Model | Ratio | PC vs HP | PCD vs HPD |
|---|---|---|---|
| 1B | 50% | +16.6% | +13.2% |
| 1B | 70% | +16.3% | +14.4% |
| 1B | 90% | +17.2% | +11.8% |
| 8B | 50% | +43.6% ⚠ (3/4 mixed HW) | +48.8% ⚠ (3/4 mixed HW) |
| 8B | 70% | +42.7% ⚠ (4/4 mixed HW) | +11.5% ⚠ (4/4 mixed HW) |
| 8B | 90% | +79.3% | +55.4% ⚠ (4/4 mixed HW) |
