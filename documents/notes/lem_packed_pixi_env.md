# lem: packed pixi env, unpacked to node-local disk per job

## Why

The lem grant `plgllmefficont3` (Lustre project 649 on `/lustre/pd03`) went over its file
quota: 1.03M files against a soft limit of 900,096, with the grace period expired. After
that every file create failed with "Disk quota exceeded", and Slurm jobs died at start
(exit 0:53, no slurm-*.out). Most of those files were per-user pixi envs, about 60k files
each, plus their package caches. Since pixi and uv hard-link env files into the cache, an
env and its cache together cost about one set of inodes.

lem jobs therefore keep no env on Lustre. Each `pixi.lock` version is packed once into a
single tar on Lustre, and every job unpacks it to its own node-local NVMe at start.

## Layout

```
/lustre/pd03/plgrid/plgllmefficont3/nano/pretraining/$USER/pixi_packs/   (infrastructure.pixi_pack_dir)
  nano-env-<hash>.tar           the env: conda packages + wheels, from pixi-pack 0.7.12
  nano-env-<hash>.pixi-unpack   the matching unpacker, a static binary
  builds/<timestamp>_<hash>/    one per pack job: the shipped pixi.lock, the two scripts, slurm-<id>.out
```

`<hash>` is the first 16 hex characters of `sha256(pixi.lock)`. Each pack adds about 6
files to Lustre.

## How it works

- **Pack**: `update_pixi.py` with a lem config. It sees `infrastructure.pixi_pack_dir` and
  goes to pack mode instead of `pixi install`. It copies the local `pixi.lock` and the two
  scripts into `builds/`, then runs `scripts/pixi_pack_env.sh` as a batch job (`sbatch --wait`,
  1 GPU). The script:
  1. builds a wheel from each sdist in the lock, after checking the sdist's sha256 against
     the lock. pixi-pack only packs wheels, and four of our deps ship as sdists:
     `antlr4-python3-runtime` (hydra/omegaconf needs it), `rouge-score`, `sqlitedict` and
     `word2number` (all three for lm-eval);
  2. runs `pixi exec pixi-pack` with `--ignore-pypi-non-wheel --inject <those wheels>`.
     pixi-pack reads `pixi.lock` (v6 is fine) and never re-solves it. The torch cu128
     wheel is downloaded from the URL written in the lock;
  3. downloads the static `pixi-unpack` release binary and checks it against a pinned sha256;
  4. publishes the two files, then tests them by unpacking with the job script and
     importing torch, hydra, lm_eval and the rest, and running `torchrun --help`. If the
     test fails, it deletes what it published.

  Downloads, caches and the test unpack go to the job's `$TMPDIR_LOCAL` and are deleted
  when the script exits. The tar is written to Lustre directly. If the tar already
  exists, the step does nothing; `+force=true` rebuilds it.

  The pack job runs for about a minute but can wait an hour in the GPU queue. With
  `+wait=false`, `update_pixi.py` submits it with `sbatch --parsable` and returns at once,
  printing its job id and the override `++infrastructure.slurm.dependency=afterok:<id>`.
  Pass that override to `run_exp.py`, together with `++infrastructure.slurm.kill_on_invalid_dep=yes`,
  and the training jobs queue right away and start once the pack has succeeded. If the
  pack fails its self-test they never start. Submit each dependent job with its own command,
  so one failed submission doesn't silently skip the next one.
- **Job**: the `lem.yaml` preamble exports `PIXI_PACK_DIR` and `PIXI_LOCK_HASH`, then
  sources `scripts/pixi_unpack_env.sh`. The hash comes from `${pixi_lock_hash:}`
  (`resolver/__init__.py`), which runs on the laptop when `exp.job` is generated. That
  makes it the lock of the code being submitted. The hash can't be computed in the job,
  because `run_exp.version_code` drops `pixi.lock` from experiment branches. The script:
  - exits the job if the tar for that hash is missing, if `$TMPDIR_LOCAL` is unset, or if
    the node-local disk is too small;
  - unpacks to `$TMPDIR_LOCAL/nano_env`. pixi-unpack installs at that path, so prefixes,
    shebangs (`torchrun`) and `activate.sh` are all correct. In a multi-node job every node
    unpacks its own copy, in one `srun --ntasks-per-node=1` step before training. The path
    is the same on every node, so the env activated on the batch node is valid on all of
    them;
  - sources `activate.sh` and checks that `python` is the env's;
  - sets an EXIT trap that deletes the env, on every node in a multi-node job.
- **Changing `pixi.lock`**: run `update_pixi.py` again before submitting. A job whose lock
  has no pack fails at its first line, with the hash in the error. Old packs can be
  removed with `rm pixi_packs/nano-env-<old hash>.*`.

## WCSS TMPDIR facts this relies on

From the [WCSS TMPDIR docs](https://man.e-science.pl/i/422):

- A single-node job on the lem GPU partitions gets `TMPDIR_LOCAL=/mnt/lscratch/slurm/$SLURM_JOB_ID`
  on NVMe, 40G by default. Ask for more with `--gres=gpu:hopper:4,storage:local:<N>G`,
  which in a config is the `infrastructure.slurm.gres` string. `$TMPDIR` points there too.
- **Multi-node jobs default to Lustre TMPDIR.** A multi-node config must ask for local
  disk itself: `gres: "gpu:hopper:<n>,storage:local:40G"`, as in
  `configs/jm/feature/h4_lem_packed_env_2node.yaml`. The script uses `$TMPDIR_LOCAL`, not
  `$TMPDIR`, so a job that forgot this stops at start rather than unpacking onto Lustre.
- **A finished job's TMPDIR, local disk included, is moved to
  `/lustre/tmp/slurm/finished_jobs/<jobid>` and kept 14 days**, unless the job was
  submitted with `--extra=FORCE_RM_TMPDIR`. Without the flag, every job would drop a
  60k-file env back onto Lustre. `lem.yaml` sets it (`slurm.extra`), and the EXIT trap
  deletes the env as a second line of defence. The flag means nothing a job writes to
  TMPDIR survives the job.

## Sizes and timings

**lem, measured 2026-11-25.** The line-3 MoE branch (torch 2.11 lock `bb3ff591bf09a721`)
ran first. Its pack job 6439972 used `/mnt/lscratch/slurm/<id>` as scratch and was accepted
with `--extra=FORCE_RM_TMPDIR`. It built a 4.02 GiB tar, and its test unpack took 14 s for a
5.4G env, with `cuda available: True` on an H100. The job ran 1:03 but waited about an
hour in the GPU queue. Four single-node training jobs (2 and 4 GPUs) then unpacked in
12-15 s each, with 40G free, and completed.

The table below is for `pixi.lock` `cf69ad40a59f5c76`, measured with the full pack script
(test unpack included) on a laptop (8 cores, NVMe) and in a CPU-only entropy srun (8 CPUs on
asusgpu4; pack dir on NFS home, unpack to NVMe `/tmp`).

| | |
|---|---|
| tar on shared storage | 3.92 GiB (28 conda packages + 216 wheels), plus the 15 MB unpacker |
| unpacked env | 7.5G, 66,657 files |
| node-local peak during unpack | ~11.7G, i.e. the extracted tar plus the env, about 3x the tar |
| unpack time | 36-41 s on entropy, 59-82 s on the laptop; 125 s with 2 entropy nodes reading the tar from NFS at once |
| whole pack job | 8-11 min on entropy, ~7 min on the laptop, mostly downloading |

The multi-node path was tested with a 2-node CPU-only entropy batch job (46662). Both nodes
unpacked, and a 2-node × 2-process `torchrun` launched as `exp.job` does it all-reduced over
gloo, with every rank running the unpacked env's Python. Both nodes were clean afterwards.
On lem it has not run yet: NCCL over InfiniBand across nodes is a separate thing to check
with the 2-node smoke config.

The default 40G `storage:local` covers it, and the unpack script fails early if fewer
than 4x the tar size is free. The same space is shared with anything else that uses
`$TMPDIR`, such as torch Inductor's compile cache, which is usually well under 1G.

## Adopting it on another branch

The change is self-contained. Bring over these files, by cherry-picking the commit that
added this note or copying them:

- `scripts/pixi_pack_env.sh`, `scripts/pixi_unpack_env.sh` (new)
- `configs/_cluster/lem.yaml`: `pixi_pack_dir`, `slurm.extra`, and the preamble lines
  from `PIXI_PACK_DIR` to `source scripts/pixi_unpack_env.sh`, which replace the
  `PIXI_HOME` / `pixi shell-hook` lines
- `resolver/__init__.py`: `pixi_lock_hash` and its `register_new_resolver` line
- `update_pixi.py`: `pack_remote_pixi` and the `pixi_pack_dir` dispatch
- optionally `tests/test_pixi_pack_env.py` and `configs/jm/feature/h4_lem_packed_env.yaml`
  (the test composes it), plus `h4_lem_packed_env_2node.yaml`

Then, from that branch, using any lem experiment config:

```
pixi run python update_pixi.py --config-path configs/<dir> --config-name <lem config>
```

Packs are per user (`$USER` in the path) and per lock, so branches with different locks
don't conflict. Running `scripts/lem_packed_env_test_plan.sh` step by step checks
everything. Once a smoke run passes, the old Lustre env and its caches can be deleted
(this is the step that frees the inodes): `$USER/pixi`, `$USER/cache/rattler`,
`$USER/cache/pixi` and `$USER/cache/uv` under `/lustre/pd03/plgrid/plgllmefficont3/nano/pretraining/`.

## Not covered

- Multi-node on lem itself: tested only on entropy CPU nodes so far (see above).
- pixi-pack doesn't verify the sha256 of the PyPI wheels it downloads; it does verify
  conda packages. The sdists we build are checked by the pack script.
- The pack job holds a GPU for about 20 minutes: the partition requires one, and the test
  uses it to check `torch.cuda.is_available()`.
- helios and entropy are unchanged and still use `PIXI_HOME` with `pixi install`.
