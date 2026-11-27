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

## Using it

From any branch, with any experiment config that has `_cluster@_here_: lem` in its defaults:

```
pixi run python update_pixi.py --config-path configs/<dir> --config-name <lem config> +dry_run=true
pixi run python update_pixi.py --config-path configs/<dir> --config-name <lem config>
pixi run python run_exp.py --config-path configs/<dir> --config-name <lem config>
```

- The dry run prints the lock hash, the pack path and the sbatch flags, and touches nothing.
- The second command packs this branch's `pixi.lock` in a 1-GPU job and waits for it. It
  ends with `✓ Packed .../nano-env-<hash>.tar`. If that lock is already packed, it returns
  at once; `+force=true` rebuilds it.
- Run it again whenever `pixi.lock` changes. A job whose lock has no pack stops at its
  first line, with the hash in the error.
- Packs are per user (`$USER` in the path) and per lock, so branches with different locks
  don't conflict.
- To avoid waiting while the pack job queues, add `+wait=false`. Then `update_pixi.py`
  submits it with `sbatch --parsable`, prints its job id and
  `++infrastructure.slurm.dependency=afterok:<id>`, and returns. Pass that override to
  `run_exp.py`, together with `++infrastructure.slurm.kill_on_invalid_dep=yes`. The
  training jobs then queue right away and start only once the pack has passed its
  self-test. Submit each dependent job with its own command, so one failed submission
  doesn't silently skip the next one.

**Moving off the old env.** Before this, lem jobs activated a per-user env installed on
Lustre. Once a job has run on the packed env, delete that env and its caches. This is the
step that frees the inodes:

```
rm -rf /lustre/pd03/plgrid/plgllmefficont3/nano/$USER/{pixi,cache/rattler,cache/pixi,cache/uv}
```

## Layout

```
/lustre/pd03/plgrid/plgllmefficont3/nano/$USER/pixi_packs/   (infrastructure.pixi_pack_dir)
  nano-env-<hash>.tar           the env: conda packages + wheels, from pixi-pack 0.7.12
  nano-env-<hash>.pixi-unpack   the matching unpacker, a static binary
  builds/<timestamp>_<hash>/    one per pack job: the shipped pixi.lock, the two scripts, slurm-<id>.out
```

`<hash>` is the first 16 hex characters of `sha256(pixi.lock)`. Each pack adds about 6
files to Lustre. Old packs can be removed with `rm pixi_packs/nano-env-<old hash>.*`.

## How it works

- **Pack**: `update_pixi.py` sees `infrastructure.pixi_pack_dir` in the config and goes to
  pack mode instead of `pixi install`. It copies the local `pixi.lock` and the two scripts
  into `builds/`, then runs `scripts/pixi_pack_env.sh` there as a batch job. The script:
  1. builds a wheel from each sdist in the lock, after checking the sdist's sha256 against
     the lock. pixi-pack only packs wheels. The sdists are found in the lock, so a branch
     that adds one needs no change here. Today there are four: `antlr4-python3-runtime`
     (hydra/omegaconf needs it), `rouge-score`, `sqlitedict` and `word2number` (all three
     for lm-eval);
  2. runs `pixi exec pixi-pack` with `--ignore-pypi-non-wheel --inject <those wheels>`.
     pixi-pack reads `pixi.lock` (v6 is fine) and never re-solves it. The torch cu128
     wheel is downloaded from the URL written in the lock;
  3. downloads the static `pixi-unpack` release binary and checks it against a pinned
     sha256. conda-forge's build links the env's OpenSSL, so it can't unpack that env;
  4. publishes the two files, then tests them by unpacking with the job script. The test
     checks that every PyPI package pinned in the lock is installed at its locked version,
     imports torch and reports whether CUDA is available, and runs `torchrun --help`. If
     the test fails, it deletes what it published.

  Downloads, caches and the test unpack go to the job's `$TMPDIR_LOCAL` and are deleted
  when the script exits. The tar is written to Lustre directly.
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

## When something fails

| Message | Where | Fix |
|---|---|---|
| `no packed env for this pixi.lock (<hash>)` | job log, first lines | Run `update_pixi.py` with that config, from the commit you submit. |
| `TMPDIR_LOCAL is unset` | job log | Multi-node job: add `storage:local:40G` to `infrastructure.slurm.gres`. |
| `need ~<N>G free under ...` | job log | Raise `storage:local:<N>G` in the gres string. |
| `not installed at the locked version: ...` | pack log in `builds/` | A wheel is missing from the pack; usually an sdist that failed to build in step 1. |
| `pixi-unpack checksum mismatch` | pack log | The release binary changed or the download broke; check `PIXI_UNPACK_SHA256`. |
| `Disk quota exceeded`, or exit 0:53 with no slurm-*.out | any Lustre write, job start | Project file quota: `lfs quota -h -p 649 /lustre/pd03`. |

## WCSS TMPDIR facts this relies on

From the [WCSS TMPDIR docs](https://man.e-science.pl/i/422):

- A single-node job on the lem GPU partitions gets `TMPDIR_LOCAL=/mnt/lscratch/slurm/$SLURM_JOB_ID`
  on NVMe, 40G by default. Ask for more with `--gres=gpu:hopper:4,storage:local:<N>G`,
  which in a config is the `infrastructure.slurm.gres` string. `$TMPDIR` points there too.
- **Multi-node jobs default to Lustre TMPDIR.** A multi-node config must ask for local
  disk itself: `gres: "gpu:hopper:<n>,storage:local:40G"`. The script uses `$TMPDIR_LOCAL`,
  not `$TMPDIR`, so a job that forgot this stops at start rather than unpacking onto Lustre.
- **A finished job's TMPDIR, local disk included, is moved to
  `/lustre/tmp/slurm/finished_jobs/<jobid>` and kept 14 days**, unless the job was
  submitted with `--extra=FORCE_RM_TMPDIR`. Without the flag, every job would drop a
  60k-file env back onto Lustre. `lem.yaml` sets it (`slurm.extra`), and the EXIT trap
  deletes the env as a second line of defence. The flag means nothing a job writes to
  TMPDIR survives the job.

## Sizes and timings

**lem, measured 2026-11-25** (a torch 2.11 lock). The pack job used `/mnt/lscratch/slurm/<id>`
as scratch and was accepted with `--extra=FORCE_RM_TMPDIR`. It built a 4.02 GiB tar, and its
test unpack took 14 s for a 5.4G env, with `cuda available: True` on an H100. The job ran
about a minute but waited about an hour in the GPU queue. Four single-node training jobs
(2 and 4 GPUs) then unpacked in 12-15 s each, with 40G free, and completed.

The table below is for a torch 2.7.1 lock, measured with the full pack script (test unpack
included) on a laptop (8 cores, NVMe) and in a CPU-only entropy srun (8 CPUs; pack dir on
NFS home, unpack to NVMe `/tmp`).

| | |
|---|---|
| tar on shared storage | 3.92 GiB (28 conda packages + 216 wheels), plus the 15 MB unpacker |
| unpacked env | 7.5G, 66,657 files |
| node-local peak during unpack | ~11.7G, i.e. the extracted tar plus the env, about 3x the tar |
| unpack time | 36-41 s on entropy, 59-82 s on the laptop; 125 s with 2 entropy nodes reading the tar from NFS at once |
| whole pack job | 8-11 min on entropy, ~7 min on the laptop, mostly downloading |

The default 40G `storage:local` covers it, and the unpack script fails early if fewer
than 4x the tar size is free. The same space is shared with anything else that uses
`$TMPDIR`, such as torch Inductor's compile cache, which is usually well under 1G.

The multi-node path was tested with a 2-node CPU-only entropy batch job. Both nodes
unpacked, and a 2-node × 2-process `torchrun` launched as `exp.job` did an all-reduce over
gloo, with every rank running the unpacked env's Python. Both nodes were clean afterwards.

## Not covered

- Multi-node on lem itself has not run yet: NCCL over InfiniBand across nodes is a
  separate thing to check with a 2-node smoke run.
- pixi-pack doesn't verify the sha256 of the PyPI wheels it downloads; it does verify
  conda packages. The sdists we build are checked by the pack script.
- The pack job takes a GPU, because the partition requires one; the test uses it to check
  `torch.cuda.is_available()`.
- helios and entropy are unchanged and still use `PIXI_HOME` with `pixi install`.
