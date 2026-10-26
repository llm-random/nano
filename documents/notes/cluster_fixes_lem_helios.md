# Fix plan: lem and helios cluster blockers (1-3)

Written 2026-10-24, from the health check recorded in `merge_to_main.md`
("Lem / Helios health check + 1-GPU PC smoketests"). Covers only the three
**cluster-side** blockers. The two PC config API drifts found at the same time are repo
bugs present on `origin/main`, are already fixed in the `mas/nips-merge-prep` worktree,
and are out of scope here.

Both clusters currently work: lem job `6155713` and helios job `25023297` both COMPLETED
exit 0. Each of the three items below is either a workaround that should become a real
fix, or a stopgap with a hazard attached. **Nothing in this file has been executed.**
Every destructive step is marked and left to you.

Ordering: do **3** first (it has a live hazard), then **1**, then **2**. 2 is already
working for you and is only a hardening item.

---

## 1. lem `$HOME` is over quota

### Verified state (2026-10-23)

| fact | value |
|---|---|
| `quota -s`, the one `hnud:/plgrid-users/...` row | **28 822 M used / 20 480 M limit**, flagged `*`, grace expired |
| inode use on that row | 176 k / 1000 k -- not the binding constraint |
| `~/llmrandom_cemetery` | **13 G**, 253 clones, dated 2026-01-21 .. 2026-05-06 |
| `~/.local` | 7.1 G |
| `~/lmeval_env` | 5.3 G |
| `~/.cache` | 2.9 G |
| everything else in `$HOME` | < 0.3 G total |
| `/lustre/pd03` user and group quota | **none** (`limit 0`), 537 T free |
| `squeue -u plgmstefaniak` | empty |

The four directories above sum to ~28.3 G, which matches the 28 822 M the quota row
reports, so that row is `$HOME`'s quota.

> Caveat, unexplained: `df` reports `$HOME` on `hnud:/plgrid-users/plgmstefaniak`, while
> the quota row is labelled `hnud:/plgrid-users/plgprzewiez`. The usage figures match
> `$HOME`'s actual contents, so the attribution is right, but the label mismatch was not
> chased down. If a fix below does not move the number, re-check this first.

Symptom when it bites: `run_exp.py` fails with a bare
`invoke.exceptions.UnexpectedExit ... Exit code: 128`; the real message only appears if
you run the clone by hand -- `fatal: could not create work tree dir '...': Disk quota
exceeded`.

### What has to happen

To clear the over-quota state, **at least 8.4 G must leave `$HOME`** (more for headroom).
There are only two ways: remove data from `$HOME`, or get the quota raised. A symlink to
Lustre does **not** free anything by itself -- the bytes still sit in `$HOME` until the
original is removed. State that plainly before choosing.

### Option 1A (recommended): relocate the cemetery clones to Lustre

Copy first, verify, and only then remove. The removal is the single destructive step and
it is yours to run.

```bash
DST=/lustre/pd03/plgrid/plgllmefficont3/mstefaniak/llmrandom_cemetery

# 0. nothing may be running with a cwd inside these clones
squeue -u $USER

# 1. copy (resumable; writes to Lustre, so the full $HOME is not an obstacle)
rsync -a --info=progress2 ~/llmrandom_cemetery/ "$DST"/

# 2. verify before removing anything
diff <(cd ~/llmrandom_cemetery && find . | sort) \
     <(cd "$DST" && find . | sort) && echo TREES-MATCH
du -sb ~/llmrandom_cemetery "$DST"          # byte totals must match

# 3. rename aside (free, same filesystem) and point the old path at Lustre
mv ~/llmrandom_cemetery ~/llmrandom_cemetery.pre-move
ln -s "$DST" ~/llmrandom_cemetery

# 4. >>> DESTRUCTIVE, and the only step that actually frees quota <<<
#     run it once you are satisfied with step 2
rm -rf ~/llmrandom_cemetery.pre-move
quota -s                                     # the '*' should be gone
```

Frees 13 G, taking `$HOME` to roughly 15 G of the 20 G limit.

Note on step 3: wandb run directories live inside these clones, so the symlink matters --
it keeps every absolute path that was ever logged resolvable.

### Option 1B: ask WCSS to raise the quota

Deletes nothing. Worth doing in parallel with 1A if 253 clones are worth keeping in
`$HOME`-like storage. 20 G is small for a repo-clone-per-experiment workflow.

### Option 1C: if 1A is not enough

`~/lmeval_env` (5.3 G) relocates the same way. `~/.cache` (2.9 G) is regenerable. Leave
`~/.local` (7.1 G) alone unless you know what is in it -- relocating it tends to break
`pip --user` tooling.

### Prevent recurrence (no deletion, do this regardless)

Add to `configs/_cluster/lem.yaml`, under `infrastructure:`, so future submissions never
touch `$HOME`:

```yaml
  cluster_cemetery_dir: "/lustre/pd03/plgrid/plgllmefficont3/nano/$USER/llmrandom_cemetery"
```

This is lem-scoped and leaves entropy and helios on the `_misc/default.yaml` value
`~/llmrandom_cemetery`.

> **Not `$USER_HOME_PATH` -- but `$USER` is fine.** This value is used by `run_exp.py` on
> the *login node* (`mkdir -p`, then `git clone`), before `exp.job` runs, so none of the
> `infrastructure.script` exports exist yet: `$USER_HOME_PATH` would expand to empty.
> `$USER` is different -- it is set in every login shell, and this value is only ever
> interpolated into *remote shell* commands (`run_exp.py:204-205`), so the remote shell
> expands it. `~` works for the same reason.
>
> **As shipped it uses `$USER`**, which also removes the hardcoded `mstefaniak` problem
> that §2 describes.

### Verify

`quota -s` shows no `*`, then resubmit the lem smoketest and confirm the clone lands
under `/lustre/...` and the job reaches 10/10 steps.

---

## 2. lem shared HF cache is not writable by us

### Verified state

- `configs/_cluster/lem.yaml` exports `HF_HOME=$PROJECT_HOME_PATH/hf_cache`
  -> `/lustre/pd03/plgrid/plgllmefficont3/nano/hf_cache`.
- That directory is owner `plgggwozdz02`, group `plggllmeffi3`, mode `drwxr-sr-x`
  -- **no group write**. Its `hub/` holds only `models--gpt2`; no Llama.
- So `llama_tokenize_fn` -> `AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")`
  cannot create its cache entry and dies with
  `OSError: PermissionError at .../models--meta-llama--Llama-3.1-8B ... Check cache
  directory permissions. Common causes: 1) another user is downloading the same model
  ...; 2) a previous download was canceled and the lock file needs manual removal.`
  Both suggested causes are wrong; it is a permissions problem.
- `/lustre/pd03/plgrid/plgllmefficont3/mstefaniak/hf_cache` already holds
  `models--meta-llama--Llama-3.1-8B` with `tokenizer.json`, `tokenizer_config.json` and
  `special_tokens_map.json`, in the flat `models--org--name` hub layout.

### Already applied (in `mas/nips-merge-prep`, proven by job 6155713)

One line in `configs/_cluster/lem.yaml`:

```yaml
    - 'export HF_HUB_CACHE="$USER_HOME_PATH/hf_cache"'
```

`HF_HUB_CACHE` overrides the hub location independently of `HF_HOME`, so the shared
`HF_HOME` keeps working for anything else that reads it.

### Remaining weakness -- RESOLVED, see below

The original diagnosis: `USER_HOME_PATH` was hardcoded to
`/lustre/pd03/plgrid/plgllmefficont3/mstefaniak`, so another group member would point their
hub cache at **your** directory (itself `drwxr-sr-x`) and hit the same `PermissionError`
somewhere else. The reasoning at the time was that `$USER` could not be substituted,
because on lem the username is `plgmstefaniak` while that directory is `mstefaniak`.

**That reasoning applied to the old path, and no longer holds.** The shipped config does
not point at `<grant-root>/mstefaniak` at all -- it uses `<grant-root>/nano/$USER`, and
`nano/` already uses login-named per-user directories (`plgj321m`, `plgmtboro`,
`plgggwozdz02`). So `$USER` is exactly the right key there, and no account name remains in
any `configs/_cluster/*` file:

```yaml
    - 'export USER_HOME_PATH=$PROJECT_HOME_PATH/$USER'
```

Verified by expanding the exported `script:` block under two different `$USER` values.

**One migration step this implies:** the existing 414 G of data (pixi env, HF caches,
checkpoints) still lives at the old `<grant-root>/mstefaniak`, and **10
`configs/pc_project_mas/*.yaml` files hardcode that path** for checkpoints and evals. It
was therefore *not* moved. Instead a symlink makes both resolve:

```
/lustre/pd03/plgrid/plgllmefficont3/nano/plgmstefaniak -> ../mstefaniak
```

Nothing was copied, the 10 configs keep working, and it is reversible by removing the
symlink. Moving the data for real is a separate task that must update those configs in
lockstep.

> **Verify propagation before relying on this.** `exp.job` runs under `#!/bin/bash -l` and
> Slurm defaults to `--export=ALL`, so the submitting login shell's environment should
> carry through -- but that has not been tested on lem. One-line check: export a sentinel
> on the login node, submit, and grep the job's `env` dump for it.

### Do not

Do not `chmod` `/lustre/.../nano/hf_cache`; it is not ours. If the group wants a genuinely
shared cache, either ask plgggwozdz02 to `chmod -R g+w` it, or create a new
group-writable one (`chmod g+ws`) and point `HF_HUB_CACHE` there. Be aware that
concurrent downloads into a shared cache rely on HF's lock files, which is supported but
is exactly the situation the misleading "stale lock" message describes.

### Verify

Inside an `srun` on lem:
`python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')"`
should succeed with no network access.

---

## 3. helios `PIXI_HOME` does not exist -- for anyone

### Verified state

- `configs/_cluster/helios.yaml` sets `PIXI_HOME=$PROJECT_HOME_PATH/$USER/pixi`, with
  `PROJECT_HOME_PATH=$PLG_GROUPS_STORAGE/plggllmeffi3/pc`. Introduced by commit
  `7e1e115` "user dependent pixi paths (#174)".
- `$PROJECT_HOME_PATH` contains only `cache hf_cache hf_datasets hf_modules nano pixi`.
  There are **no per-user directories at all**, so this path is empty for every member,
  not just you. Nothing has run on helios since 2026-04-18.
- The working env is `$PROJECT_HOME_PATH/pixi`: `.pixi/envs/default`, aarch64,
  python 3.12, **torch 2.6.0+cu126**. Its `pixi.lock` is **version 6**. The `pixi` binary
  is dated 2026-04-20 and is **owned by `plgmstefaniak`** -- this "shared" env is yours,
  and the group uses it.
- `pixi.toml` pins, for `linux-aarch64`, `torch = "==2.6"` on the cu126 index -- exactly
  what that env has. So it matches the repo's aarch64 target.
- The helios **login node is x86_64** while compute is aarch64, so the cluster `pixi`
  binary cannot run on the login node (`Exec format error`). Its version could not be
  read; the v6 lockfile is the only version signal we have.
- Group storage: 11.29 TiB used of 29.30 TiB.

### Stopgap already applied (proven by job 25023297)

```
$PROJECT_HOME_PATH/plgmstefaniak/pixi -> $PROJECT_HOME_PATH/pixi
```

### >>> Hazard introduced by that stopgap -- read before anything else <<<

With the symlink in place, `python update_pixi.py` targeting helios resolves `PIXI_HOME`
**through it and onto the shared env**. `update_pixi.py` `mv`s the existing `pixi.toml`
and `pixi.lock` into `$PIXI_HOME/old_pixi_files/obsolete_since_<ts>/` and then runs
`pixi install` in place. That would mutate the environment the whole group depends on.

**Do not run `update_pixi.py` for helios while the symlink exists.** Removing the symlink
is the first step of Option 3A.

### Option 3A (recommended): build a real per-user env

```bash
# 1. ship a v6 lockfile -- `pixi run` rewrites it v6 -> v7 locally (pixi 0.81.0)
cd /home/maciej/code/nano-merge
git show HEAD:pixi.lock > pixi.lock

# 2. >>> DESTRUCTIVE (removes the symlink only, never the env) <<<
ssh helios 'rm /net/storage/pr3/plgrid/plggllmeffi3/pc/plgmstefaniak/pixi'

# 3. build it
python update_pixi.py --config-path configs/pc_project_mas \
                      --config-name smoketest_pc_1gpu_helios
```

Cautions:

- **Lockfile version.** `update_pixi.py` copies your *local* `pixi.lock`. Local pixi is
  0.81.0 and rewrites v6 -> **v7** on every `pixi run`. The helios env is on v6 and its
  binary is from April; it may not read v7. Either ship v6 as in step 1, or first read the
  remote version from inside an `srun` -- the only place that binary can execute:
  `srun -p plgrid-gpu-gh200 -A plgllmefficont3-gpu-gh200 --gres=gpu:1 -c 1 --mem=2G -t 00:02:00 pixi --version`
- It allocates a **gh200 node for a CPU-only install** -- the script drops only
  `gres`/`cpus_per_gpu`/`mem_per_gpu` and sets `cpus-per-task=4`, `mem=8G` -- and blocks
  until the install finishes.
- Space is fine (11.29 of 29.30 TiB used), but a full torch env is not small; check
  `hpc-fs` afterwards.

### Option 3B: make the shared env an explicit fallback

```yaml
    - 'export PIXI_HOME="${PIXI_HOME:-$PROJECT_HOME_PATH/pixi}"'
```

Smaller change, but it reverses the intent of `7e1e115`, and it leaves everyone sharing
one env -- which is what the group is doing in practice anyway. Same `--export=ALL`
propagation caveat as §2.

### Option 3C: ask first

`7e1e115` rolled out a per-user layout that nobody ever populated. Before committing to
either option, it is worth asking its author what was intended -- the answer decides
between 3A and 3B for the whole group, not just for you.

### Verify

Resubmit `configs/pc_project_mas/smoketest_pc_1gpu_helios.yaml`; expect COMPLETED with
20/20 steps and a loss trace near 22.0 -> 9.56.

---

## Summary of destructive steps (all left to you)

| § | step | what it removes |
|---|---|---|
| 1A.4 | `rm -rf ~/llmrandom_cemetery.pre-move` | the `$HOME` copy, **after** a verified Lustre duplicate exists |
| 3A.2 | `rm .../plgmstefaniak/pixi` | a symlink only; the env it points at is untouched |

Nothing else in this plan deletes anything.

---

# Execution log (2026-10-24/25)

Executed on instruction "fix those as planned". Five things in the plan above turned out to
be wrong or incomplete once run; they are corrected here and in the config.

## Corrections to the plan

**C1 -- `${VAR:-default}` cannot be used in `infrastructure.script` (§2, §3B).**
OmegaConf parses `${...}` inside these strings as an *interpolation*, not shell syntax --
the file already relies on that for `'${export_env_variables_placeholders:}'`. The
suggested `export USER_HOME_PATH="${USER_HOME_PATH:-...}"` makes every lem config fail to
compose with an `InterpolationResolutionError`. Use an explicit test instead:

```yaml
- 'if [ -z "$USER_HOME_PATH" ]; then export USER_HOME_PATH=/lustre/pd03/plgrid/plgllmefficont3/mstefaniak; fi'
```

**C2 -- the `--export=ALL` caveat is closed; it works.** Verified with two real lem jobs
(`6160840`, `6160841`) running the exact `if [ -z ... ]` line under `#!/bin/bash -l`:

| submitted with | job printed |
|---|---|
| `USER_HOME_PATH=/tmp/sentinel_override sbatch ...` | `USER_HOME_PATH=/tmp/sentinel_override` |
| plain `sbatch ...` | `USER_HOME_PATH=/lustre/.../mstefaniak` |

So another member exporting `USER_HOME_PATH` in their lem `~/.bashrc` does get it honoured,
and the hardcoded default still applies for everyone else.

**C3 -- putting `cemetery_experiments_dir` in `_cluster/lem.yaml` does not work (§1).**
`_misc` is listed *after* `_cluster` in every config's defaults list, so
`_misc/default.yaml`'s `~/llmrandom_cemetery` silently clobbers anything a cluster file
sets. Confirmed by composing: the key still resolved to `~/llmrandom_cemetery`.
Fixed with one indirection instead -- `_misc/default.yaml`:

```yaml
cemetery_experiments_dir: "${oc.select:infrastructure.cluster_cemetery_dir,'~/llmrandom_cemetery'}"
```

and cluster files set `cluster_cemetery_dir`. Verified after the change:

| config | resolves to |
|---|---|
| `smoketest_pc_1gpu_lem` | `/lustre/pd03/plgrid/plgllmefficont3/mstefaniak/llmrandom_cemetery` |
| `smoketest_pc_1gpu_helios` | `/net/storage/pr3/plgrid/plggllmeffi3/pc/plgmstefaniak/llmrandom_cemetery` |
| `tiny_remote` (entropy_a100) | `~/llmrandom_cemetery` -- unchanged |

**C4 -- §1's step order is wrong.** `$HOME` is so far over quota that it cannot allocate a
single new inode: `ln -s /tmp ~/.quota_probe_link` fails with `Disk quota exceeded`. So the
symlink of step 1A.3 **cannot be created until after the deletion** of step 1A.4. Corrected
order is: copy -> verify -> `mv` aside (a rename needs no new inode, so this still works)
-> delete -> `ln -s`.

**C5 -- §3A's lockfile caution was right, and is now quantified.** Remote helios pixi is
**0.67.1** (read from inside an `srun`, job `25058879`; the login node is x86_64 and cannot
execute the aarch64 binary). Local pixi is 0.81.0 and rewrites the lock to **v7**, which
0.67.1 will not read. Two things follow:
- ship the committed v6 lock (`git show HEAD:pixi.lock > pixi.lock`), and
- **do not invoke the launcher through `pixi run`** -- that rewrites the lock *before* the
  script reads it. Call the env's python directly:
  `./.pixi/envs/default/bin/python update_pixi.py ...`. Verified: imports resolve and the
  lock stays v6. This is a better default for `run_exp.py` too, and removes the recurring
  "restore pixi.lock afterwards" dance.

## Status

| item | state |
|---|---|
| §3 helios `PIXI_HOME` | **DONE.** Symlink removed; real per-user env built by `update_pixi.py` from the v6 lock ("Pixi environment updated successfully", srun job `25058952`). |
| §3 verification | **PASSED.** Job `25059418` COMPLETED 00:11:06 exit 0, 20/20 steps, loss 22.0 -> 9.5, wandb `09zv6a3f`. |
| §2 lem HF cache | **DONE** (`HF_HUB_CACHE`), plus C1/C2 hardening so other members can override. |
| §1 config half | **DONE** via C3; lem and helios both clone to group storage, entropy untouched. |
| §1 + §2 verification | **PASSED.** Lem job `6163167` COMPLETED 00:12:26 exit 0, 10/10 steps, loss 11.375 -> 8.000, wandb `jv6wi8i8`. Its clone landed on Lustre, so the `oc.select` indirection and the `HF_HUB_CACHE` fix both work on a real submit. |
| §1 quota half | **NOT DONE** -- see below. |

## §1 is not finished, and the last step is still yours

The copy of `~/llmrandom_cemetery` -> Lustre is the slow part: the tree is ~176k small
files on NFS, so it is metadata-bound at roughly **0.5 MB/s** serially -- hours for 13 G.
Parallelising it 10 ways coincided with lem refusing new SSH sessions for ~20 minutes.
That was **probably not** the cluster's fault: when it came back the login node's load
average was 0.04, and the failure was
`mux_client_request_session: read from master failed: Broken pipe` -- i.e. the *local*
SSH ControlMaster socket (`ControlMaster auto` / `ControlPersist 10m` in `~/.ssh/config`)
had wedged. If lem goes quiet mid-session, clear the stale master
(`rm ~/.ssh/cm-plgmstefaniak@ui.wcss.pl:22`) or connect with `-o ControlMaster=no` before
concluding the cluster is down. Even so, 10 concurrent rsyncs on a shared login node is
impolite; 3-4 is a reasonable ceiling:

```bash
ls -1 ~/llmrandom_cemetery | xargs -P 3 -I{} rsync -a "$HOME/llmrandom_cemetery/{}" \
  /lustre/pd03/plgrid/plgllmefficont3/mstefaniak/llmrandom_cemetery/
```

### The copy is DONE and verified (2026-10-25)

| check | result |
|---|---|
| `rsync -ani` dry run, src -> dst | **no output** -- every source file present and identical |
| set difference, files in src not in dst | **empty** |
| regular-file bytes | src `13 540 592 289` == dst `13 540 592 289` -- **exact match** |
| regular files | src 57 632, dst 57 633 |

The one extra destination file is
`llama1_pcd_memeff_2026-05-05_13-44-44/.aggregated_results_llama1b.json.ccAlFg`, an rsync
temp left behind when the first serial copy was stopped. Harmless, and Lustre has no quota;
remove it whenever you like. The three `smoketest_lem_pc90_1gpu_*` directories also exist
only on the destination -- they were cloned straight there and were never in `$HOME`.

So the data is safely duplicated. **Only the two final steps remain, and both are yours:**

```bash
DST=/lustre/pd03/plgrid/plgllmefficont3/mstefaniak/llmrandom_cemetery
mv ~/llmrandom_cemetery ~/llmrandom_cemetery.pre-move
rm -rf ~/llmrandom_cemetery.pre-move          # >>> DESTRUCTIVE, frees ~13G <<<
ln -s "$DST" ~/llmrandom_cemetery             # only works after the rm (see C4)
quota -s                                      # the '*' should be gone
```

If you would rather re-verify yourself first, the original recipe follows.

```bash
DST=/lustre/pd03/plgrid/plgllmefficont3/mstefaniak/llmrandom_cemetery
rsync -ani ~/llmrandom_cemetery/ "$DST"/      # empty output == complete
mv ~/llmrandom_cemetery ~/llmrandom_cemetery.pre-move
rm -rf ~/llmrandom_cemetery.pre-move          # >>> DESTRUCTIVE, yours <<<
ln -s "$DST" ~/llmrandom_cemetery             # only works after the rm (see C4)
quota -s                                      # the '*' should be gone
```

Nothing has been deleted. Submissions already work without this, because the config half
of §1 routes new clones to Lustre -- clearing the quota is what is still outstanding.
