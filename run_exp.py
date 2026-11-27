#!/usr/bin/env python
import datetime
import logging
import os
import signal
import re
import time
from git import Repo
from contextlib import contextmanager
import copy
import getpass
from typing import Generator, Optional, cast
from fabric import Connection
import hydra
from omegaconf import DictConfig, OmegaConf
import paramiko.ssh_exception

from grid_generator.generate_configs import create_grid_config
from grid_generator.sbatch_builder import generate_sbatch_script
from main import dump_grid_configs, run
from src.definitions import RootSchema, check_point, resolve_n_steps, tokens_per_step
import resolver

logger = logging.getLogger(__name__)

# Remote failures that are worth naming rather than leaving as an exit code.
_REMOTE_DISK_FULL = re.compile(
    r"quota exceeded|No space left on device|Disk quota", re.IGNORECASE
)


def run_remote(connection, command, what):
    """Run a remote command, and say what went wrong when it is something known.

    fabric raises UnexpectedExit with the exit status and nothing else, so a cluster
    home that is over its disk quota surfaces as `exit 128` from `git clone` with the
    quota message buried in the remote stderr. configs/_cluster/lem.yaml carries a
    comment about exactly that, which is a sign it cost somebody an afternoon.

    It is a one-line diagnosis once you can see it, so show it.
    """
    result = connection.run(command, warn=True)
    if not result.failed:
        return result

    output = (result.stdout or "") + (result.stderr or "")
    if _REMOTE_DISK_FULL.search(output):
        raise RuntimeError(
            f"{what} failed on {connection.host}: the remote filesystem is out of "
            f"space or over its disk quota.\n"
            f"  command: {command}\n"
            f"  remote says: {output.strip().splitlines()[-1] if output.strip() else '(nothing)'}\n"
            f"Check the quota there (`quota -s`, `df -h`). If this is the cemetery "
            f"directory, infrastructure.cluster_cemetery_dir in the cluster config "
            f"redirects it to project storage; helios and lem already do that."
        )
    raise RuntimeError(
        f"{what} failed on {connection.host} with exit {result.exited}.\n"
        f"  command: {command}\n"
        f"  remote says: {output.strip() or '(nothing)'}"
    )


# A sparse_exclude entry goes into a shell command and a sparse-checkout pattern, so
# only plain top-level folder names are accepted (.github yes; ., .., .git and paths no).
_FOLDER_NAME = re.compile(r"[A-Za-z0-9_.][A-Za-z0-9_.-]*")
_NOT_A_FOLDER = {".", "..", ".git"}
# Top-level folders a job reads on the cluster: main.py imports src/ and resolver/
# and runs on generated_configs/, and the PC evals read configs/pc_project.
_NEEDED_BY_A_JOB = {"src", "resolver", "configs", "generated_configs"}


def sparse_exclude_names(names) -> list[str]:
    """infrastructure.git.sparse_exclude as a checked list of top-level folder names.

    Called before anything is pushed, so a bad entry fails on the laptop. A string
    is refused rather than read letter by letter: on the command line,
    `sparse_exclude=tests` arrives as one, and would exclude nothing.
    """
    if isinstance(names, str):
        raise ValueError(
            "infrastructure.git.sparse_exclude takes a list of folder names, got the "
            f"string {names!r}; write it as a list, e.g. sparse_exclude=[{names}]"
        )
    checked = []
    for name in names or []:
        name = str(name).strip("/")
        if not _FOLDER_NAME.fullmatch(name) or name in _NOT_A_FOLDER:
            raise ValueError(
                "infrastructure.git.sparse_exclude takes top-level folder names "
                f"(letters, digits, _ . -), got {name!r}"
            )
        if name in _NEEDED_BY_A_JOB:
            raise ValueError(
                f"infrastructure.git.sparse_exclude: {name!r} is read by jobs on the "
                "cluster, so the clone cannot leave it out"
            )
        checked.append(name)
    return checked


def clone_command(remote_url, branch, experiment_dir, sparse_exclude=()) -> str:
    """The shell command that clones an experiment's cemetery branch on the cluster.

    The cemetery branch on GitHub always holds the whole repo; only the clone on the
    cluster leaves out the top-level folders named in `sparse_exclude`
    (infrastructure.git.sparse_exclude, configs/_misc/default.yaml). A job runs main.py
    on generated_configs/, so it never reads tests/, scripts/ or documents/, while every
    clone counts once per experiment against the cluster's file quota. On LEM the group
    shares one quota of 1,048,576 files; on 2026-11-19 it was full, and eight runs failed
    at their end-of-run save.

    The clone uses git's classic sparse checkout (core.sparseCheckout, the patterns in
    .git/info/sparse-checkout, then read-tree) instead of `git sparse-checkout`, so it
    works with whatever git a cluster has. The left-out files stay in the clone's .git,
    and `git -C <dir> sparse-checkout disable` brings them back without the network.

    An empty `sparse_exclude` gives the plain full clone. The paths are not quoted:
    cemetery dirs carry a literal $USER for the remote shell to expand.
    """
    excluded = sparse_exclude_names(sparse_exclude)
    if not excluded:
        return f"git clone --depth 1 -b {branch} {remote_url} {experiment_dir}"
    patterns = "\\n".join(["/*"] + [f"!/{name}/" for name in excluded])
    return " && ".join(
        [
            f"git clone --depth 1 --no-checkout -b {branch} {remote_url} {experiment_dir}",
            f"git -C {experiment_dir} config core.sparseCheckout true",
            f"mkdir -p {experiment_dir}/.git/info",
            f"printf '{patterns}\\n' > {experiment_dir}/.git/info/sparse-checkout",
            f"git -C {experiment_dir} read-tree -mu HEAD",
        ]
    )


_SSH_HOSTS_TO_PASSPHRASES = {}


def commit_pending_changes(repo: Repo):
    if len(repo.index.diff("HEAD")) > 0:
        repo.git.commit(m="Versioning code", no_verify=True)


def reset_to_original_repo_state(
    repo: Repo,
    original_branch: str,
    original_branch_commit_hash: str,
    versioning_branch: str,
):
    repo.git.checkout(original_branch, "-f")
    if versioning_branch in repo.branches:
        repo.git.branch("-D", versioning_branch)
    repo.head.reset(original_branch_commit_hash, index=True)
    print("Successfully restored working tree to the original state!")


@contextmanager
def restoring_repo_state(
    repo: Repo,
    original_branch: str,
    original_branch_commit_hash: str,
    versioning_branch: str,
):
    """Restore the working tree on the way out, including on SIGTERM and SIGINT.

    A `finally` only runs when Python unwinds. The default SIGTERM handler kills the
    process outright, so a cancelled or preempted submission left the repo standing on
    the experiment branch with the original branch unrestored -- which is how
    `run_exp.py leaves the branch broken if killed` got into documents/issues.md.

    The handler restores, puts the previous handler back and re-raises the signal, so
    the process still dies the way the caller asked it to.
    """
    restored = False

    def _restore():
        nonlocal restored
        if restored:
            return
        restored = True
        reset_to_original_repo_state(
            repo, original_branch, original_branch_commit_hash, versioning_branch
        )

    previous = {}

    def _handler(signum, _frame):
        _restore()
        signal.signal(signum, previous.get(signum, signal.SIG_DFL))
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            previous[sig] = signal.signal(sig, _handler)
        except ValueError:
            # signal() only works on the main thread; leave the finally as the guard.
            pass
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        _restore()


def version_code(
    remote_url: str,
    experiment_config_path: Optional[str] = None,
    exp_job_path: Optional[str] = None,
    job_name: Optional[str] = None,
) -> str:
    repo = Repo(".", search_parent_directories=True)

    experiment_branch_name = (
        f"{job_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    original_branch = repo.active_branch.name
    original_branch_commit_hash = repo.head.object.hexsha

    repo.git.add(experiment_config_path, force=True)
    repo.git.add(exp_job_path, force=True)
    repo.git.add(all=True)

    # Remove pixi files from the *commit snapshot* for the experiment branch
    for fname in ("pixi.toml", "pixi.lock"):
        try:
            repo.git.rm("--cached", fname)
        except Exception:
            # ignore if file is not tracked / doesn't exist
            pass

    with restoring_repo_state(
        repo, original_branch, original_branch_commit_hash, experiment_branch_name
    ):
        commit_pending_changes(repo)
        repo.git.checkout(b=experiment_branch_name)
        repo.git.push(remote_url, experiment_branch_name)

    return experiment_branch_name


@contextmanager
def ConnectWithPassphrase(*args, **kwargs) -> Generator[Connection, None, None]:
    """Connect to a remote host using a passphrase if the key is encrypted. The passphrase is preserved for subsequent connections to the same host."""
    try:
        connection = Connection(*args, **kwargs)
        connection.run('echo "Connection successful."')
        yield connection
    except paramiko.ssh_exception.PasswordRequiredException:
        if connection.host not in _SSH_HOSTS_TO_PASSPHRASES:
            passphrase = getpass.getpass(
                f"SSH key encrypted, provide the passphrase ({connection.host}): "
            )
            _SSH_HOSTS_TO_PASSPHRASES[connection.host] = passphrase
        else:
            passphrase = _SSH_HOSTS_TO_PASSPHRASES[connection.host]
        kwargs["connect_kwargs"] = copy.deepcopy(
            kwargs.get("connect_kwargs", {})
        )  # avoid modifying the original connect_kwargs
        kwargs["connect_kwargs"]["passphrase"] = passphrase
        connection = Connection(*args, **kwargs)
        yield connection
    finally:
        connection.close()


def tmux_session_name(experiment_branch_name: str) -> str:
    """The tmux session of an experiment: its branch name with "." and ":" as "_".

    tmux reads "." in a target (`-t name.0`) as the window-pane separator and ":"
    as the session-window one, and it turns a "." into "_" when it creates the
    session. So a run named after a float learning rate (`..._lr9.0_...`, which
    every config loading /base_schema produces) could not be addressed: the
    submit failed after the clone, with nothing queued
    (documents/notes/archive/stage_2_report.md, decision 2).
    """
    return experiment_branch_name.replace(".", "_").replace(":", "_")


def wait_for_job_id(connection, tmux_pane, tries: int = 3):
    """
    Wait for a SLURM job ID to appear in the output of a tmux pane.

    Repeatedly checks the pane for a successful `sbatch` message and returns
    the job ID. Raises RuntimeError if an error is found or if no job ID
    appears after the given number of tries.
    """
    while tries > 0:
        output = connection.run(
            f"tmux capture-pane -pt {tmux_pane}.0", hide=True
        ).stdout

        match = re.search(r"Submitted batch job (\d+)", output)
        if not match:
            match_error = re.search(r"sbatch: error: (.*)\n", output)
            if not match_error:
                time.sleep(0.5)
                tries -= 1
                if tries == 0:
                    raise RuntimeError("Failed to get job ID from sbatch output.")
                continue
            else:
                err_msg = match_error.group(1)
                raise RuntimeError(f"Error submitting job: {err_msg}")
        else:
            job_id = match.group(1)
            break
    return job_id


def check_missing_keys(cfg: DictConfig) -> None:
    """Fail on any ??? left in the composed config, before anything is pushed.

    Checked on every grid point, each on its own copy with its token budget
    resolved. OmegaConf.missing_keys resolves every interpolation it walks past,
    and a length given in tokens leaves trainer.n_steps null until resolve_n_steps
    fills it in, so the warmup's ${trainer.n_steps} would fail on None. On the
    unexpanded config a sweep over the budget (`^n_tokens_b`) or over what a step
    holds (`^batch_size`, `^sequence_length`) could not be resolved at all
    (documents/notes/archive/review_issues.md, N5). cfg keeps n_tokens_b as written, for
    the grid expansion and the dump.
    """
    for point, overrides in create_grid_config(copy.deepcopy(cfg)):
        probe = OmegaConf.create(point)
        resolve_n_steps(probe)
        missing_keys: set[str] = OmegaConf.missing_keys(probe)
        if missing_keys:
            where = f" at grid point {overrides}" if overrides else ""
            raise RuntimeError(f"Got missing keys in config{where}:\n{missing_keys}")


@hydra.main(version_base=None, config_path="configs", config_name="exp")
def submit_experiment(
    cfg: RootSchema,
):
    check_missing_keys(cast(DictConfig, cfg))
    # Checked now, before version_code pushes anything; used by the remote clone.
    sparse_exclude = sparse_exclude_names(
        cfg.infrastructure.get("git", {}).get("sparse_exclude", None)
    )

    # cast: the runtime object is a DictConfig; RootSchema is Pylance sugar.
    configs_grid = create_grid_config(cast(DictConfig, cfg))
    # Schema-check every expanded grid point before anything is dumped or
    # submitted. `^key` grids are expanded by now, so a mistyped sweep key or
    # out-of-range value fails here, on the laptop, naming the point.
    for point_dict, point_overrides in configs_grid:
        try:
            checked = check_point(point_dict)
        except Exception as exc:
            raise RuntimeError(
                f"grid point {point_overrides or '(no overrides)'} failed the "
                f"schema check: {exc}"
            ) from exc
        # check_point resolved a token budget into steps on its own copy; show
        # the result before anything is queued. The dumped point keeps
        # n_tokens_b, and main.run resolves it the same way on the cluster.
        if checked.trainer.n_tokens_b is not None:
            per_step = tokens_per_step(
                checked.common.batch_size, checked.common.sequence_length
            )
            print(
                f"grid point {point_overrides or '(no overrides)'}: "
                f"trainer.n_tokens_b={checked.trainer.n_tokens_b} -> "
                f"trainer.n_steps={checked.trainer.n_steps} "
                f"({checked.trainer.n_steps * per_step:,} tokens trained on at "
                f"{per_step:,} per step, rounded up)"
            )
    dump_grid_configs(configs_grid, cfg.infrastructure.generated_configs_path)

    script = cfg.infrastructure.get("script", None)
    max_concurrent_jobs = cfg.infrastructure.get("max_concurrent_jobs", None)
    generate_sbatch_script(
        cfg.infrastructure.slurm,
        cfg.infrastructure.generated_configs_path,
        len(configs_grid),
        max_concurrent_jobs,
        script,
    )
    if cfg.infrastructure.server == "local":
        config, _overrides = configs_grid[0]
        omega_conf = OmegaConf.create(config)
        run(cast(RootSchema, omega_conf))
    else:
        experiment_branch_name = version_code(
            remote_url=cfg.infrastructure.git.remote_url,
            experiment_config_path=cfg.infrastructure.generated_configs_path,
            exp_job_path="exp.job",
            job_name=cfg.infrastructure.metric_logger.name,
        )

        with ConnectWithPassphrase(
            host=cfg.infrastructure.server, inline_ssh_env=True
        ) as connection:
            cemetery_dir = cfg.infrastructure.cemetery_experiments_dir
            run_remote(
                connection,
                f"mkdir -p {cemetery_dir}",
                f"Creating the cemetery directory {cemetery_dir}",
            )

            if "WANDB_API_KEY" in os.environ:
                connection.config["run"]["env"]["WANDB_API_KEY"] = os.environ[
                    "WANDB_API_KEY"
                ]

            experiment_dir = f"{cemetery_dir}/{experiment_branch_name}"
            if connection.run(f"test -d {experiment_dir}", warn=True).failed:
                run_remote(
                    connection,
                    clone_command(
                        cfg.infrastructure.git.remote_url,
                        experiment_branch_name,
                        experiment_dir,
                        sparse_exclude,
                    ),
                    f"Cloning the experiment into {experiment_dir}",
                )
            else:
                print(f"Experiment {experiment_branch_name} already exists. Skipping.")

            try:
                tmux_name = tmux_session_name(experiment_branch_name)
                connection.run(f"tmux new -d -s {tmux_name}")
                logger.info(
                    "Will try to replace the placeholders of the following env variables with values pulled from the local machine: %s",
                    resolver.ENV_VARS_TO_FORWARD,
                )
                for var in resolver.ENV_VARS_TO_FORWARD:
                    # We are using placeholders to avoid exposing secrets in the sbatch script pushed to git
                    if var not in os.environ:
                        logger.warning(
                            "%s not found in local environment variables. This might lead to issues. Skipping replacing the placeholder.",
                            var,
                        )
                    else:
                        connection.run(
                            f"sed -i 's/{resolver.env_var_name_to_placeholder(var)}/{os.environ[var]}/g' {experiment_dir}/exp.job"
                        )
                connection.run(
                    f'tmux send -t {tmux_name}.0 "cd {experiment_dir}" ENTER'
                )
                connection.run(f'tmux send -t {tmux_name}.0 "sbatch exp.job" ENTER')
                job_id = wait_for_job_id(connection, tmux_name)
                print(f"Job ID: {job_id}")
                connection.run(
                    f'tmux send -t {tmux_name}.0 "tail -f --retry slurm-{job_id}_0.out" ENTER'
                )
                LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
                if LOGLEVEL == "DEBUG":
                    connection.run(f"tmux attach-session -t {tmux_name}", pty=True)
            except Exception as e:
                print("Exception while running an experiment: ", e)


if __name__ == "__main__":
    submit_experiment()
