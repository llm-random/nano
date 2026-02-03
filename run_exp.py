#!/usr/bin/env python
import datetime
import logging
import os
import re
import time
from git import Repo
from contextlib import contextmanager
import copy
import getpass
from typing import Generator, Optional
from fabric import Connection
import hydra
from omegaconf import OmegaConf
import paramiko.ssh_exception

from grid_generator.generate_configs import create_grid_config
from grid_generator.sbatch_builder import generate_sbatch_script
from main import dump_grid_configs, run
import resolver

logger = logging.getLogger(__name__)

_SSH_HOSTS_TO_PASSPHRASES = {}


def ensure_remote_config_exist(repo: Repo, remote_name: str, remote_url: str):
    for remote in repo.remotes:
        if remote.name == remote_name:
            if remote.url != remote_url:
                old_remote_url = remote.url
                remote.set_url(remote_url)
                print(
                    f"Updated url of '{remote_name}' remote from '{old_remote_url}' to '{remote_url}'"
                )
            return

    repo.create_remote(remote_name, url=remote_url)
    print(f"Added remote '{remote_name}' with url '{remote_url}'")


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

    try:
        commit_pending_changes(repo)
        repo.git.checkout(b=experiment_branch_name)
        repo.git.push(remote_url, experiment_branch_name)
    finally:
        reset_to_original_repo_state(
            repo, original_branch, original_branch_commit_hash, experiment_branch_name
        )

    return experiment_branch_name


@contextmanager
def ConnectWithPassphrase(*args, **kwargs) -> Generator[Connection, None, None]:
    """Connect to a remote host using a passphrase if the key is encrypted. The passphrase is preserved for subsequent connections to the same host."""
    try:
        connection = Connection(*args, **kwargs)
        connection.run('echo "Connection successful."')
        yield connection
    except paramiko.ssh_exception.PasswordRequiredException as e:
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


def get_experiment_components(
    hydra_config: OmegaConf,
) -> str:
    # this is a workaround as hydra does not provide a way to get the config path
    # https://github.com/facebookresearch/hydra/discussions/2750
    config_name = hydra_config.job.config_name
    config_path = [
        path["path"]
        for path in hydra_config.runtime.config_sources
        if path["schema"] == "file"
    ][0]
    return config_path, config_name


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


@hydra.main(version_base=None, config_path="configs", config_name="exp")
def submit_experiment(
    cfg: OmegaConf,
):
    missing_keys: set[str] = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Got missing keys in config:\n{missing_keys}")

    configs_grid = create_grid_config(cfg)
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
        run(omega_conf)
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
            connection.run(f"mkdir -p {cemetery_dir}")

            if "NEPTUNE_API_TOKEN" in os.environ:
                connection.config["run"]["env"]["NEPTUNE_API_TOKEN"] = os.environ[
                    "NEPTUNE_API_TOKEN"
                ]

            if "WANDB_API_KEY" in os.environ:
                connection.config["run"]["env"]["WANDB_API_KEY"] = os.environ[
                    "WANDB_API_KEY"
                ]

            experiment_dir = f"{cemetery_dir}/{experiment_branch_name}"
            if connection.run(f"test -d {experiment_dir}", warn=True).failed:
                connection.run(
                    f"git clone --depth 1 -b {experiment_branch_name} {cfg.infrastructure.git.remote_url} {experiment_dir}"
                )
            else:
                print(f"Experiment {experiment_branch_name} already exists. Skipping.")

            try:
                connection.run(f"tmux new -d -s {experiment_branch_name}")
                print(
                    f"Will try to replace the placeholders of the following env variables with values pulled from the local machine: {resolver.ENV_VARS_TO_FORWARD}"
                )
                for var in resolver.ENV_VARS_TO_FORWARD:
                    if var not in os.environ:
                        print(
                            f"Warning: {var} not found in local environment variables. This might lead to issues. Skip replacing the placeholder."
                        )
                    else:
                        connection.run(
                            f"sed -i 's/{resolver.env_var_name_to_placeholder(var)}/{os.environ[var]}/g' {experiment_dir}/exp.job"
                        )
                connection.run(
                    f'tmux send -t {experiment_branch_name}.0 "cd {experiment_dir}" ENTER'
                )
                connection.run(
                    f'tmux send -t {experiment_branch_name}.0 "sbatch exp.job" ENTER'
                )
                job_id = wait_for_job_id(connection, experiment_branch_name)
                print(f"Job ID: {job_id}")
                connection.run(
                    f'tmux send -t {experiment_branch_name}.0 "tail -f --retry slurm-{job_id}_0.out" ENTER'
                )
                LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
                if LOGLEVEL == "DEBUG":
                    connection.run(
                        f"tmux attach-session -t {experiment_branch_name}", pty=True
                    )
            except Exception as e:
                print("Exception while running an experiment: ", e)


if __name__ == "__main__":
    submit_experiment()
