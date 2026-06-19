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
    force_add_paths: Optional[list[str]] = None,
    job_name: Optional[str] = None,
) -> str:
    repo = Repo(".", search_parent_directories=True)

    experiment_branch_name = (
        f"{job_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    original_branch = repo.active_branch.name
    original_branch_commit_hash = repo.head.object.hexsha

    # force-add gitignored artifacts that must travel with the experiment branch
    for path in force_add_paths or []:
        repo.git.add(path, force=True)
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


def _fmt_n(n):
    if n is None:
        return "?"
    if not isinstance(n, (int, float)):
        return str(n)
    if abs(n) >= 1e9:
        return f"{n / 1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"{n / 1e6:.2f}M"
    if abs(n) >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


def _parse_n_gpu(slurm: dict) -> Optional[int]:
    if not slurm:
        return None
    nodes = slurm.get("nodes", 1) or 1
    gres = slurm.get("gres")
    gpus_per_node = None
    if isinstance(gres, str):
        m = re.search(r"gpu:(\d+)", gres)
        if m:
            gpus_per_node = int(m.group(1))
    if gpus_per_node is None:
        return None
    return nodes * gpus_per_node


def summarize_config(config: dict) -> dict:
    oc = OmegaConf.create(copy.deepcopy(config))
    try:
        OmegaConf.resolve(oc)
    except Exception:
        pass
    common = OmegaConf.to_container(oc.get("common", {}), resolve=True) or {}
    trainer = OmegaConf.to_container(oc.get("trainer", {}), resolve=True) or {}
    infra = OmegaConf.to_container(oc.get("infrastructure", {}), resolve=True) or {}

    n_steps = trainer.get("n_steps")
    bs = common.get("batch_size")
    seq = common.get("sequence_length")
    n_tokens = (
        n_steps * bs * seq
        if all(isinstance(x, (int, float)) for x in (n_steps, bs, seq))
        else None
    )

    # Approx params (notebook formula): attn = 2*(1 + kv/q) * d^2, ff = 3*d*dff*n_experts, ff_active = ff*top_k/n_experts.
    n_params_total, n_params_active = None, None
    try:
        n_blocks = common["n_blocks"]
        d = common["dmodel"]
        dff = common["dff"]
        q_heads = common["q_heads"]
        kv_heads = common["kv_heads"]
        vocab = common["vocab_size"]
        ff_layer = (
            ((oc.get("model") or {}).get("encoder") or {}).get("block_fn") or {}
        ).get("ff_layer_fn") or {}
        ff_layer = (
            OmegaConf.to_container(ff_layer, resolve=True)
            if not isinstance(ff_layer, dict)
            else ff_layer
        )
        n_experts = ff_layer.get("num_experts", 1) or 1
        top_k = ff_layer.get("topk", 1) or 1
        attn = 2 * (1 + kv_heads / q_heads) * d * d
        ff_total = 3 * d * dff * n_experts
        ff_active = ff_total * top_k / n_experts
        embed = vocab * d
        n_params_total = int(n_blocks * (attn + ff_total) + 2 * embed)
        n_params_active = int(n_blocks * (attn + ff_active) + embed)
    except Exception:
        pass

    ckpt_save = (trainer.get("checkpoint") or {}).get("save") or {}
    ckpt_path = ckpt_save.get("path")
    ckpt_steps_cfg = ckpt_save.get("steps") or []
    ckpt_interval = ckpt_save.get("interval")
    tokens_per_step = (
        bs * seq
        if isinstance(bs, (int, float)) and isinstance(seq, (int, float))
        else None
    )
    if ckpt_path is None:
        ckpt_step_list = None
    else:
        # Display in "training step count" convention: subtract 1 from explicit-list values
        # to align with periodic (interval) save points. Final stays at n_steps - 1.
        steps_set = set()
        for s in ckpt_steps_cfg:
            if n_steps is None or s < n_steps:
                steps_set.add(int(s) - 1)
        if ckpt_interval and ckpt_interval > 0 and n_steps:
            steps_set.update(range(int(ckpt_interval), n_steps, int(ckpt_interval)))
        if n_steps:
            steps_set.add(n_steps - 1)
        ckpt_step_list = sorted(steps_set)

    n_gpu = _parse_n_gpu(infra.get("slurm") or {})

    token_param_ratio = (
        n_tokens / n_params_active if n_tokens and n_params_active else None
    )
    return {
        "n_params_total": n_params_total,
        "n_params_active": n_params_active,
        "total_tokens": n_tokens,
        "token_param_ratio": token_param_ratio,
        "checkpoint_steps": ckpt_step_list,
        "tokens_per_step": tokens_per_step,
        "n_steps": n_steps,
        "n_gpu": n_gpu,
    }


def _format_ckpt_steps(steps, tokens_per_step, n_steps) -> str:
    if steps is None:
        return "disabled (path=null)"
    if not steps:
        return "none"
    final_step = n_steps - 1 if n_steps else None
    lines = []
    for s in steps:
        tag = " (final)" if s == final_step else ""
        tok = f"{_fmt_n(s * tokens_per_step)} tokens" if tokens_per_step else "?"
        lines.append(f"      step {s:<8} → {tok}{tag}")
    return "\n" + "\n".join(lines)


def format_summary(s: dict) -> str:
    is_moe = (
        s["n_params_total"] is not None
        and s["n_params_active"] is not None
        and s["n_params_total"] != s["n_params_active"]
    )
    if is_moe:
        size_line = f"  model size:       active={_fmt_n(s['n_params_active'])}, total={_fmt_n(s['n_params_total'])}"
    else:
        size_line = f"  model size:       {_fmt_n(s['n_params_total'])}"
    ratio = s["token_param_ratio"]
    ratio_str = f"{ratio:.2f}" if ratio is not None else "?"
    ckpt_str = _format_ckpt_steps(
        s["checkpoint_steps"], s["tokens_per_step"], s["n_steps"]
    )
    return "\n".join(
        [
            size_line,
            f"  tokens:           {_fmt_n(s['total_tokens'])}",
            f"  tok/active_param: {ratio_str}",
            f"  checkpoint steps:{ckpt_str}",
            f"  n_gpu:            {s['n_gpu']}",
        ]
    )


def print_grid_summary(configs_grid, output_folder: str):
    n = len(configs_grid)
    first_summary = summarize_config(configs_grid[0][0])
    print(f"\n=== Experiment summary ({n} config{'s' if n != 1 else ''}) ===")
    print("First config:")
    print(format_summary(first_summary))

    if n > 1:
        os.makedirs(output_folder, exist_ok=True)
        summary_path = os.path.join(output_folder, "summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            for i, (cfg_dict, overrides) in enumerate(configs_grid):
                s = summarize_config(cfg_dict)
                f.write(f"=== Config {i + 1}/{n} ===\n")
                if overrides:
                    f.write(f"overrides: {overrides}\n")
                f.write(format_summary(s) + "\n\n")
        print(f"\n{n} configs total — full summary written to: {summary_path}")
    print()


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
    configs_grid = create_grid_config(cfg)
    for config, _overrides in configs_grid:
        missing_keys: set[str] = OmegaConf.missing_keys(OmegaConf.create(config))
        if missing_keys:
            raise RuntimeError(f"Got missing keys in config:\n{missing_keys}")

    dump_grid_configs(configs_grid, cfg.infrastructure.generated_configs_path)
    print_grid_summary(configs_grid, cfg.infrastructure.generated_configs_path)

    if cfg.get("dry_run", False):
        print("dry_run=true — exiting after summary.")
        return

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
            force_add_paths=[cfg.infrastructure.generated_configs_path, "exp.job"],
            job_name=cfg.infrastructure.metric_logger.name,
        )

        with ConnectWithPassphrase(
            host=cfg.infrastructure.server, inline_ssh_env=True
        ) as connection:
            cemetery_dir = cfg.infrastructure.cemetery_experiments_dir
            connection.run(f"mkdir -p {cemetery_dir}")

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
