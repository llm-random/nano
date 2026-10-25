#!/usr/bin/env python
import os
import logging
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from grid_generator.sbatch_builder import generate_eval_sbatch_script
from run_exp import (
    version_code,
    ConnectWithPassphrase,
    wait_for_job_id,
)
import resolver

logger = logging.getLogger(__name__)

EVAL_CONFIG_PATH = "configs/pc_project"
EVAL_CONFIG_NAME = "eval_hf_models"


@hydra.main(version_base=None, config_path=EVAL_CONFIG_PATH, config_name=EVAL_CONFIG_NAME)
def submit_eval(cfg: OmegaConf):
    config_name = HydraConfig.get().job.config_name
    script = cfg.infrastructure.get("script", None)
    generate_eval_sbatch_script(
        slurm_config=cfg.infrastructure.slurm,
        script=script,
        config_path=EVAL_CONFIG_PATH,
        config_name=config_name,
    )

    experiment_branch_name = version_code(
        remote_url=cfg.infrastructure.git.remote_url,
        experiment_config_path=EVAL_CONFIG_PATH,
        exp_job_path="exp.job",
        job_name=cfg.wandb.name,
    )

    with ConnectWithPassphrase(host=cfg.infrastructure.server, inline_ssh_env=True) as connection:
        cemetery_dir = cfg.infrastructure.cemetery_experiments_dir
        connection.run(f"mkdir -p {cemetery_dir}")

        if "WANDB_API_KEY" in os.environ:
            connection.config["run"]["env"]["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

        experiment_dir = f"{cemetery_dir}/{experiment_branch_name}"
        if connection.run(f"test -d {experiment_dir}", warn=True).failed:
            connection.run(
                f"git clone --depth 1 -b {experiment_branch_name} {cfg.infrastructure.git.remote_url} {experiment_dir}"
            )
        else:
            print(f"Experiment {experiment_branch_name} already exists. Skipping.")

        try:
            connection.run(f"tmux new -d -s {experiment_branch_name}")
            for var in resolver.ENV_VARS_TO_FORWARD:
                if var not in os.environ:
                    logger.warning("%s not found in environment, skipping placeholder replacement.", var)
                else:
                    connection.run(
                        f"sed -i 's/{resolver.env_var_name_to_placeholder(var)}/{os.environ[var]}/g' {experiment_dir}/exp.job"
                    )
            connection.run(f'tmux send -t {experiment_branch_name}.0 "cd {experiment_dir}" ENTER')
            connection.run(f'tmux send -t {experiment_branch_name}.0 "sbatch exp.job" ENTER')
            job_id = wait_for_job_id(connection, experiment_branch_name)
            print(f"Job submitted: {job_id}")
            connection.run(
                f'tmux send -t {experiment_branch_name}.0 "tail -f --retry slurm-{job_id}.out" ENTER'
            )
        except Exception as e:
            print("Exception while submitting eval job:", e)


if __name__ == "__main__":
    submit_eval()
