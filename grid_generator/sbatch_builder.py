import shlex
from typing import Any

from omegaconf import DictConfig


def create_slurm_parameters(slurm_config: DictConfig) -> list[str]:
    def _as_sbatch_flag(key: str, value: Any) -> str:
        key = key.replace("_", "-")
        if value is True:
            return f"#SBATCH --{key}"
        value = shlex.quote(str(value))
        return f"#SBATCH --{key}={value}"

    lines = []
    for param in sorted(slurm_config):
        lines.append(_as_sbatch_flag(param, slurm_config[param]))

    return lines


def create_master_node_configuration() -> list[str]:
    return [
        "",
        "export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)",
        'if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then',
        "    export MASTER_PORT=$((40000 + ${SLURM_JOB_ID} % 10000))",
        "else",
        "    export MASTER_PORT=$((30000 + (${SLURM_JOB_ID} % 1250) * 8 + (${SLURM_ARRAY_TASK_ID} % 8)))",
        "fi",
    ]


def create_program_call(config_folder):
    return [
        "srun torchrun --nnodes=${SLURM_NNODES}\\",
        "  --nproc-per-node=${SLURM_GPUS_ON_NODE} \\",
        "  --rdzv-id=${SLURM_JOBID} \\",
        "  --rdzv-backend=c10d \\",
        "  --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \\",
        "  main.py \\",
        f"    --config-path={config_folder} \\",
        "    --config-name=config_${SLURM_ARRAY_TASK_ID}.yaml \\",
        "    +checkpoint_config.slurm_array_task_id=${SLURM_ARRAY_TASK_ID}"
    ]


def generate_sbatch_script(
    slurm_config, config_folder, n_experiments, venv_path, modules_to_add
) -> list[str]:
    lines = ["#!/bin/bash -l", ""]

    slurm_parameters = create_slurm_parameters(slurm_config)
    lines.append(f"#SBATCH --array=0-{n_experiments - 1}")

    lines.extend(slurm_parameters)

    lines.extend(create_master_node_configuration())

    if modules_to_add is not None:
        for module in modules_to_add:
            lines.append(f"module load {module}")

    lines.append(f"source {venv_path}")
    lines.extend(create_program_call(config_folder))

    with open("exp.job", "w") as f:
        f.write("\n".join(lines))
