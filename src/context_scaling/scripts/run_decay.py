import os
import copy
import datetime
import warnings
import logging
from pathlib import Path
from typing import Optional
import argparse
import json
import yaml

from wandb_utils import (
    get_wandb_table,
    save_yaml_config_from_row,
)

logger = logging.getLogger(__name__)


def find_checkpoint_steps(ckpt_path: str) -> list[int]:
    """List all step_* checkpoint directories, sorted ascending."""
    ckpt_dir = Path(ckpt_path)
    if not ckpt_dir.exists():
        warnings.warn(f"Checkpoint directory not found: {ckpt_dir}")
        return []

    steps = []
    for name in os.listdir(ckpt_dir):
        if name.startswith("step_"):
            try:
                steps.append(int(name.split("_", 1)[1]))
            except ValueError:
                continue
    steps.sort()
    return steps


def build_decay_config(
    base_config: dict,
    ckpt_load_path: str,
    source_step: int,
    decay_steps: int,
    save_base_path: Optional[str],
    train_data_seed: Optional[int],
    eval_config: Optional[dict],
) -> dict:
    """Modify a training config for a pure-decay run from a checkpoint.

    decay_steps == 0 produces an eval-only job: no LR schedule change, no
    checkpoint save, training loop runs zero iterations, and the trainer's
    post-loop final_lm_eval hook fires the evaluator once.
    """
    cfg = copy.deepcopy(base_config)
    eval_only = decay_steps == 0

    # Backfill for base runs submitted before the final_lm_eval field existed.
    cfg["trainer"].setdefault("final_lm_eval", False)

    # Training runs from source_step to source_step + decay_steps (== source_step for eval-only)
    cfg["trainer"]["n_steps"] = source_step + decay_steps

    if not eval_only:
        original_train_seed = cfg["trainer"]["train_dataloader"]["dataset"]["seed"]
        assert train_data_seed is not None and train_data_seed != original_train_seed, (
            f"train_data_seed ({train_data_seed}) must be set and differ from the "
            f"base run's training data seed ({original_train_seed})."
        )
        cfg["trainer"]["train_dataloader"]["dataset"]["seed"] = train_data_seed

        # Pure linear decay from peak LR to 0
        cfg["trainer"]["scheduler"] = {
            "_partial_": True,
            "_target_": "torch.optim.lr_scheduler.LinearLR",
            "start_factor": 1.0,
            "end_factor": 0.0,
            "total_iters": decay_steps,
        }

    # Custom evaluator override + post-training eval hook
    if eval_config is not None:
        cfg["evaluator"] = copy.deepcopy(eval_config)
        cfg["trainer"]["final_lm_eval"] = True

    # Add "decay" tag so these runs don't get picked up by the same query
    tags = cfg.get("infrastructure", {}).get("metric_logger", {}).get("tags", [])
    if "decay" not in tags:
        tags.append("decay")

    save_block = {
        "type": "nano",
        "interval": -1,
        "path": None if eval_only else save_base_path,
        "model_checkpoint_filename": "__model_checkpoint_filename.pt",
        "training_state_filename": "__training_state_filename.pt",
    }

    cfg["trainer"]["checkpoint"] = {
        "load": {
            "type": "nano",
            "path": ckpt_load_path,
            "model_checkpoint_filename": "__model_checkpoint_filename.pt",
            "training_state_filename": "__training_state_filename.pt",
            "only_weights": False,
            "reset_scheduler": not eval_only,
            "rewind_data": False,
        },
        "save": save_block,
    }

    return cfg


def dump_decay_configs(configs, config_dir):
    """Write decay configs to disk in the same format as run_exp.py's dump_grid_configs."""
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)

    class CustomDumper(yaml.SafeDumper):
        def write_line_break(self, data=None):
            super().write_line_break(data)
            if len(self.indents) == 1:
                super().write_line_break()

    for idx, cfg_dict in enumerate(configs):
        cfg_dict["_run_"] = True
        cfg_dict.setdefault("overrides", [])
        cfg_dict.pop("checkpoint_config", None)

        out_path = config_dir / f"config_{idx}.yaml"
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg_dict, f, Dumper=CustomDumper, sort_keys=True)


def load_eval_config(eval_config_path: Optional[str]) -> Optional[dict]:
    """Load an override for the `evaluator:` config block from a YAML file.

    The file must contain a top-level `evaluator:` key; its value replaces the
    base run's evaluator block in every generated decay config.
    """
    if eval_config_path is None:
        return None
    path = Path(eval_config_path)
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict) or "evaluator" not in loaded:
        raise ValueError(
            f"Eval config {path} must contain a top-level 'evaluator:' key."
        )
    return loaded["evaluator"]


def generate_configs(args):
    """Fetch runs from wandb and generate decay configs. Returns (configs, infrastructure)."""
    df = get_wandb_table(tags=args.tags, negative_tags=args.negative_tags)
    if df.empty:
        print("No runs found. Exiting.")
        return [], None

    yaml_dir = Path(args.out_dir) / "yaml_cache"
    eval_config = load_eval_config(args.eval_config)
    eval_only = args.decay_fraction == 0.0

    configs = []
    infrastructure = None

    for _, row in df.iterrows():
        run_id = str(row["sys/id"])
        run_name = str(row.get("sys/name", run_id))
        ckpt_path = str(row.get("summary/full_save_checkpoints_path", ""))

        if not ckpt_path:
            warnings.warn(f"Run {run_id} ({run_name}): no checkpoint path, skipping.")
            continue

        # Reconstruct original config from wandb
        yaml_path = yaml_dir / f"{run_id}.yaml"
        if not yaml_path.exists() or yaml_path.stat().st_size == 0:
            save_yaml_config_from_row(row, yaml_path)

        with open(yaml_path, "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f)

        # Use infrastructure from first run for sbatch generation
        if infrastructure is None:
            infrastructure = base_config.get("infrastructure", {})

        # Determine which checkpoint steps to decay from
        if args.steps is not None:
            steps_to_decay = sorted(args.steps)
        else:
            all_steps = find_checkpoint_steps(ckpt_path)
            min_needed = 1 if eval_only else 2
            if len(all_steps) < min_needed:
                warnings.warn(
                    f"Run {run_id} ({run_name}): need at least {min_needed} checkpoint(s), "
                    f"found {len(all_steps)}. Skipping."
                )
                continue
            # Eval-only: evaluate every available checkpoint.
            # Decay: skip the last one (base run already covers that endpoint).
            steps_to_decay = all_steps if eval_only else all_steps[:-1]

        print(
            f"Run {run_id} ({run_name}): {len(steps_to_decay)} decay jobs "
            f"(steps {steps_to_decay}, decay_fraction={args.decay_fraction})"
        )

        for step in steps_to_decay:
            if eval_only:
                decay_steps = 0
            else:
                # decay_steps = f * (source_step + decay_steps) → decay_steps = f * source_step / (1 - f)
                decay_steps = int(
                    args.decay_fraction * step / (1 - args.decay_fraction)
                )

            step_ckpt_path = f"{ckpt_path}/step_{step}"

            save_path = None
            if args.save_ckpt_base and not eval_only:
                save_path = f"{args.save_ckpt_base}/{run_id}/from_step_{step}"

            decay_cfg = build_decay_config(
                base_config=base_config,
                ckpt_load_path=step_ckpt_path,
                source_step=step,
                decay_steps=decay_steps,
                save_base_path=save_path,
                train_data_seed=args.train_data_seed,
                eval_config=eval_config,
            )
            configs.append(decay_cfg)

    return configs, infrastructure


def submit(infrastructure, config_dir, n_experiments, job_name):
    """Version code, SSH to cluster, and submit the sbatch job."""
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    from grid_generator.sbatch_builder import generate_sbatch_script
    from run_exp import version_code, ConnectWithPassphrase, wait_for_job_id
    import resolver

    slurm_config = infrastructure.get("slurm", {})
    script = infrastructure.get("script", None)
    max_concurrent_jobs = infrastructure.get("max_concurrent_jobs", None)

    generate_sbatch_script(
        slurm_config,
        str(config_dir),
        n_experiments,
        max_concurrent_jobs,
        script,
    )

    server = infrastructure.get("server")
    if server == "local":
        raise ValueError("Local execution not supported for decay runs.")

    remote_url = infrastructure["git"]["remote_url"]
    experiment_branch_name = version_code(
        remote_url=remote_url,
        experiment_config_path=str(config_dir),
        exp_job_path="exp.job",
        job_name=job_name,
    )

    cemetery_dir = infrastructure.get(
        "cemetery_experiments_dir", "~/llmrandom_cemetery"
    )

    with ConnectWithPassphrase(host=server, inline_ssh_env=True) as connection:
        connection.run(f"mkdir -p {cemetery_dir}")

        if "WANDB_API_KEY" in os.environ:
            connection.config["run"]["env"]["WANDB_API_KEY"] = os.environ[
                "WANDB_API_KEY"
            ]

        experiment_dir = f"{cemetery_dir}/{experiment_branch_name}"
        if connection.run(f"test -d {experiment_dir}", warn=True).failed:
            connection.run(
                f"git clone --depth 1 -b {experiment_branch_name} {remote_url} {experiment_dir}"
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
                if var not in os.environ:
                    logger.warning(
                        "%s not found in local environment variables. Skipping placeholder replacement.",
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
        except Exception as e:
            print("Exception while running experiment: ", e)


def main():
    parser = argparse.ArgumentParser(
        description="Generate and submit decay-phase training jobs from intermediate checkpoints."
    )
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--negative_tags", nargs="+", default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument(
        "--decay_fraction",
        type=float,
        default=0.1,
        help="Fraction of original training steps to use for decay (default: 0.1).",
    )
    parser.add_argument(
        "--save_ckpt_base",
        type=str,
        default=None,
        help="Base path for saving decay run checkpoints. If not set, decay runs won't save checkpoints.",
    )
    parser.add_argument(
        "--train_data_seed",
        type=int,
        default=None,
        help="Seed for the training data stream in decay runs. Shared across all "
        "generated jobs; must differ from the base run's training data seed. "
        "Required when --decay_fraction > 0; ignored when --decay_fraction == 0.",
    )
    parser.add_argument(
        "--eval_config",
        type=str,
        default=None,
        help="Path to a YAML file with a top-level 'evaluator:' block. When set, "
        "it replaces the base run's evaluator in every generated job and enables "
        "trainer.final_lm_eval so the evaluator fires once at the end of training. "
        "Combine with --decay_fraction 0.0 for eval-only sweeps.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=None,
        help="Specific checkpoint steps to decay from. If not set, uses all checkpoints except the last.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Version code, push to cluster, and submit via sbatch.",
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="decay",
        help="Job name for sbatch and git branch (default: 'decay').",
    )
    parser.add_argument(
        "--max_concurrent_jobs",
        type=int,
        default=None,
        help="Max concurrent slurm array jobs. Overrides the value from the source run config.",
    )
    parser.add_argument(
        "--slurm_time",
        type=str,
        default=None,
        help="Slurm time limit (e.g. '0-12:00:00'). Overrides the value from the source run config.",
    )

    args = parser.parse_args()

    if args.decay_fraction > 0.0 and args.train_data_seed is None:
        parser.error("--train_data_seed is required when --decay_fraction > 0.")
    if args.decay_fraction == 0.0 and args.eval_config is None:
        parser.error(
            "--decay_fraction 0.0 only makes sense with --eval_config "
            "(otherwise the job has nothing to do)."
        )

    if args.out_dir is None:
        now = datetime.datetime.now()
        args.out_dir = str(
            Path("outputs") / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
        )
    out_dir = Path(args.out_dir)
    config_dir = out_dir / "generated_configs"

    configs, infrastructure = generate_configs(args)
    if not configs:
        return

    dump_decay_configs(configs, config_dir)

    # Write jobs metadata
    records = []
    for idx, cfg in enumerate(configs):
        records.append(
            {
                "array_idx": idx,
                "ckpt_load_path": cfg["trainer"]["checkpoint"]["load"]["path"],
                "decay_steps": cfg["trainer"]["n_steps"],
                "config_path": str(config_dir / f"config_{idx}.yaml"),
            }
        )

    jobs_path = out_dir / "jobs.json"
    with open(jobs_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\n{len(configs)} total decay jobs")
    print(f"  configs: {config_dir}")
    print(f"  jobs metadata: {jobs_path}")

    if args.submit:
        if infrastructure is None:
            raise RuntimeError("No infrastructure config found in wandb runs.")
        if args.max_concurrent_jobs is not None:
            infrastructure["max_concurrent_jobs"] = args.max_concurrent_jobs
        if args.slurm_time is not None:
            infrastructure.setdefault("slurm", {})["time"] = args.slurm_time
        submit(infrastructure, config_dir, len(configs), args.job_name)
    else:
        print("\nDry run. Pass --submit to version code and submit to cluster.")


if __name__ == "__main__":
    main()
