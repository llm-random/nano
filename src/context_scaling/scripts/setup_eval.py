import os
import re
import warnings
from pathlib import Path
import argparse
import json
import yaml

from wandb_utils import WANDB_PROJECT, get_wandb_table, save_yaml_config_from_row


def resolve_model_step(ckpt_path: str, model_step: int | None) -> int | str:
    """Return the step number to eval, or a warning message if unresolvable."""
    print("resolving model_step")
    print(f"ckpt_path: {ckpt_path}")
    print(f"model_step: {model_step}")
    if not ckpt_path:
        if model_step is not None:
            return model_step
        return "MISSING_CKPT_PATH"

    ckpt_dir = Path(ckpt_path)

    if model_step is not None:
        step_dir = ckpt_dir / f"step_{model_step}"
        if not step_dir.exists():
            warnings.warn(f"Checkpoint not found: {step_dir}")
            return f"NOT_FOUND:step_{model_step}"
        return model_step

    # find latest step_*
    steps = []
    if ckpt_dir.exists():
        for name in os.listdir(ckpt_dir):
            if name.startswith("step_"):
                try:
                    steps.append(int(name.split("_", 1)[1]))
                except ValueError:
                    continue
    if not steps:
        warnings.warn(f"No step_* checkpoints found in {ckpt_dir}")
        return "NO_CHECKPOINTS"
    return max(steps)


def update_slurm_array_line(sbatch_path: Path, num_jobs: int) -> None:
    """
    Update (in-place) the first '#SBATCH --array=...' line to match num_jobs.

    - If num_jobs == 0: raises.
    - Replaces with '#SBATCH --array=0-(num_jobs-1)'.
    - Preserves an optional concurrency cap like '%4' if present.
      e.g. '#SBATCH --array=0-19%1' -> '#SBATCH --array=0-7%1'
    """
    if num_jobs <= 0:
        raise ValueError(f"num_jobs must be > 0, got {num_jobs}")

    sbatch_path = Path(sbatch_path)
    text = sbatch_path.read_text(encoding="utf-8")

    # Match: #SBATCH --array=... optionally with %<cap>
    # Examples:
    #   #SBATCH --array=0-19
    #   #SBATCH --array=0-19%1
    #   #SBATCH --array=3,5,7%2 (we will overwrite anyway)
    m = re.search(r"(?m)^(#SBATCH\s+--array=)([^\s]+)\s*$", text)
    if not m:
        raise RuntimeError(f"No '#SBATCH --array=...' line found in {sbatch_path}")

    old_spec = m.group(2)
    cap = ""
    mcap = re.match(r".*(%[0-9]+)$", old_spec)
    if mcap:
        cap = mcap.group(1)

    new_spec = f"0-{num_jobs - 1}{cap}"
    new_line = f"{m.group(1)}{new_spec}"

    new_text = text[: m.start()] + new_line + text[m.end() :]
    sbatch_path.write_text(new_text, encoding="utf-8")
    print(f"Updated {sbatch_path}: --array={old_spec} -> --array={new_spec}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--negative_tags", nargs="+", default=None)
    parser.add_argument("--out_dir", type=str, default="eval_grid")

    # Optional: update an sbatch script to have correct array length
    parser.add_argument(
        "--sbatch_path",
        type=str,
        default=None,
        help="Optional path to an sbatch .sh file. If provided, updates '#SBATCH --array=...' to match number of runs.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="Eval sequence length for all jobs. If not specified, each job uses its training sequence length.",
    )
    parser.add_argument(
        "--model_step",
        type=int,
        default=None,
        help="Checkpoint step to eval. If not specified, uses the latest step_* in each ckpt_dir.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = get_wandb_table(tags=args.tags, negative_tags=args.negative_tags)

    csv_path = out_dir / "main.csv"
    df.to_csv(csv_path, index=False)

    yaml_dir = out_dir / "yaml_cache"
    records = []

    for _, row in df.iterrows():
        run_id = str(row["sys/id"])
        ckpt_path = str(row.get("summary/full_save_checkpoints_path", ""))

        yaml_path = yaml_dir / f"{run_id}.yaml"
        if not yaml_path.exists() or yaml_path.stat().st_size == 0:
            save_yaml_config_from_row(row, yaml_path)

        with open(yaml_path, "r", encoding="utf-8") as f:
            run_cfg = yaml.safe_load(f)
        seq_len = run_cfg["common"]["sequence_length"]

        if args.seq_len is not None and args.seq_len < seq_len:
            seq_len = args.seq_len

        model_step = resolve_model_step(ckpt_path, args.model_step)

        records.append(
            {
                "jobID": run_id,
                "ckpt_path": ckpt_path,
                "yaml_config_path": str(yaml_path),
                "seq_len": seq_len,
                "model_step": model_step,
                "wandb_project": WANDB_PROJECT,
            }
        )

    json_path = out_dir / "jobs.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # Optional sbatch update
    if args.sbatch_path is not None:
        update_slurm_array_line(Path(args.sbatch_path), num_jobs=len(records))


if __name__ == "__main__":
    main()
