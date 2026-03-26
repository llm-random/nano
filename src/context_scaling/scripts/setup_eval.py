import os
import re
from pathlib import Path
import argparse
import pandas as pd
import wandb
import json
import yaml


def get_wandb_table(
    tags,
    project="ideas_cv/llm-random-test",
    negative_tags=None,
    columns=None,
    print_columns=False,
):
    # WandB automatically looks for the WANDB_API_KEY environment variable.
    # No need to explicitly pass it like Neptune's token, but you must ensure it's exported in your env.
    api = wandb.Api()

    print(f"tags: {tags}")

    # Ensure tags is a list
    if isinstance(tags, str):
        tags = [tags]

    # Build WandB filter using MongoDB-like syntax.
    # $and ensures the run contains ALL of the specified positive tags (matching Neptune's default).
    filters = {"$and": [{"tags": tag} for tag in tags]} if tags else {}

    # Fetch runs via API
    runs = api.runs(path=project, filters=filters)

    # Extract data into a list of dictionaries to convert to pandas
    runs_data = []
    for run in runs:
        run_dict = {
            "sys/id": run.id,
            "sys/name": run.name,
            "sys/state": run.state,
            "sys/tags": run.tags,  # WandB tags are already a list of strings
        }
        # Flatten config (hyperparameters) and summary (metrics) into the dict.
        # Adding prefixes keeps the structure clean and prevents key collisions.
        run_dict.update({f"config/{k}": v for k, v in run.config.items()})
        run_dict.update({f"summary/{k}": v for k, v in run.summary._json_dict.items()})

        runs_data.append(run_dict)

    runs_table = pd.DataFrame(runs_data)

    if runs_table.empty:
        print("df shape: (0, 0)")
        return runs_table

    # Filter down to specific columns if provided
    if columns:
        # We must keep "sys/tags" temporarily if negative_tags are provided
        cols_to_keep = set(columns)
        if negative_tags:
            cols_to_keep.add("sys/tags")

        available_cols = [col for col in cols_to_keep if col in runs_table.columns]
        runs_table = runs_table[available_cols]

    # Apply negative tags filter
    if negative_tags:
        if isinstance(negative_tags, str):
            negative_tags = [negative_tags]

        for neg_tag in negative_tags:
            # Filter out rows where the negative tag exists in the list of tags
            runs_table = runs_table[
                ~runs_table["sys/tags"].apply(
                    lambda x: neg_tag in x if isinstance(x, list) else False
                )
            ]

    # Print columns
    if print_columns:
        print("\n=== Available columns ===")
        for col in sorted(runs_table.columns):
            print(f"\t{col}")
        print("========================\n")

    print(f"df shape: {runs_table.shape}")

    return runs_table


def save_yaml_config_from_row(
    row: pd.Series,
    out_yaml_path: Path,
) -> None:
    """Reconstruct the run config from dataframe row's config/* columns."""
    PREFIX = "config/"
    config = {}
    for col in row.index:
        if isinstance(col, str) and col.startswith(PREFIX):
            config[col[len(PREFIX) :]] = row[col]

    out_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=True)


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
        ckpt_path = str(row.get("job/full_save_checkpoints_path", ""))

        yaml_path = yaml_dir / f"{run_id}.yaml"
        if not yaml_path.exists() or yaml_path.stat().st_size == 0:
            save_yaml_config_from_row(row, yaml_path)

        with open(yaml_path, "r", encoding="utf-8") as f:
            run_cfg = yaml.safe_load(f)
        seq_len = run_cfg["common"]["sequence_length"]

        if args.seq_len is not None and args.seq_len < seq_len:
            seq_len = args.seq_len

        records.append(
            {
                "jobID": run_id,
                "ckpt_path": ckpt_path,
                "yaml_config_path": str(yaml_path),
                "seq_len": seq_len,
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
