import os
import re
from pathlib import Path
import argparse
import pandas as pd
import neptune
import json


def get_neptune_table(
    tags,
    project="pmtest/llm-random",
    negative_tags=None,
    columns=None,
    print_columns=False,
):
    project = neptune.init_project(
        project=project,
        mode="read-only",
        api_token=os.environ["NEPTUNE_API_TOKEN"],
    )
    runs_table = project.fetch_runs_table(tag=tags, columns=columns).to_pandas()
    print(f"df shape: {runs_table.shape}")

    runs_table["sys/tags"] = runs_table["sys/tags"].apply(
        lambda x: x.split(",") if isinstance(x, str) else x
    )

    if negative_tags:
        for neg_tag in negative_tags:
            runs_table = runs_table[~runs_table["sys/tags"].apply(lambda x: neg_tag in x)]

    if print_columns:
        print("\n=== Available columns ===")
        for col in sorted(runs_table.columns):
            print(f"\t{col}")
        print("========================\n")

    return runs_table


def download_yaml_config(run_id: str, out_yaml_path: Path) -> None:
    run = neptune.init_run(
        project="pmtest/llm-random",
        with_id=run_id,
        mode="read-only",
        api_token=os.environ["NEPTUNE_API_TOKEN"],
    )
    out_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    run["yaml_config"].download(destination=str(out_yaml_path))
    run.stop()


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
    parser.add_argument("--out_dir", type=str, default="eval_grid")

    # Optional: update an sbatch script to have correct array length
    parser.add_argument(
        "--sbatch_path",
        type=str,
        default=None,
        help="Optional path to an sbatch .sh file. If provided, updates '#SBATCH --array=...' to match number of runs.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = get_neptune_table(tags=args.tags)

    csv_path = out_dir / "main.csv"
    df.to_csv(csv_path, index=False)

    yaml_dir = out_dir / "yaml_cache"
    records = []

    for _, row in df.iterrows():
        run_id = str(row["sys/id"])
        ckpt_path = str(row.get("job/full_save_checkpoints_path", ""))

        yaml_path = yaml_dir / f"{run_id}.yaml"
        if not yaml_path.exists() or yaml_path.stat().st_size == 0:
            download_yaml_config(run_id, yaml_path)

        records.append(
            {
                "jobID": run_id,
                "ckpt_path": ckpt_path,
                "yaml_config_path": str(yaml_path),
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
