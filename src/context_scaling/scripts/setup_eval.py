import os
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--out_dir", type=str, default="eval_grid")
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

        # save yaml_config to eval_grid/yaml_cache/<RUN_ID>.yaml
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

if __name__ == "__main__":
    main()
