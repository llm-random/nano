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
    """
    Fetches a Neptune runs table filtered by tags and returns it as a pandas DataFrame.

    Parameters:
    - tags (list): List of tags to filter the runs.
    - negative_tags (list, optional): List of tags to exclude from the runs.
    - columns (list, optional): Additional columns to include in the runs table.
    - print_columns (bool, optional): If True, prints all available columns.

    Returns:
    - pandas.DataFrame: The runs table with the specified filters and columns.
    """

    # Initialize the Neptune project
    project = neptune.init_project(
        project=project,
        mode="read-only",
        api_token=os.environ["NEPTUNE_API_TOKEN"],
    )

    # Fetch the runs table with the specified tags and columns
    runs_table = project.fetch_runs_table(tag=tags, columns=columns).to_pandas()

    print(f"df shape: {runs_table.shape}")
    # Ensure 'sys/tags' is a list for each run
    # print(f"runs_table: {runs_table}")
    runs_table["sys/tags"] = runs_table["sys/tags"].apply(
        lambda x: x.split(",") if isinstance(x, str) else x
    )

    # Exclude runs containing any of the negative tags
    if negative_tags:
        for neg_tag in negative_tags:
            runs_table = runs_table[
                ~runs_table["sys/tags"].apply(lambda x: neg_tag in x)
            ]

    print(f"Table downloaded\nShape: {runs_table.shape}")

    if print_columns:
        print("\n=== Available columns ===")
        for col in sorted(runs_table.columns):
            print(f"  {col}")
        print("========================\n")

    return runs_table

def main():
    parser = argparse.ArgumentParser(
        description="Compute per-token losses and export to CSV"
    )

    parser.add_argument(
        "--tags",
        nargs="+",
        required=True,
        help="Neptune tags to INCLUDE (space-separated)",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=".",
        help="Output path for neptune CSV",
    )


    args = parser.parse_args()

    df = get_neptune_table(tags=args.tags)

    csv_dir = os.path.dirname(args.csv_path)
    os.makedirs(csv_dir, exist_ok=True)

    df.to_csv(args.csv_path, index=False)

    records = [
        {"jobID": job_id, "ckpt_path": ckpt}
        for job_id, ckpt in zip(df["sys/id"].astype(str), df["job/full_save_checkpoints_path"].astype(str))
    ]

    json_path = Path(args.csv_path).with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
