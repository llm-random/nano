from pathlib import Path
import pandas as pd
import torch
import wandb
import yaml


WANDB_PROJECT = "ideas_cv/llm-random-test"


def upload_mean_loss_to_wandb(
    run_id: str,
    project: str,
    mean_losses: torch.Tensor,
    model_step: int,
    eval_seq_len: int,
):
    """Resume the wandb run and log per-position mean loss as a wandb.Table.
    Stored as a media artifact (no step-counter pollution, no large-array
    truncation that summary lists hit at this size).
    """
    parts = project.split("/", 1)
    entity, proj = (parts[0], parts[1]) if len(parts) == 2 else (None, project)

    run = wandb.init(entity=entity, project=proj, id=run_id, resume="must")
    try:
        key = f"eval/per_position_loss/step{model_step}_seq{eval_seq_len}"
        table = wandb.Table(
            columns=["position", "loss"],
            data=[[i, float(v)] for i, v in enumerate(mean_losses.tolist())],
        )
        run.log({key: table})
        print(f"logged {len(mean_losses)} points → {key}")
    finally:
        run.finish()


def get_wandb_table(
    tags,
    project=WANDB_PROJECT,
    negative_tags=None,
    columns=None,
    print_columns=False,
):
    # WandB automatically looks for the WANDB_API_KEY environment variable.
    api = wandb.Api()

    print(f"tags: {tags}")

    if isinstance(tags, str):
        tags = [tags]

    # $and ensures the run contains ALL of the specified positive tags.
    filters = {"$and": [{"tags": tag} for tag in tags]} if tags else {}

    runs = api.runs(path=project, filters=filters)

    runs_data = []
    for run in runs:
        run_dict = {
            "sys/id": run.id,
            "sys/name": run.name,
            "sys/state": run.state,
            "sys/tags": run.tags,
        }
        run_dict.update({f"config/{k}": v for k, v in run.config.items()})
        run_dict.update({f"summary/{k}": v for k, v in run.summary._json_dict.items()})

        runs_data.append(run_dict)

    runs_table = pd.DataFrame(runs_data)

    if runs_table.empty:
        print("df shape: (0, 0)")
        return runs_table

    if columns:
        cols_to_keep = set(columns)
        if negative_tags:
            cols_to_keep.add("sys/tags")

        available_cols = [col for col in cols_to_keep if col in runs_table.columns]
        runs_table = runs_table[available_cols]

    if negative_tags:
        if isinstance(negative_tags, str):
            negative_tags = [negative_tags]

        for neg_tag in negative_tags:
            runs_table = runs_table[
                ~runs_table["sys/tags"].apply(
                    lambda x: neg_tag in x if isinstance(x, list) else False
                )
            ]

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
