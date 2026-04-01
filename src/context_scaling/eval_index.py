"""Load and filter model evaluation CSVs using the index.jsonl metadata."""

import json
from pathlib import Path
import pandas as pd


def load_eval_index(eval_dir: str | Path) -> pd.DataFrame:
    """Load index.jsonl from an eval directory into a DataFrame.

    Each row has the full flat config of the evaluated model plus
    csv, run_id, model_step, eval_seq_len columns.
    """
    index_path = Path(eval_dir) / "index.jsonl"
    records = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def load_eval_csvs(
    index: pd.DataFrame,
    eval_dir: str | Path,
    label_cols: list[str] | None = None,
) -> tuple[list[str], list[pd.DataFrame]]:
    """Load CSV DataFrames for a (filtered) index.

    Args:
        index: DataFrame from load_eval_index, possibly filtered.
        eval_dir: directory containing the CSV files.
        label_cols: config columns to include in the label string.
            If None, uses ["run_id"].

    Returns:
        (labels, dfs) — parallel lists of label strings and loss DataFrames.
    """
    if label_cols is None:
        label_cols = ["run_id"]

    eval_dir = Path(eval_dir)
    labels = []
    dfs = []

    for _, row in index.iterrows():
        csv_path = eval_dir / row["csv"]
        if not csv_path.exists():
            continue

        parts = []
        for col in label_cols:
            if col in row.index:
                parts.append(f"{col}={row[col]}")
        label = "+".join(parts) if parts else row["csv"]

        labels.append(label)
        dfs.append(pd.read_csv(csv_path))

    return labels, dfs
