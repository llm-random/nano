#!/usr/bin/env python
r"""Per-run MFU averaged over training, for W&B runs selected by tags.

The W&B summary keeps only the last step's MFU. Three per-run numbers instead:
  mfu_mean       training steps only: total work / total time over them; eval and
                 checkpoint pauses between steps count against it
  mfu_median     the typical training step; for formula runs the typical 100-step
                 window, which still holds eval pauses, so it sits closer to the mean
  mfu_wallclock  the notebook's method, for every run: Setup FLOPs per step x steps
                 done / (W&B _runtime x peak); also counts startup, compilation and
                 whatever follows the last step
Where mfu_mean and mfu_median come from (the `source` column):
  logged      runs since 2026-10-30 (commit 997a16f7) log train/mfu every step,
              with FLOPs measured on the live model; averaged as is
  formula     older runs: FLOPs from the run's config via Setup
              (jupyter_notebooks/mfu_calc.py), time from steps/100/time
n/a: neither is available (run died before step 100, unknown GPU or FF type).

The table is id, the --columns you ask for, then the three mfu columns. A column is
  a built-in: name state tags gpus steps hours source
  a config key, dotted: common.h common.sequence_length compile_blocks_full
  or else a summary key, with its steps/ prefix: steps/eval/loss
Missing values show n/a.

--sort-by takes any of those columns (and id and the mfu columns), applied in order;
append :desc to reverse one. A sort key not in --columns is added to the table.
Without it, newest run first.

Usage:
pixi run python wandb_mfu.py \
    --entity ideas_cv \
    --project llm-random-test \
    --tags jm_mfu_benchmarking \
    --exclude-tags remove \
    --columns common.h common.sequence_length compile_blocks_full steps/eval/loss \
    --sort-by common.h common.sequence_length:desc \
    --csv wandb_mfu.csv
"""

import argparse
import statistics
import sys
from pathlib import Path

import pandas as pd
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent / "jupyter_notebooks"))
from mfu_calc import PEAK_BF16, Setup  # noqa: E402

MFU_KEY = "steps/train/mfu"
STEP_TIME_KEY = "steps/train/step_time"
WINDOW_TIME_KEY = "steps/100/time"  # mean step time over each 100-step window
FF_MATRICES = {"SwiGLU": 3, "GatedFeedForward": 3, "GeluFeedForward": 2}
MFU_COLUMNS = ("mfu_mean", "mfu_median", "mfu_wallclock")
FORMATS = {"hours": "{:.2f}"} | {column: "{:.2%}" for column in MFU_COLUMNS}


def n_gpus(run):
    return run.config.get("world_size") or (run.metadata or {}).get("gpu_count")


def gpu_name(run):
    return (run.metadata or {}).get("gpu", "?")


def steps_done(run):
    last_step = run.summary.get("step")
    return None if last_step is None else last_step + 1


BUILTIN_COLUMNS = {
    "name": lambda run: run.name,
    "state": lambda run: run.state,
    "tags": lambda run: " ".join(run.tags),
    "gpus": lambda run: f"{n_gpus(run)}x {gpu_name(run).removeprefix('NVIDIA ')}",
    "steps": steps_done,
    "hours": lambda run: run.summary.get("_runtime", 0) / 3600,
}


def logged_mfu(run):
    if MFU_KEY not in run.summary:
        return None, None  # never logged, skip the history download
    rows = run.scan_history(keys=[MFU_KEY, STEP_TIME_KEY], page_size=10_000)
    mfus, times = [], []
    for row in rows:
        mfus.append(row[MFU_KEY])
        times.append(row[STEP_TIME_KEY])
    if not mfus:
        return None, None
    # mfu * step_time is the step's work in peak-seconds
    mean = sum(m * t for m, t in zip(mfus, times)) / sum(times)
    return mean, statistics.median(mfus)


def setup_from_config(run, gpu, n_gpus):
    common = run.config["common"]
    if common.get("datt", common["dmodel"]) != common["dmodel"]:
        return None  # Setup assumes the attention width is dmodel
    ff = run.config["model"]["encoder"]["block_fn"]["ff_layer_fn"]
    kind = ff["_target_"].rsplit(".", 1)[-1]
    if kind == "MoE":
        n_experts, topk, dff = ff["num_experts"], ff["topk"], ff["dff"]
        ff_matrices = 3 if ff["activation_function"] == "swiglu" else 2
    elif kind in FF_MATRICES:
        n_experts, topk, dff, ff_matrices = 1, 1, common["dff"], FF_MATRICES[kind]
    else:
        return None
    return Setup(
        n_layers=common["n_blocks"],
        dmodel=common["dmodel"],
        dff=dff,
        q_heads=common["q_heads"],
        kv_heads=common["kv_heads"],
        ff_matrices=ff_matrices,
        n_experts=n_experts,
        topk=topk,
        vocab=common["vocab_size"],
        sl=common["sequence_length"],
        bs=common["batch_size"],
        gpu=gpu,
        n_gpus=n_gpus,
        include_attention_scores=True,
    )


def recalculated_mfu(run, setup):
    rows = run.scan_history(keys=[WINDOW_TIME_KEY], page_size=10_000)
    times = [row[WINDOW_TIME_KEY] for row in rows]
    if not times:
        return None, None
    # equal-length windows, so their plain mean is the time-weighted mean step time
    ideal_step_time = setup.flops_per_step / setup.peak
    return ideal_step_time / statistics.mean(
        times
    ), ideal_step_time / statistics.median(times)


def run_setup(run):
    gpu = next((key for key in PEAK_BF16 if key in gpu_name(run).lower()), None)
    if gpu is None or not n_gpus(run):
        return None
    return setup_from_config(run, gpu, n_gpus(run))


def run_mfu(run, setup):
    """(mfu_mean, mfu_median, source)"""
    mfu_mean, mfu_median = logged_mfu(run)
    if mfu_mean is not None:
        return mfu_mean, mfu_median, "logged"
    if setup:
        mfu_mean, mfu_median = recalculated_mfu(run, setup)
        if mfu_mean is not None:
            return mfu_mean, mfu_median, "formula"
    return None, None, None


def wallclock_mfu(run, setup):
    steps, runtime = steps_done(run), run.summary.get("_runtime")
    if not (setup and steps and runtime):
        return None
    return setup.flops_per_step * steps / (runtime * setup.peak)  # Setup.solve's MFU


def column_value(run, column):
    if column in BUILTIN_COLUMNS:
        return BUILTIN_COLUMNS[column](run)
    value = run.config
    for part in column.split("."):
        if not isinstance(value, dict) or part not in value:
            return run.summary.get(column)
        value = value[part]
    return value


def describe(run, columns):
    setup = run_setup(run)
    mfu_mean, mfu_median, source = run_mfu(run, setup)
    row = {"id": run.id}
    for column in columns:
        row[column] = source if column == "source" else column_value(run, column)
    row["mfu_mean"] = mfu_mean
    row["mfu_median"] = mfu_median
    row["mfu_wallclock"] = wallclock_mfu(run, setup)
    return row


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--entity", default="ideascv_llms")
    parser.add_argument("--project", required=True)
    parser.add_argument(
        "--tags", nargs="+", required=True, help="a run must carry all of them"
    )
    parser.add_argument(
        "--exclude-tags", nargs="+", default=[], help="and none of these"
    )
    parser.add_argument(
        "--columns", nargs="+", default=[], help="shown between id and mfu"
    )
    parser.add_argument(
        "--sort-by",
        nargs="+",
        default=[],
        help="columns in order; key:desc reverses one",
    )
    parser.add_argument("--csv", help="also write the table to this file")
    args = parser.parse_args()

    sort_keys = [key.removesuffix(":desc") for key in args.sort_by]
    ascending = [not key.endswith(":desc") for key in args.sort_by]
    fixed = {"id", *MFU_COLUMNS}
    columns = args.columns + [
        k for k in sort_keys if k not in args.columns and k not in fixed
    ]

    filters = {"$and": [{"tags": {"$in": [tag]}} for tag in args.tags]}
    filters["$and"] += [{"tags": {"$nin": [tag]}} for tag in args.exclude_tags]
    runs = wandb.Api(timeout=60).runs(
        f"{args.entity}/{args.project}", filters=filters, order="-created_at"
    )
    rows = []
    for run in runs:
        print(f"reading {run.name} ({run.id})", file=sys.stderr)
        rows.append(describe(run, columns))
    if not rows:
        sys.exit(
            f"no runs in {args.entity}/{args.project} carry all of {args.tags}"
            f" and none of {args.exclude_tags}"
        )

    table = pd.DataFrame(rows, dtype=object)
    if sort_keys:
        table = table.sort_values(
            sort_keys, ascending=ascending, na_position="last", kind="stable"
        )
    shown = table.copy()
    for column, fmt in FORMATS.items():
        if column in shown:
            shown[column] = [
                None if pd.isna(x) else fmt.format(x) for x in table[column]
            ]
    print(shown.fillna("n/a").to_string(index=False))
    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
