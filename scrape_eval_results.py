#!/usr/bin/env python
"""Scrape downstream-eval results from W&B into a CSV, one row per eval run.

Reads the `<task>/<metric>` keys that evaluate_hf_models.py logs and keeps one
primary metric per task (acc_norm where it is the reported number, acc
otherwise), plus an `avg` over the accuracy columns.

Examples:
    pixi run python scrape_eval_results.py --tags olmo2 0_numshot --out olmo_eval.csv
    pixi run python scrape_eval_results.py --tags smollm2 0_numshot --after 2026-07-01
"""

import argparse
import csv
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import wandb

logger = logging.getLogger("scrape_eval_results")

DEFAULT_PROJECT = "ideas_cv/llm-random-test"

# eval runs carry the training tags too, so pin to the eval jobs by default
DEFAULT_REQUIRED_TAGS = ["eval", "hf_model"]

# metric reported for each task; anything else falls back to acc_norm -> acc
PRIMARY_METRIC = {
    "arc_easy": "acc_norm",
    "arc_challenge": "acc_norm",
    "hellaswag": "acc_norm",
    "piqa": "acc_norm",
    "openbookqa": "acc_norm",
    "winogrande": "acc",
    "sciq": "acc",
    "social_iqa": "acc",
    "lambada_openai": "acc",
}
METRIC_FALLBACK = ["acc_norm", "acc", "exact_match", "f1"]

# column order; tasks outside this list are appended alphabetically
TASK_ORDER = list(PRIMARY_METRIC)

# excluded from the `avg` column — not on an accuracy scale
NON_ACCURACY_METRICS = {
    "perplexity",
    "word_perplexity",
    "byte_perplexity",
    "bits_per_byte",
}

METHOD_ORDER = ["pcd", "hpd", "pc", "hp"]
RATIO_RE = re.compile(r"^(\d+)p$")


@dataclass
class Row:
    model: str
    method: str
    ratio: str
    run_id: str
    state: str
    created_at: str
    num_fewshot: Optional[int]
    scores: dict[str, float] = field(default_factory=dict)
    metrics_used: dict[str, str] = field(default_factory=dict)


def parse_name(name: str) -> tuple[str, str]:
    """Pull method (pcd/hpd/pc/hp) and ratio (10p/…) out of a `name:` from the eval config."""
    tokens = name.split("_")
    method = next((t for t in tokens if t in METHOD_ORDER), "")
    ratio = next((t for t in tokens if RATIO_RE.match(t)), "")
    if not method:
        logger.warning("No method token in run name %r", name)
    return method, ratio


def collect_task_metrics(summary: dict) -> dict[str, dict[str, float]]:
    """Group the flat `<task>/<metric>` summary keys by task, dropping stderr."""
    per_task: dict[str, dict[str, float]] = {}
    for key, value in summary.items():
        if "/" not in key or not isinstance(value, (int, float)):
            continue
        task, metric = key.split("/", 1)
        if metric.endswith("_stderr") or metric == "alias":
            continue
        per_task.setdefault(task, {})[metric] = float(value)
    return per_task


def pick_metric(task: str, metrics: dict[str, float]) -> Optional[str]:
    preferred = PRIMARY_METRIC.get(task)
    if preferred and preferred in metrics:
        return preferred
    for candidate in METRIC_FALLBACK:
        if candidate in metrics:
            return candidate
    return next(iter(metrics), None)


def build_row(run) -> Optional[Row]:
    per_task = collect_task_metrics(run.summary._json_dict)
    if not per_task:
        logger.warning("Run %s (%s) logged no task metrics, skipping", run.name, run.id)
        return None

    method, ratio = parse_name(run.name or "")
    scores, metrics_used = {}, {}
    for task, metrics in per_task.items():
        metric = pick_metric(task, metrics)
        if metric is None:
            continue
        scores[task] = metrics[metric]
        metrics_used[task] = metric

    if run.state != "finished":
        logger.warning("Run %s (%s) is state=%s", run.name, run.id, run.state)

    return Row(
        model=run.name or run.id,
        method=method,
        ratio=ratio,
        run_id=run.id,
        state=run.state,
        created_at=str(run.created_at),
        num_fewshot=(run.config or {}).get("num_fewshot"),
        scores=scores,
        metrics_used=metrics_used,
    )


def fetch_rows(project: str, tags: list[str], after: Optional[str]) -> list[Row]:
    conditions = [{"tags": tag} for tag in tags]
    if after:
        conditions.append({"createdAt": {"$gte": f"{after}T00:00:00"}})
    filters = {"$and": conditions} if conditions else {}
    runs = list(wandb.Api().runs(path=project, filters=filters))
    logger.info("Fetched %d run(s) matching tags %s", len(runs), tags)
    return [row for row in (build_row(run) for run in runs) if row is not None]


def task_columns(rows: list[Row]) -> list[str]:
    found = {task for row in rows for task in row.scores}
    ordered = [task for task in TASK_ORDER if task in found]
    return ordered + sorted(found - set(ordered))


def average(row: Row, tasks: list[str]) -> Optional[float]:
    values = [
        row.scores[task]
        for task in tasks
        if task in row.scores and row.metrics_used[task] not in NON_ACCURACY_METRICS
    ]
    return sum(values) / len(values) if values else None


def sort_key(row: Row):
    method_rank = (
        METHOD_ORDER.index(row.method)
        if row.method in METHOD_ORDER
        else len(METHOD_ORDER)
    )
    ratio_match = RATIO_RE.match(row.ratio)
    return (
        method_rank,
        int(ratio_match.group(1)) if ratio_match else 999,
        row.model,
        row.created_at,
    )


def render(rows: list[Row], handle, precision: int) -> None:
    tasks = task_columns(rows)
    header = (
        ["model", "method", "ratio"]
        + tasks
        + [
            "avg",
            "num_fewshot",
            "state",
            "run_id",
            "created_at",
        ]
    )
    writer = csv.writer(handle)
    writer.writerow(header)

    def fmt(value: Optional[float]) -> str:
        return "" if value is None else f"{value:.{precision}f}"

    for row in sorted(rows, key=sort_key):
        writer.writerow(
            [row.model, row.method, row.ratio]
            + [fmt(row.scores.get(task)) for task in tasks]
            + [
                fmt(average(row, tasks)),
                "" if row.num_fewshot is None else row.num_fewshot,
                row.state,
                row.run_id,
                row.created_at,
            ]
        )

    for task in tasks:
        used = {row.metrics_used[task] for row in rows if task in row.metrics_used}
        if len(used) > 1:
            logger.warning("Task %s mixes metrics across runs: %s", task, sorted(used))
        else:
            logger.info("%s -> %s", task, next(iter(used)))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tags", nargs="+", required=True, help="keep runs carrying ALL of these tags"
    )
    parser.add_argument(
        "--require-tags",
        nargs="+",
        default=DEFAULT_REQUIRED_TAGS,
        help=f"always-required tags, pass none to disable (default: {' '.join(DEFAULT_REQUIRED_TAGS)})",
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help=f"wandb entity/project (default: {DEFAULT_PROJECT})",
    )
    parser.add_argument(
        "--after",
        default=None,
        help="only runs created on/after this date (YYYY-MM-DD)",
    )
    parser.add_argument("--precision", type=int, default=5, help="decimal places")
    parser.add_argument(
        "--out", default=None, help="write the CSV here instead of stdout"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr
    )

    tags = list(dict.fromkeys(list(args.require_tags) + list(args.tags)))
    rows = fetch_rows(args.project, tags, args.after)
    if not rows:
        logger.error("No runs with task metrics matched, nothing to write")
        return 1

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as handle:
            render(rows, handle, args.precision)
        logger.info("Wrote %s (%d row(s))", args.out, len(rows))
    else:
        render(rows, sys.stdout, args.precision)
        logger.info("%d row(s)", len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
