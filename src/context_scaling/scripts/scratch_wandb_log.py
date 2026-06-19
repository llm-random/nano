"""Scratch: try various ways to attach a sequence to an existing wandb run.

Reads the first jobID from 2503_restart/jobs.json, resumes that run,
attempts a few different upload strategies, and prints what's queryable
back through the API for each.

Run:
    pixi run python src/context_scaling/scripts/scratch_wandb_log.py
"""
import json
import sys
from pathlib import Path

import wandb

REPO_ROOT = Path(__file__).resolve().parents[3]
JOBS_JSON = REPO_ROOT / "2503_restart/jobs.json"

# small + large list, so we can see where wandb starts choking
SHORT_LIST = [float(i) * 0.1 for i in range(8)]
LONG_LIST = [float(i) * 0.001 for i in range(2048)]


def split_project(project: str):
    parts = project.split("/", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, project


def main():
    with open(JOBS_JSON) as f:
        job = json.load(f)[0]
    run_id = job["jobID"]
    project = job["wandb_project"]
    entity, proj = split_project(project)
    print(f"run_id={run_id} project={project}")

    # ---- strategy 1: summary list (the one currently failing for 2048) ----
    print("\n[1] summary[key] = LONG_LIST")
    run = wandb.init(entity=entity, project=proj, id=run_id, resume="must")
    try:
        run.summary["scratch/summary_short"] = SHORT_LIST
        run.summary["scratch/summary_long"] = LONG_LIST
    finally:
        run.finish()

    # ---- strategy 2: per-position run.log with define_metric custom x-axis ----
    print("\n[2] define_metric + per-position run.log")
    run = wandb.init(entity=entity, project=proj, id=run_id, resume="must")
    try:
        x = "scratch/pos"
        y = "scratch/per_pos_loss"
        run.define_metric(x)
        run.define_metric(y, step_metric=x)
        for i, v in enumerate(LONG_LIST):
            run.log({x: i, y: v})
    finally:
        run.finish()

    # ---- strategy 3: wandb.Table ----
    print("\n[3] wandb.Table")
    run = wandb.init(entity=entity, project=proj, id=run_id, resume="must")
    try:
        table = wandb.Table(
            columns=["position", "loss"],
            data=[[i, v] for i, v in enumerate(LONG_LIST)],
        )
        run.log({"scratch/per_pos_table": table})
    finally:
        run.finish()

    # ---- read it all back via API ----
    print("\n=== readback via API ===")
    api = wandb.Api()
    api_run = api.run(f"{project}/{run_id}")
    for key in ["scratch/summary_short", "scratch/summary_long", "scratch/per_pos_table"]:
        try:
            val = api_run.summary[key]
            print(f"  {key}: type={type(val).__name__} repr={val!r}")
        except KeyError:
            print(f"  {key}: <missing>")

    print("\n=== history scan for strategy 2 ===")
    hist = api_run.history(keys=["scratch/per_pos_loss", "scratch/pos"], pandas=False)
    print(f"  rows returned: {len(hist)}")
    if hist:
        print(f"  first: {hist[0]}")
        print(f"  last:  {hist[-1]}")


if __name__ == "__main__":
    main()
