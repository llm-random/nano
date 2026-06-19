"""Local setup for context-scaling eval. Run before run_ctx_eval.py.

Hydra entrypoint. Fetches wandb runs matching eval.tags / eval.negative_tags,
writes {out_dir}/jobs.json and per-run yaml_cache/<run_id>.yaml. Inspect (and
hand-edit) the json, then submit via run_ctx_eval.py — sbatch is generated
there from the current jobs.json.
"""
import json
import sys
from pathlib import Path

import hydra
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src/context_scaling/scripts"))

from wandb_utils import (  # noqa: E402
    WANDB_PROJECT,
    get_wandb_table,
    save_yaml_config_from_row,
)


def _resolve_steps(eval_cfg, run_cfg) -> list[int]:
    """eval.model_step:
      - null → defaults to trainer.n_steps - 1 (last training step)
      - list → each step gets its own array task (per run)
    """
    if eval_cfg.model_step is None:
        return [int(run_cfg["trainer"]["n_steps"]) - 1]
    return [int(s) for s in eval_cfg.model_step]


@hydra.main(version_base=None, config_path="../../../configs", config_name="ctx_eval")
def main(cfg):
    eval_cfg = cfg.eval
    out_dir = Path(eval_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    negative_tags = list(eval_cfg.negative_tags) if eval_cfg.negative_tags else None
    df = get_wandb_table(
        tags=list(eval_cfg.tags),
        project=WANDB_PROJECT,
        negative_tags=negative_tags,
    )
    df.to_csv(out_dir / "main.csv", index=False)

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
        if eval_cfg.seq_len is not None and eval_cfg.seq_len < seq_len:
            seq_len = int(eval_cfg.seq_len)

        common = run_cfg["common"]
        records.append(
            {
                "jobID": run_id,
                "ckpt_path": ckpt_path,
                "yaml_config_path": str(yaml_path),
                "seq_len": seq_len,
                "model_step": _resolve_steps(eval_cfg, run_cfg),  # always a list
                "wandb_project": WANDB_PROJECT,
                "common.dmodel": common.get("dmodel"),
                "common.kv_heads": common.get("kv_heads"),
                "common.dff": common.get("dff"),
                "common.sequence_length": common.get("sequence_length"),
            }
        )

    jobs_path = out_dir / "jobs.json"
    with open(jobs_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    n_tasks = sum(len(r["model_step"]) for r in records)
    print(f"wrote {len(records)} runs / {n_tasks} array tasks → {jobs_path}")
    print("next: pixi run python src/context_scaling/scripts/run_ctx_eval.py")


if __name__ == "__main__":
    main()
