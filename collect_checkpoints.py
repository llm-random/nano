#!/usr/bin/env python
"""Build a paste-ready `models:` block for eval_hf_models.yaml from W&B run tags.

Takes each run's `full_save_checkpoints_path` (logged to the W&B summary by
main.py) and names it `{model}_{method}_{ratio}_{tokens}BT`. Runs that did not
finish are emitted commented out with the reason, so the output stays paste-able.

Examples:
    pixi run python collect_checkpoints.py --tags pcd llama_1
    pixi run python collect_checkpoints.py --tags pc llama_1 --verify --host lem
"""

import argparse
import logging
import os
import re
import shlex
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import wandb

logger = logging.getLogger("collect_checkpoints")

DEFAULT_PROJECT = "ideas_cv/llm-random-test"

# dmodel -> label fallback, used only when a run carries no `<N>p` tag (llama dims)
DMODEL_RATIO_FALLBACK = {448: "10p", 960: "30p", 1344: "50p", 1728: "75p"}
RATIO_TAG_RE = re.compile(r"^(\d+)p$")

# checked in order, so longer/more specific patterns come first
MODEL_LABEL_PATTERNS = [
    ("smollm2-1.7b", "smollm2_1700"),
    ("smollm", "smollm2"),
    ("olmo-2-0425-1b", "olmo1b"),
    ("olmo", "olmo"),
    ("qwen", "qwen"),
    ("8b", "llama8b"),
    ("1b", "llama1b"),
]
MODEL_TAG_LABELS = {
    "llama_1": "llama1b",
    "llama_8": "llama8b",
    "olmo2": "olmo1b",
    "smollm2": "smollm2_1700",
}

# eval runs carry the training tags too, so drop them unless asked otherwise
DEFAULT_NEGATIVE_TAGS = ["eval", "hf_model"]

# grouping order in the rendered output
METHOD_ORDER = ["pc", "pcd", "hp", "hpd"]


@dataclass
class Entry:
    name: str
    path: Optional[str]
    method: str
    ratio: str
    tokens: Optional[float]
    run_id: str
    job_id: Optional[str]
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.path is not None and not self.problems


def get_in(config: dict, dotted: str, default: Any = None) -> Any:
    """Read a nested wandb config value by dotted path."""
    node = config
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def first_of(config: dict, dotted_paths: list[str], default: Any = None) -> Any:
    for dotted in dotted_paths:
        value = get_in(config, dotted)
        if value is not None:
            return value
    return default


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def model_label(config: dict, tags: list[str]) -> str:
    source = first_of(
        config,
        [
            "trainer.original_llama_path",
            "common.original_llama_path",
            "projected_compression.original_llama_path",
            "trainer.checkpoint.load.path",
            "distillation.load.path",
            "projected_compression.source_model_path",
        ],
    )
    if source:
        basename = str(source).rstrip("/").split("/")[-1].lower()
        for pattern, label in MODEL_LABEL_PATTERNS:
            if pattern in basename:
                return label
    for tag in tags:
        if tag in MODEL_TAG_LABELS:
            return MODEL_TAG_LABELS[tag]
    return slug(str(source).split("/")[-1]) if source else "unknown_model"


def is_pruning(config: dict, tags: list[str]) -> bool:
    if "hard_pruning" in tags:
        return True
    functions = get_in(config, "apply_functions") or []
    if isinstance(functions, list):
        return any(
            "pruning.prune" in str(fn.get("_target_", ""))
            for fn in functions
            if isinstance(fn, dict)
        )
    return False


def is_distillation(config: dict, tags: list[str]) -> bool:
    target = str(get_in(config, "trainer._target_", ""))
    if "DistillationTrainer" in target:
        return True
    if get_in(config, "distillation") is not None:
        return True
    return "distillation" in tags


def method_label(config: dict, tags: list[str]) -> str:
    base = "hp" if is_pruning(config, tags) else "pc"
    return base + "d" if is_distillation(config, tags) else base


def ratio_label(config: dict, tags: list[str]) -> str:
    ratios = sorted(
        (tag for tag in tags if RATIO_TAG_RE.match(tag)),
        key=lambda tag: int(RATIO_TAG_RE.match(tag).group(1)),
    )
    if len(ratios) == 1:
        return ratios[0]
    if len(ratios) > 1:
        logger.warning("Multiple ratio tags %s, falling back to dmodel", ratios)
    dmodel = first_of(config, ["common.dmodel", "common.target_dmodel"])
    if dmodel in DMODEL_RATIO_FALLBACK:
        return DMODEL_RATIO_FALLBACK[dmodel]
    logger.warning("No ratio tag and unmapped dmodel=%s", dmodel)
    return "unkp"


def target_tokens(summary: dict, n_steps: Optional[int]) -> Optional[float]:
    """Token budget the run was configured for; extrapolated so partial runs label right."""
    tokens = summary.get("token_count")
    step = summary.get("step")
    if tokens and step is not None and n_steps:
        return tokens / (step + 1) * n_steps
    return tokens


def tokens_label(tokens: Optional[float]) -> str:
    if not tokens:
        return "unkBT"
    return f"{round(tokens / 1e9)}BT"


def checkpoint_path(
    config: dict, summary: dict, last_step: Optional[int]
) -> tuple[Optional[str], list[str]]:
    """The HF checkpoint dir: the path the run logged, completed if it predates the suffix."""
    save_type = get_in(config, "trainer.checkpoint.save.type")
    if get_in(config, "trainer.checkpoint.save.path") is None:
        return None, ["checkpoint saving disabled (save.path is null)"]
    if save_type not in ("hf_only", "nano_and_hf", "huggingface"):
        return None, [f"save.type={save_type!r} is not HF-loadable"]

    logged = summary.get("full_save_checkpoints_path")
    if logged:
        path = str(logged).rstrip("/")
        if save_type == "huggingface" or path.endswith("/hf"):
            return path, []
        # runs from before main.py logged the suffix (May 2026 and earlier)
        if last_step is None:
            return None, ["logged path has no /hf suffix and no n_steps to derive it"]
        return f"{path}/step_{last_step}/hf", []

    base = get_in(config, "trainer.checkpoint.save.path")
    job_id = first_of(config, ["run_env.slurm_job_id", "job/SLURM_JOB_ID"])
    task_id = first_of(
        config, ["run_env.slurm_array_task_id", "job/SLURM_ARRAY_TASK_ID"]
    )
    if not (job_id and task_id and last_step is not None):
        return None, ["no logged path and not enough config to reconstruct it"]
    suffix = "" if save_type == "huggingface" else f"/step_{last_step}/hf"
    return f"{str(base).rstrip('/')}/{job_id}/{task_id}{suffix}", []


def build_entry(run, name_suffix: Optional[str]) -> Entry:
    config = run.config
    tags = list(run.tags)
    summary = run.summary._json_dict

    n_steps = get_in(config, "trainer.n_steps")
    logged_step = summary.get("step")
    tokens = target_tokens(summary, n_steps)

    problems: list[str] = []
    if run.state != "finished":
        problems.append(f"state={run.state}")
    if n_steps is None:
        problems.append("no trainer.n_steps in config")
    elif logged_step is None:
        problems.append("no logged step")
    elif logged_step < n_steps - 1:
        problems.append(f"stopped at step {int(logged_step)}/{n_steps}")

    # the final checkpoint is always written at n_steps-1 (periodic or final save)
    last_step = n_steps - 1 if n_steps is not None else None
    path, path_problems = checkpoint_path(config, summary, last_step)
    problems.extend(path_problems)

    method = method_label(config, tags)
    ratio = ratio_label(config, tags)
    name = f"{model_label(config, tags)}_{method}_{ratio}_{tokens_label(tokens)}"
    if name_suffix:
        name = f"{name}_{name_suffix}"

    return Entry(
        name=name,
        path=path,
        method=method,
        ratio=ratio,
        tokens=tokens,
        run_id=run.id,
        job_id=first_of(config, ["run_env.slurm_job_id", "job/SLURM_JOB_ID"]),
        problems=problems,
    )


def fetch_entries(project, tags, negative_tags, name_suffix, after) -> list[Entry]:
    conditions = [{"tags": tag} for tag in tags]
    if after:
        conditions.append({"createdAt": {"$gte": f"{after}T00:00:00"}})
    filters = {"$and": conditions} if conditions else {}
    runs = list(wandb.Api().runs(path=project, filters=filters))
    logger.info("Fetched %d run(s) matching tags %s", len(runs), tags)

    negative = set(negative_tags or [])
    entries = []
    for run in runs:
        if negative & set(run.tags):
            continue
        entries.append(build_entry(run, name_suffix))
    return entries


def deduplicate_names(entries: list[Entry]) -> None:
    """Names key both the eval W&B run and the results filename, so make them unique.

    Only usable entries compete: a commented-out entry writes no results, so a failed
    run must not force a job-id suffix onto the rerun that succeeded.
    """
    counts = Counter(entry.name for entry in entries if entry.ok)
    for entry in entries:
        qualifier = entry.job_id or entry.run_id
        if entry.ok and counts[entry.name] > 1:
            logger.warning(
                "Duplicate name %s, qualifying with %s", entry.name, qualifier
            )
            entry.name = f"{entry.name}_{qualifier}"
        elif not entry.ok:
            entry.name = f"{entry.name}_{qualifier}"  # keep skipped lines identifiable


def verify_paths(entries: list[Entry], host: str) -> None:
    """Mark entries whose checkpoint dir is not actually on disk (one ssh round trip)."""
    from run_exp import ConnectWithPassphrase  # lazy: pulls in torch via main

    paths = [entry.path for entry in entries if entry.path]
    if not paths:
        return

    checks = " ".join(
        f"if [ -f {shlex.quote(p)}/config.json ] && ls {shlex.quote(p)}/*.safetensors "
        f'>/dev/null 2>&1; then echo "OK {p}"; else echo "MISSING {p}"; fi;'
        for p in paths
    )
    with ConnectWithPassphrase(host=host, inline_ssh_env=True) as connection:
        output = connection.run(checks, hide=True).stdout

    missing = {
        line.split(" ", 1)[1]
        for line in output.splitlines()
        if line.startswith("MISSING ")
    }
    for entry in entries:
        if entry.path in missing:
            entry.problems.append("no config.json/safetensors on disk")
    logger.info("Verified %d path(s), %d missing", len(paths), len(missing))


def sort_key(entry: Entry):
    method_rank = (
        METHOD_ORDER.index(entry.method)
        if entry.method in METHOD_ORDER
        else len(METHOD_ORDER)
    )
    ratio_match = RATIO_TAG_RE.match(entry.ratio)
    return (
        method_rank,
        int(ratio_match.group(1)) if ratio_match else 999,
        entry.tokens or 0,
    )


def render(entries: list[Entry]) -> str:
    lines = ["models:"]
    current_group = None
    for entry in sorted(entries, key=sort_key):
        group = (entry.method, entry.ratio)
        if group != current_group:
            lines.append(f"\n  # {entry.method.upper()} {entry.ratio}")
            current_group = group

        if entry.ok:
            lines.append(f"  - path: {entry.path}")
            lines.append(f"    name: {entry.name}")
        else:
            reason = "; ".join(entry.problems) or "unresolved checkpoint path"
            lines.append(f"  # SKIPPED {entry.name}: {reason}")
            if entry.path:
                lines.append(f"  # - path: {entry.path}")
                lines.append(f"  #   name: {entry.name}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tags", nargs="+", required=True, help="keep runs carrying ALL of these tags"
    )
    parser.add_argument(
        "--not-tags",
        nargs="+",
        default=DEFAULT_NEGATIVE_TAGS,
        help=f"drop runs carrying ANY of these tags (default: {' '.join(DEFAULT_NEGATIVE_TAGS)})",
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help=f"wandb entity/project (default: {DEFAULT_PROJECT})",
    )
    parser.add_argument(
        "--name-suffix", default=None, help="appended to every name, e.g. ns0"
    )
    parser.add_argument(
        "--after",
        default=None,
        help="only runs created on/after this date (YYYY-MM-DD); skips old reruns",
    )
    parser.add_argument(
        "--verify", action="store_true", help="ssh to --host and check the dirs exist"
    )
    parser.add_argument("--host", default=None, help="ssh host for --verify, e.g. lem")
    parser.add_argument(
        "--out", default=None, help="write the block here instead of stdout"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr
    )

    if args.verify and not args.host:
        parser.error("--verify requires --host")

    entries = fetch_entries(
        args.project, args.tags, args.not_tags, args.name_suffix, args.after
    )
    if not entries:
        logger.error("No runs matched, nothing to write")
        return 1

    deduplicate_names(entries)
    if args.verify:
        verify_paths(entries, args.host)

    block = render(entries)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            handle.write(block)
        logger.info("Wrote %s", args.out)
    else:
        print(block, end="")

    usable = sum(entry.ok for entry in entries)
    logger.info("%d/%d run(s) usable", usable, len(entries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
