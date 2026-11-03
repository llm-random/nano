"""Audit-table row for one MFU run, read from a job log instead of wandb.

The wandb twin is scripts/mfu_window.py; this one parses the lines the stdout
metric logger prints (`infrastructure.metric_logger.type=stdout`), which is how
runs on a cluster without a wandb key are measured -- every multi-GPU baseline
in documents/notes/multigpu_efficiency.md came through here.

Same reducer as the single-GPU audit: mean over the steady-state window (steps
10-29 by default), steps 0-9 dropped for allocator warmup, compile and
dataloader spin-up. Only rank 0's lines are read; every rank prints, and they
would otherwise be averaged together.

Usage:
    python scripts/mfu_window_log.py <logfile> [--window 10 29]
"""

import argparse
import re
import statistics

# [device:0] step:12 tokens:12576768 -> train/mfu: 0.4559
LINE = re.compile(
    r"\[device:(?P<rank>\d+)\] step:(?P<step>\d+) tokens:(?P<tokens>\d+) -> "
    r"(?P<name>[\w/]+): (?P<value>[-\d.eE+]+)"
)
WANTED = ("train/step_time", "train/tokens_per_s", "train/mfu", "train/loss")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile")
    parser.add_argument("--window", nargs=2, type=int, default=[10, 29])
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    lo, hi = args.window

    series = {name: {} for name in WANTED}
    peak_memory = None
    with open(args.logfile, errors="replace") as handle:
        for line in handle:
            match = LINE.search(line)
            if not match or int(match["rank"]) != args.rank:
                continue
            name, step = match["name"], int(match["step"])
            if name in series:
                series[name][step] = float(match["value"])
            elif name == "train/max_memory_gib":
                peak_memory = max(peak_memory or 0.0, float(match["value"]))

    def window(name):
        return [v for step, v in sorted(series[name].items()) if lo <= step <= hi]

    step_times = window("train/step_time")
    if not step_times:
        raise SystemExit(
            f"no train/step_time for rank {args.rank} in steps {lo}-{hi}; "
            f"saw steps {sorted(series['train/step_time'])[:5]}..."
        )

    mean_step = statistics.mean(step_times)
    stdev_step = statistics.stdev(step_times) if len(step_times) > 1 else 0.0
    tokens_per_s = window("train/tokens_per_s")
    mfu = window("train/mfu")
    losses = window("train/loss")

    print(f"window steps {lo}-{hi}  ({len(step_times)} steps)")
    print(f"  step time    {mean_step * 1000:9.0f} ms  (cv {stdev_step / mean_step:.2%})")
    if tokens_per_s:
        print(f"  tokens/s     {statistics.mean(tokens_per_s):9,.0f}")
    if mfu:
        print(f"  MFU          {statistics.mean(mfu):9.2%}")
    if peak_memory:
        print(f"  peak memory  {peak_memory:9.1f} GiB")
    if losses:
        print(f"  loss         {losses[0]:.4f} -> {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
