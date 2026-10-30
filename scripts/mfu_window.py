"""Audit-table row for one MFU run: window stats + the E-criteria checks.

Pulls a run's history from wandb and reports, over the steady-state window
(steps 10-29 by default): mean tokens/s, mean MFU, step-time mean/std, peak
memory, plus the pass criteria from documents/notes/pretraining_stage3_mfu_plan.md
(first loss ~ ln(50304), loss decreasing, exact token accounting, step-time
stability).

Usage:
    python scripts/mfu_window.py <run_name_or_id> [--project nano-mfu-audit]
        [--entity ideas_cv] [--window 10 29] [--tokens-per-step 1048576]
"""

import argparse
import math
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", help="run name or id in the project")
    parser.add_argument("--entity", default="ideas_cv")
    parser.add_argument("--project", default="nano-mfu-audit")
    parser.add_argument("--window", nargs=2, type=int, default=[10, 29])
    parser.add_argument("--tokens-per-step", type=int, default=512 * 2048)
    args = parser.parse_args()

    import wandb

    api = wandb.Api()
    matches = [
        r
        for r in api.runs(f"{args.entity}/{args.project}")
        if args.run in (r.name, r.id)
    ]
    if not matches:
        raise SystemExit(f"no run named/id {args.run!r} in {args.project}")
    run = matches[0]

    rows = list(run.scan_history())
    by_step = {}
    for row in rows:
        step = row.get("step")
        if step is not None:
            by_step.setdefault(step, {}).update(
                {k: v for k, v in row.items() if v is not None}
            )

    lo, hi = args.window
    window = [by_step[s] for s in range(lo, hi + 1) if s in by_step]

    def series(key, rows_):
        return [r[key] for r in rows_ if key in r]

    losses_all = [by_step[s]["train/loss"] for s in sorted(by_step) if "train/loss" in by_step[s]]
    step_times = series("train/step_time", window)
    tps = series("train/tokens_per_s", window)
    mfu = series("train/mfu", window)

    print(f"run: {run.name} ({run.id})  state={run.state}  url={run.url}")
    print(f"gpu/config: {run.config.get('infrastructure', {}).get('slurm', {})}")
    if tps:
        print(f"tokens/s  (steps {lo}-{hi}): mean {statistics.mean(tps):,.0f}")
    if mfu:
        print(f"MFU       (steps {lo}-{hi}): mean {statistics.mean(mfu)*100:.1f}%")
    if step_times:
        mean_t = statistics.mean(step_times)
        std_t = statistics.stdev(step_times) if len(step_times) > 1 else 0.0
        print(f"step time (steps {lo}-{hi}): {mean_t:.3f}s ± {std_t:.3f} "
              f"(cv {100*std_t/mean_t:.1f}%)")
    mem = [r["train/max_memory_gib"] for r in rows if "train/max_memory_gib" in r]
    if mem:
        print(f"peak memory: {mem[-1]:.2f} GiB")

    # --- pass criteria ---
    verdicts = []
    if losses_all:
        first, last = losses_all[0], losses_all[-1]
        expected = math.log(50304)
        verdicts.append(("first loss ~ ln(50304)", abs(first - expected) < 0.15,
                         f"{first:.4f} vs {expected:.4f}"))
        verdicts.append(("loss decreased", last < first, f"{first:.4f} -> {last:.4f}"))
        verdicts.append(("loss finite", all(l == l for l in losses_all), ""))
    if step_times and len(step_times) > 1:
        cv = statistics.stdev(step_times) / statistics.mean(step_times)
        verdicts.append(("step-time cv < 10%", cv < 0.10, f"{100*cv:.1f}%"))
    # exact token accounting: tokens at the last logged step
    last_step = max(by_step)
    tokens_logged = by_step[last_step].get("tokens")
    if tokens_logged is not None:
        # the model consumes width-1 tokens per sequence, so the exact per-step
        # count is batch_size * (sequence_length - 1) ... unless the config packs
        # to sequence_length and the spec counts raw tokens; report both.
        expected_raw = (last_step + 1) * args.tokens_per_step
        print(f"tokens at step {last_step}: {tokens_logged:,} "
              f"(raw-spec expectation {expected_raw:,})")
    for name, ok, detail in verdicts:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


if __name__ == "__main__":
    main()
