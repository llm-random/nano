"""Reduce a nano training job's slurm log to the numbers a results table needs.

    python scripts/run_summary.py slurm-32543_0.out [more.out ...]
    python scripts/run_summary.py --tsv slurm-*.out      # paste-ready rows

Everything it reports is printed by the training loop itself, so this works on
any finished run without the experiment tracker:

    MFU accounting:   device, world size, GFLOPs/token, peak TFLOP/s
                      -- the line that settles which FLOPs/token denominator a
                      run actually used, which matters whenever several model
                      shapes are being compared (GPT-2-arch h20 is 4.122,
                      H-balanced h20 3.97, AME 3.77)
    timers_ms:        per-step section split, present only with
                      trainer.profile_timers. Averaged over a steady-state
                      window, skipping warmup and the two profiled steps
    Peak CUDA memory: one line at the end of training, rank 0
    train/mfu etc.:   the tracker's end-of-run summary, when it is in the log

MFU is reported PER GPU: src/core/trainer.py divides global tokens/s by the
GLOBAL peak (one device's peak x world size), so the logged figure is already
per-GPU and is the only metric comparable across GPU counts. Raw tokens/s is
global and is not.

Exit status is 1 if any log shows a failure signature, so this doubles as a
"did it actually finish" check -- a log that merely stops looks identical to a
log that succeeded unless something inspects it.
"""

import argparse
import re
import statistics
import sys

# Steps before this are warmup/compile; steps 12-13 carry the kernel profiler
# when it is on, which distorts them. 15 clears both.
STEADY_FROM = 15

FAIL = re.compile(
    r"Traceback|CUDA out of memory|OutOfMemoryError|RuntimeError|"
    r"srun: error|slurmstepd: error|Killed|NCCL.*(?:error|timeout)|AssertionError",
    re.I,
)
# Order matters only for the printed table. bank_reduce and bank_step appear
# only with trainer.mem_parallel_size set (Tiers 2+3); they are absent, not
# zero, in every earlier run, and _section_table skips what it does not find.
SECTIONS = (
    "data_fetch",
    "data_h2d",
    "forward",
    "backward",
    "clip",
    "bank_reduce",
    "optim",
    "bank_step",
    "log",
)


def _fatal_lines(text):
    """Failure signatures, excluding ones the run already handled.

    A caught-and-logged exception is not a failed run. The trainer reports a
    broken per-layer diagnostic as `WARNING - instrumentation for <m> failed
    ...: RuntimeError: ...` and carries on; matching "RuntimeError" there marks
    a perfectly good run as failed, which is exactly as misleading as missing a
    real crash. Observed on lem 2026-11-04: 6237108/6237109 trained 30 steps
    each and were reported as failures.
    """
    hits = []
    for line in text.splitlines():
        if "WARNING" in line or "[WARNING]" in line:
            continue
        hits.extend(FAIL.findall(line))
    return sorted(set(hits))


def parse(path):
    text = open(path, errors="replace").read()
    out = {"path": path, "failures": _fatal_lines(text)}
    out["handled_warnings"] = len(
        [l for l in text.splitlines() if "WARNING" in l and FAIL.search(l)]
    )

    m = re.search(
        r"MFU accounting: (.+?) x(\d+), ([\d.]+) GFLOPs/token at seq_len (\d+), "
        r"peak (\S+) TFLOP/s",
        text,
    )
    if m:
        out.update(
            device=m.group(1),
            world=int(m.group(2)),
            gflops_token=float(m.group(3)),
            seq_len=int(m.group(4)),
            peak_tflops=m.group(5),
        )

    m = re.search(r"Peak CUDA memory: ([\d.]+) GiB", text)
    if m:
        out["peak_gib"] = float(m.group(1))

    # Tracker end-of-run summary, if the log carries it. LAST match, not first:
    # the tracker prints a history block before the summary block, so the first
    # occurrence is an early step rather than the final value.
    for key in ("train/mfu", "train/tokens_per_s", "train/step_time"):
        found = re.findall(rf"{re.escape(key)}\s+([\d.eE+-]+)", text)
        if found:
            out[key] = float(found[-1])

    rows = []
    for m in re.finditer(
        r"timers_ms: rank=(\d+) step=(\d+) (.+?) step_total=([\d.]+)", text
    ):
        if int(m.group(1)) != 0 or int(m.group(2)) < STEADY_FROM:
            continue
        row = {k: float(v) for k, v in re.findall(r"(\w+)=([\d.]+)", m.group(3))}
        row["step_total"] = float(m.group(4))
        rows.append(row)
    if rows:
        out["timers"] = {
            name: statistics.mean(r.get(name, 0.0) for r in rows)
            for name in (*SECTIONS, "step_total")
        }
        out["timer_steps"] = len(rows)

    losses = re.findall(r"train/loss\s+([\d.]+)", text)
    if losses:
        out["final_loss"] = float(losses[-1])
    out["has_kprof"] = "kprof_us:" in text
    return out


def show(info):
    print(f"\n=== {info['path']} ===")
    if info["failures"]:
        print(f"  !! FAILURE SIGNATURES: {', '.join(info['failures'])}")
    if info.get("handled_warnings"):
        print(
            f"  note: {info['handled_warnings']} warning line(s) mention an error "
            "but were handled by the run (not counted as failures)"
        )
    if "gflops_token" in info:
        print(
            f"  {info['device']} x{info['world']}   "
            f"{info['gflops_token']:.3f} GFLOPs/token @ seq {info['seq_len']}   "
            f"peak {info['peak_tflops']} TFLOP/s (global)"
        )
    bits = []
    if "train/tokens_per_s" in info:
        bits.append(f"tokens/s(global)={info['train/tokens_per_s']:,.0f}")
    if "train/mfu" in info:
        bits.append(f"MFU/GPU={info['train/mfu'] * 100:.2f}%")
    if "train/step_time" in info:
        bits.append(f"step={info['train/step_time']:.3f}s")
    if "peak_gib" in info:
        bits.append(f"peak={info['peak_gib']:.1f}GiB")
    if "final_loss" in info:
        bits.append(f"loss={info['final_loss']:.4f}")
    if bits:
        print("  " + "   ".join(bits))

    if "timers" in info:
        t = info["timers"]
        total = t["step_total"] or 1.0
        print(
            f"  section split, mean of {info['timer_steps']} steps from {STEADY_FROM}:"
        )
        covered = 0.0
        for name in SECTIONS:
            ms = t[name]
            covered += ms
            print(f"      {name:12s} {ms:9.1f} ms  {100 * ms / total:5.1f}%")
        print(f"      {'step_total':12s} {total:9.1f} ms")
        print(
            f"      covered by sections: {100 * covered / total:.1f}%  "
            f"(GPU and host sections overlap, so this can exceed 100)"
        )
    elif "gflops_token" in info:
        print("  (no timers_ms lines -- run without trainer.profile_timers)")
    if info["has_kprof"]:
        print(
            "  kprof_us lines present -> python scripts/kprof_agg.py "
            f"{info['path']} -v"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--tsv", action="store_true", help="one tab-separated row per log")
    args = ap.parse_args()

    infos = [parse(p) for p in args.logs]
    if args.tsv:
        print("log\tgflops_token\tworld\ttokens_per_s\tmfu_pct\tstep_s\tpeak_gib")
        for i in infos:
            print(
                f"{i['path']}\t{i.get('gflops_token', '')}\t{i.get('world', '')}\t"
                f"{i.get('train/tokens_per_s', '')}\t"
                f"{100 * i['train/mfu'] if 'train/mfu' in i else ''}\t"
                f"{i.get('train/step_time', '')}\t{i.get('peak_gib', '')}"
            )
    else:
        for i in infos:
            show(i)
    return 1 if any(i["failures"] for i in infos) else 0


if __name__ == "__main__":
    sys.exit(main())
