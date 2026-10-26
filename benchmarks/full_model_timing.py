"""Step-time extraction for full-model PC vs HP timing sweeps logged to wandb.

Per optimizer step the trainer logs one wandb row; the gap between consecutive row
timestamps is that step's wall-clock time (rank 0). We report the mean over the last
`--last` steps of each run and the PC-vs-HP overhead at equal total batch.

Usage (repo root, pixi env):
  python benchmarks/full_model_timing.py --pc pc90_gacc64_sweep --hp hp90_gacc64_sweep \
      --out benchmarks/results/full_90p_gacc64_10step.jsonl
"""

import argparse
import json
import statistics

import wandb


def collect(api, project, name, mode, n_last):
    out = {}
    for r in api.runs(project, filters={"display_name": name}):
        c = r.config
        seqs = c["common"]["batch_size"]
        n_steps = c["trainer"]["n_steps"]
        ts = sorted(
            (x["_step"], x["_timestamp"])
            for x in r.scan_history(keys=["_step", "_timestamp"])
            if x.get("_timestamp") is not None
        )
        deltas = [b[1] - a[1] for a, b in zip(ts, ts[1:])]
        ok = r.state == "finished" and len(deltas) >= n_last
        rec = {
            "mode": mode,
            "run_id": r.id,
            "state": r.state,
            "seqs": seqs,
            "tokens": seqs * c["common"]["sequence_length"],
            "grad_acc": c["trainer"]["gradient_accumulation_steps"],
            "micro": seqs // c["trainer"]["gradient_accumulation_steps"],
            "n_steps": n_steps,
            "logged_rows": len(ts),
            "status": "ok" if ok else r.state,
            "step_times": deltas,
        }
        if ok:
            last = deltas[-n_last:]
            rec.update(
                s_per_step=statistics.mean(last),
                s_std=statistics.stdev(last),
                last_n=n_last,
            )
        prev = out.get(seqs)
        if prev is None or (prev["status"] != "ok" and rec["status"] == "ok"):
            out[seqs] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="ideas_cv/pc_time")
    ap.add_argument("--pc", required=True)
    ap.add_argument("--hp", required=True)
    ap.add_argument("--last", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    api = wandb.Api(timeout=120)
    pc = collect(api, args.project, args.pc, "pc", args.last)
    hp = collect(api, args.project, args.hp, "hp", args.last)
    with open(args.out, "w") as f:
        for rec in list(pc.values()) + list(hp.values()):
            f.write(json.dumps(rec) + "\n")

    print(
        "| total tokens | micro×acc | HP s/step | PC s/step | PC−HP (s) | **PC overhead** |"
    )
    print("|---|---|---|---|---|---|")
    for seqs in sorted(set(pc) | set(hp)):
        a, b = pc.get(seqs, {}), hp.get(seqs, {})
        if a.get("status") == "ok" and b.get("status") == "ok":
            d = a["s_per_step"] - b["s_per_step"]
            print(
                f"| {a['tokens']:,} | {a['micro']}×{a['grad_acc']} | {b['s_per_step']:.2f} | {a['s_per_step']:.2f} "
                f"| {d:+.2f} | **{100 * d / b['s_per_step']:.1f}%** |"
            )
        else:
            print(
                f"| {seqs * 1024:,} | — | {b.get('status', 'missing')} | {a.get('status', 'missing')} | — | — |"
            )
    print("wrote", args.out)


if __name__ == "__main__":
    main()
