"""Smoke test for src.core.zephon_dataset, no GPU needed.

    python -m scripts.zephon_smoke <cache_dir> <index_dir> [steps]

Builds the train DataLoader the way the trainer does (get_dataloader, total batch 512,
sequence 1024) at world sizes 1, 2 and 4 -- each rank in its own process, one after
another -- and checks that every step's global batch is the same set of rows, in the
same order, at all three. Also prints the shape and the BOS/EOS count of the first batch.
"""

import hashlib
import json
import os
import subprocess
import sys
import time

HF_PATH = "hf://HuggingFaceFW/fineweb-edu@87f09149ef4734204d70ed1d046ddc9ca3f2b8f9~original/default/train"
BOS, EOS = 128000, 128001


def one_rank(cache_dir, index_dir, steps):
    from src.core.datasets import get_dataloader
    from src.core.zephon_dataset import ZephonDataset

    t0 = time.perf_counter()
    ds = ZephonDataset(sequence_length=1024, total_batch_size=512, hf_path=HF_PATH,
                       cache_dir=cache_dir, index_dir=index_dir, seed=123, canonical_replicas=4)
    rows, times = [], []
    for step, batch in enumerate(get_dataloader(ds, total_batch_size=512, num_workers=0)):
        if step == steps:
            break
        times.append(time.perf_counter() - t0)
        if step == 0:
            print(f"rank {os.environ['RANK']}/{os.environ['WORLD_SIZE']}: batch {tuple(batch.shape)} "
                  f"{batch.dtype}, BOS {(batch == BOS).sum().item()}, EOS {(batch == EOS).sum().item()}",
                  file=sys.stderr, flush=True)
        rows.append([hashlib.sha256(r.numpy().tobytes()).hexdigest()[:16] for r in batch])
    print(json.dumps({"rows": rows, "times": times}))
    os._exit(0)


def run(world_size, cache_dir, index_dir, steps):
    per_rank = []
    for rank in range(world_size):
        env = dict(os.environ, WORLD_SIZE=str(world_size), RANK=str(rank))
        out = subprocess.run([sys.executable, "-m", "scripts.zephon_smoke", "--rank",
                              cache_dir, index_dir, str(steps)], env=env, check=True,
                             capture_output=True, text=True)
        sys.stderr.write(out.stderr[-300:])
        per_rank.append(json.loads(out.stdout.strip().splitlines()[-1]))
    # the global batch of a step is the ranks' batches in rank order
    steps_rows = [sum((r["rows"][s] for r in per_rank), []) for s in range(steps)]
    return steps_rows, per_rank[0]["times"]


if __name__ == "__main__":
    if sys.argv[1] == "--rank":
        one_rank(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    cache_dir, index_dir = sys.argv[1], sys.argv[2]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    results = {ws: run(ws, cache_dir, index_dir, steps) for ws in (1, 2, 4)}
    base = results[1][0]
    for ws, (glob, times) in results.items():
        same = sum(sorted(a) == sorted(b) for a, b in zip(glob, base))
        ordered = sum(a == b for a, b in zip(glob, base))
        print(f"world_size {ws}: steps with the same rows as ws1 {same}/{steps}, "
              f"same order {ordered}/{steps}; rank-0 time to batches {[round(t, 1) for t in times]}")
