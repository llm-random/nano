"""Smoke test for src.core.zephon_dataset with configs/_dataset/fineweb_zephon.yaml's
train/eval split, no GPU needed.

    python -m scripts.zephon_smoke <cache_dir> <index_dir> [steps]

Builds the train and eval DataLoaders the way the trainer does (get_dataloader, total
batch 512, sequence 1024) at world sizes 1, 2 and 4 -- each rank in its own process, one
after another -- and checks:
- train: every step's global batch is the same rows, in the same order, at all three;
  the eval dump's 50 files are gone from the train index (2410 -> 2360);
- eval: exactly max_samples rows, the same set on every iter() and at every world size;
- the two are disjoint, and train keeps going after an eval in the same process (they
  need separate caches).
"""

import hashlib
import json
import os
import subprocess
import sys
import time

COMMIT = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
TRAIN = f"hf://HuggingFaceFW/fineweb-edu@{COMMIT}~original/default/train"
EVAL = f"hf://HuggingFaceFW/fineweb-edu@{COMMIT}~original/CC-MAIN-2025-26/train"
EXCLUDE = ["data/CC-MAIN-2025-26/"]
BATCH, EVAL_SAMPLES = 512, 8192
BOS, EOS = 128000, 128001


def hashes(batch):
    return [hashlib.sha256(r.numpy().tobytes()).hexdigest()[:16] for r in batch]


def one_rank(cache_dir, index_dir, steps):
    from src.core.datasets import get_dataloader
    from src.core.zephon_dataset import ZephonDataset

    t0 = time.perf_counter()
    common = dict(sequence_length=1024, total_batch_size=BATCH, index_dir=index_dir,
                  seed=123, canonical_replicas=4)
    train = ZephonDataset(hf_path=TRAIN, cache_dir=cache_dir, exclude_prefixes=EXCLUDE,
                          name="fineweb_edu", **common)
    evald = ZephonDataset(hf_path=EVAL, cache_dir=cache_dir + "_eval", max_samples=EVAL_SAMPLES,
                          name="fineweb_edu_eval", **common)
    train_it = iter(get_dataloader(train, total_batch_size=BATCH, num_workers=0))
    train_rows, times = [], []
    for step in range(steps):
        batch = next(train_it)
        times.append(time.perf_counter() - t0)
        if step == 0:
            print(f"rank {os.environ['RANK']}/{os.environ['WORLD_SIZE']}: train batch "
                  f"{tuple(batch.shape)} {batch.dtype}, BOS {(batch == BOS).sum().item()}, "
                  f"EOS {(batch == EOS).sum().item()}", file=sys.stderr, flush=True)
        train_rows.append(hashes(batch))

    from zephon.io import Dataset
    n_train_shards = Dataset.from_path("fineweb_edu", TRAIN).shard_count()

    eval_loader = get_dataloader(evald, total_batch_size=BATCH, num_workers=0)
    eval_rows = [hashes(b) for b in eval_loader]           # full pass, as far as max_samples
    again = [hashes(b) for _, b in zip(range(2), eval_loader)]  # a new iter(), cut short
    after_eval = hashes(next(train_it))                    # train still alive after eval
    print(json.dumps({"train": train_rows, "times": times, "eval": eval_rows, "again": again,
                      "after_eval": after_eval, "n_train_shards": n_train_shards}), flush=True)
    os._exit(0)  # zephon threads; flush above, os._exit skips it


def run(world_size, cache_dir, index_dir, steps):
    per_rank = []
    for rank in range(world_size):
        env = dict(os.environ, WORLD_SIZE=str(world_size), RANK=str(rank))
        out = subprocess.run([sys.executable, "-m", "scripts.zephon_smoke", "--rank",
                              cache_dir, index_dir, str(steps)], env=env, check=True,
                             capture_output=True, text=True)
        sys.stderr.write(out.stderr[-300:])
        per_rank.append(json.loads(out.stdout.strip().splitlines()[-1]))
    # a step's global batch is the ranks' batches in rank order
    glob = lambda key, n: [sum((r[key][s] for r in per_rank), []) for s in range(n)]
    n_eval = len(per_rank[0]["eval"])
    return {"train": glob("train", steps), "eval": glob("eval", n_eval),
            "again": glob("again", 2), "after_eval": sum((r["after_eval"] for r in per_rank), []),
            "times": per_rank[0]["times"], "n_train_shards": {r["n_train_shards"] for r in per_rank}}


if __name__ == "__main__":
    if sys.argv[1] == "--rank":
        one_rank(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    cache_dir, index_dir = sys.argv[1], sys.argv[2]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    results = {ws: run(ws, cache_dir, index_dir, steps) for ws in (1, 2, 4)}
    base = results[1]
    ok = True
    for ws, r in results.items():
        eval_rows = [h for step in r["eval"] for h in step]
        train_rows = {h for step in r["train"] for h in step} | set(r["after_eval"])
        checks = {
            "train = ws1 (rows, order)": r["train"] == base["train"],
            "train shards 2360": r["n_train_shards"] == {2360},
            f"eval {EVAL_SAMPLES} rows": len(eval_rows) == EVAL_SAMPLES,
            "eval = ws1": r["eval"] == base["eval"],
            "eval same on re-iter": r["again"] == r["eval"][:2],
            "eval & train disjoint": not (set(eval_rows) & train_rows),
            "train after eval = ws1": r["after_eval"] == base["after_eval"],
        }
        ok &= all(checks.values())
        print(f"world_size {ws}: " + ", ".join(f"{k}: {'OK' if v else 'FAIL'}" for k, v in checks.items())
              + f"; rank-0 time to train batches {[round(t, 1) for t in r['times']]}")
    print("ALL OK" if ok else "SOME CHECKS FAILED")
