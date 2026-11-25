"""AME readouts from a final checkpoint, per layer and averaged over the layers.

src/core/ame.py's `instrumentation` logs these to W&B. Under FSDP2 it failed and
was disabled until 8b78b805 (ame_experiments.md, Stage 0), so the runs from
before that fix -- all of Stage 2c -- logged none of them, and this script reads
them from each final checkpoint instead. Runs since log them under FSDP2 too,
except attention entropy and top-k mass on a sharded layer. tau, alpha and beta
are parameters, and the ALiBi slopes a persistent buffer, so the checkpoint
holds them exactly.

Per layer, computed as `instrumentation` computes them:
- tau/mean, tau/min: the learned per-head temperature;
- horizon/min, horizon/max: tau / slope per head, in tokens (QK-norm on, so tau
  scales the scores; without it the horizon would be 1 / slope);
- alpha_2/absmean, beta_2/absmean: |coefficient on the previous token|, the
  mean over heads of column 1 of each head's (c1, c2) pair; also min and max.
Attention entropy and top-8 mass need a forward pass, so they are not here.

Run it on the cluster with the project env's python:
    python ame_ckpt_readouts.py <checkpoint step dir> [...]
A step dir holds either an FSDP2 distributed checkpoint (.metadata plus
__<rank>_0.distcp files; only the needed tensors are read) or a single
__model_checkpoint_filename.pt (memory-mapped, so only they are paged in).
"""

import os
import re
import sys

import torch

PAT = re.compile(
    r"(?:^|\.)encoder\.blocks\.(\d+)\.attention_layer\.layer\.(tau|alpha|beta|slopes)$"
)


def load(step_dir):
    if os.path.exists(os.path.join(step_dir, ".metadata")):
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint import FileSystemReader

        reader = FileSystemReader(step_dir)
        md = reader.read_metadata()
        sd = {
            k: torch.empty(m.size, dtype=m.properties.dtype)
            for k, m in md.state_dict_metadata.items()
            if PAT.search(k)
        }
        dcp.load(sd, storage_reader=reader, no_dist=True)
        return sd
    obj = torch.load(
        os.path.join(step_dir, "__model_checkpoint_filename.pt"),
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    flat = {}

    def walk(prefix, o):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif torch.is_tensor(o) and PAT.search(prefix):
            flat[prefix] = o.detach().clone()

    walk("", obj)
    return flat


def readouts(sd):
    layers = {}
    for k, v in sd.items():
        m = PAT.search(k)
        layers.setdefault(int(m.group(1)), {})[m.group(2)] = v.float()
    rows = []
    for i in sorted(layers):
        d = layers[i]
        missing = {"tau", "alpha", "beta", "slopes"} - set(d)
        assert not missing, f"layer {i}: no {sorted(missing)} in the checkpoint"
        tau, hor = d["tau"], d["tau"] / d["slopes"]
        a2, b2 = d["alpha"][:, 1], d["beta"][:, 1]
        rows.append(
            {
                "tau/mean": tau.mean().item(),
                "tau/min": tau.min().item(),
                "horizon/min": hor.min().item(),
                "horizon/max": hor.max().item(),
                "alpha_2/absmean": a2.abs().mean().item(),
                "beta_2/absmean": b2.abs().mean().item(),
                "alpha_2/min": a2.min().item(),
                "alpha_2/max": a2.max().item(),
                "beta_2/min": b2.min().item(),
                "beta_2/max": b2.max().item(),
            }
        )
    return rows


def main(paths):
    for path in paths:
        rows = readouts(load(path))
        if not rows:
            sys.exit(
                f"{path}: no AME layers found (no key matches {PAT.pattern}); "
                "a dense model, or a renamed module path?"
            )
        print(f"== {path}: {len(rows)} layers")
        if rows and max(r["tau/mean"] for r in rows) < 1:
            print("   tau < 1 everywhere: no QK-norm? horizon/* assume it is on")
        for key in rows[0]:
            vals = [r[key] for r in rows]
            print(
                f"   {key:16s} mean {sum(vals) / len(vals):9.4f}  per layer "
                + " ".join(f"{v:.3g}" for v in vals)
            )


if __name__ == "__main__":
    main(sys.argv[1:])
