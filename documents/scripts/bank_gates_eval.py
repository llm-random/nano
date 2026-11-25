#!/usr/bin/env python
"""Bank gates, the secondary evaluation: is a rarely updated row less influential?

documents/notes/ame_bank_gates_plan.md, "Evaluation". Loads a final checkpoint
into the common evaluator's model (`ame_ref32`: fp32, eager, reference attention,
banks from the fp32 masters; documents/scripts/parity_numerics.py), buckets the
vocabulary by `steps_seen` u from the run's token_counts.pt -- the number of
optimizer steps a token occurred in, i.e. the steps in which its bank rows had a
gradient (AdamW's momentum moves them in later steps too) -- and reports per
bucket:

  loss          mean next-token CE at held-out positions whose INPUT token is in
                the bucket (the whole held-out set)
  as queries    attention entropy, normalized by ln(visible keys) per row; and
                the spread (std over the visible keys) of the identity part of
                the logits, q . k without the distance bias
  as keys       the attention mass the bucket's keys receive, divided by what
                uniform attention over the visible keys would give them (1.0 is
                exactly their share; below 1, less)
  per token id  the bank rows' length per head; B1's trust s = sigmoid(c);
                B3's cos(row, base)
  per position  B2's gate g, and ST's gate RMS, from real activations

Attention statistics come from the first --attn-seqs held-out sequences, rounded
up to whole micro-batches (the report's attn_seqs is the number measured), every
query row, averaged over heads and then over layers (per-layer values are in the
JSON). Buckets: u = 0, 1-10, 11-100, 101-1000, 1001-(S-1), S, where S is the
number of steps the counts cover (3815 for a full run).

    python documents/scripts/bank_gates_eval.py --config CFG --ckpt FOLDER \
        --data HELDOUT --out DIR --tag NAME [--counts FILE] [--attn-seqs 512]

The total CE it prints is the common evaluator's (`ame_ref32`), computed in the
same pass.
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parity_numerics as pn  # noqa: E402  (sets up REPO on sys.path and env)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

BUCKETS = 6


def bucket_labels(steps):
    return ["0", "1-10", "11-100", "101-1000", f"1001-{steps - 1}", f"{steps}"]


def bucket_of_ids(steps_seen, steps):
    """[vocab] bucket index per token id, from its steps_seen u."""
    u = steps_seen
    bucket = torch.full_like(u, 4)
    bucket[u == 0] = 0
    bucket[(u >= 1) & (u <= 10)] = 1
    bucket[(u >= 11) & (u <= 100)] = 2
    bucket[(u >= 101) & (u <= 1000)] = 3
    bucket[u >= steps] = 5
    return bucket


def _bincount(index, weights=None):
    return torch.bincount(index, weights=weights, minlength=BUCKETS).double()


class AttentionStats:
    """Per-layer, per-bucket accumulators, filled by a forward hook on every
    AMEAttention while `collect` is on."""

    def __init__(self, layers, bucket_of, device):
        self.layers = layers
        self.bucket_of = bucket_of
        self.collect = False
        zeros = lambda: torch.zeros(len(layers), BUCKETS, dtype=torch.float64)  # noqa
        self.sums = {
            name: zeros()
            for name in (
                "entropy",
                "rows",
                "spread",
                "spread_rows",
                "mass",
                "uniform",
                "gate",
            )
        }
        self.handles = [
            layer.register_forward_hook(self._hook(i)) for i, layer in enumerate(layers)
        ]

    def _hook(self, index):
        def hook(module, args, output):
            if self.collect:
                self._measure(index, module, *args)

        return hook

    @torch.no_grad()
    def _measure(self, index, module, x, token_ids):
        q, k, _ = module._mixed_qkv(x, token_ids)  # [B, H, n, D], tau and gates in q
        k, _, offset = module._with_sink(k, k)
        batch, _, n, _ = q.shape
        device = q.device
        bias = module.score_bias(n, device, torch.float32, offset)  # [H, n, n+off]
        rows = torch.arange(n, device=device)
        visible = (rows + 1 + offset).double()
        causal = torch.tril(torch.ones(n, n, device=device, dtype=torch.bool))
        # What uniform attention over the visible keys (the sink included) gives
        # key j, summed over the query rows that can see it: sum_{i>=j} 1/(i+1+off).
        per_row = 1.0 / visible
        uniform = per_row.flip(0).cumsum(0).flip(0)
        for b in range(batch):
            ids = token_ids[b]
            bucket = self.bucket_of[ids]
            scores = torch.einsum("hid,hjd->hij", q[b].float(), k[b].float())
            weights = (scores + bias).softmax(dim=-1)
            entropy = -(weights * weights.clamp_min(1e-30).log()).sum(-1)  # [H, n]
            entropy = (entropy / visible.log().clamp_min(1e-6).float()).mean(0)
            # The identity part over the visible SEQUENCE keys (the sink is not
            # a token): std over j <= i, from row 1 on.
            seq = scores[:, :, offset:]
            count = (rows + 1).float()
            mean = seq.masked_fill(~causal, 0.0).sum(-1) / count
            var = ((seq - mean.unsqueeze(-1)) ** 2).masked_fill(~causal, 0.0).sum(-1)
            spread = (var / count).sqrt().mean(0)  # [n]
            received = weights[:, :, offset:].sum(1).mean(0)  # [n] keys, over heads
            sums = self.sums
            sums["entropy"][index] += _bincount(bucket, entropy.double()).cpu()
            sums["rows"][index] += _bincount(bucket).cpu()
            sums["spread"][index] += _bincount(bucket[1:], spread[1:].double()).cpu()
            sums["spread_rows"][index] += _bincount(bucket[1:]).cpu()
            sums["mass"][index] += _bincount(bucket, received.double()).cpu()
            sums["uniform"][index] += _bincount(bucket, uniform).cpu()
            if module.bank_gate == "context":
                gate = torch.sigmoid(module.gate(x[b]).float()).mean(-1)  # [n]
                sums["gate"][index] += _bincount(bucket, gate.double()).cpu()
            elif module.bank_gate == "stem":
                gate = F.silu(module.gate(x[b]).float()).pow(2).mean(-1).sqrt()
                sums["gate"][index] += _bincount(bucket, gate.double()).cpu()

    def per_layer(self):
        s = self.sums
        out = {
            "entropy_normalized": s["entropy"] / s["rows"],
            "identity_spread": s["spread"] / s["spread_rows"],
            "key_mass_vs_uniform": s["mass"] / s["uniform"],
        }
        if self.layers and self.layers[0].bank_gate in ("context", "stem"):
            out["gate"] = s["gate"] / s["rows"]
        return out


@torch.no_grad()
def vocab_stats(layers, bucket_of):
    """Per-layer, per-bucket means over token ids of parameter readouts."""
    stats = {}

    def add(name, layer_index, values):
        table = stats.setdefault(
            name, torch.zeros(len(layers), BUCKETS, dtype=torch.float64)
        )
        n = _bincount(bucket_of.cpu())
        table[layer_index] = _bincount(bucket_of.cpu(), values.double().cpu()) / n

    for i, layer in enumerate(layers):
        heads, dhead = layer.q_heads, layer.dhead
        datt = heads * dhead
        for which in ("q", "k"):
            weight = getattr(layer, f"{which}_bank").weight.detach().float()
            rows = weight[:, :datt].view(-1, heads, dhead)
            add(f"row_length_{which}", i, rows.norm(dim=-1).mean(-1))
            if layer.bank_gate == "confidence":
                add(f"trust_{which}", i, torch.sigmoid(weight[:, datt:]).mean(-1))
            if layer.bank_gate == "backoff":
                base = getattr(layer, f"{which}_base").detach().float()
                cos = F.cosine_similarity(rows, base.unsqueeze(0), dim=-1)
                add(f"cos_row_base_{which}", i, cos.mean(-1))
    return stats


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--counts", default=None, help="default: CKPT/token_counts.pt")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", default="run")
    parser.add_argument("--attn-seqs", type=int, default=512)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    args.smoke = False  # pn.load_heldout reads it
    os.makedirs(args.out, exist_ok=True)
    pn.EXTRA_OVERRIDES.extend(args.override)

    import main as main_module
    from src.core.ame import AMEAttention

    main_module.setup_enviroment()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert not torch.backends.cuda.matmul.allow_tf32, "fp32 references need TF32 off"
    t0 = time.time()

    counts = torch.load(
        args.counts or os.path.join(args.ckpt, "token_counts.pt"),
        map_location="cpu",
        weights_only=False,
    )
    steps = int(counts["steps"])
    bucket_of = bucket_of_ids(counts["steps_seen"], steps).to(device)
    labels = bucket_labels(steps)

    built = pn.build("ame_ref32", args.config, device)
    pn.load_checkpoint(built, args.ckpt)
    model = built.model.eval()
    layers = [m for m in model.modules() if isinstance(m, AMEAttention)]
    gate = layers[0].bank_gate
    attention = AttentionStats(layers, bucket_of, device)

    held = pn.load_heldout(args, device)
    micro = built.micro
    loss_sum = torch.zeros(BUCKETS, dtype=torch.float64)
    loss_count = torch.zeros(BUCKETS, dtype=torch.float64)
    attn_measured = 0  # whole micro-batches: up to micro - 1 past --attn-seqs
    with torch.no_grad():
        for start in range(0, held.shape[0], micro):
            chunk = held[start : start + micro]
            inp, tgt = chunk[:, :-1], chunk[:, 1:]
            attention.collect = start < args.attn_seqs
            if attention.collect:
                attn_measured += chunk.shape[0]
            logits = model(inp)
            losses = F.cross_entropy(
                logits.float().flatten(0, -2), tgt.reshape(-1).long(), reduction="none"
            )
            bucket = bucket_of[inp.reshape(-1)]
            loss_sum += _bincount(bucket, losses.double()).cpu()
            loss_count += _bincount(bucket).cpu()
    attention.collect = False

    total_ce = float(loss_sum.sum() / loss_count.sum())
    per_layer = attention.per_layer()
    per_layer.update(vocab_stats(layers, bucket_of))
    ids_per_bucket = _bincount(bucket_of.cpu())

    summary = {}
    for b, label in enumerate(labels):
        row = {
            "token_ids": int(ids_per_bucket[b]),
            "heldout_positions": int(loss_count[b]),
            "loss": (
                float(loss_sum[b] / loss_count[b]) if loss_count[b] else float("nan")
            ),
        }
        for name, table in per_layer.items():
            column = table[:, b]
            row[name] = float(column.mean()) if torch.isfinite(column).all() else None
        summary[label] = row

    report = {
        "tag": args.tag,
        "config": args.config,
        "ckpt": args.ckpt,
        "bank_gate": gate,
        "steps": steps,
        "attn_seqs": attn_measured,
        "attn_seqs_requested": args.attn_seqs,
        "heldout_sequences": held.shape[0],
        "ce_ref32": total_ce,
        "buckets": summary,
        "per_layer": {k: v.tolist() for k, v in per_layer.items()},
        "seconds": time.time() - t0,
    }
    pn.save_json(os.path.join(args.out, f"buckets_{args.tag}.json"), report)

    columns = ["token_ids", "heldout_positions", "loss"] + list(per_layer)
    pn.log(f"{args.tag} ({gate}): ce_ref32 {total_ce:.6f} over {held.shape[0]} seqs")
    print("| u | " + " | ".join(columns) + " |")
    print("|---" * (len(columns) + 1) + "|")
    for label in labels:
        cells = []
        for column in columns:
            value = summary[label][column]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                cells.append("-")
            elif isinstance(value, int):
                cells.append(f"{value:,}")
            else:
                cells.append(f"{value:.4f}")
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
