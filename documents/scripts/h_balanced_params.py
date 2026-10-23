"""Dimensions and parameter counts of the H-balanced transformer family.

See documents/concepts/h-balanced-model.md. Counting convention: GQA with kv_heads = H/2, gated
feed-forward (gate + up + down = 3 matrices of dmodel x dff), untied embedding and head, norms
without bias.

    python documents/scripts/h_balanced_params.py [H ...]
"""

import sys

VOCAB = 50257
HEAD_DIM = 64


def dims(H: int) -> dict:
    """Every dimension of an H-balanced model, derived from H alone."""
    if H % 2 != 0:
        raise ValueError(f"H must be divisible by 2, got {H}")
    dmodel = 64 * H
    return dict(
        dmodel=dmodel,
        dff=4 * dmodel,
        datt=dmodel,
        n_blocks=H,
        q_heads=H,
        kv_heads=H // 2,
    )


def params(
    H: int, vocab: int = VOCAB, gated_ff: bool = True, tied: bool = False
) -> dict:
    d = dims(H)
    dmodel, dff, L = d["dmodel"], d["dff"], d["n_blocks"]
    ff_matrices = 3 if gated_ff else 2  # gate + up + down, vs. a plain up + down FF
    embedding = vocab * dmodel * (1 if tied else 2)
    attention = L * (2 * dmodel * dmodel + 2 * dmodel * HEAD_DIM * d["kv_heads"])
    feedforward = L * ff_matrices * dmodel * dff
    norms = L * 2 * dmodel + dmodel
    return dict(
        embedding=embedding,
        attention=attention,
        feedforward=feedforward,
        norms=norms,
        total=embedding + attention + feedforward + norms,
    )


def main(hs):
    print(f"vocab {VOCAB}, GQA (kv = H/2), gated FF (gate + up + down), untied head")
    print(
        f"{'H':>3} {'dmodel':>6} {'L':>3} {'q':>3} {'kv':>3} {'dff':>6} {'params':>11} {'non-emb':>11}"
    )
    for H in hs:
        d, p = dims(H), params(H)
        print(
            f"{H:>3} {d['dmodel']:>6} {d['n_blocks']:>3} {d['q_heads']:>3} {d['kv_heads']:>3}"
            f" {d['dff']:>6} {p['total'] / 1e6:>10.1f}M {(p['total'] - p['embedding']) / 1e6:>10.1f}M"
        )


if __name__ == "__main__":
    main(
        [int(a) for a in sys.argv[1:]]
        or [12, 16, 20, 24, 32, 40, 50, 64, 80, 100, 126, 158, 200, 252]
    )
