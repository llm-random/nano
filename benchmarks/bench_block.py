"""Single-transformer-block step-time benchmark: PC vs HP (paper Fig. 2).

Drops the embedding and the LM head: random bf16 hidden states go into ONE real
Llama block (repo classes, same dims as training) and a dummy loss comes out, so
no 128k-vocab logits limit the batch size.

  PC step = materialize W_c = P_out W P_in + W_b (fp32, no grad)
            -> block fwd/bwd (bf16 autocast, like FSDP2 param_dtype=bf16)
            -> projection backward (fp32, as backward_compressed_weights)
            -> per-block clip 0.25 -> AdamW(projections); clip 1.0 -> AdamW(norms)
  HP step = block fwd/bwd (bf16 autocast) -> clip 1.0 -> AdamW(block params)

The block input requires grad (a middle layer also computes dL/dx). Weights and
data are random: values do not affect timing. For each batch size we report the
median step time over the timed steps and the exact peak memory, stopping at
the first OOM.

Usage (on a GPU node, from the repo root):
  PYTHONPATH=$PWD python benchmarks/bench_block.py --config-dir $PWD/configs/pc_project_mas \
      --config-name llama_1B_pc_1blk_bs_probe_50p_entropy --mode pc --seqs 98 196 391
"""

import argparse
import json
import statistics
import time

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import resolver  # noqa: F401  (registers OmegaConf resolvers used by the configs)
from src.projected_compression.mem_eff import CompressibleBlock

# (CompressibleBlock attribute, Linear module path inside TransformerBlock)
LINEARS = [
    ("compressible_q", "attention_layer.layer.q_proj"),
    ("compressible_k", "attention_layer.layer.k_proj"),
    ("compressible_v", "attention_layer.layer.v_proj"),
    ("compressible_o", "attention_layer.layer.o_proj"),
    ("compressible_ff_pre", "ff_layer.layer.ff_pre_act"),
    ("compressible_ff_gate", "ff_layer.layer.gate"),
    ("compressible_ff_post", "ff_layer.layer.ff_post_act"),
]


def _get(module, path):
    for name in path.split("."):
        module = getattr(module, name)
    return module


def build(config_dir, config_name, mode, device):
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)
    c = cfg.common
    target = instantiate(cfg.model.target_model.encoder.block_fn)(0).to(device)
    norm_params = list(target.attention_layer.norm.parameters()) + list(
        target.ff_layer.norm.parameters()
    )
    m = {"mode": mode, "target": target, "dmodel": c.dmodel, "dff": c.dff}

    if mode == "hp":
        params = list(target.parameters())
        m["opt"] = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.1)
        m["params"] = params
        return m

    source = instantiate(cfg.model.source_model.encoder.block_fn)(0).to(device)
    source.requires_grad_(False)
    dmodel_idx = torch.randperm(c.base_dmodel)[: c.dmodel]
    dff_idx = torch.randperm(c.base_dff)[: c.dff]
    proj = CompressibleBlock(
        q_heads=c.q_heads,
        kv_heads=c.kv_heads,
        dhead=c.dhead,
        base_dmodel=c.base_dmodel,
        target_dmodel=c.dmodel,
        base_dff=c.base_dff,
        target_dff=c.dff,
        dmodel_topk_indices=dmodel_idx,
        dff_topk_indices=dff_idx,
        cast_bfloat16=False,
    )
    proj.init_projection_weights(dmodel_idx, dff_idx)
    proj = proj.to(device)
    m["triplets"] = [
        (getattr(proj, cl), _get(target, path).weight, _get(source, path).weight)
        for cl, path in LINEARS
    ]
    m["proj"] = proj
    m["norm_params"] = norm_params
    m["opt_proj"] = torch.optim.AdamW(proj.parameters(), lr=1e-4, weight_decay=0.1)
    m["opt_norm"] = torch.optim.AdamW(norm_params, lr=1e-4, weight_decay=0.1)
    return m


def step(m, x, grad_acc=1):
    """One optimizer step. Like PCTrainer.train: W_c is materialized once, `grad_acc`
    microbatch fwd/bwd passes accumulate into W_c.grad, then the projection backward and
    optimizer step run once. `x` is one microbatch, reused for every accumulation pass."""
    target = m["target"]
    if m["mode"] == "pc":
        with torch.no_grad():
            for cl, w_target, w_source in m["triplets"]:
                w_target.copy_(cl.get_projected_weight(w_source))

    for _ in range(grad_acc):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = target(x).mean() / grad_acc
        loss.backward()
        x.grad = None

    if m["mode"] == "pc":
        for cl, w_target, w_source in m["triplets"]:
            cl.get_projected_weight(w_source).backward(w_target.grad)
            w_target.grad = None
        torch.nn.utils.clip_grad_norm_(m["proj"].parameters(), 0.25)
        m["opt_proj"].step()
        m["opt_proj"].zero_grad(set_to_none=True)
        torch.nn.utils.clip_grad_norm_(m["norm_params"], 1.0)
        m["opt_norm"].step()
        m["opt_norm"].zero_grad(set_to_none=True)
    else:
        torch.nn.utils.clip_grad_norm_(m["params"], 1.0)
        m["opt"].step()
        m["opt"].zero_grad(set_to_none=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", required=True)
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--mode", choices=["pc", "hp"], required=True)
    ap.add_argument("--seqs", type=int, nargs="+", required=True)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=2, help="untimed steps before measuring")
    ap.add_argument("--steps", type=int, default=10, help="timed steps")
    ap.add_argument("--stat", choices=["median", "mean"], default="median")
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds random weights, projection index choice and input data")
    ap.add_argument("--grad-acc", type=int, default=1,
                    help="microbatches per optimizer step; --seqs is the TOTAL per step")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    m = build(args.config_dir, args.config_name, args.mode, device)
    tag = {"config": args.config_name, "mode": args.mode, "dmodel": m["dmodel"], "dff": m["dff"],
           "grad_acc": args.grad_acc, "seed": args.seed}

    for seqs in sorted(args.seqs):
        assert seqs % args.grad_acc == 0, f"--seqs {seqs} not divisible by --grad-acc {args.grad_acc}"
        tokens = seqs * args.seq_len
        x = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            x = torch.randn(
                seqs // args.grad_acc, args.seq_len, m["dmodel"], device=device, dtype=torch.bfloat16,
                requires_grad=True,
            )
            # Time every step (warm-up included) so warm-up behaviour can be inspected;
            # the reported statistic uses only the last `steps` of them.
            all_times = []
            for _ in range(args.warmup + args.steps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                step(m, x, args.grad_acc)
                torch.cuda.synchronize()
                all_times.append(time.perf_counter() - t0)
            times = all_times[args.warmup:]
            t = statistics.median(times) if args.stat == "median" else statistics.mean(times)
            rec = {**tag, "seqs": seqs, "tokens": tokens, "status": "ok", "stat": args.stat,
                   "warmup": args.warmup, "steps": args.steps, "s_per_step": t,
                   "s_std": statistics.stdev(times) if len(times) > 1 else 0.0,
                   "tok_per_s": tokens / t, "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                   "step_times": times, "all_step_times": all_times}
        except torch.OutOfMemoryError:
            rec = {**tag, "seqs": seqs, "tokens": tokens, "status": "oom"}
        print("BENCH " + json.dumps(rec), flush=True)
        if rec["status"] == "oom":
            break
        del x


if __name__ == "__main__":
    main()
