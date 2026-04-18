"""Audit simpleP param groups: which params land in which group and at what LR.

Usage:
    pixi run python scratch_simpleP_audit.py --config-name=simpleP/k8
"""
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

from main import build_simpleP_param_groups


@hydra.main(version_base=None, config_path="configs/simpleP", config_name="k8")
def main(cfg: OmegaConf):
    base_lr = 1.0  # use 1.0 so printed lr IS the scale factor
    dmodel = cfg.common.dmodel
    dff = cfg.common.dff
    simpleP = cfg.get("simpleP", None)
    print(f"dmodel={dmodel}, dff={dff}")
    if simpleP is None:
        print("simpleP: NOT configured (vanilla)")
    else:
        print(f"base_dmodel={simpleP.base_model.dmodel}, base_dff={simpleP.base_model.dff}")
        print(f"expected scale for dmodel fan_in: {simpleP.base_model.dmodel/dmodel}")
        print(f"expected scale for dff fan_in:    {simpleP.base_model.dff/dff}")

    model = instantiate(cfg.model, _convert_="all")

    # Build name -> param and id -> name maps.
    id_to_name = {id(p): n for n, p in model.named_parameters()}
    all_param_ids = set(id(p) for p in model.parameters())

    groups = build_simpleP_param_groups(model, base_lr)

    print(f"\nTotal param groups: {len(groups)}")
    print(f"Total named params: {len(id_to_name)}")

    # Bucket groups by their LR value for readability.
    by_lr = {}
    seen_param_ids = set()
    dup_ids = []
    for g in groups:
        lr = g["lr"]
        by_lr.setdefault(lr, []).append(g)
        for p in g["params"]:
            if id(p) in seen_param_ids:
                dup_ids.append(id(p))
            seen_param_ids.add(id(p))

    for lr in sorted(by_lr.keys()):
        gs = by_lr[lr]
        n_params = sum(sum(p.numel() for p in g["params"]) for g in gs)
        print(f"\n=== LR = {lr:.6g}  (× base) ===  [{len(gs)} groups, {n_params:,} params total]")
        # Collect and unique (name, shape) rows.
        rows = []
        for g in gs:
            for p in g["params"]:
                rows.append((id_to_name.get(id(p), "<unknown>"), tuple(p.shape)))
        # Show first 12 + count of the rest.
        rows_sorted = sorted(set(rows))
        for name, shape in rows_sorted[:12]:
            print(f"  {name}  {shape}")
        if len(rows_sorted) > 12:
            print(f"  ... +{len(rows_sorted) - 12} more")

    print("\n--- SANITY ---")
    missing = all_param_ids - seen_param_ids
    extra = seen_param_ids - all_param_ids
    print(f"model params not in any group (LEAK): {len(missing)}")
    for pid in list(missing)[:10]:
        name = id_to_name.get(pid, "<unknown>")
        print(f"  MISSING: {name}")
    print(f"group params not in model (ORPHAN): {len(extra)}")
    print(f"duplicated param ids across groups: {len(dup_ids)}")

    # Also: compare against what AdamW would use.
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    print(f"\nAdamW built with {len(opt.param_groups)} groups.")
    lr_histogram = {}
    for g in opt.param_groups:
        lr_histogram.setdefault(g["lr"], 0)
        lr_histogram[g["lr"]] += sum(p.numel() for p in g["params"])
    for lr in sorted(lr_histogram.keys()):
        print(f"  LR {lr:.6g}: {lr_histogram[lr]:,} params")


if __name__ == "__main__":
    main()
