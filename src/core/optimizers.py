"""Parameter roles and the one optimizer builder main.py calls (line 2 of the plan)."""

import logging
import os

import torch
import torch.nn as nn
from hydra.utils import instantiate

from src.core.model import TransformerHead

logger = logging.getLogger(__name__)

# Order of the param groups; the role table in the plan.
ROLES = ("vector", "embedding", "head", "hidden")


def param_roles(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    """Each parameter's role from its shape and owning module; first rule wins."""
    embedding = {
        id(p)
        for m in model.modules()
        if isinstance(m, nn.Embedding)
        for p in m.parameters(recurse=False)
    }
    head = {
        id(p)
        for m in model.modules()
        if isinstance(m, TransformerHead)
        for p in m.parameters()
    }
    roles = {role: [] for role in ROLES}
    for p in model.parameters():
        if p.ndim < 2:
            roles["vector"].append(p)
        elif id(p) in embedding:
            roles["embedding"].append(p)
        elif id(p) in head:
            roles["head"].append(p)
        else:
            roles["hidden"].append(p)
    return roles


def log_param_groups(model, groups):
    """Each group's LR, decay and parameters, once, on rank 0."""
    if os.environ.get("RANK", "0") != "0":
        return
    names = {id(p): name for name, p in model.named_parameters()}
    for group in groups:
        n_params = sum(p.numel() for p in group["params"])
        logger.info(
            f"param group {group['role']}: lr {group['lr']:.4g}, weight_decay "
            f"{group['weight_decay']:g}, {len(group['params'])} tensors, {n_params:,} params"
        )
        for p in group["params"]:
            logger.info(f"  {names[id(p)]} {tuple(p.shape)}")


def build_optimizer_and_scheduler(cfg, model, learning_rate):
    roles = cfg.optimizer.roles
    base_dmodel = cfg.optimizer.base_dmodel
    if roles is None:
        assert base_dmodel is None, "simpleP (optimizer.base_dmodel) needs param groups"
        params = model.parameters()
    else:
        # simpleP (muP's LR rule): hidden and head LR / m, m = dmodel / base_dmodel.
        scale = 1 if base_dmodel is None else base_dmodel / cfg.common.dmodel
        params = [
            {
                "params": ps,
                "role": role,
                "lr": learning_rate * (scale if role in ("hidden", "head") else 1),
                "weight_decay": roles[role].weight_decay,
            }
            for role, ps in param_roles(model).items()
            if ps
        ]
        log_param_groups(model, params)
    optimizer = torch.optim.AdamW(
        params,
        lr=learning_rate,
        weight_decay=cfg.trainer.weight_decay,
        # The fused kernel exists only for CUDA tensors; CPU runs keep the
        # default path, which also keeps the seeded CPU gates bit-identical.
        fused=torch.cuda.is_available(),
    )
    scheduler = instantiate(cfg.trainer.scheduler)(
        optimizer=optimizer, n_steps=cfg.trainer.n_steps
    )
    return optimizer, scheduler
