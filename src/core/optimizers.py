"""Parameter roles and the one optimizer builder main.py calls (line 2 of the plan)."""

import torch
import torch.nn as nn
from hydra.utils import instantiate

from src.core.model import TransformerHead

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


def build_optimizer_and_scheduler(cfg, model, learning_rate):
    roles = cfg.optimizer.roles
    if roles is None:
        params = model.parameters()
    else:
        params = [
            {
                "params": ps,
                "role": role,
                "lr": learning_rate * roles[role].lr_multiplier,
                "weight_decay": roles[role].weight_decay,
            }
            for role, ps in param_roles(model).items()
            if ps
        ]
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
