"""Diagnostic: which parameter group makes a SliceGPT-sliced model fragile to Adam's first step?

Adam's first update moves every element by ~ +-lr (sign of the gradient). We emulate it with a random +-eps
perturbation applied to ONE parameter group at a time and measure the eval loss. Also reports residual-stream
magnitudes (massive activations) per block for the sliced model.

Usage (1 GPU, from repo root): python scripts/slicegpt_perturb_diag.py <sliced.pt> <ratio> [eps]
"""

import json
import os
import sys

for k, v in {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"}.items():
    os.environ.setdefault(k, v)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402

import resolver  # noqa: E402,F401
from src.projected_compression.slicegpt.run_slicing import (
    evaluate_loss,
    sample_token_batches,
)  # noqa: E402

ckpt, ratio = sys.argv[1], sys.argv[2]
eps = (
    float(sys.argv[3]) if len(sys.argv) > 3 else 2.44140625e-05
)  # first-step lr of the S4 smoke (2^-12 * 0.1)
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with initialize_config_dir(
    version_base=None, config_dir=os.path.join(root, "configs/pc_project_mas")
):
    cfg = compose(
        config_name="llama_1B_slicegpt_hyb_entropy",
        overrides=[f"slicegpt_ratio={ratio}"],
    )
with initialize_config_dir(
    version_base=None, config_dir=os.path.join(root, "configs/slicegpt")
):
    scfg = compose(config_name="llama_1B_slice_hyb_entropy")

device = torch.device("cuda")
state = torch.load(ckpt, map_location="cpu")
batches = sample_token_batches(scfg.trainer.train_dataloader.dataset, 1024, 7, 16, 8)

GROUPS = {
    "none": lambda n: False,
    "shortcut_Q": lambda n: n.endswith("shortcut_Q"),
    "embedding": lambda n: n == "embedding.weight",
    "head.linear": lambda n: n == "head.linear.weight",
    "norms": lambda n: n.endswith("norm.weight"),
    "attn q/k/v": lambda n: any(f"{p}_proj.weight" in n for p in "qkv"),
    "attn o_proj": lambda n: "o_proj.weight" in n,
    "mlp in (up/gate)": lambda n: "ff_pre_act.weight" in n or "gate.weight" in n,
    "mlp out (down)": lambda n: "ff_post_act.weight" in n,
    "all": lambda n: True,
}

results = {}
g = torch.Generator().manual_seed(0)
for name, pick in GROUPS.items():
    model = instantiate(cfg.model, _convert_="all")
    sd = {k: v.clone() for k, v in state.items()}
    for k, v in sd.items():
        if pick(k) and v.is_floating_point():
            sd[k] = v + eps * (
                torch.randint(0, 2, v.shape, generator=g).float() * 2 - 1
            )
    model.load_state_dict(sd, strict=True)
    model = model.to(device=device, dtype=torch.float32)
    results[name] = evaluate_loss(model, batches, device)
    print(f"perturb {name:18s} -> loss {results[name]:.4f}", flush=True)

# residual-stream magnitudes of the unperturbed sliced model
model = instantiate(cfg.model, _convert_="all")
model.load_state_dict(state, strict=True)
model = model.to(device=device, dtype=torch.float32).eval()
stats = []
with torch.no_grad():
    x = model.embedding(batches[0][:, :-1].to(device))
    for i, block in enumerate(model.encoder.blocks):
        x = block(x)
        rms = x.pow(2).mean(-1).sqrt()
        stats.append(
            {
                "block": i,
                "rms_median": rms.median().item(),
                "rms_max": rms.max().item(),
                "abs_max": x.abs().max().item(),
            }
        )
for s in stats:
    print("residual", json.dumps(s))
print("DIAG_RESULT", json.dumps({"eps": eps, "loss": results}))
