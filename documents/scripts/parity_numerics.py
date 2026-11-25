#!/usr/bin/env python
"""The common evaluator: the final CE of a checkpoint on the full held-out set.

Ported from `parity_audit`'s documents/scripts/parity_numerics.py for
documents/notes/ame_bank_gates_plan.md. Only the scoring is here -- its `score`
check, the `dump_data` that produced the held-out set, and the helpers both
need -- with one change: a model is composed from a run config (`--config`)
rather than from the parity configs, so it builds whatever the run trained,
`bank_gate` included. The production references (doi36p8k: 3.125227 under
`ame_ref32`) were scored by the original, on the same held-out tensor.

Not every run config works. An FSDP2 run (Stage 2c, standard/ame_h20) does not:
`ame_ref32` leaves `trainer.distributed` as the run set it, so the build reaches
setup_distributed_training with no process group, and `load_checkpoint` reads
only the single-file checkpoint and the rank-0 sidecar, not a distributed
checkpoint with one sidecar per rank. A config with a null learning rate needs
`--override trainer.learning_rate=<n>`. A config that gives its length as
`trainer.n_tokens_b` fails, because `build` never calls `resolve_n_steps`
(documents/notes/review_upgrades.md, U5).

Run from the root of a checkout, so that every model is built by that commit's
code (`main.get_model_optimizer_scheduler`, the same Hydra configs):

    python documents/scripts/parity_numerics.py dump_data --config CFG --out DIR
    python documents/scripts/parity_numerics.py score --config CFG --ckpt FOLDER \
        --data FILE --out DIR --tag NAME [--evaluators ame_ref32,ame_prod]

CHECK is one of:

    dump_data   the whole held-out set, exactly as the Trainer's eval dataloader
                yields it, plus its fingerprint; with --compare FILE, whether it
                equals another dump's held-out tensor
    score       the precise CE (fp32 loss on each evaluator's logits, summed in
                fp64) of one checkpoint on the held-out set, under each
                evaluator variant: the common evaluator (`ame_ref32`: fp32,
                eager, reference attention, banks from the fp32 masters) and,
                for comparison, the run's own production path (`ame_prod`)

`--smoke` shrinks the model (h2, vocab 512, short sequences) and uses random
data, so the logic runs on a laptop CPU; smoke numbers mean nothing.
"""

import argparse
import contextlib
import gc
import json
import os
import sys
import time
import types

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

MP_NULL = "trainer.mem_parallel_size=null"
NO_COMPILE = "++compile_blocks_full=false"
REFERENCE = "++common.attention.ame.backend=reference"
FP32 = "trainer.autocast_bf16=false"

# name: (overrides on the run's config, autocast, micro-batch)
VARIANTS = {
    # The common evaluator: fp32 everywhere, reference attention, eager, the banks
    # as fp32 nn.Embedding weights loaded from the sidecar's masters.
    "ame_ref32": ([MP_NULL, NO_COMPILE, REFERENCE, FP32], False, 8),
    # The run as trained: Tier 2 banks (bf16 compute copy), compiled blocks, flex.
    "ame_prod": ([], True, 32),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    log(f"wrote {path}")


# ---- configs and models ------------------------------------------------------


# Extra Hydra overrides from --override, applied to every compose (e.g. the arm
# of a smoke config: common.attention.ame.bank_gate=confidence).
EXTRA_OVERRIDES = []


def compose(config, overrides=(), smoke=False):
    """A run config (a path to its yaml), composed the way run_exp.py composes
    it: `--config-path` its directory, `--config-name` its file name."""
    from hydra import compose as hydra_compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    import resolver  # noqa: F401  registers ${eval:...}
    import src.definitions  # noqa: F401  registers base_schema

    path = os.path.abspath(config)
    extra = ["infrastructure.metric_logger.name=parity_numerics"]
    if smoke:
        extra += ["common.h=2", "common.vocab_size=512", "common.sequence_length=129"]
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.dirname(path), version_base=None):
        name = os.path.splitext(os.path.basename(path))[0]
        return hydra_compose(
            config_name=name, overrides=extra + EXTRA_OVERRIDES + list(overrides)
        )


class _Quiet(types.SimpleNamespace):
    heavy_metrics_calculation_interval = 100


def quiet_logger():
    """A RecorderLogger at step 1 of interval 100: Residual's metric hook is
    called (as in the runs, where it broke the graph twice per block) but
    computes nothing."""
    from src.core.metric_loggers import RecorderLogger

    logger = RecorderLogger(_Quiet())
    logger.set_step(1)
    return logger


def build(variant, config, device, smoke=False):
    """Model, optimizer, scheduler and bank controller, built the way main.py
    builds them (seed, instantiate, controller, compile, AdamW groups)."""
    from hydra.utils import instantiate

    from main import get_model_optimizer_scheduler
    from src.core.memory_parallel import get_controller
    from src.core.model import Residual
    from src.core.utils import solve_config_lr

    overrides, autocast, micro = VARIANTS[variant]
    cfg = compose(config, overrides, smoke)
    if smoke and device.type == "cpu":
        # inductor miscompiles the bank scatter on CPU and compiled flex cannot
        # lower there: smoke runs never compile.
        cfg.compile_blocks_full = False
    torch._dynamo.reset()
    torch.manual_seed(cfg.trainer.train_dataloader.dataset.seed)
    model = instantiate(cfg.model, _convert_="all").to(device)
    logger = quiet_logger()
    for module in model.modules():
        if isinstance(module, Residual):
            module.set_metric_logger(logger)
    lr, _ = solve_config_lr(float(cfg.trainer.learning_rate))
    model, optimizer, scheduler = get_model_optimizer_scheduler(cfg, model, lr)
    banks = get_controller(model)
    if banks is not None:
        banks.materialize()
    return types.SimpleNamespace(
        variant=variant,
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        banks=banks,
        autocast=autocast and device.type == "cuda",
        micro=micro,
        logger=logger,
    )


def bank_slots(model):
    """[(module, attr, fully-qualified weight name)] in the controller's order."""
    from src.core.memory_parallel import ShardedBank

    slots = []
    for name, module in model.named_modules():
        for attr in ("q_bank", "k_bank"):
            bank = getattr(module, attr, None)
            if isinstance(bank, (nn.Embedding, ShardedBank)):
                slots.append((module, attr, f"{name}.{attr}.weight"))
    return slots


def load_checkpoint(built, folder):
    """A run's final checkpoint into any variant.

    Tier 2 (mp 1) reads the model state dict and the bank sidecar the way
    `load_checkpoint_from_file` does. The ordinary path (mp null) has the banks
    as nn.Embedding weights the state dict does not carry, so they come from
    the sidecar's fp32 masters, in the controller's bank order.
    """
    t0 = time.time()
    state = torch.load(
        os.path.join(folder, "__model_checkpoint_filename.pt"),
        map_location="cpu",
        weights_only=False,
    )
    model = built.model
    sidecar_path = os.path.join(folder, "memory_parallel_rank0.pt")
    if not os.path.exists(sidecar_path):
        # A checkpoint of an ordinary-path run (mem_parallel_size null): the
        # banks are nn.Embedding weights inside the model state dict.
        bank_keys = {
            k for k in state["model"] if k.endswith(("q_bank.weight", "k_bank.weight"))
        }
        assert bank_keys, f"{folder}: no sidecar and no bank weights in the state dict"
        if built.banks is None:
            model.load_state_dict(state["model"])
        else:
            rest = {k: v for k, v in state["model"].items() if k not in bank_keys}
            missing, unexpected = model.load_state_dict(rest, strict=False)
            assert not missing and not unexpected, (missing[:4], unexpected[:4])
            slot = {id(getattr(m, a)): n for m, a, n in bank_slots(model)}
            with torch.no_grad():
                for bank in built.banks.banks:
                    bank.master.copy_(
                        state["model"][slot[id(bank)]].to(bank.master.device)
                    )
                    bank.weight.copy_(bank.master.to(bank.weight.dtype))
    elif built.banks is not None:
        model.load_state_dict(state["model"])
        side = torch.load(sidecar_path, map_location="cpu", weights_only=False)
        # The bank AdamW's moments are only needed to resume; scoring gets the
        # masters alone.
        built.banks.load_state_dict({**side, "optimizer": None})
    else:
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        slots = bank_slots(model)
        expected = {name for _, _, name in slots}
        assert not unexpected, unexpected
        assert set(missing) == expected, (sorted(set(missing) ^ expected))[:6]
        side = torch.load(sidecar_path, map_location="cpu", weights_only=False)
        assert len(side["masters"]) == len(slots)
        with torch.no_grad():
            for (module, attr, _), master in zip(slots, side["masters"]):
                getattr(module, attr).weight.copy_(master)
    log(f"loaded {folder} ({time.time() - t0:.0f} s)")
    return state


def free():
    """Release GPU memory. Callers must drop their own references first
    (`x = None`). Compiled graphs can hold a model's tensors, so dynamo's caches
    go too."""
    torch._dynamo.reset()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---- data ------------------------------------------------------------------------


def dump_data(args):
    """The full held-out set, exactly as the Trainer's eval dataloader yields it."""
    from hydra.utils import instantiate

    from src.core.utils import create_batch_fingerprint

    cfg = compose(args.config)
    held = instantiate(cfg.trainer.eval_dataloader)
    held_batches = list(held)
    out = {
        "heldout": torch.cat(held_batches).to(torch.int32),
        "heldout_fingerprint": create_batch_fingerprint(held_batches[0]),
        "heldout_batches": len(held_batches),
    }
    path = os.path.join(args.out, "heldout.pt")
    torch.save(out, path)
    log(
        f"heldout {tuple(out['heldout'].shape)} in {len(held_batches)} batches -> {path}"
    )
    summary = {k: v for k, v in out.items() if not torch.is_tensor(v)}
    if args.compare:
        other = torch.load(args.compare, map_location="cpu", weights_only=False)
        summary["compare"] = args.compare
        summary["identical_to_compare"] = bool(
            torch.equal(other["heldout"].to(torch.int32), out["heldout"])
        )
        log(
            f"held-out set identical to {args.compare}: {summary['identical_to_compare']}"
        )
    save_json(os.path.join(args.out, "heldout.json"), summary)


def load_heldout(args, device):
    if args.smoke:
        g = torch.Generator().manual_seed(0)
        return torch.randint(0, 512, (64, 129), generator=g).to(device)
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    return data["heldout"].to(torch.int64).to(device)


# ---- the precise CE ----------------------------------------------------------------


def autocast_ctx(on):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16) if on else contextlib.nullcontext()
    )


@torch.no_grad()
def precise_ce(model, held, autocast, micro):
    """Mean next-token CE over `held`: each evaluator's logits, upcast to fp32 for
    the loss, accumulated per token in fp64."""
    total = torch.zeros((), dtype=torch.float64, device=held.device)
    count = 0
    for start in range(0, held.shape[0], micro):
        chunk = held[start : start + micro]
        inp, tgt = chunk[:, :-1], chunk[:, 1:]
        with autocast_ctx(autocast):
            logits = model(inp)
        losses = F.cross_entropy(
            logits.float().flatten(0, -2), tgt.reshape(-1).long(), reduction="none"
        )
        total += losses.double().sum()
        count += losses.numel()
    return float(total / count)


def run_score(args, device):
    """Precise CE of one checkpoint on the full held-out set, under each
    evaluator variant."""
    held = load_heldout(args, device)
    report = {"config": args.config, "ckpt": args.ckpt, "tag": args.tag}
    for variant in args.evaluators.split(","):
        t0 = time.time()
        built = build(variant, args.config, device, args.smoke)
        report.setdefault(
            "bank_gate",
            str(built.cfg.common.attention.get("ame", {}).get("bank_gate", "none")),
        )
        if not args.smoke:
            load_checkpoint(built, args.ckpt)
        built.model.eval()
        report[variant] = precise_ce(
            built.model, held, autocast=built.autocast, micro=built.micro
        )
        log(f"{args.tag} {variant}: {report[variant]:.6f} ({time.time() - t0:.0f} s)")
        built = None
        free()
    save_json(os.path.join(args.out, f"score_{args.tag}.json"), report)


# ---- main -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("check", choices=["dump_data", "score"])
    parser.add_argument("--config", required=True, help="the run's config yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", default=None, help="score: the held-out dump")
    parser.add_argument("--ckpt", default=None, help="score: the checkpoint folder")
    parser.add_argument("--evaluators", default="ame_ref32", help="score: variants")
    parser.add_argument("--tag", default="run", help="score: a name for the output")
    parser.add_argument("--compare", default=None, help="dump_data: another dump")
    parser.add_argument(
        "--override", action="append", default=[], help="an extra Hydra override"
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    EXTRA_OVERRIDES.extend(args.override)
    if args.check == "score" and not args.smoke:
        assert args.ckpt and args.data, "score needs --ckpt and --data"

    import main as main_module

    main_module.setup_enviroment()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import subprocess

    commit_file = os.path.join(REPO, "EVAL_COMMIT")
    if os.path.exists(commit_file):
        commit = open(commit_file).read().strip()
    else:
        commit = subprocess.run(
            ["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    meta = {
        "check": args.check,
        "commit": commit,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
    }
    log(json.dumps(meta))
    assert not torch.backends.cuda.matmul.allow_tf32, "fp32 references need TF32 off"
    save_json(os.path.join(args.out, f"meta_{args.check}_{args.tag}.json"), meta)
    torch._dynamo.config.cache_size_limit = 64
    if args.check == "dump_data":
        dump_data(args)
    else:
        run_score(args, device)
    log("done")


if __name__ == "__main__":
    main()
