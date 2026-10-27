"""Generate the SliceGPT smoke-test configs from the *current* configs in the repo.

The point of regenerating rather than committing the composed configs: a composed config
is a snapshot. It bakes in whatever cluster paths, model dims and defaults were active
when it was written, so a committed one quietly goes stale the moment anything it
composed from changes -- which defeats the purpose of a suite you run *to check that
things still work after a change*.

Everything that varies between users and clusters comes from arguments or the cluster
config being composed, never from a constant in here:

  --work-dir  where the smoke run reads and writes its own artefacts
  --out-dir   where the generated configs are written
  --source-model / --importances
              the two assets the suite needs; without them the composed configs inherit
              whatever paths the source config names, which are one researcher's
  --override  any other Hydra override, repeatable -- e.g. dataset paths on another
              cluster

The cluster itself comes from the composed config's own defaults list. It cannot be
overridden from the command line: those entries are relative (`../_cluster@_here_`), and
Hydra's override lexer rejects both `..` and `@_here_`. To run on a different cluster,
either add a cluster variant of the source config or retarget the paths explicitly:

    python smoke/generate.py --work-dir /scratch/$USER/smoke \
        --override cluster_switch.train_path_fineweb=/data/fineweb/train

Usage:

    python smoke/generate.py --work-dir /scratch/$USER/slicegpt_smoke
    bash smoke/run_smoke.sh s1_invariance

Scenarios:

  s1_invariance   slicing at d = D with no dff pruning -> loss must be unchanged.
                  The strongest single check that fusion + rotation are correct.
  s2_slice_r50    real hybrid r50 slicing, small calibration, saves a checkpoint that
                  s4/s5 then load.
  s4_rft          30-step recovery training from the s2 checkpoint (plain).
  s4_rft_distil   the same with distillation.
  s5_lr<n>        100-step LR sweep from the s2 checkpoint, learning_rate = n.
"""

import argparse
import os
import sys

sys.path.insert(0, os.getcwd())

import resolver  # noqa: F401  (registers the repo's custom OmegaConf resolvers)
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from grid_generator.generate_configs import create_grid_config

STDOUT = ["infrastructure.metric_logger.type=stdout"]
CHECKPOINT = "llama1b_slicegpt_hyb_r50_smoke"


def build(config_dir, name, overrides, out_path, edit=None):
    """Compose, expand the grid, apply `edit`, and write one concrete config."""
    with initialize_config_dir(
        version_base=None, config_dir=os.path.abspath(config_dir)
    ):
        cfg = compose(config_name=name, overrides=list(overrides))
    config = create_grid_config(cfg)[0][0]
    if edit:
        edit(config)
    config["_run_"] = True
    OmegaConf.save(OmegaConf.create(config), out_path)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--work-dir",
        default=os.environ.get("SMOKE_WORK_DIR", "./smoke_work"),
        help="where the smoke run keeps its checkpoints (default $SMOKE_WORK_DIR or ./smoke_work)",
    )
    p.add_argument(
        "--out-dir", default="smoke/generated", help="where to write the configs"
    )
    p.add_argument(
        "--lrs", default="11,12,13,14,15", help="learning rates for the s5 sweep"
    )
    p.add_argument(
        "--source-model",
        default=os.environ.get("SMOKE_SOURCE_MODEL"),
        help="nano checkpoint to slice; overrides slicegpt.source.path (default: whatever the config names)",
    )
    p.add_argument(
        "--importances",
        default=os.environ.get("SMOKE_IMPORTANCES"),
        help="dimension-importances .pt; overrides slicegpt.dimensions_importances_path",
    )
    p.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="extra",
        help="extra Hydra override, repeatable (e.g. a data path on another cluster)",
    )
    args = p.parse_args()

    work = os.path.abspath(args.work_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    extra = list(args.extra)
    # The asset overrides only apply to the slicing scenarios: the recovery-training
    # configs have no slicegpt.source.* keys, and Hydra errors on overriding a key a
    # config does not define.
    assets = []
    if args.source_model:
        assets.append(f"slicegpt.source.path={args.source_model}")
    if args.importances:
        assets.append(f"slicegpt.dimensions_importances_path={args.importances}")
    small = ["slicegpt.calibration.n_samples=64", "slicegpt.eval.n_samples=16", *STDOUT]

    # s1: d = D, no dff pruning -> the model must be unchanged, so the loss must be too.
    build(
        "configs/slicegpt",
        "llama_1B_slice_hyb_entropy",
        small
        + extra
        + assets
        + [
            "slicegpt.target_d=2048",
            "slicegpt.target_dff=8192",
            "slicegpt.orientation=pca",
            "slicegpt.save_path=''",
        ],
        f"{args.out_dir}/s1_invariance.yaml",
    )

    # s2: real r50 slicing, saves the checkpoint s4/s5 consume.
    build(
        "configs/slicegpt",
        "llama_1B_slice_hyb_entropy",
        small
        + extra
        + assets
        + [
            "slicegpt_ratio=r50",
            f"slicegpt.save_path={work}",
            f"slicegpt.save_name={CHECKPOINT}",
        ],
        f"{args.out_dir}/s2_slice_r50.yaml",
    )

    def rft(steps, lr=None):
        def edit(config):
            config["trainer"]["n_steps"] = steps
            config["trainer"]["checkpoint"]["save"]["path"] = f"{work}/rft_ckpt"
            if lr is not None:
                config["trainer"]["learning_rate"] = lr

        return edit

    load = [f"trainer.checkpoint.load.path={work}/{CHECKPOINT}.pt"]

    for name, out in (
        ("llama_1B_slicegpt_hyb_entropy", "s4_rft"),
        ("llama_1B_slicegpt_hyb_distillation_entropy", "s4_rft_distil"),
    ):
        build(
            "configs/pc_project_mas",
            name,
            ["slicegpt_ratio=r50", *load, *STDOUT, *extra],
            f"{args.out_dir}/{out}.yaml",
            rft(30),
        )

    for n in [int(x) for x in args.lrs.split(",") if x.strip()]:
        build(
            "configs/pc_project_mas",
            "llama_1B_slicegpt_hyb_entropy",
            ["slicegpt_ratio=r50", *load, *STDOUT, *extra],
            f"{args.out_dir}/s5_lr{n}.yaml",
            rft(100, lr=n),
        )

    print(f"\n  work dir: {work}")
    print(f"  run with: bash smoke/run_smoke.sh <scenario>")


if __name__ == "__main__":
    main()
