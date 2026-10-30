"""Expand a Hydra config the way run_exp.py does, and write it out as a plain YAML file.

Why this exists: several configs use `^`-prefixed grid keys (e.g. `^n_steps: [1024, 2048]`).
`run_exp.py` expands those into one concrete config per grid point before anything is
instantiated. Running `main.py` against such a config directly does not, so the raw key
reaches `instantiate(cfg.trainer)` and dies with:

    TypeError: Trainer.__init__() got an unexpected keyword argument '^n_steps'

Hydra cannot remove it either -- `~trainer.^n_steps` fails in the override lexer, because
`^` is not a legal character there. So the only way to run a grid config locally (a smoke
test, a single-GPU repro) is to expand it first, which is what this does.

Usage:

    python scripts/expand_config.py --config-path configs \\
        --config-name tiny_remote -o /tmp/tiny.yaml \\
        trainer.n_steps=6 infrastructure.metric_logger.type=stdout

    python main.py --config-path=/tmp --config-name=tiny

`--grid-index` picks which grid point to emit when the config expands to several
(default 0). `--list` prints how many there are and exits.

Note the ordering trap: a Hydra override of a key that is *also* a grid key is
overwritten by the expansion, because the grid sets that key afterwards. Passing
`trainer.n_steps=6` to a config with `^n_steps` silently yields the grid's value.
Use `--set trainer.n_steps=6`, which is applied *after* expansion.
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.getcwd())

import resolver  # noqa: F401  (registers the repo's custom OmegaConf resolvers)
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from grid_generator.generate_configs import create_grid_config


def apply_set(config, assignments):
    """Apply `dotted.key=value` assignments to the expanded config, after grid expansion."""
    for item in assignments:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise SystemExit(f"--set {key}: no such section {part!r}")
            node = node[part]
        node[parts[-1]] = yaml.safe_load(raw)
    return config


def expand(config_path, config_name, overrides, grid_index=0):
    """Compose `config_name` and return one grid point as a plain dict."""
    with initialize_config_dir(
        version_base=None, config_dir=os.path.abspath(config_path)
    ):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    grid = create_grid_config(cfg)
    if not 0 <= grid_index < len(grid):
        raise SystemExit(
            f"--grid-index {grid_index} out of range: this config expands to {len(grid)} point(s)"
        )
    config, grid_overrides = grid[grid_index]
    config["overrides"] = grid_overrides
    config["_run_"] = True
    return config, len(grid)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config-path", required=True, help="directory holding the config")
    p.add_argument(
        "--config-name", required=True, help="config file name, without .yaml"
    )
    p.add_argument("-o", "--out", help="where to write the expanded YAML")
    p.add_argument(
        "--grid-index", type=int, default=0, help="which grid point (default 0)"
    )
    p.add_argument("--list", action="store_true", help="report the grid size and exit")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="set_after",
        help="assignment applied AFTER grid expansion; use this for keys the grid also sets",
    )
    p.add_argument(
        "overrides", nargs="*", help="Hydra overrides, e.g. trainer.n_steps=6"
    )
    args = p.parse_args()

    config, n = expand(
        args.config_path, args.config_name, args.overrides, args.grid_index
    )

    if args.list:
        print(f"{args.config_name}: {n} grid point(s)")
        return
    if not args.out:
        raise SystemExit("--out is required unless --list is given")

    config = apply_set(config, args.set_after)
    OmegaConf.save(OmegaConf.create(config), args.out)
    leftover = [k for k in config.get("trainer", {}) if k.startswith("^")]
    print(f"wrote {args.out} (grid point {args.grid_index} of {n})")
    if leftover:
        print(f"  warning: unexpanded grid keys remain: {leftover}")


if __name__ == "__main__":
    main()
