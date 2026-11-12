"""Create and read channel-selection records (src/projected_compression/channel_selection.py).

Show a record:

    python scripts/channel_selection.py show keep70_hf.json

Turn an importances file (weight_importances.py output) into a record that keeps the
same channels pruning.prune() would:

    python scripts/channel_selection.py from-importances \\
        <ckpt>/minitron_dimensions_importances.pt \\
        --target-dmodel 960 --target-dff 3840 \\
        --source-model meta-llama/Llama-3.2-1B --out minitron_30p.json
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.projected_compression.channel_selection import (  # noqa: E402
    ChannelSelection,
    describe,
    load_selection,
    save_selection,
)
from src.projected_compression.channel_selectors import topk_keep  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("show", help="print a record's summary")
    show.add_argument("path")

    conv = sub.add_parser("from-importances", help="importances .pt -> record JSON")
    conv.add_argument("importances_path")
    conv.add_argument("--target-dmodel", type=int, required=True)
    conv.add_argument("--target-dff", type=int, required=True)
    conv.add_argument("--source-model", required=True)
    conv.add_argument("--out", required=True)
    conv.add_argument("--notes", default="")

    args = parser.parse_args(argv)
    if args.command == "show":
        print(describe(load_selection(args.path)))
        return

    # No model needed: the source dims come from the importances tensors.
    import torch

    imp = torch.load(args.importances_path, map_location="cpu")
    dff_imp = imp["dff_importances"]
    dmodel_keep, dff_keep = topk_keep(imp, args.target_dmodel, args.target_dff)
    selection = ChannelSelection(
        source_model=args.source_model,
        dmodel=len(imp["dmodel_importances"]),
        dff=len(dff_imp[0]),
        n_blocks=len(dff_imp),
        method="importance_topk",
        method_params={
            "importances_path": args.importances_path,
            "target_dmodel": args.target_dmodel,
            "target_dff": args.target_dff,
        },
        dmodel_keep=dmodel_keep,
        dff_keep=dff_keep,
        notes=args.notes,
    )
    save_selection(selection, args.out)
    print(f"wrote {args.out}\n{describe(selection)}")


if __name__ == "__main__":
    main()
