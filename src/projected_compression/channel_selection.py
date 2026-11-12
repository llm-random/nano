"""A record of which channels a structured pruning keeps, and where they came from.

`ChannelSelection` is plain data: the kept dmodel channels and the kept FFN (dff)
channels of every block, the model they index into, when the record was made, the
method that chose them (a name plus its parameters) and free-form notes. It knows
nothing about models or torch; the functions that *produce* a selection from a model
live in channel_selectors.py.

    sel = select_by_importances(model, "imp.pt", 960, 3840, source_model="...")
    save_selection(sel, "minitron_30p.json")
    print(describe(load_selection("minitron_30p.json")))

Apply a saved selection at training start through `apply_functions`::

    apply_functions:
      - _target_: src.projected_compression.pruning.prune_from_selection
        _partial_: true
        selection_path: /path/to/keep70_hf.json

The model config keeps the source dims; the pruning shrinks the loaded weights. For
`trainer.checkpoint.load.type: nano`, main.py runs apply_functions *before* loading the
weights, so pruning on load works with the huggingface / llm-random / sliced_nano /
finalized_pc load types, not nano.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import Optional

FORMAT_VERSION = 1


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _check_indices(indices, dim, name):
    if len(indices) == 0:
        raise ValueError(f"{name}: keeps no channels")
    if any(not isinstance(i, int) or isinstance(i, bool) for i in indices):
        raise ValueError(f"{name}: indices must be ints")
    if any(a >= b for a, b in zip(indices, indices[1:])):
        raise ValueError(
            f"{name}: indices must be strictly increasing (sorted, unique)"
        )
    if indices[0] < 0 or indices[-1] >= dim:
        raise ValueError(f"{name}: indices must lie in [0, {dim})")


@dataclass
class ChannelSelection:
    """Channels kept by a structured pruning of one model, plus provenance.

    `dmodel_keep` / `dff_keep` set to None mean "keep every channel on that axis".
    """

    source_model: str
    dmodel: int
    dff: int
    n_blocks: int
    method: str
    method_params: dict = field(default_factory=dict)
    dmodel_keep: Optional[list[int]] = None
    dff_keep: Optional[list[list[int]]] = None
    created: str = field(default_factory=_now)
    notes: str = ""
    format_version: int = FORMAT_VERSION

    def __post_init__(self):
        if self.format_version != FORMAT_VERSION:
            raise ValueError(
                f"unsupported format_version {self.format_version}, "
                f"expected {FORMAT_VERSION}"
            )
        if self.dmodel_keep is not None:
            _check_indices(self.dmodel_keep, self.dmodel, "dmodel_keep")
        if self.dff_keep is not None:
            if len(self.dff_keep) != self.n_blocks:
                raise ValueError(
                    f"dff_keep has {len(self.dff_keep)} blocks, model has {self.n_blocks}"
                )
            for i, block in enumerate(self.dff_keep):
                _check_indices(block, self.dff, f"dff_keep[{i}]")
            # The model config has a single dff, so a pruned nano checkpoint must be
            # uniform across blocks to be loadable again.
            counts = {len(block) for block in self.dff_keep}
            if len(counts) != 1:
                raise ValueError(
                    "every block must keep the same number of dff channels, "
                    f"got {sorted(counts)}"
                )

    @property
    def target_dmodel(self) -> int:
        return self.dmodel if self.dmodel_keep is None else len(self.dmodel_keep)

    @property
    def target_dff(self) -> int:
        return self.dff if self.dff_keep is None else len(self.dff_keep[0])


def save_selection(selection: ChannelSelection, path: str):
    """Write JSON with one line per field / per block, so it diffs and greps well."""
    lines = []
    for key, value in asdict(selection).items():
        if key == "dff_keep" and value is not None:
            blocks = ",\n".join(f"    {json.dumps(b)}" for b in value)
            lines.append(f'  "{key}": [\n{blocks}\n  ]')
        else:
            lines.append(f"  {json.dumps(key)}: {json.dumps(value)}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("{\n" + ",\n".join(lines) + "\n}\n")


def load_selection(path: str) -> ChannelSelection:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    unknown = set(d) - {f.name for f in fields(ChannelSelection)}
    if unknown:
        raise ValueError(f"unknown fields in channel selection: {sorted(unknown)}")
    return ChannelSelection(**d)


def describe(selection: ChannelSelection) -> str:
    s = selection
    lines = [
        f"source model : {s.source_model}",
        f"created      : {s.created}",
        f"method       : {s.method} {json.dumps(s.method_params)}",
        f"dmodel       : {s.target_dmodel} / {s.dmodel} kept"
        f" ({s.target_dmodel / s.dmodel:.1%})",
        f"dff          : {s.target_dff} / {s.dff} kept per block"
        f" ({s.target_dff / s.dff:.1%}), {s.n_blocks} blocks",
    ]
    if s.notes:
        lines.append(f"notes        : {s.notes}")
    return "\n".join(lines)
