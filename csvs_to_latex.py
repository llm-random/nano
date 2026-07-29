#!/usr/bin/env python
"""Turn one or more scrape_eval_results.py CSVs into a LaTeX table.

Rows are the methods (PC, PCD, HP, HPD), columns are compression ratios grouped
under a \\multicolumn header per input CSV (one model per CSV). Ratio tags are
relabelled as the fraction compressed away: 50p -> 50\\%, 30p -> 70\\%, 10p -> 90\\%.

Examples:
    pixi run python csvs_to_latex.py olmo_eval.csv smollm2_eval.csv --out table.tex
    pixi run python csvs_to_latex.py olmo_eval.csv --labels OLMo-2-1B --metric hellaswag
"""

import argparse
import csv
import logging
import os
import re
import sys
from typing import Optional

logger = logging.getLogger("csvs_to_latex")

METHOD_ORDER = ["pc", "pcd", "hp", "hpd"]
METHOD_LABELS = {"pc": "PC", "pcd": "PCD", "hp": "HP", "hpd": "HPD"}
RATIO_RE = re.compile(r"^(\d+)p$")

MISSING = "--"


def ratio_label(ratio: str) -> str:
    """Ratio tag -> fraction compressed away, e.g. 30p (30% kept) -> 70\\%."""
    match = RATIO_RE.match(ratio)
    if not match:
        return ratio
    return f"{100 - int(match.group(1))}\\%"


def model_label(rows: list[dict], path: str) -> str:
    """Name shared by every model in the CSV: the name tokens before the method token."""
    prefixes = set()
    for row in rows:
        tokens = row["model"].split("_")
        method = row.get("method", "")
        if method in tokens:
            prefixes.add("_".join(tokens[: tokens.index(method)]))
    if len(prefixes) == 1:
        return next(iter(prefixes)).replace("_", "\\_")
    logger.warning(
        "%s holds %d model prefixes %s, labelling from the filename",
        path,
        len(prefixes),
        sorted(prefixes),
    )
    return os.path.splitext(os.path.basename(path))[0].replace("_", "\\_")


def load_csv(path: str, metric: str) -> tuple[str, dict[tuple[str, str], float]]:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path} has no data rows")
    if metric not in rows[0]:
        raise SystemExit(f"{path} has no column {metric!r} (has: {', '.join(rows[0])})")

    cells: dict[tuple[str, str], float] = {}
    freshest: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (row.get("method", ""), row.get("ratio", ""))
        value = row[metric].strip()
        if not value:
            logger.warning("%s: %s has no %s, skipping", path, row["model"], metric)
            continue
        created = row.get("created_at", "")
        if key in cells and created <= freshest.get(key, ""):
            logger.warning(
                "%s: %s is an older rerun of %s, skipping", path, row["model"], key
            )
            continue
        if key in cells:
            logger.warning(
                "%s: %s supersedes an earlier row for %s", path, row["model"], key
            )
        cells[key] = float(value)
        freshest[key] = created
    return model_label(rows, path), cells


def sorted_ratios(all_cells: list[dict[tuple[str, str], float]]) -> list[str]:
    """Least compressed first, so labels read 50\\%, 70\\%, 90\\%."""
    found = {ratio for cells in all_cells for _, ratio in cells}
    known = sorted(
        (r for r in found if RATIO_RE.match(r)),
        key=lambda r: -int(RATIO_RE.match(r).group(1)),
    )
    return known + sorted(found - set(known))


def sorted_methods(all_cells: list[dict[tuple[str, str], float]]) -> list[str]:
    found = {method for cells in all_cells for method, _ in cells}
    known = [m for m in METHOD_ORDER if m in found]
    return known + sorted(found - set(known))


def best_per_column(
    all_cells: list[dict[tuple[str, str], float]], methods: list[str], ratios: list[str]
) -> set[tuple[int, str, str]]:
    """(group, method, ratio) keys holding the best value in their column; ties all win."""
    best: set[tuple[int, str, str]] = set()
    for index, group in enumerate(all_cells):
        for ratio in ratios:
            column = {m: group[(m, ratio)] for m in methods if (m, ratio) in group}
            if not column:
                continue
            top = max(column.values())
            best |= {(index, m, ratio) for m, v in column.items() if v == top}
    return best


def render(
    labels: list[str],
    all_cells: list[dict[tuple[str, str], float]],
    precision: int,
    scale: float,
    caption: Optional[str],
    label: Optional[str],
    bold_best: bool,
) -> str:
    ratios = sorted_ratios(all_cells)
    methods = sorted_methods(all_cells)
    n_ratios = len(ratios)
    n_groups = len(labels)
    best = best_per_column(all_cells, methods, ratios) if bold_best else set()

    lines = [
        "\\begin{table*}[t!]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
    ]
    lines.append("\\begin{tabular}{l" + "|".join(["c" * n_ratios] * n_groups) + "}")
    lines.append("\\toprule")

    last_column = 1 + n_ratios * n_groups
    lines.append(
        f"& \\multicolumn{{{n_ratios * n_groups}}}{{c}}{{\\textbf{{Model}}}} \\\\"
    )
    lines.append(f"\\cmidrule(lr){{2-{last_column}}}")

    group_cells = [
        f"\\multicolumn{{{n_ratios}}}{{c{'|' if i < n_groups - 1 else ''}}}{{\\textbf{{{name}}}}}"
        for i, name in enumerate(labels)
    ]
    lines.append("& " + " & ".join(group_cells) + " \\\\")
    lines.append(
        "".join(
            f"\\cmidrule(lr){{{2 + i * n_ratios}-{1 + (i + 1) * n_ratios}}}"
            for i in range(n_groups)
        )
    )
    lines.append(
        "\\textbf{Method} & "
        + " & ".join(ratio_label(r) for _ in labels for r in ratios)
        + " \\\\"
    )
    lines.append("\\midrule")

    for method in methods:
        cells = []
        for index, group in enumerate(all_cells):
            for ratio in ratios:
                value = group.get((method, ratio))
                if value is None:
                    cells.append(MISSING)
                    continue
                text = f"{value * scale:.{precision}f}"
                cells.append(
                    f"\\textbf{{{text}}}" if (index, method, ratio) in best else text
                )
        lines.append(
            f"\\textbf{{{METHOD_LABELS.get(method, method.upper())}}} & "
            + " & ".join(cells)
            + " \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "csvs", nargs="+", help="CSVs from scrape_eval_results.py, one per model"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="column-group headers, one per CSV (default: derived from the model names)",
    )
    parser.add_argument(
        "--metric", default="avg", help="CSV column to tabulate (default: avg)"
    )
    parser.add_argument(
        "--precision", type=int, default=3, help="decimal places (default: 3)"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="multiply values by this, e.g. 100 for percent (default: 1)",
    )
    parser.add_argument(
        "--no-bold-best",
        dest="bold_best",
        action="store_false",
        help="don't bold the best method in each model/ratio column",
    )
    parser.add_argument("--caption", default=None, help="\\caption text")
    parser.add_argument("--label", default=None, help="\\label key")
    parser.add_argument(
        "--out", default=None, help="write the .tex here instead of stdout"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr
    )

    if args.labels and len(args.labels) != len(args.csvs):
        parser.error(f"got {len(args.labels)} labels for {len(args.csvs)} CSVs")

    loaded = [load_csv(path, args.metric) for path in args.csvs]
    labels = args.labels or [label for label, _ in loaded]
    all_cells = [cells for _, cells in loaded]

    table = render(
        labels,
        all_cells,
        args.precision,
        args.scale,
        args.caption,
        args.label,
        args.bold_best,
    )
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            handle.write(table)
        logger.info("Wrote %s", args.out)
    else:
        print(table, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
