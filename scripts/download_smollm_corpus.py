#!/usr/bin/env python
"""Download SmolLM corpus datasets to local storage via save_to_disk (Arrow format)."""
import os
import argparse
from datasets import load_dataset

DATASETS = [
    ("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", "fineweb-edu-dedup"),
    ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "cosmopedia-v2"),
    ("HuggingFaceTB/smollm-corpus", "python-edu", "python-edu"),
    ("HuggingFaceTB/smollm-corpus", "open-web-math", "open-web-math"),
]


def main(output_dir: str, splits: list):
    for repo_id, config, name in DATASETS:
        for split in splits:
            path = os.path.join(output_dir, name, split)
            if os.path.exists(path):
                print(f"Skipping {name}/{split} — already exists at {path}")
                continue
            print(f"Downloading {repo_id} ({config}) split={split} ...")
            ds = load_dataset(repo_id, config, split=split)
            os.makedirs(path, exist_ok=True)
            ds.save_to_disk(path)
            print(f"Saved {name}/{split} to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, help="Root directory for datasets")
    parser.add_argument("--splits", nargs="+", default=["train"], help="Splits to download")
    args = parser.parse_args()
    main(args.output_dir, args.splits)
