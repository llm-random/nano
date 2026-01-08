import os
import csv
import numpy as np
from argparse import ArgumentParser
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer


def get_arrow_files(dataset_path):
    """Returns sorted list of arrow files if dataset is sharded, else None."""
    arrow_files = sorted(
        [
            os.path.join(dataset_path, f)
            for f in os.listdir(dataset_path)
            if f.startswith("data-") and f.endswith(".arrow")
        ]
    )
    return arrow_files if len(arrow_files) > 50 else None


def process_shard_histogram(shard, tokenizer, num_proc, max_len, lin_edges, log_edges):
    shard = shard.map(
        lambda x: {
            "length": len(tokenizer(x["text"], add_special_tokens=False)["input_ids"])
        },
        num_proc=num_proc,
    )
    lengths = np.array(shard["length"], dtype=np.int64)
    clipped = np.clip(lengths, 1, max_len)

    h, _ = np.histogram(clipped, bins=lin_edges)
    hl, _ = np.histogram(clipped, bins=log_edges)
    return h, hl, len(lengths)


def save_histogram_csv(path, edges, h):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_left", "bin_right", "count"])
        for left, right, c in zip(edges[:-1], edges[1:], h):
            w.writerow([int(left), int(right), int(c)])


def generate_histogram(
    dataset_path,
    tokenizer_name,
    save_dir,
    num_shards=10,
    num_proc=None,
    max_len=10000,
    bins=200,
):
    if num_proc is None:
        num_proc = os.cpu_count()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    lin_edges = np.linspace(10, max_len, bins + 1)
    log_edges = np.exp(np.linspace(np.log(10.0), np.log(max_len), bins + 1))

    hist = np.zeros(bins, dtype=np.int64)
    hist_log = np.zeros(bins, dtype=np.int64)

    arrow_files = get_arrow_files(dataset_path)

    if arrow_files:
        num_files = len(arrow_files)
        for i, arrow_file in enumerate(arrow_files):
            print(f"Processing file {i+1}/{num_files}...")
            shard = load_dataset("arrow", data_files=arrow_file, split="train")
            h, hl, n = process_shard_histogram(
                shard, tokenizer, num_proc, max_len, lin_edges, log_edges
            )
            hist += h
            hist_log += hl
            print(f"File {i+1}/{num_files}: {n} documents")
    else:
        dataset = load_from_disk(dataset_path)
        for i in range(num_shards):
            print(f"Processing shard {i+1}/{num_shards}...")
            shard = dataset.shard(num_shards=num_shards, index=i)
            h, hl, n = process_shard_histogram(
                shard, tokenizer, num_proc, max_len, lin_edges, log_edges
            )
            hist += h
            hist_log += hl
            print(f"Shard {i+1}/{num_shards}: {n} documents")

    os.makedirs(save_dir, exist_ok=True)
    save_histogram_csv(os.path.join(save_dir, "hist_normal.csv"), lin_edges, hist)
    save_histogram_csv(os.path.join(save_dir, "hist_log.csv"), log_edges, hist_log)

    print(f"Saved histograms to {save_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--num_shards", type=int, default=10)
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--max_len", type=int, default=10000)
    parser.add_argument("--bins", type=int, default=200)
    args = parser.parse_args()

    generate_histogram(
        args.dataset_path,
        args.tokenizer,
        args.save_dir,
        args.num_shards,
        args.num_proc,
        args.max_len,
        args.bins,
    )
