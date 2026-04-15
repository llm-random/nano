import os
import csv
import numpy as np
from argparse import ArgumentParser
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer


def save_histogram_csv(path, edges, h):
    """Saves histogram data to a CSV file."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_left", "bin_right", "count"])
        for left, right, c in zip(edges[:-1], edges[1:], h):
            w.writerow([int(left), int(right), int(c)])


def generate_histogram(
    dataset_path,
    tokenizer_name,
    save_dir,
    num_shards=32,
    max_len=10000,
    bins=200,
):
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # 1. Load Dataset from disk
    print(f"Loading dataset from: {dataset_path}")
    hf_dataset = load_from_disk(dataset_path)

    # 2. Convert to Iterable Dataset as requested
    print(f"Converting to iterable dataset with {num_shards} shards...")
    hf_dataset = hf_dataset.to_iterable_dataset(num_shards=num_shards)

    # 3. Define Tokenization Logic
    def compute_length(batch):
        # We compute length on the fly
        encoding = tokenizer(batch["text"], add_special_tokens=False)
        return {"length": [len(ids) for ids in encoding["input_ids"]]}

    # Map the function over the stream
    # Note: batched=True is efficient here as it reduces python overhead
    print("mapping...")
    dataset_stream = hf_dataset.map(compute_length, batched=True)

    # 4. Prepare Histogram Bins
    lin_edges = np.linspace(10, max_len, bins + 1)
    log_edges = np.exp(np.linspace(np.log(10.0), np.log(max_len), bins + 1))

    hist_lin = np.zeros(bins, dtype=np.int64)
    hist_log = np.zeros(bins, dtype=np.int64)

    # 5. Iterate and Aggregate
    print("Processing stream...")
    total_docs = 0
    batch_size = 1000  # Number of examples to pull at once for numpy vectorization

    # We use .iter() with batch_size so we get a list of items (a batch)
    # allowing us to use numpy on the whole chunk at once.
    for i, batch in enumerate(dataset_stream.iter(batch_size=batch_size)):
        lengths = np.array(batch["length"], dtype=np.int64)

        # Clip lengths to max_len to ensure they fit in bins
        clipped = np.clip(lengths, 1, max_len)

        # Update histograms
        h_lin, _ = np.histogram(clipped, bins=lin_edges)
        h_log, _ = np.histogram(clipped, bins=log_edges)

        hist_lin += h_lin
        hist_log += h_log
        total_docs += len(lengths)

        if (i + 1) % 10 == 0:
            print(f"Processed {total_docs} documents...", end="\r")

    print(f"\nFinished processing {total_docs} documents.")

    # 6. Save Results
    os.makedirs(save_dir, exist_ok=True)
    save_histogram_csv(os.path.join(save_dir, "hist_normal.csv"), lin_edges, hist_lin)
    save_histogram_csv(os.path.join(save_dir, "hist_log.csv"), log_edges, hist_log)
    print(f"Saved histograms to {save_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--save_dir", type=str, required=True)
    # Renamed num_proc to num_shards to align with the prompt's logic requirements
    parser.add_argument(
        "--num_shards",
        type=int,
        default=32,
        help="Number of shards for iterable dataset",
    )
    parser.add_argument("--max_len", type=int, default=10000)
    parser.add_argument("--bins", type=int, default=200)
    args = parser.parse_args()

    generate_histogram(
        args.dataset_path,
        args.tokenizer,
        args.save_dir,
        args.num_shards,
        args.max_len,
        args.bins,
    )
