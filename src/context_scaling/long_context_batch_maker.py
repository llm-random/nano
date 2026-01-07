import os
import csv
from argparse import ArgumentParser
import numpy as np
import multiprocessing as mp
from functools import partial
from datasets import load_from_disk, load_dataset, IterableDataset
from transformers import AutoTokenizer


# Worker function for multiprocessing
def process_batch_lengths(texts, tokenizer_name):
    """
    Worker function to tokenize text and return lengths.
    We re-initialize the tokenizer inside the process to avoid deadlocks/pickling issues.
    """
    # Fast tokenizer is crucial here
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # Tokenize without adding special tokens (usually desired for raw length)
    # and return length directly to save memory (don't return full input_ids)
    enc = tokenizer(texts, add_special_tokens=False, return_length=True)

    # Some tokenizers return 'length', others just 'input_ids'. Handle both.
    if "length" in enc:
        return enc["length"]
    else:
        return [len(ids) for ids in enc["input_ids"]]


def plot_token_length_hist_robust(
    dataset_path,
    tokenizer_name,
    min_length=0,
    text_field="text",
    save_batch_size=None,
    save_data_dir=None,
    save_hist_dir=None,
    batch_size=1000,
    num_proc=1,
    max_len_for_hist=262144,
    bins=200,
    is_streaming=False,
):
    # 1. Flexible Loading (Disk or Stream)
    print(f"Loading dataset from: {dataset_path}")
    if os.path.exists(dataset_path):
        ds = load_from_disk(dataset_path)
    else:
        # Fallback to loading from Hub if path doesn't exist locally
        print("Path not found on disk, attempting to load from HF Hub...")
        ds = load_dataset(dataset_path, split="train", streaming=is_streaming)

    # If it's a map-style dataset (local disk), we can convert to iterable for consistent logic
    # or just iterate it. If num_proc > 1, we use a pool.

    # Determine total size if possible for progress bar
    try:
        n = len(ds)
        print(f"Dataset size: {n} documents")
    except:
        n = None
        print("Dataset size: Unknown (Streaming)")

    # 2. Setup Histogram Bins
    lin_edges = np.linspace(1, max_len_for_hist, bins + 1)
    log_edges = np.exp(np.linspace(np.log(1.0), np.log(max_len_for_hist), bins + 1))

    hist = np.zeros(bins, dtype=np.int64)
    hist_log = np.zeros(bins, dtype=np.int64)

    selected_indices = (
        []
    )  # This might need to store IDs if streaming, but indices work for local
    saved_subset_count = 0
    total_processed = 0
    count_ge = 0

    # 3. Setup Multiprocessing
    # We use a Pool to tokenize batches in parallel.
    # We use 'spawn' or 'fork' context carefully depending on OS, but default Pool usually works.
    pool = mp.Pool(processes=num_proc) if num_proc > 1 else None

    print(
        f"Starting processing with batch_size={batch_size} and num_proc={num_proc}..."
    )

    # Helper to update histograms from a list of lengths
    def update_stats(lengths, start_idx, current_batch_texts):
        nonlocal count_ge, saved_subset_count, hist, hist_log

        lengths = np.array(lengths, dtype=np.int64)

        # Update histograms
        clipped = np.clip(lengths, 1, max_len_for_hist)
        h, _ = np.histogram(clipped, bins=lin_edges)
        hl, _ = np.histogram(clipped, bins=log_edges)
        hist += h
        hist_log += hl

        # Handle filtering and saving
        # Note: If streaming, 'start_idx' is relative to the current session
        mask = lengths >= min_length
        local_count_ge = int(mask.sum())
        count_ge += local_count_ge

        # Saving subset logic
        if save_batch_size is not None and saved_subset_count < save_batch_size:
            # Get indices in the current batch that satisfy the condition
            valid_indices_in_batch = np.nonzero(mask)[0]

            # If we need to save the actual data, we must hold the text in memory
            # or rely on indices. For robustness, we collect the TEXT data immediately
            # if we are in streaming mode, or indices if local.

            # To be safe for both streaming and local: collect the data itself
            candidates = [current_batch_texts[i] for i in valid_indices_in_batch]

            slots_needed = save_batch_size - saved_subset_count
            to_save = candidates[:slots_needed]

            if to_save and save_data_dir:
                # Append to a temporary CSV or JSONL to avoid holding in RAM
                # Here we just buffer in memory assuming save_batch_size is small (e.g. 1k)
                # If save_batch_size is huge, this should be a file append.
                selected_indices.extend(to_save)

            saved_subset_count += len(to_save)

    # 4. Main Processing Loop
    # usage of iter(batch_size) allows efficient traversal without loading full columns
    iterator = ds.iter(batch_size=batch_size)

    # Accumulate futures if using multiprocessing
    async_results = []

    # We track absolute index manually for logging
    current_idx = 0

    for batch in iterator:
        texts = batch[text_field]
        batch_len = len(texts)

        if num_proc > 1:
            # Submit job to pool
            # We must pass the tokenizer name, not the object, to avoid pickling the tokenizer
            res = pool.apply_async(process_batch_lengths, (texts, tokenizer_name))
            async_results.append((res, current_idx, texts if save_batch_size else []))

            # Manage memory: don't let the queue grow infinitely
            # If we have too many pending results, wait for some
            if len(async_results) > num_proc * 4:
                res, s_idx, b_texts = async_results.pop(0)
                lengths = res.get()  # Block until ready
                update_stats(lengths, s_idx, b_texts)

                # Report progress
                if (s_idx + batch_size) % (batch_size * 50) == 0:
                    print(
                        f"Processed {s_idx + batch_size} docs... (>= {min_length}: {count_ge})"
                    )

        else:
            # Single process fallback
            # Re-instantiating tokenizer every batch is slow, so we do it outside loop in single-proc
            # But for consistency with the worker function logic:
            if "tokenizer_obj" not in locals():
                tokenizer_obj = AutoTokenizer.from_pretrained(
                    tokenizer_name, use_fast=True
                )

            enc = tokenizer_obj(texts, add_special_tokens=False, return_length=True)
            if "length" in enc:
                lengths = enc["length"]
            else:
                lengths = [len(ids) for ids in enc["input_ids"]]

            update_stats(lengths, current_idx, texts if save_batch_size else [])

            if (current_idx + batch_size) % (batch_size * 50) == 0:
                print(
                    f"Processed {current_idx + batch_size} docs... (>= {min_length}: {count_ge})"
                )

        current_idx += batch_len

    # Process remaining async results
    if num_proc > 1:
        for res, s_idx, b_texts in async_results:
            lengths = res.get()
            update_stats(lengths, s_idx, b_texts)

    if pool:
        pool.close()
        pool.join()

    print(
        f"Total processed: {current_idx}. Documents ≥ {min_length} tokens: {count_ge}"
    )

    # 5. Saving Results
    if save_hist_dir is not None:
        os.makedirs(save_hist_dir, exist_ok=True)

        def _save(path, edges, h):
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["bin_left", "bin_right", "count"])
                for left, right, c in zip(edges[:-1], edges[1:], h):
                    w.writerow([int(left), int(right), int(c)])

        _save(os.path.join(save_hist_dir, "hist_normal.csv"), lin_edges, hist)
        _save(os.path.join(save_hist_dir, "hist_log.csv"), log_edges, hist_log)
        print(f"Saved histogram CSVs to: {save_hist_dir}")

    # Save subset (Robust method: Write generic list of dicts/strings)
    if (
        save_batch_size is not None
        and save_data_dir is not None
        and len(selected_indices) > 0
    ):
        os.makedirs(save_data_dir, exist_ok=True)
        # Since we collected raw text strings (to be streaming safe), we save differently
        # We can construct a simple HF dataset from memory
        from datasets import Dataset

        subset_ds = Dataset.from_dict({text_field: selected_indices})
        subset_ds.save_to_disk(save_data_dir)
        print(f"Saved {len(subset_ds)} documents to: {save_data_dir}")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to local disk dataset or Hub ID",
    )
    p.add_argument("--tokenizer", type=str, default="gpt2")
    p.add_argument("--min_length", type=int, default=0)
    p.add_argument("--save_batch_size", type=int, default=None)
    p.add_argument("--save_data_dir", type=str, default=None)
    p.add_argument("--save_hist_dir", type=str, default=None)
    p.add_argument("--text_field", type=str, default="text")
    p.add_argument("--batch_size", type=int, default=1000)
    p.add_argument(
        "--num_proc",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes",
    )
    p.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode (load_dataset streaming=True)",
    )
    p.add_argument("--max_len_for_hist", type=int, default=262144)
    args = p.parse_args()

    # Disable parallel tokenization in the main process to prevent fighting with multiprocessing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    plot_token_length_hist_robust(
        args.dataset_path,
        args.tokenizer,
        min_length=args.min_length,
        text_field=args.text_field,
        save_batch_size=args.save_batch_size,
        save_data_dir=args.save_data_dir,
        save_hist_dir=args.save_hist_dir,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        max_len_for_hist=args.max_len_for_hist,
        is_streaming=args.streaming,
    )
