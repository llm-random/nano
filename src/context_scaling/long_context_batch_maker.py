from datasets import load_from_disk
import os, csv
import numpy as np
from argparse import ArgumentParser
from transformers import AutoTokenizer


def plot_token_length_hist_streaming(
    dataset_path,
    tokenizer,
    min_length=0,
    text_field="text",
    save_batch_size=None,
    save_data_dir=None,
    save_hist_dir=None,
    batch_size=64,
    max_len_for_hist=262144,
    bins=200,
):
    ds = load_from_disk(dataset_path)

    # fixed binning to avoid two-pass over the dataset
    lin_edges = np.linspace(1, max_len_for_hist, bins + 1)
    log_edges = np.exp(np.linspace(np.log(1.0), np.log(max_len_for_hist), bins + 1))

    hist = np.zeros(bins, dtype=np.int64)
    hist_log = np.zeros(bins, dtype=np.int64)

    selected_indices = []
    count_ge = 0

    # iterate in batches to keep memory bounded
    n = len(ds)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = ds[start:end]
        texts = batch[text_field]

        # IMPORTANT: avoid spawning huge intermediate structures across 16 processes
        enc = tokenizer(texts, add_special_tokens=False)
        lengths = np.fromiter((len(ids) for ids in enc["input_ids"]), dtype=np.int64)

        # update hist (clip long outliers so bins stay stable)
        clipped = np.clip(lengths, 1, max_len_for_hist)
        h, _ = np.histogram(clipped, bins=lin_edges)
        hl, _ = np.histogram(clipped, bins=log_edges)
        hist += h
        hist_log += hl

        # count + collect indices for saving
        if min_length > 0:
            mask = lengths >= min_length
            count_ge += int(mask.sum())
            if save_batch_size is not None and len(selected_indices) < save_batch_size:
                # add indices in original dataset coordinates
                new_idx = (np.nonzero(mask)[0] + start).tolist()
                need = save_batch_size - len(selected_indices)
                selected_indices.extend(new_idx[:need])
        else:
            count_ge += len(lengths)
            if save_batch_size is not None and len(selected_indices) < save_batch_size:
                need = save_batch_size - len(selected_indices)
                selected_indices.extend(
                    list(range(start, start + min(need, end - start)))
                )

        if start % (batch_size * 200) == 0:
            print(
                f"Processed {end}/{n} docs. >= {min_length}: {count_ge}. Saved candidates: {len(selected_indices)}"
            )

    print(f"number of documents ≥ {min_length} tokens: {count_ge}")

    # save hists
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

    # save subset
    if save_batch_size is not None and save_data_dir is not None:
        os.makedirs(save_data_dir, exist_ok=True)
        subset = ds.select(selected_indices)
        subset.save_to_disk(save_data_dir)
        print(f"Saved {len(subset)} documents to: {save_data_dir}")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--tokenizer", type=str, default="gpt2")
    p.add_argument("--min_length", type=int, default=0)
    p.add_argument("--save_batch_size", type=int, default=None)
    p.add_argument("--save_data_dir", type=str, default=None)
    p.add_argument("--save_hist_dir", type=str, default=None)
    p.add_argument("--text_field", type=str, default="text")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_len_for_hist", type=int, default=262144)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    # avoid tokenizer thread explosion on clusters
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    plot_token_length_hist_streaming(
        args.dataset_path,
        tok,
        min_length=args.min_length,
        text_field=args.text_field,
        save_batch_size=args.save_batch_size,
        save_data_dir=args.save_data_dir,
        save_hist_dir=args.save_hist_dir,
        batch_size=args.batch_size,
        max_len_for_hist=args.max_len_for_hist,
    )
