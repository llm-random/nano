import os
import shutil
from argparse import ArgumentParser
from datasets import load_from_disk, load_dataset, concatenate_datasets
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


def concatenate_shards(shards_dir, output_path):
    shard_dirs = sorted(
        [
            os.path.join(shards_dir, d)
            for d in os.listdir(shards_dir)
            if d.startswith("shard_")
        ]
    )
    shards = [load_from_disk(d) for d in shard_dirs]
    final = concatenate_datasets(shards)
    final.save_to_disk(output_path)
    return final


def process_shard_filter(shard, tokenizer, seq_len, num_proc):
    shard = shard.map(
        lambda x: tokenizer(x["text"], add_special_tokens=False),
        batched=True,
        num_proc=num_proc,
    )
    shard = shard.filter(
        lambda x: len(x["input_ids"]) >= seq_len,
        num_proc=num_proc,
    )
    return shard


def save_long_ctx_batch(
    dataset_path, tokenizer_name, seq_len, save_path, num_shards=10, num_proc=None
):
    if num_proc is None:
        num_proc = os.cpu_count()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    shards_dir = f"{save_path}_shards"
    os.makedirs(shards_dir, exist_ok=True)

    arrow_files = get_arrow_files(dataset_path)

    if arrow_files:
        num_files = len(arrow_files)
        for i, arrow_file in enumerate(arrow_files):
            print(f"Processing file {i+1}/{num_files}...")
            shard = load_dataset("arrow", data_files=arrow_file, split="train")
            shard = process_shard_filter(shard, tokenizer, seq_len, num_proc)
            if len(shard) > 0:
                shard.save_to_disk(os.path.join(shards_dir, f"shard_{i}"))
            print(f"File {i+1}/{num_files}: found {len(shard)} long sequences")
    else:
        dataset = load_from_disk(dataset_path)
        for i in range(num_shards):
            print(f"Processing shard {i+1}/{num_shards}...")
            shard = dataset.shard(num_shards=num_shards, index=i)
            shard = process_shard_filter(shard, tokenizer, seq_len, num_proc)
            shard.save_to_disk(os.path.join(shards_dir, f"shard_{i}"))
            print(f"Shard {i+1}/{num_shards}: found {len(shard)} long sequences")

    print("Concatenating shards...")
    final = concatenate_shards(shards_dir, save_path)

    print(f"Total: {len(final)} sequences >= {seq_len} tokens")
    print(f"Saved to {save_path}")

    shutil.rmtree(shards_dir)
    print(f"Cleaned up {shards_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--num_shards", type=int, default=10)
    parser.add_argument("--num_proc", type=int, default=None)
    args = parser.parse_args()

    save_long_ctx_batch(
        args.dataset_path,
        args.tokenizer,
        args.seq_len,
        args.save_path,
        args.num_shards,
        args.num_proc,
    )
