import os
import shutil
from argparse import ArgumentParser
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer
import time


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


def save_long_ctx_batch(
    dataset_path, tokenizer_name, seq_len, save_path, num_shards=10, num_proc=None
):
    if num_proc is None:
        num_proc = os.cpu_count()

    dataset = load_from_disk(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    shards_dir = f"{save_path}_shards"
    os.makedirs(shards_dir, exist_ok=True)

    for i in range(num_shards):
        print(f"Processing shard {i+1}/{num_shards}...")
        shard = dataset.shard(num_shards=num_shards, index=i)

        shard = shard.map(
            lambda x: tokenizer(x["text"], add_special_tokens=False),
            batched=True,
            num_proc=num_proc,
        )

        shard = shard.filter(
            lambda x: len(x["input_ids"]) >= seq_len,
            num_proc=num_proc,
        )

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
