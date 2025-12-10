from datasets import load_from_disk
import os
from argparse import ArgumentParser
from transformers import AutoTokenizer

def plot_token_length_hist(dataset_path, tokenizer, min_length=0,
                           text_field="text", bins=100,
                           save_batch_size=None, save_dir=None):
    ds = load_from_disk(dataset_path)

    # Compute token lengths
    def _token_len(batch):
        enc = tokenizer(batch[text_field], add_special_tokens=False)
        return {"length": [len(ids) for ids in enc["input_ids"]]}

    ds = ds.map(
        _token_len,
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        keep_in_memory=True,
    )

    # Filter by minimum token length
    if min_length > 0:
        ds = ds.filter(lambda x: x["length"] >= min_length)

    lengths = ds["length"]
    print(f"number of documents ≥ {min_length} tokens: {len(lengths)}")

    # ---- SAVE FIRST BATCH ----
    if save_batch_size is not None and save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        subset = ds.select(range(min(save_batch_size, len(ds))))
        subset.save_to_disk(save_dir)
        print(f"Saved {len(subset)} documents to: {save_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        
    )
    tok = AutoTokenizer.from_pretrained("gpt2")
    plot_token_length_hist("/home/janek/Documents/IDEAS/nano/data", tok, min_length=512, save_batch_size=64, save_dir='../data_long_ctx')