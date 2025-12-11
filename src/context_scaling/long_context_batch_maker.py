from datasets import load_from_disk
import os
from argparse import ArgumentParser
from transformers import AutoTokenizer

def plot_token_length_hist(dataset_path, tokenizer, min_length=0,
                           text_field="text",
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
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        default="gpt2",
        help="Tokenizer model name or path"
    )
    parser.add_argument(
        "--min_length",
        type=int,
        required=True,
        help="Minimum token length to filter documents"
    )
    parser.add_argument(
        "--save_batch_size",
        type=int,
        required=True,
        help="Number of documents to save"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save the filtered dataset"
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="Name of the text field in the dataset (default: text)"
    )

    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    plot_token_length_hist(
        args.dataset_path,
        tok,
        min_length=args.min_length,
        text_field=args.text_field,
        save_batch_size=args.save_batch_size,
        save_dir=args.save_dir
    )