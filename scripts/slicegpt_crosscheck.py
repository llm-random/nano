"""V3: cross-check our nano SliceGPT port against the official implementation (slicegpt.md, Step 3).

Runs OUR port in FAITHFUL mode (no dff pruning, embedding centering ON like their `fuse_modules`) on a model the
official repo supports (meta-llama/Meta-Llama-3-8B, no RoPE scaling) with THEIR calibration / test data pipeline
(WikiText2, copied below) and THEIR perplexity metric, so the numbers are directly comparable with:

    python experiments/run_slicegpt.py --model meta-llama/Meta-Llama-3-8B --sparsity 0.25 --dtype fp32 \
        --cal-dataset wikitext2 --cal-nsamples 128 --cal-max-seqlen 2048 --cal-batch-size 16 \
        --final-orientation pca --round-interval 8 --ppl-eval-batch-size 4 --eval-baseline --eval-fused-model --no-wandb

Usage (1 GPU):  python scripts/slicegpt_crosscheck.py [--sparsity 0.25] [--cal-nsamples 128]
"""

import argparse
import json
import logging
import os
import sys

import datasets
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import resolver  # noqa: E402,F401
from src.core.llama import copy_llama_model_weights_from_HF  # noqa: E402
from src.projected_compression.slicegpt import (
    finalize_norms,
    fuse_norms,
    rotate_and_slice,
)  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("crosscheck")


# ---- copied from TransformerCompression/src/slicegpt/data_utils.py (MIT, Microsoft) -------------------------------
def prepare_test_dataloader(dataset, tokenizer, seqlen=2048, batch_size=1):
    class TestDataset(Dataset):
        def __init__(self, ds, tokenizer, seqlen=2048):
            tokenized_ds = tokenizer("\n\n".join(ds["text"]), return_tensors="pt")
            nsamples = tokenized_ds.input_ids.numel() // seqlen
            self.input_ids = tokenized_ds.input_ids[0, : nsamples * seqlen].reshape(
                nsamples, seqlen
            )

        def __getitem__(self, idx):
            return {"input_ids": self.input_ids[idx]}

        def __len__(self):
            return len(self.input_ids)

    return DataLoader(TestDataset(dataset, tokenizer, seqlen), batch_size=batch_size)


def prepare_dataloader(
    dataset, tokenizer, max_seqlen=2048, batch_size=1, nsamples=128, seed=42
):
    data_name = dataset.column_names[0]
    ds = dataset.filter(lambda x: len(x[data_name]) > 0)
    data_list = ds[data_name]
    new_data_list = []
    torch.manual_seed(seed)
    indices = list(range(len(data_list)))
    while len(new_data_list) < nsamples and len(indices) > 0:
        start_idx = torch.randint(0, len(indices), (1,)).item()
        idx = start_idx
        tokens = []
        while len(tokens) < max_seqlen and idx < len(indices):
            item = data_list[indices[idx]]
            sep = "" if not tokens else "\n\n"
            tokens += tokenizer.tokenize(sep + item)
            idx += 1
        indices = indices[:start_idx] + indices[idx:]
        if len(tokens) >= max_seqlen:
            tokens = tokens[:max_seqlen]
            new_data_list.append(tokenizer.convert_tokens_to_string(tokens))
    ds = datasets.Dataset.from_dict({data_name: new_data_list})

    def tokenize(data_batch):
        return tokenizer(
            data_batch[data_name],
            padding="longest",
            max_length=max_seqlen,
            truncation=True,
            return_tensors="pt",
        )

    ds.set_transform(tokenize)
    torch.manual_seed(seed)
    sampler = SubsetRandomSampler(torch.randperm(len(ds))[:nsamples])
    return DataLoader(ds, batch_size=batch_size, sampler=sampler)


# ---- their gpu_utils.evaluate_ppl, for a nano model (logits = model(input_ids)) -----------------------------------
@torch.no_grad()
def evaluate_ppl(model, pad_token_id, loader, device):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
    nlls = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        logits = model(input_ids)[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        nll = loss_fn(logits.permute(0, 2, 1).float(), shift_labels).float()
        mask = shift_labels != loss_fn.ignore_index
        nlls.append((nll * mask).sum(dim=1) / mask.sum(dim=1))
    return torch.exp(torch.cat(nlls).mean()).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    p.add_argument("--sparsity", type=float, default=0.25)
    p.add_argument("--round-interval", type=int, default=8)
    p.add_argument("--cal-nsamples", type=int, default=128)
    p.add_argument("--cal-batch-size", type=int, default=16)
    p.add_argument("--cal-max-seqlen", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ppl-eval-batch-size", type=int, default=4)
    p.add_argument("--orientation", default="pca")
    p.add_argument("--out", default=None)
    p.add_argument(
        "--token-file",
        default=None,
        help="token ids dumped by slicegpt_dump_ref_tokens.py (official env)",
    )
    args = p.parse_args()
    device = torch.device("cuda")

    if (
        args.token_file
    ):  # exact token ids from the official pipeline (preferred: removes tokenizer-version noise)
        tok = torch.load(args.token_file)
        pad_token_id = tok["pad_token_id"]
        calib = [ids for ids, _ in tok["calibration"]]
        masks = [m for _, m in tok["calibration"]]
        test_loader = DataLoader(
            [{"input_ids": x} for x in tok["test"]], batch_size=args.ppl_eval_batch_size
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        tokenizer.pad_token = (
            tokenizer.eos_token
        )  # as their LlamaModelAdapter.post_init
        pad_token_id = tokenizer.pad_token_id
        ds = datasets.load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1"
        )  # == "wikitext" (moved on the Hub)
        train_loader = prepare_dataloader(
            ds["train"],
            tokenizer,
            args.cal_max_seqlen,
            args.cal_batch_size,
            args.cal_nsamples,
            args.seed,
        )
        calib = [b["input_ids"] for b in train_loader]
        masks = [b["attention_mask"] for b in train_loader]
        test_loader = prepare_test_dataloader(
            ds["test"], tokenizer, batch_size=args.ppl_eval_batch_size
        )

    # nano Llama-3-8B (Meta-Llama-3-8B: rope theta 500000, NO llama3 rope scaling)
    with initialize(version_base=None, config_path="../configs/slicegpt"):
        cfg = compose(
            config_name="llama_8B_slice_hyb_entropy",
            overrides=[
                "model.encoder.block_fn.attention_fn.rope_scale_freqs=false",
                "common.sequence_length=2048",
            ],
        )
    model = instantiate(cfg.model, _convert_="all")
    copy_llama_model_weights_from_HF(model, args.model)
    model = model.to(device=device, dtype=torch.float32).eval()
    res = {"original_params": sum(x.numel() for x in model.parameters())}

    res["original_ppl"] = evaluate_ppl(model, pad_token_id, test_loader, device)
    log.info(f"Original ppl: {res['original_ppl']:.4f}")

    fuse_norms(
        model, center_embeddings=True
    )  # their fuse_modules always centers embeddings
    res["post_fusion_ppl"] = evaluate_ppl(model, pad_token_id, test_loader, device)
    log.info(f"Post-fusion: {res['post_fusion_ppl']:.4f}")

    D = model.embedding.weight.shape[1]
    d = int((1 - args.sparsity) * D)
    d -= d % args.round_interval
    # their CLI uses ConstSlicingScheduler with do_slice_head=False -> last block output + head stay at width D
    rotate_and_slice(
        model,
        calib,
        d=d,
        orientation=args.orientation,
        device=device,
        slice_head=False,
        ignore_masks=masks,
    )
    finalize_norms(model)
    res.update(
        {"d": d, "sliced_ppl": evaluate_ppl(model, pad_token_id, test_loader, device)}
    )
    res["sliced_params"] = sum(x.numel() for x in model.parameters())
    log.info(f"After rotating and slicing {res['sliced_ppl']:.4f}  (d={d})")
    print("CROSSCHECK_RESULT " + json.dumps(res))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
