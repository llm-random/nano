"""Run in the OFFICIAL SliceGPT env (transformers 4.41 + their src on PYTHONPATH): dump the exact WikiText2
calibration and test token ids their run_slicegpt.py uses (same functions, same defaults, seed 42), so that
scripts/slicegpt_crosscheck.py --token-file feeds our port identical data."""

import sys

import torch
from transformers import AutoTokenizer

from slicegpt import data_utils

model, out = sys.argv[1], sys.argv[2]
tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # LlamaModelAdapter.post_init
ds = data_utils.get_dataset("wikitext2")
train_loader = data_utils.prepare_dataloader(
    ds["train"],
    tokenizer,
    max_seqlen=2048,
    batch_size=16,
    nsamples=128,
    varied_seqlen=False,
    seed=42,
)
test_loader = data_utils.prepare_test_dataloader(ds["test"], tokenizer, batch_size=4)
# keep their exact batches (padding="longest" per batch) + attention masks; PCA sums over all samples, so the
# (RNG-dependent) batch order of their SubsetRandomSampler does not matter, only the sample set does.
calib = [(b["input_ids"], b["attention_mask"]) for b in train_loader]
for (
    _,
    m,
) in calib:  # right padding only (causal attention => padding can't affect real tokens)
    assert torch.equal(
        m, m.sort(dim=1, descending=True).values
    ), "expected right padding"
test = torch.cat([b["input_ids"] for b in test_loader])
torch.save(
    {
        "calibration": calib,
        "test": test,
        "pad_token_id": tokenizer.pad_token_id,
        "padding_side": tokenizer.padding_side,
    },
    out,
)
print(
    "DUMPED",
    len(calib),
    "batches",
    sum(int(m.sum()) for _, m in calib),
    "real tokens; test",
    tuple(test.shape),
)
