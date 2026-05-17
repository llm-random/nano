import logging
import os 
from typing import Callable, Optional, override, List

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from transformers import GPT2TokenizerFast, AutoTokenizer

from src.core.datasets import AbstractDataset, collate_wrapper
from itertools import zip_longest

logger = logging.getLogger(__name__)


class GlueDataset(AbstractDataset):
    BUFFER_SIZE = 1000
    NUM_SHARDS = 64

    def __init__(
        self,
        sequence_length,
        tokenize_fn: Callable,
        path: Optional[str] = None,
        split: Optional[str] = None,
        seed: Optional[int] = None,
        use_new_sampling_method: bool = True,
        shuffle: bool = True,
        world_size_independent: bool = False,
        task_name: Optional[str] = None,
    ):
        super().__init__(
            sequence_length,
            tokenize_fn,
            path,
            split,
            seed,
            use_new_sampling_method,
            shuffle,
            world_size_independent,
        )
        self.task_name = task_name
        self._load_dataset(path, split, seed, tokenize_fn, shuffle)

    def _get_hf_dataset(self, path, split, seed, shuffle):
        if path is None:
            logger.debug(
                f"Loading 'nyu-mll/glue' dataset task '{self.task_name}' from HuggingFace with split={split}"
            )
            hf_dataset = load_dataset(
                "nyu-mll/glue",
                self.task_name,
                split=split,
                streaming=True,
                trust_remote_code=True,
            )
        else:
            logger.info(f"Loading dataset from path '{path}'")
            logger.info(f"Split: {split}")
            hf_dataset = load_dataset(path, split=split)
            if not hasattr(hf_dataset, "set_epoch"): # Check if it's already an IterableDataset
                hf_dataset = hf_dataset.to_iterable_dataset(num_shards=self.NUM_SHARDS)

        if not self.world_size_independent:
            hf_dataset = split_dataset_by_node(
                hf_dataset, rank=self.rank, world_size=self.world_size
            )

        if shuffle:
            hf_dataset = hf_dataset.shuffle(buffer_size=self.BUFFER_SIZE, seed=seed)

        return hf_dataset

    def _load_dataset(self, path, split, seed, tokenize_fn, shuffle: bool):
        hf_dataset = self._get_hf_dataset(path, split, seed, shuffle)

        # Map task specific columns to 'text' for generic tokenize_fn
        if self.task_name == "sst2":
            hf_dataset = hf_dataset.map(lambda x: {"text": x["sentence"]})

        elif self.task_name == "mnli":
            hf_dataset = hf_dataset.map(lambda x: {"text_1": x["premise"], 
                                                   "text_2": x["hypothesis"]})

        self.data_generator = hf_dataset.map(tokenize_fn, batched=True)

    def sample_packer(self):
        sampler = iter(self.get_infinite_sampler())
        while True:
            full_sample = next(sampler)
            tokens = full_sample['input_ids']
            label = full_sample['label']
            attention_mask = full_sample['attention_mask']
            yield (tokens, label, attention_mask)

    def get_infinite_sampler(self):
        epoch = 0
        while True:
            self.data_generator.set_epoch(epoch)
            for next_sample in self.data_generator:
                yield next_sample
            epoch += 1

    def full_sample_packer(self):
        sampler = iter(self.get_full_sampler())
        for full_sample in sampler:
            tokens = full_sample['input_ids']
            label = full_sample['label']
            attention_mask = full_sample['attention_mask']
            yield (tokens, label, attention_mask)

    def get_full_sampler(self):
        for next_sample in self.data_generator:
            yield next_sample


    def full_iter(self):
        if self.world_size_independent:
            return itertools.islice(
                self.full_sample_packer(), self.rank, None, self.world_size
            )
        else:
            return self.full_sample_packer()


class GlueLengthSplitDataset(GlueDataset):
    def __init__(
        self,
        sequence_length,
        tokenize_fn: Callable,
        path: Optional[str] = None,
        split: Optional[str] = None,
        seed: Optional[int] = None,
        use_new_sampling_method: bool = True,
        shuffle: bool = True,
        world_size_independent: bool = False,
        task_name: Optional[str] = None,
        split_values: Optional[List[int]] = None
    ):
        self.split_values = split_values
        super().__init__(
            sequence_length,
            tokenize_fn,
            path,
            split,
            seed,
            use_new_sampling_method,
            shuffle,
            world_size_independent,
            task_name,
        )

    def add_sequence_length_column(self, dataset):
        return dataset.map(lambda x: {"sequence_length": sum(x["attention_mask"])})
    
    def filter_by_length(self, dataset, min_value: int, max_value: int):
        return dataset.filter(lambda x: x["sequence_length"] >= min_value and x["sequence_length"] < max_value)

    @override
    def _load_dataset(self, path, split, seed, tokenize_fn, shuffle: bool):
        hf_dataset = self._get_hf_dataset(path, split, seed, shuffle)
        tokenized_dataset = hf_dataset.map(tokenize_fn, batched=True)
        tokenized_dataset = self.add_sequence_length_column(tokenized_dataset)
        
        min_value = 0
        split_datasets = []

        full_split_values = self.split_values + [float("inf")]
        
        for max_value in full_split_values:
            filtered_dataset = self.filter_by_length(tokenized_dataset, min_value, max_value)
            split_datasets.append(filtered_dataset)
            min_value = max_value
        
        self.data_generator = split_datasets

    def full_sample_packer(self):
        sampler = iter(self.get_full_sampler())
        for full_samples in sampler:

            full_sample_list = [(sample['input_ids'], sample['label'], sample['attention_mask']) if sample is not None else None for sample in full_samples]
            
            # tokens_list = [sample['input_ids'] if sample is not None else None for sample in full_samples]
            # labels_list = [sample['label'] if sample is not None else None for sample in full_samples]
            # attention_mask_list = [sample['attention_mask'] if sample is not None else None for sample in full_samples]
            yield full_sample_list

    def get_full_sampler(self):
        for next_samples in zip_longest(*self.data_generator, fillvalue=None):
            yield next_samples


    def full_iter(self):
        if self.world_size_independent:
            return itertools.islice(
                self.full_sample_packer(), self.rank, None, self.world_size
            )
        else:
            return self.full_sample_packer()


def gpt2_mask_tokenize_fn():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Add additional special tokens so that the vocab size is a multiple of 64
    # https://x.com/karpathy/status/1621578354024677377
    current_size = tokenizer.vocab_size
    diff_multiple_64 = (((current_size // 64) + 1) * 64 - current_size) % 64
    tokens_to_add = diff_multiple_64 - 1  # -1 for the mask token
    additional_special_tokens = [f"<|extra_token_{i}|>" for i in range(tokens_to_add)]
    tokenizer.add_special_tokens(
        {
            "mask_token": "<|mask|>",
            "additional_special_tokens": additional_special_tokens,
        }
    )

    def tokenize_function(examples):
        texts = [text + tokenizer.eos_token for text in examples["text"]]
        batch_encodings = tokenizer(
            texts,
            truncation=False,
            max_length=int(1e10),
        )
        return batch_encodings

    return tokenize_function

# gpt2 specific, need to add additional tokenize functions for different
def glue_tokenize_fn(seq_len: int):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    current_size = tokenizer.vocab_size
    diff_multiple_64 = (((current_size // 64) + 1) * 64 - current_size) % 64
    tokens_to_add = diff_multiple_64 - 2  # mask token, cls token
    additional_special_tokens = [f"<|extra_token_{i}|>" for i in range(tokens_to_add)]
    tokenizer.add_special_tokens(
        {
            "mask_token": "<|mask|>",
            "cls_token": "<|cls|>",
            "additional_special_tokens": additional_special_tokens,
        }
    )
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        examples['text'] = [f"{tokenizer.cls_token} {text}" for text in examples['text']]
        batch_encodings = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=seq_len,
        ) 
        return batch_encodings

    return tokenize_function

def glue_2_sentences_tokenize_fn(seq_len: int):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    current_size = tokenizer.vocab_size
    diff_multiple_64 = (((current_size // 64) + 1) * 64 - current_size) % 64
    tokens_to_add = diff_multiple_64 - 3  # mask token, cls token, sep token
    additional_special_tokens = [f"<|extra_token_{i}|>" for i in range(tokens_to_add)]
    tokenizer.add_special_tokens(
        {
            "mask_token": "<|mask|>",
            "cls_token": "<|cls|>",
            "sep_token": "<|sep|>",
            "additional_special_tokens": additional_special_tokens,
        }
    )
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):        
        examples['text'] = [f"{tokenizer.cls_token} {text1} {tokenizer.sep_token} {text2}"
                            for text1, text2 in zip(examples['text_1'], examples['text_2'])]
        batch_encodings = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=seq_len,
        ) 
        return batch_encodings

    return tokenize_function


def glue_collate_wrapper(examples):
    inputs = [item[0] for item in examples]
    labels = [item[1] for item in examples]
    attention_masks = [item[2] for item in examples]

    collated_inputs = collate_wrapper(inputs)
    collated_labels = torch.tensor(labels, dtype=torch.int64)
    collated_attention_masks = collate_wrapper(attention_masks).bool()

    return collated_inputs, collated_labels, collated_attention_masks


def glue_split_collate_wrapper(examples):
    collated = []
    for example in zip(*examples):
        valid_items = [item for item in example if item is not None]
        if valid_items:
            try:
                collated.append(glue_collate_wrapper(valid_items))
            except Exception as e:
                # logger.error(f"Error collating example: {e}")
                collated.append(None)
        else:
            collated.append(None)
    return collated


class FullIterDataset(IterableDataset):
    def __init__(self, dataset: GlueDataset):
        self.dataset = dataset

    def __iter__(self):
        return self.dataset.full_iter()
