import os
from typing import Callable
import torch
from typing import Optional, List
from datasets import load_from_disk
import itertools
import numpy as np
import random
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
import logging

logger = logging.getLogger(__name__)


def take_circular(iterable, start, stop):
    cycle = itertools.cycle(iterable)
    return itertools.islice(cycle, start, stop)


def get_tokenize_fn(model_name: str):
    """
    Factory function to create a tokenize function for a given model.

    Args:
        model_name: The HuggingFace model name/path to load the tokenizer from.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        add_bos_token=True,
        add_eos_token=True,
        legacy=False,
    )

    # Lazy initialization to avoid using tokenizer before DataLoader forks workers
    manual_add_eos = None

    def tokenize_function(examples):
        nonlocal manual_add_eos

        # Check if tokenizer actually respects add_eos_token (only once)
        # Some tokenizers (e.g., GPT-2) silently ignore this option
        if manual_add_eos is None:
            _test_tokens = tokenizer("test")["input_ids"]
            manual_add_eos = tokenizer.eos_token_id is not None and _test_tokens[-1] != tokenizer.eos_token_id

        if "text" in examples and "content" in examples:
            raise KeyError("Both 'text' and 'content' found in examples.")
        elif "text" in examples:
            source_col = "text"
        elif "content" in examples:
            source_col = "content"
        else:
            raise KeyError(f"Neither 'text' nor 'content' found. Available keys: {list(examples.keys())}")

        texts = examples[source_col]

        # Manually add EOS token for tokenizers that don't support it natively (e.g., GPT-2)
        if manual_add_eos:
            texts = [text + tokenizer.eos_token for text in texts]

        batch_encodings = tokenizer(
            texts,
            truncation=False,
            max_length=int(1e10),
        )
        return batch_encodings

    return tokenize_function


class GenericDataset(IterableDataset):
    """
    world_size_independent - if True, we take the whole dataset and take every 'mod rank' element. If world_size == 1 it does not matter.
    shuffle - if True, we shuffle the dataset independently on each rank part (unless world_size_independent is True)


    world_size_independent should be 'True' for tests and eval
    """

    BUFFER_SIZE = 10000
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
    ):
        self.world_size = int(os.environ.get("WORLD_SIZE"))
        self.rank = int(os.environ.get("RANK"))
        self.rng = random.Random(seed)
        self.sequence_length = sequence_length
        self.tokenize_fn = tokenize_fn
        self.path = path
        self.split = split
        self.seed = seed
        self.use_new_sampling_method = use_new_sampling_method
        self.shuffle = shuffle
        self.world_size_independent = world_size_independent
        self.data_generator = None
        self._load_dataset(path, split, seed, tokenize_fn, shuffle)

    def _load_hf_dataset(self, path, split):
        logger.debug(f"Loading dataset from path '{path}'")
        hf_dataset = load_from_disk(path)
        hf_dataset = hf_dataset.to_iterable_dataset(num_shards=self.NUM_SHARDS)
        return hf_dataset

    def _load_dataset(self, path, split, seed, tokenize_fn, shuffle: bool):
        hf_dataset = self._load_hf_dataset(path, split)

        if not self.world_size_independent:
            hf_dataset = split_dataset_by_node(
                hf_dataset, rank=self.rank, world_size=self.world_size
            )

        if shuffle:
            hf_dataset = hf_dataset.shuffle(buffer_size=self.BUFFER_SIZE, seed=seed)

        self.data_generator = hf_dataset.map(tokenize_fn, batched=True)

    def sample_packer(self):
        buffer: List[int] = []
        sampler = iter(self.get_infinite_sampler())
        if self.use_new_sampling_method:

            while True:
                sample = next(sampler)["input_ids"]

                if len(buffer) == 0:
                    rand_num = self.rng.randint(0, len(sample) - 1)
                    sample = sample[rand_num:]

                buffer.extend(sample)

                if len(buffer) >= self.sequence_length:
                    yield buffer[: self.sequence_length]
                    buffer = []
        else:
            document_lengths: List[int] = []
            while True:
                tokens = next(sampler)["input_ids"]
                buffer.extend(tokens)

                document_lengths.append(len(tokens))
                if (
                        sum(document_lengths) - max(document_lengths)
                ) > self.sequence_length:
                    sample_start = self.rng.randint(0, len(buffer) - 1)
                    sample_end = sample_start + self.sequence_length
                    input_ids = list(take_circular(buffer, sample_start, sample_end))
                    yield input_ids
                    buffer, document_lengths = [], []

    def __iter__(self):
        self.rng.seed(self.seed)
        if self.world_size_independent:
            return itertools.islice(
                self.sample_packer(), self.rank, None, self.world_size
            )
        else:
            return self.sample_packer()

    def get_infinite_sampler(self):
        epoch = 0
        while True:
            self.data_generator.set_epoch(epoch)
            for next_sample in self.data_generator:
                yield next_sample
            epoch += 1


class MixtureOfDatasets(IterableDataset):
    BUFFER_SIZE = 10000
    NUM_SHARDS = 64

    def __init__(
            self,
            sequence_length,
            tokenize_fn: Callable,
            paths: Optional[List[str]] = None,
            weights: Optional[List[float]] = None,
            split: Optional[str] = None,
            seed: Optional[int] = None,
            use_new_sampling_method: bool = True,
            shuffle: bool = True,
            world_size_independent: bool = False,
    ):
        self.world_size = int(os.environ.get("WORLD_SIZE"))
        self.rank = int(os.environ.get("RANK"))
        self.rng = random.Random(seed)
        self.sequence_length = sequence_length
        self.tokenize_fn = tokenize_fn
        self.paths = paths
        self.weights = weights
        self.split = split
        self.seed = seed
        self.use_new_sampling_method = use_new_sampling_method
        self.shuffle = shuffle
        self.world_size_independent = world_size_independent
        self.data_generator = None
        self.datasets = [GenericDataset(
            sequence_length=sequence_length,
            split=split,
            tokenize_fn=tokenize_fn,
            path=path,
            seed=seed,
            use_new_sampling_method=use_new_sampling_method,
            shuffle=shuffle,
            world_size_independent=world_size_independent,
        ) for path in paths]

    def __iter__(self):
        rng = random.Random(self.seed)
        dataset_iterators = [iter(dataset) for dataset in self.datasets]
        used = np.zeros(len(self.datasets), dtype=np.int64)
        step = 0

        while True:
            step += 1
            expected = (step * np.array(self.weights)).astype(np.int64)
            diffs = expected - used
            max_diff = np.max(diffs)
            candidate_indices = [i for i, diff in enumerate(diffs) if diff == max_diff]
            chosen_index = rng.choice(candidate_indices)
            used[chosen_index] += 1
            try:
                sample = next(dataset_iterators[chosen_index])
            except StopIteration:
                dataset_iterators[chosen_index] = iter(self.datasets[chosen_index])
                sample = next(dataset_iterators[chosen_index])
            if self.rank == 0:
                logger.debug(f"{self.split}, step {step}: Chose dataset {self.paths[chosen_index]}")
            yield sample


def collate_wrapper(examples):
    return torch.from_numpy(np.array(examples))


def get_mixture_of_datasets_dataloader(datasets: list[dict], dataset_split, tokenize_fn, total_batch_size,
                                       sequence_length,
                                       num_workers, seed, shuffle, use_new_sampling_method, world_size_independent,
                                       collate_fn: Callable = collate_wrapper):
    # print(datasets)
    dataset_paths = [d["path"] for d in datasets]
    dataset_weights = [d["weight"] for d in datasets]

    # Validate paths exist
    for path in dataset_paths:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Directory {path} not found")

    assert abs(sum(dataset_weights) - 1) < 1e-6, f"Dataset weights must sum to 1, current sum: {sum(dataset_weights)}"
    world_size = int(os.environ["WORLD_SIZE"])
    batch_size_per_device = total_batch_size // world_size
    logger.debug(f"Batch size per device: {batch_size_per_device}")
    logger.debug(f"Total: {total_batch_size}")

    dataset = MixtureOfDatasets(
            sequence_length=sequence_length + 1,
            split=dataset_split,
            tokenize_fn=tokenize_fn,
            paths=dataset_paths,
            weights=dataset_weights,
            seed=seed,
            use_new_sampling_method=use_new_sampling_method,
            shuffle=shuffle,
            world_size_independent=world_size_independent,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size_per_device,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=num_workers,
    )

    return dataloader

