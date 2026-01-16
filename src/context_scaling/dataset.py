import os
from typing import Callable
import torch
from typing import Optional, List
from datasets import load_from_disk
import itertools
import numpy as np
import random
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from transformers import GPT2TokenizerFast, AutoTokenizer
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
import logging

logger = logging.getLogger(__name__)


def take_circular(iterable, start, stop):
    cycle = itertools.cycle(iterable)
    return itertools.islice(cycle, start, stop)


def gpt2_tokenize_fn():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def tokenize_function(examples):
        texts = [text + tokenizer.eos_token for text in examples["text"]]
        batch_encodings = tokenizer(
            texts,
            truncation=False,
            max_length=int(1e10),
        )
        return batch_encodings

    return tokenize_function


def llama_tokenize_fn():
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B", add_bos_token=True, add_eos_token=True, legacy=False
    )

    def tokenize_function(examples):
        batch_encodings = tokenizer(
            examples["text"],
            truncation=False,
            max_length=int(1e10),
        )
        return batch_encodings

    return tokenize_function


def smollm_135_tokenize_fn():
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolLM-135M",
        add_bos_token=True,
        add_eos_token=True,
        legacy=False,
    )

    def tokenize_function(examples):
        batch_encodings = tokenizer(
            examples["text"],
            truncation=False,
            max_length=int(1e10),
        )
        return batch_encodings

    return tokenize_function


def smollm_360_tokenize_fn():
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolLM-360M",
        add_bos_token=True,
        add_eos_token=True,
        legacy=False,
    )

    def tokenize_function(examples):
        batch_encodings = tokenizer(
            examples["text"],
            truncation=False,
            max_length=int(1e10),
        )
        return batch_encodings

    return tokenize_function


def smollm_1700_tokenize_fn():
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolLM-1.7B",
        add_bos_token=True,
        add_eos_token=True,
        legacy=False,
    )

    def tokenize_function(examples):
        batch_encodings = tokenizer(
            examples["text"],
            truncation=False,
            max_length=int(1e10),
        )
        return batch_encodings

    return tokenize_function


def get_dataset(
    path: str,  # hydra
    load_dataset_fn: Callable,  # hydra
    shuffle: bool,  # hydra
    seed: int,
    tokenize_fn: Callable,
    num_shards: int,
    world_size_independent: bool,
    rank: int,
    world_size: int,
    buffer_size: int,
):
    if path is None:
        logger.debug(f"Loading with custom function (possibly streaming from HF)")
        hf_dataset = load_dataset_fn()
    else:
        logger.debug(f"Loading dataset from path '{path}'")
        hf_dataset = load_from_disk(path)
        hf_dataset = hf_dataset.to_iterable_dataset(num_shards=num_shards)

    if not world_size_independent:
        hf_dataset = split_dataset_by_node(hf_dataset, rank=rank, world_size=world_size)

    if shuffle:
        hf_dataset = hf_dataset.shuffle(buffer_size=buffer_size, seed=seed)

    return hf_dataset.map(tokenize_fn, batched=True)


def new_packer(
    get_infinite_sampler: Callable,
    seed: int,
    sequence_length: int,
):
    buffer: List[int] = []
    sampler = iter(get_infinite_sampler())
    rng = random.Random(seed)

    while True:
        sample = next(sampler)["input_ids"]

        if len(buffer) == 0:
            rand_num = rng.randint(0, len(sample) - 1)
            sample = sample[rand_num:]

        buffer.extend(sample)

        if len(buffer) >= sequence_length:
            yield buffer[:sequence_length]
            buffer = []


def long_context_packer(
    get_infinite_sampler: Callable,
    seed: int,
    sequence_length: int,
):
    rng = random.Random(seed)

    for sample in get_infinite_sampler():
        tokens = sample["input_ids"]
        if len(tokens) < sequence_length:
            continue
        start = rng.randint(0, len(tokens) - sequence_length)
        yield tokens[start : start + sequence_length]


def old_packer(
    get_infinite_sampler: Callable,
    seed: int,
    sequence_length: int,
):
    buffer: List[int] = []
    sampler = iter(get_infinite_sampler())
    rng = random.Random(seed)

    document_lengths: List[int] = []
    while True:
        tokens = next(sampler)["input_ids"]
        buffer.extend(tokens)

        document_lengths.append(len(tokens))
        if (sum(document_lengths) - max(document_lengths)) > sequence_length:
            sample_start = rng.randint(0, len(buffer) - 1)
            sample_end = sample_start + sequence_length
            input_ids = list(take_circular(buffer, sample_start, sample_end))
            yield input_ids
            buffer, document_lengths = [], []


class AbstractDataset(IterableDataset):

    BUFFER_SIZE = 10000
    NUM_SHARDS = 64

    def __init__(
        self,
        sequence_length,
        tokenize_fn: Callable,
        load_dataset_fn: Callable,
        sample_packer_fn: Callable,
        seed: Optional[int] = None,
        use_new_sampling_method: bool = True,
        world_size_independent: bool = False,
    ):
        self.world_size = int(os.environ.get("WORLD_SIZE"))
        self.rank = int(os.environ.get("RANK"))
        self.rng = random.Random(seed)
        self.sequence_length = sequence_length
        self.tokenize_fn = tokenize_fn
        self.seed = seed
        self.use_new_sampling_method = use_new_sampling_method
        self.world_size_independent = world_size_independent
        self.sample_packer_fn = sample_packer_fn

        self.data_generator = load_dataset_fn(
            seed=self.seed,
            tokenize_fn=self.tokenize_fn,
            num_shards=self.NUM_SHARDS,
            world_size_independent=self.world_size_independent,
            rank=self.rank,
            world_size=self.world_size,
            buffer_size=self.BUFFER_SIZE,
        )

    def _get_effective_seed(self):
        worker_info = get_worker_info()
        if worker_info is None:
            return self.seed
        return self.seed + worker_info.id

    def sample_packer(self):
        effective_seed = self._get_effective_seed()
        yield from self.sample_packer_fn(
            get_infinite_sampler=self.get_infinite_sampler,
            seed=effective_seed,
            sequence_length=self.sequence_length,
        )

    def __iter__(self):
        effective_seed = self._get_effective_seed()
        self.rng.seed(effective_seed)
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


def collate_wrapper(examples):
    return torch.from_numpy(np.array(examples))


def get_dataloader(
    dataset: AbstractDataset,
    total_batch_size: int,
    num_workers: int,
    collate_fn: Callable = collate_wrapper,
):
    world_size = int(os.environ["WORLD_SIZE"])
    batch_size_per_device = total_batch_size // world_size
    logger.debug(f"Batch size per device: {batch_size_per_device}")
    logger.debug(f"Total: {total_batch_size}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size_per_device,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=num_workers,
    )

    return dataloader
