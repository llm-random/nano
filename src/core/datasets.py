import os
from typing import Callable
import torch
from typing import Optional, List
from datasets import load_from_disk
import itertools
import numpy as np
import random
from torch.utils.data import IterableDataset, DataLoader
from transformers import GPT2TokenizerFast, PreTrainedTokenizerBase
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
import logging

logger = logging.getLogger(__name__)

def take_circular(iterable, start, stop):
    cycle = itertools.cycle(iterable)
    return itertools.islice(cycle, start, stop)


def _process_document(document, encode_fn, eot_str):
    """
    Used for processing the dataset in the C4Dataset class.
    It should be lambda inside a map function, but apparently it is not possible to pickle it.
    """
    return {
        "tokens": encode_fn(
            document["text"] + eot_str,
            truncation=False,
            max_length=int(1e10),
        )
    }


class C4Dataset(IterableDataset):
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
        path: Optional[str] = None,
        split: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        seed: Optional[int] = None,
        eot_str: str = "<|endoftext|>",
        use_new_sampling_method: bool = True,
        shuffle: bool = True,
        world_size_independent: bool = False,
    ):
        self.world_size = int(os.environ.get("WORLD_SIZE"))
        self.rank = int(os.environ.get("RANK"))
        self.use_new_sampling_method = use_new_sampling_method
        self.world_size_independent = world_size_independent
        if tokenizer is None:
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self._load_dataset(path, split, seed, tokenizer, eot_str, shuffle)
        self.sequence_length = sequence_length
        self.seed = seed
        self.rng = random.Random(seed)

    def _load_dataset(self, path, split, seed, tokenizer, eot_str, shuffle: bool):
        if path is None:
            logger.debug(
                f"Loading 'allenai/c4' dataset from HuggingFace with split={split}"
            )
            hf_dataset = load_dataset(
                "allenai/c4",
                "en",
                split=split,
                streaming=True,
                trust_remote_code=True,
            )
        else:
            logger.debug(f"Loading dataset from path '{path}'")
            hf_dataset = load_from_disk(path)
            hf_dataset = hf_dataset.to_iterable_dataset(num_shards=self.NUM_SHARDS)

        if not self.world_size_independent:
            hf_dataset = split_dataset_by_node(
                hf_dataset, rank=self.rank, world_size=self.world_size
            )

        if shuffle:
            hf_dataset = hf_dataset.shuffle(buffer_size=self.BUFFER_SIZE, seed=seed)

        self.data_generator = hf_dataset.map(
            _process_document,
            fn_kwargs={"encode_fn": tokenizer.encode, "eot_str": eot_str},
        )

    def get_infinite_sampler(self):
        epoch = 0
        while True:
            self.data_generator.set_epoch(epoch)
            for next_sample in self.data_generator:
                yield next_sample
            epoch += 1

    def sample_packer(self):
        buffer: List[int] = []
        sampler = iter(self.get_infinite_sampler())
        if self.use_new_sampling_method:

            while True:
                sample = next(sampler)["tokens"]

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
                tokens = next(sampler)["tokens"]
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


def collate_wrapper(examples):
    return torch.from_numpy(np.array(examples))


def get_dataloader(
    dataset_type: str,
    dataset_path: str,
    dataset_split: str,
    total_batch_size: int,
    sequence_length: int,
    num_workers: int,
    seed: int,
    shuffle: bool,
    use_new_sampling_method: bool,
    world_size_independent: bool,
    collate_fn: Callable = collate_wrapper,
):
    world_size = int(os.environ["WORLD_SIZE"])
    batch_size_per_device = total_batch_size // world_size
    logger.debug(f"Batch size per device: {batch_size_per_device}")
    logger.debug(f"Total: {total_batch_size}")
    if dataset_type == "c4":
        dataset = C4Dataset(
            sequence_length=sequence_length + 1,
            split=dataset_split,
            path=dataset_path,
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
    else:
        raise ValueError(f"Unsupported dataset type: '{dataset_type}'")

    return dataloader

