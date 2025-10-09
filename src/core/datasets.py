import os
from typing import Callable
import torch
from typing import Optional, List
from datasets import load_from_disk
import itertools
import numpy as np
import random
from torch.utils.data import IterableDataset, DataLoader
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
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", add_bos_token=True, add_eos_token=True, legacy=False)
    def tokenize_function(examples):
        batch_encodings = tokenizer(
            examples["text"],
            truncation=False,
            max_length=int(1e10),
        )
        return batch_encodings
    return tokenize_function

class AbstractDataset(IterableDataset):

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

class FineWebEduDataset(AbstractDataset):

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
        world_size_independent: bool = False
    ):
        super().__init__(sequence_length, tokenize_fn, path, split, seed, use_new_sampling_method, shuffle,
                         world_size_independent)
        self._load_dataset(path, seed, tokenize_fn, shuffle)

    def _load_dataset(self, path, seed, tokenize_fn, shuffle: bool):
        if path is None:
            raise ValueError("Path to dataset must be provided for FineWebEduDataset")
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
            tokenize_fn,
            batched=True
        )

    def get_infinite_sampler(self):
        epoch = 0
        while True:
            self.data_generator.set_epoch(epoch)
            for next_sample in self.data_generator:
                yield next_sample
            epoch += 1

class C4Dataset(AbstractDataset):
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
        super().__init__(sequence_length, tokenize_fn, path, split, seed, use_new_sampling_method, shuffle,
                         world_size_independent)
        self._load_dataset(path, split, seed, tokenize_fn, shuffle)

    def _load_dataset(self, path, split, seed, tokenize_fn, shuffle: bool):
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
            tokenize_fn,
            batched=True
        )

    def get_infinite_sampler(self):
        epoch = 0
        while True:
            self.data_generator.set_epoch(epoch)
            for next_sample in self.data_generator:
                yield next_sample
            epoch += 1


def collate_wrapper(examples):
    return torch.from_numpy(np.array(examples))


def download_dummy_dataset(dataset_type: str = "c4", num_samples: int = 100, output_dir: str = "data"):
    """
    Download a small portion of a dataset for local debugging.

    Args:
        dataset_type: Type of dataset to download ("c4" or "fineweb-edu")
        num_samples: Number of samples to download
        output_dir: Directory to save the dataset
    """
    import os

    if os.path.exists(output_dir):
        logger.info(f"Dataset already exists at {output_dir}, skipping download")
        return

    logger.info(f"Downloading {num_samples} samples of {dataset_type} to {output_dir}")

    if dataset_type == "c4":
        dataset = load_dataset(
            "allenai/c4",
            "en",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    elif dataset_type == "fineweb-edu":
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    # Take only num_samples
    dataset = dataset.take(num_samples)

    # Convert to regular dataset and save
    from datasets import Dataset
    samples = list(dataset)
    dataset_dict = Dataset.from_dict({k: [sample[k] for sample in samples] for k in samples[0].keys()})
    dataset_dict.save_to_disk(output_dir)

    logger.info(f"Successfully saved {num_samples} samples to {output_dir}")


def get_dataloader(
    dataset_type: str,
    dataset_path: str,
    dataset_split: str,
    tokenize_fn: str,
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
            tokenize_fn=tokenize_fn,
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
    elif dataset_type == "fineweb-edu":
        dataset = FineWebEduDataset(
            sequence_length=sequence_length + 1,
            split=dataset_split,
            tokenize_fn=tokenize_fn,
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

