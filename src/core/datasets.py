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
        max_samples: Optional[int] = None,
    ):
        self.max_samples = max_samples
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
        """Restart the stream.

        Reseeding here is what makes a fresh iterator replay from the beginning, which
        the resume path depends on. `max_samples` caps the stream, turning an otherwise
        endless packer into a fixed, finite set -- an eval set has to be the *same*
        samples every time it is measured, or the number moves for reasons that have
        nothing to do with the model.
        """
        self.rng.seed(self.seed)
        if self.world_size_independent:
            stream = itertools.islice(
                self.sample_packer(), self.rank, None, self.world_size
            )
        else:
            stream = self.sample_packer()
        if self.max_samples is not None:
            stream = itertools.islice(stream, self.max_samples)
        return stream


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
        world_size_independent: bool = False,
        max_samples: Optional[int] = None,
        holdout: Optional[str] = None,
        holdout_rows: int = 262144,
    ):
        """holdout carves a disjoint evaluation slice out of a single directory.

        fineweb ships only a `train` directory -- no validation split was ever
        downloaded -- so eval was pointed at the training data and scored the batches
        the model was training on. There is no second directory to point at, but there
        is no need for one: the last `holdout_fraction` of the rows can be reserved.
        Pass holdout="train" to take everything before the cut and holdout="eval" to
        take everything after it. Both are computed from the same row count, so the two
        are disjoint by construction.

        The reserve is a fixed number of rows rather than a fraction, so the eval set
        does not grow with the corpus: an eval number is only comparable across runs if
        it is measured on the same data, and a fraction of a changing dataset is not.

        holdout=None uses every row, for callers that genuinely want the whole dataset.
        """
        self.holdout = holdout
        self.holdout_rows = holdout_rows
        super().__init__(
            sequence_length,
            tokenize_fn,
            path,
            split,
            seed,
            use_new_sampling_method,
            shuffle,
            world_size_independent,
            max_samples,
        )
        self._load_dataset(path, seed, tokenize_fn, shuffle)

    def _holdout_split(self, hf_dataset):
        if self.holdout is None:
            return hf_dataset
        if self.holdout not in ("train", "eval"):
            raise ValueError(
                f"holdout must be None, 'train' or 'eval', got {self.holdout!r}"
            )
        if self.holdout_rows < 1:
            raise ValueError(f"holdout_rows must be >= 1, got {self.holdout_rows}")

        total = len(hf_dataset)
        if self.holdout_rows >= total:
            raise ValueError(
                f"holdout_rows {self.holdout_rows} leaves no training rows in a "
                f"dataset of {total}"
            )
        cut = total - self.holdout_rows

        if self.holdout == "train":
            selected = hf_dataset.select(range(cut))
        else:
            selected = hf_dataset.select(range(cut, total))
        logger.info(
            f"holdout='{self.holdout}': {len(selected)} of {total} rows (cut at {cut})"
        )
        return selected

    def _load_dataset(self, path, seed, tokenize_fn, shuffle: bool):
        if path is None:
            raise ValueError("Path to dataset must be provided for FineWebEduDataset")
        else:
            logger.debug(f"Loading dataset from path '{path}'")
            hf_dataset = load_from_disk(path)
            hf_dataset = self._holdout_split(hf_dataset)
            hf_dataset = hf_dataset.to_iterable_dataset(
                num_shards=min(self.NUM_SHARDS, len(hf_dataset))
            )

        if not self.world_size_independent:
            hf_dataset = split_dataset_by_node(
                hf_dataset, rank=self.rank, world_size=self.world_size
            )

        if shuffle:
            hf_dataset = hf_dataset.shuffle(buffer_size=self.BUFFER_SIZE, seed=seed)

        self.data_generator = hf_dataset.map(tokenize_fn, batched=True)

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
        max_samples: Optional[int] = None,
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
            max_samples,
        )
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

        self.data_generator = hf_dataset.map(tokenize_fn, batched=True)

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
