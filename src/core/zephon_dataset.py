"""Training data streamed with zephon (https://github.com/datologyai/zephon).

FineWebEduDataset reads a copy of fineweb saved with `save_to_disk`. This reads the
parquet shards straight from the Hugging Face hub instead, caches them on local disk,
and tokenizes and packs on the fly. What zephon adds over a hand-rolled stream is that
the order is fixed by the seed and the number of *lanes* (`canonical_replicas`), not by
the number of GPUs: a run on 4 GPUs and its resume on 2 see the same global batches.

How the lanes line up with nano's batches: one zephon batch is one lane's share of an
optimizer step, `total_batch_size // canonical_replicas` rows. A rank owning k lanes
gets k of them per step, round-robin, which is exactly its DataLoader batch of
`total_batch_size // world_size` rows -- so every step consumes each lane once and the
global batch is the same at any world size that divides canonical_replicas.

Three things zephon 0.2.1 does not do on its own, set up in `_install_zephon_fixes`:
- `persist_state`: CacheOptions has no such field and the store never passes it to the
  CacheManager, so every new process wipes the shard cache. Forced on here.
- A shard index for hf:// data. Without one, every start opens all 2410 parquet footers
  of fineweb-edu (5-8 minutes, and enough requests to get HTTP 429 from the hub). The
  index can't live in the hub repo, so it is a local file (`index_dir`), built once by
  `python -m src.core.zephon_dataset build-index <hf_path> <index_dir>`.
- `hf_path` should be the frozen URI (commit-pinned, `~original`). Resolving a plain
  `hf://org/name/config/split` needs datasets>=4.4, and nano pins 3.1; a frozen URI is
  used as is, with no resolution and no `datasets` import.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)

DATASET_NAME = "fineweb_edu"  # part of the zephon cache fingerprint; keep it stable


def _index_file(index_dir: str, dataset_root: str) -> Path:
    key = hashlib.sha256(dataset_root.rstrip("/").encode()).hexdigest()[:16]
    return Path(index_dir) / f"{key}.json"


def _install_zephon_fixes(index_dir: Optional[str]) -> None:
    from zephon._internal.io.resolvers.cache.manager import CacheManager
    import zephon._internal.io.formats.parquet as zephon_parquet
    import zephon.io.dataset as zephon_dataset

    if not getattr(CacheManager, "_nano_persist", False):
        init = CacheManager.__init__

        def init_persist(self, *args, **kwargs):
            kwargs["persist_state"] = True
            init(self, *args, **kwargs)

        CacheManager.__init__ = init_persist
        CacheManager._nano_persist = True

    if index_dir is not None and not getattr(zephon_parquet, "_nano_index", False):
        find = zephon_parquet.find_and_load_index

        def find_with_local_index(dir_path, storage):
            f = _index_file(index_dir, dir_path)
            if f.exists():
                return json.loads(f.read_text())
            logger.warning(
                f"No local zephon index {f} for {dir_path}: every start will scan all "
                f"shard footers. Build it once with `python -m src.core.zephon_dataset "
                f"build-index {dir_path} {index_dir}`."
            )
            return find(dir_path, storage)

        zephon_parquet.find_and_load_index = find_with_local_index
        zephon_dataset._find_and_load_index = find_with_local_index
        zephon_parquet._nano_index = True


class ZephonDataset(IterableDataset):
    """Yields packed rows of `sequence_length` token ids for this rank.

    The trainer shifts by one itself (`batch[:, :-1]` / `batch[:, 1:]`), so rows are
    `sequence_length` long, like FineWebEduDataset's. Documents are tokenized with BOS
    and EOS and packed back to back (`wrap`): a document that does not fit continues in
    the next row, nothing is dropped or padded.

    Use it with num_workers: 0. zephon runs its own fetch and tokenize threads; under
    DataLoader workers it would split the lanes again by worker.
    """

    def __init__(
        self,
        sequence_length: int,
        total_batch_size: int,
        hf_path: str,
        cache_dir: str,
        tokenizer_id: str = "meta-llama/Llama-3.1-8B",
        seed: int = 42,
        canonical_replicas: int = 4,
        index_dir: Optional[str] = None,
        cache_limit_gb: int = 200,
        tokenize_parallelism: int = 8,
        prefetch_parallelism: int = 8,
        tokenize_fn=None,
    ):
        # tokenize_fn is accepted because the trainer configs set it on every train
        # dataset; zephon tokenizes inside its own pipeline from tokenizer_id instead.
        # llama_tokenize_fn is meta-llama/Llama-3.1-8B with BOS and EOS, the default here.
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        if canonical_replicas % self.world_size != 0:
            raise ValueError(
                f"canonical_replicas={canonical_replicas} must be a multiple of "
                f"world_size={self.world_size}, or ranks get uneven shares of a step."
            )
        if total_batch_size % canonical_replicas != 0:
            raise ValueError(
                f"total_batch_size={total_batch_size} must be a multiple of "
                f"canonical_replicas={canonical_replicas}: one zephon batch is one "
                f"lane's share of a step."
            )
        self.sequence_length = sequence_length
        self.rows_per_lane = total_batch_size // canonical_replicas
        self.hf_path = hf_path
        self.cache_dir = cache_dir
        self.tokenizer_id = tokenizer_id
        self.seed = seed
        self.canonical_replicas = canonical_replicas
        self.index_dir = index_dir
        self.cache_limit_gb = cache_limit_gb
        self.tokenize_parallelism = tokenize_parallelism
        self.prefetch_parallelism = prefetch_parallelism

    def _pipeline(self):
        from zephon import Pipeline
        from zephon.io import CacheOptions, Dataset, StoreOptions
        from zephon.work import MixtureSpec, StaticMixtureWorkSource

        _install_zephon_fixes(self.index_dir)
        dataset = Dataset.from_path(DATASET_NAME, self.hf_path)
        work_source = StaticMixtureWorkSource(
            datasets=[dataset], mixture=MixtureSpec({DATASET_NAME: 1.0}), seed=self.seed
        )
        pipeline = (
            Pipeline(work_source)
            .prefetch(buffer_size=4096, parallelism=self.prefetch_parallelism)
            .tokenize(
                tokenizer_id=self.tokenizer_id,
                field="text",
                parallelism=self.tokenize_parallelism,
            )
            .pack_flat(max_length=self.sequence_length, algorithm="wrap")
            .batch(microbatch_size=self.rows_per_lane)
        )
        return pipeline.options(
            io_options=StoreOptions(
                cache=CacheOptions(
                    enabled=True,
                    root=str(Path(self.cache_dir) / "cache"),
                    limit_bytes=self.cache_limit_gb << 30,
                )
            ),
            world_size=self.world_size,
            global_rank=self.rank,
            canonical_replicas=self.canonical_replicas,
            aggregate_dir=str(Path(self.cache_dir) / "aggregate"),
        )

    def __iter__(self):
        """Restart the stream from the beginning (the resume path skips from here)."""
        for batch in self._pipeline():
            input_ids = batch.to_training(return_labels=False)["input_ids"]
            for row in np.asarray(input_ids, dtype=np.int64):
                yield row


def build_index(hf_path: str, index_dir: str) -> Path:
    """Scan the shard footers once and write the index `ZephonDataset` reads."""
    from zephon._internal.io.storage import RouterStorageBackend
    from zephon.build_index import build_index as zephon_build_index

    root = RouterStorageBackend().canonical_root(hf_path.rstrip("/"), fmt=None)
    out = _index_file(index_dir, root)
    out.parent.mkdir(parents=True, exist_ok=True)
    zephon_build_index("parquet", root, output_path=out)
    return out


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4 or sys.argv[1] != "build-index":
        sys.exit("usage: python -m src.core.zephon_dataset build-index <hf_path> <index_dir>")
    print(build_index(sys.argv[2], sys.argv[3]))
