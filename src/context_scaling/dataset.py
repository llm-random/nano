from src.core.datasets import AbstractDataset
import logging

logger = logging.getLogger(__name__)


class WholeDocumentDataset(AbstractDataset):
    """Yield sequence_length tokens from exactly one document."""

    def __init__(self, base_dataset: AbstractDataset):
        self.base_dataset = base_dataset

        # copy the minimal state AbstractDataset.__iter__ expects
        self.world_size = base_dataset.world_size
        self.rank = base_dataset.rank
        self.rng = base_dataset.rng
        self.seed = base_dataset.seed
        self.sequence_length = base_dataset.sequence_length
        self.world_size_independent = base_dataset.world_size_independent

        # optional: keep these if other code expects them to exist
        self.shuffle = getattr(base_dataset, "shuffle", None)
        self.use_new_sampling_method = getattr(
            base_dataset, "use_new_sampling_method", None
        )
        self.tokenize_fn = getattr(base_dataset, "tokenize_fn", None)
        self.path = getattr(base_dataset, "path", None)
        self.split = getattr(base_dataset, "split", None)

    def get_infinite_sampler(self):
        return self.base_dataset.get_infinite_sampler()

    def sample_packer(self):
        for sample in self.get_infinite_sampler():
            tokens = sample["input_ids"]
            if len(tokens) < self.sequence_length:
                logger.info("WARNING: skipping a document, seq_len too large")
                continue
            start = self.rng.randint(0, len(tokens) - self.sequence_length)
            yield tokens[start : start + self.sequence_length]
