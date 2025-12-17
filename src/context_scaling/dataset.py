from src.core.datasets import AbstractDataset, get_dataloader


class WholeDocumentDataset(AbstractDataset):
    """Dataset wrapper that yields sequences from single documents only.

    Each yielded sequence comes from exactly one document:
    - If document >= sequence_length: pick a random consecutive chunk
    - If document < sequence_length: skip it
    """

    def __init__(self, base_dataset: AbstractDataset):
        self.__dict__.update(base_dataset.__dict__)
        self.base_dataset = base_dataset

    def get_infinite_sampler(self):
        return self.base_dataset.get_infinite_sampler()

    def sample_packer(self):
        """Yield sequence_length tokens from single documents."""
        for sample in self.get_infinite_sampler():
            tokens = sample["input_ids"]

            if len(tokens) < self.sequence_length:
                continue

            start = self.rng.randint(0, len(tokens) - self.sequence_length)
            yield tokens[start : start + self.sequence_length]

def wrap_dataloader(
    get_dataloader_fun
):
    dataset = get_dataloader_fun
    wrapped_dataset = WholeDocumentDataset(
        base_dataset=dataset
    )
    return wrapped_dataset
