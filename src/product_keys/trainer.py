import os
import torch
import torch.nn.functional as F
from attr import define, field
import torch.distributed as dist
from typing import List

from src.core.trainer import Trainer


@define(slots=False)
class MaskedLMTrainer(Trainer):
    """
    A Trainer for Masked Language Modeling (MLM), similar to BERT training.
    """

    masking_percentage: float = field(default=0.15, kw_only=True)
    mask_token_id: int = field(kw_only=True)
    unmaskable_special_tokens: List[int] = field(factory=list, kw_only=True)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        # Use -100 as the ignore index for the loss function
        self.ignore_index = -100

    def _preprocess_and_mask_input(self, batch):
        """
        Prepares a batch for MLM by masking a percentage of tokens.
        """
        # remove last token for compatibility with next token prediction
        # here we do not shift labels, as we want to predict the masked tokens in place
        input_ids = batch[:, :-1].contiguous()
        labels = batch[:, :-1].contiguous()

        # Determine which tokens to mask based on the masking percentage.
        prob = torch.full(labels.shape, self.masking_percentage, device=labels.device)
        masked_indices = torch.bernoulli(prob).bool()

        # We should not mask special tokens like [CLS], [SEP], etc.
        for special_token_id in self.unmaskable_special_tokens:
            masked_indices[input_ids == special_token_id] = False

        labels[~masked_indices] = self.ignore_index
        input_ids[masked_indices] = self.mask_token_id

        return input_ids, labels

    def calculate_loss(self, batch):
        """
        Calculates the MLM loss.
        The loss is calculated only for the masked tokens.
        """

        def _mlm_loss_calculation(input_ids, target_ids):
            """
            Inner function to calculate loss, allowing for Python's garbage collector
            to free memory by keeping model output in a local scope.
            """
            predicted_logits = self.model(input_ids)
            target_ids = target_ids.to(predicted_logits.device)

            loss = F.cross_entropy(
                predicted_logits.view(-1, predicted_logits.size(-1)),
                target_ids.view(-1).long(),
                ignore_index=self.ignore_index,
                reduction="mean",
            )
            return loss / self.gradient_accumulation_steps

        losses = []
        for batch_chunk in batch.chunk(self.gradient_accumulation_steps):
            input_ids, target_ids = self._preprocess_and_mask_input(batch_chunk)
            input_ids = input_ids.to(self.device)

            if self.model.training:
                self._update_processed_tokens(input_ids)

            loss = _mlm_loss_calculation(input_ids, target_ids)
            if self.model.training:
                loss.backward()
            losses.append(loss.item())

        # gloo backend supports only sum reduce operation, therfore we first divide by world size and then sum
        avg_loss = torch.tensor(losses, device=loss.device).sum()
        if dist.is_initialized():
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)

        return avg_loss / float(os.environ["WORLD_SIZE"])
