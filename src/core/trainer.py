import os
from attr import define
import torch
import torch.nn.functional as F
from typing import Optional
from torch.utils.data import IterableDataset
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import logging

from src.core.checkpointing import save_checkpoint, step_checkpoint_path
from src.core.metric_loggers import MetricLogger
from src.core.utils import create_batch_fingerprint

logger = logging.getLogger(__name__)

@define(slots=False)
class Trainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    gradient_accumulation_steps: int
    training_state: dict
    n_steps: int
    train_dataloader: IterableDataset
    eval_dataloader: IterableDataset
    metric_logger: MetricLogger
    eval_interval: int
    n_eval_steps: int
    gradient_clipping: Optional[float]
    checkpoint: Optional[dict]
    learning_rate: float
    weight_decay: float
    distributed: Optional[dict]

    def __attrs_post_init__(self):
        self.processed_tokens = self.training_state["processed_tokens"]
        self.start_step = self.training_state["next_step"]
        self.device = next(self.model.parameters()).device
        self.loss_interval_100 = 0.0
        self.eval_iterator = iter(self.eval_dataloader)
        self.step = self.start_step - 1

        if self.start_step > 0:
            n_skip_eval_batches = (
                (self.start_step - 1) // self.eval_interval * self.n_eval_steps
            )
            logger.debug(f"Skipping {n_skip_eval_batches} eval batches")
            for _ in range(n_skip_eval_batches):
                next(self.eval_iterator)

    @property
    def _should_evaluate(self) -> bool:
        return (
            self.eval_interval > 0
            and self.step % self.eval_interval == 0
            and self.step != 0
        )

    @property
    def _should_log_eval_input(self) -> bool:
        return self.step % (self.eval_interval * 100) == 0

    @property
    def _should_save_checkpoint(self) -> bool:
        return (
            self.checkpoint.save.interval > 0
            and (self.step) % self.checkpoint.save.interval == 0
            and self.step != 0
            and self.checkpoint.save.path is not None
        )
    
    @property
    def _should_save_final_checkpoint(self) -> bool:
        return (
            not self._should_save_checkpoint # checkpoint was already saved
            and self.step >= self.n_steps - 1
            and self.checkpoint.save.path is not None
        )

    def train(self):
        for step, batch in zip(
            range(self.start_step, self.n_steps), self.train_dataloader
        ):
            self.step = step
            self.metric_logger.set_step(step)
            self.model.train()
            loss = self.calculate_loss(batch)

            grad_norm = self.clip_gradient()

            self.log_metrics(loss, grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            
            if self._should_save_checkpoint:
                save_checkpoint(
                    checkpoint_path=step_checkpoint_path(self.checkpoint.save.path, self.step),
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    training_state={
                        "next_step": self.step + 1,
                        "run_id": self.training_state["run_id"],
                        "processed_tokens": self.processed_tokens,
                    },
                )

            if self._should_evaluate:
                self.eval()

    def _preprocess_input(self, batch):  # TODO test it
        input_ids = batch[:, :-1].contiguous()
        target_ids = batch[:, 1:].contiguous()

        return input_ids, target_ids

    def calculate_loss(self, batch):
        def _hack_for_python_garbage_collection(input_ids, target_ids):
            """we want to have no reference to model output while backpropagating to allow torch to free memory,
            so we wrap loss calculation in a function"""
            predicted_ids = self.model(input_ids)

            # Tensors should be on the same device for loss calculation #TODO check
            target_ids = target_ids.to(predicted_ids.device)

            mask_loss = F.cross_entropy(
                predicted_ids.flatten(0, -2),
                target_ids.reshape(-1).long(),
                reduction="none",
            )
            loss = mask_loss.mean() / self.gradient_accumulation_steps
            return loss

        losses = []
        for batch_chunk in batch.chunk(self.gradient_accumulation_steps):
            input_ids, target_ids = self._preprocess_input(batch_chunk)
            input_ids = input_ids.to(self.device)
            if self.model.training:
                self._update_processed_tokens(input_ids)

            loss = _hack_for_python_garbage_collection(input_ids, target_ids)
            if self.model.training:
                loss.backward()
            losses.append(loss.item())

        # gloo backend supports only sum reduce operation, therfore we first divide by world size and then sum
        avg_loss = torch.tensor(losses, device=loss.device).sum()
        if dist.is_initialized():
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)

        return avg_loss / float(os.environ["WORLD_SIZE"])

    def eval(self):
        self.model.eval()
        saved_step = self.step
        self.metric_logger.set_step(None)  # disables heavy logging
        losses = []
        eval_fingerprint = []
        with torch.no_grad():
            for _ in range(self.n_eval_steps):
                batch = next(self.eval_iterator)
                batch_fingerprint = create_batch_fingerprint(batch)
                eval_fingerprint.extend(batch_fingerprint)
                batch = batch.to(self.device)
                loss = self.calculate_loss(batch)
                losses.append(loss.item())
                self.metric_logger.flush_accumulated_metrics(self.step)
            avg_loss = torch.tensor(losses).mean()
            self.metric_logger.log("steps/eval/loss", self.step, avg_loss.item())
            self.metric_logger.log(
                "tokens/eval/loss", self.processed_tokens, avg_loss.item()
            )

        if self._should_log_eval_input:
            self.metric_logger.log(
                f"steps/eval/batch", self.step, str(eval_fingerprint)
            )
        
        self.step = saved_step  # Restore step

    def clip_gradient(self):
        if self.gradient_clipping is not None:
            if isinstance(self.model, FSDP):
                return self.model.clip_grad_norm_(self.gradient_clipping)
            else:
                return torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clipping
                )

    def _update_processed_tokens(self, batch):
        self.processed_tokens += batch.numel() * int(os.environ["WORLD_SIZE"])

    def log_metrics(self, loss, grad_norm):
        self.metric_logger.log("step", self.step, self.step)
        self.metric_logger.log("steps/train/loss", self.step, loss.item())
        self.metric_logger.log(
            "steps/train/lr", self.step, (self.scheduler.get_last_lr()[0])
        )
        self.metric_logger.log("steps/train/grad_norm", self.step, grad_norm.item())
        self.metric_logger.log(
            "steps/train/processed_tokens", self.step, self.processed_tokens
        )

        self.metric_logger.log("tokens/train/loss", self.processed_tokens, loss.item())
        self.metric_logger.log(
            "tokens/lr", self.processed_tokens, (self.scheduler.get_last_lr()[0])
        )
        self.metric_logger.log(
            "tokens/train/grad_norm", self.processed_tokens, grad_norm.item()
        )
        self.metric_logger.flush_accumulated_metrics(self.step)
