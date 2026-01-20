import logging
import os
import torch
from src.core.checkpointing import step_checkpoint_path
from src.core.trainer import Trainer
from attr import define
import torch.distributed.checkpoint as dcp

logger = logging.getLogger(__name__)


@define(slots=False)
class PCTrainer(Trainer):
    only_compress_model_gradient_clipping: bool

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if type(self.optimizer) is list:
            self.block_optimizers = self.optimizer[1:]
            self.block_schedulers = self.scheduler[1:]
            self.optimizer = self.optimizer[0]
            self.scheduler = self.scheduler[0]
        else:
            self.block_optimizers = None
            self.block_schedulers = None

    def train(self):
        for step, batch in zip(
            range(self.start_step, self.n_steps), self.train_dataloader
        ):
            self.step = step
            self.metric_logger.set_step(step)
            self.model.train()

            self.model.prepare_compressed_weights()
            loss = self.calculate_loss(batch)

            if self.only_compress_model_gradient_clipping:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clipping
                )
                self.model.pass_gradient_to_projections(
                    self.block_optimizers, self.block_schedulers, False
                )
            else:
                grad_norm = self.model.pass_gradient_to_projections(
                    self.block_optimizers, self.block_schedulers, self.gradient_clipping
                )
                torch.nn.utils.clip_grads_with_norm_(
                    self.model.parameters(), self.gradient_clipping, grad_norm
                )

            self.log_metrics(loss, grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            if self._should_save_checkpoint:
                self.save_checkpoint()

            if self._should_save_final_checkpoint:
                self.save_checkpoint()

    def save_checkpoint(self):
        checkpoint_folder = step_checkpoint_path(self.checkpoint.save.path, self.step)
        dcp.save(self.model.state_dict(), checkpoint_id=f"{checkpoint_folder}/model")
        dcp.save(
            self.optimizer.state_dict(), checkpoint_id=f"{checkpoint_folder}/optimizer"
        )

        if self.block_optimizers is not None:
            for idx, block_optimizer in enumerate(self.block_optimizers):
                dcp.save(
                    block_optimizer.state_dict(),
                    checkpoint_id=f"{checkpoint_folder}/block_optimizer_{idx}",
                )
        # TODO: add scheduler saving
        logger.info(f"Saved checkpoint after step {self.step}")
