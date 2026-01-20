import logging
import os
import torch
import torch.distributed.checkpoint as dcp
from attr import define
from src.core.checkpointing import step_checkpoint_path
from src.core.trainer_distillation import TrainerDistillation

logger = logging.getLogger(__name__)

@define(slots=False)
class PCDistillationTrainer(TrainerDistillation):
    """Works only for mem_eff_pc model with distillation"""

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

            if self._should_evaluate:
                self.eval()


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

        # TODO: add scheduler saving/loading, saving below:
        # if self.block_schedulers is not None:
        #     for idx, block_sched in enumerate(self.block_schedulers):
        #         dcp.save(
        #             block_sched.state_dict(),
        #             checkpoint_id=f"{checkpoint_folder}/block_scheduler_{idx}",
        #         )
        
        if os.environ["RANK"] == "0":
            from src.core.checkpointing import save_training_state
            save_training_state(
                save_config=self.checkpoint.save,
                step=self.step,
                processed_tokens=self.processed_tokens,
                metric_logger=self.metric_logger,
            )
        logger.info(f"Saved checkpoint after step {self.step}")
