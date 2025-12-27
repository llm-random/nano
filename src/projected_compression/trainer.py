import torch
from src.core.trainer import Trainer
from attr import define

@define(slots=False)
class PCTrainer(Trainer):

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.block_optimizers = self.optimizer[1:]
        self.block_schedulers = self.scheduler[1:]
        self.optimizer = self.optimizer[0]
        self.scheduler = self.scheduler[0]


    def train(self):
        self.model.source_model.embedding.weight.requires_grad = False
        for step, batch in zip(
            range(self.start_step, self.n_steps), self.train_dataloader
        ):
            self.step = step
            self.metric_logger.set_step(step)
            self.model.train()

            self.model.prepare_compressed_weights()
            loss = self.calculate_loss(batch)
            self.model.pass_gradient_to_projections(self.block_optimizers, self.block_schedulers, self.gradient_clipping)

            grad_norm = self.clip_gradient()
            self.log_metrics(loss, grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            self.model.zero_grad()

            if self._should_save_checkpoint:
                self.save_checkpoint()

            # if self._should_evaluate:
            #     self.eval()



