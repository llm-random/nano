import torch
from src.core.trainer import Trainer
from attr import define

@define(slots=False)
class PCTrainer(Trainer):

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

            grad_norm = self.model.pass_gradient_to_projections(self.block_optimizers, self.block_schedulers, self.gradient_clipping)
            torch.nn.utils.clip_grads_with_norm_(self.model.parameters(), self.gradient_clipping, grad_norm)

            self.log_metrics(loss, grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()


            if self._should_save_checkpoint:
                self.save_checkpoint()

            # if self._should_evaluate:
            #     self.eval()
