import os
from attr import define, field
import logging 
import torch
import torch.nn
from typing import Optional

from src.core.trainer import Trainer, cast_state_dict_to_tensors
from src.product_keys.model_sequence_classifiaction import ModelSequenceClassification

# for now focus solely on sst2
HIDDEN_SIZE = 128256
SST2_LABELS: int = 2

logger = logging.getLogger(__name__)

def show_gradients(model: torch.nn.Module):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                # Calculate some basic statistics to see what the gradients look like
                grad_mean = param.grad.abs().mean().item()
                grad_max = param.grad.abs().max().item()
                print(f"Layer: {name:<30} | Grad Mean: {grad_mean:.6f} | Grad Max: {grad_max:.6f}")
            else:
                print(f"Layer: {name:<30} | NO GRADIENT DEPOSITED (Disconnected layer?)")


def create_classifier_model(model: torch.nn.Module,
                            device: torch.device,
                            distributed: Optional[dict] = None) -> torch.nn.Module:
    logger.info("Printing model shapes...")
    for name, layer in model.named_modules():
        logger.info(f"Layer name: {name}")

        if hasattr(layer, 'weight') and layer.weight is not None:
            logger.info(f"Layer: {name} | Size: {layer.weight.shape}")
    
    model = ModelSequenceClassification(model, hidden_size=HIDDEN_SIZE, num_labels=SST2_LABELS,
                                        distributed=distributed).to(device)

    model_dtypes = set([param.dtype for param in model.parameters()])
    logger.info(f"Model dtypes: {list(model_dtypes)}")
    return model


@define(slots=False)
class FinetuningTrainer(Trainer):
    freeze_backbone: bool = field(default=False)
    trainable_modules: list = field(factory=list) 
    loss_fct = torch.nn.CrossEntropyLoss()

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

        logger.info(f"{self.distributed=}")

        if self.freeze_backbone:
            self._freeze_model_layers()

        self.model = create_classifier_model(self.model,
                self.device, self.distributed)
            
        
    def _freeze_model_layers(self):
        logger.info("Freezing backbone layers...")
        for name, param in self.model.named_parameters():
            should_train = any(mod in name for mod in self.trainable_modules)
            
            if not should_train:
                param.requires_grad = False
            else:
                param.requires_grad = True

    
    def train(self):
        logger.info(type(self.train_dataloader))
        for step, batch in zip(
            range(self.start_step, self.n_steps), self.train_dataloader
        ):
            self.step = step
            self.metric_logger.set_step(step)
            self.model.train()
            texts, labels = batch
            labels = labels.to(self.device)

            loss = self.calculate_loss(texts, labels)

            grad_norm = self.clip_gradient()

            self.log_metrics(loss, grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            if self._should_save_checkpoint:
                self.save_checkpoint()

            if self._should_evaluate:
                self.eval()

        if self._should_save_final_checkpoint:
            self.save_checkpoint()
        
        eval()

    
    def eval(self):
        pass


    def calculate_loss(self, texts, labels):
        logits = self.model(texts)
        # logger.info(f"{logits=}")
        loss = self.loss_fct(logits, labels)
        
        if self.model.training:
            logger.info("Backward loss")
            loss.backward()

        return loss
