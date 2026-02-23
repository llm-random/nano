from attr import define, field
import logging 
import torch

from src.core.trainer import Trainer

logger = logging.getLogger(__name__)
from nano.src.product_keys.finetuning.model_sequence_classifiaction import ModelSequenceClassification

# for now focus solely on sst2
HIDDEN_SIZE = 1024
SST2_LABELS: int = 2


def create_classifier_model(model: torch.nn.Module) -> torch.nn.Module:
    logger.info("Printing model shapes...")
    for name, layer in model.named_modules():
        # We filter for Linear layers to keep the output readable
        if isinstance(layer, torch.nn.Linear):
            logger.info(f"Layer: {name} | Size: {layer.weight.shape}")
    return ModelSequenceClassification(model, hidden_size=HIDDEN_SIZE, num_labels=SST2_LABELS)


@define(slots=False)
class FinetuningTrainer(Trainer):
    freeze_backbone: bool = field(default=False)
    trainable_modules: list = field(factory=list) 

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

        if self.freeze_backbone:
            self._freeze_model_layers()

        model = create_classifier_model(model)
        
    def _freeze_model_layers(self):
        logger.info("Freezing backbone layers...")
        for name, param in self.model.named_parameters():
            should_train = any(mod in name for mod in self.trainable_modules)
            
            if not should_train:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def save_checkpoint(self):
        logger.info("Saving finetune checkpoint...")
        super().save_checkpoint()

