import torch
import torch.nn as nn
import logging 
from typing import Optional

logger = logging.getLogger(__name__)
from omegaconf import OmegaConf
from src.core.distributed_training import setup_distributed_training


class ModelSequenceClassification(nn.Module):
    def __init__(self, base_model: nn.Module, hidden_size: int, 
                 num_labels: int, distributed: Optional[dict] = None):
        super().__init__()
        self.backbone = base_model
        self.score = nn.Linear(hidden_size, num_labels, bias=False, dtype=torch.float32)
        
        if distributed is not None:
            logger.info("Using distributed on model sequence classifier")
            distributed_config = OmegaConf.create(distributed)
            logger.info(f"Distributed config: {distributed_config}")
            self.score = setup_distributed_training(self.score, 
                                                    distributed_config=distributed_config)

        
    def forward(self, input_ids, attention_mask=None):
        model_dtypes = set([param.dtype for param in self.backbone.parameters()])
        logger.info(f"Backbone model dtypes: {list(model_dtypes)}")

        outputs = self.backbone(input_ids)

        logger.info(f"Outputs shape: {outputs.shape}")
        
        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
        
        logger.info(f"Hidden states shape: {hidden_states.shape}")
        logger.info(f"Hidden states type: {hidden_states.dtype}")

        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = input_ids.shape[0]
            last_token_hidden_states = hidden_states[
                torch.arange(batch_size, device=hidden_states.device), 
                sequence_lengths
            ]
        else:
            last_token_hidden_states = hidden_states[:, -1, :]

        logits = self.score(last_token_hidden_states)
            
       #  finetune_object = {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

        return logits
