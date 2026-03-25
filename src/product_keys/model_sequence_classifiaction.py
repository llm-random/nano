import torch
import torch.nn as nn
import logging 
from typing import Optional

logger = logging.getLogger(__name__)
from omegaconf import OmegaConf
from src.core.distributed_training import setup_distributed_training


class TransformerHead(nn.Module):
    def __init__(self, d_model: int, num_labels: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        self.linear = nn.Linear(d_model, num_labels, bias=False, dtype=torch.float32)

    def forward(self, x):
        x = self.norm(x)
        # logger.info(f"{x[:, :4, :4]=}")
        return self.linear(x)


class ModelSequenceClassification(nn.Module):
    def __init__(self, base_model: nn.Module, d_model: int, 
                 num_labels: int, distributed: Optional[dict] = None):
        super().__init__()
        self.backbone = base_model

        # replace head
        assert hasattr(self.backbone, "head"), "Model provided for sequence classification should have head attribute"
        self.backbone.head = TransformerHead(d_model, num_labels) 
        
        if distributed is not None:
            logger.info("Using distributed on model sequence classifier")
            distributed_config = OmegaConf.create(distributed)
            logger.info(f"Distributed config: {distributed_config}")
            self.backbone.head = setup_distributed_training(
                self.backbone.head, distributed_config=distributed_config)

    count = 0

    def forward(self, input_ids, attention_mask=None):

        model_dtypes = set([param.dtype for param in self.backbone.parameters()])
        logger.debug(f"Backbone model dtypes: {list(model_dtypes)}")

        hidden_states = self.backbone(input_ids, attention_mask=attention_mask)

        logger.debug(f"Hidden states shape: {hidden_states.shape}")
        logger.debug(f"Hidden states type: {hidden_states.dtype}")

        # take hidden states from [CLS] token
        logits = hidden_states[:, 0, :]
        logger.debug(f"CLS token hidden states shape: {logits.shape}")

        # logits = self.score(cls_token_hidden_states)

        if self.count % 5 == 0:
            logger.info(f"Logits shape: {logits.shape}")
            logger.info(f"{logits=}")

        self.count += 1

        return logits
