import torch
import torch.nn as nn

class ModelSequenceClassification(nn.Module):
    def __init__(self, base_model: nn.Module, hidden_size: int, num_labels: int):
        super().__init__()
        self.backbone = base_model
        
        self.score = nn.Linear(hidden_size, num_labels, bias=False)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.backbone(input_ids)
        
        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
        
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
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.score.out_features), labels.view(-1))
            
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}
