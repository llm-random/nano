from pathlib import Path
import torch

def load_finalized_pc_checkpoint(model, load_config):
    checkpoint = torch.load(str(Path(load_config.path, load_config.model_checkpoint_filename)))
    model.load_state_dict(checkpoint["model"]) 