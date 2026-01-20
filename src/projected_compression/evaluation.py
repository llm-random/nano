
from transformers import AutoConfig, LlamaForCausalLM

from src.projected_compression.initialization import create_model
import torch.distributed.checkpoint as dcp
import os
import torch
import torch.nn.functional as F
    

def evaluation(cfg, metric_logger):

    model = create_model(cfg.model, cfg.projected_compression)

    # dcp.load(model.state_dict(), checkpoint_id=f"{checkpoint_folder}/model")

    id = "/net/scratch/hscra/plgrid/plgcrewtool/tutaj_pc_hej_8b_minitron_testy_17/11472834/0/step_1023/model"
    dcp.load(model.state_dict(), checkpoint_id=id)

    model.prepare_compressed_weights()


    # with torch.no_grad():
    #     embedding_base = model.source_model.embedding.weight.full_tensor()
    #     if torch.distributed.get_rank() == 0:
    #         embedding_base = embedding_base.cpu()
    #     else:
    #         del embedding_base

    #     linear = model.projections.embedding.full_tensor()
    #     if torch.distributed.get_rank() == 0:
    #         linear = linear.cpu()
    #     else:
    #         del linear
        
    #     aux_w = model.projections.auxiliary_embedding_weights.weight.full_tensor()
    #     if torch.distributed.get_rank() == 0:
    #         aux_w = aux_w.cpu()
    #     else:
    #         del aux_w
        
    #     if torch.distributed.get_rank() == 0:
    #         embedding_weight = F.linear(embedding_base, linear, bias=None) + aux_w

    # Maybe use:
    # torch.distributed.checkpoint.state_dict.get_model_state_dict()
    # torch.distributed.checkpoint.state_dict.StateDictOptions

    if os.environ["RANK"] == "0":
        conf = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B")
        # _vocab_size, dmodel = embedding_weight.shape
        conf.num_hidden_layers = cfg.common.n_blocks
        conf.num_key_value_heads = cfg.common.kv_heads
        conf.num_attention_heads = cfg.common.q_heads
        # conf._name_or_path = f"{PROJECT_ORG}/{MODEL_NAME}"
        conf.hidden_size = cfg.common.dmodel
        conf.intermediate_size =  cfg.common.dff 
        conf.head_dim = cfg.common.dhead

        model2 = LlamaForCausalLM(conf)

    print("yes")
    return None, None, None, None, None

    


