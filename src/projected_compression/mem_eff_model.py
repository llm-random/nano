import torch
import torch.nn as nn
from typing import Callable, Optional
from torch.nn.modules.normalization import RMSNorm

from src.core.model import LLM


class MemoryEfficientProjectedCompression(nn.Module):
    def __init__(
        self,
        embedding: nn.Module,
        encoder: nn.Module,
        head: nn.Module,
        projections: nn.Module
    ):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.head = head
        self.projections = projections

    def forward(self, *args, **kwargs):
        self.prepare_compressed_weights()
        x = self.embedding(*args, **kwargs)
        x = self.encoder(x)
        x = self.head(x)
        return x

    def prepare_compressed_weights(self):
        with torch.no_grad():
            for block, block_proj in zip(self.encoder.blocks, self.projections.blocks):
                block.attention_layer.layer.q_proj.weight.copy_(block_proj.compressible_Q.get_projected_weight())
                block.attention_layer.layer.k_proj.weight.copy_(block_proj.compressible_K.get_projected_weight())
                    # block.attention_layer.layer.v_proj.weight.data = block_proj.compressible_V.get_projected_weight()
                    # block.attention_layer.layer.o_proj.weight.data = block_proj.compressible_O.get_projected_weight()
                    # block.ff_layer.layer.ff_pre_act.weight.data = block_proj.compressible_FF_pre.get_projected_weight()
                    # block.ff_layer.layer.ff_post_act.weight.data = block_proj.compressible_FF_post.get_projected_weight()
            

    def pass_gradient_to_projections(self):
        for block, block_proj in zip(self.encoder.blocks, self.projections.blocks):
            # ej = block_proj.compressible_Q.get_projected_weight()
            # block.attention_layer.layer.q_proj.weight.data = ej
            waga = block_proj.compressible_Q.get_projected_weight()
            waga.backward(block.attention_layer.layer.q_proj.weight.grad)
            print("xd")
        
    def init_weights(self):
        pass
        #Maybe load_state_dict



class CompressibleLinear(nn.Module):
    def __init__(
        self,
        base_in_features: int,
        result_in_features: int,
        base_out_features: int,
        result_out_features: int,
        proj_in_topk_indices: Optional[torch.Tensor],
        proj_out_topk_indices: Optional[torch.Tensor]
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(base_out_features, base_in_features), requires_grad=False)
        self.base_in_features = base_in_features
        self.result_in_features = result_in_features
        self.base_out_features = base_out_features
        self.result_out_features = result_out_features
        self.proj_in_topk_indices = proj_in_topk_indices
        self.proj_out_topk_indices = proj_out_topk_indices
        
    # def initialize_projections(self):
        if self.base_in_features != self.result_in_features:
            assert self.proj_in_topk_indices is not None, "proj_in_topk_indices must be provided if result_in_features is specified."
            weight = torch.zeros(
                self.base_in_features, self.result_in_features
            )
            weight[self.proj_in_topk_indices, torch.arange(self.result_in_features)] = 1
            self.projection_in_weight = nn.Parameter(weight, requires_grad=True)

        if self.base_out_features != self.result_out_features:
            assert self.proj_out_topk_indices is not None, "proj_out_topk_indices must be provided if result_out_features is specified."
            # weight = torch.zeros(
            #     self.base_out_features, self.result_out_features
            # )
            # weight[self.proj_out_topk_indices, torch.arange(self.result_out_features)] = 1
            
            weight = torch.zeros(
                self.result_out_features, self.base_out_features
            )
            weight[torch.arange(self.result_out_features), self.proj_out_topk_indices ] = 1
            self.projection_out_weight = nn.Parameter(weight, requires_grad=True)
            
        if self.result_in_features is not None or self.result_out_features is not None:
            final_in_features = (
                self.result_in_features
                if self.result_in_features is not None
                else self.base_in_features
            )
            final_out_features = (
                self.result_out_features
                if self.result_out_features is not None
                else self.base_out_features
            )
            weight = torch.zeros(
                final_out_features, final_in_features
            )
            self.auxiliary_weight = nn.Parameter(weight, requires_grad=True)
            
    def get_projected_weight(self):
        weight = self.weight
        if hasattr(self, 'projection_in_weight'):
            weight = weight @ self.projection_in_weight
        if hasattr(self, 'projection_out_weight'):
            weight = self.projection_out_weight @ weight
            # weight = weight.T @ self.projection_out_weight
        if hasattr(self, 'auxiliary_weight'):
            # weight += self.auxiliary_weight
            weight = weight + self.auxiliary_weight
        return weight


    def extra_repr(self) -> str:
        if hasattr(self, 'projection_in_weight'):
            in_features, out_features = self.projection_in_weight.shape
            result = f"(projection_in_weight) ({in_features}, {out_features})\n"
        else:
            result = ""
        in_features, out_features = self.W.shape
        result += f"(weight) ({in_features}, {out_features})"
        if hasattr(self, 'projection_ouweight_weight'):
            out_features, in_features = self.projection_out_weight.shape
            result += f"\n(projection_out_weight) ({out_features}, {in_features})"
        return result

class CompressibleBlock(nn.Module):
    def __init__(
        self,
        q_heads: int,
        kv_heads: int,
        dhead: int,
        base_dmodel: int,
        target_dmodel: int,
        base_dff: int,
        target_dff: int,
        dmodel_topk_indices: Optional[torch.Tensor],
        dff_topk_indices: Optional[torch.Tensor]
    ):
        super().__init__()
        
            #         # Attention norms
            # sub_key = sub_key.replace(
            #     "input_layernorm.weight", "attention_layer.norm.weight"
            # )
            # sub_key = sub_key.replace(
            #     "post_attention_layernorm.weight", "ff_layer.norm.weight"
            # )
            
            
        self.attention_layer_norm = RMSNorm(normalized_shape=base_dmodel, eps=1e-5)
        self.ff_layer_norm = RMSNorm(normalized_shape=base_dmodel, eps=1e-5)

        self.compressible_q = CompressibleLinear(
            base_in_features=base_dmodel,
            result_in_features=target_dmodel,
            base_out_features=q_heads * dhead,
            result_out_features=q_heads * dhead,
            proj_in_topk_indices=dmodel_topk_indices,
            proj_out_topk_indices=dmodel_topk_indices
        )
        self.compressible_k = CompressibleLinear(
            base_in_features=base_dmodel,
            result_in_features=target_dmodel,
            base_out_features=kv_heads * dhead,
            result_out_features=kv_heads * dhead,   
            proj_in_topk_indices=dmodel_topk_indices,
            proj_out_topk_indices=None
        )
        self.compressible_v = CompressibleLinear(
            base_in_features=base_dmodel,
            result_in_features=target_dmodel,
            base_out_features=kv_heads * dhead,
            result_out_features=kv_heads * dhead,
            proj_in_topk_indices=dmodel_topk_indices,
            proj_out_topk_indices=None
        )
        self.compressible_o = CompressibleLinear(
            base_in_features=base_dmodel,
            result_in_features=base_dmodel,
            base_out_features=base_dmodel,
            result_out_features=target_dmodel,
            proj_in_topk_indices=None,
            proj_out_topk_indices=dmodel_topk_indices
        )
        self.compressible_ff_pre = CompressibleLinear(
            base_in_features=base_dmodel,
            result_in_features=target_dmodel,
            base_out_features=base_dff,
            result_out_features=target_dff,
            proj_in_topk_indices=dmodel_topk_indices,
            proj_out_topk_indices=dff_topk_indices
        )      
        self.compressible_ff_gate = CompressibleLinear(
            base_in_features=base_dmodel,
            result_in_features=target_dmodel,
            base_out_features=base_dff,
            result_out_features=target_dff,
            proj_in_topk_indices=dmodel_topk_indices,
            proj_out_topk_indices=dff_topk_indices
        )
        self.compressible_ff_post = CompressibleLinear(
            base_in_features=base_dff,
            result_in_features=target_dff,
            base_out_features=base_dmodel,
            result_out_features=target_dmodel,
            proj_in_topk_indices=dff_topk_indices,
            proj_out_topk_indices=dmodel_topk_indices
        )

class CompressibleEmbedding(nn.Module):
    def __init__(
        self,
        base_dmodel: int,
        target_dmodel: int,
        vocab_size: int,
        proj_topk_indices: Optional[torch.Tensor]
    ):
        super().__init__()
        self.weight = nn.Embedding(vocab_size, base_dmodel)
        self.projection = nn.Parameter(
            torch.zeros(base_dmodel, target_dmodel), requires_grad=True
        )

        if base_dmodel != target_dmodel:
            assert proj_topk_indices is not None, "proj_in_topk_indices must be provided if result_in_features is specified."
            weight = torch.zeros(
                base_dmodel, target_dmodel
            )
            weight[proj_topk_indices, torch.arange(target_dmodel)] = 1
            self.projection_out_weight = nn.Parameter(weight, requires_grad=True)
            
    def get_projected_weight(self):
        weight = self.weight.weight
        if hasattr(self, "projection_out_weight"):
            weight = weight @ self.projection_out_weight
        weight = weight @ self.projection
        return weight
        
            

        #     CompressibleLinear(
        #     base_in_features=base_dmodel,
        #     result_in_features=target_dmodel,
        #     base_out_features=base_dmodel,
        #     result_out_features=target_dmodel,
        #     proj_in_topk_indices=dmodel_topk_indices,
        #     proj_out_topk_indices=None
        # )
   
      
class Projections(nn.Module):
    def __init__(
        self,
        q_heads: int,
        kv_heads: int,
        base_dmodel: int,
        base_dff: int,
        target_dmodel: int,
        target_dff: int,
        # dmodel_topk_indices: torch.Tensor, #TODO
        # dff_topk_indices: list[torch.Tensor],
        n_blocks: int,
        vocab_size: int,
    ):
        dmodel_topk_indices =  torch.arange(target_dmodel) #TODO remove
        dff_topk_indices = [ torch.arange(target_dff) for _ in range(n_blocks)]  #TODO remove
        super().__init__()
        self.blocks = nn.ModuleList([
            CompressibleBlock(
                q_heads=q_heads,
                kv_heads=kv_heads,
                dhead=base_dmodel // q_heads,
                base_dmodel=base_dmodel,
                target_dmodel=target_dmodel,
                base_dff=base_dff,
                target_dff=target_dff,
                dmodel_topk_indices=dmodel_topk_indices,
                dff_topk_indices=dff_topk_indices[i]
                ) for i in range(n_blocks)
            ])
        self.head = CompressibleLinear(
            base_in_features=base_dmodel,
            result_in_features=target_dmodel,
            base_out_features=vocab_size,
            result_out_features=vocab_size,
            proj_in_topk_indices=dmodel_topk_indices,
            proj_out_topk_indices=None
        )
        
        self.embedding = CompressibleEmbedding(
            base_dmodel=base_dmodel,
            target_dmodel=target_dmodel,
            vocab_size=vocab_size,
            proj_topk_indices=dmodel_topk_indices
        )
        self.head_norm = RMSNorm(normalized_shape=base_dmodel, eps=1e-5)
        
        
        
        # weight = torch.zeros(len(dmodel_topk_indices), base_dmodel)
        # weight[torch.arange(self.result_out_features), dmodel_topk_indices] = 1
        # self.projection = nn.Parameter(weight, requires_grad=True)
        # self.initialized_compression = True
        # self.embedding_proj =



# print("XD")
# proj = Projections(
#     q_heads=4,
#     kv_heads=2,
#     base_dmodel=32,
#     base_dff=96,
#     target_dmodel=8,
#     target_dff=24,
#     n_blocks=3,
#     vocab_size=17,
# )
# print(proj)

# lol = proj.head.get_projected_weight()

# print(lol.shape)



class ProjectedCompressionModel(nn.Module):
    def __init__(
        self,
        source_model: nn.Module,
        target_model: nn.Module,
        # load_config: dict,
    ):
        super().__init__()
        self.model = target_model

        self.projections = Projections(
            q_heads=source_model.encoder.blocks[0].attention_layer.layer.q_heads,
            kv_heads=source_model.encoder.blocks[0].attention_layer.layer.kv_heads,
            base_dmodel=source_model.encoder.blocks[0].attention_layer.layer.dmodel,
            base_dff=source_model.encoder.blocks[0].ff_layer.layer.ff_pre_act.out_features,
            target_dmodel=target_model.encoder.blocks[0].attention_layer.layer.dmodel,
            target_dff=target_model.encoder.blocks[0].ff_layer.layer.ff_pre_act.out_features,
            n_blocks=len(source_model.encoder.blocks),
            vocab_size=source_model.embedding.num_embeddings
        )
        self.move_weights_from_source_model(source_model)

        
    def move_weights_from_source_model(self, source_model: nn.Module):
        with torch.no_grad():
            self.projections.embedding.weight.weight.copy_(source_model.embedding.weight)
 
            for (target_block, source_block) in zip(self.projections.blocks, source_model.encoder.blocks):
                target_block.compressible_q.weight.copy_(source_block.attention_layer.layer.q_proj.weight)
                target_block.compressible_k.weight.copy_(source_block.attention_layer.layer.k_proj.weight)
                target_block.compressible_v.weight.copy_(source_block.attention_layer.layer.v_proj.weight)
                target_block.compressible_o.weight.copy_(source_block.attention_layer.layer.o_proj.weight)
                target_block.compressible_ff_pre.weight.copy_(source_block.ff_layer.layer.ff_pre_act.weight)
                target_block.compressible_ff_gate.weight.copy_(source_block.ff_layer.layer.gate.weight)
                target_block.compressible_ff_post.weight.copy_(source_block.ff_layer.layer.ff_post_act.weight)
                # target_block.attention_layer_norm.weight.copy_(source_block.attention_layer.norm.weight) #TODO
                # target_block.ff_layer_norm.weight.copy_(source_block.ff_layer.norm.weight)               #TODO
                
            self.projections.head.weight.copy_(source_model.head.linear.weight)
            self.projections.head_norm.weight.copy_(source_model.head.norm.weight)

    def prepare_compressed_weights(self):
        with torch.no_grad():
            for (projections_block, model_block) in zip(self.projections.blocks, self.model.encoder.blocks):
                model_block.attention_layer.layer.q_proj.weight.copy_(projections_block.compressible_q.get_projected_weight())
                model_block.attention_layer.layer.k_proj.weight.copy_(projections_block.compressible_k.get_projected_weight())
                model_block.attention_layer.layer.v_proj.weight.copy_(projections_block.compressible_v.get_projected_weight())
                model_block.attention_layer.layer.o_proj.weight.copy_(projections_block.compressible_o.get_projected_weight())
                
                model_block.ff_layer.layer.ff_pre_act.weight.copy_(projections_block.compressible_ff_pre.get_projected_weight())
                model_block.ff_layer.layer.gate.weight.copy_(projections_block.compressible_ff_gate.get_projected_weight())
                model_block.ff_layer.layer.ff_post_act.weight.copy_(projections_block.compressible_ff_post.get_projected_weight())

    def pass_gradient_to_projections(self):
        for (projections_block, model_block) in zip(self.projections.blocks, self.model.encoder.blocks):
            self.backward_compressed_weights(projections_block.compressible_q, model_block.attention_layer.layer.q_proj.weight.grad)
            self.backward_compressed_weights(projections_block.compressible_k, model_block.attention_layer.layer.k_proj.weight.grad)
            self.backward_compressed_weights(projections_block.compressible_v, model_block.attention_layer.layer.v_proj.weight.grad)
            self.backward_compressed_weights(projections_block.compressible_o, model_block.attention_layer.layer.o_proj.weight.grad)
            self.backward_compressed_weights(projections_block.compressible_ff_pre, model_block.ff_layer.layer.ff_pre_act.weight.grad)
            self.backward_compressed_weights(projections_block.compressible_ff_gate, model_block.ff_layer.layer.gate.weight.grad)
            self.backward_compressed_weights(projections_block.compressible_ff_post, model_block.ff_layer.layer.ff_post_act.weight.grad)
            
    def backward_compressed_weights(self, compressible_weights, gradient):
        weights = compressible_weights.get_projected_weight()
        weights.backward(gradient)
        
        # self.prepare_compressed_weights()
        # self.model.zero_grad()
        # loss = self.compute_loss()  # Placeholder for actual loss computation
        # loss.backward()
        # self.pass_gradient_to_projections()

    def forward(self, *args, **kwargs):
        x = self.model(*args, **kwargs)
        return x

class PretrainedLLM(LLM):
    def __init__(
        self,
        embedding: nn.Module,
        encoder: nn.Module,
        head: nn.Module,
        initialize_weights: Callable[[nn.Module], None],
    ):  
        with torch.device('meta'):
            super().__init__(embedding, encoder, head)
        initialize_weights(self)
        
        