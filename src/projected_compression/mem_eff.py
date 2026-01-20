import torch
import torch.nn as nn
from typing import List, Optional
import torch.nn.functional as F
from src.projected_compression.initialization import get_topk_indices
from torch.distributed.tensor import distribute_tensor, DTensor


class MemoryEfficientProjectedCompression(nn.Module):
    # fmt: off
    def __init__(
        self,
        source_model: nn.Module,
        target_model: nn.Module,
        path_to_importances: str,
        cast_bfloat16: bool,
        adjust_grad_norm: bool,
    ):
        super().__init__()
        self.source_model = source_model
        self.target_model = target_model
        self.cast_bfloat16 = cast_bfloat16
        self.adjust_grad_norm = adjust_grad_norm
        self.projections = Projections(
            q_heads=target_model.encoder.blocks[0].attention_layer.layer.q_heads,
            kv_heads=target_model.encoder.blocks[0].attention_layer.layer.kv_heads,
            base_dmodel=source_model.encoder.blocks[0].attention_layer.layer.dmodel,
            base_dff=source_model.encoder.blocks[0].ff_layer.layer.ff_pre_act.out_features,
            target_dmodel=target_model.encoder.blocks[0].attention_layer.layer.dmodel,
            target_dff=target_model.encoder.blocks[0].ff_layer.layer.ff_pre_act.out_features,
            n_blocks=len(target_model.encoder.blocks),
            vocab_size=target_model.embedding.num_embeddings,
            path_to_importances=path_to_importances,
            cast_bfloat16=cast_bfloat16,
        )
    # fmt: on

    def forward(self, *args, **kwargs):
        x = self.source_model.embedding(*args, **kwargs)
        x = F.linear(x, self.projections.embedding, bias=None)
        x = x + self.projections.auxiliary_embedding_weights(*args)
        x = self.target_model.encoder(x)
        x = self.target_model.head(x)
        return x

    def prepare_compressed_weights(self):
        """
        Copies the projected weights from source_model to target_model using the projections.
        cast_bfloat16: whether to cast the source weights to bfloat16 before projection. This argument only exists to have backward compatibility with previous implementation.
                       after testing, we can remove it and never cast to bfloat16.
        """
        with torch.no_grad():
            if not self.cast_bfloat16:
                for block_target, block_source, block_proj in zip(
                    self.target_model.encoder.blocks,
                    self.source_model.encoder.blocks,
                    self.projections.blocks,
                ):
                    block_target.attention_layer.layer.q_proj.weight.copy_(
                        block_proj.compressible_q.get_projected_weight(
                            block_source.attention_layer.layer.q_proj.weight
                        )
                    )
                    block_target.attention_layer.layer.k_proj.weight.copy_(
                        block_proj.compressible_k.get_projected_weight(
                            block_source.attention_layer.layer.k_proj.weight
                        )
                    )
                    block_target.attention_layer.layer.v_proj.weight.copy_(
                        block_proj.compressible_v.get_projected_weight(
                            block_source.attention_layer.layer.v_proj.weight
                        )
                    )
                    block_target.attention_layer.layer.o_proj.weight.copy_(
                        block_proj.compressible_o.get_projected_weight(
                            block_source.attention_layer.layer.o_proj.weight
                        )
                    )

                    block_target.ff_layer.layer.ff_pre_act.weight.copy_(
                        block_proj.compressible_ff_pre.get_projected_weight(
                            block_source.ff_layer.layer.ff_pre_act.weight
                        )
                    )
                    block_target.ff_layer.layer.gate.weight.copy_(
                        block_proj.compressible_ff_gate.get_projected_weight(
                            block_source.ff_layer.layer.gate.weight
                        )
                    )
                    block_target.ff_layer.layer.ff_post_act.weight.copy_(
                        block_proj.compressible_ff_post.get_projected_weight(
                            block_source.ff_layer.layer.ff_post_act.weight
                        )
                    )

                self.target_model.head.linear.weight.copy_(
                    self.projections.head.get_projected_weight(
                        self.source_model.head.linear.weight
                    )
                )
            else:
                for block_target, block_source, block_proj in zip(
                    self.target_model.encoder.blocks,
                    self.source_model.encoder.blocks,
                    self.projections.blocks,
                ):
                    block_target.attention_layer.layer.q_proj.weight.copy_(
                        block_proj.compressible_q.get_projected_weight(
                            block_source.attention_layer.layer.q_proj.weight.bfloat16()
                        )
                    )
                    block_target.attention_layer.layer.k_proj.weight.copy_(
                        block_proj.compressible_k.get_projected_weight(
                            block_source.attention_layer.layer.k_proj.weight.bfloat16()
                        )
                    )
                    block_target.attention_layer.layer.v_proj.weight.copy_(
                        block_proj.compressible_v.get_projected_weight(
                            block_source.attention_layer.layer.v_proj.weight.bfloat16()
                        )
                    )
                    block_target.attention_layer.layer.o_proj.weight.copy_(
                        block_proj.compressible_o.get_projected_weight(
                            block_source.attention_layer.layer.o_proj.weight.bfloat16()
                        )
                    )

                    block_target.ff_layer.layer.ff_pre_act.weight.copy_(
                        block_proj.compressible_ff_pre.get_projected_weight(
                            block_source.ff_layer.layer.ff_pre_act.weight.bfloat16()
                        )
                    )
                    block_target.ff_layer.layer.gate.weight.copy_(
                        block_proj.compressible_ff_gate.get_projected_weight(
                            block_source.ff_layer.layer.gate.weight.bfloat16()
                        )
                    )
                    block_target.ff_layer.layer.ff_post_act.weight.copy_(
                        block_proj.compressible_ff_post.get_projected_weight(
                            block_source.ff_layer.layer.ff_post_act.weight.bfloat16()
                        )
                    )

                self.target_model.head.linear.weight.copy_(
                    self.projections.head.get_projected_weight(
                        self.source_model.head.linear.weight.bfloat16()
                    )
                )

    def pass_gradient_to_projections(
        self, optimizers: List, schedulers, gradient_clipping
    ):

        def get_compressed_matrices(block):
            params = []
            params.append(block.attention_layer.layer.q_proj)
            params.append(block.attention_layer.layer.k_proj)
            params.append(block.attention_layer.layer.v_proj)
            params.append(block.attention_layer.layer.o_proj)
            params.append(block.ff_layer.layer.ff_pre_act)
            params.append(block.ff_layer.layer.gate)
            params.append(block.ff_layer.layer.ff_post_act)
            return params

        def get_compressed_params_grad_norm(block):
            compressed_params = get_compressed_matrices(block)
            grads = [
                p.weight.grad for p in compressed_params if p.weight.grad is not None
            ]
            return torch.nn.utils.get_total_norm(grads)

        def get_module_grad_norm(module: nn.Module):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            return torch.nn.utils.get_total_norm(grads)

        self.backward_compressed_weights(
            self.projections.head,
            self.source_model.head.linear.weight,
            self.target_model.head.linear.weight,
        )

        if optimizers is None:
            for block_target, block_source, block_proj in zip(
                self.target_model.encoder.blocks,
                self.source_model.encoder.blocks,
                self.projections.blocks,
            ):
                self.backward_compressed_weights(
                    block_proj.compressible_q,
                    block_source.attention_layer.layer.q_proj.weight,
                    block_target.attention_layer.layer.q_proj.weight,
                )
                self.backward_compressed_weights(
                    block_proj.compressible_k,
                    block_source.attention_layer.layer.k_proj.weight,
                    block_target.attention_layer.layer.k_proj.weight,
                )
                self.backward_compressed_weights(
                    block_proj.compressible_v,
                    block_source.attention_layer.layer.v_proj.weight,
                    block_target.attention_layer.layer.v_proj.weight,
                )
                self.backward_compressed_weights(
                    block_proj.compressible_o,
                    block_source.attention_layer.layer.o_proj.weight,
                    block_target.attention_layer.layer.o_proj.weight,
                )
                self.backward_compressed_weights(
                    block_proj.compressible_ff_pre,
                    block_source.ff_layer.layer.ff_pre_act.weight,
                    block_target.ff_layer.layer.ff_pre_act.weight,
                )
                self.backward_compressed_weights(
                    block_proj.compressible_ff_gate,
                    block_source.ff_layer.layer.gate.weight,
                    block_target.ff_layer.layer.gate.weight,
                )
                self.backward_compressed_weights(
                    block_proj.compressible_ff_post,
                    block_source.ff_layer.layer.ff_post_act.weight,
                    block_target.ff_layer.layer.ff_post_act.weight,
                )

            grads = [v.grad for v in self.parameters() if v.grad is not None]
            final_grad_norm = torch.nn.utils.get_total_norm(grads)
        else:
            projection_blocks_grad_norms = []
            grads = [v.grad for v in self.parameters() if v.grad is not None]
            start_grad_norm = torch.nn.utils.get_total_norm(grads)
            for block_target, block_source, block_proj, optimizer, scheduler in zip(
                self.target_model.encoder.blocks,
                self.source_model.encoder.blocks,
                self.projections.blocks,
                optimizers,
                schedulers,
            ):
                compressed_params_grad_norm = get_compressed_params_grad_norm(
                    block_target
                )

                self.backward_compressed_weights(
                    block_proj.compressible_q,
                    block_source.attention_layer.layer.q_proj.weight,
                    block_target.attention_layer.layer.q_proj.weight,
                )
                self.backward_compressed_weights(
                    block_proj.compressible_k,
                    block_source.attention_layer.layer.k_proj.weight,
                    block_target.attention_layer.layer.k_proj.weight,
                )
                self.backward_compressed_weights(
                    block_proj.compressible_v,
                    block_source.attention_layer.layer.v_proj.weight,
                    block_target.attention_layer.layer.v_proj.weight,
                )
                self.backward_compressed_weights(
                    block_proj.compressible_o,
                    block_source.attention_layer.layer.o_proj.weight,
                    block_target.attention_layer.layer.o_proj.weight,
                )
                self.backward_compressed_weights(
                    block_proj.compressible_ff_pre,
                    block_source.ff_layer.layer.ff_pre_act.weight,
                    block_target.ff_layer.layer.ff_pre_act.weight,
                )
                self.backward_compressed_weights(
                    block_proj.compressible_ff_gate,
                    block_source.ff_layer.layer.gate.weight,
                    block_target.ff_layer.layer.gate.weight,
                )
                self.backward_compressed_weights(
                    block_proj.compressible_ff_post,
                    block_source.ff_layer.layer.ff_post_act.weight,
                    block_target.ff_layer.layer.ff_post_act.weight,
                )

                projection_block_grad_norm = get_module_grad_norm(block_proj)
                projection_blocks_grad_norms.append(projection_block_grad_norm)
                if gradient_clipping:
                    grad_norm_to_use = start_grad_norm
                    if self.adjust_grad_norm:
                        grad_norm_to_use = (
                            start_grad_norm**2
                            - compressed_params_grad_norm**2
                            + projection_block_grad_norm**2
                        ) ** 0.5

                    torch.nn.utils.clip_grads_with_norm_(
                        block_proj.parameters(), gradient_clipping, grad_norm_to_use
                    )

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            grads = [v.grad for v in self.parameters() if v.grad is not None]
            final_grad_norm = torch.nn.utils.get_total_norm(
                grads + projection_blocks_grad_norms
            )

        return final_grad_norm

    def backward_compressed_weights(self, proj, source_weight, target_weight):
        source_weight = source_weight.detach()
        if self.cast_bfloat16:
            source_weight = source_weight.bfloat16()
        weights = proj.get_projected_weight(source_weight)
        weights.backward(target_weight.grad)
        target_weight.grad = None


class CompressibleLinear(nn.Module):
    def __init__(
        self,
        base_in_features: int,
        result_in_features: int,
        base_out_features: int,
        result_out_features: int,
        proj_in_topk_indices: Optional[torch.Tensor],
        proj_out_topk_indices: Optional[torch.Tensor],
        cast_bfloat16: bool,
    ):
        super().__init__()
        self.cast_bfloat16 = cast_bfloat16
        self.base_in_features = base_in_features
        self.result_in_features = result_in_features
        self.base_out_features = base_out_features
        self.result_out_features = result_out_features
        self.proj_in_topk_indices = proj_in_topk_indices
        self.proj_out_topk_indices = proj_out_topk_indices

        if self.base_in_features != self.result_in_features:
            assert (
                self.proj_in_topk_indices is not None
            ), "proj_in_topk_indices must be provided if result_in_features is specified."
            weight = torch.zeros(self.base_in_features, self.result_in_features)
            self.projection_in_weight = nn.Parameter(weight, requires_grad=True)

        if self.base_out_features != self.result_out_features:
            assert (
                self.proj_out_topk_indices is not None
            ), "proj_out_topk_indices must be provided if result_out_features is specified."

            weight = torch.zeros(self.result_out_features, self.base_out_features)
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
            weight = torch.zeros(final_out_features, final_in_features)
            self.auxiliary_weight = nn.Parameter(weight, requires_grad=True)

    def init_projection_weights(self, proj_in_topk_indices, proj_out_topk_indices):
        with torch.no_grad():
            if hasattr(self, "projection_in_weight"):
                weight = torch.zeros(self.projection_in_weight.data.shape)
                weight[proj_in_topk_indices, torch.arange(self.result_in_features)] = 1

                if isinstance(self.projection_in_weight, DTensor):
                    self.projection_in_weight.data.copy_(
                        distribute_tensor(
                            weight,
                            self.projection_in_weight.device_mesh,
                            self.projection_in_weight.placements,
                        )
                    )
                else:
                    self.projection_in_weight.data.copy_(weight)

            if hasattr(self, "projection_out_weight"):
                weight = torch.zeros(self.projection_out_weight.data.shape)
                weight[
                    torch.arange(self.result_out_features), proj_out_topk_indices
                ] = 1

                if isinstance(self.projection_out_weight, DTensor):
                    self.projection_out_weight.data.copy_(
                        distribute_tensor(
                            weight,
                            self.projection_out_weight.device_mesh,
                            self.projection_out_weight.placements,
                        )
                    )
                else:
                    self.projection_out_weight.data.copy_(weight)

            if hasattr(self, "auxiliary_weight"):
                self.auxiliary_weight.data.copy_(
                    torch.zeros_like(self.auxiliary_weight.data)
                )

    def get_projected_weight(self, source_weight):
        if not self.cast_bfloat16:
            weight = source_weight
            if hasattr(self, "projection_in_weight"):
                weight = weight @ self.projection_in_weight
            if hasattr(self, "projection_out_weight"):
                weight = self.projection_out_weight @ weight
            if hasattr(self, "auxiliary_weight"):
                weight = weight + self.auxiliary_weight
            return weight
        else:
            weight = source_weight
            if hasattr(self, "projection_in_weight"):
                weight = weight @ self.projection_in_weight.bfloat16()
            if hasattr(self, "projection_out_weight"):
                weight = self.projection_out_weight.bfloat16() @ weight
            if hasattr(self, "auxiliary_weight"):
                weight = weight + self.auxiliary_weight.bfloat16()
            return weight.float()

    def extra_repr(self) -> str:
        if hasattr(self, "projection_in_weight"):
            in_features, out_features = self.projection_in_weight.shape
            result = f"(projection_in_weight) ({in_features}, {out_features})\n"
        else:
            result = ""
        in_features, out_features = self.W.shape
        result += f"(weight) ({in_features}, {out_features})"
        if hasattr(self, "projection_ouweight_weight"):
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
        dff_topk_indices: Optional[torch.Tensor],
        cast_bfloat16: bool,
    ):
        super().__init__()

        self.compressible_q = CompressibleLinear(
            base_in_features=base_dmodel,
            result_in_features=target_dmodel,
            base_out_features=q_heads * dhead,
            result_out_features=q_heads * dhead,
            proj_in_topk_indices=dmodel_topk_indices,
            proj_out_topk_indices=dmodel_topk_indices,
            cast_bfloat16=cast_bfloat16,
        )
        self.compressible_k = CompressibleLinear(
            base_in_features=base_dmodel,
            result_in_features=target_dmodel,
            base_out_features=kv_heads * dhead,
            result_out_features=kv_heads * dhead,
            proj_in_topk_indices=dmodel_topk_indices,
            proj_out_topk_indices=None,
            cast_bfloat16=cast_bfloat16,
        )
        self.compressible_v = CompressibleLinear(
            base_in_features=base_dmodel,
            result_in_features=target_dmodel,
            base_out_features=kv_heads * dhead,
            result_out_features=kv_heads * dhead,
            proj_in_topk_indices=dmodel_topk_indices,
            proj_out_topk_indices=None,
            cast_bfloat16=cast_bfloat16,
        )
        self.compressible_o = CompressibleLinear(
            base_in_features=base_dmodel,
            result_in_features=base_dmodel,
            base_out_features=base_dmodel,
            result_out_features=target_dmodel,
            proj_in_topk_indices=None,
            proj_out_topk_indices=dmodel_topk_indices,
            cast_bfloat16=cast_bfloat16,
        )
        self.compressible_ff_pre = CompressibleLinear(
            base_in_features=base_dmodel,
            result_in_features=target_dmodel,
            base_out_features=base_dff,
            result_out_features=target_dff,
            proj_in_topk_indices=dmodel_topk_indices,
            proj_out_topk_indices=dff_topk_indices,
            cast_bfloat16=cast_bfloat16,
        )
        self.compressible_ff_gate = CompressibleLinear(
            base_in_features=base_dmodel,
            result_in_features=target_dmodel,
            base_out_features=base_dff,
            result_out_features=target_dff,
            proj_in_topk_indices=dmodel_topk_indices,
            proj_out_topk_indices=dff_topk_indices,
            cast_bfloat16=cast_bfloat16,
        )
        self.compressible_ff_post = CompressibleLinear(
            base_in_features=base_dff,
            result_in_features=target_dff,
            base_out_features=base_dmodel,
            result_out_features=target_dmodel,
            proj_in_topk_indices=dff_topk_indices,
            proj_out_topk_indices=dmodel_topk_indices,
            cast_bfloat16=cast_bfloat16,
        )

    def init_projection_weights(self, dmodel_topk_indices, dff_topk_indices):
        self.compressible_q.init_projection_weights(
            dmodel_topk_indices, dmodel_topk_indices
        )
        self.compressible_k.init_projection_weights(dmodel_topk_indices, None)
        self.compressible_v.init_projection_weights(dmodel_topk_indices, None)
        self.compressible_o.init_projection_weights(None, dmodel_topk_indices)
        self.compressible_ff_pre.init_projection_weights(
            dmodel_topk_indices, dff_topk_indices
        )
        self.compressible_ff_gate.init_projection_weights(
            dmodel_topk_indices, dff_topk_indices
        )
        self.compressible_ff_post.init_projection_weights(
            dff_topk_indices, dmodel_topk_indices
        )


class Projections(nn.Module):
    def __init__(
        self,
        q_heads: int,
        kv_heads: int,
        base_dmodel: int,
        base_dff: int,
        target_dmodel: int,
        target_dff: int,
        n_blocks: int,
        vocab_size: int,
        path_to_importances: str,
        cast_bfloat16: bool,
    ):
        dmodel_topk_indices, dff_topk_indices = get_topk_indices(
            path_to_importances, target_dmodel, target_dff
        )

        super().__init__()
        self.target_dmodel = target_dmodel
        self.target_dff = target_dff

        # --- Embedding start
        weight = torch.zeros(target_dmodel, base_dmodel)
        weight[torch.arange(target_dmodel), dmodel_topk_indices] = 1
        self.embedding = nn.Parameter(weight, requires_grad=True)

        zeros = torch.zeros(vocab_size, target_dmodel)
        self.auxiliary_embedding_weights = nn.Embedding(
            vocab_size, target_dmodel, _weight=zeros
        )
        # --- Embedding end

        self.blocks = nn.ModuleList(
            [
                CompressibleBlock(
                    q_heads=q_heads,
                    kv_heads=kv_heads,
                    dhead=base_dmodel // q_heads,
                    base_dmodel=base_dmodel,
                    target_dmodel=target_dmodel,
                    base_dff=base_dff,
                    target_dff=target_dff,
                    dmodel_topk_indices=dmodel_topk_indices,
                    dff_topk_indices=dff_topk_indices[i],
                    cast_bfloat16=cast_bfloat16,
                )
                for i in range(n_blocks)
            ]
        )
        self.head = CompressibleLinear(
            base_in_features=base_dmodel,
            result_in_features=target_dmodel,
            base_out_features=vocab_size,
            result_out_features=vocab_size,
            proj_in_topk_indices=dmodel_topk_indices,
            proj_out_topk_indices=None,
            cast_bfloat16=cast_bfloat16,
        )

    def init_projection_weights(self, path_to_importances):
        dmodel_topk_indices, dff_topk_indices = get_topk_indices(
            path_to_importances, self.target_dmodel, self.target_dff
        )

        with torch.no_grad():
            weight = torch.zeros(self.embedding.data.shape)
            weight[torch.arange(self.target_dmodel), dmodel_topk_indices] = 1

            if isinstance(self.embedding, DTensor):
                self.embedding.data.copy_(
                    distribute_tensor(
                        weight,
                        self.embedding.device_mesh,
                        self.embedding.placements,
                    )
                )
            else:
                self.embedding.data.copy_(weight)

            self.auxiliary_embedding_weights.weight.data.copy_(
                torch.zeros_like(self.auxiliary_embedding_weights.weight.data)
            )

            for block, block_dff_topk_indices in zip(self.blocks, dff_topk_indices):
                block.init_projection_weights(
                    dmodel_topk_indices, block_dff_topk_indices
                )

            self.head.init_projection_weights(self.head.proj_in_topk_indices, None)
