import logging
import os
import torch
import torch.distributed as dist
from typing import Optional, Union
from src.core.checkpointing import step_checkpoint_path
from src.core.trainer import Trainer
from src.projected_compression.mem_eff import get_global_grad_norm
from attr import define
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
)
from torch.distributed.tensor import DTensor

logger = logging.getLogger(__name__)


@define(slots=False)
class PCTrainer(Trainer):
    only_compress_model_gradient_clipping: bool
    only_target_model_gradient_clipping: Optional[Union[float, str]] = (
        None  # float = per-block projection clip threshold; "no_projection_clip" = clip target model only
    )
    original_llama_path: Optional[str] = (
        None  # if set, also saves HF-format checkpoint for lm_eval
    )

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

    def _clip_model_grads(self, grad_norm):
        """Clip all model grads with a pre-computed global norm.
        When cpu_offload_projections is active, some params (head/embedding projections)
        are plain GPU tensors while target_model params are FSDP2 DTensors.
        torch._foreach_mul_ cannot mix the two, so we split by grad type."""
        if not self.gradient_clipping:
            return
        params_with_grads = [p for p in self.model.parameters() if p.grad is not None]
        dtensor_grads = [p for p in params_with_grads if hasattr(p.grad, "device_mesh")]
        plain_grads = [
            p for p in params_with_grads if not hasattr(p.grad, "device_mesh")
        ]
        if dtensor_grads:
            torch.nn.utils.clip_grads_with_norm_(
                dtensor_grads, self.gradient_clipping, grad_norm
            )
        if plain_grads:
            torch.nn.utils.clip_grads_with_norm_(
                plain_grads, self.gradient_clipping, grad_norm
            )

    def train(self):
        for step, batch in zip(
            range(self.start_step, self.n_steps), self.train_dataloader
        ):

            self.step = step
            self.metric_logger.set_step(step)
            self.metric_logger.set_tokens(self.processed_tokens)
            self.model.train()

            self.model.prepare_compressed_weights()
            loss_metrics = self.calculate_loss(batch)

            if self.only_target_model_gradient_clipping:
                # Clip target model params (Wc + norms) globally as a normal model,
                # then propagate those clipped Wc grads to projection params unclipped.
                target_params_with_grad = [
                    p
                    for p in self.model.target_model.parameters()
                    if p.grad is not None
                ]
                target_norm = get_global_grad_norm(target_params_with_grad)
                if self.gradient_clipping:
                    torch.nn.utils.clip_grads_with_norm_(
                        target_params_with_grad, self.gradient_clipping, target_norm
                    )
                projection_clip = (
                    None
                    if self.only_target_model_gradient_clipping == "no_projection_clip"
                    else self.only_target_model_gradient_clipping
                )
                total_grad_norm, projection_grad_norms, head_norm, embedding_norm = (
                    self.model.pass_gradient_to_projections(
                        self.block_optimizers,
                        self.block_schedulers,
                        gradient_clipping=projection_clip,
                        shared_gradient_norms=False,
                    )
                )
                grad_norm = target_norm
            elif self.only_compress_model_gradient_clipping:
                total_grad_norm, projection_grad_norms, head_norm, embedding_norm = (
                    self.model.pass_gradient_to_projections(
                        self.block_optimizers,
                        self.block_schedulers,
                        self.gradient_clipping,
                        shared_gradient_norms=False,
                    )
                )
                grad_norm = total_grad_norm
                self._clip_model_grads(grad_norm)
            else:
                total_grad_norm, projection_grad_norms, head_norm, embedding_norm = (
                    self.model.pass_gradient_to_projections(
                        self.block_optimizers,
                        self.block_schedulers,
                        self.gradient_clipping,
                        shared_gradient_norms=True,
                    )
                )
                grad_norm = total_grad_norm
                self._clip_model_grads(grad_norm)

            self.log_metrics(loss_metrics, grad_norm)
            self.metric_logger.log("train/total_grad_norm", total_grad_norm.item())
            self.log_projection_grad_norms(
                projection_grad_norms, head_norm, embedding_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            if self._should_save_checkpoint:
                self.save_checkpoint()

            if self._should_save_final_checkpoint:
                self.save_checkpoint()

            if self._should_evaluate:
                self.eval()

            self.metric_logger.flush()  # <--- THIS SENDS TO WANDB

    def log_projection_grad_norms(
        self, projection_grad_norms, head_norm=None, embedding_norm=None
    ):
        if not projection_grad_norms:
            return
        total_proj_norm = torch.tensor([n.item() for n in projection_grad_norms]).norm()
        self.metric_logger.log("train/projection_grad_norm", total_proj_norm.item())
        for i, norm in enumerate(projection_grad_norms):
            self.metric_logger.log(f"train/projection_grad_norm_block_{i}", norm.item())
        if head_norm is not None:
            self.metric_logger.log("train/projection_grad_norm_head", head_norm.item())
        if embedding_norm is not None:
            self.metric_logger.log(
                "train/projection_grad_norm_embedding", embedding_norm.item()
            )

    def save_checkpoint(self):
        checkpoint_folder = step_checkpoint_path(self.checkpoint.save.path, self.step)
        save_type = self.checkpoint.save.type  # "nano" | "nano_and_hf" | "hf_only"

        if save_type in ("nano", "nano_and_hf"):
            dcp.save(
                self.model.state_dict(), checkpoint_id=f"{checkpoint_folder}/model"
            )
            dcp.save(
                self.optimizer.state_dict(),
                checkpoint_id=f"{checkpoint_folder}/optimizer",
            )
            if self.block_optimizers is not None:
                for idx, block_optimizer in enumerate(self.block_optimizers):
                    dcp.save(
                        block_optimizer.state_dict(),
                        checkpoint_id=f"{checkpoint_folder}/block_optimizer_{idx}",
                    )
            # TODO: add scheduler saving
            logger.info(f"Saved nano checkpoint after step {self.step}")

        if save_type in ("nano_and_hf", "hf_only"):
            self.save_hf_checkpoint(f"{checkpoint_folder}/hf")

    def save_hf_checkpoint(self, save_path):
        """Save Wc weights as a HuggingFace LLaMA model (fp32) for lm_eval harness.

        prepare_compressed_weights() is called at the start of each training step,
        so target_model already holds current Wc weights when this is called.
        target_model.embedding is None (set in create_model), so we compute the
        projected embedding separately and add it to the state dict before remapping.

        All collective ops (get_model_state_dict, DTensor.full_tensor) must be called
        on every rank; only rank 0 writes files.
        """
        from src.projected_compression.convert_memeff_to_hf import (
            _load_pc_state_dict_to_hf,
        )
        from transformers import AutoTokenizer

        # Gather full Wc state dict from FSDP2-sharded target_model (collective)
        target_sd = get_model_state_dict(
            model=self.model.target_model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )

        # Compute projected embedding on all ranks (DTensor.full_tensor is collective),
        # then use on rank 0 only.
        with torch.no_grad():
            src_emb = self.model.source_model.embedding.weight
            proj_emb = self.model.projections.embedding
            aux_emb = self.model.projections.auxiliary_embedding_weights.weight

            src_emb = src_emb.full_tensor() if isinstance(src_emb, DTensor) else src_emb
            proj_emb = (
                proj_emb.full_tensor() if isinstance(proj_emb, DTensor) else proj_emb
            )
            aux_emb = aux_emb.full_tensor() if isinstance(aux_emb, DTensor) else aux_emb

            embedding = (
                src_emb.float().cpu() @ proj_emb.float().cpu().T + aux_emb.float().cpu()
            )

        if int(os.environ.get("RANK", "0")) == 0:
            target_sd["embedding"] = embedding
            hf_model = _load_pc_state_dict_to_hf(target_sd, self.original_llama_path)
            hf_model.save_pretrained(save_path)
            tokenizer = AutoTokenizer.from_pretrained(self.original_llama_path)
            tokenizer.save_pretrained(save_path)
            logger.info(f"Saved HF checkpoint at step {self.step} to '{save_path}'")
