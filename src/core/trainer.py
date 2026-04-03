import os
import time
from attr import define
import torch
import torch.nn.functional as F
from typing import Optional
from torch.utils.data import IterableDataset
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from src.projected_compression.compression import finalize_projection_weights
from src.core.conversion_to_hf import save_to_llama_3_hf
import torch.distributed.checkpoint as dcp
import logging

from src.core.checkpointing import (
    TrainingState,
    get_full_checkpoint_path,
    save_training_state,
    step_checkpoint_path,
)
from src.core.metric_loggers import AveDiffMetric, AveMetric, MetricLogger
from src.core.utils import cast_state_dict_to_tensors, create_batch_fingerprint

logger = logging.getLogger(__name__)


@define(frozen=True)
class LossMetrics:
    total_loss: torch.Tensor
    reported_loss: torch.Tensor
    moe_load_balancing_loss: Optional[torch.Tensor] = None
    moe_router_z_loss: Optional[torch.Tensor] = None
    distill_loss: Optional[torch.Tensor] = None


@define(slots=False)
class Trainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    gradient_accumulation_steps: int
    training_state: dict
    n_steps: int
    train_dataloader: IterableDataset
    eval_dataloader: IterableDataset
    metric_logger: MetricLogger
    eval_interval: int
    n_eval_steps: int
    gradient_clipping: Optional[float]
    checkpoint: Optional[dict]
    learning_rate: float
    weight_decay: float
    distributed: Optional[dict]

    def __attrs_post_init__(self):
        self.processed_tokens = self.training_state["processed_tokens"]
        self.start_step = self.training_state["next_step"]
        self.device = next(self.model.parameters()).device
        self._has_moe_modules = any(
            getattr(module, "is_moe", False) for module in self.model.modules()
        )

        if self.eval_dataloader is not None and hasattr(
            self.eval_dataloader, "__iter__"
        ):
            self.eval_iterator = iter(self.eval_dataloader)
        self.step = self.start_step - 1

        if self.start_step > 0:
            n_skip_eval_batches = (
                (self.start_step - 1) // self.eval_interval * self.n_eval_steps
            )
            logger.debug(f"Skipping {n_skip_eval_batches} eval batches")
            for _ in range(n_skip_eval_batches):
                next(self.eval_iterator)

        self.loss_averaged_100 = AveMetric(100, "100/train/loss")
        if self._has_moe_modules:
            self.total_loss_averaged_100 = AveMetric(100, "100/train/total_loss")
        self.time_diff_averaged_100 = AveDiffMetric(100, "100/time", time.time())

    @property
    def _should_evaluate(self) -> bool:
        return (
            self.eval_interval > 0
            and self.step % self.eval_interval == 0
            and self.step != 0
        )

    @property
    def _should_log_eval_input(self) -> bool:
        return self.step % (self.eval_interval * 100) == 0

    @property
    def _should_save_checkpoint(self) -> bool:
        return (
            self.checkpoint.save.interval > 0
            and (self.step) % self.checkpoint.save.interval == 0
            and self.step != 0
            and self.checkpoint.save.path is not None
        )

    @property
    def _should_save_final_checkpoint(self) -> bool:
        return (
            not self._should_save_checkpoint  # checkpoint was already saved
            and self.step >= self.n_steps - 1
            and self.checkpoint.save.path is not None
        )

    def train(self):
        for step, batch in zip(
            range(self.start_step, self.n_steps), self.train_dataloader
        ):
            self.step = step
            self.metric_logger.set_step(step)
            self.metric_logger.set_tokens(self.processed_tokens)
            self.model.train()
            loss_metrics = self.calculate_loss(batch)

            grad_norm = self.clip_gradient()

            self.log_metrics(loss_metrics, grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            if self._should_save_checkpoint:
                self.save_checkpoint()

            if self._should_evaluate:
                self.eval()

            self.metric_logger.flush()

        if self._should_save_final_checkpoint:
            if self.checkpoint.save.type == "nano":
                self.save_checkpoint()
            elif self.checkpoint.save.type == "huggingface":
                # self.model.unshard() # alternative that might not work for a very large > 1gpu memory models
                model_state_dict = self.model.state_dict()
                full_state = cast_state_dict_to_tensors(model_state_dict)

                if os.environ["RANK"] == "0":
                    dmodel, dff, n_att_heads, n_kvatt_heads, head_dim, nlayers = (
                        self.model.encoder.get_model_dimensions()
                    )

                    save_to_llama_3_hf(  # dev fixed values
                        full_state,
                        save_dir=get_full_checkpoint_path(self.checkpoint.save.path),
                        dmodel=dmodel,
                        dff=dff,
                        n_att_heads=n_att_heads,
                        n_kvatt_heads=n_kvatt_heads,
                        head_dim=head_dim,
                        nlayers=nlayers,
                    )
            elif self.checkpoint.save.type == "pc_finalize":
                self.save_pc_finalized_checkpoint()

    def _preprocess_input(self, batch):  # TODO test it
        input_ids = batch[:, :-1].contiguous()
        target_ids = batch[:, 1:].contiguous()

        return input_ids, target_ids

    def calculate_loss(self, batch) -> LossMetrics:
        def _hack_for_python_garbage_collection(input_ids, target_ids):
            """we want to have no reference to model output while backpropagating to allow torch to free memory,
            so we wrap loss calculation in a function"""
            predicted_ids = self.model(input_ids)

            # Tensors should be on the same device for loss calculation #TODO check
            target_ids = target_ids.to(predicted_ids.device)

            mask_loss = F.cross_entropy(
                predicted_ids.flatten(0, -2),
                target_ids.reshape(-1).long(),
                reduction="none",
            )
            # Keep the reported loss as pure CE so train/loss stays comparable to eval/loss;
            # MoE auxiliary terms are optimized and logged separately.
            reported_loss = mask_loss.mean()
            loss = reported_loss
            moe_load_balancing_loss = torch.zeros((), device=predicted_ids.device)
            moe_router_z_loss = torch.zeros((), device=predicted_ids.device)
            if self.model.training and self._has_moe_modules:
                moe_load_balancing_loss = self._calculate_moe_load_balancing_loss(
                    device=predicted_ids.device,
                )
                moe_router_z_loss = self._calculate_moe_router_z_loss(
                    device=predicted_ids.device,
                )
                loss = loss + moe_load_balancing_loss + moe_router_z_loss
            reported_loss = reported_loss / self.gradient_accumulation_steps
            loss = loss / self.gradient_accumulation_steps
            moe_load_balancing_loss = (
                moe_load_balancing_loss / self.gradient_accumulation_steps
            )
            moe_router_z_loss = moe_router_z_loss / self.gradient_accumulation_steps
            return loss, reported_loss, moe_load_balancing_loss, moe_router_z_loss

        total_losses = []
        reported_losses = []
        moe_load_balancing_losses = []
        moe_router_z_losses = []
        for batch_chunk in batch.chunk(self.gradient_accumulation_steps):
            input_ids, target_ids = self._preprocess_input(batch_chunk)
            input_ids = input_ids.to(self.device)
            if self.model.training:
                self._update_processed_tokens(input_ids)

            loss, reported_loss, moe_load_balancing_loss, moe_router_z_loss = (
                _hack_for_python_garbage_collection(input_ids, target_ids)
            )
            if self.model.training:
                loss.backward()
            total_losses.append(loss.detach())
            reported_losses.append(reported_loss.detach())
            moe_load_balancing_losses.append(moe_load_balancing_loss.detach())
            moe_router_z_losses.append(moe_router_z_loss.detach())

        # gloo backend supports only sum reduce operation, therfore we first divide by world size and then sum
        avg_total_loss = torch.stack(total_losses).sum()
        avg_reported_loss = torch.stack(reported_losses).sum()
        avg_moe_load_balancing_loss = torch.stack(moe_load_balancing_losses).sum()
        avg_moe_router_z_loss = torch.stack(moe_router_z_losses).sum()
        if dist.is_initialized():
            dist.all_reduce(avg_total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg_reported_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg_moe_load_balancing_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg_moe_router_z_loss, op=dist.ReduceOp.SUM)

        world_size = float(os.environ["WORLD_SIZE"])
        return LossMetrics(
            total_loss=avg_total_loss / world_size,
            reported_loss=avg_reported_loss / world_size,
            moe_load_balancing_loss=avg_moe_load_balancing_loss / world_size,
            moe_router_z_loss=avg_moe_router_z_loss / world_size,
        )

    def _calculate_weighted_moe_loss(
        self,
        device,
        loss_attr,
        factor_attr,
    ):
        loss = torch.zeros((), device=device)
        for module in self.model.modules():
            module_loss = getattr(module, loss_attr, None)
            factor = getattr(module, factor_attr, 0.0)
            if module_loss is not None and factor:
                loss = loss + module_loss.to(device=device) * factor
        return loss

    def _calculate_moe_load_balancing_loss(self, device):
        return self._calculate_weighted_moe_loss(
            device=device,
            loss_attr="moe_load_balancing_loss",
            factor_attr="moe_load_balancing_loss_factor",
        )

    def _calculate_moe_router_z_loss(self, device):
        return self._calculate_weighted_moe_loss(
            device=device,
            loss_attr="router_z_loss",
            factor_attr="moe_router_z_loss_factor",
        )

    def eval(self):
        self.model.eval()
        saved_step = self.step
        self.metric_logger.set_step(None)  # disables heavy logging
        losses = []
        eval_fingerprint = []
        with torch.no_grad():
            for _ in range(self.n_eval_steps):
                batch = next(self.eval_iterator)
                batch_fingerprint = create_batch_fingerprint(batch)
                eval_fingerprint.extend(batch_fingerprint)
                batch = batch.to(self.device)
                loss_metrics = self.calculate_loss(batch)
                losses.append(loss_metrics.reported_loss.detach())
                self.metric_logger.flush_accumulated_metrics()
            avg_loss = torch.stack(losses).mean()
            self.metric_logger.log("eval/loss", avg_loss.item())

        if self._should_log_eval_input:
            self.metric_logger.log("eval/batch", str(eval_fingerprint))

        self.step = saved_step  # Restore step
        self.metric_logger.set_step(saved_step)

    def clip_gradient(self):
        if self.gradient_clipping is not None:
            if isinstance(self.model, FSDP):
                return self.model.clip_grad_norm_(self.gradient_clipping)
            else:
                return torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clipping
                )

    def _update_processed_tokens(self, batch):
        self.processed_tokens += batch.numel() * int(os.environ["WORLD_SIZE"])

    def log_metrics(self, loss_metrics: LossMetrics, grad_norm):
        self.metric_logger.set_tokens(self.processed_tokens)
        self.metric_logger.log("train/loss", loss_metrics.reported_loss.item())
        if self._has_moe_modules:
            self.metric_logger.log("train/total_loss", loss_metrics.total_loss.item())
            self.metric_logger.log(
                "train/moe_load_balancing_loss",
                loss_metrics.moe_load_balancing_loss.item(),
            )
            self.metric_logger.log(
                "train/moe_router_z_loss",
                loss_metrics.moe_router_z_loss.item(),
            )
        self.metric_logger.log("train/lr", self.scheduler.get_last_lr()[0])
        if grad_norm is not None:
            self.metric_logger.log("train/grad_norm", grad_norm.item())

        self.loss_averaged_100.log(
            self.metric_logger, loss_metrics.reported_loss.item()
        )
        if self._has_moe_modules:
            self.total_loss_averaged_100.log(
                self.metric_logger, loss_metrics.total_loss.item()
            )
        self.time_diff_averaged_100.log(self.metric_logger, time.time())

        self.metric_logger.flush_accumulated_metrics()

    def save_checkpoint(self):
        if (
            isinstance(self.model, FSDP)
            or self.model.__module__
            == "torch.distributed.fsdp._fully_shard._fully_shard"
        ):
            # Sharded save
            checkpoint_folder = step_checkpoint_path(
                self.checkpoint.save.path, self.step
            )
            state_dict = {
                "app": TrainingState(self.model, self.optimizer, self.scheduler)
            }
            dcp.save(state_dict, checkpoint_id=checkpoint_folder)
            logger.info(f"Saved sharded model checkpoint in {checkpoint_folder}")
        else:
            # Non-sharded save
            if os.environ["RANK"] == "0":
                checkpoint_folder = step_checkpoint_path(
                    self.checkpoint.save.path, self.step
                )
                os.makedirs(checkpoint_folder, exist_ok=True)
                checkpoint_path = f"{checkpoint_folder}/{self.checkpoint.save.model_checkpoint_filename}"
                state_to_save = {
                    "model": self.model.state_dict(),
                    "optim": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                }
                torch.save(state_to_save, checkpoint_path)
                logger.info(
                    f"Saved non-sharded model checkpoint in '{checkpoint_path}'"
                )

        if os.environ["RANK"] == "0":
            save_training_state(
                save_config=self.checkpoint.save,
                step=self.step,
                processed_tokens=self.processed_tokens,
                metric_logger=self.metric_logger,
            )

    def save_pc_finalized_checkpoint(self):
        with torch.no_grad():
            finalize_projection_weights(self.model)
            model_state_dict = cast_state_dict_to_tensors(self.model.state_dict())

        if os.environ["RANK"] == "0":
            checkpoint_folder = step_checkpoint_path(
                self.checkpoint.save.path, self.step
            )
            os.makedirs(checkpoint_folder, exist_ok=True)
            checkpoint_path = (
                f"{checkpoint_folder}/{self.checkpoint.save.model_checkpoint_filename}"
            )
            state_to_save = {
                "model": model_state_dict,
                "optim": None,
                "scheduler": None,
            }
            torch.save(state_to_save, checkpoint_path)
            logger.info(
                f"Saved non-sharded Finalized PC model checkpoint in '{checkpoint_path}'"
            )
