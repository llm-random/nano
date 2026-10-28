import itertools
import os
import time
from attr import define
import torch
import torch.nn.functional as F
from typing import Optional
from torch.utils.data import IterableDataset
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

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

    # Checkpoint save types this trainer can actually write. Subclasses widen this;
    # anything outside it is rejected before training starts.
    SUPPORTED_SAVE_TYPES = ("nano", "huggingface")

    # Of those, the ones written once when training ends rather than on the periodic
    # interval. Subclasses that add such a type widen this too.
    FINAL_ONLY_SAVE_TYPES = ("huggingface",)

    def __attrs_post_init__(self):
        self._validate_save_type()
        self.processed_tokens = self.training_state["processed_tokens"]
        self.start_step = self.training_state["next_step"]
        self.device = next(self.model.parameters()).device
        self._has_moe_modules = any(
            getattr(module, "is_moe", False) for module in self.model.modules()
        )

        self.eval_iterator = None
        if self.eval_dataloader is not None and hasattr(
            self.eval_dataloader, "__iter__"
        ):
            self.eval_iterator = iter(self.eval_dataloader)
        self.step = self.start_step - 1

        # Resuming has to advance the eval stream to where it was -- but only when there
        # is one. eval_interval is 0 whenever eval is switched off, which 31 shipped
        # configs do (the PCD-50 repro among them), and dividing by it raised
        # ZeroDivisionError before training began: those runs could not be resumed at
        # all. eval_dataloader: None leaves no iterator to advance either.
        if (
            self.start_step > 0
            and self.eval_interval > 0
            and self.eval_iterator is not None
        ):
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

    def _batches_from_start_step(self):
        """Training batches, fast-forwarded past the steps a previous run already took.

        The datasets are deterministic iterables that reseed in __iter__ (see
        AbstractDataset), and train() consumes exactly one batch per step. A resumed run
        therefore gets a *fresh* iterator starting at batch 0 and re-trains on data it
        has already seen, while the step counter carries on from the checkpoint.

        That fails quietly in the worst way: the loss after a resume comes out lower,
        not higher, because the model is being shown examples it already learned.

        Skipping start_step batches reproduces the sequence an uninterrupted run would
        have seen. It costs one pass over the skipped batches, which is the price of a
        dataset that cannot seek.
        """
        if not self.start_step:
            return self.train_dataloader
        logger.info(
            f"Resuming at step {self.start_step}: skipping that many batches so the run "
            "does not repeat data it has already trained on."
        )
        return itertools.islice(self.train_dataloader, self.start_step, None)

    def train(self):
        for step, batch in zip(
            range(self.start_step, self.n_steps), self._batches_from_start_step()
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

        self.final_eval()

        if self._should_save_final_checkpoint:
            self.save_final_checkpoint()

    def final_eval(self):
        """Score the held-out set once, after training, and log it as `final_ce`.

        The periodic eval reads `n_eval_steps` batches so it stays cheap enough to run
        during training. This one reads the whole held-out set, which
        `_dataset/fineweb.yaml` pins to 8192 samples of 2048 tokens -- the same samples
        for every run, so the number is comparable across them.

        It runs before the final checkpoint on purpose: `pc_finalize` merges the
        projections and sets the parameters to None off rank 0, so after saving there is
        no model left to evaluate.
        """
        if self.eval_dataloader is None or not hasattr(
            self.eval_dataloader, "__iter__"
        ):
            logger.info("No eval dataloader; skipping the final eval.")
            return

        dataset = getattr(self.eval_dataloader, "dataset", None)
        max_samples = getattr(dataset, "max_samples", None)
        batches = iter(self.eval_dataloader)
        if max_samples is None:
            # An uncapped eval stream never ends, so read the periodic eval's budget
            # rather than hang. Set max_samples on the eval dataset for a fixed set.
            logger.warning(
                "Eval dataset has no max_samples, so the stream is endless; "
                f"final_ce will cover {self.n_eval_steps} batches instead of a fixed set."
            )
            batches = itertools.islice(batches, self.n_eval_steps)

        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in batches:
                batch = batch.to(self.device)
                loss_metrics = self.calculate_loss(batch)
                losses.append(loss_metrics.reported_loss.detach())
                self.metric_logger.flush_accumulated_metrics()

        if not losses:
            logger.warning("Final eval produced no batches; not logging final_ce.")
            return

        final_ce = torch.stack(losses).mean().item()
        logger.info(f"final_ce over {len(losses)} batches: {final_ce}")
        self.metric_logger.log("final_ce", final_ce)
        self.metric_logger.flush()

    def save_final_checkpoint(self):
        """Write the checkpoint that is produced once, when training ends.

        Subclasses that widen SUPPORTED_SAVE_TYPES override this to handle the type they
        added, and defer to super() for the rest.
        """
        if self.checkpoint.save.type in ("nano", "hf_only", "nano_and_hf"):
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
        else:
            raise ValueError(
                f"No final-checkpoint writer for save.type="
                f"'{self.checkpoint.save.type}' in {type(self).__name__}."
            )

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
        # Restart the eval stream every time. The iterator used to be created once and
        # carried forward, so each evaluation scored whatever came next rather than the
        # same held-out samples -- eval loss then moved between evaluations partly
        # because the data moved, which is not what an eval number is for. The dataset
        # reseeds in __iter__, so a fresh iterator replays the same samples in the same
        # order.
        self.eval_iterator = iter(self.eval_dataloader)
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
                # Plain clip_grad_norm_ on FSDP2 DTensor sharded grads only computes
                # the local shard norm without cross-rank reduction, underestimating
                # the global norm by ~sqrt(world_size). Compute it correctly:
                params = [p for p in self.model.parameters() if p.grad is not None]
                local_norm_sq = torch.tensor(0.0, device=self.device)
                for p in params:
                    g = p.grad
                    local_g = g.to_local() if hasattr(g, "to_local") else g
                    local_norm_sq += local_g.float().norm(2.0) ** 2
                if dist.is_initialized():
                    dist.all_reduce(local_norm_sq, op=dist.ReduceOp.SUM)
                total_norm = local_norm_sq.sqrt()
                torch.nn.utils.clip_grads_with_norm_(
                    params, self.gradient_clipping, total_norm
                )
                return total_norm

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

    def _validate_save_type(self):
        """Reject a checkpoint save.type this trainer cannot write, before training starts.

        Without this the run trains to completion and only then discovers it has nowhere
        to save to, which on a multi-hour job means losing the whole run.
        """
        save_type = None
        if self.checkpoint is not None and "save" in self.checkpoint:
            save_type = self.checkpoint.save.get("type")

        if save_type is None or save_type in self.SUPPORTED_SAVE_TYPES:
            return

        raise ValueError(
            f"Unknown checkpoint save.type='{save_type}'. {type(self).__name__} supports: "
            f"{', '.join(self.SUPPORTED_SAVE_TYPES)}."
        )

    def save_checkpoint(self):
        save_type = self.checkpoint.save.type  # see SUPPORTED_SAVE_TYPES
        checkpoint_folder = step_checkpoint_path(self.checkpoint.save.path, self.step)

        # Fail before writing anything rather than part-writing a checkpoint.
        self._validate_save_type()

        if save_type in self.FINAL_ONLY_SAVE_TYPES:
            # These are written once, when training ends, by train(). Reaching here means
            # periodic checkpointing was switched on for them, which used to write the
            # training state and no model at all -- a checkpoint interval that silently
            # produced nothing.
            raise NotImplementedError(
                f"Periodic checkpointing is not implemented for save.type='{save_type}', "
                "which is written once when training ends. Set "
                "checkpoint.save.interval: -1, or use save.type='nano' if you need "
                "intermediate checkpoints."
            )

        if save_type == "nano":
            if (
                isinstance(self.model, FSDP)
                or self.model.__module__
                == "torch.distributed.fsdp._fully_shard._fully_shard"
            ):
                # Sharded save
                state_dict = {
                    "app": TrainingState(self.model, self.optimizer, self.scheduler)
                }
                dcp.save(state_dict, checkpoint_id=checkpoint_folder)
                logger.info(f"Saved sharded model checkpoint in {checkpoint_folder}")
            else:
                # Non-sharded save
                if os.environ["RANK"] == "0":
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
