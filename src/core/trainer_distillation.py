import os
import time
import torch
import torch.nn.functional as F
from attr import define
from typing import Optional
import torch.distributed as dist
import logging

from src.core.trainer import Trainer
from src.core.metric_loggers import AveMetric
from src.core.utils import create_batch_fingerprint

logger = logging.getLogger(__name__)

@define(slots=False)
class TrainerDistillation(Trainer):
    """
    Trainer for online knowledge distillation from a teacher model to a student model.
    Inherits from Trainer and overrides methods for distillation-specific behavior.
    
    Additional Args:
        teacher_model: The teacher model (frozen) used for distillation
        distillation_alpha: Weight for distillation loss (0.0 to 1.0, default 0.5)
        distillation_temperature: Temperature for softening probability distributions (default 2.0)
    """
    teacher_model: torch.nn.Module
    distillation_alpha: float
    distillation_temperature: float
    teacher_distributed: Optional[dict]

    def __attrs_post_init__(self):
        # Call parent initialization
        super().__attrs_post_init__()
        
        # Freeze teacher model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # Add distillation-specific metrics
        self.total_loss_averaged_100 = AveMetric(100, "steps/100/train/total_loss")
        self.distill_loss_averaged_100 = AveMetric(100, "steps/100/train/distill_loss")
        
        logger.info(f"Distillation trainer initialized with alpha={self.distillation_alpha}, "
                   f"temperature={self.distillation_temperature}")

    @property
    def student_model(self):
        """Alias for clarity in distillation context"""
        return self.model

    def compute_distillation_loss(self, student_logits, teacher_logits):
        """
        Compute distillation loss using KL divergence with temperature scaling.
        
        Args:
            student_logits: Logits from student model [batch, seq_len, vocab_size]
            teacher_logits: Logits from teacher model [batch, seq_len, vocab_size]
        
        Returns:
            distillation_loss: KL divergence loss
        """
        # Apply temperature scaling
        student_log_probs = F.log_softmax(
            student_logits / self.distillation_temperature, dim=-1
        )
        teacher_probs = F.softmax(
            teacher_logits / self.distillation_temperature, dim=-1
        )
        
        # KL divergence loss
        kl_loss = F.kl_div(
            student_log_probs.flatten(0, -2),
            teacher_probs.flatten(0, -2),
            reduction="batchmean"
        )
        
        # Scale by temperature^2 to normalize
        return kl_loss * (self.distillation_temperature ** 2)

    def calculate_loss(self, batch):
        """Override to compute both CE loss and distillation loss"""
        def _compute_losses(input_ids, target_ids):
            """Compute both CE loss and distillation loss"""
            # Student forward pass
            student_logits = self.model(input_ids)
            
            # Teacher forward pass (no gradients)
            with torch.no_grad():
                teacher_logits = self.teacher_model(input_ids)
            
            # Move target_ids to same device as student_logits
            target_ids = target_ids.to(student_logits.device)
            
            # Cross-entropy loss (standard supervised loss)
            ce_loss = F.cross_entropy(
                student_logits.flatten(0, -2),
                target_ids.reshape(-1).long(),
                reduction="mean"
            )
            
            # Distillation loss (KL divergence between student and teacher)
            distill_loss = self.compute_distillation_loss(student_logits, teacher_logits)
            
            # Combined loss
            total_loss = (
                (1.0 - self.distillation_alpha) * ce_loss +
                self.distillation_alpha * distill_loss
            )
            
            total_loss = total_loss / self.gradient_accumulation_steps
            ce_loss = ce_loss / self.gradient_accumulation_steps
            distill_loss = distill_loss / self.gradient_accumulation_steps
            
            return total_loss, ce_loss, distill_loss

        total_losses = []
        ce_losses = []
        distill_losses = []
        
        for batch_chunk in batch.chunk(self.gradient_accumulation_steps):
            input_ids, target_ids = self._preprocess_input(batch_chunk)
            input_ids = input_ids.to(self.device)
            
            if self.model.training:
                self._update_processed_tokens(input_ids)

            total_loss, ce_loss, distill_loss = _compute_losses(input_ids, target_ids)
            
            if self.model.training:
                total_loss.backward()
            
            total_losses.append(total_loss.item())
            ce_losses.append(ce_loss.item())
            distill_losses.append(distill_loss.item())

        # Average and synchronize across devices
        avg_total_loss = torch.tensor(total_losses, device=self.device).sum()
        avg_ce_loss = torch.tensor(ce_losses, device=self.device).sum()
        avg_distill_loss = torch.tensor(distill_losses, device=self.device).sum()
        
        if dist.is_initialized():
            dist.all_reduce(avg_total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg_ce_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg_distill_loss, op=dist.ReduceOp.SUM)
        
        world_size = float(os.environ["WORLD_SIZE"])
        
        # Store individual losses for logging
        self._last_ce_loss = avg_ce_loss / world_size
        self._last_distill_loss = avg_distill_loss / world_size
        
        return avg_total_loss / world_size

    def log_metrics(self, loss, grad_norm):
        """Override to add distillation-specific metrics"""
        # Call parent log_metrics
        self.metric_logger.log("step", self.step, self.step)
        self.metric_logger.log(
            "steps/train/lr", self.step, (self.scheduler.get_last_lr()[0])
        )
        self.metric_logger.log("steps/train/grad_norm", self.step, grad_norm.item())
        self.metric_logger.log(
            "steps/train/processed_tokens", self.step, self.processed_tokens
        )

        self.metric_logger.log(
            "tokens/lr", self.processed_tokens, (self.scheduler.get_last_lr()[0])
        )
        self.metric_logger.log(
            "tokens/train/grad_norm", self.processed_tokens, grad_norm.item()
        )

        self.time_diff_averaged_100.log(self.metric_logger, self.step, time.time())

        
        # Add distillation-specific metrics
        if hasattr(self, '_last_ce_loss'):
            self.metric_logger.log("steps/train/loss", self.step, self._last_ce_loss.item()) # dont be confused! - `loss` is cross entropy loss on language modeling; `total_loss` is training loss!
            self.metric_logger.log("tokens/train/loss", self.processed_tokens, self._last_ce_loss.item())
            self.metric_logger.log("steps/train/total_loss", self.step, loss.item())
            self.metric_logger.log("tokens/train/total_loss", self.processed_tokens, loss.item())
            self.metric_logger.log("steps/train/distill_loss", self.step, self._last_distill_loss.item())
            self.metric_logger.log("tokens/train/distill_loss", self.processed_tokens, self._last_distill_loss.item())
            
            self.loss_averaged_100.log(self.metric_logger, self.step, self._last_ce_loss.item())
            self.total_loss_averaged_100.log(self.metric_logger, self.step, loss.item())
            self.distill_loss_averaged_100.log(self.metric_logger, self.step, self._last_distill_loss.item())
        else:
            self.metric_logger.log("steps/train/loss", self.step, loss.item())
            self.metric_logger.log("tokens/train/loss", self.processed_tokens, loss.item())

            self.loss_averaged_100.log(self.metric_logger, self.step, loss.item())

        self.metric_logger.flush_accumulated_metrics(self.step)


    def eval(self):
        """Override to track distillation metrics during evaluation"""
        self.model.eval()
        saved_step = self.step
        self.metric_logger.set_step(None)
        
        losses = []
        ce_losses = []
        distill_losses = []
        eval_fingerprint = []
        
        with torch.no_grad():
            for _ in range(self.n_eval_steps):
                batch = next(self.eval_iterator)
                batch_fingerprint = create_batch_fingerprint(batch)
                eval_fingerprint.extend(batch_fingerprint)
                batch = batch.to(self.device)
                
                loss = self.calculate_loss(batch)
                losses.append(loss.item())
                
                if hasattr(self, '_last_ce_loss'):
                    ce_losses.append(self._last_ce_loss.item())
                    distill_losses.append(self._last_distill_loss.item())
                
                self.metric_logger.flush_accumulated_metrics(self.step)
            
            avg_loss = torch.tensor(losses).mean()
            
            if ce_losses:
                avg_ce_loss = torch.tensor(ce_losses).mean()
                avg_distill_loss = torch.tensor(distill_losses).mean()
                self.metric_logger.log("steps/eval/loss", self.step, avg_ce_loss.item())  # dont be confused! - `loss` is cross entropy loss on language modeling; `total_loss` is training loss!
                self.metric_logger.log("tokens/eval/loss", self.processed_tokens, avg_ce_loss.item()) 
                self.metric_logger.log("steps/eval/distill_loss", self.step, avg_distill_loss.item())
                self.metric_logger.log("tokens/eval/distill_loss", self.processed_tokens, avg_distill_loss.item())

                self.metric_logger.log("steps/eval/total_loss", self.step, avg_loss.item())
                self.metric_logger.log("tokens/eval/total_loss", self.processed_tokens, avg_loss.item())
            else:
                self.metric_logger.log("steps/eval/loss", self.step, avg_loss.item())
                self.metric_logger.log("tokens/eval/loss", self.processed_tokens, avg_loss.item())
                

        if self._should_log_eval_input:
            self.metric_logger.log(f"steps/eval/batch", self.step, str(eval_fingerprint))
        
        self.step = saved_step

