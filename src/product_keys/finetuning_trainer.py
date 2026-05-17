import os
from attr import define, field
import logging 
import torch
import torch.nn
from torch.utils.data import DataLoader, IterableDataset
from typing import Optional, override

from src.product_keys.trainer import TrainerWithVocabSize
from src.core.utils import create_batch_fingerprint
from src.core.metric_loggers import AveDiffMetric, AveMetric, MetricLogger, WandbLogger
from src.product_keys.model_sequence_classifiaction import ModelSequenceClassification
from src.product_keys.datasets import GlueDataset, FullIterDataset, GlueLengthSplitDataset
import math


logger = logging.getLogger(__name__)

def show_gradients(model: torch.nn.Module):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                # Calculate some basic statistics to see what the gradients look like
                grad_mean = param.grad.abs().mean().item()
                grad_max = param.grad.abs().max().item()
                print(f"Layer: {name:<30} | Grad Mean: {grad_mean:.6f} | Grad Max: {grad_max:.6f}")
            else:
                print(f"Layer: {name:<30} | NO GRADIENT DEPOSITED (Disconnected layer?)")


def create_classifier_model(model: torch.nn.Module,
                            device: torch.device,
                            d_model: int,
                            num_labels: int,
                            distributed: Optional[dict] = None) -> torch.nn.Module:
    logger.debug("Printing model shapes...")
    for name, layer in model.named_modules():
        logger.debug(f"Layer name: {name}")

        if hasattr(layer, 'weight') and layer.weight is not None:
            logger.debug(f"Layer: {name} | Size: {layer.weight.shape}")
    
    model = ModelSequenceClassification(model, d_model=d_model, num_labels=num_labels,
                                        distributed=distributed).to(device)

    model_dtypes = set([param.dtype for param in model.parameters()])
    logger.debug(f"Model dtypes: {list(model_dtypes)}")
    return model


@define(slots=False)
class FinetuningTrainer(TrainerWithVocabSize):
    d_model: int
    num_labels: int
    full_eval_rows: Optional[int] = None
    freeze_backbone: bool = field(default=False)
    trainable_modules: list = field(factory=list)
    loss_fct = torch.nn.CrossEntropyLoss()

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

        logger.info(f"{self.distributed=}")

        self.model = create_classifier_model(
            model=self.model,
            device=self.device,
            d_model=self.d_model,
            num_labels=self.num_labels,
            distributed=self.distributed)

        if self.freeze_backbone:
            self._freeze_model_layers()
            
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer.param_groups[0]['params'] = trainable_params

        self.eval_function = self.eval
        eval_dataset = self.eval_dataloader.dataset

        is_length_split = isinstance(eval_dataset, GlueLengthSplitDataset)
        assert not (is_length_split and self.full_eval_rows is None), "full_eval_rows is required for length split dataset in eval"

        if self.full_eval_rows is not None:
            self.eval_dataset = eval_dataset
            self.eval_function = self.full_eval_length_split if is_length_split else self.eval
        
        if is_length_split:
            self.split_values = eval_dataset.split_values + [float("inf")]

        self.full_eval_batches = (
            int(math.ceil(self.full_eval_rows / self.eval_dataloader.batch_size)) 
            if self.full_eval_rows is not None else None
        )
        
    def _freeze_model_layers(self):
        logger.info("Freezing backbone layers...")
        for name, param in self.model.named_parameters():
            should_train = any(mod in name for mod in self.trainable_modules)
            if not should_train:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def _print_trainable_parameters(self):
        print("Trainable parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)

    def train(self):
        for step, batch in zip(
            range(self.start_step, self.n_steps), self.train_dataloader
        ):
            self.step = step
            self.metric_logger.set_step(step)
            self.model.train()

            loss = self.calculate_loss(batch)

            grad_norm = self.clip_gradient()

            self.log_metrics(loss, grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            if self._should_save_checkpoint:
                self.save_checkpoint()

            if self._should_evaluate:
                self.eval_function()

        if self._should_save_final_checkpoint:
            self.save_checkpoint()
        
        self.eval_function()

    @override
    def eval(self):
        self.model.eval()
        saved_step = self.step
        self.metric_logger.set_step(None)  # disables heavy logging
        losses = []
        eval_fingerprint = []
        with torch.no_grad():
            for _ in range(self.n_eval_steps):
                batch = next(self.eval_iterator)
                text, _, _ = batch
                text_fingerprint = create_batch_fingerprint(text)
                eval_fingerprint.extend(text_fingerprint)
                loss = self.calculate_loss(batch)
                losses.append(loss.item())
                self.metric_logger.flush_accumulated_metrics(self.step)
            avg_loss = torch.tensor(losses).mean()
            self.metric_logger.log("steps/eval/loss", self.step, avg_loss.item())
            if not isinstance(self.metric_logger, (WandbLogger)):
                self.metric_logger.log(
                    "tokens/eval/loss", self.processed_tokens, avg_loss.item()
                )

        if self._should_log_eval_input:
            self.metric_logger.log(
                f"steps/eval/batch", self.step, str(eval_fingerprint)
            )

        self.step = saved_step 

    @override
    def calculate_loss(self, batch, is_dummy=False):
        texts, labels, attention_masks = batch
        
        def _hack_for_python_garbage_collection(texts_chunk, labels_chunk, attention_masks_chunk):
            logits = self.model(texts_chunk, attention_mask=attention_masks_chunk)
           
            loss = self.loss_fct(logits, labels_chunk)
            loss = loss / self.gradient_accumulation_steps
            return loss

        losses = []
        texts_chunks = texts.chunk(self.gradient_accumulation_steps)
        labels_chunks = labels.chunk(self.gradient_accumulation_steps)
        attention_masks_chunks = attention_masks.chunk(self.gradient_accumulation_steps)

        for texts_chunk, labels_chunk, attention_masks_chunk in zip(texts_chunks, labels_chunks, attention_masks_chunks):
            texts_chunk = texts_chunk.to(self.device)
            labels_chunk = labels_chunk.to(self.device)
            attention_masks_chunk = attention_masks_chunk.to(self.device)
            
            loss = _hack_for_python_garbage_collection(texts_chunk, labels_chunk, attention_masks_chunk)
            
            if self.model.training:
                loss.backward()

            losses.append(loss.item())

        if is_dummy:
            losses = [0.0] * len(losses)

        avg_loss = torch.tensor(losses, device=self.device).sum()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(avg_loss, op=torch.distributed.ReduceOp.SUM)

        world_size = float(os.environ.get("WORLD_SIZE", 1))
        return avg_loss / world_size

    def _get_full_eval_iterator(self):
        saved_step = self.step
        self.metric_logger.set_step(None)
        full_eval_dataloader = DataLoader(
            FullIterDataset(self.eval_dataset),
            batch_size=self.eval_dataloader.batch_size,
            collate_fn=self.eval_dataloader.collate_fn,
            pin_memory=self.eval_dataloader.pin_memory,
            num_workers=self.eval_dataloader.num_workers,
        )
        eval_iter = iter(full_eval_dataloader)
        return eval_iter


    def full_eval(self):
        self.model.eval()
        saved_step = self.step
        self.metric_logger.set_step(None)  # disables heavy logging
        losses = []
        eval_fingerprint = []
        eval_iter = self._get_full_eval_iterator()

        with torch.no_grad():
            for _ in range(self.full_eval_batches):
                batch = next(eval_iter)
                text, _, _ = batch
                text_fingerprint = create_batch_fingerprint(text)
                eval_fingerprint.extend(text_fingerprint)
                loss = self.calculate_loss(batch)
                losses.append(loss.item())
                self.metric_logger.flush_accumulated_metrics(self.step)
            avg_loss = torch.tensor(losses).mean()
            self.metric_logger.log("steps/eval/loss", self.step, avg_loss.item())
            if not isinstance(self.metric_logger, (WandbLogger)):
                self.metric_logger.log(
                    "tokens/eval/loss", self.processed_tokens, avg_loss.item()
                )

        if self._should_log_eval_input:
            self.metric_logger.log(
                f"steps/eval/batch", self.step, str(eval_fingerprint)
            )

        self.step = saved_step

    def full_eval_length_split(self):
        self.model.eval()
        saved_step = self.step
        self.metric_logger.set_step(None)  # disables heavy logging
        eval_fingerprint = []
        eval_iter = self._get_full_eval_iterator()

        losses = {split_value: [] for split_value in self.split_values}
        n_splits = len(self.split_values)

        def make_dummy_batch():
            seq_len = self.eval_dataset.sequence_length
            bsz = max(1, self.gradient_accumulation_steps)
            dummy_text = torch.zeros((bsz, seq_len), dtype=torch.long)
            dummy_labels = torch.zeros((bsz,), dtype=torch.long)
            dummy_mask = torch.zeros((bsz, seq_len), dtype=torch.bool)
            return (dummy_text, dummy_labels, dummy_mask)

        def handle_split_batch(batch, is_dummy=False):
            if not is_dummy:
                text, _, _ = batch
                text_fingerprint = create_batch_fingerprint(text)
                eval_fingerprint.extend(text_fingerprint)
            loss = self.calculate_loss(batch, is_dummy=is_dummy)
            return loss

        with torch.no_grad():
            while True:
                exhausted = torch.tensor([0.0], device=self.device)  # 0=ok, 1=done
                try:
                    batch_list = next(eval_iter)
                except StopIteration:
                    exhausted[0] = 1.0
                    batch_list = [None] * n_splits

                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(exhausted, op=torch.distributed.ReduceOp.MAX)

                if exhausted[0] > 0.0:
                    break

                has_data = torch.zeros(n_splits, device=self.device)
                for i, b in enumerate(batch_list):
                    if b is not None:
                        has_data[i] = 1.0

                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(has_data, op=torch.distributed.ReduceOp.MAX)

                for i in range(n_splits):
                    if has_data[i] == 0.0:
                        continue

                    batch = batch_list[i] if i < len(batch_list) else None
                    is_dummy = (batch is None)
                    if is_dummy:
                        batch = make_dummy_batch()

                    split_value = self.split_values[i]
                    loss = handle_split_batch(batch, is_dummy=is_dummy)

                    if not is_dummy:
                        losses[split_value].append(loss.item())

                self.metric_logger.flush_accumulated_metrics(self.step)

            min_value = 0
            for split_value, loss_list in losses.items():
                if len(loss_list) > 0:
                    avg_loss = torch.tensor(loss_list).mean()
                    self.metric_logger.log(f"steps/eval/loss_({min_value}-{split_value})", self.step, avg_loss.item())
                    if not isinstance(self.metric_logger, (WandbLogger)):
                        self.metric_logger.log(
                            f"tokens/eval/loss_({min_value}-{split_value})", self.processed_tokens, avg_loss.item()
                        )
                min_value = split_value

        if self._should_log_eval_input:
            self.metric_logger.log(
                f"steps/eval/batch", self.step, str(eval_fingerprint)
            )

        self.step = saved_step
