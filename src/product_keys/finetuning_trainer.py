import os
from attr import define, field
import logging 
import torch
import torch.nn
from torch.utils.data import DataLoader, IterableDataset
from typing import NamedTuple, Optional, override

from src.product_keys.trainer import TrainerWithVocabSize
from src.core.utils import create_batch_fingerprint
from src.core.metric_loggers import AveDiffMetric, AveMetric, MetricLogger, WandbLogger
from src.product_keys.model_sequence_classifiaction import ModelSequenceClassification
from src.product_keys.datasets import GlueDataset, FullIterDataset, GlueLengthSplitDataset
import math


class LossResult(NamedTuple):
    loss: torch.Tensor
    correct: int
    total: int
    tp: int = 0
    fp: int = 0
    fn: int = 0


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
            self.eval_function = self.full_eval_length_split if is_length_split else self.full_eval
        
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

            loss = self.calculate_loss(batch).loss

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
                loss = self.calculate_loss(batch).loss
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


    def _pad_eval_batch(self, texts, labels, attention_masks):
        """Optionally pad a batch to eval_dataloader.batch_size during eval.

        The base class does not pad (single-label batches are always full-size).
        Override in subclasses that need FSDP-safe uniform chunk counts.
        Returns (texts, labels, attention_masks, original_bsz, target_bsz).
        """
        return texts, labels, attention_masks, texts.size(0), self.eval_dataloader.batch_size

    def _forward_chunk(self, texts_chunk, labels_chunk, attention_masks_chunk, valid_in_chunk):
        """Run one gradient-accumulation chunk and return (loss, chunk_correct, chunk_total, chunk_tp, chunk_fp, chunk_fn).

        loss is already divided by gradient_accumulation_steps.
        """
        logits = self.model(texts_chunk, attention_mask=attention_masks_chunk)
        loss = self.loss_fct(logits, labels_chunk)
        loss = loss / self.gradient_accumulation_steps
        preds = logits.detach().argmax(dim=-1)
        chunk_correct = (preds == labels_chunk).sum().item()
        chunk_total = labels_chunk.size(0)
        return loss, chunk_correct, chunk_total, 0, 0, 0

    def _make_dummy_labels(self, bsz: int) -> torch.Tensor:
        """Return a zero label tensor for a dummy batch of the given batch size."""
        return torch.zeros((bsz,), dtype=torch.long)

    def _log_extra_metrics(self, result_totals: dict):
        """Log any metrics beyond loss/accuracy after a full_eval pass.

        `result_totals` contains summed tp, fp, fn keys if available.
        Override in subclasses that track additional metrics (e.g. micro-F1).
        """
        pass

    def _log_extra_metrics_split(self, split_key: str, result_totals: dict):
        """Same as _log_extra_metrics but for a length-split bucket."""
        pass


    @override
    def calculate_loss(self, batch, is_dummy=False) -> LossResult:
        texts, labels, attention_masks = batch

        texts, labels, attention_masks, original_bsz, target_bsz = self._pad_eval_batch(
            texts, labels, attention_masks
        )
        if is_dummy:
            original_bsz = 0

        losses = []
        correct = 0
        total = 0
        tp = 0
        fp = 0
        fn = 0

        texts_chunks = texts.chunk(self.gradient_accumulation_steps)
        labels_chunks = labels.chunk(self.gradient_accumulation_steps)
        attention_masks_chunks = attention_masks.chunk(self.gradient_accumulation_steps)

        chunk_size = target_bsz // self.gradient_accumulation_steps

        for c, (texts_chunk, labels_chunk, attention_masks_chunk) in enumerate(
            zip(texts_chunks, labels_chunks, attention_masks_chunks)
        ):
            texts_chunk = texts_chunk.to(self.device)
            labels_chunk = labels_chunk.to(self.device)
            attention_masks_chunk = attention_masks_chunk.to(self.device)

            valid_in_chunk = max(0, min((c + 1) * chunk_size, original_bsz) - c * chunk_size)

            # Wrap in a function to avoid Python closure issues with loop variables
            def _run_chunk(texts_chunk, labels_chunk, attention_masks_chunk, valid_in_chunk):
                if valid_in_chunk > 0:
                    return self._forward_chunk(
                        texts_chunk[:valid_in_chunk],
                        labels_chunk[:valid_in_chunk],
                        attention_masks_chunk[:valid_in_chunk],
                        valid_in_chunk,
                    )
                else:
                    # Still run a forward pass (needed for FSDP all-gather sync),
                    # but discard all outputs.
                    self._forward_chunk(texts_chunk, labels_chunk, attention_masks_chunk, 0)
                    return torch.tensor(0.0, device=self.device), 0, 0, 0, 0, 0

            loss, cc, ct, ctp, cfp, cfn = _run_chunk(
                texts_chunk, labels_chunk, attention_masks_chunk, valid_in_chunk
            )

            if self.model.training:
                loss.backward()

            losses.append(loss.item())
            correct += cc
            total += ct
            tp += ctp
            fp += cfp
            fn += cfn

        avg_loss = torch.tensor(losses, device=self.device).sum()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(avg_loss, op=torch.distributed.ReduceOp.SUM)

        world_size = float(os.environ.get("WORLD_SIZE", 1))
        return LossResult(
            loss=avg_loss / world_size,
            correct=correct,
            total=total,
            tp=tp,
            fp=fp,
            fn=fn,
        )

    # -------------------------------------------------------------------------
    # Full-dataset evaluation helpers
    # -------------------------------------------------------------------------

    def _get_full_eval_iterator(self):
        full_eval_dataloader = DataLoader(
            FullIterDataset(self.eval_dataset),
            batch_size=self.eval_dataloader.batch_size,
            collate_fn=self.eval_dataloader.collate_fn,
            pin_memory=self.eval_dataloader.pin_memory,
            num_workers=self.eval_dataloader.num_workers,
        )
        return iter(full_eval_dataloader)

    def _make_dummy_batch(self) -> tuple:
        seq_len = self.eval_dataset.sequence_length
        bsz = self.eval_dataloader.batch_size
        dummy_text = torch.zeros((bsz, seq_len), dtype=torch.long)
        dummy_labels = self._make_dummy_labels(bsz)
        dummy_mask = torch.zeros((bsz, seq_len), dtype=torch.bool)
        return (dummy_text, dummy_labels, dummy_mask)

    def _log_eval_results(self, losses: list, correct_count: int, total_count: int,
                          tp: int = 0, fp: int = 0, fn: int = 0,
                          name_suffix: str = ""):
        """Log loss, accuracy, and any extra metrics.

        name_suffix is appended to each metric name, e.g. '_(0-512)' for split evals.
        """
        avg_loss = torch.tensor(losses).mean()
        self.metric_logger.log(f"steps/eval/loss{name_suffix}", self.step, avg_loss.item())
        if not isinstance(self.metric_logger, WandbLogger):
            self.metric_logger.log(f"tokens/eval/loss{name_suffix}", self.processed_tokens, avg_loss.item())

        if total_count > 0:
            accuracy = correct_count / total_count
            self.metric_logger.log(f"steps/eval/accuracy{name_suffix}", self.step, accuracy)
            if not isinstance(self.metric_logger, WandbLogger):
                self.metric_logger.log(f"tokens/eval/accuracy{name_suffix}", self.processed_tokens, accuracy)

        self._log_extra_metrics({"tp": tp, "fp": fp, "fn": fn, "name_suffix": name_suffix})

    def full_eval(self):
        self.model.eval()
        saved_step = self.step
        self.metric_logger.set_step(None)  # disables heavy logging
        losses = []
        correct_count = 0
        total_count = 0
        tp_count = 0
        fp_count = 0
        fn_count = 0
        eval_fingerprint = []
        eval_iter = self._get_full_eval_iterator()

        with torch.no_grad():
            for _ in range(self.full_eval_batches):
                exhausted = torch.tensor([0.0], device=self.device)
                try:
                    batch = next(eval_iter)
                    is_dummy = False
                except StopIteration:
                    exhausted[0] = 1.0
                    is_dummy = True
                    batch = self._make_dummy_batch()

                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(exhausted, op=torch.distributed.ReduceOp.MAX)

                if exhausted[0] > 0.0:
                    break

                if not is_dummy:
                    text, _, _ = batch
                    eval_fingerprint.extend(create_batch_fingerprint(text))

                result = self.calculate_loss(batch, is_dummy=is_dummy)
                if not is_dummy:
                    losses.append(result.loss.item())
                    correct_count += result.correct
                    total_count += result.total
                    tp_count += result.tp
                    fp_count += result.fp
                    fn_count += result.fn
                self.metric_logger.flush_accumulated_metrics(self.step)

            if losses:
                self._log_eval_results(losses, correct_count, total_count, tp_count, fp_count, fn_count)

        if self._should_log_eval_input:
            self.metric_logger.log(f"steps/eval/batch", self.step, str(eval_fingerprint))

        self.step = saved_step

    def full_eval_length_split(self):
        self.model.eval()
        saved_step = self.step
        self.metric_logger.set_step(None)  # disables heavy logging
        eval_fingerprint = []
        eval_iter = self._get_full_eval_iterator()

        losses = {sv: [] for sv in self.split_values}
        correct_counts = {sv: 0 for sv in self.split_values}
        total_counts = {sv: 0 for sv in self.split_values}
        tp_counts = {sv: 0 for sv in self.split_values}
        fp_counts = {sv: 0 for sv in self.split_values}
        fn_counts = {sv: 0 for sv in self.split_values}
        n_splits = len(self.split_values)

        with torch.no_grad():
            batch_count = 0
            while True:
                exhausted = torch.tensor([0.0], device=self.device)
                try:
                    batch_list = next(eval_iter)
                except StopIteration:
                    exhausted[0] = 1.0
                    batch_list = [None] * n_splits

                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(exhausted, op=torch.distributed.ReduceOp.MAX)

                if exhausted[0] > 0.0:
                    break

                batch_count += 1
                if self.full_eval_batches is not None and batch_count > self.full_eval_batches:
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
                        batch = self._make_dummy_batch()
                    else:
                        text, _, _ = batch
                        eval_fingerprint.extend(create_batch_fingerprint(text))

                    split_value = self.split_values[i]
                    result = self.calculate_loss(batch, is_dummy=is_dummy)

                    if not is_dummy:
                        losses[split_value].append(result.loss.item())
                        correct_counts[split_value] += result.correct
                        total_counts[split_value] += result.total
                        tp_counts[split_value] += result.tp
                        fp_counts[split_value] += result.fp
                        fn_counts[split_value] += result.fn

                self.metric_logger.flush_accumulated_metrics(self.step)

            min_value = 0
            for split_value, loss_list in losses.items():
                if len(loss_list) > 0:
                    self._log_eval_results(
                        loss_list,
                        correct_counts[split_value],
                        total_counts[split_value],
                        tp_counts[split_value],
                        fp_counts[split_value],
                        fn_counts[split_value],
                        name_suffix=f"_({min_value}-{split_value})",
                    )
                min_value = split_value

        if self._should_log_eval_input:
            self.metric_logger.log(f"steps/eval/batch", self.step, str(eval_fingerprint))

        self.step = saved_step


"""
    Do not use for now.
    Introducing learned embeddings during finetuning
    can degrade performance if the model was pretrained without it.
"""
@define(slots=False)
class FinetuningTrainerTwoSentences(FinetuningTrainer):
    """Fine-tuning trainer for sentence-pair tasks (e.g. MNLI).

    Adds a learned segment embedding (token-type embedding) that is summed with
    the token embeddings produced by the backbone's embedding layer.  Tokens
    belonging to sentence A receive segment id 0; tokens at or after the first
    [SEP] token receive segment id 1.

    The segment_embedding module is attached directly to ``self.model`` so that
    its parameters are included in ``self.model.parameters()`` and are therefore
    visible to and updated by the optimizer.

    Requires the backbone to expose ``embed()`` and ``forward_from_embeddings()``
    (as defined in ``src.product_keys.model.LLM``).
    """

    sep_token_id: int = field(kw_only=True)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

        # Attach the segment embedding to self.model so the optimizer sees it.
        self.model.segment_embedding = torch.nn.Embedding(2, self.d_model).to(self.device)
        torch.nn.init.normal_(self.model.segment_embedding.weight, mean=0.0, std=0.02)
        logger.info(
            f"FinetuningTrainerTwoSentences: initialised segment_embedding "
            f"(2 x {self.d_model}) and attached it to self.model. "
            f"sep_token_id={self.sep_token_id}"
        )

        if torch.distributed.is_initialized():
            from torch.distributed.tensor import distribute_tensor, Replicate, DTensor
            device_mesh = None
            for p in self.model.parameters():
                if isinstance(p, DTensor):
                    device_mesh = p.device_mesh
                    break
            
            if device_mesh is not None:
                logger.info("FinetuningTrainerTwoSentences: Converting segment_embedding weight to a replicated DTensor.")
                replicated_weight = distribute_tensor(
                    self.model.segment_embedding.weight.data,
                    device_mesh,
                    [Replicate()]
                )
                self.model.segment_embedding.weight = torch.nn.Parameter(replicated_weight)

        # Re-collect trainable params so the newly added segment_embedding is
        # included (super().__attrs_post_init__ already ran, but it set params
        # before we attached segment_embedding).
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer.param_groups[0]['params'] = trainable_params

    def _get_token_type_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return a (B, T) tensor of 0s and 1s.

        Segment 0: [CLS] + sentence A tokens up to (but not including) [SEP].
        Segment 1: [SEP] token and everything after (sentence B + padding).
        """
        is_sep = (input_ids == self.sep_token_id)  # (B, T) bool
        # argmax returns the index of the first True; if no SEP is found it
        # returns 0, which is safe because segment 0 is the default anyway.
        sep_pos = is_sep.long().argmax(dim=1, keepdim=True)  # (B, 1)
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        token_type_ids = (positions >= sep_pos).long()  # (B, T)
        return token_type_ids

    @override
    def _forward_chunk(self, texts_chunk, labels_chunk, attention_masks_chunk, valid_in_chunk):
        backbone = self.model.backbone

        # Compute segment embeddings to inject after the token embedding lookup.
        token_type_ids = self._get_token_type_ids(texts_chunk)
        seg_embeds = self.model.segment_embedding(token_type_ids)  # (B, T, d_model)

        def _add_seg_embeds_hook(module, input, output):
            return output + seg_embeds

        handle = backbone.embedding.register_forward_hook(_add_seg_embeds_hook)
        try:
            logits = self.model(texts_chunk, attention_mask=attention_masks_chunk)
        finally:
            handle.remove()

        loss = self.loss_fct(logits, labels_chunk)
        loss = loss / self.gradient_accumulation_steps
        preds = logits.detach().argmax(dim=-1)
        chunk_correct = (preds == labels_chunk).sum().item()
        chunk_total = labels_chunk.size(0)
        return loss, chunk_correct, chunk_total, 0, 0, 0


class FinetuningTrainerMultiLabel(FinetuningTrainer):
    loss_fct = torch.nn.BCEWithLogitsLoss()

    @override
    def _make_dummy_labels(self, bsz: int) -> torch.Tensor:
        return torch.zeros((bsz, self.num_labels), dtype=torch.float32)

    @override
    def _pad_eval_batch(self, texts, labels, attention_masks):
        """Pad to eval batch size so all ranks run equal forward-pass chunks (FSDP safety)."""
        original_bsz = texts.size(0)
        target_bsz = self.eval_dataloader.batch_size
        if not self.model.training and original_bsz < target_bsz:
            pad = target_bsz - original_bsz
            texts = torch.cat([texts, torch.zeros((pad, texts.size(1)), dtype=texts.dtype, device=texts.device)], dim=0)
            labels = torch.cat([labels, torch.zeros((pad, self.num_labels), dtype=labels.dtype, device=labels.device)], dim=0)
            attention_masks = torch.cat([attention_masks, torch.zeros((pad, attention_masks.size(1)), dtype=attention_masks.dtype, device=attention_masks.device)], dim=0)
        return texts, labels, attention_masks, original_bsz, target_bsz

    @override
    def _forward_chunk(self, texts_chunk, labels_chunk, attention_masks_chunk, valid_in_chunk):
        logits = self.model(texts_chunk, attention_mask=attention_masks_chunk)
        loss = self.loss_fct(logits, labels_chunk)
        loss = loss / self.gradient_accumulation_steps

        preds = (logits.detach() >= 0.0).float()
        chunk_correct = (preds == labels_chunk).sum().item()
        chunk_total = labels_chunk.numel()
        chunk_tp = ((preds == 1.0) & (labels_chunk == 1.0)).sum().item()
        chunk_fp = ((preds == 1.0) & (labels_chunk == 0.0)).sum().item()
        chunk_fn = ((preds == 0.0) & (labels_chunk == 1.0)).sum().item()
        return loss, chunk_correct, chunk_total, chunk_tp, chunk_fp, chunk_fn

    @override
    def _log_extra_metrics(self, result_totals: dict):
        tp = result_totals["tp"]
        fp = result_totals["fp"]
        fn = result_totals["fn"]
        name_suffix = result_totals["name_suffix"]
        if tp + fp + fn > 0:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            micro_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            self.metric_logger.log(f"steps/eval/micro_f1{name_suffix}", self.step, micro_f1)
            if not isinstance(self.metric_logger, WandbLogger):
                self.metric_logger.log(f"tokens/eval/micro_f1{name_suffix}", self.processed_tokens, micro_f1)
