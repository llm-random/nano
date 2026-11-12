from lm_eval import evaluator
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.evaluator_utils import get_task_list
from lm_eval.tasks import TaskManager, get_task_dict
from attr import define
from typing import Optional
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.core.metric_loggers import MetricLogger


class _SimpleAccelerator:
    """Minimal accelerator shim for lm_eval distributed sync."""

    def __init__(self, device: torch.device):
        self._device = device

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return torch.stack(gathered)

    def wait_for_everyone(self):
        dist.barrier()


class NanoLM(LM):
    """lm_eval wrapper for nano models, enabling evaluation without HF conversion."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer_name: str,
        max_length: int,
        max_gen_toks: int,
        batch_size: int,
    ):
        super().__init__()
        self._model = model
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self._device = next(model.parameters()).device
        self._max_length = max_length
        self._max_gen_toks = max_gen_toks
        self._batch_size = batch_size

        if dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()

        self.accelerator = _SimpleAccelerator(self._device)

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string, **kwargs):
        return self._tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens, **kwargs):
        return self._tokenizer.decode(tokens)

    def _model_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._model(input_ids)

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        results = []
        for i in range(0, len(requests), self._batch_size):
            batch = requests[i : i + self._batch_size]

            # Encode all requests in this batch
            encodings = []
            cont_lens = []
            for request in batch:
                context, continuation = request.args
                context_enc = self.tok_encode(context)
                continuation_enc = self.tok_encode(continuation)
                full_enc = (context_enc + continuation_enc)[-self.max_length :]
                encodings.append(full_enc)
                cont_lens.append(len(continuation_enc))

            # Right-pad to same length for batched forward pass.
            # With causal attention, real tokens never attend to right-side
            # pad tokens, so no attention mask is needed.
            max_len = max(len(e) for e in encodings)
            pad_id = (
                self._tokenizer.pad_token_id
                if self._tokenizer.pad_token_id is not None
                else 0
            )
            padded = [e + [pad_id] * (max_len - len(e)) for e in encodings]
            input_ids = torch.tensor(padded, device=self.device)

            logits = self._model_forward(input_ids)

            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            log_probs = F.log_softmax(shift_logits, dim=-1)

            for j, cont_len in enumerate(cont_lens):
                seq_len = len(encodings[j])
                cont_log_probs = log_probs[j, seq_len - cont_len - 1 : seq_len - 1]
                cont_labels = shift_labels[j, seq_len - cont_len - 1 : seq_len - 1]

                token_log_probs = cont_log_probs.gather(
                    1, cont_labels.unsqueeze(1)
                ).squeeze(1)
                total_log_prob = token_log_probs.sum().item()

                greedy_tokens = cont_log_probs.argmax(dim=-1)
                is_greedy = (greedy_tokens == cont_labels).all().item()

                results.append((total_log_prob, is_greedy))

        return results

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float]]:
        results = []
        for i in range(0, len(requests), self._batch_size):
            batch = requests[i : i + self._batch_size]

            encodings = []
            for request in batch:
                (string,) = request.args
                encodings.append(self.tok_encode(string)[-self.max_length :])

            max_len = max(len(e) for e in encodings)
            pad_id = (
                self._tokenizer.pad_token_id
                if self._tokenizer.pad_token_id is not None
                else 0
            )
            padded = [e + [pad_id] * (max_len - len(e)) for e in encodings]
            seq_lens = [len(e) for e in encodings]
            input_ids = torch.tensor(padded, device=self.device)

            logits = self._model_forward(input_ids)

            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            log_probs = F.log_softmax(shift_logits, dim=-1)

            for j, seq_len in enumerate(seq_lens):
                # Only score the real tokens (right-padding is ignored by causal attention)
                item_log_probs = log_probs[j, : seq_len - 1]
                item_labels = shift_labels[j, : seq_len - 1]
                token_log_probs = item_log_probs.gather(
                    1, item_labels.unsqueeze(1)
                ).squeeze(1)
                results.append(token_log_probs.sum().item())

        return results

    def generate_until(self, requests: list[Instance]) -> list[str]:
        results = []
        for request in requests:
            context, gen_kwargs = request.args
            max_gen = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            until = gen_kwargs.get("until", [self._tokenizer.eos_token])

            context_enc = self.tok_encode(context)
            generated = list(context_enc)

            for _ in range(max_gen):
                input_ids = torch.tensor(
                    [generated[-self.max_length :]], device=self.device
                )
                logits = self._model_forward(input_ids)
                next_token = logits[0, -1].argmax().item()
                generated.append(next_token)

                decoded = self.tok_decode(generated[len(context_enc) :])
                if any(s in decoded for s in until):
                    break

            results.append(self.tok_decode(generated[len(context_enc) :]))

        return results


@define(slots=False)
class Evaluator:
    """Scores `model` on lm-eval tasks through NanoLM and logs eval/<task>/<metric>.

    NanoLM above is ctx_scl's, unchanged, so the two branches' scores are comparable.
    """

    tokenizer: str
    tasks: list[str]
    limit: Optional[int]
    max_length: int
    max_gen_toks: int
    batch_size: int
    # Read by Trainer: score every eval_interval steps (0 = off) and/or once at the end.
    eval_interval: int
    final_eval: bool
    metric_logger: MetricLogger
    model: nn.Module

    def __attrs_post_init__(self):
        self._refuse_generative_tasks_on_multi_gpu()

    def _refuse_generative_tasks_on_multi_gpu(self):
        # Under FSDP2 every forward is a collective; generate_until runs a
        # different number of forwards per rank and would deadlock.
        if int(os.environ.get("WORLD_SIZE", 1)) == 1:
            return
        task_dict = get_task_dict(list(self.tasks), TaskManager())
        generative = sorted(
            {
                t.task_name
                for t in get_task_list(task_dict)
                if t.task.OUTPUT_TYPE == "generate_until"
            }
        )
        if generative:
            raise ValueError(
                f"Generative tasks {generative} can deadlock under FSDP2 with "
                "WORLD_SIZE > 1. Score them on one GPU, or drop them from "
                "downstream_evaluator.tasks."
            )

    def eval(self):
        self.model.eval()
        lm = NanoLM(
            model=self.model,
            tokenizer_name=self.tokenizer,
            max_length=self.max_length,
            max_gen_toks=self.max_gen_toks,
            batch_size=self.batch_size,
        )
        # lm-eval's bootstrap otherwise forks one worker per node CPU; serial gives the same stderr.
        os.environ["DISABLE_MULTIPROC"] = "1"
        step = self.metric_logger.step
        # Step None stops Residual accumulating heavy metrics on every eval forward.
        self.metric_logger.set_step(None)
        try:
            # Eager: eval batches vary in length, which would recompile compiled blocks.
            with torch.compiler.set_stance("force_eager"):
                results = evaluator.simple_evaluate(
                    model=lm,
                    tasks=list(self.tasks),
                    limit=self.limit,
                    log_samples=False,
                    # Only lambada's perplexity stderr bootstraps; 100k draws keep the
                    # other ranks waiting on rank 0. 0 would drop every stderr.
                    bootstrap_iters=1000,
                    confirm_run_unsafe_code=True,
                )
        finally:
            self.metric_logger.set_step(step)

        # lm-eval returns results on rank 0 only.
        if results is not None:
            self.log_eval(results)

    def log_eval(self, eval_results: dict):
        for task_name, metrics in eval_results["results"].items():
            for metric_name, value in metrics.items():
                # Skips the `alias` string and "N/A" stderrs.
                if not isinstance(value, (int, float)):
                    continue
                clean_metric_name = metric_name.replace(",none", "")
                self.metric_logger.log(f"eval/{task_name}/{clean_metric_name}", value)

        if self.limit is not None:
            self.metric_logger.log("eval/limit", self.limit)
