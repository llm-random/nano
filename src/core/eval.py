from lm_eval import evaluator
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from attr import define
from typing import Optional
import json
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
        for request in requests:
            context, continuation = request.args

            context_enc = self.tok_encode(context)
            continuation_enc = self.tok_encode(continuation)

            full_enc = (context_enc + continuation_enc)[-self.max_length :]
            input_ids = torch.tensor([full_enc], device=self.device)

            logits = self._model_forward(input_ids)

            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]

            log_probs = F.log_softmax(shift_logits, dim=-1)

            cont_len = len(continuation_enc)
            cont_log_probs = log_probs[0, -cont_len:]
            cont_labels = shift_labels[0, -cont_len:]

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
        for request in requests:
            (string,) = request.args

            encoding = self.tok_encode(string)[-self.max_length :]
            input_ids = torch.tensor([encoding], device=self.device)

            logits = self._model_forward(input_ids)

            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]

            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(2)).squeeze(2)

            total_log_prob = token_log_probs.sum().item()
            results.append(total_log_prob)

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
    tokenizer: str
    tasks: list[str]
    limit: Optional[int]
    max_length: int
    max_gen_toks: int
    batch_size: int
    metric_logger: MetricLogger
    model: nn.Module

    def eval(self):
        self.model.eval()
        lm = NanoLM(
            model=self.model,
            tokenizer_name=self.tokenizer,
            max_length=self.max_length,
            max_gen_toks=self.max_gen_toks,
            batch_size=self.batch_size,
        )
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=list(self.tasks),
            limit=self.limit,
            log_samples=False,
            confirm_run_unsafe_code=True,
        )

        if lm.rank == 0:
            with open("eval_results.json", "w") as f:
                json.dump(results, f, indent=2, default=str)

            self.log_eval(results)

    def log_eval(self, eval_results: dict):
        """Log evaluation results."""
        for task_name, metrics in eval_results["results"].items():
            for metric_name, value in metrics.items():
                clean_metric_name = metric_name.replace(",none", "")
                self.metric_logger.log(f"eval/{task_name}/{clean_metric_name}", value)

        self.metric_logger.log("eval/limit", eval_results["config"]["limit"])
