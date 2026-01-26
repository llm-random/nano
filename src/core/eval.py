from lm_eval import evaluator
from attr import define
from typing import Optional
import json

from src.core.metric_loggers import MetricLogger


@define(slots=False)
class Evaluator:
    checkpoint_path: str
    tokenizer: str
    tasks: list[str]
    limit: Optional[int]
    device: str
    metric_logger: MetricLogger

    def eval(self):
        eval_model_args = (
            f"pretrained={self.checkpoint_path}," f"tokenizer={self.tokenizer}"
        )

        results_no_fewshot = evaluator.simple_evaluate(
            model="hf",
            model_args=eval_model_args,
            tasks=list(self.tasks),
            limit=self.limit,
            device=self.device,
            log_samples=False,
        )
        with open(f"{self.checkpoint_path}/eval_results_no_fewshot.json", "w") as f:
            json.dump(results_no_fewshot, f, indent=2, default=str)

        self.log_eval(results_no_fewshot, prefix="eval_no_fewshot")

        # results_with_fewshot_5 = evaluator.simple_evaluate(
        #     model="hf",
        #     model_args=eval_model_args,
        #     tasks=list(self.tasks),
        #     limit=self.limit,
        #     device=self.device,
        #     log_samples=False,
        #     num_fewshot=5,
        # )
        # with open(f"{self.checkpoint_path}/eval_results_with_fewshot5.json", "w") as f:
        #     json.dump(results_with_fewshot_5, f, indent=2, default=str)

        # self.log_eval(results_with_fewshot_5, prefix="eval_with_fewshot_5")

    def log_eval(self, eval_results: dict, prefix: str):
        """Log evaluation results to Neptune."""
        for task_name, metrics in eval_results["results"].items():
            for metric_name, value in metrics.items():
                clean_metric_name = metric_name.replace(",none", "")
                self.metric_logger.run[f"{prefix}/{task_name}/{clean_metric_name}"] = (
                    value
                )

        self.metric_logger.run[f"{prefix}/limit"] = eval_results["config"]["limit"]
