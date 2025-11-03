from lm_eval import evaluator
from attr import define
from typing import Optional
import json

from src.core.metric_loggers import AveDiffMetric, AveMetric, MetricLogger


@define(slots=False)
class Evaluator:
    checkpoint_path: str
    tokenizer: str
    tasks: list[str]
    limit: Optional[int]
    device: str
    metric_logger: MetricLogger

    def __attrs_post_init__(self):
        self.tasks = list(
            self.tasks
        )  # it's omegaconf.listconfig.ListConfig, lm_eval doesn't accept this type

    def eval(self):
        eval_model_args = (
            f"pretrained={self.checkpoint_path}," f"tokenizer={self.tokenizer}"
        )

        results = evaluator.simple_evaluate(
            model="hf",
            model_args=eval_model_args,
            tasks=self.tasks,
            limit=self.limit,
            device=self.device,
        )

        # Save results to JSON with default=str to handle torch dtypes
        with open("eval_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # print(results)
        self.log_eval(results)

    def log_eval(self, eval_results: dict):
        """Log evaluation results to Neptune."""
        # Log individual task metrics
        for task_name, metrics in eval_results["results"].items():
            for metric_name, value in metrics.items():
                # Strip ,none suffix and log
                clean_metric_name = metric_name.replace(",none", "")
                self.metric_logger.run[f"eval/{task_name}/{clean_metric_name}"] = value

        self.metric_logger.run["eval/limit"] = eval_results["config"]["limit"]

        # Upload full results as artifact
        # disabled for now, it is a huge dict.
        # self.metric_logger.run["eval/full_results"].upload("eval_results.json")
