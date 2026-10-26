import os
import json
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from lm_eval import evaluator as lm_evaluator

logger = logging.getLogger(__name__)


def log_results_to_wandb(run, results: dict, model_path: str):
    metrics = {}
    for task_name, task_metrics in results["results"].items():
        for metric_name, value in task_metrics.items():
            if isinstance(value, (int, float)):
                clean = metric_name.replace(",none", "")
                metrics[f"{task_name}/{clean}"] = value
    metrics["model_path"] = model_path
    run.log(metrics)


@hydra.main(
    config_path="configs/pc_project", config_name="eval_hf_models", version_base=None
)
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))

    for model_cfg in cfg.models:
        model_path = model_cfg.path
        model_name = model_cfg.get("name", os.path.basename(model_path.rstrip("/")))
        tokenizer = model_cfg.get("tokenizer", cfg.default_tokenizer)

        logger.info(f"Evaluating {model_name} from {model_path}")

        cfg_to_log = {
            k: v
            for k, v in OmegaConf.to_container(cfg, resolve=False).items()
            if k != "infrastructure"
        }
        cfg_to_log["experiment_dir"] = os.getcwd()
        run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=model_name,
            tags=list(cfg.wandb.get("tags", [])),
            config=cfg_to_log,
            reinit=True,
        )

        results = lm_evaluator.simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_path},tokenizer={tokenizer},batch_size={cfg.batch_size},trust_remote_code=True",
            tasks=list(cfg.tasks),
            limit=cfg.get("limit", None),
            num_fewshot=cfg.get("num_fewshot", None),
            device=cfg.get("device", "cuda"),
            log_samples=False,
        )

        output_dir = cfg.get("output_dir", None) or model_path
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{model_name}_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")

        log_results_to_wandb(run, results, model_path)
        logger.info(f"Results logged for {model_name}")
        run.finish()


if __name__ == "__main__":
    main()
