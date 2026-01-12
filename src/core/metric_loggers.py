import os
import statistics
import neptune
import wandb
import torch
from typing import Optional
from abc import ABC, abstractmethod
from abc import ABC, abstractmethod
import torch.distributed as dist
import logging
from src.definitions import MetricLoggerConfig

logger = logging.getLogger(__name__)


class MetricLogger(ABC):
    def __init__(self, config=None):
        self.heavy_metrics_calculation_interval = (
            1 if config is None else config.heavy_metrics_calculation_interval
        )
        self.accumulators = {}
        self.step = 0

    @property
    def _should_log_heavy_metrics(self) -> bool:
        return (
            self.step is not None
            and self.heavy_metrics_calculation_interval > 0
            and (self.step) % self.heavy_metrics_calculation_interval == 0
        )

    @abstractmethod
    def log(self, name, step, value):
        pass

    def accumulate_metrics(self, layer_name, calculate_fn, metrics, transform_fn=None):
        if self._should_log_heavy_metrics:
            if transform_fn is not None:
                metrics = transform_fn(**metrics)

            if layer_name not in self.accumulators:
                self.accumulators[layer_name] = MetricAccumulator(calculate_fn, metrics)
            else:
                self.accumulators[layer_name].append(metrics)

    def flush_accumulated_metrics(self, step):
        if self._should_log_heavy_metrics:
            for name, accumulator in self.accumulators.items():
                for metric, result in accumulator.calculate().items():
                    self.log(f"steps/{name}/{metric}", step, result)
                accumulator.reset()

    def set_step(self, step):
        self.step = step


class MetricAccumulator:
    def __init__(self, calculate_fn, metrics):
        self.acc_dict = {}
        self.calculate_fn = calculate_fn
        self.append(metrics)

    def append(self, metrics):
        for key, value in metrics.items():
            if key in self.acc_dict:
                self.acc_dict[key].append(value)
            else:
                self.acc_dict[key] = [value]

    def calculate(self):
        return self.calculate_fn(**self.acc_dict)

    def reset(self):
        for key in self.acc_dict:
            self.acc_dict[key] = []


class DummyLogger(MetricLogger):
    def log(self, _name, _step, _value):
        pass


class NeptuneLogger(MetricLogger):
    def __init__(self, run, rank, config=None):
        super().__init__(config)
        self.run = run
        self.rank = rank

    def log(self, name, step, value):
        if self.rank == 0:
            self.run[name].append(value=value, step=step)


class WandbLogger(MetricLogger):
    def __init__(self, run, should_log, config=None):
        super().__init__(config)
        self.run = run
        self.should_log = should_log

    def log(self, name, step, value):
        if self.should_log:
            self.run.log({name: value}, step=step)


class StdoutLogger(MetricLogger):
    def __init__(self, config=None):
        super().__init__(config)
        self.rank = os.environ.get("RANK", 0)
        logger.info("Logging to stdout.")

    def log(self, name, step, value):
        logger.info(f"[device:{self.rank}] on step:{step} -> {name}: {value}")


class RecorderLogger(MetricLogger):
    def __init__(self, config=None):
        super().__init__(config)
        self.data = {}

    def log(self, name, step, value):
        if name not in self.data:
            self.data[name] = []
        self.data[name].append((value, step))

    def clear(self):
        self.data = {}


def get_composition_file_path(hydra_config):
    """
    Get the path to the file passed to main
    """
    config_name = hydra_config.job.config_name
    config_path = [
        path["path"]
        for path in hydra_config.runtime.config_sources
        if path["schema"] == "file"
    ][0]

    return f"{config_path}/{config_name}.yaml"


def get_metric_logger(
    metric_logger_config: Optional[MetricLoggerConfig] = None,
    tracker_run_id: Optional[str] = None,
):
    _metric_logger = None
    if metric_logger_config.type == "neptune":
        neptune_run_id = (
            None if metric_logger_config.new_neptune_job else tracker_run_id
        )
        rank = int(os.environ["RANK"])

        if rank == 0:
            neptune_logger = neptune.init_run(
                project=metric_logger_config.project_name,
                name=metric_logger_config.name,
                tags=metric_logger_config.tags,
                with_id=neptune_run_id,
            )
            _metric_logger = NeptuneLogger(neptune_logger, rank, metric_logger_config)

            if int(os.environ["WORLD_SIZE"]) > 1:
                run_id_container = [None]
                if neptune_run_id is None:
                    neptune_run_id = neptune_logger["sys/id"].fetch()

                run_id_container[0] = neptune_run_id
                dist.broadcast_object_list(run_id_container, src=0)
        else:
            run_id_container = [neptune_run_id]
            dist.broadcast_object_list(run_id_container, src=0)
            neptune_run_id = run_id_container[0]

            neptune_logger = neptune.init_run(
                project=metric_logger_config.project_name,
                with_id=neptune_run_id,
                capture_hardware_metrics=False,
                name=metric_logger_config.name,
                tags=metric_logger_config.tags,
            )
            _metric_logger = NeptuneLogger(neptune_logger, rank, metric_logger_config)
    elif metric_logger_config.type == "wandb":
        if os.environ.get("WORLD_SIZE") != os.environ.get("LOCAL_WORLD_SIZE"):
            # TODO: Implement W&B multinode logging (https://docs.wandb.ai/models/track/log/distributed-training)
            raise NotImplementedError("W&B multinode logging is not implemented yet.")
        wandb_run_id = None if metric_logger_config.new_wandb_job else tracker_run_id
        rank = os.environ.get("RANK")
        if rank is not None:
            rank = int(rank)

        if rank == 0 or rank is None:
            wandb_logger = wandb.init(
                entity=metric_logger_config.wandb_entity,
                project=metric_logger_config.project_name,
                name=metric_logger_config.name,
                tags=metric_logger_config.tags,
                id=wandb_run_id,
                resume="allow",
            )
            _metric_logger = WandbLogger(
                run=wandb_logger, should_log=True, config=metric_logger_config
            )
        else:
            _metric_logger = WandbLogger(
                run=None, should_log=False, config=metric_logger_config
            )

    elif metric_logger_config.type == "stdout":
        _metric_logger = StdoutLogger(metric_logger_config)
    elif metric_logger_config.type == "record":
        _metric_logger = RecorderLogger(metric_logger_config)
    elif metric_logger_config.type == "dummy":
        _metric_logger = DummyLogger(metric_logger_config)
    elif metric_logger_config.type == None:
        raise RuntimeError("Metric logger is not initialized yet.")
    else:
        raise ValueError(f"Unknown logger type: {metric_logger_config.type}")
    return _metric_logger


class AveMetric:
    def __init__(self, average_tail_len, name):
        self.name = name
        self.tail_len = average_tail_len
        self.metric_stack = []

    def log(self, mlogger: MetricLogger, step, metric_val):
        do_log = False
        self.metric_stack.append(metric_val)
        while len(self.metric_stack) > self.tail_len:
            self.metric_stack.pop(0)
            do_log = True
        if do_log:
            mlogger.log(self.name, step, statistics.mean(self.metric_stack))


class AveDiffMetric(AveMetric):
    def __init__(self, average_tail_len, name, first_metric_val):
        super().__init__(average_tail_len, name)
        self.last_metric_val = first_metric_val

    def log(self, mlogger, step, metric_val):
        metric_val_diff = metric_val - self.last_metric_val
        self.last_metric_val = metric_val
        super().log(mlogger, step, metric_val_diff)
