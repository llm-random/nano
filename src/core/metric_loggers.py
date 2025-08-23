import os
import neptune
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
    neptune_run_id: Optional[str] = None,
):
    _metric_logger = None
    if metric_logger_config.type == "neptune":
        neptune_run_id = (
            None if metric_logger_config.new_neptune_job else neptune_run_id
        )
        rank = int(os.environ["RANK"])
        if int(os.environ["WORLD_SIZE"]) > 1:
            if rank == 0:
                neptune_logger = neptune.init_run(
                    project=metric_logger_config.project_name,
                    with_id=neptune_run_id,
                    name=metric_logger_config.name,
                    tags=metric_logger_config.tags,
                )
                if neptune_run_id is None:
                    neptune_run_id = neptune_logger["sys/id"].fetch()
                    broadcast_message(rank, neptune_run_id)
                _metric_logger = NeptuneLogger(
                    neptune_logger, rank, metric_logger_config
                )
            else:
                if neptune_run_id is None:
                    neptune_run_id = broadcast_message(rank)
                neptune_logger = neptune.init_run(
                    project=metric_logger_config.project_name,
                    with_id=neptune_run_id,
                    capture_hardware_metrics=False,
                    name=metric_logger_config.name,
                    tags=metric_logger_config.tags,
                )
                _metric_logger = NeptuneLogger(
                    neptune_logger, rank, metric_logger_config
                )

        else:
            neptune_logger = neptune.init_run(
                project=metric_logger_config.project_name,
                name=metric_logger_config.name,
                tags=metric_logger_config.tags,
                with_id=neptune_run_id,
            )
            _metric_logger = NeptuneLogger(neptune_logger, rank, metric_logger_config)

    elif metric_logger_config.type == "stdout":
        _metric_logger = StdoutLogger(metric_logger_config)
    elif metric_logger_config.type == "record":
        _metric_logger = RecorderLogger(metric_logger_config)
    elif metric_logger_config.type == "dummy":
        _metric_logger = DummyLogger(metric_logger_config)
    elif metric_logger_config.type == None:
        raise RuntimeError("Metric logger is not initialized yet.")
    else:
        raise ValueError(f"Unknown logger type: { metric_logger_config.type}")
    return _metric_logger


def broadcast_message(rank, message=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Broadcast the length of the message
    if rank == 0:
        message_length_tensor = torch.tensor(
            len(message), dtype=torch.int32, device=device
        )
    else:
        message_length_tensor = torch.empty(1, dtype=torch.int32, device=device)

    dist.broadcast(message_length_tensor, src=0)

    # Prepare the tensor to hold the message of the correct length
    if rank == 0:
        message_tensor = torch.tensor(
            list(message.encode("utf-8")), dtype=torch.uint8, device=device
        )
    else:
        message_tensor = torch.empty(
            message_length_tensor.item(), dtype=torch.uint8, device=device
        )

    # Broadcast the message string data
    dist.broadcast(message_tensor, src=0)

    return message_tensor.cpu().numpy().tobytes().decode("utf-8")

