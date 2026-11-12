"""Cluster job: load a model, run a channel selector on it, save the ChannelSelection.

Used as `init_model_opt_sched_fn` (like slicegpt/run_slicing.py), so it goes through
run_exp.py / main.py and runs on a compute node, not the login node. It trains nothing:
it returns no model, so main.run() skips the trainer.

Config::

    init_model_opt_sched_fn:
      _target_: src.projected_compression.channel_selection_job.run_channel_selection
      _partial_: true

    channel_selection:
      source_model: meta-llama/Llama-3.2-1B   # HF id or local HF directory
      selector:                               # a channel_selectors function, partial
        _target_: src.projected_compression.channel_selectors.select_closest_to_init
        _partial_: true
        keep_fraction: 0.7
      notes: ""
      save_dir: /path/to/dir
      save_name: keep70_huggingface           # written as <save_dir>/<save_name>.json

See configs/bdd/llama_1B_select_closest_to_init.yaml.
"""

import json
import logging
import os
import time

import torch
from hydra.utils import instantiate

from src.core.llama import copy_llama_model_weights_from_HF
from src.core.metric_loggers import get_metric_logger
from src.projected_compression.channel_selection import describe, save_selection

logger = logging.getLogger(__name__)


def run_channel_selection(cfg, metric_logger=None):
    job = cfg.channel_selection
    if metric_logger is None:
        metric_logger = get_metric_logger(
            metric_logger_config=cfg.infrastructure.metric_logger,
            tracker_run_id=None,
            full_config=cfg,
        )

    path = os.path.join(job.save_dir, f"{job.save_name}.json")
    logger.info(f"Channel selection on {job.source_model} -> {path}")

    t0 = time.time()
    model = instantiate(cfg.model, _convert_="all")
    copy_llama_model_weights_from_HF(model, job.source_model)
    model = model.to(torch.float32).eval()
    logger.info(f"Loaded {job.source_model} in {time.time() - t0:.0f}s")

    t0 = time.time()
    selector = instantiate(job.selector, _convert_="all")
    selection = selector(
        model, source_model=job.source_model, notes=job.get("notes", "")
    )
    logger.info(f"Selection took {time.time() - t0:.0f}s")

    os.makedirs(job.save_dir, exist_ok=True)
    save_selection(selection, path)
    logger.info(f"Saved {path}\n{describe(selection)}")
    for i, stats in enumerate(selection.method_params.get("per_block", [])):
        logger.info(f"block {i}: {json.dumps(stats)}")

    return None, None, None, None, metric_logger
