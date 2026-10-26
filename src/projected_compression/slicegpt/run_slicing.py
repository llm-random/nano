"""Offline SliceGPT job (one GPU, no training), launched through main.py:

    init_model_opt_sched_fn:
      _target_: src.projected_compression.slicegpt.run_slicing.slice_llama_model
      _partial_: true

`cfg.model` is the FULL base model (built with sequence_length = calibration length). The job:
  1. loads base weights (nano .pt or Hugging Face),
  2. [hybrid] prunes dff exactly like HP,
  3. fuses norms, rotates and slices the residual width to `slicegpt.target_d` (FineWeb-Edu calibration),
  4. logs base / one-shot (0-token) loss, params, kept-variance diagnostics,
  5. saves the sliced state dict (+ JSON) loadable by `_model/llama_3_*_sliced` with checkpoint type `sliced_nano`.
"""

import json
import logging
import os
import pathlib
import time

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.core.llama import copy_llama_model_weights_from_HF
from src.core.metric_loggers import get_metric_logger
from src.projected_compression.slicegpt.fusion import finalize_norms, fuse_norms
from src.projected_compression.slicegpt.slice import prune_dff, rotate_and_slice

logger = logging.getLogger(__name__)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def sample_token_batches(
    dataset_cfg, sequence_length: int, seed: int, n_samples: int, batch_size: int
):
    """Draw `n_samples` packed sequences from the configured (FineWeb-Edu) dataset."""
    dataset = instantiate(dataset_cfg, sequence_length=sequence_length, seed=seed)
    samples = []
    for sample in dataset:
        samples.append(torch.tensor(sample, dtype=torch.long))
        if len(samples) == n_samples:
            break
    ids = torch.stack(samples)
    return list(torch.split(ids, batch_size))


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, batches, device) -> float:
    """Mean next-token cross-entropy, computed like the trainer (input = ids[:, :-1], target = ids[:, 1:])."""
    model.eval()
    total, count = 0.0, 0
    for ids in batches:
        ids = ids.to(device)
        logits = model(ids[:, :-1])
        loss = F.cross_entropy(
            logits.flatten(0, -2).float(), ids[:, 1:].reshape(-1), reduction="sum"
        )
        total += loss.item()
        count += ids[:, 1:].numel()
    return total / count


def load_base_weights(model, source_cfg):
    if source_cfg.type == "nano":
        state_dict = torch.load(source_cfg.path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    elif source_cfg.type == "huggingface":
        copy_llama_model_weights_from_HF(model, source_cfg.path)
    else:
        raise ValueError(f"Unknown slicegpt.source.type '{source_cfg.type}'")


def slice_llama_model(cfg, metric_logger=None):
    sg = cfg.slicegpt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(sg.seed)

    if metric_logger is None:
        metric_logger = get_metric_logger(
            metric_logger_config=cfg.infrastructure.metric_logger,
            tracker_run_id=None,
            full_config=cfg,
        )
    run = getattr(metric_logger, "run", None)

    def log(values: dict):
        logger.info(json.dumps(values))
        if run is not None:
            run.log(values)

    target_d, target_dff = int(sg.target_d), int(sg.target_dff)
    logger.info(
        f"SliceGPT: target_d={target_d}, target_dff={target_dff}, orientation={sg.orientation}"
    )

    # 1. base model
    t0 = time.time()
    model = instantiate(cfg.model, _convert_="all")
    load_base_weights(model, sg.source)
    model = model.to(device=device, dtype=torch.float32).eval()
    D = model.embedding.weight.shape[1]
    base_dff = model.encoder.blocks[0].ff_layer.layer.ff_pre_act.weight.shape[0]
    base_params = count_params(model)

    eval_batches = sample_token_batches(
        cfg.trainer.train_dataloader.dataset,
        sg.eval.sequence_length,
        sg.eval.seed,
        sg.eval.n_samples,
        sg.eval.batch_size,
    )
    results = {
        "base_params": base_params,
        "base_loss": evaluate_loss(model, eval_batches, device),
    }

    # optional reference: one-shot HARD PRUNING (the HP baseline's exact `prune`) on the same eval batches
    hp_ref = sg.get("hp_reference")
    if hp_ref is not None and hp_ref.get("enabled", False):
        import copy

        from src.projected_compression.pruning import prune

        # keep only one full-size model on the GPU at a time (8B fp32 = 32 GB): park the base model on CPU
        model = model.to("cpu")
        hp_model = copy.deepcopy(model).to(device)
        prune(
            hp_model,
            sg.dimensions_importances_path,
            int(hp_ref.target_dmodel),
            int(hp_ref.target_dff),
        )
        results["hp_oneshot_loss"] = evaluate_loss(hp_model, eval_batches, device)
        results["hp_params"] = count_params(hp_model)
        results["hp_dmodel"], results["hp_dff"] = int(hp_ref.target_dmodel), int(
            hp_ref.target_dff
        )
        del hp_model
        torch.cuda.empty_cache()
        model = model.to(device)
    log({f"slicegpt/{k}": v for k, v in results.items()})

    # 2. hybrid: prune dff exactly like HP
    if target_dff < base_dff:
        prune_dff(model, sg.dimensions_importances_path, target_dff)
        results["dff_pruned_loss"] = evaluate_loss(model, eval_batches, device)

    # 3. fuse norms (function-preserving unless center_embeddings) and slice
    fuse_norms(model, center_embeddings=sg.center_embeddings)
    if sg.eval.fused_model:
        results["fused_loss"] = evaluate_loss(model, eval_batches, device)

    calibration_batches = sample_token_batches(
        cfg.trainer.train_dataloader.dataset,
        sg.calibration.sequence_length,
        sg.calibration.seed,
        sg.calibration.n_samples,
        sg.calibration.batch_size,
    )
    diagnostics = rotate_and_slice(
        model,
        calibration_batches,
        d=target_d,
        orientation=sg.orientation,
        seed=sg.seed,
        device=device,
    )
    del calibration_batches
    finalize_norms(model)

    results.update(
        {
            "sliced_params": count_params(model),
            "sliced_loss": evaluate_loss(model, eval_batches, device),  # 0-token point
            "D": D,
            "d": target_d,
            "base_dff": base_dff,
            "dff": target_dff,
            "slicing_time_s": time.time() - t0,
        }
    )
    results["kept_param_fraction"] = results["sliced_params"] / base_params
    log({f"slicegpt/{k}": v for k, v in results.items()})
    for name, kept in diagnostics["kept_variance"]:
        logger.info(f"kept variance {name}: {kept:.4f}")
    if run is not None:
        run.log(
            {
                f"slicegpt/kept_variance/{name}": kept
                for name, kept in diagnostics["kept_variance"]
            }
        )

    # 4. save
    if sg.save_path:
        save_dir = pathlib.Path(sg.save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        model_file = save_dir / f"{sg.save_name}.pt"
        torch.save(
            {k: v.detach().cpu() for k, v in model.state_dict().items()}, model_file
        )
        meta = {
            **results,
            "orientation": sg.orientation,
            "center_embeddings": sg.center_embeddings,
            "calibration": OmegaConf.to_container(sg.calibration, resolve=True),
            "source": OmegaConf.to_container(sg.source, resolve=True),
            "kept_variance": diagnostics["kept_variance"],
        }
        model_file.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        logger.info(f"Saved sliced model to {model_file}")

    return None, None, None, None, metric_logger
