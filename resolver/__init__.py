import hashlib
import random
from pathlib import Path
from omegaconf import OmegaConf
import platform
import os
import re
import logging

logger = logging.getLogger(__name__)


def get_cluster_name(hostname=None, username=None) -> str:
    if "LLMRANDOM_CLUSTER" in os.environ:
        return os.environ.get("LLMRANDOM_CLUSTER")

    if hostname is None:
        hostname = platform.uname().node

    conf = OmegaConf.load("configs/clusters.yaml")

    for cluster in conf:
        if "hosts" in cluster:
            for host_pattern in cluster.hosts:
                if re.match(host_pattern, hostname):
                    return cluster.name

    return "default"


ENV_VARS_TO_FORWARD = [
    "WANDB_API_KEY",
    "HF_TOKEN",
]


def env_var_name_to_placeholder(var_name: str) -> str:
    return f"__{var_name}_PLACEHOLDER__"


def get_env_vars_placeholders_export():
    # We are using placeholders to avoid exposing secrets in the sbatch script pushed to git
    placeholders = []
    for var in ENV_VARS_TO_FORWARD:
        export_placeholder = f'export {var}="{env_var_name_to_placeholder(var)}"'
        if var not in os.environ:
            logger.warning(
                "%s not found in environment variables. This might lead to issues.",
                var,
            )

            export_placeholder = f"# {export_placeholder}  # Warning: {var} not found in environment. This might lead to issues."
        placeholders.append(export_placeholder)
    return "\n".join(placeholders)


def pixi_lock_hash() -> str:
    """Names the packed env a lem job unpacks: the first 16 hex of sha256(pixi.lock), as scripts/pixi_pack_env.sh."""
    lock = Path(__file__).resolve().parent.parent / "pixi.lock"
    if lock.exists():
        return hashlib.sha256(lock.read_bytes()).hexdigest()[:16]
    # Cemetery clones have no pixi.lock (run_exp.version_code); there exp.job has exported the hash.
    return os.environ.get("PIXI_LOCK_HASH", "unknown")


OmegaConf.register_new_resolver("__llmrandom_cluster_config", get_cluster_name)

OmegaConf.register_new_resolver("random_seed", lambda: random.randint(0, 100000))

OmegaConf.register_new_resolver("eval", eval)

OmegaConf.register_new_resolver(
    "export_env_variables_placeholders", get_env_vars_placeholders_export
)

OmegaConf.register_new_resolver("pixi_lock_hash", pixi_lock_hash)
