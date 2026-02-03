import random
from omegaconf import OmegaConf
import platform
import os
import re


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
    placeholders = []
    for var in ENV_VARS_TO_FORWARD:
        export_placeholder = f'export {var}="{env_var_name_to_placeholder(var)}"'
        if var not in os.environ:
            print(
                f"Warning: {var} not found in environment variables. This might lead to issues."
            )
            export_placeholder = f"# {export_placeholder}  # Warning: {var} not found in environment. This might lead to issues."
        placeholders.append(export_placeholder)
    return "\n".join(placeholders)


OmegaConf.register_new_resolver("__llmrandom_cluster_config", get_cluster_name)

OmegaConf.register_new_resolver("random_seed", lambda: random.randint(0, 100000))

OmegaConf.register_new_resolver("eval", eval)

OmegaConf.register_new_resolver(
    "export_env_variables_placeholders", get_env_vars_placeholders_export
)
