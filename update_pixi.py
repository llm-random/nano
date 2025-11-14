import os
import re
import sys
import shlex
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from run_exp import ConnectWithPassphrase
from grid_generator.sbatch_builder import create_slurm_parameters


def extract_pixi_home(script_lines) -> str:
    text = "\n".join(script_lines)
    match = re.search(r"^export\s+PIXI_HOME=([^\s#]+)", text, re.MULTILINE)

    if not match:
        raise ValueError("PIXI_HOME not found in cluster configuration script")

    return match.group(1).strip().replace('"', "")


def get_project_root() -> Path:
    # Get the project root directory (where pixi.toml is located).
    current = Path(__file__).resolve().parent

    # Look for pixi.toml in current and parent directories
    for _ in range(5):  # Limit search depth
        if (current / "pixi.toml").exists():
            return current
        current = current.parent

    raise FileNotFoundError("Could not find pixi.toml in parent directories")


@hydra.main(version_base=None)
def update_remote_pixi(cfg: OmegaConf):
    """
    Update pixi environment on a remote cluster.

    Args:
        cfg: Hydra configuration object
    """
    # Extract server and PIXI_HOME
    server = cfg.infrastructure.server
    script_lines = cfg.infrastructure.script

    # Convert to plain dict to avoid struct mode restrictions
    slurm_config = OmegaConf.to_container(cfg.infrastructure.slurm, resolve=True)

    try:
        pixi_home = extract_pixi_home(script_lines)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Override SLURM config for pixi install (doesn't need GPUs, shorter time)
    slurm_config["time"] = "00:15:00"
    slurm_config["gres"] = "gpu:1"
    slurm_config["job-name"] = "update_pixi"
    slurm_config["output"] = f"{pixi_home}/pixi_install_%j.out"

    print(f"Cluster: {server}")
    print(f"PIXI_HOME: {pixi_home}")

    # Get local pixi files
    project_root = get_project_root()
    pixi_toml = project_root / "pixi.toml"
    pixi_lock = project_root / "pixi.lock"

    if not pixi_toml.exists():
        print(f"Error: pixi.toml not found at {pixi_toml}")
        sys.exit(1)

    print(f"\nLocal pixi files:")
    print(f"  - {pixi_toml}")
    if pixi_lock.exists():
        print(f"  - {pixi_lock}")
    else:
        print(f"  - pixi.lock not found (will only copy pixi.toml)")

    dry_run = cfg.get("dry_run", False)

    if dry_run:
        print("\n[DRY RUN] Would perform the following actions:")
        print(f"  1. Connect to {server}")
        print(f"  2. Create directory {pixi_home} if it doesn't exist")
        print(f"  3. Copy pixi.toml to {pixi_home}/")
        if pixi_lock.exists():
            print(f"  4. Copy pixi.lock to {pixi_home}/")
        print(f"  5. Run 'pixi install' on compute node using srun with SLURM params:")
        slurm_params = create_slurm_parameters(slurm_config)
        for param in slurm_params:
            print(f"      {param}")
        return

    # Connect to remote server
    print(f"\nConnecting to {server}...")
    with ConnectWithPassphrase(host=server, inline_ssh_env=True) as connection:
        print("Connected successfully!")

        # Copy pixi.toml
        print(f"Copying pixi.toml to {pixi_home}/...")
        connection.put(str(pixi_toml), remote=f"{pixi_home}/pixi.toml")

        # Copy pixi.lock if it exists
        if pixi_lock.exists():
            print(f"Copying pixi.lock to {pixi_home}/...")
            connection.put(str(pixi_lock), remote=f"{pixi_home}/pixi.lock")

        # Run pixi install on compute node using srun
        print(f"\nRunning 'pixi install' on compute node...")

        # Build srun command with SLURM parameters
        slurm_params = create_slurm_parameters(slurm_config)
        # Convert #SBATCH flags to srun flags (remove #SBATCH prefix)
        srun_flags = [param.replace("#SBATCH ", "") for param in slurm_params]
        srun_cmd = "srun " + " ".join(srun_flags)

        # Set up PATH and other environment variables from the cluster config
        text = "\n".join(script_lines)
        env_setup = []
        for raw in text.splitlines():
            line = raw.strip()

            # skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # include module load
            if line.startswith("module load"):
                env_setup.append(line)
                continue

            # include all pixi-related exports
            if line.startswith("export") and (
                "PIXI" in line
                or "XDG_" in line
                or 'PATH="$PIXI_HOME' in line
            ):
                env_setup.append(line)
                continue

        env_commands = " && ".join(env_setup) if env_setup else ""

        # mkdir happens on the compute node, via srun
        base_command = f"mkdir -p {pixi_home} && cd {pixi_home} && pixi install"

        if env_commands:
            install_command = f"{env_commands} && {base_command}"
        else:
            install_command = base_command

        cmd_quoted = shlex.quote(install_command)
        full_command = f"{srun_cmd} bash -lc {cmd_quoted}"

        result = connection.run(full_command, pty=True)

        if result.ok:
            print("\n✓ Pixi environment updated successfully!")
        else:
            print("\n✗ Error running pixi install")
            sys.exit(1)


if __name__ == "__main__":
    update_remote_pixi()
