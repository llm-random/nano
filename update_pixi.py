import sys
from pathlib import Path
from datetime import datetime

import hydra
from omegaconf import OmegaConf
from run_exp import ConnectWithPassphrase
from grid_generator.sbatch_builder import create_slurm_parameters

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

    assert 'export PIXI_HOME' in "\n".join(script_lines), "PIXI_HOME must be set in the cluster script."

    # This job is just "pixi install" → no GPU needed.
    # Remove GPU-related keys inherited from the main cluster config.
    for key in ("gres", "cpus_per_gpu", "mem_per_gpu"):
        slurm_config.pop(key, None)

    if cfg.infrastructure.server == "lem":
        slurm_config["gres"] = "gpu:hopper:1"

    # Optionally set simple CPU-only params (tune if you like)
    slurm_config["job-name"] = "update_pixi"
    slurm_config["cpus-per-task"] = (
        4  # requires your create_slurm_parameters to map this
    )
    slurm_config["mem"] = "8G"  # normal mem, not mem-per-gpu

    print(f"Cluster: {server}")

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
        print(f"  2. Create temporary directory in $HOME for pixi config")
        print(f"  3. Copy pixi.toml (and pixi.lock if present) there")
        print(f"  4. Run 'pixi install' on compute node using srun with SLURM params:")
        slurm_params = create_slurm_parameters(slurm_config)
        for param in slurm_params:
            print(f"      {param}")
        print(
            "  5. Inside srun: mkdir -p PIXI_HOME, copy from temp dir to PIXI_HOME, run pixi install"
        )
        return

    # Connect to remote server
    print(f"\nConnecting to {server}...")
    with ConnectWithPassphrase(host=server, inline_ssh_env=True) as connection:
        print("Connected successfully!")

        # Figure out remote $HOME and temp dir for pixi config
        home_dir = connection.run("cd && pwd", hide=True).stdout.strip()
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        remote_tmp_dir = f"{home_dir}/update_pixi/{timestamp}"

        print(f"\nEnsuring temporary directory {remote_tmp_dir} exists...")
        connection.run(f"mkdir -p {remote_tmp_dir}", hide=True)

        # Now that we know the real remote path, set SLURM output there
        slurm_config["output"] = f"{remote_tmp_dir}/pixi_install_%j.out"

        # Copy pixi.toml / pixi.lock into $HOME temp dir
        print(f"Copying pixi.toml to {remote_tmp_dir}/...")
        connection.put(str(pixi_toml), remote=f"{remote_tmp_dir}/pixi.toml")

        if pixi_lock.exists():
            print(f"Copying pixi.lock to {remote_tmp_dir}/...")
            connection.put(str(pixi_lock), remote=f"{remote_tmp_dir}/pixi.lock")

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
            if line.startswith("export"):
                env_setup.append(line)
                continue

        env_commands = "\n".join(env_setup) if env_setup else ""

        # mkdir + copy happen on the compute node, via srun
        base_command = f"""\
ts=$(date +%Y_%m_%d_%H_%M_%S)
mkdir -p "$PIXI_HOME"

if [ -f "$PIXI_HOME/pixi.toml" ] || [ -f "$PIXI_HOME/pixi.lock" ]; then
  mkdir -p "$PIXI_HOME/old_pixi_files/obsolete_since_${{ts}}"
  [ -f "$PIXI_HOME/pixi.toml" ] && mv -f "$PIXI_HOME/pixi.toml" "$PIXI_HOME/old_pixi_files/obsolete_since_${{ts}}/"
  [ -f "$PIXI_HOME/pixi.lock" ] && mv -f "$PIXI_HOME/pixi.lock" "$PIXI_HOME/old_pixi_files/obsolete_since_${{ts}}/"
fi

mv -f "{remote_tmp_dir}/pixi.toml" "$PIXI_HOME/"
chmod 777 "$PIXI_HOME/pixi.toml"

if [ -f "{remote_tmp_dir}/pixi.lock" ]; then
  mv -f "{remote_tmp_dir}/pixi.lock" "$PIXI_HOME/"
  chmod 777 "$PIXI_HOME/pixi.lock"
fi

env

cd "$PIXI_HOME"
pixi install
"""


        if env_commands:
            install_command = f"{env_commands} && {base_command}"
        else:
            install_command = base_command

        command_file = f"{remote_tmp_dir}/run_pixi_install.sh"
        connection.run(f'echo \'{install_command}\' > {command_file}')
        full_command = f"{srun_cmd} bash -le {command_file}"

        result = connection.run(full_command, pty=True)

        if result.ok:
            print("\n✓ Pixi environment updated successfully!")
        else:
            print("\n✗ Error running pixi install")
            sys.exit(1)


if __name__ == "__main__":
    update_remote_pixi()
