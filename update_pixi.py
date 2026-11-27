import sys
from pathlib import Path
from datetime import datetime

import hydra
from omegaconf import OmegaConf
from run_exp import ConnectWithPassphrase
from grid_generator.sbatch_builder import create_slurm_parameters
from resolver import pixi_lock_hash


def get_project_root() -> Path:
    # Get the project root directory (where pixi.toml is located).
    current = Path(__file__).resolve().parent

    # Look for pixi.toml in current and parent directories
    for _ in range(5):  # Limit search depth
        if (current / "pixi.toml").exists():
            return current
        current = current.parent

    raise FileNotFoundError("Could not find pixi.toml in parent directories")


def pack_remote_pixi(
    server: str,
    pack_dir: str,
    slurm_config: dict,
    dry_run: bool,
    force: bool,
    wait: bool,
):
    """Pack the local pixi.lock into pack_dir on the cluster, in a batch job (scripts/pixi_pack_env.sh)."""
    project_root = get_project_root()
    lock_hash = pixi_lock_hash()
    files = [
        project_root / "pixi.lock",
        project_root / "scripts" / "pixi_pack_env.sh",
        project_root / "scripts" / "pixi_unpack_env.sh",
    ]
    # More than the install job: the test unpack holds the extracted tar and the env, and imports torch.
    slurm_config.update(
        {"job-name": "pack_pixi", "cpus-per-task": 8, "mem": "32G", "time": "01:00:00"}
    )

    print(f"Cluster: {server}")
    print(f"pixi.lock hash: {lock_hash}")
    print(f"Pack: {pack_dir}/nano-env-{lock_hash}.tar")
    if dry_run:
        print(
            "\n[DRY RUN] Would copy these into <pack dir>/builds/<timestamp>_<hash>/:"
        )
        for f in files:
            print(f"  - {f}")
        mode = "--wait" if wait else "--parsable"
        print(f"and run scripts/pixi_pack_env.sh there with sbatch {mode} and:")
        for param in create_slurm_parameters(slurm_config):
            print(f"      {param}")
        return

    print(f"\nConnecting to {server}...")
    with ConnectWithPassphrase(host=server, inline_ssh_env=True) as connection:
        # The remote shell expands $USER; sftp paths below need it expanded.
        pack_dir = connection.run(f'echo "{pack_dir}"', hide=True).stdout.strip()
        tar = f"{pack_dir}/nano-env-{lock_hash}.tar"
        if not force and connection.run(f'test -f "{tar}"', hide=True, warn=True).ok:
            print(f"Already packed: {tar}\nPass +force=true to rebuild it.")
            if not wait:
                print(
                    "No pack job, so nothing to wait for: submit without a dependency."
                )
            return

        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        stage = f"{pack_dir}/builds/{timestamp}_{lock_hash}"
        connection.run(f'mkdir -p "{stage}"', hide=True)
        for f in files:
            connection.put(str(f), remote=f"{stage}/{f.name}")

        slurm_config["output"] = f"{stage}/slurm-%j.out"
        sbatch_flags = " ".join(
            p.replace("#SBATCH ", "") for p in create_slurm_parameters(slurm_config)
        )
        force_flag = " --force" if force else ""
        command = f"bash -l {stage}/pixi_pack_env.sh {stage} {pack_dir}{force_flag}"
        if not wait:
            # +wait=false: submit and print the job id, so jobs can wait for it with afterok
            result = connection.run(
                f'sbatch --parsable {sbatch_flags} --wrap="{command}"',
                pty=True,
                warn=True,
            )
            if not result.ok:
                print("\n✗ Submitting the pack job failed")
                sys.exit(1)
            # with a pty, sbatch's stderr notes land in stdout too; the id is the last line
            job_id = result.stdout.strip().splitlines()[-1].split(";")[0].strip()
            print(
                f"\nSubmitted pack job {job_id}, log in {stage}/; it has not run yet."
            )
            print(
                f"Jobs that need it: ++infrastructure.slurm.dependency=afterok:{job_id}"
            )
            return
        print(f"\nSubmitting the pack job (sbatch --wait), log in {stage}/")
        result = connection.run(
            f'sbatch --wait {sbatch_flags} --wrap="{command}"', pty=True, warn=True
        )
        log = connection.run(
            f"ls {stage}/slurm-*.out", hide=True, warn=True
        ).stdout.strip()
        connection.run(f"tail -n 25 {log}", warn=True)
        print(f"\nLog: {log}")
        if not result.ok:
            print("\n✗ Packing failed")
            sys.exit(1)
        print(f"\n✓ Packed {tar}")


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

    # Clusters with a pixi_pack_dir (lem) unpack a packed env per job instead of installing one.
    pack_dir = cfg.infrastructure.get("pixi_pack_dir", None)
    if pack_dir is not None:
        pack_remote_pixi(
            server,
            pack_dir,
            slurm_config,
            cfg.get("dry_run", False),
            cfg.get("force", False),
            cfg.get("wait", True),
        )
        return

    assert "export PIXI_HOME" in "\n".join(
        script_lines
    ), "PIXI_HOME must be set in the cluster script."

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
        connection.run(f"echo '{install_command}' > {command_file}")
        full_command = f"{srun_cmd} bash -le {command_file}"

        result = connection.run(full_command, pty=True)

        if result.ok:
            print("\n✓ Pixi environment updated successfully!")
        else:
            print("\n✗ Error running pixi install")
            sys.exit(1)


if __name__ == "__main__":
    update_remote_pixi()
