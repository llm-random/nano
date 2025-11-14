import argparse
import os
import re
import sys
from pathlib import Path

from omegaconf import OmegaConf
from run_exp import ConnectWithPassphrase


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


def update_remote_pixi(cluster_config_path: str, dry_run: bool = False):
    """
    Update pixi environment on a remote cluster.

    Args:
        cluster_config_path: Path to the cluster yaml configuration file
        dry_run: If True, only print actions without executing them
    """
    # Load cluster configuration
    if not os.path.exists(cluster_config_path):
        raise FileNotFoundError(f"Cluster config not found: {cluster_config_path}")

    cfg = OmegaConf.load(cluster_config_path)

    # Extract server and PIXI_HOME
    server = cfg.infrastructure.server
    script_lines = cfg.infrastructure.script

    try:
        pixi_home = extract_pixi_home(script_lines)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

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

    if dry_run:
        print("\n[DRY RUN] Would perform the following actions:")
        print(f"  1. Connect to {server}")
        print(f"  2. Create directory {pixi_home} if it doesn't exist")
        print(f"  3. Copy pixi.toml to {pixi_home}/")
        if pixi_lock.exists():
            print(f"  4. Copy pixi.lock to {pixi_home}/")
        print(f"  5. Run 'pixi install' in {pixi_home}")
        return

    # Connect to remote server
    print(f"\nConnecting to {server}...")
    with ConnectWithPassphrase(host=server, inline_ssh_env=True) as connection:
        print("Connected successfully!")

        # Ensure PIXI_HOME directory exists
        print(f"\nEnsuring {pixi_home} exists...")
        connection.run(f"mkdir -p {pixi_home}", hide=True)

        # Copy pixi.toml
        print(f"Copying pixi.toml to {pixi_home}/...")
        connection.put(str(pixi_toml), remote=f"{pixi_home}/pixi.toml")

        # Copy pixi.lock if it exists
        if pixi_lock.exists():
            print(f"Copying pixi.lock to {pixi_home}/...")
            connection.put(str(pixi_lock), remote=f"{pixi_home}/pixi.lock")

        # Run pixi install
        print(f"\nRunning 'pixi install' in {pixi_home}...")

        # Set up PATH and other environment variables from the cluster config
        text = "\n".join(script_lines)
        env_setup = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("export") and "PIXI" in stripped:
                env_setup.append(stripped)

        env_commands = " && ".join(env_setup) if env_setup else ""

        if env_commands:
            install_command = f"{env_commands} && cd {pixi_home} && pixi install"
        else:
            install_command = f"cd {pixi_home} && pixi install"

        result = connection.run(install_command, pty=True)

        if result.ok:
            print("\n✓ Pixi environment updated successfully!")
        else:
            print("\n✗ Error running pixi install")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Update pixi environment on remote clusters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--cluster",
        type=str,
        required=True,
        help="Path to cluster configuration YAML file",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it",
    )

    args = parser.parse_args()

    try:
        update_remote_pixi(args.cluster, dry_run=args.dry_run)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
