"""The lem packed-env pieces that can be checked off the cluster (documents/notes/lem_packed_pixi_env.md).

A job finds its env by the pixi.lock hash baked into exp.job by resolver.pixi_lock_hash, and
scripts/pixi_pack_env.sh names the tar with sha256sum; the two must agree, or every job dies.
"""

import os
import subprocess

import resolver

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNPACK = os.path.join(REPO_ROOT, "scripts", "pixi_unpack_env.sh")


def _source_unpack(env):
    """Source the unpack script as exp.job does; REACHED means it let the job carry on."""
    return subprocess.run(
        ["bash", "-c", f'source "{UNPACK}"; echo REACHED'],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={"PATH": os.environ["PATH"], **env},
    )


def test_resolver_hash_matches_the_pack_script():
    shell = subprocess.run(
        "sha256sum pixi.lock | cut -c1-16",
        shell=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    )
    assert resolver.pixi_lock_hash() == shell.stdout.strip()


def test_lem_preamble_bakes_in_the_hash():
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        config_dir=os.path.join(REPO_ROOT, "configs", "jm", "feature"),
        version_base=None,
    ):
        cfg = compose(config_name="h4_lem_packed_env")
    script = list(cfg.infrastructure.script)
    assert f"export PIXI_LOCK_HASH={resolver.pixi_lock_hash()}" in script
    assert script[-1] == "source scripts/pixi_unpack_env.sh"
    assert cfg.infrastructure.slurm.extra == "FORCE_RM_TMPDIR"


def test_unpack_fails_the_job_when_the_pack_is_missing(tmp_path):
    result = _source_unpack(
        {
            "PIXI_PACK_DIR": str(tmp_path),
            "PIXI_LOCK_HASH": "0123456789abcdef",
            "TMPDIR_LOCAL": str(tmp_path),
        }
    )
    assert result.returncode == 1
    assert "REACHED" not in result.stdout
    assert "no packed env for this pixi.lock (0123456789abcdef)" in result.stderr


def test_unpack_refuses_to_unpack_without_node_local_disk(tmp_path):
    (tmp_path / "nano-env-0123456789abcdef.tar").write_bytes(b"")
    unpacker = tmp_path / "nano-env-0123456789abcdef.pixi-unpack"
    unpacker.write_text("#!/bin/sh\nexit 0\n")
    unpacker.chmod(0o755)
    result = _source_unpack(
        {"PIXI_PACK_DIR": str(tmp_path), "PIXI_LOCK_HASH": "0123456789abcdef"}
    )
    assert result.returncode == 1
    assert "REACHED" not in result.stdout
    assert "TMPDIR_LOCAL is unset" in result.stderr
