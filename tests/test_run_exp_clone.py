"""run_exp.clone_command: the clone on the cluster leaves out infrastructure.git.sparse_exclude.

The cemetery branch keeps the whole repo; the experiment's clone on the cluster leaves out
the folders a job never reads (tests/, scripts/, documents/ by default), because every
clone counts against the cluster's file quota. The integration test runs the exact command
run_exp sends to the cluster against a throwaway local repository, offline.
"""

import shutil
import subprocess

import pytest

from run_exp import clone_command, sparse_exclude_names, tmux_session_name


def test_no_exclusion_is_the_plain_full_clone():
    expected = "git clone --depth 1 -b exp git@host:cemetery.git /c/$USER/exp"
    assert clone_command("git@host:cemetery.git", "exp", "/c/$USER/exp", []) == expected
    assert (
        clone_command("git@host:cemetery.git", "exp", "/c/$USER/exp", None) == expected
    )


def test_names_are_stripped_of_slashes():
    assert sparse_exclude_names(["/tests/", "scripts", ".github"]) == [
        "tests",
        "scripts",
        ".github",
    ]


@pytest.mark.parametrize(
    "bad",
    ["", "a b", "x;rm -rf ~", "../up", "src/core", "$HOME", ".", "..", ".git", "*"],
)
def test_rejects_anything_but_a_plain_top_level_folder_name(bad):
    with pytest.raises(ValueError, match="sparse_exclude"):
        sparse_exclude_names([bad])


@pytest.mark.parametrize(
    "branch, session",
    [
        # A float learning rate in the run name: what /base_schema configs produce.
        (
            "s2c_ps_h20_lr9.0_2026-11-08_10-00-00",
            "s2c_ps_h20_lr9_0_2026-11-08_10-00-00",
        ),
        ("a:b.c", "a_b_c"),
        # A name tmux can already address is left alone.
        ("std_ame_h12_lr8_2026-11-21_16-47-08", "std_ame_h12_lr8_2026-11-21_16-47-08"),
    ],
)
def test_tmux_session_name_has_no_dot_or_colon(branch, session):
    assert tmux_session_name(branch) == session


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_sparse_clone_leaves_out_only_the_named_top_level_folders(tmp_path):
    source = tmp_path / "cemetery_repo"
    files = {
        "main.py": "print('train')\n",
        "exp.job": "#!/bin/bash -l\n",
        "src/core/model.py": "x = 1\n",
        "src/tests/helper.py": "y = 2\n",  # nested: only top-level folders are left out
        "generated_configs/config_0.yaml": "a: 1\n",
        "configs/base.yaml": "b: 2\n",
        "tests/test_x.py": "def test():\n    pass\n",
        "scripts/run.sh": "echo run\n",
        "documents/notes.md": "# notes\n",
    }
    for relative, text in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def git(*args, cwd):
        return subprocess.run(
            [
                "git",
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                "-c",
                "commit.gpgsign=false",
                *args,
            ],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init", "-q", cwd=source)
    git("checkout", "-q", "-b", "exp", cwd=source)
    git("add", "-A", cwd=source)
    git("commit", "-q", "-m", "experiment snapshot", cwd=source)

    clone = tmp_path / "cemetery" / "exp_2026-11-20_12-00-00"
    clone.parent.mkdir()
    command = clone_command(
        f"file://{source}", "exp", str(clone), ["tests", "scripts", "documents"]
    )
    subprocess.run(["bash", "-c", command], check=True, capture_output=True, text=True)

    checked_out = {
        str(path.relative_to(clone))
        for path in clone.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(clone).parts
    }
    assert checked_out == {
        "main.py",
        "exp.job",
        "src/core/model.py",
        "src/tests/helper.py",
        "generated_configs/config_0.yaml",
        "configs/base.yaml",
    }
    # The left-out files are marked skip-worktree, not deleted: the clone is clean.
    assert git("status", "--porcelain", cwd=clone).stdout == ""
    # And they come back from the clone's own .git, with no network.
    git("sparse-checkout", "disable", cwd=clone)
    assert (clone / "tests" / "test_x.py").read_text() == "def test():\n    pass\n"
    assert (clone / "documents" / "notes.md").exists()
