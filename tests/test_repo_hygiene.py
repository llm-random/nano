"""Tier 0 of the baseline audit: static checks that need no GPU and no data.

See documents/notes/main_baseline_audit.md. Each test here locks in a defect that was
found on main and fixed, so that branching from main cannot silently reintroduce it.

Everything in this file runs in seconds and is safe for CI.
"""

import importlib
import os
import subprocess
import warnings

import pytest
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tracked(*patterns):
    """Files git tracks, so generated and ignored trees are never scanned."""
    out = subprocess.run(
        ["git", "ls-files", *patterns],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [f for f in out if "product_keys" not in f]


# --------------------------------------------------------------------------------------
# imports


def test_every_src_module_imports():
    """A module that does not import is broken for everyone, whatever else passes."""
    failures = []
    for path in _tracked("src/*.py", "src/**/*.py"):
        module = path[:-3].replace("/", ".")
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 - we want to report, not raise
            failures.append(f"{module}: {type(exc).__name__}: {exc}")
    assert not failures, "modules that do not import:\n  " + "\n  ".join(failures)


# --------------------------------------------------------------------------------------
# layering


def test_core_does_not_depend_on_projected_compression():
    """src/core must stay usable by a project that has nothing to do with PC.

    src/core/trainer.py used to import finalize_projection_weights, advertise a
    "pc_finalize" save type and carry save_pc_finalized_checkpoint(). That moved to
    src.projected_compression.trainer.PCFinalizeTrainer. Core may be imported *by* PC,
    never the other way round.
    """
    offenders = []
    for path in _tracked("src/core/*.py", "src/core/**/*.py"):
        text = open(os.path.join(REPO_ROOT, path)).read()
        for needle in ("projected_compression", "slicegpt", "pc_finalize"):
            if needle in text:
                lines = [
                    f"{path}:{n}: {line.strip()}"
                    for n, line in enumerate(text.splitlines(), 1)
                    if needle in line
                ]
                offenders.extend(lines)
    assert (
        not offenders
    ), "src/core must not depend on projected-compression:\n  " + "\n  ".join(offenders)


# --------------------------------------------------------------------------------------
# config tree


def test_no_duplicate_yaml_keys():
    """A duplicate key makes Hydra reject the file outright.

    configs/pc_project/llmr_300.yaml carried `batch_size` twice and so could not be
    loaded at all. PyYAML's default loader silently keeps the last value, which is why
    this needs an explicit check rather than a plain yaml.safe_load.
    """

    class StrictLoader(yaml.SafeLoader):
        pass

    def no_duplicates(loader, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate key {key!r}", key_node.start_mark
                )
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    StrictLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, no_duplicates
    )

    failures = []
    for path in _tracked("*.yaml", "*.yml"):
        try:
            with open(os.path.join(REPO_ROOT, path)) as handle:
                yaml.load(handle, Loader=StrictLoader)
        except yaml.constructor.ConstructorError as exc:
            failures.append(f"{path}:{exc.problem_mark.line + 1}: {exc.problem}")
        except yaml.YAMLError as exc:
            failures.append(f"{path}: {type(exc).__name__}: {exc}")
    assert not failures, "duplicate or malformed YAML:\n  " + "\n  ".join(failures)


def _runnable_configs():
    """Top-level experiment configs: the ones a human passes to --config-name.

    Component groups live in underscore-prefixed directories, plus configs/ff_layer,
    which predates that convention.
    """
    return [
        path
        for path in _tracked("configs/*/*.yaml")
        if "/_" not in path and not path.startswith("configs/ff_layer/")
    ]


def test_every_runnable_config_composes_and_resolves():
    """Compose each config the way run_exp.py does, then resolve every interpolation.

    Invocation matters: the repo uses `--config-path configs/<dir>` with a bare config
    name. Composing as `--config-name <dir>/<name>` instead nests the whole tree under
    the directory name and breaks absolute interpolations, so this must mirror the real
    call (documents/notes/merge_to_main.md).

    `???` is left alone -- a mandatory value supplied at submit time is not a defect --
    but it must not hide a broken interpolation behind it, which is why this resolves
    with throw_on_missing=False rather than stopping at the first missing value. That
    distinction is what exposed llama_1B_pretrain's missing common.batch_size.
    """
    warnings.filterwarnings("ignore")
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    import resolver  # noqa: F401  (registers the repo's custom OmegaConf resolvers)
    from grid_generator.generate_configs import create_grid_config

    failures = []
    for path in _runnable_configs():
        directory, filename = os.path.split(path)
        try:
            with initialize_config_dir(
                version_base=None,
                config_dir=os.path.join(REPO_ROOT, directory),
            ):
                cfg = compose(config_name=filename[:-5])
            grid = create_grid_config(cfg)
            OmegaConf.to_container(
                OmegaConf.create(grid[0][0]), resolve=True, throw_on_missing=False
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{path}: {type(exc).__name__}: {str(exc).splitlines()[0]}")
    assert not failures, "configs that do not compose or resolve:\n  " + "\n  ".join(
        failures
    )


# --------------------------------------------------------------------------------------
# environment


def test_pixi_lock_stays_at_version_6():
    """helios runs pixi 0.67.1, which cannot read a v7 lock.

    Any local `pixi run` rewrites the lock to v7 without asking, and the diff is large
    enough to wave through in review. Committing that breaks submission on helios.
    """
    with open(os.path.join(REPO_ROOT, "pixi.lock")) as handle:
        first = handle.readline().strip()
    assert first == "version: 6", (
        f"pixi.lock is '{first}', expected 'version: 6'. A local `pixi run` rewrites it "
        "to v7, which helios's pixi 0.67.1 cannot read. Run `git checkout -- pixi.lock`, "
        "and use `pixi run --frozen` to avoid it."
    )


# Known account names in shared config, with nothing yet verified to replace them.
# Shrink this list; never add to it. See F9/F10 in the baseline audit notes.
KNOWN_USER_SPECIFIC = {
    "configs/_cluster/lem.yaml",  # fineweb paths under another account's directory
}

USER_SPECIFIC_PATTERNS = ("j321m", "crewtool", "m1kush", "ggwozdz", "mstefaniak")

# Framework component groups: inherited by every experiment, so they must stay neutral.
# configs/_mas is a personal overlay and is deliberately not one of these.
FRAMEWORK_CONFIG_GROUPS = (
    "_cluster",
    "_model",
    "_trainer",
    "_dataset",
    "_checkpoints",
    "_misc",
    "_eval",
)


def test_no_new_account_names_in_shared_config():
    """Shared config keys per-user paths on $USER, never on an account name.

    Scoped to the framework component groups -- the config every project inherits.
    configs/_mas is deliberately excluded: it is a personal overlay directory whose
    whole purpose is per-account paths. Experiment configs naming one researcher's
    scratch path are a different matter and are not covered here either.
    """
    offenders = []
    for path in _tracked(
        *(f"configs/{g}/**/*.yaml" for g in FRAMEWORK_CONFIG_GROUPS),
        *(f"configs/{g}/*.yaml" for g in FRAMEWORK_CONFIG_GROUPS),
    ):
        if path in KNOWN_USER_SPECIFIC:
            continue
        text = open(os.path.join(REPO_ROOT, path)).read()
        for lineno, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue  # a commented-out path is documentation, not configuration
            for pattern in USER_SPECIFIC_PATTERNS:
                if pattern in line:
                    offenders.append(f"{path}:{lineno}: {line.strip()}")
    assert (
        not offenders
    ), "account name in shared config -- key it on $USER instead:\n  " + "\n  ".join(
        offenders
    )


# Cluster/dataset pairs whose eval path is knowingly the same as its train path.
# Shrink this list; never add to it. See F10 in the baseline audit notes.
KNOWN_EVAL_PATH_EQUALS_TRAIN = {
    ("local", "fineweb"),
    ("local", "c4"),  # toy local dataset, one directory on purpose
    ("entropy", "fineweb"),
    ("helios", "fineweb"),
    ("lem", "fineweb"),
}


def test_eval_data_is_not_the_training_data():
    """An eval path equal to the train path measures nothing.

    The datasets are a deterministic function of their seed (see
    test_core_framework.py), and `_dataset/fineweb.yaml` gives its eval dataloader the
    same `split: train` and the same `seed: 123` as the train one. So where the paths
    also coincide, the eval dataloader yields *the batches the model is training on*, in
    the same order, and the reported eval loss is a re-scoring of training data rather
    than a measure of generalisation -- wrong in the flattering direction.

    c4 gets this right: its eval points at a `validation` directory with `split:
    validation`. fineweb does not, on any cluster, because no validation split was ever
    downloaded -- `datasets/fineweb/` contains only `train` while `datasets/c4/` has
    both. 29 of the 64 configs using fineweb have eval enabled.

    Those pairs are allowlisted rather than silently changed: the fix is to build a
    held-out split or to stop reporting fineweb eval, and both are decisions about
    experiments, not cleanups. The point of this test is that no *other* dataset
    acquires the same shape.
    """
    offenders = []
    for path in _tracked("configs/_cluster/*.yaml"):
        cluster = os.path.basename(path)[:-5]
        with open(os.path.join(REPO_ROOT, path)) as handle:
            doc = yaml.safe_load(handle) or {}
        switch = doc.get("cluster_switch") or {}
        for key, train_value in switch.items():
            if not key.startswith("train_path_"):
                continue
            family = key[len("train_path_") :]
            eval_value = switch.get(f"eval_path_{family}")
            if train_value is None or eval_value is None:
                continue  # not configured on this cluster
            if (
                train_value == eval_value
                and (cluster, family) not in KNOWN_EVAL_PATH_EQUALS_TRAIN
            ):
                offenders.append(
                    f"{path}: {family} eval path == train path ({train_value!r})"
                )
    assert not offenders, (
        "eval data is the training data, so eval loss measures nothing:\n  "
        + "\n  ".join(offenders)
    )
