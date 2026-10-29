"""Tier 0 of the baseline audit: static checks that need no GPU and no data.

See documents/notes/main_baseline_audit.md. Each test here locks in a defect that was
found on main and fixed, so that branching from main cannot silently reintroduce it.

Everything in this file runs in seconds and is safe for CI.
"""

import importlib
import os
import re
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


# The entry points people actually invoke. src/ was covered above, but run_exp.py,
# run_eval.py and main.py are not under src/ and were import-checked by nothing -- a
# broken one would have been invisible here and shown up only when somebody tried to
# submit a job. Found when a test importing run_exp failed in CI for want of fabric,
# which no CI dependency list had ever needed.
ENTRY_POINTS = ("main", "run_exp", "run_eval", "evaluate_hf_models", "resolver")


@pytest.mark.parametrize("module", ENTRY_POINTS)
def test_entry_point_imports(module):
    """Importing an entry point must not raise; it is the cheapest check there is."""
    importlib.import_module(module)


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

    The special case is deliberate, not an oversight waiting to be tidied. Renaming the
    directory to _ff_layer would remove this line, but model configs reference the group
    by path, so the rename touches every referencing config and collides with any
    in-flight branch that touches one. One line and this comment is the cheaper trade.
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


def test_no_dataloader_outruns_the_models_sequence_length():
    """Both dataloaders must feed the length the model was built for.

    The model is built once, from common.sequence_length, and LlamaRoPE sizes its
    sin/cos buffers from it and only ever *slices* them (src/core/llama.py). Hand it a
    longer batch and the slice silently returns fewer rows than the input has, and the
    broadcast raises `The size of tensor a (2047) must match the size of tensor b
    (1024)`.

    The fineweb eval dataloader pinned sequence_length: 2048, to keep final_ce
    comparable between runs that train at different lengths. The intent was right and
    the mechanism impossible: a model built for 1024 cannot score 2048 tokens. Every
    fineweb run below 2048 would have hit this -- including this branch's own
    smoketest_fineweb_holdout_helios, which runs at 1024 with eval on at interval 4.

    Checked on composed configs rather than raw YAML, because a dataset config may
    legitimately pin a literal (local_dummy pins 16 on both dataloaders) as long as the
    configs that use it set common.sequence_length to match.
    """
    warnings.filterwarnings("ignore")
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    import resolver  # noqa: F401
    from grid_generator.generate_configs import create_grid_config

    offenders = []
    for path in _runnable_configs():
        directory, filename = os.path.split(path)
        try:
            with initialize_config_dir(
                version_base=None,
                config_dir=os.path.join(REPO_ROOT, directory),
            ):
                cfg = compose(config_name=filename[:-5])
            grid = create_grid_config(cfg)
            resolved = OmegaConf.to_container(
                OmegaConf.create(grid[0][0]), resolve=True, throw_on_missing=False
            )
        except Exception:  # noqa: BLE001
            continue  # composition failures are test_every_runnable_config's business

        trainer = resolved.get("trainer") or {}
        model_length = (resolved.get("common") or {}).get("sequence_length")
        if not isinstance(model_length, int):
            continue
        for which in ("train_dataloader", "eval_dataloader"):
            loader = trainer.get(which)
            if not isinstance(loader, dict):
                continue
            dataset = loader.get("dataset")
            if not isinstance(dataset, dict):
                continue
            length = dataset.get("sequence_length")
            if isinstance(length, int) and length > model_length:
                offenders.append(
                    f"{path}: {which} feeds {length} tokens to a model built for "
                    f"{model_length}"
                )
    assert not offenders, "dataloader outruns the model:\n  " + "\n  ".join(offenders)


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


# Cluster/dataset pairs whose eval path is the same directory as its train path AND
# which do not carve a disjoint holdout out of it. Shrink this list; never add to it.
# See F10 in the baseline audit notes.
KNOWN_EVAL_PATH_EQUALS_TRAIN = {
    ("local", "c4"),  # toy local dataset, one directory on purpose
}


def _holdout_pair(dataset_family):
    """The (train, eval) holdout settings a dataset config declares, if any."""
    config = os.path.join(REPO_ROOT, "configs", "_dataset", f"{dataset_family}.yaml")
    if not os.path.isfile(config):
        return (None, None)
    with open(config) as handle:
        doc = yaml.safe_load(handle) or {}
    loaders = doc.get("trainer") or {}
    out = []
    for key in ("train_dataloader", "eval_dataloader"):
        loader = loaders.get(key) or {}
        dataset = loader.get("dataset") or {}
        out.append(dataset.get("holdout"))
    return tuple(out)


def test_eval_data_is_not_the_training_data():
    """An eval path equal to the train path measures nothing, unless it is held out.

    The datasets are a deterministic function of their seed (see
    test_core_framework.py), and `_dataset/fineweb.yaml` gives its eval dataloader the
    same `split: train` and the same `seed: 123` as the train one. So where the paths
    also coincide and nothing separates the rows, the eval dataloader yields *the
    batches the model is training on*, in the same order, and the reported eval loss is
    a re-scoring of training data rather than a measure of generalisation -- wrong in
    the flattering direction.

    fineweb has only a `train` directory on every cluster: no validation split was ever
    downloaded, which is why eval was pointed at the training data. There is no second
    directory to point at, so the fix is `holdout`, which reserves the tail of the rows
    for eval and gives the rest to train. The two slices come from the same row count,
    so they cannot overlap, and the shared path is then no longer a defect.

    This test accepts a shared path only when the config declares that split. Anything
    else sharing a path, with no holdout and no allowlist entry, fails.
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
            if train_value != eval_value:
                continue
            if (cluster, family) in KNOWN_EVAL_PATH_EQUALS_TRAIN:
                continue
            train_holdout, eval_holdout = _holdout_pair(family)
            if train_holdout == "train" and eval_holdout == "eval":
                continue  # disjoint by construction
            offenders.append(
                f"{path}: {family} eval path == train path ({train_value!r}) "
                f"with holdout train={train_holdout!r} eval={eval_holdout!r}"
            )
    assert not offenders, (
        "eval data is the training data, so eval loss measures nothing:\n  "
        + "\n  ".join(offenders)
    )


def test_eval_dataloaders_are_world_size_independent():
    """Eval must not depend on the GPU count.

    AbstractDataset's own contract says world_size_independent "should be 'True' for
    tests and eval", but every shipped dataset config set it False on both dataloaders.
    With False the rows are split across ranks by split_dataset_by_node, so each rank
    evaluates on a different slice and the reported number changes with the number of
    GPUs -- two runs of the same config are not comparable unless they used the same
    world size.
    """
    offenders = []
    for path in _tracked("configs/_dataset/*.yaml"):
        with open(os.path.join(REPO_ROOT, path)) as handle:
            doc = yaml.safe_load(handle) or {}
        loader = (doc.get("trainer") or {}).get("eval_dataloader") or {}
        dataset = loader.get("dataset") or {}
        if "world_size_independent" not in dataset:
            continue
        if dataset["world_size_independent"] is not True:
            offenders.append(
                f"{path}: eval_dataloader world_size_independent="
                f"{dataset['world_size_independent']!r}, must be True"
            )
    assert not offenders, "eval depends on the GPU count:\n  " + "\n  ".join(offenders)


# Functions whose call sites are checked against their real signature. Keep this to
# widely-used factories: the point is a refactor that renames a parameter and misses a
# caller, which import-time checks cannot see because the bad call sits in a function
# body that nothing in the suite executes.
#
# get_metric_logger is here because exactly that happened. The neptune -> wandb
# transition (d25c326b) renamed neptune_run_id to tracker_run_id, and
# convert_memeff_to_hf.convert_pc_to_hf kept passing the old name. It is the entry point
# for both llama_{1B,8B}_memeff_to_hf_conversion configs, so both had been dead since
# March with a TypeError on the first line of real work. Found 2026-10-28 by running the
# conversion on lem (job 6183879), not by any test.
SIGNATURE_CHECKED_FUNCTIONS = {
    "get_metric_logger": "src.core.metric_loggers",
}


def test_call_sites_match_the_function_signature():
    import ast
    import importlib
    import inspect

    offenders = []
    for name, module_path in SIGNATURE_CHECKED_FUNCTIONS.items():
        module = importlib.import_module(module_path)
        allowed = set(inspect.signature(getattr(module, name)).parameters)
        for path in _tracked("src/**/*.py") + _tracked("*.py"):
            full = os.path.join(REPO_ROOT, path)
            try:
                tree = ast.parse(open(full).read(), filename=path)
            except SyntaxError:  # pragma: no cover
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                called = getattr(func, "id", None) or getattr(func, "attr", None)
                if called != name:
                    continue
                for keyword in node.keywords:
                    if keyword.arg is None:  # **kwargs, cannot check
                        continue
                    if keyword.arg not in allowed:
                        offenders.append(
                            f"{path}:{node.lineno}: {name}() got "
                            f"'{keyword.arg}', which is not in its signature "
                            f"({', '.join(sorted(allowed))})"
                        )
    assert not offenders, "call sites disagree with the function signature:\n  " + (
        "\n  ".join(offenders)
    )


def test_pixi_version_is_bounded_below_the_lock_bump():
    """A pixi new enough to rewrite the lock must be refused, not left to do it.

    Measured across four versions: 0.63.2 writes lock v6, 0.67.1 leaves it at v6, and
    0.77.0 and 0.81.0 both bump it to v7. helios runs 0.67.1 and cannot read v7, so one
    `pixi run` on a modern pixi rewrote the lock and broke the cluster. `requires-pixi`
    turns that into a refusal with a clear message.

    The upper bound has to stay at or below 0.77, the first version observed to bump the
    format, and the lower bound must admit lem's 0.63.2 and helios's 0.67.1.
    """
    import tomllib

    with open(os.path.join(REPO_ROOT, "pixi.toml"), "rb") as handle:
        manifest = tomllib.load(handle)
    requires = (manifest.get("workspace") or {}).get("requires-pixi")
    assert (
        requires
    ), "pixi.toml declares no requires-pixi, so a too-new pixi will rewrite the lock"

    upper = re.search(r"<\s*([0-9]+\.[0-9]+)", requires)
    assert upper, f"requires-pixi {requires!r} has no upper bound"
    major, minor = (int(part) for part in upper.group(1).split("."))
    assert (major, minor) <= (0, 77), (
        f"requires-pixi upper bound {upper.group(1)} admits pixi 0.77+, which bumps the "
        "lock to v7 and breaks helios"
    )

    lower = re.search(r">=\s*([0-9]+\.[0-9]+)", requires)
    assert lower, f"requires-pixi {requires!r} has no lower bound"
    lo_major, lo_minor = (int(part) for part in lower.group(1).split("."))
    assert (lo_major, lo_minor) <= (
        0,
        63,
    ), f"requires-pixi lower bound {lower.group(1)} excludes lem's pixi 0.63.2"


# Model configs referenced by nothing. Dead weight still has to be maintained -- the
# orphan below was edited under two earlier issues purely to keep it consistent with
# configs that are actually used. They are allowlisted rather than deleted: an orphan
# model config is plausibly staging for work that has not landed, and deleting someone
# else's work-in-progress is not an audit's call. The point is that the set must not
# grow.
KNOWN_ORPHAN_MODEL_CONFIGS = {
    # documents/issues.md recorded one orphan. Running the check found five: the entry
    # was written from a spot inspection rather than a search.
    "configs/_model/llama_3_8_pc.yaml",
    "configs/_model/llama_3_1_pc.yaml",
    "configs/_model/llama/projected_tiny.yaml",
    "configs/_model/llama/projected_mem_eff_tiny.yaml",
    "configs/_model/llama/small.yaml",
    "configs/_model/smollm/projected_tiny.yaml",
}


def test_no_new_orphan_model_configs():
    """A model config nothing references is maintenance cost with no consumer."""
    referenced = set()
    for path in _tracked("configs/**/*.yaml", "configs/*.yaml"):
        text = open(os.path.join(REPO_ROOT, path)).read()
        for match in re.finditer(r"_model[^:\n]*:\s*([A-Za-z0-9_/]+)", text):
            referenced.add(match.group(1).split("/")[-1])

    orphans = set()
    for path in _tracked("configs/_model/**/*.yaml", "configs/_model/*.yaml"):
        stem = os.path.basename(path)[:-5]
        if stem.startswith("base_") or stem == "default":
            continue  # base fragments are pulled in by defaults lists, not by name
        if stem not in referenced:
            orphans.add(path)

    new = orphans - KNOWN_ORPHAN_MODEL_CONFIGS
    assert not new, (
        "model configs referenced by nothing -- wire them up or drop them:\n  "
        + "\n  ".join(sorted(new))
    )


def test_the_logged_save_path_is_never_a_none_path():
    """With checkpointing off, the run record must not name a path built from `None`.

    _build_save_path feeds wandb's `full_save_checkpoints_path`. When
    checkpoint.save.path is null it used to interpolate the literal string, giving
    "None/<jobid>/<task>/step_N" -- nothing is written there, so it is cosmetic, but it
    is a path in the run record that does not exist, which is how F17 wasted time.
    """
    from omegaconf import OmegaConf

    import main

    cfg = OmegaConf.create(
        {
            "trainer": {
                "checkpoint": {"save": {"path": None, "type": "nano"}},
                "n_steps": 10,
            }
        }
    )
    assert "None/" not in main._build_save_path(cfg)

    cfg.trainer.checkpoint.save.path = "/tmp/ckpt"
    assert main._build_save_path(cfg).startswith("/tmp/ckpt")


def test_world_size_is_recorded_with_the_run():
    """A run is reproducible at a fixed (seed, world_size), so both belong in the record.

    world_size_independent: false shards the corpus per rank, so changing the GPU count
    changes which data a run trains on. The seed is in the config; without world_size
    beside it the run record does not say what data the run actually saw.
    """
    import inspect

    import main

    source = inspect.getsource(main.initialize_training_components)
    assert '"world_size"' in source, (
        "world_size is not logged with the run, so the record cannot tell you what data "
        "the run trained on"
    )


def test_no_config_writes_a_bare_none():
    """A bare `None` in YAML is the string "None", not a null.

    YAML spells the null `null` or `~`; `None` is just a three-letter scalar. Every one
    of the 37 configs that switched eval off wrote `eval_dataloader: None`, so the
    trainer received the string. A string is not None and does have __iter__, so it
    passed every "is there a dataloader" guard and only failed once something called
    .to() on one of its characters -- after training had finished, and before the final
    checkpoint was written.

    The trainer now normalises a str eval_dataloader to None, so this is belt and
    braces: it keeps the configs honest about what they mean, and catches the same
    mistake on a key that has no such normalisation.
    """
    offenders = []
    for path in _tracked("configs"):
        if not path.endswith(".yaml"):
            continue
        with open(os.path.join(REPO_ROOT, path)) as handle:
            for number, line in enumerate(handle, start=1):
                if re.match(r"^\s*[\w.]+:\s*None\s*(#.*)?$", line):
                    offenders.append(f"{path}:{number}: {line.strip()}")
    assert not offenders, (
        "YAML reads a bare `None` as the string 'None'; write `null`:\n  "
        + "\n  ".join(offenders)
    )
