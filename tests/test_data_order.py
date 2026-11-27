"""What a training step trains on is a function of the seed and the step.

Not of the GPU count, and not of gradient_accumulation_steps (Maciej, 2026-11-26;
documents/notes/archive/review_issues.md, A2). fineweb's train loader is
world_size_independent: every GPU reads the whole stream and keeps every
world_size-th sequence, so the GPUs of step k together hold exactly the sequences
a 1-GPU run trains on at step k. The trainer then cuts each GPU's share into
gradient_accumulation_steps equal micro-batches, which changes how the step is
computed but not what it trains on (tests/test_training_smoke.py checks that
end to end); the checks below refuse any batch that would not split evenly.

The loader worker count is still part of the order (review_issues.md, A1) and
stays at 2.

These build a small corpus in tmp_path and tokenize it with a stub, so they run
anywhere: no local data, no tokenizer download.
"""

import os
import random
from types import SimpleNamespace

import pytest
from datasets import Dataset
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import resolver  # noqa: F401  (registers the repo's custom OmegaConf resolvers)
from src.core.datasets import FineWebEduDataset, get_dataloader
from src.core.trainer import _check_micro_batches
from src.definitions import check_point, job_gpu_count

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STANDARD = os.path.join(REPO_ROOT, "configs", "standard")
STANDARD_CONFIGS = [
    "ame_h12",
    "dense_h12",
    "ame_h16",
    "dense_h16",
    "ame_h20",
    "dense_h20",
]
EOS = 0
SEQ = 8
BATCH = 16  # sequences per step: splits evenly over 1, 2 and 4 GPUs
STEPS = 6


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """256 documents of 3 to 40 words, each word its own token id."""
    rng = random.Random(0)
    texts = [
        " ".join(str(rng.randrange(1, 5000)) for _ in range(rng.randint(3, 40)))
        for _ in range(256)
    ]
    path = tmp_path_factory.mktemp("corpus") / "fineweb"
    Dataset.from_dict({"text": texts}).save_to_disk(str(path))
    return str(path)


def _tokenize_fn():
    def tokenize(examples):
        return {
            "input_ids": [
                [int(word) for word in text.split()] + [EOS]
                for text in examples["text"]
            ]
        }

    return tokenize


def _steps(corpus, monkeypatch, world_size, num_workers, world_size_independent=True):
    """Each step's sequences, gathered over the ranks in rank order."""
    steps = [[] for _ in range(STEPS)]
    for rank in range(world_size):
        monkeypatch.setenv("WORLD_SIZE", str(world_size))
        monkeypatch.setenv("RANK", str(rank))
        dataset = FineWebEduDataset(
            sequence_length=SEQ,
            tokenize_fn=_tokenize_fn(),
            path=corpus,
            seed=123,
            shuffle=True,
            world_size_independent=world_size_independent,
            holdout="train",
            holdout_rows=16,
        )
        loader = get_dataloader(
            dataset, total_batch_size=BATCH, num_workers=num_workers
        )
        for step, batch in zip(range(STEPS), loader):
            assert tuple(batch.shape) == (BATCH // world_size, SEQ)
            steps[step].extend(tuple(row.tolist()) for row in batch)
    return steps


@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("world_size", [2, 4])
def test_a_step_trains_on_the_same_sequences_at_any_gpu_count(
    corpus, monkeypatch, world_size, num_workers
):
    """The GPUs of each step together hold the 1-GPU run's step, no more, no less."""
    one_gpu = _steps(corpus, monkeypatch, 1, num_workers)
    many = _steps(corpus, monkeypatch, world_size, num_workers)
    for step, (single, split) in enumerate(zip(one_gpu, many)):
        assert len(single) == BATCH
        assert sorted(split) == sorted(single), (
            f"step {step} trains on other sequences at {world_size} GPUs "
            f"(num_workers={num_workers})"
        )


@pytest.mark.parametrize("num_workers", [0, 2])
def test_one_gpu_trains_on_exactly_what_it_did_before_the_switch(
    corpus, monkeypatch, num_workers
):
    """At one GPU world_size_independent is a no-op, batch for batch and row for row.

    So turning it on left every 1-GPU run's data as it was: the v2 baselines still
    pair with the v1 references in data (review_upgrades.md, U11).
    """
    before = _steps(corpus, monkeypatch, 1, num_workers, world_size_independent=False)
    after = _steps(corpus, monkeypatch, 1, num_workers, world_size_independent=True)
    assert after == before


def test_without_the_switch_the_gpu_count_changes_the_data(corpus, monkeypatch):
    """What the switch removes, as a fact: per-GPU shards train on other data.

    Also proves the comparison above can fail.
    """
    one_gpu = _steps(corpus, monkeypatch, 1, 0, world_size_independent=False)
    two_gpus = _steps(corpus, monkeypatch, 2, 0, world_size_independent=False)
    assert [sorted(s) for s in two_gpus] != [sorted(s) for s in one_gpu]


# ---------------------------------------------------------------------------
# The shipped configs, and the checks that refuse an uneven split.


def _standard(name, overrides=()):
    with initialize_config_dir(version_base=None, config_dir=STANDARD):
        return compose(config_name=name, overrides=list(overrides))


def _point(name, overrides=()):
    """A grid point as run_exp.py hands it to check_point: a plain, unresolved dict."""
    return OmegaConf.to_container(_standard(name, overrides), resolve=False)


@pytest.mark.parametrize("name", STANDARD_CONFIGS)
def test_the_standard_runs_read_the_whole_stream_and_split_evenly(name):
    cfg = _standard(name)
    train = cfg.trainer.train_dataloader
    assert train.dataset.world_size_independent is True
    assert train.num_workers == 2, "num_workers is part of the data order (A1)"
    check_point(_point(name))  # batch_size divides GPUs x accumulation


def test_the_job_gpu_count_comes_from_the_gres():
    assert job_gpu_count(_standard("ame_h12")) == 1
    assert job_gpu_count(_standard("ame_h20")) == 4
    two_nodes = OmegaConf.create(
        {"infrastructure": {"slurm": {"gres": "gpu:h100:4", "nodes": 2}}}
    )
    assert job_gpu_count(two_nodes) == 8
    assert (
        job_gpu_count(
            OmegaConf.create({"infrastructure": {"slurm": {"gres": "gpu:1"}}})
        )
        == 1
    )
    # a local run, and a gres the cluster config has yet to fill
    assert job_gpu_count(OmegaConf.create({})) is None
    assert (
        job_gpu_count(OmegaConf.create({"infrastructure": {"slurm": {"gres": "???"}}}))
        is None
    )


@pytest.mark.parametrize(
    "overrides, message",
    [
        # 512 over 3 GPUs: 170 each, a step of 510
        (["infrastructure.slurm.gres=gpu:hopper:3"], "the job's 3 GPUs x"),
        # 512 on 1 GPU in 3 micro-batches: 171, 171, 170, weighted equally
        (
            ["trainer.gradient_accumulation_steps=3"],
            "gradient_accumulation_steps=3 = 3",
        ),
    ],
)
def test_submission_refuses_a_batch_that_does_not_split(overrides, message):
    with pytest.raises(ValueError, match=message):
        check_point(_point("ame_h12", overrides))


def test_the_loader_refuses_a_batch_that_does_not_split_over_the_gpus(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "3")
    monkeypatch.setenv("RANK", "0")
    with pytest.raises(ValueError, match="does not split evenly over 3 GPUs"):
        get_dataloader(None, total_batch_size=512, num_workers=0)


@pytest.mark.parametrize("per_gpu, accumulation", [(10, 4), (10, 6), (512, 3)])
def test_the_trainer_refuses_unequal_micro_batches(per_gpu, accumulation):
    """torch.chunk would give 3/3/3/1, five chunks of 2, or 171/171/170."""
    with pytest.raises(ValueError, match="equal micro-batches"):
        _check_micro_batches(SimpleNamespace(batch_size=per_gpu), accumulation)


def test_even_micro_batches_pass_and_no_loader_is_no_check():
    _check_micro_batches(SimpleNamespace(batch_size=32), 16)
    _check_micro_batches(None, 16)  # an eval-only trainer has no train loader
