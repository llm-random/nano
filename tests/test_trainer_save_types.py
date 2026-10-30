"""Tier 1: the checkpoint save-type contract between core and its subclasses.

See documents/notes/main_baseline_audit.md. src/core/trainer.py used to hard-code the
projected-compression save type; it now declares only its own and lets subclasses widen
SUPPORTED_SAVE_TYPES and override save_final_checkpoint. These tests pin that contract
down, so the layering cannot quietly come undone.

The validation matters beyond tidiness: a save type the trainer cannot write is rejected
before training starts rather than after, which on a multi-hour job is the difference
between a clear error and a lost run.
"""

import pytest
import torch
from omegaconf import OmegaConf

from src.core.trainer import Trainer
from src.core.trainer_distillation import TrainerDistillation
from src.projected_compression.trainer import PCFinalizeTrainer, PCTrainer


def _validator(trainer_cls, save_type):
    """Exercise _validate_save_type without building a whole Trainer.

    A Trainer needs a model, dataloaders, an optimizer and a metric logger; the
    validation only reads self.checkpoint and the class's SUPPORTED_SAVE_TYPES.
    """

    class Stub:
        SUPPORTED_SAVE_TYPES = trainer_cls.SUPPORTED_SAVE_TYPES
        FINAL_ONLY_SAVE_TYPES = trainer_cls.FINAL_ONLY_SAVE_TYPES
        _validate_save_type = trainer_cls._validate_save_type

    Stub.__name__ = trainer_cls.__name__
    stub = Stub()
    stub.checkpoint = OmegaConf.create({"save": {"type": save_type}})
    return stub


@pytest.mark.parametrize("save_type", ["nano", "huggingface"])
def test_core_trainer_accepts_its_own_save_types(save_type):
    _validator(Trainer, save_type)._validate_save_type()


@pytest.mark.parametrize(
    "save_type", ["pc_finalize", "hf_only", "nano_and_hf", "nonsense"]
)
def test_core_trainer_rejects_save_types_it_cannot_write(save_type):
    """Including pc_finalize: that is a projected-compression concern now."""
    with pytest.raises(ValueError, match="save.type"):
        _validator(Trainer, save_type)._validate_save_type()


def test_pc_finalize_trainer_accepts_pc_finalize():
    _validator(PCFinalizeTrainer, "pc_finalize")._validate_save_type()


@pytest.mark.parametrize("save_type", ["hf_only", "nano_and_hf"])
def test_pc_trainer_accepts_the_hf_save_types(save_type):
    _validator(PCTrainer, save_type)._validate_save_type()


def test_subclasses_only_widen_the_supported_set():
    """A subclass may add save types; it must never drop one core supports."""
    for subclass in (TrainerDistillation, PCTrainer, PCFinalizeTrainer):
        missing = set(Trainer.SUPPORTED_SAVE_TYPES) - set(subclass.SUPPORTED_SAVE_TYPES)
        assert not missing, f"{subclass.__name__} dropped {sorted(missing)}"


def test_pc_finalize_is_written_only_at_the_end():
    """Finalization merges the projections into the weights in place, so running it on
    a periodic checkpoint interval would corrupt the run rather than save it."""
    assert "pc_finalize" in PCFinalizeTrainer.FINAL_ONLY_SAVE_TYPES
    assert "pc_finalize" not in Trainer.FINAL_ONLY_SAVE_TYPES
    assert set(Trainer.FINAL_ONLY_SAVE_TYPES) <= set(Trainer.SUPPORTED_SAVE_TYPES)
    assert set(PCFinalizeTrainer.FINAL_ONLY_SAVE_TYPES) <= set(
        PCFinalizeTrainer.SUPPORTED_SAVE_TYPES
    )


def test_core_trainer_has_no_projected_compression_attributes():
    """The method that used to live here moved to PCFinalizeTrainer."""
    assert not hasattr(Trainer, "save_pc_finalized_checkpoint")
    assert hasattr(PCFinalizeTrainer, "save_pc_finalized_checkpoint")


# ---------------------------------------------------------------------------
# Resuming, and what __attrs_post_init__ does with the eval stream.
#
# __attrs_post_init__ used to advance the eval stream to where a resumed run left off:
#
#     if self.start_step > 0:
#         n_skip_eval_batches = (start_step - 1) // self.eval_interval * n_eval_steps
#
# eval_interval is 0 whenever eval is off, which 31 shipped configs do -- the
# PCD-50 repro among them -- so every such run raised ZeroDivisionError on
# resume, before training began. A preempted job simply could not be restarted.
# It needs no GPU and no data to reproduce, but it only fires on a resume, which
# is why it sat behind the one path nothing routinely runs. Found on helios job
# 25327019.
#
# The skip is now gone entirely, because eval() restarts the stream on every call so
# that each evaluation scores the same held-out samples. A wound-forward iterator is
# discarded unread by that restart, so the skip bought nothing and cost ~10k batches
# through the tokenizer at the start of a long resume. These tests pin both halves:
# construction must not touch the eval stream at all, and must not divide by
# eval_interval.


def _post_init_with(eval_interval, next_step, eval_dataloader=None):
    """Run Trainer.__attrs_post_init__ against the smallest object that satisfies it."""

    class Stub:
        __attrs_post_init__ = Trainer.__attrs_post_init__
        _validate_save_type = Trainer._validate_save_type
        final_eval = Trainer.final_eval
        SUPPORTED_SAVE_TYPES = Trainer.SUPPORTED_SAVE_TYPES
        FINAL_ONLY_SAVE_TYPES = Trainer.FINAL_ONLY_SAVE_TYPES

    stub = Stub()
    stub.checkpoint = OmegaConf.create({"save": {"type": "nano"}})
    stub.training_state = {"processed_tokens": 0, "next_step": next_step}
    stub.model = torch.nn.Linear(2, 2)
    stub.eval_dataloader = eval_dataloader
    stub.eval_interval = eval_interval
    stub.n_eval_steps = 2
    stub.__attrs_post_init__()
    return stub


def test_resuming_with_eval_disabled_does_not_divide_by_zero():
    """eval_interval: 0 plus a resume must not raise -- 31 configs ship that way."""
    stub = _post_init_with(eval_interval=0, next_step=10)
    assert stub.start_step == 10
    assert stub.step == 9


def test_construction_never_reads_the_eval_stream():
    """Not one batch, even resuming deep into a run with eval on.

    This is the case the old skip loop was written for: next_step 11 at eval_interval 5
    wound the stream forward by (11 - 1) // 5 * 2 == 4 batches. eval() then threw that
    iterator away and built a fresh one, so the 4 batches were read and dropped. On a
    real resume the number is in the thousands, and every one of them goes through the
    tokenizer before the first training step.
    """
    consumed = []

    class CountingLoader:
        def __iter__(self):
            def gen():
                for i in range(1000):
                    consumed.append(i)
                    yield i

            return gen()

    stub = _post_init_with(
        eval_interval=5, next_step=11, eval_dataloader=CountingLoader()
    )
    assert consumed == [], f"construction read {len(consumed)} eval batches"
    assert stub.eval_iterator is None


def test_starting_from_scratch_skips_nothing():
    stub = _post_init_with(eval_interval=0, next_step=0)
    assert stub.start_step == 0
    assert stub.step == -1


def test_resuming_with_a_loader_but_eval_off_does_not_divide_by_zero():
    """No shipped config pairs eval_interval: 0 with a live eval_dataloader, but the
    guard has to cover it: without the eval_interval check the division still runs and
    still raises, and nothing else in the suite would notice."""

    class CountingLoader:
        def __iter__(self):
            return iter(range(1000))

    stub = _post_init_with(
        eval_interval=0, next_step=10, eval_dataloader=CountingLoader()
    )
    assert stub.start_step == 10


# ---------------------------------------------------------------------------
# Checkpoint folder naming.
#
# Trainer.step is start_step - 1, so -1 before the first step. A run with
# n_steps: 0 -- how every finalize and conversion pass is configured -- never
# advances it, and _should_save_final_checkpoint is satisfied at
# step >= n_steps - 1, so the finalized model was written to a folder named
# literally "step_-1". Observed on helios job 25325254, the first end-to-end run
# of llama_1B_comp_finalize.


def test_a_finalize_pass_does_not_write_to_a_negative_step_folder():
    from src.core.checkpointing import step_checkpoint_path

    path = step_checkpoint_path("/tmp/ckpt", -1)
    assert not path.endswith("step_-1"), f"negative step leaked into the path: {path}"
    assert path.endswith("step_0"), path


@pytest.mark.parametrize("step", [0, 1, 10, 4095])
def test_ordinary_steps_are_unchanged(step):
    from src.core.checkpointing import step_checkpoint_path

    assert step_checkpoint_path("/tmp/ckpt", step).endswith(f"step_{step}")


# ---------------------------------------------------------------------------
# The final eval.
#
# Scores the held-out set once when training ends and logs it as final_ce. It
# has to run before the final checkpoint: pc_finalize merges the projections and
# sets parameters to None off rank 0, so after saving there is no model left.


class _RecordingLogger:
    def __init__(self):
        self.logged = {}

    def log(self, name, value):
        self.logged[name] = value

    def flush(self):
        pass

    def flush_accumulated_metrics(self):
        pass


class _FinalEvalStub:
    """The smallest object Trainer.final_eval needs."""

    final_eval = Trainer.final_eval

    def __init__(
        self, loader, max_samples, n_eval_steps=2, gradient_accumulation_steps=1
    ):
        self.eval_dataloader = loader
        if loader is not None:
            loader.dataset = type("D", (), {"max_samples": max_samples})()
        self.n_eval_steps = n_eval_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = torch.device("cpu")
        self.metric_logger = _RecordingLogger()
        self.model = torch.nn.Linear(2, 2)
        self.seen = 0

    def calculate_loss(self, batch):
        self.seen += 1
        return type("L", (), {"reported_loss": torch.tensor(float(self.seen))})()


class _Loader(list):
    """A list of batches that also carries a .dataset, like a DataLoader."""


def _batches(n):
    return _Loader(torch.zeros(1, 2) for _ in range(n))


def test_final_eval_logs_final_ce_over_the_whole_fixed_set():
    stub = _FinalEvalStub(_batches(4), max_samples=4)
    stub.final_eval()
    assert "final_ce" in stub.metric_logger.logged, "final_ce was not logged"
    # losses 1,2,3,4 -> mean 2.5; all four batches were read, not just n_eval_steps
    assert stub.metric_logger.logged["final_ce"] == pytest.approx(2.5)
    assert stub.seen == 4


def test_final_eval_does_not_hang_on_an_uncapped_stream():
    """Without max_samples the eval stream is endless, so it must fall back to a budget."""
    stub = _FinalEvalStub(_batches(100), max_samples=None, n_eval_steps=3)
    stub.final_eval()
    assert stub.seen == 3, f"read {stub.seen} batches instead of n_eval_steps"


def test_final_eval_is_a_no_op_without_an_eval_dataloader():
    stub = _FinalEvalStub(None, max_samples=None)
    stub.final_eval()
    assert stub.metric_logger.logged == {}


def test_final_eval_puts_the_model_in_eval_mode():
    stub = _FinalEvalStub(_batches(2), max_samples=2)
    stub.model.train()
    stub.final_eval()
    assert not stub.model.training, "the model was not switched to eval mode"


def test_train_runs_the_final_eval_before_saving():
    """Order matters, and nothing else in the suite would notice if it changed.

    PCFinalizeTrainer's final save merges the projections and sets the parameters to
    None off rank 0, so a final eval placed after it would score a model that no longer
    exists. This reads the source rather than driving a full training run, because the
    guarantee is about call order, not about what either call computes. (Naming that
    here is fine: this is a test, not src/core, which must stay PC-free.)
    """
    import inspect

    source = inspect.getsource(Trainer.train)
    assert "self.final_eval()" in source, "train() does not run the final eval"
    assert "self.save_final_checkpoint()" in source
    assert source.index("self.final_eval()") < source.index(
        "self.save_final_checkpoint()"
    ), "the final eval runs after the checkpoint, so pc_finalize leaves nothing to score"


# ---------------------------------------------------------------------------
# `eval_dataloader: None` is a string.
#
# YAML's null is `null` or `~`; a bare `None` is the three-letter string. 37 shipped
# configs switch eval off with `eval_dataloader: None`, so the trainer gets "None".
# That string is not None and does have __iter__, so it satisfied every "is there a
# dataloader" guard, and final_eval -- which runs after the last step and *before* the
# final checkpoint -- iterated its characters and raised AttributeError on 'N'.to().
# Every PC run in the repo would have trained to completion and then died unsaved.
# __attrs_post_init__ now normalises a str to None.


@pytest.mark.parametrize("sentinel", ["None", "null", "~", ""])
def test_a_string_eval_dataloader_is_normalised_to_none(sentinel):
    stub = _post_init_with(eval_interval=0, next_step=0, eval_dataloader=sentinel)
    assert stub.eval_dataloader is None


def test_a_real_eval_dataloader_survives_normalisation():
    """The normalisation must not reach anything that is actually a dataloader."""

    class Loader:
        def __iter__(self):
            return iter(range(3))

    loader = Loader()
    stub = _post_init_with(eval_interval=2, next_step=0, eval_dataloader=loader)
    assert stub.eval_dataloader is loader


@pytest.mark.parametrize("absent", [None, "None"])
def test_final_eval_is_a_no_op_after_a_string_eval_dataloader(absent):
    """Construct exactly as a shipped config does, then run the whole final_eval.

    This is the end-to-end shape of the bug: `eval_dataloader: None` in YAML, a trainer
    built from it, and final_eval reached after the last step. The stub carries no
    model, device, calculate_loss or metric_logger, so anything past the guard raises
    AttributeError -- which is the assertion. Before the fix it raised
    "'str' object has no attribute 'to'" here.

    final_eval runs before save_final_checkpoint, so anything it raises costs the whole
    run's output, not just the metric.
    """
    stub = _post_init_with(eval_interval=0, next_step=0, eval_dataloader=absent)
    stub.final_eval()  # must not raise


# ---------------------------------------------------------------------------
# final_ce is the loss over the whole eval set, not a mean of batch means.


class _SizedLoader(list):
    """Batches of differing sizes, with a .dataset like a DataLoader's."""


def _sized_stub(sizes, losses, gradient_accumulation_steps=1):
    loader = _SizedLoader(torch.zeros(size, 2) for size in sizes)
    stub = _FinalEvalStub(
        loader,
        max_samples=sum(sizes),
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    seen = iter(losses)
    stub.calculate_loss = lambda batch: type(
        "L", (), {"reported_loss": torch.tensor(next(seen))}
    )()
    return stub


def test_final_ce_weights_each_batch_by_its_size():
    """A short last batch must not count as much as a full one.

    8 samples at loss 1.0 and 2 samples at loss 6.0 is 20 over 10 samples, so 2.0. The
    mean of the two batch means would be 3.5. This matters because a fixed-size eval set
    split across ranks does not divide evenly into batches.
    """
    stub = _sized_stub([8, 2], [1.0, 6.0])
    stub.final_eval()
    assert stub.metric_logger.logged["final_ce"] == pytest.approx(2.0)


def test_final_ce_skips_batches_too_short_to_chunk():
    """calculate_loss divides by gradient_accumulation_steps however many chunks it got.

    A batch of 2 split 4 ways yields 2 chunks, each divided by 4, so the loss comes back
    at half its true value. Folding that into final_ce would bias the number down with
    nothing to show for it; the stub batch is dropped and warned about instead, leaving
    the full batch's 1.0.
    """
    stub = _sized_stub([8, 2], [1.0, 6.0], gradient_accumulation_steps=4)
    stub.final_eval()
    assert stub.metric_logger.logged["final_ce"] == pytest.approx(1.0)
