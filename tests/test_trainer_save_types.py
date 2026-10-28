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
# Resuming with eval switched off.
#
# __attrs_post_init__ advances the eval stream to where a resumed run left off:
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


def _post_init_with(eval_interval, next_step, eval_dataloader=None):
    """Run Trainer.__attrs_post_init__ against the smallest object that satisfies it."""

    class Stub:
        __attrs_post_init__ = Trainer.__attrs_post_init__
        _validate_save_type = Trainer._validate_save_type
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


def test_resuming_with_eval_enabled_still_skips_the_eval_stream():
    """The guard must not switch the skip off when eval is genuinely on."""
    consumed = []

    class CountingLoader:
        def __iter__(self):
            def gen():
                for i in range(1000):
                    consumed.append(i)
                    yield i

            return gen()

    _post_init_with(eval_interval=5, next_step=11, eval_dataloader=CountingLoader())
    # (11 - 1) // 5 * 2 == 4 eval batches skipped
    assert len(consumed) == 4, f"expected 4 eval batches skipped, got {len(consumed)}"


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
