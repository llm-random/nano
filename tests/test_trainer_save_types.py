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
