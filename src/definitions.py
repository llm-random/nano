from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)
from dataclasses import dataclass
from typing import Literal
from typing import Optional, List
from attr import dataclass

@dataclass
class AttentionConfig:
    mode: str
    n_heads: int


@dataclass
class FeedForwardConfig:
    mode: str


@dataclass
class BlockConfig:
    attention: AttentionConfig
    feedforward: FeedForwardConfig
    residual_mode: str
    norm_class_mode: str


@dataclass
class TowerConfig:
    mode: str
    n_blocks: int
    block_config: BlockConfig


@dataclass
class EmbeddingConfig:
    mode: str


class Common(BaseModel):
    model_type: str
    sequence_length: int
    dmodel: int
    dff: int
    init_type: str
    init_scale: Optional[float]
    vocab_size: int
    head_norm: bool

class CommonCompression(Common):
    base_dmodel: int
    base_dff: int

class SchedulerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["cosine", "trapezoidal"]
    warmup_steps: NonNegativeInt


class CosineSchedulerConfig(SchedulerConfig):
    n_steps: PositiveInt
    final_lr_fraction: NonNegativeFloat


class TrapezoidalSchedulerConfig(SchedulerConfig):
    constant_steps: NonNegativeInt
    decay_steps: NonNegativeInt


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    learning_rate: PositiveFloat
    weight_decay: PositiveFloat
    scheduler: object
    gradient_accumulation_steps: PositiveInt
    n_steps: PositiveInt
    gradient_clipping: PositiveFloat
    evaluation: dict


class MetricLoggerConfig(BaseModel):
    type: Optional[str]
    project_name: Optional[str]
    name: Optional[str]
    tags: Optional[List[str]]
    heavy_metrics_calculation_interval: Optional[int]
    new_neptune_job: Optional[bool] = None
