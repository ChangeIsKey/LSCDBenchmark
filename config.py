from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Tuple, Optional, Dict

from omegaconf import MISSING, DictConfig


class UsesType(Enum):
    annotated = "annotated"
    all = "all"
    sample = "sample"

    def __repr__(self):
        return self.value


@dataclass
class Uses:
    type: UsesType
    # make n and replacement optional
    n: int = MISSING
    replacement: bool = MISSING


class Pairing(Enum):
    EARLIER = "EARLIER"
    LATER = "LATER"
    COMPARE = "COMPARE"

    def __repr__(self):
        return self.value


class Task(Enum):
    lscd = "lscd"
    semantic_proximity = "semantic_proximity"

    def __repr__(self):
        return self.value


@dataclass
class Preprocessing:
    method: str
    params: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    name: str
    language: str
    grouping_1: int
    grouping_2: int
    pairing: Pairing
    uses: Uses
    task: Task
    preprocessing: Preprocessing


@dataclass
class ModelConfig:
    name: str
    layers: str
    measure: str
    cased: Optional[bool] = False


defaults = [
    {"dataset": MISSING},
    {"model": MISSING}
]


@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    dataset: DatasetConfig = MISSING
    model: ModelConfig = MISSING
