from dataclasses import dataclass
import functools
from typing import Any
from enum import Enum
from typing_extensions import Self

from src.utils import is_float


@dataclass
class EvaluationMetric:
    method: functools.partial
    params: dict[str, Any]

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        return self.method(*args, **kwargs, **self.params)
        

class EvaluationTask(str, Enum):
    CHANGE_GRADED = "change_graded"
    CHANGE_BINARY = "change_binary"
    COMPARE = "COMPARE"
    SEMANTIC_PROXIMITY = "semantic_proximity"
    CLUSTERING = "clustering"

    @classmethod
    def from_str(cls, s: str) -> Self:
        return cls[s.upper()]


@dataclass
class Evaluation:
    task: EvaluationTask | str
    metric: EvaluationMetric
    keep: int | None

    def __post_init__(self) -> None:
        self.task: EvaluationTask = EvaluationTask.from_str(self.task)

    def __call__(self, predictions: list[float | int], labels: list[float | int]) -> float:
        score = self.metric(labels, predictions)
        if not is_float(score) and self.keep is not None:
            return score[self.keep] 
        return score


