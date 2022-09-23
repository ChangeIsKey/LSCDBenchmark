from __future__ import annotations
from dataclasses import dataclass
import functools
from genericpath import isfile
from typing import Any, Callable, Iterable, Sequence, TypeVar
from enum import Enum
from typing_extensions import Self
from src import utils



class EvaluationTask(str, Enum):
    CHANGE_GRADED = "change_graded"
    CHANGE_BINARY = "change_binary"
    COMPARE = "COMPARE"
    SEMANTIC_PROXIMITY = "semantic_proximity"
    WSI = "wsi"

    @classmethod
    def from_str(cls, s: str) -> EvaluationTask:
        return cls[s.upper()]


@dataclass
class Evaluation:
    task: EvaluationTask | str  # type: ignore
    metric: Callable[..., int | float | list[float | int]]
    keep: int | None

    def __post_init__(self) -> None:
        self.task: EvaluationTask = EvaluationTask.from_str(self.task)

    def __call__(self, predictions: list[float | int], labels: list[float | int]) -> int | float:
        score = self.metric(labels, predictions)
        if utils.is_list(score) and self.keep is not None:
            return score[self.keep]
        elif utils.is_number(score):
            return score
        raise ValueError("evaluation metric does not return a number")
