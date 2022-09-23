from typing import Callable, Sequence
from enum import Enum

from pydantic import BaseModel
from src import utils


class EvaluationTask(str, Enum):
    CHANGE_GRADED = "change_graded"
    CHANGE_BINARY = "change_binary"
    COMPARE = "COMPARE"
    SEMANTIC_PROXIMITY = "semantic_proximity"
    WSI = "wsi"


class Evaluation(BaseModel):
    task: EvaluationTask
    metric: Callable[..., int | float | list[float | int]]
    keep: int | None

    def __call__(self, predictions: list[float | int], labels: list[float | int]) -> int | float:
        score = self.metric(labels, predictions)
        if utils.is_list(score) and self.keep is not None:
            return score[self.keep]
        elif utils.is_number(score):
            return score
        raise ValueError("evaluation metric does not return a number")
