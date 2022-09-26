from typing import Callable
from enum import Enum
import numpy as np

from pydantic import BaseModel
from src import utils


class EvaluationTask(str, Enum):
    CHANGE_GRADED = "change_graded"
    CHANGE_BINARY = "change_binary"
    COMPARE = "COMPARE"
    SEMANTIC_PROXIMITY = "semantic_proximity"
    WSI = "wsi"


class Evaluation(BaseModel):
    task: EvaluationTask | None
    metric: Callable[..., int | float | list[float | int]] | None
    keep: int | None

    def __call__(self, predictions: list[float | int], labels: list[float | int]) -> int | float:
        if self.metric is not None:
            labels, predictions = self.filter_inputs(labels, predictions)
            score = self.metric(labels, predictions)
            if utils.is_list(score) and self.keep is not None:
                return score[self.keep]
            if utils.is_number(score):
                return score
        return np.nan

    @staticmethod
    def filter_inputs(labels: list[float | int], predictions: list[float | int]):
        combined = list(zip(labels, predictions))
        # https://stackoverflow.com/questions/8081545/how-to-convert-list-of-tuples-to-multiple-lists
        return map(list, zip(*combined))