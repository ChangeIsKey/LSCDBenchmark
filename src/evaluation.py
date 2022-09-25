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
            score = self.metric(labels, predictions)
            if utils.is_list(score) and self.keep is not None:
                return score[self.keep]
            elif utils.is_number(score):
                return score
        return np.nan
