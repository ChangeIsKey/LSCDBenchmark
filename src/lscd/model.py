from abc import (
    ABC,
    abstractmethod,
)
from typing import Callable

from pydantic import BaseModel

from src.lemma import Lemma


class GradedModel(BaseModel, ABC):
    @abstractmethod
    def predict(self, lemma: Lemma) -> float:
        ...


class BinaryModel(BaseModel, ABC):
    ...    

class BinaryThresholdModel(BinaryModel):
    threshold_fn: Callable[[list[float]], list[int]]
    graded_model: GradedModel

    def predict(self, graded_predictions: list[float]) -> list[int]:
        return self.threshold_fn(graded_predictions)
