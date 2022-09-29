from abc import ABC, abstractmethod
from typing import Callable

from pydantic import BaseModel

from src.target import Target


class GradedModel(BaseModel, ABC):

    @abstractmethod
    def predict(self, targets: list[Target]) -> dict[str, float]:
        ...


class BinaryThresholdModel(BaseModel):
    threshold_fn: Callable[[list[float]], list[int]]
    graded_model: GradedModel

    def predict(self, targets: list[Target]) -> dict[str, int]:
        predictions = self.graded_model.predict(targets)
        values = list(predictions.values())
        return dict(zip(predictions.keys(), self.threshold_fn(values)))
