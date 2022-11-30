from abc import (
    ABC,
    abstractmethod,
)
from typing import Callable

from pydantic import BaseModel

from src.lemma import Lemma


class GradedLSCDModel(BaseModel, ABC):
    @abstractmethod
    def predict(self, lemma: Lemma) -> float:
        ...

    def predict_all(self, lemmas: list[Lemma]) -> list[float]:
        use_pairs = []
        for lemma in lemmas:
            predictions.update({lemma.name: model.predict(lemma)})

class BinaryModel(BaseModel, ABC):
    ...


class BinaryThresholdModel(BinaryModel):
    threshold_fn: Callable[[list[float]], list[int]]
    graded_model: GradedLSCDModel

    def predict(self, graded_predictions: list[float]) -> list[int]:
        return self.threshold_fn(graded_predictions)
