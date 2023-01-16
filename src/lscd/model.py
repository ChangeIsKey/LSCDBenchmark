from abc import (
    ABC,
    abstractmethod,
)
import functools
from typing import Callable

from pydantic import BaseModel

from src.lemma import Lemma


class GradedLSCDModel(BaseModel, ABC):
    class Config:
        json_encoders = {
            functools.partial: lambda f: f.func.__name__
        }

    @abstractmethod
    def predict(self, lemma: Lemma) -> float:
        ...

    @abstractmethod
    def predict_all(self, lemmas: list[Lemma]) -> list[float]:
        ...
        

class BinaryModel(BaseModel, ABC):
    class Config:
        json_encoders = {
            functools.partial: lambda f: f.func.__name__
        }


class BinaryThresholdModel(BinaryModel):
    class Config:
        json_encoders = {
            functools.partial: lambda f: f.func.__name__
        }
    threshold_fn: Callable[[list[float]], list[int]]
    graded_model: GradedLSCDModel

    def predict(self, graded_predictions: list[float]) -> list[int]:
        return self.threshold_fn(graded_predictions)
