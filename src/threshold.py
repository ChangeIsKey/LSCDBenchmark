from dataclasses import dataclass
from typing import Any, Iterable
from abc import ABC, abstractmethod
import numpy as np


class ThresholdFn(ABC):
    @abstractmethod
    def __call__(self, predictions: list[float]) -> float:
        raise NotImplementedError


@dataclass
class MeanStd(ThresholdFn):
    t: float

    def __call__(self, predictions: list[float]) -> float:
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        return mean + self.t * std