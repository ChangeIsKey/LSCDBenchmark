from abc import (
    ABC,
    abstractmethod,
)

import numpy as np
import more_itertools as mit
from pandas import DataFrame
from pydantic import BaseModel
from tqdm import tqdm

from src.lemma import Lemma
from src.use import (
    Use,
    UseID,
)


class WICModel(BaseModel, ABC):
    @abstractmethod
    def predict(self, use_pairs: list[tuple[Use, Use]]) -> list[float]:
        ...


class ThresholdedWicModel(BaseModel):
    thresholds: list[float]
    wic: WICModel

    def predict(self, use_pairs: list[tuple[Use, Use]]) -> list[float]:
        predictions = self.wic.predict(use_pairs)
        threshold_spans = list(
            mit.windowed([float("-inf"), *self.thresholds, float("inf")], n=2, step=1)
        )
        for i, x in enumerate(predictions):
            for j, (floor, ceil) in enumerate(threshold_spans):
                assert floor is not None and ceil is not None
                if floor <= x < ceil:
                    predictions[i] = float(j)
                    break
        return predictions
