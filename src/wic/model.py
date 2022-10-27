from abc import (
    ABC,
    abstractmethod,
)

import numpy as np
import more_itertools as mit
from pandas import DataFrame
from pydantic import BaseModel
from tqdm import tqdm

from src.target import Lemma
from src.use import (
    Use,
    UseID,
)


class WICModel(BaseModel, ABC):
    @abstractmethod
    def predict(
        self, use_pairs: list[tuple[Use, Use]]
    ) -> dict[tuple[UseID, UseID], float]:
        ...



class ThresholdedWicModel(BaseModel):
    thresholds: list[float]
    wic: WICModel

    def predict(
        self, use_pairs: list[tuple[Use, Use]]
    ) -> dict[tuple[UseID, UseID], float]:
        predictions = self.wic.predict(use_pairs)
        new_predictions = {}
        threshold_spans = list(
            mit.windowed([float("-inf"), *self.thresholds, float("inf")], n=2, step=1)
        )
        for use_pair_id, pred in predictions.items():
            for i, (floor, ceil) in enumerate(threshold_spans):
                if floor <= pred < ceil:
                    new_predictions[use_pair_id] = float(i)
                    break
        return new_predictions
