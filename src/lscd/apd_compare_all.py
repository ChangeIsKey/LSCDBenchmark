from typing import Callable
import numpy as np

from src.lscd.model import Model
from src.target import Target
from src import wic


class ApdCompareAll(Model):
    wic: wic.Model
    threshold_fn: Callable[[list[float]], list[int]] | None

    def predict(self, targets: list[Target]) -> list[int] | list[float]:
        predictions = []
        for target in targets:
            use_pairs = target.use_pairs(pairing="COMPARE", sampling="all")
            similarities = self.wic.predict(use_pairs)
            apd = np.mean(similarities).item()
            predictions.append(apd)

        if self.threshold_fn is not None:
            predictions = self.threshold_fn(predictions)

        return predictions
