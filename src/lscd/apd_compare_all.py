from typing import Callable
import numpy as np

from src.lscd.model import Model
from src.target import Target
from src import wic


class ApdCompareAll(Model):
    wic: wic.Model
    threshold_fn: Callable[[list[float]], float] | None

    def predict(self, targets: list[Target]) -> list[float | int]:
        predictions: dict[str, float | int] = {}
        for target in targets:
            use_pairs = target.use_pairs(pairing="COMPARE", sampling="all")
            similarities = self.wic.predict(use_pairs)
            predictions[target.name] = np.mean(similarities).item()

        if self.threshold_fn is not None:
            values = list(predictions.values())
            threshold = self.threshold_fn(values)
            predictions = {target_name: int(p >= threshold) for target_name, p in predictions.items()}

        return list(predictions.values())
