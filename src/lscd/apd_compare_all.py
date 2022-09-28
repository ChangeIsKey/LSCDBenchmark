import numpy as np

from src.lscd.model import GradedModel
from src.target import Target
from src import wic


class ApdCompareAll(GradedModel):
    wic: wic.Model

    def predict(self, targets: list[Target]) -> dict[str, float]:
        predictions = {}
        for target in targets:
            use_pairs = target.use_pairs(pairing="COMPARE", sampling="all")
            similarities = self.wic.similarities(use_pairs)
            apd = np.mean(similarities).item()
            predictions[target.name] = apd

        return predictions
