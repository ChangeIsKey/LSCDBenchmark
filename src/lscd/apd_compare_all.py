import numpy as np
from tqdm import tqdm

from src import wic
from src.lscd.model import GradedModel
from src.target import Target


class ApdCompareAll(GradedModel):
    wic: wic.Model

    def predict(self, targets: list[Target]) -> dict[str, float]:
        predictions = {}
        for target in tqdm(targets, desc="Generating target predictions"):
            use_pairs = target.use_pairs(pairing="COMPARE", sampling="all")
            with self.wic:
                similarities = self.wic.similarities(use_pairs)
            apd = np.mean(similarities).item()
            predictions[target.name] = apd

        return predictions
