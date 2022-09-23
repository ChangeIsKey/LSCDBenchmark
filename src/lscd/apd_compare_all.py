from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
from tqdm import tqdm

from src.lscd.lscd_model import LSCDModel
from src.target import Pairing, Sampling, Target
from src.wic.model import WICModel


@dataclass
class ApdCompareAll(LSCDModel):
    wic: WICModel
    threshold_fn: Callable[[list[float]], list[int]] | None

    def predict(self, targets: list[Target]) -> tuple[list[str], list[float | int]]:
        predictions = {}
        for target in targets:
            use_pairs = target.use_pairs(pairing=Pairing.COMPARE, sampling=Sampling.ALL)
            similarities = self.wic.predict(use_pairs)
            predictions[target.name] = np.mean(similarities)

        # if self.threshold_fn is not None:
        #     threshold = self.threshold_fn(predictions)
        #     predictions = {target_name: int(p >= threshold) for target_name, p in predictions.items()}

        return list(predictions.keys()), list(predictions.values())
