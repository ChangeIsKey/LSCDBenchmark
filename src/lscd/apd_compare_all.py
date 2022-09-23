from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from src.lscd.lscd_model import LSCDModel
from src.target import Pairing, Sampling, Target
from src.threshold import ThresholdFn
from src.wic.model import WICModel


@dataclass
class ApdCompareAll(LSCDModel):
    wic: WICModel
    threshold_fn: ThresholdFn | None

    def predict(self, targets: list[Target]) -> list[float | int]:
        predictions = []
        for target in targets:
            use_pairs = target.use_pairs(pairing=Pairing.COMPARE, sampling=Sampling.ALL)
            similarities = self.wic.predict(use_pairs)
            mean = np.mean(similarities)
            predictions.append(mean)

        if self.threshold_fn is not None:
            threshold = self.threshold_fn(predictions)
            predictions = [int(p >= threshold) for p in predictions]

        return predictions
