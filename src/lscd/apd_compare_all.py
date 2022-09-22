import numpy as np
from tqdm import tqdm

from lscd_model import LSCDModel
from src.target import Pairing, Sampling, Target


class ApdCompareAll(LSCDModel):
    def predict(self, targets: list[Target]) -> list[float]:
        predictions = []
        for target in tqdm(targets, desc="Processing targets"):
            use_pairs = target.use_pairs(pairing=Pairing.COMPARE, sampling=Sampling.ALL)
            similarities = self.wic_model.predict(use_pairs)
            predictions.append(np.mean(similarities))
        return predictions
