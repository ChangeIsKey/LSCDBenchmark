import numpy as np
from tqdm import tqdm

from src import wic
from src.lscd.model import GradedLSCDModel
from src.lemma import Lemma


class ApdCompareAll(GradedLSCDModel):
    wic: wic.WICModel

    def predict(self, lemma: Lemma) -> float:
        use_pairs = lemma.use_pairs(pairing="COMPARE", sampling="all")
        similarities = self.wic.predict(use_pairs)
        return np.mean(similarities).item()
