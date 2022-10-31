import numpy as np
from tqdm import tqdm

from src import wic
from src.lscd.model import GradedModel
from src.lemma import Lemma


class ApdCompareAll(GradedModel):
    wic: wic.WICModel

    def predict(self, lemma: Lemma) -> float:
        use_pairs = lemma.use_pairs(pairing="COMPARE", sampling="all")
        similarities = self.wic.predict(use_pairs)
        return np.mean(similarities).item()
