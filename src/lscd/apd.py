import numpy as np
from tqdm import tqdm
from functools import partial

from src import wic
from src.lscd.model import GradedLSCDModel
from src.lemma import Lemma, Pairing, Sampling

class APD(GradedLSCDModel):
    wic: wic.WICModel
    pairing: Pairing
    sampling: Sampling

    def predict(self, lemma: Lemma) -> float:
        use_pairs = lemma.use_pairs(pairing=self.pairing, sampling=self.sampling)
        similarities = self.wic.predict(use_pairs)
        return np.mean(similarities).item()

    def predict_all(self, lemmas: list[Lemma]) -> list[float]:
        use_pairs_nested = [lemma.use_pairs(pairing=self.pairing, sampling=self.sampling) for lemma in tqdm(lemmas, desc="Building use pairs", leave=False)]
        use_pairs = [use_pair for sublist in use_pairs_nested for use_pair in sublist]
        self.wic.predict_all(use_pairs=use_pairs)
        return [self.predict(lemma) for lemma in tqdm(lemmas, desc="Processing lemmas")]


ApdCompareAll = partial(APD, pairing="COMPARE", sampling="all")
ApdEarlierAll = partial(APD, pairing="EARLIER", sampling="all")
ApdLaterAll = partial(APD, pairing="LATER", sampling="all")
ApdCompareAnnotated = partial(APD, pairing="COMPARE", sampling="annotated")
ApdEarlierAnnotated = partial(APD, pairing="EARLIER", sampling="annotated")
ApdLaterAnnotated = partial(APD, pairing="LATER", sampling="annotated")