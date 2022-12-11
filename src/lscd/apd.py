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
        use_pairs_nested = [
            lemma.use_pairs(pairing=self.pairing, sampling=self.sampling) 
            for lemma in tqdm(lemmas, desc="Building use pairs", leave=False)
        ]
        use_pairs = [use_pair for sublist in use_pairs_nested for use_pair in sublist]
        id_pairs = [(use_0.identifier, use_1.identifier) for use_0, use_1 in use_pairs]
        self.wic.predictions = dict(zip(id_pairs, self.wic.predict_all(use_pairs=use_pairs)))
        return [self.predict(lemma) for lemma in tqdm(lemmas, desc="Processing lemmas")]


class DiaSense(GradedLSCDModel):
    wic: wic.WICModel

    def predict(self, lemma: Lemma) -> float:
        use_pairs_0 = lemma.use_pairs(pairing="COMPARE", sampling="all")
        use_pairs_1 = lemma.use_pairs(pairing="ALL", sampling="all")
        similarities_0 = self.wic.predict(use_pairs_0)
        similarities_1 = self.wic.predict(use_pairs_1)
        return np.mean(similarities_0).item() - np.mean(similarities_1).item()

    def predict_all(self, lemmas: list[Lemma]) -> list[float]:
        use_pairs_nested = [
            lemma.use_pairs(pairing="ALL", sampling="all") 
            for lemma in tqdm(lemmas, desc="Building use pairs", leave=False)
        ]
        use_pairs = [use_pair for sublist in use_pairs_nested for use_pair in sublist]
        id_pairs = [(use_0.identifier, use_1.identifier) for use_0, use_1 in use_pairs]
        self.wic.predictions = dict(zip(id_pairs, self.wic.predict_all(use_pairs=use_pairs)))
        return [self.predict(lemma) for lemma in tqdm(lemmas, desc="Processing lemmas")]

ApdCompareAll = partial(APD, pairing="COMPARE", sampling="all")
ApdEarlierAll = partial(APD, pairing="EARLIER", sampling="all")
ApdLaterAll = partial(APD, pairing="LATER", sampling="all")
ApdCompareAnnotated = partial(APD, pairing="COMPARE", sampling="annotated")
ApdEarlierAnnotated = partial(APD, pairing="EARLIER", sampling="annotated")
ApdLaterAnnotated = partial(APD, pairing="LATER", sampling="annotated")