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


ApdCompareAll = partial(APD, pairing="COMPARE", sampling="all")
ApdEarlierAll = partial(APD, pairing="EARLIER", sampling="all")
ApdLaterAll = partial(APD, pairing="LATER", sampling="all")
ApdCompareAnnotated = partial(APD, pairing="COMPARE", sampling="annotated")
ApdEarlierAnnotated = partial(APD, pairing="EARLIER", sampling="annotated")
ApdLaterAnnotated = partial(APD, pairing="LATER", sampling="annotated")