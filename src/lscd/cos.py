import numpy as np
from scipy.spatial import distance
from tqdm import tqdm

from src.lscd.model import GradedLSCDModel
from src.lemma import Lemma
from src.use import Use
from src.wic import ContextualEmbedder


class Cos(GradedLSCDModel):
    wic: ContextualEmbedder

    def predict(self, lemma: Lemma) -> float:
        earlier_df = lemma.uses_df[lemma.uses_df.grouping == lemma.groupings[0]]
        later_df = lemma.uses_df[lemma.uses_df.grouping == lemma.groupings[1]]

        earlier = [Use.from_series(s) for _, s in earlier_df.iterrows()]
        later = [Use.from_series(s) for _, s in later_df.iterrows()]

        with self.wic:
            earlier_vectors = np.vstack([self.wic.encode(use) for use in earlier])
            later_vectors = np.vstack([self.wic.encode(use) for use in later])

        earlier_avg = earlier_vectors.mean(axis=0)
        later_avg = later_vectors.mean(axis=0)
        cos = distance.cosine(earlier_avg, later_avg)
        return float(cos)
    
    def predict_all(self, lemmas: list[Lemma]) -> list[float]:
        return [self.predict(lemma) for lemma in lemmas]