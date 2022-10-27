import numpy as np
from scipy.spatial import distance
from tqdm import tqdm

from src.lscd.model import GradedLSCDModel
from src.target import Lemma
from src.use import Use
from src.wic import ContextualEmbedder


class Cos(GradedLSCDModel):
    wic: ContextualEmbedder

    def predict(self, targets: list[Lemma]) -> dict[str, float]:
        predictions = {}
        for target in tqdm(targets, desc="Generating target predictions"):
            earlier_df = target.uses_df[target.uses_df.grouping == target.groupings[0]]
            later_df = target.uses_df[target.uses_df.grouping == target.groupings[1]]

            earlier = [Use.from_series(s) for _, s in earlier_df.iterrows()]
            later = [Use.from_series(s) for _, s in later_df.iterrows()]

            with self.wic:
                earlier_vectors = np.vstack([self.wic.encode(use) for use in earlier])
                later_vectors = np.vstack([self.wic.encode(use) for use in later])

            earlier_avg = earlier_vectors.mean(axis=0)
            later_avg = later_vectors.mean(axis=0)
            cos = distance.cosine(earlier_avg, later_avg)
            predictions[target.name] = cos

        return predictions
