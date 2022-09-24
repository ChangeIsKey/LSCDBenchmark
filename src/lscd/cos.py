from typing import Callable
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm

from src.lscd.model import Model
from src.target import Target
from src.use import Use
from src.wic import ContextualEmbedder


class Cos(Model):
    wic: ContextualEmbedder
    threshold_fn: Callable[[list[float]], float] | None

    def predict(self, targets: list[Target]) -> tuple[list[str], list[float | int]]:
        predictions = {}
        for target in tqdm(targets):
            earlier_df = target.uses_df[target.uses_df.grouping == target.groupings[0]]
            later_df = target.uses_df[target.uses_df.grouping == target.groupings[1]]

            earlier = [Use.from_series(s) for _, s in earlier_df.iterrows()]
            later = [Use.from_series(s) for _, s in later_df.iterrows()]
            earlier_vectors = np.vstack([self.wic.encode(use) for use in earlier])
            later_vectors = np.vstack([self.wic.encode(use) for use in later])

            earlier_avg = earlier_vectors.mean(axis=0)
            later_avg = later_vectors.mean(axis=0)

            predictions[target.name] = -distance.cosine(earlier_avg, later_avg)

        if self.threshold_fn is not None:
            values = list(predictions.values())
            threshold = self.threshold_fn(values)
            predictions = {
                target_name: int(p >= threshold)
                for target_name, p in predictions.items()
            }

        return list(predictions.keys()), list(predictions.values())
