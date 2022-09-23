from dataclasses import dataclass
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm
from typing import Callable

from src.lscd.lscd_model import LSCDModel
from src.target import Target
from src.use import Use
from src.wic.bert import ContextualEmbedderWIC



@dataclass
class Cos(LSCDModel):
    wic: ContextualEmbedderWIC
    threshold_fn: Callable[[list[float]], list[int]] | None

    def predict(self, targets: list[Target]) -> tuple[list[str], list[float | int]]:
        predictions = {}
        for target in targets:
            earlier = (
                target.uses[target.uses.grouping == target.groupings[0]]
                .apply(Use.from_series, axis=1)
                .tolist()
            )
            later = (
                target.uses[target.uses.grouping == target.groupings[1]]
                .apply(Use.from_series, axis=1)
                .tolist()
            )

            earlier_vectors = np.vstack([self.wic.encode(use) for use in earlier])
            later_vectors = np.vstack([self.wic.encode(use) for use in later])

            earlier_avg = earlier_vectors.mean(axis=0)
            later_avg = later_vectors.mean(axis=0)

            predictions[target.name] = -distance.cosine(earlier_avg, later_avg)

        # if self.threshold_fn is not None:
        #     threshold = self.threshold_fn(predictions)
        #     predictions = {target_name: int(p >= threshold) for target_name, p in predictions.items()}


        return list(predictions.keys()), list(predictions.values())
