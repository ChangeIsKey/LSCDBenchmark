from dataclasses import dataclass
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm

from src.lscd.lscd_model import LSCDModel
from src.target import Target
from src.use import Use
from src.wic.bert import ContextualEmbedderWIC
from src.threshold import ThresholdFn



@dataclass
class Cos(LSCDModel):
    wic: ContextualEmbedderWIC
    threshold_fn: ThresholdFn | None

    def predict(self, targets: list[Target]) -> list[float | int]:
        predictions = []
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

            predictions.append(-distance.cosine(earlier_avg, later_avg))

        if self.threshold_fn is not None:
            threshold = self.threshold_fn(predictions)
            predictions = [int(p >= threshold) for p in predictions]


        return predictions