from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from src import wic
from src.target import Target
from src.use import UseID


class Model(BaseModel, ABC):
    wic: wic.Model

    class Config:
        arbitrary_types_allowed = True

    def predict(self, targets: list[Target]) -> dict[UseID, int]:
        preds = {}
        for target in targets:
            preds.update(self.predict_target(target))
        return preds

    @abstractmethod
    def predict_target(self, target: Target) -> dict[UseID, int]:
        ...

    def make_freq_dists(
            self,
            clusters: dict[UseID, int],
            use_to_grouping: dict[UseID, str],
            groupings: tuple[str, str]
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        cluster_to_freq1 = {}
        cluster_to_freq2 = {}
        for use, cluster in clusters.items():
            if not cluster in cluster_to_freq1:
                cluster_to_freq1[cluster] = 0
            if not cluster in cluster_to_freq2:
                cluster_to_freq2[cluster] = 0

            if use_to_grouping[use] == groupings[0]:
                cluster_to_freq1[cluster] += 1
            elif use_to_grouping[use] == groupings[1]:
                cluster_to_freq2[cluster] += 1
            else:
                raise ValueError

        return (
            np.array(list(cluster_to_freq1.values())),
            np.array(list(cluster_to_freq2.values()))
        )

    @staticmethod
    def normalize_cluster(cluster: list[int]) -> list[float]:
        normalized = []
        counts = Counter(cluster)
        n = len(cluster)
        for elem in cluster:
            normalized.append(counts[elem] / n)

        return normalized
