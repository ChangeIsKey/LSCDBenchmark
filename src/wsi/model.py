from abc import (
    ABC,
    abstractmethod,
)
from collections import Counter

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from src.use import UseID, Use
from src import wic


class WSIModel(BaseModel, ABC):
    wic: wic.WICModel

    def similarity_matrix(
        self, use_pairs: list[tuple[Use, Use]]
    ) -> npt.NDArray[np.float64]:
        predictions = self.wic.predict(use_pairs)
        pairs_to_similarities = dict(zip(use_pairs, predictions))

        # get a sorted list of unique uses
        uses = sorted(
            {use for use_pair in list(pairs_to_similarities.keys()) for use in use_pair}
        )
        ids = [use.identifier for use in uses]
        n_ids = len(ids)

        similarity_matrix = np.zeros((n_ids, n_ids))
        for i, use_1 in enumerate(uses):
            for j, use_2 in enumerate(uses):
                try:
                    similarity_matrix[i, j] = pairs_to_similarities[(use_1, use_2)]
                except KeyError:
                    similarity_matrix[i, j] = pairs_to_similarities[(use_2, use_1)]

        return similarity_matrix

    @abstractmethod
    def predict(self, uses: list[Use]) -> list[int]:
        ...

    def make_freq_dists(
        self,
        clusters: dict[UseID, int],
        use_to_grouping: dict[UseID, str],
        groupings: tuple[str, str],
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        cluster_to_freq1 = {}
        cluster_to_freq2 = {}
        for use, cluster in clusters.items():
            if cluster not in cluster_to_freq1:
                cluster_to_freq1[cluster] = 0
            if cluster not in cluster_to_freq2:
                cluster_to_freq2[cluster] = 0

            if use_to_grouping[use] == groupings[0]:
                cluster_to_freq1[cluster] += 1
            elif use_to_grouping[use] == groupings[1]:
                cluster_to_freq2[cluster] += 1
            else:
                raise ValueError

        return np.array(list(cluster_to_freq1.values())), np.array(
            list(cluster_to_freq2.values())
        )

    @staticmethod
    def normalize_cluster(cluster: list[int]) -> list[float]:
        normalized = []
        counts = Counter(cluster)
        n = len(cluster)
        for elem in cluster:
            normalized.append(counts[elem] / n)

        return normalized
