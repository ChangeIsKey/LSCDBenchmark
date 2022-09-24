from abc import ABC, abstractmethod
from collections import Counter

from pydantic import BaseModel
from src.target import Target

from src.use import UseID
from src import wic


class Model(BaseModel, ABC):
    wic: wic.Model
    @abstractmethod
    def predict(self, target: Target) -> dict[UseID, int]:
        ...

    def split_clusters(
        self, clusters: dict[UseID, int], grouping_to_use: dict[str, list[UseID]]
    ) -> tuple[list[int], list[int]]:

        groupings = list(grouping_to_use.keys())
        groups = (
            [clusters[use] for use in grouping_to_use[groupings[0]]],
            [clusters[use] for use in grouping_to_use[groupings[1]]]
        )

        return groups
        
    @staticmethod
    def normalize_cluster(cluster: list[int]) -> list[float]:
        normalized = []
        counts = Counter(cluster)
        n = len(cluster)
        for elem in cluster:
            normalized.append(counts[elem] / n)

        return normalized
