from abc import ABC, abstractmethod
from typing import Callable, TYPE_CHECKING
from pandas import DataFrame
import numpy as np
from src.target import Target, Sampling, Pairing


if TYPE_CHECKING:
    from src.config.config import UseID 

class DistanceModel(ABC):
    @abstractmethod
    def distances(
        self,
        target: Target,
        sampling: Sampling,
        pairing: Pairing,
        method: Callable,
        return_pairs: bool,
        **kwargs
    ) -> tuple[list[tuple["UseID", "UseID"]], list[float]] | list[float]:
        pass

    def distance_matrix(self, target: Target) -> DataFrame:
        compare_pairs, compare_distances = self.distances(target=target, sampling=Sampling.all, pairing=Pairing.COMPARE, return_pairs=True)
        later_pairs, later_distances = self.distances(target=target, sampling=Sampling.all, pairing=Pairing.LATER, return_pairs=True)
        earlier_pairs, earlier_distances = self.distances(target=target, sampling=Sampling.all, pairing=Pairing.EARLIER, return_pairs=True)

        pairs_to_distances = dict(zip(
            compare_pairs + later_pairs + earlier_pairs, 
            compare_distances + later_distances + earlier_distances
        ))

        ids = sorted({id_ for l in list(pairs_to_distances.keys()) for id_ in l})
        n_ids = len(ids)

        distance_matrix = np.zeros((n_ids, n_ids))
        for i, id1 in enumerate(ids):
            for j, id2 in enumerate(ids):
                try:
                    distance_matrix[i, j] = pairs_to_distances[id1, id2]
                except KeyError:
                    distance_matrix[i, j] = pairs_to_distances[id2, id1]
        return DataFrame(distance_matrix, index=ids, columns=ids)

