import numpy as np
from pandas import DataFrame
from pydantic import BaseModel
from abc import ABC, abstractmethod
from src.target import Target

from src.use import Use


class Model(BaseModel, ABC):
    @abstractmethod
    def predict(self, use_pairs: list[tuple[Use, Use]]) -> list[float]:
        raise NotImplementedError

    def distance_matrix(self, target: Target):
        compare_pairs = target.use_pairs(sampling="all", pairing="COMPARE")
        later_pairs = target.use_pairs(sampling="all", pairing="LATER")
        earlier_pairs = target.use_pairs(sampling="all", pairing="EARLIER")

        pairs = compare_pairs + later_pairs + earlier_pairs
        predictions = self.predict(pairs)
        pairs_to_distances = dict(zip(pairs, predictions))

        # get a sorted list of unique uses
        uses = sorted({use for use_pair in list(pairs_to_distances.keys()) for use in use_pair})
        ids = [use.identifier for use in uses]
        n_ids = len(ids)

        distance_matrix = np.zeros((n_ids, n_ids))
        for i, use_1 in enumerate(uses):
            for j, use_2 in enumerate(uses):
                try:
                    distance_matrix[i, j] = pairs_to_distances[(use_1, use_2)]
                except KeyError:
                    distance_matrix[i, j] = pairs_to_distances[(use_2, use_1)]

        return DataFrame(distance_matrix, index=ids, columns=ids)
 