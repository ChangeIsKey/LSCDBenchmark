import torch

from typing import List, Dict, Tuple

import numpy as np
from pandas import DataFrame
from scipy.spatial import distance
from torch import Tensor

from src.config import ModelConfig
from src.use import Use
from src.vectorizer import Vectorizer, Bert, XLMR


class VectorModel:
    def __init__(self, config: ModelConfig, uses_1: List[Use], uses_2: List[Use]):
        # TODO check if model can just take the identifiers (optionally)
        self.config = config
        self.uses = (uses_1, uses_2)
        self._vectorizer = None
        self._vectors = None
        self._id_to_row = None

    @property
    def id_to_row(self):
        if self._id_to_row is None:
            self._id_to_row = {k: v for d in [{use.identifier: i for i, use in enumerate(self.uses[j])} for j in range(2)]
                               for k, v in d.items()}
        return self._id_to_row

    @property
    def vectors(self):
        if self._vectors is None:
            self._vectors = tuple([
                self.vectorizer(
                    contexts=[u.context_preprocessed for u in self.uses[i]],
                    target_indices=[(u.target_index_begin, u.target_index_end) for u in self.uses[i]]
                )
                for i in range(2)
            ])
        return self._vectors

    @property
    def vectorizer(self) -> Vectorizer:
        if self._vectorizer is None:
            if self.config.name == "bert":
                self._vectorizer = Bert(self.config)
            elif self.config.name == "xlmr":
                self._vectorizer = XLMR(self.config)
        return self._vectorizer

    def apd(self, pairs: List[Tuple[str, str]], distance_measure=distance.cosine):
        return np.mean([
            distance_measure(self.vectors[0][self.id_to_row[id1]], self.vectors[1][self.id_to_row[id2]])
            for id1, id2 in pairs
        ])

    def cosine(self):
        return distance.cosine(*[torch.mean(v, dim=0, keepdim=True) for v in self.vectors])
