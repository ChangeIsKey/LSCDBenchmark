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
        self.config = config
        self.vectorizer = self.get_vectorizer()
        # TODO check if model can just take the identifiers (optionally)
        self.id1_to_row = {use.identifier: i for i, use in enumerate(uses_1)}
        self.id2_to_row = {use.identifier: i for i, use in enumerate(uses_2)}
        self.vectors_1 = self.vectorizer.vectorize(
            contexts=[use.context_preprocessed for use in uses_1],
            target_indices=[(use.target_index_begin, use.target_index_end) for use in uses_1]
        )
        self.vectors_2 = self.vectorizer.vectorize(
            contexts=[use.context_preprocessed for use in uses_2],
            target_indices=[(use.target_index_begin, use.target_index_end) for use in uses_2]
        )
        pass

    def apd(self, pairs: List[Tuple[str, str]], distance_measure=distance.cosine):
        return np.mean([distance_measure(self.vectors_1[self.id1_to_row[id1]], self.vectors_2[self.id2_to_row[id2]])
                        for id1, id2 in pairs])

    def cosine(self):
        v1 = torch.mean(self.vectors_1, dim=0, keepdim=True)
        v2 = torch.mean(self.vectors_2, dim=0, keepdim=True)
        return distance.cosine(v1, v2)

    def get_vectorizer(self) -> Vectorizer:
        if self.config.name == "bert":
            return Bert(self.config)
        elif self.config.name == "xlmr":
            return XLMR(self.config)