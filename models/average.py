from typing import Tuple, List, Dict

from config import ModelConfig
import torch
import numpy as np
import scipy.spatial.distance as distance
from torch import Tensor

BERT = None


class LSCDModel:
    def __init__(self, vectors_1: List[Tensor], vectors_2: List[Tensor], config: ModelConfig, id1_to_row: Dict[str, int], id2_to_row: Dict[str, int]):
        self.config = config
        self.id1_to_row = id1_to_row
        self.id2_to_row = id2_to_row
        self.vectors_1 = vectors_1
        self.vectors_2 = vectors_2

    def apd(self, pairs: List[Tuple[str, str]], distance_measure=distance.cosine):
        return np.mean([distance_measure(self.vectors_1[self.id1_to_row[id1]], self.vectors_2[self.id2_to_row[id2]])
                        for id1, id2 in pairs])

    def cos(self):
        pass
