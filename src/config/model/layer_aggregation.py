import numpy as np
from enum import Enum

class LayerAggregator(str, Enum):
    AVERAGE = "average"
    CONCAT = "concat"
    SUM = "sum"

    def __call__(self, layers: np.ndarray) -> np.ndarray:
        match self:
            case self.AVERAGE:
                return np.mean(layers, axis=0)
            case self.SUM:
                return np.sum(layers, axis=0)
            case self.CONCAT:
                return np.ravel(layers)


