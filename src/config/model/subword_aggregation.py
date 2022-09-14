from enum import Enum
import numpy as np

class SubwordAggregator(str, Enum):
    AVERAGE = "average"
    FIRST = "first"
    LAST = "last"
    SUM = "sum"

    def __call__(self, vectors: np.ndarray) -> np.ndarray:
        match self:
            case self.AVERAGE:
                return np.mean(vectors, axis=0, keepdims=True)
            case self.SUM:
                return np.sum(vectors, axis=0, keepdims=True)
            case self.FIRST:
                return vectors[0]
            case self.LAST:
                return vectors[-1]



