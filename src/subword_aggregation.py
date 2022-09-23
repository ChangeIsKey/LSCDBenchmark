from enum import Enum
from typing_extensions import Self
import torch

class SubwordAggregator(str, Enum):
    AVERAGE = "average"
    FIRST = "first"
    LAST = "last"
    SUM = "sum"

    def __call__(self, vectors: torch.Tensor) -> torch.Tensor:
        match self:
            case self.AVERAGE:
                return torch.mean(vectors, dim=0, keepdim=True)
            case self.SUM:
                return torch.sum(vectors, dim=0, keepdim=True)
            case self.FIRST:
                return vectors[0]
            case self.LAST:
                return vectors[-1]

    @classmethod
    def from_str(cls, s: str) -> Self:
        return cls[s.upper()]