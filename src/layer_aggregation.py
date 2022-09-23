from typing_extensions import Self
import torch
from enum import Enum

class LayerAggregator(str, Enum):
    AVERAGE = "average"
    CONCAT = "concat"
    SUM = "sum"

    def __call__(self, layers: torch.Tensor) -> torch.Tensor:
        match self:
            case self.AVERAGE:
                return torch.mean(layers, dim=0)
            case self.SUM:
                return torch.sum(layers, dim=0)
            case self.CONCAT:
                return torch.ravel(layers)

    @classmethod
    def from_str(cls, s: str) -> Self:
        return cls[s.upper()]