from enum import Enum
import torch


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
            case _:
                raise ValueError
