from abc import ABC, abstractmethod
from src.config import UseID, pairing, sampling
from typing import Callable, List, Tuple, Union

from src.lscd.target import Target


class DistanceModel(ABC):
    @abstractmethod
    def distances(
        self,
        target: Target,
        sampling: sampling,
        pairing: pairing,
        method: Callable,
        return_pairs: bool,
        **kwargs
    ) -> Union[Tuple[List[Tuple[UseID, UseID]], List[float]], List[float]]:
        pass
