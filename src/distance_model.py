from abc import ABC, abstractmethod
from src.config import ID, pairing, sampling
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
    ) -> Union[Tuple[List[Tuple[ID, ID]], List[float]], List[float]]:
        pass
