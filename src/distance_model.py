from abc import ABC, abstractmethod
from src.config import ID
from typing import Callable, List, Tuple


class DistanceModel(ABC):
    @abstractmethod
    def distances(ids: List[Tuple[ID, ID]], distance_measure: Callable, **kwargs):
        pass