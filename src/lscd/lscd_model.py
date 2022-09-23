from abc import ABC, abstractmethod
from typing import Any
from src.target import Target


class LSCDModel(ABC):
    @abstractmethod
    def predict(self, targets: list[Target]) -> tuple[list[str], list[float | int]]:
        ...
