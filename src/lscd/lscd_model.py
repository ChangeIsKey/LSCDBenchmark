from abc import ABC, abstractmethod
from src.target import Target


class LSCDModel(ABC):
    @abstractmethod
    def predict(self, targets: list[Target]) -> list[float | int]:
        ...
