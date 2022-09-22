from abc import ABC, abstractmethod

from src.use import Use


class WICModel(ABC):
    @abstractmethod
    def predict(self, use_pairs: list[tuple[Use, Use]]) -> list[float]:
        pass
