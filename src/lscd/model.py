from pydantic import BaseModel
from abc import ABC, abstractmethod
from src.target import Target


class Model(BaseModel, ABC):
    @abstractmethod
    def predict(self, targets: list[Target]) -> list[float | int]:
        ...
