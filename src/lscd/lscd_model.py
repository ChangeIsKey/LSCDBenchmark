from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import Any, Callable
from src.target import Target
from src.wic import ContextualEmbedder, DeepMistake
from src.wic import WICModel


class LSCDModel(BaseModel, ABC):
    wic: WICModel
    threshold_fn: Callable[[list[float]], float] | None

    @abstractmethod
    def predict(self, targets: list[Target]) -> tuple[list[str], list[float | int]]:
        ...
