from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import Any, Callable
from src.target import Target
from src.wic.bert import ContextualEmbedderWIC
from src.wic.deepmistake import DeepMistakeWIC


class LSCDModel(BaseModel, ABC):
    wic: ContextualEmbedderWIC | DeepMistakeWIC
    threshold_fn: Callable[[list[float]], float] | None

    @abstractmethod
    def predict(self, targets: list[Target]) -> tuple[list[str], list[float | int]]:
        ...
