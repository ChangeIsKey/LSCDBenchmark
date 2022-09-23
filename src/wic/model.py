from pydantic import BaseModel
from abc import ABC, abstractmethod

from src.use import Use


class WICModel(BaseModel, ABC):
    @abstractmethod
    def predict(self, use_pairs):
        raise NotImplementedError
