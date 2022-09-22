from abc import ABC, abstractmethod
from src.config.model.config import ContextualEmbedderConfig, DeepMistakeConfig
from src.target import Target
from src.wic.deepmistake import DeepMistakeWIC
from src.wic.bert import ContextualEmbedderWIC


class LSCDModel(ABC):
    _target_: str
    wic: ContextualEmbedderConfig | DeepMistakeConfig | None
    wic_model: ContextualEmbedderWIC | DeepMistakeWIC | None

    @abstractmethod
    def predict(self, targets: list[Target]) -> list[float | int]:
        ...
