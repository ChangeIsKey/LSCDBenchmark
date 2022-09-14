import json
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, List,
                    Optional, Tuple, TypeAlias)

from pydantic import BaseModel, Field
from src.config.dataset.config import DatasetConfig
from src.config.evaluation.config import EvaluationConfig
from src.config.model.config import ModelConfig

UseID: TypeAlias = str
Grouping: TypeAlias = str


class Config(BaseModel):
    model: ModelConfig
    dataset: DatasetConfig
    evaluation: EvaluationConfig
    gpu: Optional[int] = Field(exclude=True)

    def yaml(self) -> str:
        as_dict = self.json(exclude={"model": {"measure": {"func", "default"}}, "dataset": {"preprocessing": {"func", "default"}}})
        print(as_dict) 
        