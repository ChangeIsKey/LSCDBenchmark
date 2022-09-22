import json
from typing import Optional

import yaml
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

from src.config.dataset.config import DatasetConfig
from src.config.evaluation.config import EvaluationConfig
from src.config.model.config import ModelConfig


class Config(BaseModel):
    model: ModelConfig
    dataset: DatasetConfig
    evaluation: EvaluationConfig
    gpu: Optional[int]

    @classmethod
    def from_dictconfig(cls, cfg: DictConfig):
        obj = OmegaConf.to_object(cfg)
        return cls(**obj)

    def yaml(self) -> str:
        return yaml.dump(json.loads(self.json(by_alias=True)))
        