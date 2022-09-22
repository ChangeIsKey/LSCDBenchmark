from typing import TYPE_CHECKING, Any, Optional

import torch
from hydra import utils
from pydantic import BaseModel, Field
from src.config.evaluation.config import Method
from src.config.model.layer_aggregation import LayerAggregator
from src.config.model.subword_aggregation import SubwordAggregator
from src.wic.bert import ContextualEmbedderWIC
from src.wic.deepmistake import DeepMistakeWIC

if TYPE_CHECKING:
    from src.config.config import Config


class ContextualEmbedderConfig(BaseModel):
    id: str
    layers: list[int] | torch.Tensor
    layer_aggregation: LayerAggregator
    subword_aggregation: SubwordAggregator
    truncation_tokens_before_target: float
    distance_metric: Method

    class Config:
        arbitrary_types_allowed: bool = True


class DeepMistakeConfig(BaseModel):
    id: str = "deepmistake"


class ModelConfig(BaseModel):
    target: str = Field(alias="_target_")
    wic: Optional[ContextualEmbedderConfig] = Field(default=None)
    threshold_fn: Optional[Method] = Field(default=None)

    def instantiate(self, config: "Config") -> Any:
        model = utils.instantiate(self.dict(by_alias=True, exclude={"wic"}))
        model.config = config
        model.wic = self.wic

        if self.wic is not None:
            model.wic = self.wic
            model.wic_model = (
                ContextualEmbedderWIC(config)
                if isinstance(model.wic, ContextualEmbedderConfig)
                else DeepMistakeWIC()
            )
            if isinstance(model.wic_model, ContextualEmbedderWIC):
                model.wic.layers = torch.tensor(model.wic.layers, dtype=torch.int32).to(
                    model.wic_model.device
                )

        return model
