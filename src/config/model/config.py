from pydantic import BaseModel
from src.config.model.layer_aggregation import LayerAggregator
from src.config.model.subword_aggregation import SubwordAggregator
from src.config.model.truncation import Truncation
from src.config.model.measure import Measure

class ModelConfig(BaseModel):
    name: str
    layers: list[int]
    layer_aggregation: LayerAggregator
    subword_aggregation: SubwordAggregator
    truncation: Truncation
    measure: Measure
