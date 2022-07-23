from typing import Any, Tuple, Optional, Dict

from pydantic import conint, BaseModel, validator, root_validator, Field

import warnings

long2short = dict(
    english="en",
    en="en",
    spanish="es",
    es="es",
    swedish="sv",
    sv="sv",
    german="de",
    de="de",
)


class SamplingParams(BaseModel):
    n: conint(ge=0)
    replacement: bool


class Uses(BaseModel):
    type: str = "annotated"
    pairing: str = "COMPARE"
    params: Optional[SamplingParams] = None

    @root_validator()
    def valid_use_type_and_params(cls, values: Dict):
        allowed_types = ['annotated', 'all', 'sampled']
        assert (type_ := values.get("type")) in allowed_types, f"value '{type_}' is not one of {allowed_types}"

        if values.get("params") is None and type_ == "sampled":
            raise ValueError("you didn't provide sampling parameters")
        elif values.get("params") and type_ != "sampled":
            warnings.warn("you defined some sampling parameters, but the use type is not 'sampled'")

        return values

    @validator("pairing")
    def pairing_is_supported(cls, pairing: str):
        supported_pairings = ["COMPARE", "EARLIER", "LATER"]
        assert pairing in supported_pairings, f"value '{pairing}' not in {supported_pairings}"
        return pairing


class Preprocessing(BaseModel):
    method: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


class DatasetConfig(BaseModel):
    name: str
    language: str
    grouping_1: int
    grouping_2: int
    uses: Uses
    task: str
    preprocessing: Preprocessing

    @validator("task")
    def task_is_supported(cls, task: str):
        supported_tasks = ["lscd"]
        assert task in supported_tasks, f"value '{task}' is not one of {supported_tasks}"

    @validator("language")
    def language_is_supported(cls, lang: str):
        assert lang in long2short.keys(), f"value '{lang}' is not one of {list(long2short.keys())}"
        return lang


class ModelConfig(BaseModel):
    name: str
    model: str
    layers: Tuple[int, int]
    measure: str = "apd"
    subword_aggregation: str = "average"

    @validator("measure")
    def measure_supported(cls, measure: str):
        supported_measures = ["apd", "cos"]
        assert measure in supported_measures, f"value '{measure}' is not one of {[supported_measures]}"
        return measure

    @validator("subword_aggregation")
    def subword_aggregation_supported(cls, aggr: str):
        supported_aggregations = ["average", "first"]
        assert aggr in supported_aggregations, f"value '{aggr}' is not one of {[supported_aggregations]}"
        return aggr


class Config(BaseModel):
    dataset: DatasetConfig
    model: ModelConfig
