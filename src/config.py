from typing import Any, Tuple, Optional, Dict, List

from pydantic import conint, BaseModel, validator, root_validator, Field, conlist

import warnings

from enum import Enum

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


class Pairing(str, Enum):
    COMPARE = "COMPARE"
    EARLIER = "EARLIER"
    LATER = "LATER"


class UsesType(str, Enum):
    annotated = "annotated",
    sampled = "sampled",
    all = "all"


class Uses(BaseModel):
    type: UsesType = UsesType.annotated
    pairing: Pairing = Pairing.COMPARE
    params: Optional[SamplingParams] = None

    @root_validator(pre=False)
    def valid_use_type_and_params(cls, values: Dict):
        # allowed_types = ['annotated', 'all', 'sampled']
        # assert (type_ := values.get("type")) in allowed_types, f"value '{type_}' is not one of {allowed_types}"

        type_ = values.get("type")
        if values.get("params") is None and type_ == "sampled":
            raise ValueError("you didn't provide sampling parameters")
        elif values.get("params") and type_ != "sampled":
            warnings.warn("you defined some sampling parameters, but the use type is not 'sampled'")

        return values


class Preprocessing(BaseModel):
    method: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


class DatasetConfig(BaseModel):
    name: str
    language: str
    groupings: Tuple[int, int]
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
    measures: conlist(item_type=str, unique_items=True)
    subword_aggregation: str

    @validator("measures")
    def __measure_supported(cls, measures: str):
        supported_measures = ["apd", "cos"]
        assert all([m in supported_measures for m in measures]), \
            f"one of the measures in '{measures}' is not a one of {supported_measures}"
        return measures

    @validator("subword_aggregation")
    def __subword_aggregation_supported(cls, aggr: str):
        supported_aggregations = ["average", "first"]
        assert aggr in supported_aggregations, f"value '{aggr}' is not one of {[supported_aggregations]}"
        return aggr


class Config(BaseModel):
    dataset: DatasetConfig
    model: ModelConfig
