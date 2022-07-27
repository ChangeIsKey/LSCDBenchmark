import importlib.util
import sys
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Tuple, Optional, Dict, List

import torch
from pydantic import conint, BaseModel, validator, root_validator
from pydantic.dataclasses import dataclass, Field

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
    ANNOTATED = "annotated",
    SAMPLED = "sampled",
    ALL = "all"


class Uses(BaseModel):
    type: UsesType = UsesType.ANNOTATED
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


@dataclass
class Preprocessing:
    module: Path = Path("src/preprocessing.py").resolve()
    method: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


    def __post_init_post_parse__(self):
        spec = importlib.util.spec_from_file_location(
            name=self.module.stem,
            location=self.module
        )
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.module 
        spec.loader.exec_module(self.module)


@dataclass
class CleaningParam:
    threshold: float
    above: bool = True

class BooleanMethod(str, Enum):
    ALL = "all"
    ANY = "any"

@dataclass
class Cleaning:
    fields: Dict[str, CleaningParam]
    method: BooleanMethod = BooleanMethod.ALL


class DatasetConfig(BaseModel):
    name: str
    language: str
    groupings: Tuple[int, int]
    uses: Uses
    task: str
    preprocessing: Preprocessing
    cleaning: Cleaning

    @validator("task")
    def task_is_supported(cls, task: str):
        supported_tasks = ["lscd"]
        assert task in supported_tasks, f"value '{task}' is not one of {supported_tasks}"
        return task

    @validator("language")
    def language_is_supported(cls, lang: str):
        assert lang in long2short.keys(), f"value '{lang}' is not one of {list(long2short.keys())}"
        return lang


class SubwordAggregationMethod(str, Enum):
    AVERAGE = "average"
    FIRST = "first"

    def __call__(self, vectors):
        match self.name:
            case self.AVERAGE.name:
                return torch.mean(vectors, dim=0)
            case self.FIRST.name:
                return vectors[0]


class ModelConfig(BaseModel):
    name: str
    layers: Tuple[int, int]
    measures: List[str]
    subword_aggregation: SubwordAggregationMethod

    @validator("measures")
    def _measure_supported(cls, measures: str):
        supported_measures = ["apd", "cos"]
        assert all([m in supported_measures for m in measures]), \
            f"one of the measures in '{measures}' is not a one of {supported_measures}"
        return measures


@dataclass
class ResultsConfig:
    output_directory: Path = Path("results")

    def __post_init__(self):
        self.output_directory.mkdir(exist_ok=True)


class Config(BaseModel):
    dataset: DatasetConfig
    model: ModelConfig
    results: ResultsConfig = Field(default_factory=ResultsConfig)


