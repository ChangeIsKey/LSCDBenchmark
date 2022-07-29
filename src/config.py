from __future__ import annotations

import importlib.util
import sys
import warnings
from enum import Enum, unique
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from pydantic import BaseModel, conint, conlist, root_validator, validator
from pydantic.dataclasses import Field, dataclass
from torch import Tensor

if TYPE_CHECKING:
    from src.lscd import Target

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


ID = str


@unique
class pairing(str, Enum):
    COMPARE = "COMPARE"
    EARLIER = "EARLIER"
    LATER = "LATER"

    def __call__(self, target: Target, sampling: sampling) -> Tuple[List[ID], List[ID]]:
        if sampling is sampling.annotated:
            ids = (
                target.judgments.identifier1.tolist(),
                target.judgments.identifier2.tolist()
            )
        else:
            ids = (
                target.uses_1.identifier.tolist(),
                target.uses_2.identifier.tolist()
            )

        if self is self.COMPARE:
            return ids
        elif self is self.EARLIER:
            return ids[0], ids[0]
        elif self is self.LATER:
            return ids[1], ids[1]

@unique
class sampling(str, Enum):
    annotated = "annotated"
    sampled = "sampled"
    all = "all"

    def __call__(self, pairing: pairing, target: Target, **kwargs) -> List[Tuple[ID, ID]]:
        if self is self.annotated:
            return self.__annotated(target, pairing)
        elif self is self.all:
            return self.__all(target, pairing)
        elif self is self.sampled:
            return self.__sampled(target, pairing, **kwargs)

    def __annotated(self, target: Target, pairing: pairing) -> List[Tuple[ID, ID]]:
        # This simply takes
        return list(zip(pairing(target, self)))

    def __all(self, target: Target, pairing: pairing) -> List[Tuple[ID, ID]]:
        ids_1, ids_2 = pairing(target, self)
        return list(product(ids_1, ids_2))

    def __sampled(self, target: Target, pairing: pairing, n: int = 100, replace: bool = True) -> List[Tuple[ID, ID]]:
        ids_1, ids_2 = pairing(target, self)
        pairs = []
        for _ in range(n):
            pairs.append((
                np.random.choice(ids_1, replace=replace),
                np.random.choice(ids_2, replace=replace)
            ))
        return pairs

    




class Uses(BaseModel):
    type: sampling = sampling.annotated
    pairing: pairing = pairing.COMPARE
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
    method: Union[str, Callable] = None
    params: Dict[str, Any] = Field(default_factory=dict)

    def __post_init_post_parse__(self):
        spec = importlib.util.spec_from_file_location(
            name=self.module.stem,
            location=self.module
        )
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.module 
        spec.loader.exec_module(self.module)
        self.method = getattr(self.module, self.method) if self.method is not None else None


@dataclass
class Measure:
    module: Path = Path("src/measures.py").resolve()
    method: Union[str, Callable] = None
    params: Dict[str, Any] = Field(default_factory=dict)

    def __post_init_post_parse__(self):
        spec = importlib.util.spec_from_file_location(
            name=self.module.stem,
            location=self.module
        )
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.module 
        spec.loader.exec_module(self.module)
        self.method = getattr(self.module, self.method) if self.method is not None else None

    

@dataclass
class CleaningParam:
    threshold: float
    above: bool = True

@unique
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


@unique
class SubwordAggregator(str, Enum):
    AVERAGE = "average"
    FIRST = "first"

    def __call__(self, vectors: Tensor):
        if self is self.AVERAGE:
            return torch.mean(vectors, dim=0)
        elif self is self.FIRST:
            return vectors[0]


class ModelConfig(BaseModel):
    name: str
    layers: List[int]
    measure: Measure
    subword_aggregation: SubwordAggregator

@dataclass
class ResultsConfig:
    output_directory: Path = Path("results")

    def __post_init__(self):
        self.output_directory.mkdir(exist_ok=True)


class Config(BaseModel):
    dataset: DatasetConfig
    model: ModelConfig
    results: ResultsConfig = Field(default_factory=ResultsConfig)

