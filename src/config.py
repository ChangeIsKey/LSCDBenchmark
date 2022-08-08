from __future__ import annotations

import importlib.util
import sys
import warnings
import pandas as pd
from pandas import Series
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


ID = str


@unique
class pairing(str, Enum):
    COMPARE = "COMPARE"
    EARLIER = "EARLIER"
    LATER = "LATER"
    MERGE = "MERGE"

    def __call__(self, target: Target, sampling: sampling) -> Tuple[List[ID], List[ID]]:
        if sampling is sampling.annotated:
            judgments = pd.merge(target.judgments, target.uses_1, left_on="identifier1", right_on="identifier", how="left")
            judgments = pd.merge(judgments, target.uses_2, left_on="identifier2", right_on="identifier", how="left")
            # judgments = judgments[(judgments.grouping_x == target.grouping_combination[0]) & (judgments.grouping_y == target.grouping_combination[1])].copy()
            # judgments.to_csv("test.csv")
            ids = (
                judgments.identifier1.tolist(),
                judgments.identifier2.tolist()
            )
        else:
            ids1 = target.uses_1.identifier.tolist(),
            ids2 = target.uses_2.identifier.tolist(),

        if self is self.COMPARE:
            return ids1, ids2
        elif self is self.EARLIER:
            return ids1, ids1
        elif self is self.LATER:
            return ids2, ids2
        elif self is self.MERGE:
            return ids1 + ids1 + ids2, ids2 + ids1 + ids2


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
        return list(zip(*pairing(target, self)))

    def __all(self, target: Target, pairing: pairing) -> List[Tuple[ID, ID]]:
        return list(product(*pairing(target, self)))

    def __sampled(self, target: Target, pairing: pairing, n: int = 100, replace: bool = True) -> List[Tuple[ID, ID]]:
        ids_1, ids_2 = pairing(target, self)
        pairs = []
        for _ in range(n):
            pairs.append((
                np.random.choice(ids_1, replace=replace),
                np.random.choice(ids_2, replace=replace)
            ))
        return pairs

    
@dataclass
class Preprocessing:
    module: Path
    method: Optional[Union[str, Callable]]
    params: Dict[str, Any]

    @staticmethod
    def keep_intact(s: Series, **kwargs) -> Tuple[str, int, int]:
        start, end = tuple(map(int, s.indexes_target_token.split(":")))
        return s.context, start, end

    def __post_init_post_parse__(self):
        self.module = self.module.resolve()
        spec = importlib.util.spec_from_file_location(
            name=self.module.stem,
            location=self.module
        )
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.module 
        spec.loader.exec_module(self.module)

        how = self.keep_intact if self.method is None else getattr(self.module, self.method)
        self.method_name = None if self.method is None else self.method

        def func(s: Series, **kwargs) -> Series:
            context, start, end = how(s, **kwargs)
            return Series({
                "context_preprocessed": context,
                "begin_index_token_preprocessed": start,
                "end_index_token_preprocessed": end,
            })

        self.method = func


@dataclass
class Measure:
    module: Path
    method: Union[str, Callable]
    params: Dict[str, Any]

    def __post_init_post_parse__(self):
        self.module = self.module.resolve()
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
    groupings: Tuple[int, int]
    task: str
    preprocessing: Preprocessing
    cleaning: Cleaning

    @validator("task")
    def task_is_supported(cls, task: str):
        supported_tasks = ["lscd"]
        assert task in supported_tasks, f"value '{task}' is not one of {supported_tasks}"
        return task

@unique
class SubwordAggregator(str, Enum):
    AVERAGE = "average"
    FIRST = "first"
    LAST = "last"
    SUM = "sum"

    def __call__(self, vectors: np.array) -> np.array:
        if self is self.AVERAGE:
            return vectors.mean(axis=0)
        elif self is self.FIRST:
            return vectors[0]
        elif self is self.LAST:
            return vectors[-1]
        elif self is self.SUM:
            return vectors.sum(axis=0, keepdim=True)


class LayerAggregator(str, Enum):
    AVERAGE = "average"
    CONCAT = "concat"
    SUM = "sum"

    def __call__(self, layers: np.array) -> np.array:
        if self is self.AVERAGE:
            return layers.mean(axis=0)
        elif self is self.SUM:
            return layers.sum(axis=0, keepdim=True)
        elif self is self.CONCAT:
            dim = np.product(list(layers.shape))
            return layers.reshape((dim, 1))


class ModelConfig(BaseModel):
    gpu: Optional[int]
    name: str
    layers: List[int]
    layer_aggregation: LayerAggregator
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

