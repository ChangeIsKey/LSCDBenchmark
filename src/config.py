from __future__ import annotations

import importlib.util
import sys
from enum import Enum, unique
from itertools import product
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple,
                    Union)

import numpy as np
import pandas as pd
from pandas import Series
from pydantic import BaseModel
from pydantic.dataclasses import dataclass

import src.utils as utils
from src.preprocessing import normalize_spaces

if TYPE_CHECKING:
    from src.lscd import Target
    from src.distance_model import DistanceModel

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
    """
    Class representing the possible types of use pairs
    """

    COMPARE = "COMPARE"
    EARLIER = "EARLIER"
    LATER = "LATER"

    def __call__(self, target: Target, sampling: sampling) -> Tuple[List[ID], List[ID]]:
        """
        Retrieves two lists of use IDs for different types of pairings.
        In general, you don't need to call this function manually.
        By calling sampling.__call__ and passing a pairing as argument, this
        function will be automatically called
        """
        if sampling is sampling.annotated:
            judgments = pd.merge(
                target.judgments,
                target.uses,
                left_on="identifier1",
                right_on="identifier",
                how="left",
            )
            judgments = pd.merge(
                judgments,
                target.uses,
                left_on="identifier2",
                right_on="identifier",
                how="left",
            )

            pairing_to_grouping = {
                "COMPARE": target.grouping_combination,
                "LATER": (
                    target.grouping_combination[1],
                    target.grouping_combination[1],
                ),
                "EARLIER": (
                    target.grouping_combination[0],
                    target.grouping_combination[0],
                ),
            }

            conditions = [
                f"grouping_x == {pairing_to_grouping[self.name][0]}",
                f"grouping_y == {pairing_to_grouping[self.name][1]}",
            ]

            judgments = judgments.query("&".join(conditions))
            return (
                judgments.identifier1.tolist(),
                judgments.identifier2.tolist(),
            )

        else:
            if self is self.COMPARE:
                ids_1 = target.uses[target.uses.grouping == target.grouping_combination[0]].identifier.tolist()
                ids_2 = target.uses[target.uses.grouping == target.grouping_combination[1]].identifier.tolist()
                return ids_1, ids_2
            elif self is self.EARLIER:
                ids = target.uses[target.uses.grouping == target.grouping_combination[0]].identifier.tolist()
                return ids, ids
            elif self is self.LATER:
                ids = target.uses[target.uses.grouping == target.grouping_combination[1]].identifier.tolist()
                return ids, ids


@unique
class sampling(str, Enum):
    """Class representing the possible types of sampling strategies
    annotated: retrieves only use pairs that have been manually annotated before
    sampled: randomly sample use pairs
    all: cartesian product of all use pairs
    """

    annotated = "annotated"
    sampled = "sampled"
    all = "all"

    def __call__(
        self, pairing: pairing, target: Target, **kwargs
    ) -> List[Tuple[ID, ID]]:
        """
        Retrieve use pairs following the specified strategy
        """
        if self is self.annotated:
            return self.__annotated(target, pairing)
        elif self is self.all:
            return self.__all(target, pairing)
        elif self is self.sampled:
            return self.__sampled(target, pairing, **kwargs)

    def __annotated(self, target: Target, pairing: pairing) -> List[Tuple[ID, ID]]:
        ids1, ids2 = pairing(target, self)
        return list(zip(ids1, ids2))

    def __all(self, target: Target, pairing: pairing) -> List[Tuple[ID, ID]]:
        ids1, ids2 = pairing(target, self)
        return list(product(ids1, ids2))

    def __sampled(
        self, target: Target, pairing: pairing, n: int = 100, replace: bool = True
    ) -> List[Tuple[ID, ID]]:
        ids_1, ids_2 = pairing(target, self)
        pairs = []
        for _ in range(n):
            pairs.append(
                (
                    np.random.choice(ids_1, replace=replace),
                    np.random.choice(ids_2, replace=replace),
                )
            )
        return pairs

        
def load_method(module: Path, method: Optional[str], default: Callable) -> Tuple[Callable, str]:
        module = utils.path(module)
        spec = importlib.util.spec_from_file_location(
            name=module.stem,
            location=module,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        method_name = str(method)
        method = (
            default
            if method is None
            else getattr(module, method)
        )
        return method, method_name
    


@dataclass
class Preprocessing:
    module: Optional[Union[str, Path, Any]]
    method: Optional[Union[str, Callable]]
    params: Dict[str, Any]

    @staticmethod
    def __keep_intact(s: Series) -> Tuple[str, int, int]:
        start, end = tuple(map(int, s.indexes_target_token.split(":")))
        return normalize_spaces(s.context), start, end

    def __post_init_post_parse__(self):
        self.module = utils.path(self.module)
        how, self.method_name = load_method(self.module, self.method, default=self.__keep_intact)

        def func(s: Series, **kwargs) -> Series:
            context, start, end = how(s, **kwargs)
            return Series(
                {
                    "context_preprocessed": normalize_spaces(context),
                    "target_index_begin": start,
                    "target_index_end": end,
                }
            )

        self.method = func


@dataclass
class Measure:
    module: Path
    method: str
    sampling_params: Dict[str, Any]
    method_params: Dict[str, Any]

    def __post_init_post_parse__(self):
        self.module = utils.path(self.module)
        self.__method, self.method = load_method(self.module, self.method, default=None)

    def __call__(self, target: Target, model: DistanceModel, **kwargs):
        return self.__method(target, model, **kwargs)
    

class ThresholdParam(str, Enum):
    ABOVE = "above"
    BELOW = "below"


@dataclass
class CleaningParam:
    threshold: float
    keep: ThresholdParam = ThresholdParam.ABOVE


@unique
class BooleanMethod(str, Enum):
    ALL = "all"
    ANY = "any"


@dataclass
class Cleaning:
    stats: Dict[str, CleaningParam]
    method: BooleanMethod = BooleanMethod.ALL


class Task(str, Enum):
    LSCD = "lscd"
    CLUSTER = "cluster"
    SEMANTIC_PROXIMITY = "semantic_proximity"


class DatasetConfig(BaseModel):
    name: str
    groupings: Tuple[int, int]
    task: Task
    preprocessing: Preprocessing
    cleaning: Cleaning


@unique
class SubwordAggregator(str, Enum):
    AVERAGE = "average"
    FIRST = "first"
    LAST = "last"
    SUM = "sum"

    def __call__(self, vectors: np.array) -> np.array:
        if self is self.AVERAGE:
            return np.mean(vectors, axis=0, keepdims=True)
        elif self is self.FIRST:
            return vectors[0]
        elif self is self.LAST:
            return vectors[-1]
        elif self is self.SUM:
            return np.sum(vectors, axis=0, keepdims=True)


class LayerAggregator(str, Enum):
    AVERAGE = "average"
    CONCAT = "concat"
    SUM = "sum"

    def __call__(self, layers: np.array) -> np.ndarray:
        if self is self.AVERAGE:
            return np.mean(layers, axis=0)
        elif self is self.SUM:
            return np.sum(layers, axis=0)
        elif self is self.CONCAT:
            return np.ravel(layers)


class ModelConfig(BaseModel):
    gpu: Optional[int]
    name: str
    layers: List[int]
    layer_aggregation: LayerAggregator
    measure: Measure
    subword_aggregation: SubwordAggregator

class EvaluationTask(str, Enum):
    GRADED_CHANGE = "change_graded"
    BINARY_CHANGE = "change_binary"
    
    
@dataclass
class Threshold:    
    module: Union[str, Path]
    method: Optional[Union[str, Callable]]
    params: Dict[str, Any]

    def __post_init_post_parse__(self):
        self.module = utils.path(self.module)
        self.__method, self.method = load_method(self.module, self.method, default=None)
    
    def __call__(self, **kwargs):
        return self.__method(**kwargs)
    
class EvaluationConfig(BaseModel):
    task: EvaluationTask
    binary_threshold: Threshold


class Config(BaseModel):
    dataset: DatasetConfig
    model: ModelConfig
    evaluation: EvaluationConfig
