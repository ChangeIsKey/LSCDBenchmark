from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Literal,
    TypeAlias,
    TypeVar,
    TypedDict,
)

import numpy as np
import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel, PrivateAttr 

from src.utils import utils
from src.utils.utils import CsvParams
from src.use import UseID
from src.plots import Plotter

EvaluationTask: TypeAlias = Literal[
    "wic", "binary_wic", "change_graded", "change_binary", "COMPARE", "wsi"
]
K = TypeVar("K", str, tuple[str, str])
V = TypeVar("V", int, float)


class DatasetMetadata(TypedDict):
    name: str
    version: str


class Evaluation(BaseModel, ABC):
    task: EvaluationTask 
    metric: Callable[[list[float | int], list[float | int]], Any] 
    plotter: Plotter | None
    write: bool


    def __call__(self, predictions: dict[K, V], labels: dict[K, V]) -> int | float:
        if self.write and self.plotter is not None:
            self.plotter(predictions=predictions, labels=labels)

        results = self.combine_inputs(labels=labels, predictions=predictions)
        if self.write:
            results.to_csv("predictions.csv", sep="\t")
        results = results.dropna(how="any")

        y_true = results.label.tolist()
        y_pred = results.prediction.tolist()
        score = self.metric(y_true, y_pred)

        if self.write:
            with open(file="score.txt", mode="w", encoding="utf8") as f:
                f.write(str(score))

        return score

    @staticmethod
    def combine_inputs(labels: dict[K, V], predictions: dict[K, V]) -> DataFrame:
        labels_df = DataFrame(
            {"target": list(labels.keys()), "label": list(labels.values())}
        )
        predictions_df = DataFrame(
            {
                "target": list(predictions.keys()),
                "prediction": list(predictions.values()),
            }
        )
        merged = pd.merge(
            left=labels_df,
            right=predictions_df,
            how="outer",
            on="target",
            validate="one_to_one",
        )

        return merged