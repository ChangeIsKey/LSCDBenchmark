from abc import ABC
from typing import (
    Any,
    Callable,
    Literal,
    TypeAlias,
    TypeVar,
)

import numpy as np
import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel

from src.plots import Plotter

EvaluationTask: TypeAlias = Literal[
    "semantic_proximity", "change_graded", "change_binary", "COMPARE", "wsi"
]
K = TypeVar("K", str, tuple[str, str])
V = TypeVar("V", int, float)


class Evaluation(BaseModel, ABC):
    task: EvaluationTask | None
    metric: Callable[[list[float | int], list[float | int]], Any] | None
    plotter: Plotter

    def preprocess_results(self, results: DataFrame) -> DataFrame:
        return results

    def __call__(self, predictions: dict[K, V], labels: dict[K, V], write: bool) -> int | float:
        results = self.combine_inputs(labels=labels, predictions=predictions)
        if write:
            results.to_csv("predictions.csv", sep="\t")
        results = self.preprocess_results(results)
        results = results.dropna(how="any")

        score = np.nan
        if self.metric is not None:
            y_true = results.label.tolist()
            y_pred = results.prediction.tolist()
            score = self.metric(y_true, y_pred)
            if write:
                self.plotter(predictions=predictions, labels=labels)

        if write:
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


class WsiEvaluation(Evaluation):
    def preprocess_results(self, results: DataFrame) -> DataFrame:
        results["label"] = results["label"].replace(-1, np.nan)
        return results


class WicEvaluation(Evaluation):
    def preprocess_results(self, results: DataFrame) -> DataFrame:
        results["label"] = results["label"].replace(to_replace=0, value=np.nan)
        return results
