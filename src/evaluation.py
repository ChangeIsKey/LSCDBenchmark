import json
from abc import ABC
from typing import Any, Callable, Literal, TypeAlias, TypedDict, TypeVar

import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel, Field

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
    plotter: Plotter | None = Field(...)

    def __call__(self, predictions: dict[K, V], labels: dict[K, V]) -> int | float:
        if self.plotter is not None:
            self.plotter(predictions=predictions, labels=labels)

        print(predictions)
        results = self.combine_inputs(labels=labels, predictions=predictions)
        results.to_csv("predictions.csv", sep="\t")
        results = results.dropna(how="any")

        y_true = results.label.tolist()
        y_pred = results.prediction.tolist()

        #print(y_true)
        #print(y_pred)
        result = {"score": self.metric(y_true, y_pred), "metric": self.metric.func.__name__}
        
        with open(file="result.json", mode="w", encoding="utf8") as f:
            f.write(json.dumps(result, indent=4))
        return result["score"]

    @staticmethod
    def combine_inputs(labels: dict[K, V], predictions: dict[K, V]) -> DataFrame:
        labels_df = DataFrame(
            {"instance": list(labels.keys()), "label": list(labels.values())}
        )
        predictions_df = DataFrame(
            {
                "instance": list(predictions.keys()),
                "prediction": list(predictions.values()),
            }
        )
        merged = pd.merge(
            left=labels_df,
            right=predictions_df,
            how="inner",
            on="instance",
            validate="one_to_one",
        )

        first_key = list(labels.keys())[0]
        if isinstance(first_key, (tuple, list, set)):
            new_cols = merged.instance.apply(pd.Series)
            new_cols.columns = [f"instance_{i}" for i in range(len(new_cols.columns))]  # type: ignore
            merged.drop(columns=["instance"], inplace=True) 
            merged = pd.concat([new_cols, merged], axis=1)
            merged = merged[list(new_cols.columns) + ["prediction", "label"]]
        else:
            merged = merged[["instance", "prediction", "label"]]

        return merged
