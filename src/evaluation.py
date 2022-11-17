from abc import ABC, abstractmethod
import csv
import os
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
    "semantic_proximity", "change_graded", "change_binary", "COMPARE", "wsi"
]
K = TypeVar("K", str, tuple[str, str])
V = TypeVar("V", int, float)


class Evaluation(BaseModel, ABC):
    task: EvaluationTask | None
    metric: Callable[[list[float | int], list[float | int]], Any] | None
    plotter: Plotter | None
    exclude_annotators: list[str]
    __csv_params__: CsvParams = PrivateAttr(default_factory=CsvParams)

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
            if write and self.plotter is not None:
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
    
    @abstractmethod
    def get_labels(self, *args, **kwargs) -> dict[Any, float]:
        pass
        

class WsiEvaluation(Evaluation):
    def preprocess_results(self, results: DataFrame) -> DataFrame:
        results["label"] = results["label"].replace(-1, np.nan)
        return results


class WicEvaluation(Evaluation):
    binarize: bool
    def preprocess_results(self, results: DataFrame) -> DataFrame:
        results["label"] = results["label"].replace(to_replace=0, value=np.nan)
        return results

    def get_labels(
        self, 
        dataset_name: str, 
        dataset_version: str, 
        lemma: str, 
    ) -> dict[tuple[UseID, UseID], float]:
        path = utils.dataset_path(dataset_name, dataset_version) / "data" / lemma / "judgments.csv"
        judgments_df: DataFrame = pd.read_csv(path, **self.__csv_params__.dict()) # type: ignore
        judgments_df["judgment"] = judgments_df["judgment"].astype(float)
        judgments_df = judgments_df[~judgments_df["annotator"].isin(self.exclude_annotators)]
        judgments_df = judgments_df.groupby(by=["identifier1", "identifier2"])["judgment"].median().reset_index()

        if self.binarize:
            judgments_df = judgments_df[judgments_df["judgment"].isin([4.0, 1.0])]
        
        judgments_df.sort_values(by=["identifier1", "identifier2"], inplace=True)
        annotated_pairs = zip(judgments_df.identifier1, judgments_df.identifier2)
        return dict(zip(list(annotated_pairs), judgments_df.judgment))