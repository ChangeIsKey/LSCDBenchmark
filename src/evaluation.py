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


class DatasetMetadata(TypedDict):
    name: str
    version: str


class Evaluation(BaseModel, ABC):
    task: EvaluationTask | None
    metric: Callable[[list[float | int], list[float | int]], Any] | None
    plotter: Plotter | None
    exclude_annotators: list[str]
    __csv_params__: CsvParams = PrivateAttr(default_factory=CsvParams)


    def __call__(self, predictions: dict[K, V], labels: dict[K, V], write: bool) -> int | float:
        results = self.combine_inputs(labels=labels, predictions=predictions)
        if write:
            results.to_csv("predictions.csv", sep="\t")
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
    def get_labels(self, dataset: DatasetMetadata) -> dict[Any, float]:
        pass


class LSCDEvaluation(Evaluation):
    def get_stats_groupings(self, dataset: DatasetMetadata) -> DataFrame:
        stats_groupings = "stats_groupings.csv"
        dataset_path = utils.dataset_path(dataset["name"], dataset["version"])
        path = dataset_path / "stats" / "semeval" / stats_groupings
        if not path.exists():
            path = dataset_path / "stats" / "opt" / stats_groupings
        if not path.exists():
            path = dataset_path / "stats" / stats_groupings
        return pd.read_csv(path, **self.__csv_params__.dict()) # type: ignore
    

class LSCDGradedEvaluation(LSCDEvaluation):
    def get_labels(self, dataset: DatasetMetadata) -> dict[str, float]:
        stats_groupings = self.get_stats_groupings(dataset)
        return dict(zip(stats_groupings.lemma, stats_groupings.change_graded))


class LSCDBinaryEvaluation(LSCDEvaluation):
    def get_labels(self, dataset: DatasetMetadata) -> dict[str, float]:
        stats_groupings = self.get_stats_groupings(dataset)
        return dict(zip(stats_groupings.lemma, stats_groupings.change_binary))


class LSCDCompareEvaluation(LSCDEvaluation):
    def get_labels(self, dataset: DatasetMetadata) -> dict[str, float]:
        stats_groupings = self.get_stats_groupings(dataset)
        return dict(zip(stats_groupings.lemma, stats_groupings.COMPARE))


class WsiEvaluation(Evaluation):
    def get_labels(self, dataset: DatasetMetadata) -> dict[UseID, float]:
        path = utils.dataset_path(dataset["name"], dataset["version"]) / "clusters" / "opt" / "clusters.parquet"
        clusters = (
            pd.read_parquet(path, engine="pyarrow")
            .replace(to_replace=-1, value=np.nan)
        )
        return dict(zip(clusters.identifier, clusters.cluster))


class WicEvaluation(Evaluation):

    def get_labels(self, dataset: DatasetMetadata) -> dict[tuple[UseID, UseID], float]:
        judgments_path = utils.dataset_path(dataset["name"], dataset["version"]) / "data" / "judgments.parquet"
        judgments = pd.read_parquet(judgments_path, engine="pyarrow")
        judgments["judgment"] = judgments["judgment"].astype(float)
        judgments = judgments[~judgments["annotator"].isin(self.exclude_annotators)]
        judgments.replace(to_replace=0, value=np.nan, inplace=True)
        # pandas.core.groupby.GroupBy.median ignores missing values -> no need for nanmedian
        judgments = judgments.groupby(by=["identifier1", "identifier2"])["judgment"].median().reset_index()
        annotated_pairs = zip(judgments.identifier1, judgments.identifier2)
        return dict(zip(list(annotated_pairs), judgments.judgment))

class BinaryWicEvaluation(WicEvaluation):
    def get_labels(self, dataset: DatasetMetadata) -> dict[tuple[UseID, UseID], float]:
        labels = super().get_labels(dataset)
        return {use_pair: judgment for use_pair, judgment in labels.items() if judgment in [4.0, 1.0]}
