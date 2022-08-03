import pandas as pd
import numpy as np
import torch

from typing import Dict
from pathlib import Path
from pandas import DataFrame

import sklearn.metrics as metrics
import scipy.stats as stats

from src.config import Config


class Results:
    def __init__(
        self, config: Config, predictions: Dict[str, float], labels: Dict[str, float]
    ):
        self._scores = None
        self.config = config
        self.predictions = sorted(predictions.items())
        self.labels = sorted(labels.items())
        self.targets = [lemma for lemma, _ in self.predictions]

    @property
    def scores(self):
        if self._scores is None:
            try:
                self._scores = pd.read_csv(
                    self.config.results.output_directory.joinpath("scores.csv"),
                    delimiter="\t",
                )
            except FileNotFoundError:
                self._scores = DataFrame()
        return self._scores

    @scores.setter
    def scores(self, value: DataFrame):
        self._scores = value

    def score(self, task: str, metric=None, threshold: float = 0.5, t: float = 0.1):
        if task == "graded_change":
            labels = [values["change_graded"] for lemma, values in self.labels if lemma in self.targets]
            predictions = [pred for _, pred in self.predictions]

            spearman, p = stats.spearmanr(predictions, labels)
            row = {
                "n_targets": len(self.targets),
                "task": self.config.dataset.task,
                "method": "spearmanr",
                "score": spearman,
                "measure": self.config.model.measure.method.__name__,
                "model": self.config.model.name,
                "preprocessing": str(self.config.dataset.preprocessing.method_name),
                "dataset": self.config.dataset.name,
            }
            self.scores = pd.concat([self.scores, DataFrame([row])], ignore_index=True)
            self.export()
            return spearman

        elif task == "binary_change":
            # t = 0.1
            # mean = np.mean(distances, axis=0)
            # std = np.std(distances, axis=0)
            # threshold = mean + t * std

            # threshold could be a percentile

            labels = {
                lemma: values["binary_change"]
                for lemma, values in labels.items()
                if lemma in set(self.predictions.keys())
            }
            binary_scores = {
                target: int(distance >= threshold)
                for target, distance in self.predictions.items()
            }
            f1 = metrics.f1_score(list(labels.values()), list(binary_scores.values()))
            row = DataFrame([
                {
                    "n_targets": len(list(self.predictions.keys())),
                    "task": self.config.dataset.task,
                    "method": "spearmanr",
                    "score": spearman,
                    "model": self.config.model.name,
                    "preprocessing": self.config.dataset.preprocessing.method,
                    "dataset": self.config.dataset.name,
                    "threshold": threshold,
                }
            ])
            self.scores = pd.concat([self.scores, row])
            self.export()
            return f1

    def export(self):
        self.scores.to_csv(
            self.config.results.output_directory.joinpath("scores.csv"),
            sep="\t",
            index=False,
        )
