import pandas as pd
import numpy as np
import torch

from typing import Dict
from pathlib import Path
from pandas import DataFrame

import sklearn.metrics as metrics
import scipy.stats as stats
from torchmetrics.functional import spearman_corrcoef

from src.config import Config


class Results:
    def __init__(
        self, config: Config, predictions: Dict[str, float], labels: Dict[str, float]
    ):
        self._scores = None
        self.config = config
        self.predictions = predictions
        self.labels = labels

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
        # 1. graded_change with spearman
        # include number of targets after
        #

        if task == "graded_change":
            labels = torch.tensor(
                [
                    values["graded_jsd"]
                    for lemma, values in self.labels.items()
                    if lemma in set(self.predictions.keys())
                ]
            ).to("cuda:1")

            predictions = torch.stack(list(self.predictions.values())).to("cuda:1")
            spearman = spearman_corrcoef(predictions, labels)
            row = {
                "n_targets": len(list(self.predictions.keys())),
                "task": self.config.dataset.task,
                "method": "spearmanr",
                "score": spearman.item(),
                "measure": self.config.model.measure.method.__name__,
                "model": self.config.model.name,
                "preprocessing": self.config.dataset.preprocessing.method.__name__,
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
