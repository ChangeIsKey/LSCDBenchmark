import hydra
import pandas as pd
import numpy as np
import torch
import os

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
        self.predictions = predictions
        self.predictions = self.predictions
        self.labels = sorted(labels.items())
        self.targets = [lemma for lemma in self.predictions]

    def score(self, task: str, metric=None, threshold: float = 0.5, t: float = 0.1):
        if task == "graded_change":
            labels = [
                values["change_graded"]
                for lemma, values in self.labels
                if lemma in self.targets
            ]
            predictions = list(self.predictions.values())
            spearman, p = stats.spearmanr(predictions, labels)
            self.export(score=spearman)
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
            self.export(score=f1)
            return f1

    def export(self, score: float):
        predictions = DataFrame(
            data={"target": self.targets, "value": list(self.predictions.values())}
        )
        predictions.to_csv("predictions.tsv", sep="\t", index=False)
        Path("score.txt").write_text(str(score))
