from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import scipy.stats as stats
import sklearn.metrics as metrics
from pandas import DataFrame
from src.config.config import Config
from src.config.evaluation.task import EvaluationTask
import src.utils as utils


class Results:
    def __init__(
        self, config: Config, predictions: Dict[Any, float], labels: Dict[Any, float]
    ):
        self._scores = None
        self.config = config
        self.keys = sorted(
            set(list(labels.keys())).intersection(list(predictions.keys()))
        )
        self.predictions = predictions
        self.labels = labels

    def score(self):
        labels = [self.labels[key] for key in self.keys]
        predictions = [self.predictions[key] for key in self.keys]
        match self.config.evaluation.task:
            case None:
                self.export(score=None)
                return np.nan
            case EvaluationTask.GRADED_CHANGE:
                score = self.config.evaluation.metric(labels, predictions)
                self.export(score=score)
                return score
            case EvaluationTask.BINARY_CHANGE:
                threshold = self.config.evaluation.threshold_fn(predictions)
                self.predictions = [
                    int(self.predictions[i] >= threshold)
                    for i in range(len(self.predictions))
                ]

                score = self.config.evaluation.metric(labels, predictions)
                self.export(score=score)
                return score

            case EvaluationTask.CLUSTERING:
                pass

            case EvaluationTask.SEMANTIC_PROXIMITY:
                pass

    def export(self, score: Optional[float]):
        if score is not None:
            predictions = DataFrame(
                data={
                    "target": self.keys,
                    "prediction": self.predictions,
                    "label": self.labels,
                }
            )

            predictions.to_csv("predictions.tsv", sep="\t", index=False)
            score = str(score)
            Path("score.txt").write_text(score, encoding="utf8")
