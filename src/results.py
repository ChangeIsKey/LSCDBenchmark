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
        self.labels = [labels[key] for key in self.keys]
        self.predictions = [predictions[key] for key in self.keys]

        self._aggregated_results_dir = utils.path("results")
        self._aggregated_results_dir.mkdir(exist_ok=True)
        self._aggregated_results = None

    def score(self):
        match self.config.evaluation.task:
            case None:
                self.export(score=None)
                return np.nan
            case EvaluationTask.GRADED_CHANGE:
                spearman, p = stats.spearmanr(self.labels, self.predictions)
                self.export(score=spearman)
                return spearman
            case EvaluationTask.BINARY_CHANGE:
                threshold = self.config.evaluation.binary_threshold(self.predictions)
                self.predictions = [
                    int(self.predictions[i] >= threshold)
                    for i in range(len(self.predictions))
                ]

                f1 = metrics.f1_score(
                    y_true=self.labels,
                    y_pred=self.predictions,
                )
                self.export(score=f1)
                return f1

            case EvaluationTask.CLUSTERING:
                pass

            case EvaluationTask.SEMANTIC_PROXIMITY:
                pass

    def export(self, score: Optional[float]):
        predictions = DataFrame(
            data={
                "target": self.keys,
                "prediction": self.predictions,
                "label": self.labels,
            }
        )

        predictions.to_csv("predictions.tsv", sep="\t", index=False)

        if score is not None:
            score = str(score)
            Path("score.txt").write_text(score)
