from pathlib import Path
from typing import Dict, List

import scipy.stats as stats
import sklearn.metrics as metrics
from pandas import DataFrame
from src.config import Config, EvaluationConfig, EvaluationTask


class Results:
    def __init__(
        self, config: Config, predictions: Dict[str, float], labels: Dict[str, float]
    ):
        self._scores = None
        self.config = config
        self.predictions = predictions
        self.labels = labels
        self.targets = sorted(
            set([lemma for lemma in self.labels])
            .intersection([lemma for lemma in self.predictions])
        )

    def score(self):

        labels = [self.labels[target] for target in self.targets]

        if self.config.evaluation.task is EvaluationTask.GRADED_CHANGE:
            predictions = [self.predictions[target] for target in self.targets]
            spearman, p = stats.spearmanr(labels, predictions)
            self.export(score=spearman, labels=labels, predictions=predictions)
            return spearman

        elif self.config.evaluation.task is EvaluationTask.BINARY_CHANGE:
            # The parameters need to be filled
            threshold = self.config.evaluation.binary_threshold.method()
            predictions = [int(self.predictions[target] >= threshold) for target in self.targets]

            f1 = metrics.f1_score(
                y_true=labels, 
                y_pred=predictions,
            )
            self.export(score=f1, labels=labels, predictions=predictions)
            return f1

    def export(self, score: float, labels: List[float], predictions: List[float]):
        predictions = DataFrame(
            data={
                "target": self.targets,
                "prediction": predictions,
                "label": labels
            }
        )
        predictions.to_csv("predictions.tsv", sep="\t", index=False)
        Path("score.txt").write_text(str(score))
