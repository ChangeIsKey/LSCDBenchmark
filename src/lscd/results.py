from pathlib import Path
from typing import Dict, List

import scipy.stats as stats
import sklearn.metrics as metrics
from pandas import DataFrame
from src.config import Config


class Results:
    def __init__(
        self, config: Config, predictions: Dict[str, float], labels: Dict[str, float]
    ):
        self._scores = None
        self.config = config
        self.predictions = predictions
        self.labels = labels
        self.targets = set([lemma for lemma in self.labels]).intersection([lemma for lemma in self.predictions])

    def score(self, task: str, metric=None, threshold: float = 0.5, t: float = 0.1):
        if task == "change_graded":
            labels = [self.labels[target]["change_graded"] for target in self.targets]
            predictions = [self.predictions[target] for target in self.targets]
            spearman, p = stats.spearmanr(labels, predictions)
            self.export(score=spearman, labels=labels, predictions=predictions)
            return spearman

        elif task == "change_binary":
            # t = 0.1
            # mean = np.mean(distances, axis=0)
            # std = np.std(distances, axis=0)
            # threshold = mean + t * std

            # threshold could be a percentile
            binary_scores = {
                target: int(distance >= threshold)
                for target, distance in self.predictions.items()
            }

            f1 = metrics.f1_score(
                y_true=list(self.labels.values()), y_pred=list(binary_scores.values())
            )
            self.export(score=f1)
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
