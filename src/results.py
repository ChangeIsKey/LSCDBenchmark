from pathlib import Path
from typing import Dict, List, Any

from datetime import date, datetime
import json
import scipy.stats as stats
import sklearn.metrics as metrics
import pandas as pd
from pandas import DataFrame
from src.config import Config, EvaluationConfig, EvaluationTask
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
            case EvaluationTask.SEMANTIC_PROXIMITY:
                pass

    @property
    def aggregated_results(self):
        if self._aggregated_results is None:
            path = self._aggregated_results_dir / "results.tsv"
            self._aggregated_results = (
                pd.read_csv(path, engine="pyarrow", sep="\t")
                if path.exists()
                else DataFrame()
            )
        return self._aggregated_results

    def aggregated_results_row(self, score: float):
        # convert the config into a flat dictionary with keys ready for a csv export
        if self.config.evaluation.task is EvaluationTask.GRADED_CHANGE:
            self.config.evaluation.binary_threshold = None
        [dict_cfg] = pd.json_normalize(
            json.loads(self.config.json(exclude_none=True, exclude_unset=True)), sep="."
        ).to_dict(orient="records")

        return DataFrame(
            [
                {
                    "date": date.today().strftime("%d/%m/%Y"),
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "score": score,
                    "n_targets": len(self.keys),
                    **dict_cfg,
                }
            ]
        )

    @aggregated_results.setter
    def aggregated_results(self, new: DataFrame):
        self._aggregated_results = new

    def export(self, score: float):
        predictions = DataFrame(
            data={
                "target": self.keys,
                "prediction": self.predictions,
                "label": self.labels,
            }
        )

        predictions.to_csv("predictions.tsv", sep="\t", index=False)
        # self.aggregated_results = pd.concat(
        #     [self.aggregated_results, self.aggregated_results_row(score)], axis=0
        # )
        # self.aggregated_results.to_csv(
        #     self._aggregated_results_dir / "results.tsv", sep="\t", index=False
        # )
        Path("score.txt").write_text(str(score))
