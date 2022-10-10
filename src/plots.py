import matplotlib, matplotlib.ticker as mtick


from abc import ABC
from typing import (
    Any,
    Callable,
    TypeVar,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel, PrivateAttr, root_validator, validator

K = TypeVar("K", str, tuple[str, str])
V = TypeVar("V", int, float)


class Plotter(BaseModel, ABC):

    max_alpha: float
    default_alpha: float
    min_boots_in_one_tail: int
    metric: Callable[[npt.NDArray[np.float32], npt.NDArray[np.float32]], float]

    _initial_alpha: float = PrivateAttr(default=None)
    _min_n_boots: int = PrivateAttr(default=None)
    _n_boots: int = PrivateAttr(default=None)
    _alpha: float = PrivateAttr(default=None)

    @root_validator
    def validate_alphas(cls, v: dict[str, float]) -> dict[str, float]:
        default_alpha = v["default_alpha"]
        max_alpha = v["max_alpha"]
        if default_alpha > max_alpha and (0 < (1 - default_alpha) <= max_alpha):
            raise ValueError(
                f"alpha={default_alpha} > 0.5. Did you mean alpha={1-default_alpha:.9g}?"
            )
        elif not (0 < default_alpha <= max_alpha):
            raise ValueError(
                f"alpha={default_alpha} is outside allowed range: 0 < alpha <= {max_alpha}"
            )
        return v

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._min_n_boots = self._min_n_boots_from(self.max_alpha)

        # apply min to a specified OR calculated n_boots
        self._n_boots = max(
            self._min_n_boots_from(self.default_alpha), self._min_n_boots
        )
        # apply min alpha for final n_boots
        self._alpha = max(self.default_alpha, self._min_alpha_from(self._n_boots))

    def preprocess_inputs(self, results: DataFrame) -> DataFrame:
        return results.dropna(how="any")

    def __call__(self, predictions: dict[K, V], labels: dict[K, V]):
        combined_results = self.combine_inputs(labels=labels, predictions=predictions)
        preprocessed_results = self.preprocess_inputs(combined_results)

        if self.metric is not None:
            y_true = preprocessed_results.label.to_numpy()
            y_pred = preprocessed_results.prediction.to_numpy()
            self.metric_boot_histogram(y_pred, y_true)

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

    def _one_boot(
        self, y_true: npt.NDArray[np.float32], y_pred: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Usage: (t, p) = one_boot(true, pred) with true, pred, t, p arrays of same length"""
        length = len(y_true)
        index = np.random.randint(0, length, size=length)
        return y_true[index], y_pred[index]

    def _boot_generator(self, y_true: np.ndarray, y_pred: np.ndarray):
        # return Gener of boot sampl datasets, not huge list!
        return (
            self._one_boot(y_pred=y_pred, y_true=y_true) for _ in range(self._n_boots)
        )

    def _min_n_boots_from(self, alpha: float) -> int:
        # Calcul. using this formula (from abov), solved for min_n_boots:
        # MIN_BOOTS_IN_1TAIL = (min_n_boots - 1) * (0.5*alpha) + 1
        return int(np.ceil((self.min_boots_in_one_tail - 1) / (0.5 * alpha) + 1))

    def _min_alpha_from(self, n_boots: int) -> float:
        # Calc using this formula (frm above), but solved for min_alpha:
        # MIN_BOOTS_IN_1TAIL = (n_boots - 1) * (0.5*min_alpha) + 1
        return 2 * (self.min_boots_in_one_tail - 1) / (n_boots - 1)

    def metric_boot_histogram(self, y_true, y_pred):
        """Plot histogram w/ lines for 1 observed metric & its confidence interval."""
        results = pd.Series(
            [
                self.metric(y_true, y_pred)
                for y_true, y_pred in self._boot_generator(y_true, y_pred)
            ]
        )
        n_boots = len(results)  # in case some failed
        (lo, hi) = results.quantile([0.5 * self._alpha, 1 - 0.5 * self._alpha])
        matplotlib.rcParams["figure.dpi"] = 250
        ax = results.hist(bins=50, figsize=(7, 2.5), alpha=0.4, edgecolor="white")
        showing = f", showing {100*(1-self._alpha):.4g}% Confidence Interval"
        ax.set_title(f"Histogram of {n_boots} boot results" + showing)
        ax.set_xlabel(results.name)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
        for x in lo, self.metric(y_true, y_pred), hi:
            ax.plot([x, x], [0, n_boots * 0.07], lw=2.5)
        ax.figure.savefig("histogram.png")  # the name of histogram could be a parameter
