import re  # regular expressions
from sklearn import metrics
from scipy import stats
import csv
import matplotlib, matplotlib.ticker as mtick


from abc import ABC
from typing import (
    Any,
    Callable,
    Literal,
    TypeAlias,
    TypeVar,
)

import numpy as np
import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel

K = TypeVar("K", str, tuple[str, str])
V = TypeVar("V", int, float)


class Plots(BaseModel, ABC):

    def preprocess_inputs(
        self,
        results: DataFrame
    ) -> DataFrame:
        return results

    def __call__(
        self,
        predictions: dict[K, V],
        labels: dict[K, V]
    ):
        combined_results = self.combine_inputs(labels=labels, predictions=predictions)
        #combined_results.to_csv("predictions.csv", sep="\t")
        preprocessed_results = self.preprocess_inputs(combined_results)

        if self.metric is not None:
            y_true = preprocessed_results.label.tolist()
            y_pred = preprocessed_results.prediction.tolist()
            np.random.seed(13)
            metric_boot_histogram\
            (metrics.balanced_accuracy_score, y_pred, y_true) # replace with self.metric
            #score = self.metric(y_true, y_pred)

    @staticmethod
    def combine_inputs(
        labels: dict[K, V],
        predictions: dict[K, V]
    ) -> DataFrame:
        labels_df = DataFrame(
            {"target": list(labels.keys()), "label": list(labels.values())}
        )
        predictions_df = DataFrame(
            {
                "target": list(predictions.keys()), "prediction": list(predictions.values()),
            }
        )
        merged = pd.merge(
            left=labels_df, right=predictions_df, how="outer", on="target", validate="one_to_one", )

        return merged



    def one_boot(*data_args):
        """Usage: (t, p) = one_boot(true, pred) with true, pred, t, p arrays of same length
        """
        length = len(data_args[0])
        index = np.random.randint(0, length, size=length)
        return [ pd.Series(arg.values[index], name=arg.name)
             if isinstance(arg, pd.Series) else arg[index]   for  arg in data_args
           ]


    def calc_metrics(metrics, *data_args):
        """Return a list of calculated values for each metric applied to *data_args
        where metrics is a metric func or iterable of funcs e.g. [m1, m2, m3, m4]
        """
        metrics=_fix_metrics(metrics)
        mname = metrics.__name__ if hasattr(metrics, '__name__') else "Metric"
        return pd.Series\
        ([m(*data_args) for m in metrics], index=[_metric_name(m) for m in metrics], name=mname)

    def _metric_name(metric):  # use its prettified __name__
        name = re.sub(' score$', '', metric.__name__.replace('_',' ').strip())
        return name.title() if name.lower()==name else name

    def _fix_metrics(metrics_): # allow for single metric func or any iterable of metric funcs
        if callable(metrics_): metrics_=[metrics_]  # single metric func to list of one
            return pd.Series(metrics_)  # in case iterable metrics_ is generator, generate & store



    import tqdm  # progress bar
    def trange(iterable):  # narrower progress bar so it won't wrap
        return tqdm.trange(iterable, bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}")

    def raw_metric_samples(metrics, *data_args, nboots):
        """Return dataframe containing metric(s) for nboots boot sample datasets
        where metrics is a metric func or iterable of funcs e.g. [m1, m2, m3]
        """
        metrics=_fix_metrics(metrics)
        cols=[ calc_metrics(metrics, *boot_data)   for boot_data  in _boot_generator\
            #(*data_args, nboots=nboots)  if boot_data[0].nunique() >1  # >1 for log Loss, ROC
            (*data_args, nboots=nboots)  # >1 for log Loss, ROC
            ]#end of list comprehension
        return pd.DataFrame\
            ( {iboot: col for iboot,col in enumerate(cols)}#end of dict comprehension
            ).rename_axis("Boot", axis="columns").rename_axis(cols[0].name)

    def _boot_generator(*data_args, nboots): #return Gener of boot sampl datasets, not huge list!
        return (one_boot(*data_args) for _ in trange(nboots)) # generator expression



    DFLT_ALPHA = 0.05     # MIN_BOOTS_IN_1TAIL includes tail boundary:
    MIN_BOOTS_IN_1TAIL=11 # North et al.:10; Davidson&MacK ".05-->399"==>11, ".01-->1499"==>9
    MAX_ALPHA = 0.40 # Make it .25 or .20?  Min boots formula dubious even at .40 or .25?

    def _min_nboots_from(alpha): # Calcul. using this formula (from abov), solved for min_nboots:
                             # MIN_BOOTS_IN_1TAIL = (min_nboots - 1) * (0.5*alpha) + 1
                 return int(np.ceil((MIN_BOOTS_IN_1TAIL - 1)/(0.5*alpha) + 1))
    def _min_alpha_from(nboots): # Calc using this formula (frm above), but solved for min_alpha:
                             # MIN_BOOTS_IN_1TAIL = (nboots - 1) * (0.5*min_alpha) + 1
                 return 2 * (MIN_BOOTS_IN_1TAIL - 1) / (nboots - 1)
    MIN_NBOOTS = _min_nboots_from(MAX_ALPHA)

    def get_alpha_nboots(alpha=DFLT_ALPHA, nboots=None):
        """Return (alpha, nboots) with default nboots, applying MIN_NBOOTS & min alpha for nboots.
        """
        if alpha > MAX_ALPHA and (0 < (1-alpha) <= MAX_ALPHA):
            raise ValueError(f'alpha={alpha} > 0.5. Did you mean alpha={1-alpha:.9g}?')
        elif not (0 < alpha <= MAX_ALPHA):
            raise ValueError(f'alpha={alpha} is outside allowed range: 0 < alpha <= {MAX_ALPHA}')
        if pd.isna(nboots):   # by dflt use calculated min nboots for given alpha:
            nboots = _min_nboots_from(alpha)
        if int(nboots) != nboots: raise ValueError(f"nboots={nboots} isn't an integer")
        nboots = max(int(nboots), MIN_NBOOTS)  # apply min to a specified OR calculated nboots
        alpha = max(alpha, _min_alpha_from(nboots))  # apply min alpha for final nboots
        return (alpha, nboots)  # tuple of possibly-modified args




    def metric_boot_histogram(metric, *data_args, alpha=DFLT_ALPHA, nboots=None):
        """Plot histogram w/ lines for 1 observed metric & its confidence interval.
        """
        alpha, nboots = get_alpha_nboots(alpha, nboots) # 1-row df 2 series:
        series = raw_metric_samples(metric, *data_args, nboots=nboots).iloc[0,:]
        nboots=len(series) # in case some failed
        (lo, hi) = series.quantile([0.5*alpha, 1 - 0.5*alpha])
        matplotlib.rcParams["figure.dpi"] = 250
        ax = series.hist(bins=50, figsize=(7, 2.5), alpha=0.4, edgecolor='white')
        showing = f", showing {100*(1-alpha):.4g}% Confidence Interval"
        ax.set_title(f"Histogram of {nboots} boot results" + showing)
        ax.set_xlabel(series.name)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
        for x in lo, metric(*data_args), hi:
            ax.plot([x, x], [0, nboots*.07], lw=2.5)
        ax.figure.savefig('histogram.png') # the name of histogram could be a parameter
