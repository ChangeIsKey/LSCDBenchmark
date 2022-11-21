from typing import Literal
from scipy import stats


def spearmanr(y_true: list[float], y_pred: list[float], **kwargs) -> float:
    corr, _ = stats.spearmanr(a=y_true, b=y_pred, **kwargs)
    return corr
