from typing import Literal
import torch
import scipy
import numpy as np

from scipy import stats


def spearmanr(y_true: list[float], y_pred: list[float], **kwargs) -> float:
    corr, _ = stats.spearmanr(a=y_true, b=y_pred, **kwargs)
    return corr

def pearsonr(y_true: list[float], y_pred: list[float], **kwargs) -> float:
    corr, _ = stats.pearsonr(x=y_true, y=y_pred, **kwargs)
    return corr

def l1(v: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(input=v, p=1, dim=0)

def l2(v: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(input=v, p=2, dim=0)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return 1 - scipy.spatial.distance.cosine(a, b)

def euclidean_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return -scipy.spatial.distance.euclidean(a, b)