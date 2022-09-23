import numpy as np


def mean_std(predictions: list[float], t: float) -> float:
    mean = np.mean(predictions, axis=0)
    std = np.std(predictions, axis=0)
    return mean + t * std