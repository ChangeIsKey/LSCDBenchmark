from typing import Iterable
import numpy as np

def mean_std(predictions: Iterable[float], t: float):
    mean = np.mean(predictions, axis=0)
    std = np.std(predictions, axis=0)
    threshold = mean + t * std
    return threshold
