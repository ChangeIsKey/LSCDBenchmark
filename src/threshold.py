import numpy as np


def mean_std(predictions: list[float], t: float) -> list[int]:
    mean = np.mean(predictions, axis=0)
    std = np.std(predictions, axis=0)
    threshold = mean + t * std

    return [int(p >= threshold) for p in predictions]