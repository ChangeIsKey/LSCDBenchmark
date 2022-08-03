from src.config import pairing, sampling
from src.lscd.model import VectorModel
from src.lscd.target import Target
import numpy as np


def apd_compare_all(target: Target, model: VectorModel, **kwargs) -> float:
    return np.mean(model.distances(sampling.all(pairing.COMPARE, target)))


def apd_compare_annotated(target: Target, model: VectorModel, **kwargs) -> float:
    return np.mean(model.distances(sampling.annotated(pairing.COMPARE, target)))

