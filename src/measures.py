from src.config import pairing, sampling
from src.lscd.model import VectorModel
from src.lscd.target import Target
import numpy as np


def apd_compare_all(target: Target, model: VectorModel, **kwargs) -> float:
    return np.mean(model.distances(sampling.all(pairing.COMPARE, target)))


def apd_compare_annotated(target: Target, model: VectorModel, **kwargs) -> float:
    return np.mean(model.distances(sampling.annotated(pairing.COMPARE, target)))


def apd_compare_sampled(
    target: Target, model: VectorModel, n: int, replace: bool
) -> float:
    return np.mean(
        model.distances(sampling.sampled(pairing.COMPARE, target, n=n, replace=replace))
    )


def apd_compare_minus_all_annotated(target: Target, model: VectorModel) -> float:
    return np.mean(
        model.distances(sampling.annotated(pairing.COMPARE, target))
    ) - np.mean(model.distances(sampling.annotated(pairing.ALL, target)))


diasense = apd_compare_minus_all_annotated
