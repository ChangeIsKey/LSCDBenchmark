from src.config import pairing, sampling
from src.lscd.model import VectorModel
from src.lscd.target import Target
import numpy as np
import torch


def apd_compare_all(target: Target, model: VectorModel) -> float:
    return torch.mean(model.distances(sampling.all(pairing.COMPARE, target), dim=0))
