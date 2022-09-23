from pydantic import BaseModel
from typing import Any, Callable
import numpy as np
from tqdm import tqdm

from src.lscd.lscd_model import LSCDModel
from src.target import Pairing, Sampling, Target
from src.wic.bert import ContextualEmbedderWIC
from src.wic.deepmistake import DeepMistakeWIC


class ApdCompareAll(LSCDModel):
    def predict(self, targets: list[Target]) -> tuple[list[str], list[float | int]]:
        predictions: dict[str, float | int] = {}
        for target in targets:
            use_pairs = target.use_pairs(pairing=Pairing.COMPARE, sampling=Sampling.ALL)
            similarities = self.wic.predict(use_pairs)
            predictions[target.name] = np.mean(similarities).item()

        if self.threshold_fn is not None:
            values = list(predictions.values())
            threshold = self.threshold_fn(values)
            predictions = {target_name: int(p >= threshold) for target_name, p in predictions.items()}

        return list(predictions.keys()), list(predictions.values())
