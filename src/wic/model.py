from abc import (
    ABC,
    abstractmethod,
)
from typing import Any, Iterable

import numpy as np
import more_itertools as mit
from pandas import DataFrame
from pydantic import BaseModel, Field
from tqdm import tqdm

from src.lemma import Lemma
from src.use import (
    Use,
    UseID,
)


class WICModel(BaseModel, ABC):
    scaler: Any = Field(default=None)  # should be a scikit-learn scaler
    predictions: dict[tuple[UseID, UseID], float] = Field(default_factory=dict)

    @abstractmethod
    def predict(self, use_pairs: Iterable[tuple[Use, Use]], **kwargs) -> list[float]:
        ...

    def predict_all(self, use_pairs: list[tuple[Use, Use]]) -> list[float]:
        predictions = self.predict(use_pairs=tqdm(use_pairs, desc="Computing WiC predictions", leave=False))
        if self.scaler is not None:
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.fit_transform(predictions).flatten().tolist()
        return predictions
