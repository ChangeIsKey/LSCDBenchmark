from abc import (
    ABC,
    abstractmethod,
)
from typing import Any

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
    scaler: Any # should be a scikit-learn scaler
    predictions: dict[tuple[UseID, UseID], float] = Field(default_factory=dict)

    @abstractmethod
    def predict(self, use_pairs: list[tuple[Use, Use]]) -> list[float]:
        ...

    def predict_all(self, use_pairs: list[tuple[Use, Use]]) -> list[float]:
        predictions = self.predict(use_pairs)
        return self.scaler.fit_transform(predictions)