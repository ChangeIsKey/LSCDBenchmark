from abc import (
    ABC,
    abstractmethod,
)
from pathlib import Path
from typing import Any, Iterable
import json

import numpy as np
import more_itertools as mit
from pandas import DataFrame
from pydantic import BaseModel, Field
from tqdm import tqdm
from src.utils import utils

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
        predictions = self.predict(use_pairs=use_pairs)
        if self.scaler is not None:
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.fit_transform(predictions).flatten().tolist()

            with Path("scaler_parameters.json").open(mode="w", encoding="utf8") as f:
                possible_attrs = [ 
                    "mean_", "var_", "scale_", "min_", 
                    "data_min_", "data_max_", "data_range_",
                    "max_abs_", "center_"
                ]
                parameters = {attr: getattr(self.scaler, attr).tolist() for attr in possible_attrs if hasattr(self.scaler, attr)}
                as_json = json.dumps(parameters)
                f.write(as_json)
            
        return predictions
