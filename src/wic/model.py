from abc import (
    ABC,
    abstractmethod,
)
import functools
from pathlib import Path
from typing import Any, Iterable
import json

import pandas as pd
import numpy as np
import more_itertools as mit
from pandas import DataFrame
from pydantic import BaseModel, Field, PrivateAttr
from tqdm import tqdm
from src.utils import utils
from sklearn import preprocessing

from src.use import (
    Use,
    UseID,
)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, np.integer, np.floating)):
            return obj.item()
        return super().default(obj)

class WICModel(BaseModel, ABC):
    _cache: pd.DataFrame = PrivateAttr(default=None)
    _cache_path: Path = PrivateAttr(default_factory=lambda: utils.path(".wic") / "predictions.parquet")
    class Config:
        json_encoders = {
            functools.partial: lambda f: f.func.__name__,
            preprocessing.StandardScaler: lambda _: "standard",
            preprocessing.MaxAbsScaler: lambda _: "maxabs",
            preprocessing.MinMaxScaler: lambda _: "minmax",
            preprocessing.RobustScaler: lambda _: "robust"
        }
        
    scaler: Any = Field(default=None)  # should be a scikit-learn scaler
    predictions: dict[tuple[UseID, UseID], float] = Field(default_factory=dict, exclude=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        try:
            self._cache = pd.read_parquet(self._cache_path)
        except FileNotFoundError:
            self._cache = self.as_df()
            self._cache = self._cache.assign(use_0=None, use_1=None, prediction=None)
            self._cache = self._cache.iloc[0:0]

    @abstractmethod
    def as_df(self) -> DataFrame:
        ...
    
    @abstractmethod
    def predict(self, use_pairs: Iterable[tuple[Use, Use]], **kwargs) -> list[float]:
        ...

    def predict_all(self, use_pairs: list[tuple[Use, Use]]) -> list[float]:
        left = [use_0.identifier for use_0, _ in use_pairs]
        right = [use_1.identifier for _, use_1 in use_pairs]
        query = self.as_df()
        query = query.loc[query.index.repeat(len(left))]
        query = query.assign(use_0=left, use_1=right)
        query = query.assign(**{col: None for col in self._cache.columns if col not in query.columns})
        self._cache = self._cache.assign(**{col: None for col in query.columns if col not in self._cache.columns})

        query["prediction"] = query["prediction"].astype("float32")
        self._cache["prediction"] = self._cache["prediction"].astype("float32")
        

        df = self._cache.merge(query, on=[col for col in query.columns if not col.startswith("prediction")], how="right")


        non_cached = df[df["prediction_x"].isna()].copy(deep=True).reset_index(drop=True)
        
        new_left = list(non_cached["use_0"])
        new_right = list(non_cached["use_1"])
        new_use_pairs = [(use_0, use_1) for use_0, use_1 in use_pairs if use_0.identifier in new_left and use_1.identifier in new_right]
        
        new_predictions = self.predict(use_pairs=new_use_pairs)
        
        if self.scaler is not None:
            new_predictions = np.array(new_predictions).reshape(-1, 1)
            new_predictions = self.scaler.fit_transform(new_predictions).flatten().tolist()

            with Path("scaler_parameters.json").open(mode="w", encoding="utf8") as f:
                as_json = json.dumps(self.scaler.__dict__, cls=NumpyEncoder, indent=4)
                f.write(as_json)
           
        new_use_pair_ids = list(zip(new_left, new_right))
        cached = df[~df["prediction_x"].isna()].copy(deep=True).reset_index()
        old_use_pair_ids = list(zip(list(cached["use_0"]), list(cached["use_1"])))

        full_predictions = dict(zip(new_use_pair_ids, new_predictions))
        full_predictions.update(dict(zip(old_use_pair_ids, cached.prediction_x)))
        
        for i, _ in non_cached.iterrows():
            non_cached.at[i, "prediction"] = new_predictions[i]  # type: ignore
        
        non_cached.drop(columns=["prediction_x", "prediction_y"], inplace=True)
        self._cache = pd.concat([self._cache, non_cached], ignore_index=True)
        main_cols = ["use_0", "use_1", "prediction"]
        extra_cols = [col for col in self._cache.columns if col not in main_cols]
        self._cache = self._cache[["use_0", "use_1", "prediction", *extra_cols]]
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache.to_parquet(self._cache_path, index=False)
        
        return list(full_predictions.values())
