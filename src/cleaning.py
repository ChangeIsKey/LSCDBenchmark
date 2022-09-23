from __future__ import annotations
from pydantic import BaseModel, Field
from enum import Enum
from typing import List
from pandas import DataFrame

class ThresholdParam(str, Enum):
    ABOVE = "above"
    BELOW = "below"

class CleaningParam(BaseModel):
    threshold: float
    keep: ThresholdParam = Field(default_factory=lambda: ThresholdParam.ABOVE)


class BooleanMethod(str, Enum):
    ALL = "all"
    ANY = "any"


class Cleaning(BaseModel):
    stats: dict[str, CleaningParam] 
    method: BooleanMethod

    def __call__(self, agreements: DataFrame) -> DataFrame:
        conditions = [
            f"{column} >= {cleaning_param.threshold}"
            if cleaning_param.keep is ThresholdParam.ABOVE
            else f"{column} <= {cleaning_param.threshold}"
            for column, cleaning_param in self.stats.items()
        ]

        match self.method:
            case BooleanMethod.ALL:
                return agreements.query("&".join(conditions))
            case BooleanMethod.ANY:
                return agreements.query("|".join(conditions))




