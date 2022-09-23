from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List
from pandas import DataFrame

class ThresholdParam(str, Enum):
    ABOVE = "above"
    BELOW = "below"

@dataclass
class CleaningParam:
    threshold: float
    keep: ThresholdParam = field(default_factory=lambda: ThresholdParam.ABOVE)


class BooleanMethod(str, Enum):
    ALL = "all"
    ANY = "any"

@dataclass
class Cleaning:
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




