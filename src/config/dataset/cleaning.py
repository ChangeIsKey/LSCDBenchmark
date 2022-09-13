from __future__ import annotations
from pydantic.dataclasses import dataclass, Field
from enum import Enum
from typing import Dict, List
from pandas import DataFrame


@dataclass
class Cleaning:
    stats: Dict[str, Cleaning.CleaningParam]
    method: Cleaning.BooleanMethod = Field(default_factory=lambda: Cleaning.BooleanMethod.ALL)

    def __call__(self, agreements: DataFrame) -> List[str]:
        conditions = [
            f"{column} >= {cleaning_param.threshold}"
            if cleaning_param.keep is self.ThresholdParam.ABOVE
            else f"{column} <= {cleaning_param.threshold}"
            for column, cleaning_param in self.stats.items()
        ]

        match self.method:
            case self.BooleanMethod.ALL:
                return agreements.query("&".join(conditions))
            case self.BooleanMethod.ANY:
                return agreements.query("|".join(conditions))


    @dataclass
    class CleaningParam:
        threshold: float
        keep: Cleaning.ThresholdParam = Field(default_factory=lambda: Cleaning.ThresholdParam.ABOVE)

    class ThresholdParam(str, Enum):
        ABOVE = "above"
        BELOW = "below"

    class BooleanMethod(str, Enum):
        ALL = "all"
        ANY = "any"


