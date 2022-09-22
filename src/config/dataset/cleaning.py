from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List
from pandas import DataFrame
from omegaconf import MISSING


@dataclass
class Cleaning:
    defaults: list[Any] = field(default_factory=lambda: [{
        "stats": MISSING,
        "method": Cleaning.BooleanMethod.ALL
    }])
    stats: dict[str, Cleaning.CleaningParam] = MISSING
    method: Cleaning.BooleanMethod = MISSING

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
        keep: Cleaning.ThresholdParam = field(default_factory=lambda: Cleaning.ThresholdParam.ABOVE)

    class ThresholdParam(str, Enum):
        ABOVE = "above"
        BELOW = "below"

    class BooleanMethod(str, Enum):
        ALL = "all"
        ANY = "any"


