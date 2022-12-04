from __future__ import annotations

from typing import Literal

from pandas import DataFrame
from pydantic import (
    BaseModel,
    Field,
)


class CleaningParam(BaseModel):
    threshold: float
    keep: Literal["above", "below"]

class Cleaning(BaseModel):
    stats: dict[str, CleaningParam]
    match: Literal["all", "any"]

    def __call__(self, agreements: DataFrame) -> DataFrame:
        conditions = [
            f"{column} >= {cleaning_param.threshold}"
            if cleaning_param.keep == "above"
            else f"{column} <= {cleaning_param.threshold}"
            for column, cleaning_param in self.stats.items()
        ]

        match self.match:
            case "all":
                return agreements.query("&".join(conditions))
            case "any":
                return agreements.query("|".join(conditions))
