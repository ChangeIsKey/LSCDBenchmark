from __future__ import annotations

from typing import Literal

from pandas import DataFrame
from pydantic import (
    BaseModel,
    Field,
)


class CleaningParam(BaseModel):
    """Define the threshold and the condition for Cleaning.

    Attributes:
        threshold   float
        keep    If it is "above", the condition will be the column grater equal than threshold. If it is "below", the condition will be the column less equal than threshold.
    """    
    threshold: float
    keep: Literal["above", "below"]

class Cleaning(BaseModel):
    """Query the data from agreements that match the conditions. Conditions are set by 
    initiating CleaningParam.

    Attributes:
        stats   a dictionary holds column name and CleaningParam object
        match   If it is "all", the Cleaning query the data that match all conditions. If it is "any", the queried data can only match one of the condition.
    """    
    stats: dict[str, CleaningParam]
    match: Literal["all", "any"]

    def __call__(self, agreements: DataFrame) -> DataFrame:
        """This object is callable.

        :param agreements: input dataframe
        :type agreements: DataFrame
        :return: output that match the conditions
        :rtype: DataFrame
        """        
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
