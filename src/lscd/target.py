import importlib.util
from typing import Dict

import pandas as pd
from pandas import DataFrame
from src.config import ID, Config
from src.use import Use


class Target:
    def __init__(
        self,
        name: str,
        uses: DataFrame,
        labels: DataFrame,
        judgments: DataFrame,
        config: Config,
    ):
        self.name = name
        self.uses = uses
        self._ids_to_uses = None
        self.labels = labels
        self.judgments = judgments
        self.grouping_combination = config.dataset.groupings
        self.config = config

        self.preprocess()

    def preprocess(self) -> None:
        """
        :param nlp:
        :param how:
            A function to preprocess the contexts. This can be any function that takes a pandas Series (i.e., a row of
            one of the uses.csv files) as input, and possibly other parameters, and returns a string.
            The module preprocessing contains useful preprocessing functions.
        :param params: Extra keyword parameters for the preprocessing function
        :raises TypeError: if the 'how' parameter is neither None nor a function
        :raises TypeError: if the one of the returned outputs of the preprocessing function is not a string
        """

        self.uses = pd.concat(
            [
                self.uses, 
                self.uses.apply(
                    func=self.config.dataset.preprocessing, 
                    axis=1
                )
            ], 
            axis=1
        )
    
    def uses_to_grouping(self) -> Dict[ID, int]:
        print(self.uses)
        uses_to_grouping = (
            self.uses.loc[:, ["identifier", "grouping"]]
            .set_index("identifier")
            .to_dict("index")
        )
        return {
            identifier: value["grouping"] 
            for identifier, value in uses_to_grouping.items() 
            if value["grouping"] in self.grouping_combination
        }