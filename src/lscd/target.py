import csv
from typing import Dict

import src.utils as utils
import pandas as pd
from pandas import DataFrame
from src.config import ID, Config


class Target:
    def __init__(self, name: str, config: Config, translation_table: Dict[str, str]) -> None:
        self.config = config
        self.name = name
        self.translation_table = translation_table

        self.__wug = utils.path("wug") / self.config.dataset.name / self.config.dataset.version
        self.__csv_params = dict(
            delimiter="\t", encoding="utf8", quoting=csv.QUOTE_NONE
        )

        self._uses = None
        self._judgments = None
        self._clusters = None
        self.grouping_combination = config.groupings

    @property
    def uses(self) -> DataFrame:
        if self._uses is None:
            # load uses
            self._uses = pd.read_csv(self.__wug / "data" / self.name / "uses.tsv", **self.__csv_params)
            # preprocess uses
            self._uses = pd.concat([self._uses, self.uses.apply(self.config.preprocessing, axis=1, translation_table=self.translation_table)], axis=1)
        return self._uses

    @property
    def judgments(self) -> DataFrame:
        if self._judgments is None:
            self._judgments = pd.read_csv(self.__wug / "data"/ self.name / "judgments.tsv", **self.__csv_params)
        return self._judgments
    
    @property
    def clusters(self) -> DataFrame:
        if self._clusters is None:
            self._clusters = pd.read_csv(self.__wug / "clusters" / "opt" / f"{self.name}.tsv", **self.__csv_params)
        return self._clusters 
    
    def uses_to_grouping(self) -> Dict[ID, int]:
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