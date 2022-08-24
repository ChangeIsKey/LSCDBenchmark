import csv
from typing import Dict

from pathlib import Path
import src.utils as utils
import pandas as pd
from pandas import DataFrame
from src.config import ID, Config


class Target:
    def __init__(self, name: str, config: Config, translation_table: Dict[str, str], path: Path) -> None:
        self.config = config
        self.name = name
        self.translation_table = translation_table

        self.__wug = path
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
            path = self.__wug / "data" / self.name / "uses.tsv"
            if not path.exists():
                path = path.with_suffix(".csv")
            self._uses = pd.read_csv(path, **self.__csv_params)
            # preprocess uses
            self._uses = pd.concat([self._uses, self.uses.apply(self.config.preprocessing, axis=1, translation_table=self.translation_table)], axis=1)
        return self._uses

    @property
    def judgments(self) -> DataFrame:
        if self._judgments is None:
            path = self.__wug / "data" / self.name / "judgments.tsv"
            if not path.exists():
                path = path.with_suffix(".csv")
            self._judgments = pd.read_csv(path, **self.__csv_params)
        return self._judgments
    
    @property
    def clusters(self) -> DataFrame:
        if self._clusters is None:
            path = self.__wug / "clusters" / "opt" / f"{self.name}.tsv"
            if not path.exists():
                path = path.with_suffix(".csv")
            self._clusters = pd.read_csv(path, **self.__csv_params)
        return self._clusters 
    
    def uses_to_grouping(self) -> Dict[ID, int]:
        uses_to_grouping = dict(zip(self.uses.identifier, self.uses.grouping))

        return {
            identifier: value
            for identifier, value in uses_to_grouping.items() 
            if value in self.grouping_combination
        }