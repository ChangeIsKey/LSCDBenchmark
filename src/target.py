from __future__ import annotations
import csv
import pandera as pa
from typing import Dict
from pandera import DataFrameSchema, Column

from pathlib import Path
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from src.config import UseID, Config, Grouping


class Target:
    def __init__(self, name: str, config: Config, translation_table: Dict[str, str], path: Path) -> None:
        self.config = config
        self.name = name

        self.__translation_table = translation_table
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
            path = self.__wug / "data" / self.name / "uses.csv"
            self._uses = pd.read_csv(path, **self.__csv_params)
            # filter by grouping
            self._uses.grouping = self._uses.grouping.astype(Grouping)
            self._uses = self._uses[self._uses.grouping.isin(self.config.groupings)]
            # preprocess uses
            self._uses = self.__uses_pre_schema.validate(self._uses)
            self._uses = pd.concat([self._uses, self.uses.apply(self.config.preprocessing, axis=1, translation_table=self.__translation_table)], axis=1)
            self._uses = self.__uses_post_schema.validate(self._uses)
        return self._uses

    @property
    def __uses_pre_schema(self) -> DataFrameSchema:
        schema = DataFrameSchema({
            "identifier": Column(dtype=str, unique=True),
            "grouping": Column(dtype=Grouping)
        })

        if self.config.preprocessing.method in {"toklem", "tokenize"}:
            schema = schema.add_columns({
                "context_tokenized": Column(dtype=int),
                "indexes_target_token_tokenized": Column(dtype=int)
            })
        elif self.config.preprocessing.method in {"lemmatize"}:
            schema = schema.add_columns({
                "context_lemmatized": Column(dtype=str),
                "indexes_target_token_tokenized": Column(dtype=int),
                "indexes_target_token_lemmatized": Column(dtype=int, required=False)
            })
        elif self.config.preprocessing.method is None:
            def validate_indices(s: str) -> bool:
                parts = s.split(":")
                return len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit()

            schema = schema.add_columns({
                "context": Column(dtype=str),
                "indexes_target_token": Column(dtype=str, checks=pa.Check(check_fn=validate_indices))
            })
        
        return schema
    
    @property
    def __uses_post_schema(self) -> DataFrameSchema:
        return self.uses_pre_schema.add_columns({
            "context_preprocessed": Column(str),
            "target_index_begin": Column(int),
            "target_index_end": Column(int),
        })
        
    @property
    def judgments(self) -> DataFrame:
        if self._judgments is None:
            path = self.__wug / "data" / self.name / "judgments.csv"
            self._judgments = pd.read_csv(path, **self.__csv_params)
            self._judgments["judgment"] = self._judgments["judgment"].replace(to_replace=0, value=np.nan)
            self.judgments_schema.validate(self._judgments)
        return self._judgments
    
    @property
    def judgments_schema(self) -> DataFrameSchema:
        return DataFrameSchema({
            "identifier1": Column(dtype=str),
            "identifier2": Column(dtype=str),
            "judgment": Column(dtype=float, nullable=True),
        })
    
    @property
    def clusters(self) -> DataFrame:
        if self._clusters is None:
            path = self.__wug / "clusters" / "opt" / f"{self.name}.csv"
            self._clusters = pd.read_csv(path, **self.__csv_params)
            self._clusters = self.clusters_schema.validate(self._clusters)
        return self._clusters 

    @property
    def clusters_schema(self) -> DataFrameSchema:
        return DataFrameSchema({
            "identifier": Column(dtype=str, unique=True),
            "cluster": Column(int)
        })
    
    def uses_to_grouping(self) -> Dict[UseID, Grouping]:
        uses_to_grouping = dict(zip(self.uses.identifier, self.uses.grouping))

        return {
            identifier: value
            for identifier, value in uses_to_grouping.items() 
            if value in self.grouping_combination
        }

    def grouping_to_uses(self) -> Dict[UseID, Grouping]:
        uses_to_groupings = self.uses_to_grouping()
        return {
            group: [id_ for id_, grouping in uses_to_groupings.items() if grouping == group]
            for group in self.grouping_combination
        }