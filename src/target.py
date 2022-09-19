from __future__ import annotations

import csv
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict

import numpy as np
import pandas as pd
import pandera as pa
from pandas import DataFrame
from pandera import Column, DataFrameSchema

from src.config.config import Config

if TYPE_CHECKING:
    from src.config.config import UseID


class Sampling(str, Enum):
    annotated = "annotated"
    sampled = "sampled"
    all = "all"


class Pairing(str, Enum):
    COMPARE = "COMPARE"
    EARLIER = "EARLIER"
    LATER = "LATER"


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
        self.grouping_combination = config.dataset.groupings

    @property
    def uses(self) -> DataFrame:
        if self._uses is None:
            # load uses
            path = self.__wug / "data" / self.name / "uses.csv"
            self._uses = pd.read_csv(path, **self.__csv_params)
            # filter by grouping
            self._uses.grouping = self._uses.grouping.astype(str)
            self._uses = self._uses[self._uses.grouping.isin(self.config.dataset.groupings)]
            # preprocess uses
            self._uses = pd.concat([self._uses, self._uses.apply(self.config.dataset.preprocessing.__call__, axis=1, translation_table=self.__translation_table)], axis=1)
            self._uses = self.__uses_schema.validate(self._uses)
        return self._uses
    
    @property
    def __uses_schema(self) -> DataFrameSchema:
        return DataFrameSchema({
            "identifier": Column(dtype=str, unique=True),
            "grouping": Column(dtype=str),
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
    
    def uses_to_grouping(self) -> Dict[UseID, str]:
        uses_to_grouping = dict(zip(self.uses.identifier, self.uses.grouping))

        return {
            identifier: value
            for identifier, value in uses_to_grouping.items() 
            if value in self.grouping_combination
        }

    def grouping_to_uses(self) -> Dict[UseID, str]:
        uses_to_groupings = self.uses_to_grouping()
        return {
            group: [id_ for id_, grouping in uses_to_groupings.items() if grouping == group]
            for group in self.grouping_combination
        }
    
    def use_pairs(self, pairing: Pairing, sampling: Sampling, **params) -> list[tuple[UseID, UseID]]:
        if sampling is Sampling.annotated:
            ids1, ids2 = self.__split_annotated_uses(target)
        else:
            match pairing:
                case Pairing.COMPARE:
                    ids1 = target.uses[target.uses.grouping == target.grouping_combination[0]].identifier.tolist()
                    ids2 = target.uses[target.uses.grouping == target.grouping_combination[1]].identifier.tolist()
                case Pairing.EARLIER:
                    ids1 = target.uses[target.uses.grouping == target.grouping_combination[0]].identifier.tolist()
                    ids2 = ids1
                case Pairing.LATER:
                    ids1 = target.uses[target.uses.grouping == target.grouping_combination[1]].identifier.tolist()
                    ids2 = ids1

        match sampling:
            case Sampling.annotated:
                return list(zip(ids1, ids2))
            case Sampling.all:
                return list(product(ids1, ids2))
            case Sampling.sampled:
                return [(np.random.choice(ids1, replace=params["replace"]), np.random.choice(ids2, replace=params["replace"])) for _ in range(params["n"])]
        
    def __split_annotated_uses(self, target: Target) -> tuple[list[UseID], list[UseID]]:
        judgments = pd.merge(target.judgments, target.uses, left_on="identifier1", right_on="identifier", how="left")
        judgments = pd.merge(judgments, target.uses, left_on="identifier2", right_on="identifier", how="left")

        pairing_to_grouping = {
            "COMPARE": target.grouping_combination,
            "LATER": (
                target.grouping_combination[1],
                target.grouping_combination[1],
            ),
            "EARLIER": (
                target.grouping_combination[0],
                target.grouping_combination[0],
            ),
        }

        judgments = judgments[
            (judgments.grouping_x == pairing_to_grouping[self.name][0]) & 
            (judgments.grouping_y == pairing_to_grouping[self.name][1])
        ]

        return (
            judgments.identifier1.tolist(),
            judgments.identifier2.tolist(),
        )

