import csv
from doctest import UnexpectedException
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Dict, Literal, TypedDict

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandera import Column, DataFrameSchema
from pydantic import BaseModel, PrivateAttr, validate_arguments

from src.preprocessing import ContextPreprocessor
from src.use import Use, UseID
from src.utils import ShouldNotHappen


class CsvParams(TypedDict):
    delimiter: str
    encoding: str
    quoting: Literal[0, 1, 2, 3]


class Target(BaseModel):
    name: str
    groupings: tuple[str, str]
    path: Path
    preprocessing: ContextPreprocessor

    _uses: DataFrame | None = PrivateAttr(default=None)
    _judgments: DataFrame | None = PrivateAttr(default=None)
    _augmented_judgments: DataFrame | None = PrivateAttr(default=None)
    _clusters: DataFrame | None = PrivateAttr(default=None)
    _csv_params: CsvParams = PrivateAttr(default_factory=lambda: CsvParams(
        delimiter="\t",
        encoding="utf8",
        quoting=csv.QUOTE_NONE
    ))

    @property
    def uses(self) -> DataFrame:
        if self._uses is None:
            # load uses
            path = self.path / "data" / self.name / "uses.csv"
            self._uses = pd.read_csv(path, **self._csv_params)
            # filter by grouping
            self._uses.grouping = self._uses.grouping.astype(str)
            self._uses = self._uses[
                self._uses.grouping.isin(self.groupings)
            ]
            # preprocess uses
            self._uses = pd.concat(
                [
                    self._uses,
                    self._uses.apply(self.preprocessing.__call__, axis=1),
                ],
                axis=1,
            )
            self._uses = self.__uses_schema.validate(self._uses)
        return self._uses

    @property
    def __uses_schema(self) -> DataFrameSchema:
        return DataFrameSchema(
            {
                "identifier": Column(dtype=str, unique=True),
                "grouping": Column(dtype=str),
                "context_preprocessed": Column(str),
                "target_index_begin": Column(int),
                "target_index_end": Column(int),
            }
        )

    @property
    def judgments(self) -> DataFrame:
        if self._judgments is None:
            path = self.path / "data" / self.name / "judgments.csv"
            self._judgments = pd.read_csv(path, **self._csv_params)

            self._judgments["judgment"] = self._judgments["judgment"].replace(to_replace=0, value=np.nan)
            self.judgments_schema.validate(self._judgments)
        return self._judgments
    
    @property
    def augmented_judgments(self) -> DataFrame:
        if self._augmented_judgments is None:
            self._augmented_judgments = pd.merge(
                self.judgments,
                self.uses,
                left_on="identifier1",
                right_on="identifier",
                how="left",
            )
            self._augmented_judgments = pd.merge(
                self._augmented_judgments,
                self.uses,
                left_on="identifier2",
                right_on="identifier",
                how="left",
            )
        return self._augmented_judgments

    @property
    def judgments_schema(self) -> DataFrameSchema:
        return DataFrameSchema(
            {
                "identifier1": Column(dtype=str),
                "identifier2": Column(dtype=str),
                "judgment": Column(dtype=float, nullable=True),
            }
        )

    @property
    def clusters(self) -> DataFrame:
        if self._clusters is None:
            path = self.path / "clusters" / "opt" / f"{self.name}.csv"
            self._clusters = pd.read_csv(path, **self._csv_params)
            self._clusters = self.clusters_schema.validate(self._clusters)
        return self._clusters

    @property
    def clusters_schema(self) -> DataFrameSchema:
        return DataFrameSchema(
            {"identifier": Column(dtype=str, unique=True), "cluster": Column(int)}
        )

    def uses_to_grouping(self) -> Dict[UseID, str]:
        uses_to_grouping = dict(zip(self.uses.identifier, self.uses.grouping))
        return {
            identifier: value
            for identifier, value in uses_to_grouping.items()
            if value in self.groupings
        }

    def grouping_to_uses(self) -> dict[str, list[UseID]]:
        uses_to_groupings = self.uses_to_grouping()
        return {
            group: [
                id_ for id_, grouping in uses_to_groupings.items() if grouping == group
            ]
            for group in self.groupings
        }


    def _split_compare_uses(self) -> tuple[list[UseID], list[UseID]]:
        ids1 = self.uses[self.uses.grouping == self.groupings[0]]
        ids2 = self.uses[self.uses.grouping == self.groupings[1]]
        return ids1.identifier.tolist(), ids2.identifier.tolist()

    def _split_earlier_uses(self) -> tuple[list[UseID], list[UseID]]:
        ids = self.uses[self.uses.grouping == self.groupings[0]]
        return ids.identifier.tolist(), ids.identifier.tolist()

    def _split_later_uses(self) -> tuple[list[UseID], list[UseID]]:
        ids = self.uses[self.uses.grouping == self.groupings[1]]
        return ids.identifier.tolist(), ids.identifier.tolist()

    def split_uses(self, pairing: Literal["COMPARE", "EARLIER", "LATER"]) -> tuple[list[UseID], list[UseID]]:
        match pairing:
            case "COMPARE":
                return self._split_compare_uses()
            case "EARLIER":
                return self._split_earlier_uses()
            case "LATER":
                return self._split_later_uses()

    @validate_arguments
    def use_pairs(
        self, pairing: Literal["COMPARE", "EARLIER", "LATER"], 
        sampling: Literal["all", "sampled", "annotated"], 
        n: int | None = None,
        replace: bool | None = None
    ) -> list[tuple[Use, Use]]:
        
        match (sampling, pairing):
            case ("annotated", p):
                ids1, ids2 = self._split_annotated_uses(p)
                use_pairs = list(zip(ids1, ids2)) 
            case ("all", p):
                ids1, ids2 = self.split_uses(p)
                use_pairs = list(product(ids1, ids2))
            case ("sampled", p):
                if replace is None: 
                    raise ValueError("'replace' parameter not provided for sampling")
                if n is None: 
                    raise ValueError("'n' parameter not provided for sampling")

                ids1, ids2 = self.split_uses(p)
                ids1 = [np.random.choice(ids1, replace=replace) for _ in range(n)]
                ids2 = [np.random.choice(ids2, replace=replace) for _ in range(n)]
                use_pairs = list(zip(ids1, ids2))
            
            case _:
                raise ShouldNotHappen

        use_pairs_instances = []
        for id1, id2 in use_pairs:
            u1 = Use.from_series(self.uses[self.uses.identifier == id1].iloc[0])
            u2 = Use.from_series(self.uses[self.uses.identifier == id2].iloc[0])
            use_pairs_instances.append((u1, u2))

        return use_pairs_instances

    def _split_annotated_uses(
        self, 
        pairing: Literal["COMPARE", "EARLIER", "LATER"], 
    ) -> tuple[list[UseID], list[UseID]]:
        match pairing:
            case "COMPARE":
                group_0, group_1 = self.groupings
            case "EARLIER":
                group_0, group_1 = self.groupings[0], self.groupings[0]
            case "LATER":
                group_0, group_1 = self.groupings[1], self.groupings[1]

        judgments = self.augmented_judgments[
            (self.augmented_judgments.grouping_x == group_0)
            & (self.augmented_judgments.grouping_y == group_1)
        ]

        return (
            judgments.identifier1.tolist(),
            judgments.identifier2.tolist(),
        )
