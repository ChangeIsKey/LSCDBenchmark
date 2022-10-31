import csv
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import (
    Dict,
    Literal,
    TypedDict,
)

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandera import (
    Column,
    DataFrameSchema,
)
from pydantic import (
    BaseModel,
    PrivateAttr,
    validate_arguments,
)

from src.preprocessing import ContextPreprocessor
from src.use import (
    Use,
    UseID,
)
from src.utils.utils import ShouldNotHappen


class CsvParams(TypedDict):
    delimiter: str
    encoding: str
    quoting: Literal[0, 1, 2, 3]


class Lemma(BaseModel):
    name: str
    groupings: tuple[str, str]
    path: Path
    preprocessing: ContextPreprocessor

    _uses_df: DataFrame | None = PrivateAttr(default=None)
    _judgments_df: DataFrame | None = PrivateAttr(default=None)
    _augmented_judgments_df: DataFrame | None = PrivateAttr(default=None)
    _clusters_df: DataFrame | None = PrivateAttr(default=None)
    _csv_params: CsvParams = PrivateAttr(
        default_factory=lambda: CsvParams(
            delimiter="\t", encoding="utf8", quoting=csv.QUOTE_NONE
        )
    )

    @property
    def uses_df(self) -> DataFrame:
        if self._uses_df is None:
            # load uses
            path = self.path / "data" / self.name / "uses.csv"
            self._uses_df = pd.read_csv(path, **self._csv_params)
            # filter by grouping
            self._uses_df.grouping = self._uses_df.grouping.astype(str)
            self._uses_df = self._uses_df[self._uses_df.grouping.isin(self.groupings)]
            # preprocess uses
            self._uses_df = pd.concat(
                [
                    self._uses_df,
                    self._uses_df.apply(self.preprocessing.__call__, axis=1),
                ],
                axis=1,
            )
            self._uses_df = self.__uses_schema.validate(self._uses_df)
        return self._uses_df

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
    def judgments_df(self) -> DataFrame:
        if self._judgments_df is None:
            path = self.path / "data" / self.name / "judgments.csv"
            self._judgments_df = pd.read_csv(path, **self._csv_params)
            self.judgments_schema.validate(self._judgments_df)
        return self._judgments_df

    @property
    def augmented_judgments_df(self) -> DataFrame:
        if self._augmented_judgments_df is None:
            self._augmented_judgments_df = pd.merge(
                self.judgments_df,
                self.uses_df,
                left_on="identifier1",
                right_on="identifier",
                how="left",
            )
            self._augmented_judgments_df = pd.merge(
                self._augmented_judgments_df,
                self.uses_df,
                left_on="identifier2",
                right_on="identifier",
                how="left",
            )
        return self._augmented_judgments_df

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
    def clusters_df(self) -> DataFrame:
        if self._clusters_df is None:
            path = self.path / "clusters" / "opt" / f"{self.name}.csv"
            self._clusters_df = pd.read_csv(path, **self._csv_params)
            self._clusters_df = self.clusters_schema.validate(self._clusters_df)
        return self._clusters_df

    @property
    def clusters_schema(self) -> DataFrameSchema:
        return DataFrameSchema(
            {"identifier": Column(dtype=str, unique=True), "cluster": Column(int)}
        )

    def useid_to_grouping(self) -> Dict[UseID, str]:
        return dict(zip(self.uses_df.identifier, self.uses_df.grouping))

    def grouping_to_useid(self) -> dict[str, list[UseID]]:
        grouping_to_useid = defaultdict(list)
        for useid, grouping in self.useid_to_grouping().items():
            grouping_to_useid[grouping].append(useid)
        return dict(grouping_to_useid)

    def _split_compare_uses(self) -> tuple[list[UseID], list[UseID]]:
        ids1 = self.uses_df[self.uses_df.grouping == self.groupings[0]]
        ids2 = self.uses_df[self.uses_df.grouping == self.groupings[1]]
        return ids1.identifier.tolist(), ids2.identifier.tolist()

    def _split_earlier_uses(self) -> tuple[list[UseID], list[UseID]]:
        ids = self.uses_df[self.uses_df.grouping == self.groupings[0]]
        return ids.identifier.tolist(), ids.identifier.tolist()

    def _split_later_uses(self) -> tuple[list[UseID], list[UseID]]:
        ids = self.uses_df[self.uses_df.grouping == self.groupings[1]]
        return ids.identifier.tolist(), ids.identifier.tolist()

    def split_uses(
        self, pairing: Literal["COMPARE", "EARLIER", "LATER"]
    ) -> tuple[list[UseID], list[UseID]]:
        match pairing:
            case "COMPARE":
                return self._split_compare_uses()
            case "EARLIER":
                return self._split_earlier_uses()
            case "LATER":
                return self._split_later_uses()

    def get_uses(self) -> list[Use]:
        return [Use.from_series(row) for _, row in self.uses_df.iterrows()]

    @validate_arguments
    def use_pairs(
        self,
        pairing: Literal["COMPARE", "EARLIER", "LATER"],
        sampling: Literal["all", "sampled", "annotated"],
        n: int | None = None,
        replace: bool | None = None,
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
            u1 = Use.from_series(self.uses_df[self.uses_df.identifier == id1].iloc[0])
            u2 = Use.from_series(self.uses_df[self.uses_df.identifier == id2].iloc[0])
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

        judgments = self.augmented_judgments_df[
            (self.augmented_judgments_df.grouping_x == group_0)
            & (self.augmented_judgments_df.grouping_y == group_1)
        ]

        return (
            judgments.identifier1.tolist(),
            judgments.identifier2.tolist(),
        )
