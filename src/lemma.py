import csv
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import (
    Dict,
    Literal,
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
    DirectoryPath
)

from src.preprocessing import ContextPreprocessor
from src.use import (
    Use,
    UseID,
)
from src.utils.utils import ShouldNotHappen, CsvParams

Pairing = Literal["COMPARE", "EARLIER", "LATER"]
Sampling = Literal["all", "sampled", "annotated", "predefined"]

class Lemma(BaseModel):
    """The Lemma class represents one lemma in a DWUG-
    like dataset (i.e., one of the words represented as folders in the `data` directory)

    :param groupings: The time periods to extract uses and use pairs from
    :type groupings: tuple[str, str]
    :param path: The path to the folder containing uses.csv for the desired lemma
    :type path: pathlib.Path
    :param preprocessing: The kind of preprocessing to apply to the context of each use of the lemma
    """    

    groupings: tuple[str, str]
    path: DirectoryPath
    preprocessing: ContextPreprocessor

    _uses_df: DataFrame = PrivateAttr(default=None)
    _annotated_pairs_df: DataFrame = PrivateAttr(default=None)
    _augmented_annotated_pairs: DataFrame = PrivateAttr(default=None)
    _predefined_use_pairs_df: DataFrame = PrivateAttr(default=None)
    _augmented_predefined_use_pairs_df: DataFrame = PrivateAttr(default=None)
    _clusters_df: DataFrame = PrivateAttr(default=None)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def uses_df(self) -> DataFrame:
        if self._uses_df is None:
            # load uses
            path = self.path / "uses.csv"
            self._uses_df = pd.read_csv(path, delimiter="\t", encoding="utf8", quoting=csv.QUOTE_NONE)
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
            self._uses_df = self.uses_schema.validate(self._uses_df)
        return self._uses_df

    @property
    def uses_schema(self) -> DataFrameSchema:
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
    def annotated_pairs_df(self) -> DataFrame:
        if self._annotated_pairs_df is None:
            path = self.path / "judgments.csv"
            self._annotated_pairs_df = pd.read_csv(path, delimiter="\t", encoding="utf8", quoting=csv.QUOTE_NONE, usecols=["identifier1", "identifier2"])
            self.annotated_pairs_schema.validate(self._annotated_pairs_df)
        return self._annotated_pairs_df

    @property
    def augmented_annotated_pairs_df(self) -> DataFrame:
        if self._augmented_annotated_pairs is None:
            self._augmented_annotated_pairs = pd.merge(
                self.annotated_pairs_df,
                self.uses_df,
                left_on="identifier1",
                right_on="identifier",
                how="left",
            )
            self._augmented_annotated_pairs = pd.merge(
                self._augmented_annotated_pairs,
                self.uses_df,
                left_on="identifier2",
                right_on="identifier",
                how="left",
            )
            drop_cols = [col for col in self._augmented_annotated_pairs.columns 
                         if col not in ["identifier1", "identifier2", "grouping_x", "grouping_y"]]
            self._augmented_annotated_pairs.drop(columns=drop_cols)

        return self._augmented_annotated_pairs

    @property
    def annotated_pairs_schema(self) -> DataFrameSchema:
        return DataFrameSchema(
            {
                "identifier1": Column(dtype=str),
                "identifier2": Column(dtype=str),
            }
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

    def split_uses(self, pairing: Pairing) -> tuple[list[UseID], list[UseID]]:
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
        pairing: Pairing,
        sampling: Sampling,
        n: int | None = None,
        replace: bool | None = None,
    ) -> list[tuple[Use, Use]]:

        match (sampling, pairing):
            case ("annotated", p):
                ids1, ids2 = self._split_augmented_uses(p, self.augmented_annotated_pairs_df)
                use_pairs = list(zip(ids1, ids2))
            case ("predefined", p):
                ids1, ids2 = self._split_augmented_uses(p, self.augmented_predefined_use_pairs_df)
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

    @property
    def predefined_use_pairs_df(self) -> DataFrame:
        if self._predefined_use_pairs_df is None:
            self._predefined_use_pairs_df = pd.read_csv(self.path / "predefined_use_pairs.csv", encoding="utf8", delimiter="\t", quoting=csv.QUOTE_NONE)
        return self._predefined_use_pairs_df
    
    @property
    def augmented_predefined_use_pairs_df(self) -> DataFrame:
        if self._augmented_predefined_use_pairs_df is None:
            self._augmented_predefined_use_pairs_df = self.predefined_use_pairs_df.merge(
                right=self.uses_df,
                left_on="identifier1",
                right_on="identifier",
                how="left",
            )
            self._augmented_predefined_use_pairs_df = self._augmented_predefined_use_pairs_df.merge(
                right=self.uses_df,
                left_on="identifier2",
                right_on="identifier",
                how="left",
            )

            drop_cols = [col for col in self._augmented_predefined_use_pairs_df.columns 
                        if col not in ["identifier1", "identifier2", "grouping_x", "grouping_y"]]
            self._augmented_predefined_use_pairs_df.drop(columns=drop_cols)
        return self._augmented_predefined_use_pairs_df
        
    def _split_augmented_uses(self, pairing: Literal["COMPARE", "EARLIER", "LATER"], augmented_uses: DataFrame) -> tuple[list[UseID], list[UseID]]:
        match pairing:
            case "COMPARE":
                group_0, group_1 = self.groupings
            case "EARLIER":
                group_0, group_1 = self.groupings[0], self.groupings[0]
            case "LATER":
                group_0, group_1 = self.groupings[1], self.groupings[1]

        filtered = augmented_uses[
            (augmented_uses.grouping_x == group_0)
            & (augmented_uses.grouping_y == group_1)
        ]

        return (
            filtered.identifier1.tolist(),
            filtered.identifier2.tolist(),
        )
