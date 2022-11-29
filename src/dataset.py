import csv
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Literal, TypedDict
import numpy as np
from git import Repo

import pandas as pd
import pandera as pa
import requests
from pandas import (
    DataFrame,
    Series,
)
from pandera import (
    Column,
    DataFrameSchema,
)
from pydantic import BaseModel, PrivateAttr, HttpUrl, Field
from tqdm import tqdm
from src.use import UseID

import src.utils.utils as utils
from src.cleaning import Cleaning
from src.evaluation import EvaluationTask
from src.preprocessing import ContextPreprocessor
from src.lemma import (
    CsvParams,
    Lemma,
    Pairing, 
    Sampling
)


class UnknownDataset(Exception):
    pass


class SplitSize(BaseModel):
    dev: float
    test: float


class StandardSplit(BaseModel):
    dev: list[str]
    test: list[str]


class Split(BaseModel):
    how: Literal["standard"]
    use: Literal["dev", "test"]


class RandomSplit(Split):
    how: Literal["random"]  # type: ignore
    use: Literal["dev", "test"]
    sizes: SplitSize


class NoSplit(BaseModel):
    how: Literal["no_split"]

class Version(BaseModel):
    url: HttpUrl | None = Field(default=None)
    path: Path

class Dataset(BaseModel):
    name: str
    groupings: tuple[str, str]
    cleaning: Cleaning | None
    preprocessing: ContextPreprocessor
    version: str
    split: Split | RandomSplit | NoSplit
    standard_split: StandardSplit
    exclude_annotators: list[str]
    test_on: set[str] | int | None
    pairing: list[Pairing] | None
    sampling: list[Sampling] | None
    versions: dict[str, Version]

    _stats_groupings: DataFrame = PrivateAttr(default=None)
    _uses: DataFrame = PrivateAttr(default=None)
    _judgments: DataFrame = PrivateAttr(default=None)
    _agreements: DataFrame = PrivateAttr(default=None)
    _clusters: DataFrame = PrivateAttr(default=None)
    _lemmas: list[Lemma] = PrivateAttr(default=None)
    _path: Path = PrivateAttr(default=None)
    _csv_params: CsvParams = PrivateAttr(default_factory=CsvParams)

    def __init__(self, **data):
        super().__init__(**data)
        if not self.path.exists():
            self.path.parent.parent.mkdir(parents=True, exist_ok=True)
            self.__download(path=self.data_dir / self.versions[self.version].path.parts[0])

    @property
    def data_dir(self) -> Path:
        root = os.getenv("DATA_DIR")
        if root is None:
            root = "wug"
        return utils.path(root)
        
    @property
    def path(self) -> Path:
        if self._path is None:
            self._path = self.data_dir / self.versions[self.version].path
        return self._path

    @property
    def __zipped_filename(self) -> Path:
        return utils.path(f"{self.name}-{self.version}.zip")

    def __download(self, path: Path) -> None:
        version = self.versions[self.version]
        assert version.url is not None

        if "github" in version.url and version.url.endswith(".git"):
            Repo.clone_from(version.url, to_path=path)
        else:
            r = requests.get(version.url, stream=True)
            with open(file=self.__zipped_filename, mode="wb") as f:
                pbar = tqdm(
                    desc=f"Downloading dataset '{self.name}' (version {self.version})",
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    total=int(r.headers["Content-Length"]),
                    leave=False,
                )
                pbar.clear()  # clear 0% info
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        pbar.update(len(chunk))
                        f.write(chunk)
                pbar.close()
            self.__unzip(path)

    def __unzip(self, output_dir: Path) -> None:
        zipped = self.__zipped_filename
        trans_table = {"ó": "ó", "á": "á", "é": "é", "ú": "ú"}

        with zipfile.ZipFile(file=zipped) as z:
            namelist = z.namelist()
            root = output_dir
            root.mkdir(parents=True, exist_ok=True)

            for filename in tqdm(
                namelist,
                desc=f"Unzipping dataset '{self.name}' (version {self.version})",
            ):

                filename_fixed = filename
                for k, v in trans_table.items():
                    filename_fixed = filename_fixed.replace(k, v)

                path = Path(filename_fixed)
                f_parts = list(path.parts)

                f_parts[f_parts.index(root.name)] = self.version
                target_path = root.joinpath(*f_parts)

                if not filename.endswith("/"):
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with target_path.open(mode="wb") as file_obj:
                        shutil.copyfileobj(z.open(filename, mode="r"), file_obj)
        zipped.unlink()

    @property
    def stats_groupings_df(self) -> DataFrame:
        if self._stats_groupings is None:
            stats_groupings = "stats_groupings.csv"
            path = self.path / "stats" / "semeval" / stats_groupings
            if not path.exists():
                path = self.path / "stats" / "opt" / stats_groupings
            if not path.exists():
                path = self.path / "stats" / stats_groupings
            self._stats_groupings = pd.read_csv(path, delimiter="\t", encoding="utf8", quoting=csv.QUOTE_NONE)
        return self._stats_groupings

    @stats_groupings_df.setter
    def stats_groupings_df(self, other: DataFrame) -> None:
        self._stats_groupings = other

    def get_stats_groupings_schema(
        self, evaluation_task: EvaluationTask
    ) -> DataFrameSchema:
        def validate_grouping(s: Series) -> bool:
            for _, item in s.items():
                parts = item.split("_")
                if len(parts) != 2:
                    return False
            return True

        schema = DataFrameSchema(
            {
                "lemma": Column(str),
                "grouping": Column(str, checks=pa.Check(check_fn=validate_grouping)),
            }
        )

        match evaluation_task:
            case "change_graded":
                return schema.add_columns({"change_graded": Column(float)})
            case "change_binary":
                return schema.add_columns({"change_binary": Column(int)})
            case "COMPARE":
                return schema.add_columns({"COMPARE": Column(float)})
            case _:
                return schema

    @property
    def graded_change_labels(self) -> dict[str, float]:
        stats_groupings = self.get_stats_groupings_schema("change_graded").validate(
            self.stats_groupings_df
        )
        return dict(zip(stats_groupings.lemma, stats_groupings.change_graded))

    @property
    def compare_labels(self) -> dict[str, float]:
        stats_groupings = self.get_stats_groupings_schema("COMPARE").validate(
            self.stats_groupings_df
        )
        return dict(zip(stats_groupings.lemma, stats_groupings.COMPARE))

    @property
    def binary_change_labels(self) -> dict[str, int]:
        stats_groupings = self.get_stats_groupings_schema("change_binary").validate(
            self.stats_groupings_df
        )
        return dict(zip(stats_groupings.lemma, stats_groupings.change_binary))

    @property
    def wic_labels(self) -> dict[tuple[UseID, UseID], float]:
        judgments = self.judgments_df[~self.judgments_df["annotator"].isin(self.exclude_annotators)].copy(deep=True)
        judgments.replace(to_replace=0, value=np.nan, inplace=True)
        # pandas.core.groupby.GroupBy.median ignores missing values -> no need for nanmedian
        judgments = judgments.groupby(by=["identifier1", "identifier2"])["judgment"].median().reset_index()
        annotated_pairs = zip(judgments.identifier1, judgments.identifier2)
        return dict(zip(list(annotated_pairs), judgments.judgment))

    @property
    def binary_wic_labels(self) -> dict[tuple[UseID, UseID], float]:
        labels = self.wic_labels
        return {use_pair: judgment for use_pair, judgment in labels.items() if judgment in [4.0, 1.0]}

    @property
    def wsi_labels(self) -> dict[str, int]:
        clusters = self.clusters_df.replace(-1, np.nan)
        return dict(zip(clusters.identifier, clusters.cluster))

    def get_labels(
        self, evaluation_task: EvaluationTask | None
    ) -> dict[Any, Any]:
        # the get_*_labels methods return dictionaries from targets, identifiers or tuples of identifiers to labels
        # to be able to return the correct subset, we need the `keys` parameter
        # this value should be a list returned by any of the models
        match evaluation_task:
            case None:
                return {}
            case "change_graded":
                return self.graded_change_labels
            case "change_binary":
                return self.binary_change_labels
            case "COMPARE":
                return self.compare_labels
            case "wic":
                return self.wic_labels
            case "binary_wic":
                return self.binary_wic_labels
            case "wsi":
                return self.wsi_labels
            case _:
                raise ValueError

    @property
    def stats_agreement_df(self) -> DataFrame:
        if self._agreements is None:
            path = self.path / "stats" / "stats_agreement.csv"
            self._agreements = pd.read_csv(path, delimiter="\t", encoding="utf8", quoting=csv.QUOTE_NONE)
        return self._agreements

    @property
    def uses_df(self):
        if self._uses is None:
            self._uses = pd.concat([target.uses_df for target in self.lemmas])
        return self._uses

    @property
    def judgments_df(self):
        if self._judgments is None:
            tables = []
            for lemma in self.lemmas:
                path = self.path / "data" / lemma.name / "judgments.csv"
                judgments = pd.read_csv(path, delimiter="\t", encoding="utf8", quoting=csv.QUOTE_NONE)
                judgments = self.judgments_schema.validate(judgments)
                tables.append(judgments)
            self._judgments = pd.concat(tables)
        return self._judgments

    @property
    def judgments_schema(self) -> DataFrameSchema:
        return DataFrameSchema(
            {
                "identifier1": Column(dtype=str),
                "identifier2": Column(dtype=str),
                "judgment": Column(dtype=float),
                "annotator": Column(dtype=str)
            }
        )

    @property
    def clusters_df(self):
        if self._clusters is None:
            tables = []
            for lemma in self.lemmas:
                path = self.path / "clusters" / "opt" / f"{lemma.name}.csv"
                clusters = pd.read_csv(path, delimiter="\t", encoding="utf8", quoting=csv.QUOTE_NONE)
                clusters = self.clusters_schema.validate(clusters)
                tables.append(clusters)
            self._clusters = pd.concat(tables)
        return self._clusters

    @property
    def clusters_schema(self) -> DataFrameSchema:
        return DataFrameSchema(
            {"identifier": Column(dtype=str, unique=True), "cluster": Column(int)}
        )

    def dev_test_split(self) -> set[str]:
        all_lemmas = sorted(
            [
                folder.name
                for folder in (self.path / "data").iterdir()
                if folder.is_dir()
            ]
        )
        match self.split.how:
            case "no_split":
                assert isinstance(self.split, NoSplit)
                return set(all_lemmas)
            case "random":
                assert isinstance(self.split, RandomSplit)
                dev = (
                    Series(all_lemmas)
                    .sample(frac=self.split.sizes.dev, replace=False)
                    .tolist()
                )
                match self.split.use:
                    case "dev":
                        return set(dev)
                    case "test":
                        return set(all_lemmas).difference(dev)
            case "standard":
                assert isinstance(self.split, Split)
                match self.split.use:
                    case "dev":
                        return set(self.standard_split.dev)
                    case "test":
                        return set(self.standard_split.test)
            case _:
                raise ValueError

    def filter_lemmas(self, lemmas: list[Lemma]) -> list[Lemma]:
        if utils.is_str_set(self.test_on):
            keep = self.test_on
        elif utils.is_int(self.test_on):
            keep = set([lemma.name for lemma in lemmas[: self.test_on]])
        else:
            keep = self.dev_test_split()
            if self.cleaning is not None and len(self.cleaning.stats) > 0:
                # remove "data=full" row
                agreements = self.stats_agreement_df.iloc[1:, :].copy()
                agreements = self.cleaning(agreements)
                keep = set(keep).intersection(agreements.data.unique().tolist())

        return [lemma for lemma in lemmas if lemma.name in keep]

    @property
    def lemmas(self) -> list[Lemma]:
        if self._lemmas is None:
            to_load = [
                folder.name
                for folder in (self.path / "data").iterdir()
                if folder.is_dir()
            ]
            self._lemmas = [
                Lemma(
                    name=target,
                    groupings=self.groupings,
                    path=self.path,
                    preprocessing=self.preprocessing,
                )
                for target in to_load
            ]

        return self._lemmas
