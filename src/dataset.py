import csv
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

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
from pydantic import (
	BaseModel,
	PrivateAttr,
)
from tqdm import tqdm

import src.utils.utils as utils
from src.cleaning import Cleaning
from src.evaluation import EvaluationTask
from src.preprocessing import ContextPreprocessor
from src.target import (
	CsvParams,
	Target,
)


class UnknownDataset(Exception):
	pass


class Dataset(BaseModel):
	name: str
	groupings: tuple[str, str]
	cleaning: Cleaning | None
	preprocessing: ContextPreprocessor
	version: str
	test_on: list[str] | int | None

	_stats_groupings: DataFrame = PrivateAttr(default=None)
	_uses: DataFrame = PrivateAttr(default=None)
	_judgments: DataFrame = PrivateAttr(default=None)
	_agreements: DataFrame = PrivateAttr(default=None)
	_clusters: DataFrame = PrivateAttr(default=None)
	_targets: list[Target] = PrivateAttr(default=None)
	_path: Path = PrivateAttr(default=None)
	_csv_params: CsvParams = PrivateAttr(
		default_factory=lambda: CsvParams(
			delimiter="\t", encoding="utf8", quoting=csv.QUOTE_NONE
		)
	)

	def __init__(
		self,
		**data
	):
		super().__init__(**data)
		if not self.path.exists():
			if self.name not in self.wug_to_url:
				raise UnknownDataset
			self.path.parent.parent.mkdir(parents=True, exist_ok=True)
			self.__download()
			self.__unzip(self.path.parent.parent)

	@property
	def path(
		self
	) -> Path:
		if self._path is None:
			path = os.getenv("DATA_DIR")
			if path is None:
				path = "wug"
			self._path = utils.path(path) / self.name / self.version
		return self._path

	@property
	def wug_to_url(
		self
	) -> dict[str, dict[str, str]]:
		path = utils.path("datasets.json")
		with path.open(mode="r") as f:
			return json.load(f)

	@property
	def url(
		self
	) -> str:
		return self.wug_to_url[self.name][self.version]

	@property
	def __zipped_filename(
		self
	) -> Path:
		return utils.path(f"{self.name}-{self.version}.zip")

	def __download(
		self
	) -> None:
		r = requests.get(self.url, stream=True)
		with open(file=self.__zipped_filename, mode="wb") as f:
			pbar = tqdm(
				desc=f"Downloading dataset '{self.name}' (version {self.version})", unit="B", unit_scale=True,
				unit_divisor=1024, total=int(r.headers["Content-Length"]), leave=False, )
			pbar.clear()  # clear 0% info
			for chunk in r.iter_content(chunk_size=1024):
				if chunk:  # filter out keep-alive new chunks
					pbar.update(len(chunk))
					f.write(chunk)
			pbar.close()

	def __unzip(
		self,
		output_dir: Path
	) -> None:
		zipped = self.__zipped_filename
		trans_table = {"ó": "ó", "á": "á", "é": "é", "ú": "ú"}

		with zipfile.ZipFile(file=zipped) as z:
			namelist = z.namelist()
			root = output_dir / namelist[0]
			root.mkdir(parents=True, exist_ok=True)

			for filename in tqdm(
					namelist, desc=f"Unzipping dataset '{self.name}' (version {self.version})", ):

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
	def stats_groupings_df(
		self
	) -> DataFrame:
		if self._stats_groupings is None:
			stats_groupings = "stats_groupings.csv"
			path = self.path / "stats" / "semeval" / stats_groupings
			if not path.exists():
				path = self.path / "stats" / "opt" / stats_groupings
			if not path.exists():
				path = self.path / "stats" / stats_groupings
			self._stats_groupings = pd.read_csv(path, **self._csv_params)
		return self._stats_groupings

	@stats_groupings_df.setter
	def stats_groupings_df(
		self,
		other: DataFrame
	) -> None:
		self._stats_groupings = other

	def get_stats_groupings_schema(
		self,
		evaluation_task: EvaluationTask
	) -> DataFrameSchema:
		def validate_grouping(
			s: Series
		) -> bool:
			for _, item in s.items():
				parts = item.split("_")
				if len(parts) != 2:
					return False
			return True

		schema = DataFrameSchema(
			{
				"lemma": Column(str), "grouping": Column(str, checks=pa.Check(check_fn=validate_grouping)),
			}
		)

		match evaluation_task:
			case "change_graded":
				return schema.add_columns({"change_graded": Column(float)})
			case "change_binary":
				return schema.add_columns({"change_binary": Column(int)})
			case _:
				return schema

	@property
	def graded_change_labels(
		self
	) -> dict[str, float]:
		stats_groupings = self.get_stats_groupings_schema("change_graded").validate(
			self.stats_groupings_df
		)
		stats_groupings.sort_values(by="lemma", inplace=True)
		return dict(zip(stats_groupings.lemma, stats_groupings.change_graded))

	@property
	def binary_change_labels(
		self
	) -> dict[str, int]:
		stats_groupings = self.get_stats_groupings_schema("change_binary").validate(
			self.stats_groupings_df
		)
		stats_groupings.sort_values(by="lemma", inplace=True)
		return dict(zip(stats_groupings.lemma, stats_groupings.change_binary))

	@property
	def semantic_proximity_labels(
		self
	) -> dict[tuple[str, str], float]:
		self.judgments_df.sort_values(by=["identifier1", "identifier2"], inplace=True)
		annotated_pairs = list(
			zip(self.judgments_df.identifier1, self.judgments_df.identifier2)
		)
		return dict(zip(annotated_pairs, self.judgments_df.judgment))

	@property
	def wsi_labels(
		self
	) -> dict[str, int]:
		self.clusters_df.sort_values(by="identifier", inplace=True)
		return dict(zip(self.clusters_df.identifier, self.clusters_df.cluster))

	def get_labels(
		self,
		evaluation_task: EvaluationTask | None
	) -> dict[Any, float] | dict[Any, int]:
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
			case "semantic_proximity":
				return self.semantic_proximity_labels
			case "wsi":
				return self.wsi_labels
			case _:
				raise ValueError

	@property
	def stats_agreement_df(
		self
	) -> DataFrame:
		if self._agreements is None:
			path = self.path / "stats" / "stats_agreement.csv"
			self._agreements = pd.read_csv(path, **self._csv_params)
		return self._agreements

	@property
	def uses_df(
		self
	):
		if self._uses is None:
			self._uses = pd.concat([target.uses_df for target in self.targets])
		return self._uses

	@property
	def judgments_df(
		self
	):
		if self._judgments is None:
			self._judgments = pd.concat(
				[target.judgments_df for target in self.targets]
			)
		return self._judgments

	@property
	def clusters_df(
		self
	):
		if self._clusters is None:
			self._clusters = pd.concat([target.clusters_df for target in self.targets])
		return self._clusters

	@property
	def targets(
		self
	) -> list[Target]:
		if self._targets is None:
			if self.cleaning is not None and len(self.cleaning.stats) > 0:
				# remove "data=full" row
				agreements = self.stats_agreement_df.iloc[1:, :].copy()
				agreements = self.cleaning(agreements)
				to_load = agreements.data.unique().tolist()
			else:
				if utils.is_str_list(self.test_on):
					to_load = self.test_on
				else:
					to_load = sorted([folder.name for folder in (self.path / "data").iterdir()])
					if utils.is_int(self.test_on):
						to_load = to_load[: self.test_on]

			self._targets = [
				Target(
					name=target,
					groupings=self.groupings,
					path=self.path,
					preprocessing=self.preprocessing
				)
				for target in to_load
			]

		return self._targets
