# -*- coding: utf-8 -*-

import csv
import logging
import os
import zipfile
from pathlib import Path
from typing import Union

import pandas as pd
import requests
from pandas import DataFrame
from tqdm import tqdm

import src.lscd as lscd
import src.semantic_proximity as semantic_proximity
from src import utils
from src.config import Config, Task

log = logging.getLogger(f"{Path(__file__).name}:{__name__}")


class DataLoader:

    wug2url = dict(
        dwug_de="https://zenodo.org/record/5796871/files/dwug_de.zip",
        # dwug_de="https://zenodo.org/record/5543724/files/dwug_de.zip",
        dwug_la="https://zenodo.org/record/5255228/files/dwug_la.zip",
        dwug_en="https://zenodo.org/record/5796878/files/dwug_en.zip",
        dwug_sv="https://zenodo.org/record/5801358/files/dwug_sv.zip",
        dwug_es="https://zenodo.org/record/6433667/files/dwug_es.zip",
        discowug="https://zenodo.org/record/5791125/files/discowug.zip",
        refwug="https://zenodo.org/record/5791269/files/refwug.zip",
        diawug="https://zenodo.org/record/5791193/files/diawug.zip",
        surel="https://zenodo.org/record/5784569/files/surel.zip",
        durel="https://zenodo.org/record/5784453/files/durel.zip",
        dups_wug="https://zenodo.org/record/5500223/files/DUPS-WUG.zip",
    )

    def __init__(self, config: Config) -> None:

        self.config = config
        self.path = utils.path("wug") / config.dataset.name
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if not self.path.exists():
            self.download()
            self.unzip(self.path.parent)

        self.__csv_params = dict(
            delimiter="\t", encoding="utf8", quoting=csv.QUOTE_NONE
        )

        self._uses = None
        self._judgments = None
        self._clusters = None
        self._agreements = None
        self._lscd_labels = None

    @property
    def uses(self) -> DataFrame:
        if self._uses is None:
            self._uses = pd.concat(
                [
                    pd.read_csv(target / "uses.tsv", **self.__csv_params)
                    for target in (self.path / "data").iterdir()
                ]
            )
        return self._uses

    @property
    def judgments(self) -> DataFrame:
        if self._judgments is None:
            self._judgments = pd.concat(
                [
                    pd.read_csv(target / "judgments.tsv", **self.__csv_params)
                    for target in (self.path / "data").iterdir()
                ]
            )
        return self._judgments

    @property
    def clusters(self) -> DataFrame:
        if self._clusters is None:
            self._clusters = pd.concat(
                [
                    pd.read_csv(target, **self.__csv_params)
                    for target in (self.path / "clusters" / "opt").iterdir()
                ]
            )
        return self._clusters

    @property
    def agreements(self) -> DataFrame:
        if self._agreements is None:
            self._agreements = pd.read_csv(
                self.path / "stats" / "stats_agreement.tsv", **self.__csv_params
            )
        return self._agreements

    @property
    def lscd_labels(self) -> DataFrame:
        if self._lscd_labels is None:
            path = self.path / "stats" / "semeval" / "stats_groupings.tsv"
            if not path.exists():
                path = self.path / "stats" / "opt" / "stats_groupings.tsv"
            if not path.exists():
                path = self.path / "stats" / "stats_groupings.tsv"
            self._lscd_labels = pd.read_csv(path, delimiter="\t", encoding="utf8")
        return self._lscd_labels

    def load_dataset(self) -> Union[lscd.Dataset, semantic_proximity.Dataset]:
        if self.config.dataset.task is Task.LSCD:
            return lscd.Dataset(
                config=self.config,
                uses=self.uses,
                labels=self.lscd_labels,
                judgments=self.judgments,
                agreements=self.agreements,
            )
        elif self.config.dataset.task is Task.SEMANTIC_PROXIMITY:
            return semantic_proximity.Dataset()
        else:
            raise NotImplementedError

    def download(self) -> None:
        r = requests.get(self.wug2url[self.config.dataset.name.lower()], stream=True)
        filename = f"{self.config.dataset.name}.zip"

        with open(file=filename, mode="wb") as f:
            pbar = tqdm(
                desc=f"Downloading dataset '{self.config.dataset.name}'",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                total=int(r.headers["Content-Length"]),
                leave=False
            )
            pbar.clear()  # clear 0% info
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    pbar.update(len(chunk))
                    f.write(chunk)
            pbar.close()

    def unzip(self, output_dir: Path) -> None:
        filename = f"{self.config.dataset.name}.zip"
        with zipfile.ZipFile(file=filename) as z:
            z.extractall()
            for f in z.namelist():
                if not f.startswith(f"{self.config.dataset.name}/plots"):
                    z.extract(f, path=output_dir)
                    if f.endswith(".csv"):
                        f = str(output_dir / f)
                        f_new = f.replace(".csv", ".tsv")
                        os.rename(f, f_new)
        os.remove(filename)
