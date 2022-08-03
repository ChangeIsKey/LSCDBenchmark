# -*- coding: utf-8 -*-

import csv
import logging
import os
from typing import Union
import zipfile
from pathlib import Path

import pandas as pd
import requests
from pandas import DataFrame
from tqdm import tqdm

import src.lscd as lscd
import src.semantic_proximity as semantic_proximity
from src.config import Config

log = logging.getLogger(f"{Path(__file__).name}:{__name__}")



class DataLoader:

    wug2url = dict(
        dwug_de="https://zenodo.org/record/5796871/files/dwug_de.zip",
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

        params = dict(delimiter="\t", encoding="utf8", quoting=csv.QUOTE_NONE)

        self.config = config

        data_path = Path(__file__).parent.parent.joinpath("wug", config.dataset.name)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        if not data_path.exists():
            self.download()
            self.unzip(data_path.parent)

        self.uses = pd.concat([pd.read_csv(target.joinpath("uses.tsv"), **params)
                               for target in data_path.joinpath("data").iterdir()])
        self.judgments = pd.concat([pd.read_csv(target.joinpath("judgments.tsv"), **params)
                                    for target in data_path.joinpath("data").iterdir()])
        self.clusters = pd.concat([pd.read_csv(target, **params)
                                   for target in data_path.joinpath("clusters", "opt").iterdir()])
        self.agreements = pd.read_csv(data_path.joinpath("stats", "stats_agreement.tsv"), **params)
        self.lscd_labels = self.load_lscd_labels(path=data_path)

    @staticmethod
    def load_lscd_labels(path: Path) -> DataFrame:
        stats_path = path.joinpath("stats", "opt", "stats_groupings.tsv")
        if not stats_path.exists():
            stats_path = path.joinpath("stats", "stats_groupings.tsv")
        df = pd.read_csv(stats_path, delimiter="\t", encoding="utf8")
        return df

    def load_dataset(self, task: str) -> Union[lscd.Dataset, semantic_proximity.Dataset]:
        if task.lower() == "lscd":
            return lscd.Dataset(config=self.config, uses=self.uses, labels=self.lscd_labels, judgments=self.judgments, agreements=self.agreements)
        elif task.lower() == "semantic_proximity":
            return semantic_proximity.Dataset()
        else:
            raise NotImplementedError

    def download(self) -> None:
        r = requests.get(self.wug2url[self.config.dataset.name.lower()], stream=True)
        filename = f"{self.config.dataset.name}.zip"

        with open(file=filename, mode='wb') as f:
            pbar = tqdm(desc=f"Downloading dataset '{self.config.dataset.name}'", unit="B", unit_scale=True,
                        unit_divisor=1024,
                        total=int(r.headers['Content-Length']))
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
                        f = str(Path(output_dir).joinpath(f))
                        f_new = f.replace(".csv", ".tsv")
                        os.rename(f, f_new)
        os.remove(filename)
