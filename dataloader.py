# -*- coding: utf-8 -*-

import os
import requests
import csv
import zipfile
import pandas as pd

from typing import List, Callable, Tuple
from pathlib import Path
from pandas import DataFrame, Series
from tqdm import tqdm
from itertools import combinations
from dataclasses import dataclass, InitVar
from enum import Enum

import preprocessing


# noinspection SpellCheckingInspection
class Dataset(Enum):
    dwug_de = "https://zenodo.org/record/5796871/files/dwug_de.zip"
    dwug_la = "https://zenodo.org/record/5255228/files/dwug_la.zip"
    dwug_en = "https://zenodo.org/record/5796878/files/dwug_en.zip"
    dwug_sv = "https://zenodo.org/record/5090648/files/dwug_sv.zip"
    dwug_es = "https://zenodo.org/record/6433667/files/dwug_es.zip"
    discowug = "https://zenodo.org/record/5791125/files/discowug.zip"
    refwug = "https://zenodo.org/record/5791269/files/refwug.zip"
    diawug = "https://zenodo.org/record/5791193/files/diawug.zip"
    surel = "https://zenodo.org/record/5784569/files/surel.zip"
    durel = "https://zenodo.org/record/5784453/files/durel.zip"
    DUPS_WUG = "https://zenodo.org/record/5500223/files/DUPS-WUG.zip"

    def download(self) -> None:
        r = requests.get(self.value, stream=True)
        filename = f"{self.name}.zip"

        with open(file=filename, mode='wb') as f:
            pbar = tqdm(desc=f"Downloading dataset '{self.name}'", unit="B", unit_scale=True,
                        unit_divisor=1024,
                        total=int(r.headers['Content-Length']))
            pbar.clear()  # clear 0% info
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    pbar.update(len(chunk))
                    f.write(chunk)
            pbar.close()

    def unzip(self, output_dir: Path) -> None:
        filename = f"{self.name}.zip"
        with zipfile.ZipFile(file=filename) as z:
            for f in z.namelist():
                if not f.startswith(f"{self.name}/plots"):
                    z.extract(f, path=output_dir)
                    if f.endswith(".csv"):
                        f = str(Path(output_dir).joinpath(f))
                        f_new = f.replace(".csv", ".tsv")
                        os.rename(f, f_new)
        os.remove(filename)


@dataclass
class Use:
    identifier: str
    context_preprocessed: str
    target_indices: Tuple[int]


@dataclass
class Target:
    lemma: str
    uses: DataFrame
    labels: DataFrame
    judgments: DataFrame
    preprocessing: InitVar[Callable[[Series], str]]

    def __post_init__(self, preprocessing):
        self.preprocess(how=preprocessing)
        self.uses = self.uses[self.uses.columns[self.uses.columns.isin([
            "pos", "date", "grouping", "identifier", "description", "context",
            "indexes_target_token", "indexes_target_sentence", "lemma",
            "context_preprocessed"
        ])]]
        self.judgments = self.judgments[self.judgments.columns[self.judgments.columns != "lemma"]]

    def preprocess(self, how: Callable[[Series], str] = None):
        assert how is None or callable(how), TypeError("preprocessing parameter type is invalid")

        if how is None:
            self.uses["context_preprocessed"] = self.uses.context
        elif callable(how):
            self.uses["context_preprocessed"] = self.uses.apply(how, axis=1)
            assert self.uses.context_preprocessed.apply(lambda txt: isinstance(txt, str)).all(), \
                f"Invalid return type for preprocessing function {how.__name__}"

    def get_uses(self) -> List[Use]:
        return self.uses.apply(lambda row: Use(row.identifier, row.context_preprocessed, row.indexes_target_token),
                               axis=1).tolist()

    def sample_pairs_use(self, n: int) -> (Tuple[str], List[str]):
        samples = []
        ids = self.uses.groupby(self.uses.grouping).identifier
        sampled_ids = []
        for _ in range(n):
            sample = ids.sample(n=1, replace=True)
            samples.extend(combinations(sample, r=2))
            sampled_ids.extend(sample)

        return samples, sampled_ids

    def get_pairs_use_judgments(self):
        pass


# TODO ask nikolai how to avoid name space problems in pytorch
class DataLoader:
    PROJECT = Path(__file__).parent

    LABELS = {
        "change_graded": "graded_jsd",
        "change_binary": "binary_change",
        "change_binary_loss": "binary_loss",
        "change_binary_gain": "binary_gain",
        "COMPARE": "graded_compare"
    }

    def __init__(self, dataset: Dataset, targets: List[str] = None,
                 preprocessing: Callable[[Series], str] = None):

        # self.dataset = dataset.lower()
        self.dataset = dataset
        self.targets = targets
        self.preprocessing = preprocessing
        self.data_path = self.PROJECT.joinpath("wug", self.dataset.name.replace("_", "-")
                                               if dataset == Dataset.DUPS_WUG else self.dataset.name)

        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.data_path.exists() or len(os.listdir(self.data_path)) == 0:
            self.dataset.download()
            self.dataset.unzip(output_dir=self.data_path.parent)

        self.stats = self.load_stats()

    def load_stats(self) -> DataFrame:
        path = self.data_path.joinpath("stats", "opt", "stats_groupings.tsv")
        if not path.exists():
            path = self.data_path.joinpath("stats", "stats_groupings.tsv")
        df = pd.read_csv(path, delimiter="\t", encoding="utf8")
        df.drop(columns=df.columns.difference({"lemma", "grouping", *self.LABELS.keys()}), inplace=True)
        df.set_index(["lemma", "grouping"], inplace=True)
        return df

    def load_lemma(self, lemma: str) -> Target:
        uses = pd.read_csv(filepath_or_buffer=self.data_path.joinpath("data", lemma, "uses.tsv"),
                           delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf8')
        judgments = pd.read_csv(filepath_or_buffer=self.data_path.joinpath("data", lemma, "judgments.tsv"),
                                delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf8')
        labels = self.stats.loc[lemma].rename(columns={old: new for old, new in self.LABELS.items()
                                                       if old in self.stats.columns})

        return Target(lemma=lemma, uses=uses, labels=labels, judgments=judgments, preprocessing=self.preprocessing)

    def load_dataset(self) -> List[Target]:
        if self.targets is None:
            return [self.load_lemma(lemma)
                    for lemma in os.listdir(str(self.data_path.joinpath("data")))]
        else:
            return [self.load_lemma(lemma)
                    for lemma in os.listdir(str(self.data_path.joinpath("data")))
                    if lemma in self.targets]


if __name__ == "__main__":
    dataloader = DataLoader(dataset=Dataset.dwug_es, preprocessing=preprocessing.lemmatize)
    target = dataloader.load_lemma("abundar")
    print(target.sample_pairs_uses(n=1))