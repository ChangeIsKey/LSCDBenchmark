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
    name: str
    uses: DataFrame
    labels: DataFrame
    judgments: DataFrame

    def __init__(self, name: str, uses: DataFrame, labels: DataFrame, judgments: DataFrame,
                 preprocessing: Callable[[Series], str], **kwargs):
        self.name = name
        self.uses = uses
        self.labels = labels
        self.judgments = judgments

        self.preprocess(how=preprocessing, **kwargs)

    def preprocess(self, how: Callable[[Series], str] = None, **kwargs) -> None:
        """
        :param how:
            A function to preprocess the contexts. This can be any function that takes a pandas Series (i.e., a row of
            one of the uses.csv files) as input, and possibly other parameters, and returns a string.
            The module preprocessing contains useful preprocessing functions.
        :param kwargs: Extra keyword parameters for the preprocessing function
        :raises TypeError: if the 'how' parameter is neither None nor a function
        :raises TypeError: if the one of the returned outputs of the preprocessing function is not a string
        """

        assert how is None or callable(how), TypeError("preprocessing parameter type is invalid")
        self.uses["context_preprocessed"] = self.uses.context if how is None else self.uses.apply(how, axis=1, **kwargs)

        if callable(how):
            assert self.uses.context_preprocessed.apply(isinstance, args=[str]).all(), \
                TypeError(f"Invalid return type for preprocessing function {how.__name__}")

    def get_uses(self) -> List[Use]:
        return self.uses.apply(lambda row: Use(row.identifier, row.context_preprocessed, row.indexes_target_token),
                               axis=1).tolist()

    def get_pairs_use_judgments(self):
        pass

    def sample_pairs_use(self, n: int) -> (Tuple[str], List[str]):
        samples = []
        ids = self.uses.groupby(self.uses.grouping).identifier
        sampled_ids = []
        for _ in range(n):
            sample = ids.sample(n=1, replace=True)
            samples.extend(combinations(sample, r=2))
            sampled_ids.extend(sample)

        return samples, sampled_ids

    def sample_judgments(self):
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

    def __init__(self, dataset: Dataset, targets: List[str] = None):

        self.dataset = dataset
        self.targets = targets
        dataset_basename = self.dataset.name.replace("_", "-") if dataset == Dataset.DUPS_WUG else self.dataset.name
        self.data_path = self.PROJECT.joinpath("wug", dataset_basename)

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

    def load_target(self, target: str, preprocessing: Callable[[Series], str], **kwargs) -> Target:
        """
        Load a word as a Target instance

        :param target: The target word to load
        :param preprocessing:
            A function to preprocess the contexts. This can be any function that takes a pandas Series (i.e., a row of
            one of the uses.csv files) as input, and possibly other parameters, and returns a string.
            The module preprocessing contains useful preprocessing functions.
        :param kwargs: Extra keyword parameters for the preprocessing function.
        :return: A Target instance.
        """
        uses = pd.read_csv(filepath_or_buffer=self.data_path.joinpath("data", target, "uses.tsv"),
                           delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf8')
        judgments = pd.read_csv(filepath_or_buffer=self.data_path.joinpath("data", target, "judgments.tsv"),
                                delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf8')
        labels = self.stats.loc[target].rename(columns={old: new for old, new in self.LABELS.items()
                                                        if old in self.stats.columns})

        return Target(name=target, uses=uses, labels=labels, judgments=judgments, preprocessing=preprocessing, **kwargs)

    def load_dataset(self, preprocessing: Callable[[Series], str], **kwargs) -> List[Target]:
        """
        Loads the specified dataset

        :param preprocessing:
            A function to preprocess the contexts. This can be any function that takes a pandas Series (i.e., a row of
            one of the uses.csv files) as input, and possibly other parameters, and returns a string.
            The module preprocessing contains useful preprocessing functions.
        :param kwargs: Extra keyword parameters for the preprocessing function.
        :return: a list of Target instances.
        """

        if self.targets is None:
            return [self.load_target(target, preprocessing=preprocessing, **kwargs)
                    for target in os.listdir(str(self.data_path.joinpath("data")))]
        else:
            return [self.load_target(target, preprocessing=preprocessing, **kwargs)
                    for target in os.listdir(str(self.data_path.joinpath("data")))
                    if target in self.targets]


if __name__ == "__main__":
    dataloader = DataLoader(dataset=Dataset.dwug_es)
    target = dataloader.load_target("abundar", preprocessing=preprocessing.tokenize)
    print(target.get_uses())
