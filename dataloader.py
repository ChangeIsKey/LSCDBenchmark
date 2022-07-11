# -*- coding: utf-8 -*-

import os
import requests
import csv
import zipfile
import pandas as pd

from typing import List, Callable
from pathlib import Path
from pandas import DataFrame, Series
from tqdm import tqdm
from itertools import combinations
from dataclasses import dataclass, InitVar


@dataclass
class Target:
    lemma: str
    uses: DataFrame
    labels: DataFrame
    judgments: DataFrame
    preprocessing: InitVar[str | Callable[[str], str]]

    def __post_init__(self, preprocessing):
        self.preprocess(how=preprocessing)
        self.uses = self.uses[self.uses.columns[self.uses.columns.isin([
            "pos", "date", "grouping", "identifier", "description", "context",
            "indexes_target_token", "indexes_target_sentence", "lemma",
            "context_preprocessed"
        ])]]
        self.judgments = self.judgments[self.judgments.columns[self.judgments.columns != "lemma"]]

    def preprocess(self, how: str | Callable[[str], str] = None):
        if how is None:
            self.uses["context_preprocessed"] = self.uses.context
            return self.uses
        elif isinstance(how, str):
            match how:
                case "lemmatization":
                    self.uses["context_preprocessed"] = self.uses.context_lemmatized
                case "toklem":
                    raise NotImplementedError("toklem preprocessing not implemented")
                case default:
                    raise NotImplementedError(f"unknown preprocessing type '{default}'")
        elif isinstance(how, Callable):
            contexts = Series([how(row.context) for _, row in self.uses.iterrows()])
            assert contexts.apply(lambda txt: isinstance(txt, str)).all(), f"Invalid return type for preprocessing " \
                                                                           f"function '{how.__name__}' "
            self.uses["context_preprocessed"] = contexts
            return self.uses
        else:
            raise TypeError("preprocessing parameter type is invalid")

    def sample_pairs_uses(self, n: int):
        samples = []
        for _ in range(n):
            contexts = tuple(self.uses.groupby(self.uses.grouping).context_preprocessed.sample(n=1))
            samples.extend(combinations(contexts, r=2))
        return samples


# TODO ask nikolai how to avoid name space problems in pytorch
class DataLoader:
    PROJECT = Path(__file__).parent

    DATASETS = {
        "dwug_de": "https://zenodo.org/record/5796871/files/dwug_de.zip",
        "dwug_la": "https://zenodo.org/record/5255228/files/dwug_la.zip",
        "dwug_en": "https://zenodo.org/record/5796878/files/dwug_en.zip",
        "dwug_sv": "https://zenodo.org/record/5090648/files/dwug_sv.zip",
        "dwug_es": "https://zenodo.org/record/6433667/files/dwug_es.zip",
        "discowug": "https://zenodo.org/record/5791125/files/discowug.zip",
        "refwug": "https://zenodo.org/record/5791269/files/refwug.zip",
        "diawug": "https://zenodo.org/record/5791193/files/diawug.zip",
        "surel": "https://zenodo.org/record/5784569/files/surel.zip",
        "durel": "https://zenodo.org/record/5784453/files/durel.zip",
        "dups-wug": "https://zenodo.org/record/5500223/files/DUPS-WUG.zip"
        # "http://www.dianamccarthy.co.uk/downloads/WordMeaningAnno2012/cl-meaningincontext.tgz"
    }

    LABELS = {
        "change_graded": "graded_jsd",
        "change_binary": "binary_change",
        "change_binary_loss": "binary_loss",
        "change_binary_gain": "binary_gain",
        "COMPARE": "graded_compare"
    }

    def __init__(self, dataset: str, targets: List[str] = None,
                 preprocessing: str | Callable[[str], str] = None):

        self.dataset = dataset.lower()
        assert self.dataset in self.DATASETS.keys(), f"Unknown dataset. Should be one of {list(self.DATASETS.keys())}"

        self.lemmas = targets
        self.preprocessing = preprocessing
        self.data_path = self.PROJECT.joinpath("wug", dataset)

        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.data_path.exists() or len(os.listdir(self.data_path)) == 0:
            # TODO fix problem where the program tries to load data before everything is downloaded and unzipped
            self.download_dataset()
            self.unzip_dataset()

        self.stats = self.load_stats()

    def load_stats(self) -> DataFrame:
        path = self.data_path.joinpath("stats", "opt", "stats_groupings.tsv")
        if not path.exists():
            path = self.data_path.joinpath("stats", "stats_groupings.tsv")
        df = pd.read_csv(path, delimiter="\t", encoding="utf8")
        df.drop(columns=df.columns.difference({"lemma", "grouping", *self.LABELS.keys()}), inplace=True)
        return df

    def load_lemma(self, lemma: str) -> Target:
        uses = pd.read_csv(filepath_or_buffer=self.data_path.joinpath("data", lemma, "uses.tsv"),
                           delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf8')
        judgments = pd.read_csv(filepath_or_buffer=self.data_path.joinpath("data", lemma, "judgments.tsv"),
                                delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf8')
        labels = self.stats[self.stats.lemma == lemma] \
            .set_index("grouping") \
            .rename(columns={old: new for old, new in self.LABELS.items() if old in self.stats.columns})

        return Target(lemma=lemma, uses=uses, labels=labels, judgments=judgments, preprocessing=self.preprocessing)

    def load_dataset(self) -> List[Target]:
        if self.lemmas is None:
            return [self.load_lemma(lemma)
                    for lemma in os.listdir(str(self.data_path.joinpath("data")))]
        else:
            return [self.load_lemma(lemma)
                    for lemma in os.listdir(str(self.data_path.joinpath("data")))
                    if lemma in self.lemmas]

    def download_dataset(self) -> None:
        r = requests.get(self.DATASETS[self.dataset], stream=True)
        filename = f"{self.dataset}.zip"

        with open(file=filename, mode='wb') as f:
            pbar = tqdm(desc=f"Downloading dataset '{self.dataset}'", unit="B", unit_scale=True, unit_divisor=1024,
                        total=int(r.headers['Content-Length']))
            pbar.clear()  # clear 0% info
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    pbar.update(len(chunk))
                    f.write(chunk)
            pbar.close()

    def unzip_dataset(self) -> None:
        filename = f"{self.dataset}.zip"
        with zipfile.ZipFile(file=filename) as z:
            for f in z.namelist():
                if not f.startswith(f"{self.dataset}/plots"):
                    z.extract(f, path=self.data_path.parent)
                    if f.endswith(".csv"):
                        f = str(Path(self.data_path.parent).joinpath(f))
                        f_new = f.replace(".csv", ".tsv")
                        os.rename(f, f_new)
        os.remove(filename)


if __name__ == "__main__":
    dataloader = DataLoader(dataset="dwug_es", preprocessing=None)
    target = dataloader.load_lemma("abundar")
