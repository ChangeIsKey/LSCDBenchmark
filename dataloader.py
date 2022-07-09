# -*- coding: utf-8 -*-

import os
import requests
import csv
import zipfile
import pandas as pd
from typing import Tuple, Dict, List
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass

PROJECT = Path(__file__).parent


@dataclass
class Use:
    identifier: str
    pos: str
    context: str
    grouping: int
    date: int
    description: str
    indexes_target_token: Tuple[int, int]
    indexes_target_sentence: Tuple[int, int]


@dataclass
class Judgement:
    identifier1: str
    identifier2: str
    annotator: str
    judgement: int
    comment: str


@dataclass
class Target:
    lemma: str
    uses: List[Use]
    labels: Dict[Tuple[int, int], Dict[str, float]]
    judgements: List[Judgement]


Dataset = Dict[str, Target]


# TODO ask nikolai how to avoid name space problems in pytorch
class DataLoader:

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

    column_mapping = [
        ("graded_jsd", "change_graded", float),
        ("binary_change", "change_binary", bool),
        ("binary_loss", "change_binary_loss", bool),
        ("binary_gain", "change_binary_gain", bool),
        ("graded_compare", "COMPARE", float)
    ]

    # TODO: PREPROCESSING (lemmatization, toklem)
    def __init__(self, dataset: str, group_comparisons: List[str] = "all", lemmas: List[str] = None, preprocessing=None):
        self.dataset = dataset.lower()
        assert self.dataset in self.DATASETS.keys(), f"Unknown dataset. Should be one of {list(self.DATASETS.keys())}"

        self.lemmas = lemmas
        self.preprocessing = preprocessing
        self.group_comparisons = group_comparisons
        self.data_path = PROJECT.joinpath("wug", dataset)

        self.data_path.mkdir(parents=True, exist_ok=True)
        if not self.data_path.exists() or len(os.listdir(self.data_path)) == 0:
            # TODO fix problem where the program tries to load data before everything is downloaded and unzipped
            self.download_dataset()
            self.unzip_dataset()

        self.stats = self.load_stats()

    def load_stats(self):
        path = self.data_path.joinpath("stats", "opt", "stats_groupings.tsv")
        if not path.exists():
            path = self.data_path.joinpath("stats", "stats_groupings.tsv")
        return pd.read_csv(path, delimiter="\t", encoding="utf8")

    def load_lemma(self, lemma: str) -> Target:
        uses = pd.read_csv(self.data_path.joinpath("data", lemma, "uses.tsv"), delimiter="\t", quoting=csv.QUOTE_NONE,
                           encoding='utf8')
        judgements = pd.read_csv(self.data_path.joinpath("data", lemma, "judgments.tsv"), delimiter="\t",
                                 quoting=csv.QUOTE_NONE,
                                 encoding='utf8')

        target = Target(
            lemma=lemma,
            uses=[],
            labels=defaultdict(dict),
            judgements=[]
        )

        target.uses.extend([
            Use(
                identifier=use.identifier,
                pos=use.pos,
                date=use.date,
                grouping=use.grouping,
                description=use.description,
                context=use.context,
                indexes_target_token=use.indexes_target_token,
                indexes_target_sentence=use.indexes_target_sentence
            )
            for _, use in uses.iterrows()
        ])

        target.judgements.extend([
            Judgement(
                identifier1=j.identifier1,
                identifier2=j.identifier2,
                annotator=j.annotator,
                judgement=j.judgment,
                comment=j.comment
            )
            for _, j in judgements.iterrows()
        ])

        groupings = self.stats[self.stats.lemma == lemma].grouping.unique()
        for g in groupings:
            if g in self.group_comparisons or self.group_comparisons == "all":
                row = self.stats[(self.stats.lemma == lemma) & (self.stats.grouping == g)]
                g = tuple(map(int, g.split("_")))
                for field, col, func in self.column_mapping:
                    if col in row.columns:
                        target.labels[g][field] = func(row[col].values)
        return target

    def load_dataset(self) -> Dataset:
        if self.lemmas is None:
            return {l: self.load_lemma(l) for l in os.listdir(str(self.data_path.joinpath("data")))}
        else:
            return {l: self.load_lemma(l) for l in os.listdir(str(self.data_path.joinpath("data"))) if l in self.lemmas}

    def download_dataset(self):
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

    def unzip_dataset(self):
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
    dataloader = DataLoader(dataset="diawug", lemmas=["chamaco_pibe_chico"], group_comparisons=["0_1", "0_3"])
    targets = dataloader.load_dataset()
    print(targets)
