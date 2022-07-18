import csv
import re

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, product
from typing import Callable, List, Tuple, Set, Dict
from thefuzz import process, fuzz

import numpy as np
import pandas as pd
import logging

from spacy.lang.es import Spanish
from pathlib import Path

from pandas import DataFrame, Series
from torch import Tensor

import preprocessing
from config import DatasetConfig, UsesType, Pairing, Task, Uses, Preprocessing
from utils import vectorize

log = logging.getLogger(f"{Path(__file__).name}:{__name__}")


nlp = Spanish()
tokenizer = nlp.tokenizer


@dataclass
class Use:
    identifier: str
    context_preprocessed: str
    target_index_begin: int
    target_index_end: int

    # grouping: str
    # date: int
    # target_sentence_index_begin: int
    # target_sentence_index_end: int

    def vectorize(self, embedding: str, language: str = None, cased: bool = True):
        return vectorize(contexts=[self.context_preprocessed],
                         target_indices=[(self.target_index_begin, self.target_index_end)],
                         embedding=embedding, cased=cased)


class DataLoader:
    def __init__(self, name: str, language: str, grouping_1: int, grouping_2: int, pairing: Pairing, uses: Uses, task: Task, preprocessing: Preprocessing):
        self.config = DatasetConfig(name, language, grouping_1, grouping_2, pairing, uses, task, preprocessing)
        data_path = Path(__file__).parent.joinpath("wug", self.config.name)

        self.uses = pd.concat(
            [pd.read_csv(target.joinpath("uses.tsv"), delimiter="\t", encoding="utf8", quoting=csv.QUOTE_NONE)
             for target in data_path.joinpath("data").iterdir()])
        self.judgments = pd.concat([pd.read_csv(target.joinpath("judgments.tsv"), delimiter="\t", encoding="utf8")
                                    for target in data_path.joinpath("data").iterdir()])
        self.clusters = pd.concat([pd.read_csv(target, delimiter="\t", encoding="utf8", quoting=csv.QUOTE_NONE)
                                   for target in data_path.joinpath("clusters", "opt").iterdir()])
        self.lscd_labels = self.load_lscd_labels(path=data_path)

    @staticmethod
    def load_lscd_labels(path: Path) -> DataFrame:
        labels = {
            "change_graded": "graded_jsd",
            "change_binary": "binary_change",
            "change_binary_loss": "binary_loss",
            "change_binary_gain": "binary_gain",
            "COMPARE": "graded_compare"
        }
        stats_path = path.joinpath("stats", "opt", "stats_groupings.tsv")
        if not stats_path.exists():
            stats_path = path.joinpath("stats", "stats_groupings.tsv")
        df = pd.read_csv(stats_path, delimiter="\t", encoding="utf8")
        df.drop(columns=df.columns.difference({"lemma", "grouping", *labels.keys()}), inplace=True)
        df.rename(columns={old: new for old, new in labels.items() if old in df.columns}, inplace=True)

        return df

    def load_lscd_dataset(self):
        return LSCDDataset(config=self.config, uses=self.uses, labels=self.lscd_labels, judgments=self.judgments)


class LSCDDataset:
    def __init__(self, config: DatasetConfig, uses: DataFrame, labels: DataFrame, judgments: DataFrame):
        self.config = config
        self.grouping_1 = self.config.grouping_1
        self.grouping_2 = self.config.grouping_2
        self.target_names = [target.name for target in
                             Path(__file__).parent.joinpath("wug", config.name, "data").iterdir()]
        self.targets = self.make_targets(uses=uses, labels=labels, judgments=judgments, mask=False)
        self.name2target = {target.name: target for target in self.targets}

    def make_targets(self, uses: DataFrame, labels: DataFrame, judgments: DataFrame, mask: bool = False):
        return [
            LSCDTarget(
                config=self.config,
                name=target,
                grouping_1=self.grouping_1,
                grouping_2=self.grouping_2,
                uses_1=uses[(uses.lemma == target) & (uses.grouping == self.grouping_1)].copy(),
                uses_2=uses[(uses.lemma == target) & (uses.grouping == self.grouping_2)].copy(),
                judgments=judgments,
                labels=labels[(labels.lemma == target) & (
                        labels.grouping == f"{self.grouping_1}_{self.grouping_2}")] if not mask else None
            )
            for target in self.target_names]

    def get_use_id_pairs(self):
        pairs = []
        for target in self.targets:
            pairs.extend(target.get_use_id_pairs(filter=self.config.uses.type, replacement=self.config.uses.replacement,
                                                 n=self.config.uses.n, pairing=self.config.pairing))

    def get_uses(self):
        uses_1 = dict()
        uses_2 = dict()
        for target in self.targets:
            target_uses_1, target_uses_2 = target.get_uses()
            uses_1.update(target_uses_1)
            uses_2.update(target_uses_2)

        return uses_1, uses_2

    def vectorize(self, embedding: str, language: str, cased: bool):
        vectors_1 = dict()
        vectors_2 = dict()
        for target in self.targets:
            target_vectors_1, target_vectors_2 = target.vectorize(embedding, language, cased)
            vectors_1[target.name] = target_vectors_1
            vectors_2[target.name] = target_vectors_2
        return vectors_1, vectors_2


class LSCDTarget:
    def __init__(self, name: str, grouping_1: int, grouping_2: int, uses_1: DataFrame, uses_2: DataFrame,
                 labels: DataFrame, judgments: DataFrame, config: DatasetConfig, **kwargs):
        self.name = name
        self.uses_1 = uses_1
        self.uses_2 = uses_2
        self.labels = labels
        self.judgments = judgments
        self.grouping_1 = grouping_1
        self.grouping_2 = grouping_2
        self.config = config
        self.use_id_pairs, self.ids = self.get_use_id_pairs(filter=config.uses.type, pairing=config.pairing,
                                                            replacement=config.uses.replacement, n=config.uses.n)

        self.preprocess(how=getattr(preprocessing, self.config.preprocessing.method), **self.config.preprocessing.params)

    def preprocess(self, how: Callable[[Series], str] = None, **params) -> None:
        """
        :param how:
            A function to preprocess the contexts. This can be any function that takes a pandas Series (i.e., a row of
            one of the uses.csv files) as input, and possibly other parameters, and returns a string.
            The module preprocessing contains useful preprocessing functions.
        :param params: Extra keyword parameters for the preprocessing function
        :raises TypeError: if the 'how' parameter is neither None nor a function
        :raises TypeError: if the one of the returned outputs of the preprocessing function is not a string
        """

        assert how is None or callable(how), TypeError("preprocessing parameter type is invalid")

        self.uses_1["context_preprocessed"] = self.uses_1.context if how is None else self.uses_1.apply(how, axis=1, **params)
        self.uses_2["context_preprocessed"] = self.uses_2.context if how is None else self.uses_2.apply(how, axis=1, **params)
        self.uses_1["indexes_target_token_preprocessed"] = self.uses_1.apply(axis=1, func=lambda row: re.search(
            pattern=str(process.extractOne(self.name, tokenizer(row.context_preprocessed), scorer=fuzz.token_sort_ratio)[0]),
            string=row.context_preprocessed).span())
        self.uses_2["indexes_target_token_preprocessed"] = self.uses_2.apply(axis=1, func=lambda row: re.search(
            pattern=str(process.extractOne(self.name, tokenizer(row.context_preprocessed), scorer=fuzz.token_sort_ratio)[0]),
            string=row.context_preprocessed).span())

        # TODO set index attributes for target token in context_preprocessed

        if callable(how):
            assert self.uses_1.context_preprocessed.apply(isinstance, args=[str]).all(), \
                TypeError(f"Invalid return type for preprocessing function {how.__name__}")
            assert self.uses_2.context_preprocessed.apply(isinstance, args=[str]).all(), \
                TypeError(f"Invalid return type for preprocessing function {how.__name__}")

    def vectorize(self, embedding: str, language: str = None, cased: bool = True) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        uses_1, uses_2 = self.get_uses()
        vectors_1 = {identifier: use.vectorize(embedding, language, cased)
                     for identifier, uses in uses_1.items()
                     for use in uses}
        vectors_2 = {identifier: use.vectorize(embedding, language, cased)
                     for identifier, uses in uses_2.items()
                     for use in uses}

        return vectors_1, vectors_2

    def get_uses(self) -> Tuple[Dict[str, List[Use]], Dict[str, List[Use]]]:
        uses_1 = defaultdict(list)
        uses_2 = defaultdict(list)
        for _, use in self.uses_1.iterrows():
            use = Use(use.identifier, use.context_preprocessed, *use.indexes_target_token_preprocessed)
            uses_1[use.identifier].append(use)
        for _, use in self.uses_2.iterrows():
            use = Use(use.identifier, use.context_preprocessed, *use.indexes_target_token_preprocessed)
            uses_2[use.identifier].append(use)

        return dict(uses_1), dict(uses_2)

    def get_use_id_pairs(self, filter: UsesType, replacement: bool, n: int, pairing: Pairing) -> (List[Tuple[str]], Set[str]):
        ids_1 = self.uses_1.identifier.tolist()
        ids_2 = self.uses_2.identifier.tolist()

        match pairing:
            case pairing.COMPARE:
                pass
            case pairing.EARLIER:
                ids_2 = ids_1
            case pairing.LATER:
                ids_1 = ids_2

        if filter == UsesType.sample:
            pairs = []
            for _ in range(n):
                id_1 = np.random.choice(ids_1, replace=replacement)
                id_2 = np.random.choice(ids_2, replace=replacement)
                pairs.append((id_1, id_2))
        elif filter == UsesType.all:
            if pairing == "COMPARE":
                pairs = product(ids_1, ids_2)
            elif pairing in ["EARLIER", "LATER"]:
                pairs = combinations(ids_1, r=2)
            else:
                raise NotImplementedError
        elif filter == UsesType.annotated:
            ids_1 = self.judgments.identifier1
            ids_2 = self.judgments.identifier2
            pairs = list(zip(ids_1, ids_2))
        else:
            raise NotImplementedError

        sampled_ids = set()
        for id1, id2 in pairs:
            sampled_ids.add(id1)
            sampled_ids.add(id2)

        return pairs, sampled_ids