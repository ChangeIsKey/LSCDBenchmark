# -*- coding: utf-8 -*-

import os
import requests
import csv
import zipfile
import pandas as pd

from typing import List, Callable, Tuple, Set, Dict
from pathlib import Path
from pandas import DataFrame, Series
from tqdm import tqdm
from itertools import combinations
from dataclasses import dataclass, InitVar
from enum import Enum

import preprocessing
#
from config import DatasetConfig


@dataclass
class Dataset:
   name: str
   uses: DataFrame
   judgments: DataFrame
   clusters: DataFrame
   lscd_labels: DataFrame


#    targets: List[Target]
#
#    # @staticmethod
#    # def from_config(config):
#    #     return DatasetConfig(config.name, config.grouping_1, config.grouping_2, config.uses, config.pairing)
#
   def __init__(self, config):
       data_path = Path(__file__).parent.joinpath("wug", config["name"])

       self.name = config["name"]
       self.uses = pd.concat([pd.read_csv(target.joinpath("uses.tsv"), delimiter="\t", encoding="utf8", quoting=csv.QUOTE_NONE)
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
       df.set_index(["lemma", "grouping"], inplace=True)
       df.rename(columns={old: new for old, new in labels.items() if old in df.columns}, inplace=True)

       return df

   def to_lscd_dataset(self, config) -> LSCDDataset:

       d = LSCDDataset(config)


class LSCDDataset:
    def __init__(self, config: DatasetConfig, uses: DataFrame, labels: DataFrame):
        data_path = Path(__file__).parent.joinpath("wug", self.name)
        self.config = config
        self.grouping_1 = self.config.grouping_1
        self.grouping_2 = self.config.grouping_2
        #                                    encoding="utf8", quoting=csv.QUOTE_NONE) for target in targets])
        self.uses_1 = uses[(uses.grouping == self.grouping_1)]
        self.uses_2 = uses[(uses.grouping == self.grouping_2)]
        self.labels = labels[(labels.grouping == f"{self.grouping_1}_{self.grouping_2}")]

        self.targets = [LSCDTarget(name=target, grouping_1=self.grouping_1, grouping_2=self.grouping_2,
                                   uses_1=self.uses_1[self.uses_1.lemma == target], uses_2=self.uses_2[self.uses_2.lemma == target], labels=labels[labels.lemma == target])
                        for target in uses.lemma.unique()]
        # self.grouping_1 = config["grouping_1"]
        # self.grouping_2 = config["grouping_2"]
        # self.uses = config["uses"]
        # self.pairing = config["pairing"]

    def load_target(self, target: str, preprocessing: Callable[[Series], str], **kwargs) -> LSCDTarget:
        # self.config["uses"]["type"]
        LSCDTarget()



#
class ProximityDataset:
    def __init__(self, config):
        super(ProximityDataset, self).__init__(config)
    # def download(self) -> None:
#         r = requests.get(self.value, stream=True)
#         filename = f"{self.name}.zip"
#
#         with open(file=filename, mode='wb') as f:
#             pbar = tqdm(desc=f"Downloading dataset '{self.name}'", unit="B", unit_scale=True,
#                         unit_divisor=1024,
#                         total=int(r.headers['Content-Length']))
#             pbar.clear()  # clear 0% info
#             for chunk in r.iter_content(chunk_size=1024):
#                 if chunk:  # filter out keep-alive new chunks
#                     pbar.update(len(chunk))
#                     f.write(chunk)
#             pbar.close()
#
#     def unzip(self, output_dir: Path) -> None:
#         filename = f"{self.name}.zip"
#         with zipfile.ZipFile(file=filename) as z:
#             for f in z.namelist():
#                 if not f.startswith(f"{self.name}/plots"):
#                     z.extract(f, path=output_dir)
#                     if f.endswith(".csv"):
#                         f = str(Path(output_dir).joinpath(f))
#                         f_new = f.replace(".csv", ".tsv")
#                         os.rename(f, f_new)
#         os.remove(filename)
#

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


class LSCDTarget:
    def __init__(self, name: str, grouping_1: int, grouping_2: int, uses_1: DataFrame, uses_2: DataFrame, labels: DataFrame):
        self.name = name
        self.uses_1 = uses_1
        self.uses_2 = uses_2
        self.labels = labels
        self.grouping_1 = grouping_1
        self.grouping_2 = grouping_2

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
        self.uses_1["context_preprocessed"] = self.uses_1.context if how is None else self.uses_1.apply(how, axis=1, **kwargs)
        self.uses_2["context_preprocessed"] = self.uses_2.context if how is None else self.uses_2.apply(how, axis=1, **kwargs)

        if callable(how):
            assert self.uses_1.context_preprocessed.apply(isinstance, args=[str]).all(), \
                TypeError(f"Invalid return type for preprocessing function {how.__name__}")
            assert self.uses_2.context_preprocessed.apply(isinstance, args=[str]).all(), \
                TypeError(f"Invalid return type for preprocessing function {how.__name__}")

    def get_pairs_use_judgments(self):
        pass

    def get_uses(self) -> (List[Use], List[Use]):
        pass

    def get_use_pairs(self, sample: bool, replacement: bool = False, n: int = 100) -> (Tuple[str], Set[str]):
        ids_1 = self.uses_1.identifier.tolist()
        ids_2 = self.uses_2.identifier.tolist()

        if not sample:
            samples = []
            sampled_ids = []
            for _ in range(n):
                sample = ids.sample(n=1, replace=True)
                samples.extend(combinations(sample, r=2))
                sampled_ids.extend(sample)

        return samples, set(sampled_ids)

    def sample_judgments(self):
        pass

#
# class DataLoader:
#     PROJECT = Path(__file__).parent
#
#     LABELS = {
#         "change_graded": "graded_jsd",
#         "change_binary": "binary_change",
#         "change_binary_loss": "binary_loss",
#         "change_binary_gain": "binary_gain",
#         "COMPARE": "graded_compare"
#     }
#
#     def __init__(self, dataset: DatasetConfig, targets: List[str] = None):
#
#         self.dataset = dataset
#         self.targets = targets
#         dataset_basename = self.dataset.name.replace("_", "-") if dataset == DatasetConfig.DUPS_WUG else self.dataset.name
#         self.data_path = self.PROJECT.joinpath("wug", dataset_basename)
#
#         self.data_path.parent.mkdir(parents=True, exist_ok=True)
#         if not self.data_path.exists() or len(os.listdir(self.data_path)) == 0:
#             self.dataset.download()
#             self.dataset.unzip(output_dir=self.data_path.parent)
#
#         self.stats = self.load_stats()
#
#     def load_stats(self) -> DataFrame:
#         path = self.data_path.joinpath("stats", "opt", "stats_groupings.tsv")
#         if not path.exists():
#             path = self.data_path.joinpath("stats", "stats_groupings.tsv")
#         df = pd.read_csv(path, delimiter="\t", encoding="utf8")
#         df.drop(columns=df.columns.difference({"lemma", "grouping", *self.LABELS.keys()}), inplace=True)
#         df.set_index(["lemma", "grouping"], inplace=True)
#         return df
#
#     def load_target(self, target: str, preprocessing: Callable[[Series], str], **kwargs) -> Target:
#         """
#         Load a word as a Target instance
#
#         :param target: The target word to load
#         :param preprocessing:
#             A function to preprocess the contexts. This can be any function that takes a pandas Series (i.e., a row of
#             one of the uses.csv files) as input, and possibly other parameters, and returns a string.
#             The module preprocessing contains useful preprocessing functions.
#         :param kwargs: Extra keyword parameters for the preprocessing function.
#         :return: A Target instance.
#         """
#         uses = pd.read_csv(filepath_or_buffer=self.data_path.joinpath("data", target, "uses.tsv"),
#                            delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf8')
#         judgments = pd.read_csv(filepath_or_buffer=self.data_path.joinpath("data", target, "judgments.tsv"),
#                                 delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf8')
#         labels = self.stats.loc[target].rename(columns={old: new for old, new in self.LABELS.items()
#                                                         if old in self.stats.columns})
#         return Target(name=target, uses=uses, labels=labels, judgments=judgments, preprocessing=preprocessing, **kwargs)
#
#     def load_dataset(self, preprocessing: Callable[[Series], str], **kwargs) -> DatasetConfig:
#         """
#         Loads the specified dataset
#
#         :param preprocessing:
#             A function to preprocess the contexts. This can be any function that takes a pandas Series (i.e., a row of
#             one of the uses.csv files) as input, and possibly other parameters, and returns a string.
#             The module preprocessing contains useful preprocessing functions.
#         :param kwargs: Extra keyword parameters for the preprocessing function.
#         :return: a list of Target instances.
#         """
#
#         if self.targets is None:
#             return [self.load_target(target, preprocessing=preprocessing, **kwargs)
#                     for target in os.listdir(str(self.data_path.joinpath("data")))]
#         else:
#             return [self.load_target(target, preprocessing=preprocessing, **kwargs)
#                     for target in os.listdir(str(self.data_path.joinpath("data")))
#                     if target in self.targets]


if __name__ == "__main__":
    dataset = LSCDDataset(config={"name": "dwug_es.yaml"})
    # target = dataloader.load_target("abundar", preprocessing=preprocessing.lemmatize, cached=True)
    print(dataset.uses)


  #   dwug_de = "https://zenodo.org/record/5796871/files/dwug_de.zip"
  #   dwug_la = "https://zenodo.org/record/5255228/files/dwug_la.zip"
  #   dwug_en = "https://zenodo.org/record/5796878/files/dwug_en.zip"
  #   dwug_sv = "https://zenodo.org/record/5090648/files/dwug_sv.zip"
  #   dwug_es = "https://zenodo.org/record/6433667/files/dwug_es.zip"
  #   discowug = "https://zenodo.org/record/5791125/files/discowug.zip"
  #   refwug = "https://zenodo.org/record/5791269/files/refwug.zip"
  #   diawug = "https://zenodo.org/record/5791193/files/diawug.zip"
  #   surel = "https://zenodo.org/record/5784569/files/surel.zip"
  #   durel = "https://zenodo.org/record/5784453/files/durel.zip"
  #   DUPS_WUG = "https://zenodo.org/record/5500223/files/DUPS-WUG.zip"