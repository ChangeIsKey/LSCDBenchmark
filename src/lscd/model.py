import torch
import torch.nn.functional as F
import pandas as pd

from typing import List, Dict, Tuple

import numpy as np
from pandas import DataFrame
from scipy.spatial import distance

from src.config import Config
from src.use import Use
from src.vectorizer import Vectorizer


class VectorModel:
    def __init__(self, config: Config, uses_1: List[Use], uses_2: List[Use], vectorizer: Vectorizer):
        # TODO check if model can just take the identifiers (optionally)
        self.config = config
        self.uses = (uses_1, uses_2)
        self.vectorizer = vectorizer
        self._vectors = None
        self._id_to_row = None
        self._distances = None

    @property
    def distances(self):
        if self._distances is None:
            try:
                self._distances = pd.read_csv(self.config.results.output_directory.joinpath("distances.csv"), delimiter="\t")
            except FileNotFoundError:
                self._distances = DataFrame(columns=["model", "target", "preprocessing", "dataset"])

        return self._distances

    @distances.setter
    def distances(self, value: DataFrame):
        self._distances = value

    @property
    def id_to_row(self):
        if self._id_to_row is None:
            self._id_to_row = {k: v for d in [{use.identifier: i for i, use in enumerate(self.uses[j])} for j in range(2)]
                               for k, v in d.items()}
        return self._id_to_row

    @property
    def vectors(self):
        if self._vectors is None:
            self._vectors = tuple([
                self.vectorizer(
                    contexts=[u.context_preprocessed for u in self.uses[i]],
                    target_indices=[(u.target_index_begin, u.target_index_end) for u in self.uses[i]]
                )
                for i in range(2)
            ])
        return self._vectors


    def cache(self, target_name: str, measure: Dict[str, float]):
        row = {
            "model": self.config.model.name, 
            "preprocessing": self.config.dataset.preprocessing.method, 
            "target": target_name,
            "dataset": self.config.dataset.name,
            **measure
        }
        self.distances = pd.concat([self.distances, DataFrame([row])], ignore_index=True)
        
    def apd(self, target_name: str, pairs: List[Tuple[str, str]], distance_measure=lambda *args, **kwargs: 1 - F.cosine_similarity(*args, **kwargs)):
        row = self.distances[(self.distances.model == self.config.model.name) & 
                               (self.distances.target == target_name) & 
                               (self.distances.preprocessing == self.config.dataset.preprocessing.method) &
                               (self.distances.dataset == self.config.dataset.name)]
        if row.empty:  
            apd = float(torch.mean(torch.stack([
                distance_measure(self.vectors[0][self.id_to_row[id1]], self.vectors[1][self.id_to_row[id2]], dim=0)
                for id1, id2 in pairs
            ])))
            self.cache(target_name, measure={"apd": apd})
            self.distances.to_csv(self.config.results.output_directory.joinpath("distances.csv"), sep="\t", index=False)
            return apd
        else: 
            return row["apd"]

    def cosine(self):
        return 1 - F.cosine_similarity(torch.mean(self.vectors[0], dim=0, keepdim=True), torch.mean(self.vectors[1], dim=0, keepdim=True))
