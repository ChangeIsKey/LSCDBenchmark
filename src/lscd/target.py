import importlib
import importlib.util
import sys
from itertools import combinations, product
from typing import Callable, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from src.config import ID, Config
from src.use import Use


class Target:
    def __init__(
        self,
        name: str,
        uses_1: DataFrame,
        uses_2: DataFrame,
        labels: DataFrame,
        judgments: DataFrame,
        config: Config,
    ):
        self.name = name
        self.uses_1 = uses_1
        self.uses_2 = uses_2
        self._ids_to_uses = None
        self.labels = labels
        self.judgments = judgments
        self.grouping_combination = config.dataset.groupings
        self.config = config
        self._use_id_pairs, self._ids = None, None

        self.preprocess()

    def preprocess(self) -> None:
        """
        :param nlp:
        :param how:
            A function to preprocess the contexts. This can be any function that takes a pandas Series (i.e., a row of
            one of the uses.csv files) as input, and possibly other parameters, and returns a string.
            The module preprocessing contains useful preprocessing functions.
        :param params: Extra keyword parameters for the preprocessing function
        :raises TypeError: if the 'how' parameter is neither None nor a function
        :raises TypeError: if the one of the returned outputs of the preprocessing function is not a string
        """

        method = self.config.dataset.preprocessing.method
        params = self.config.dataset.preprocessing.params

        self.uses_1 = pd.concat(
            [self.uses_1, self.uses_1.apply(method, axis=1, **params)], axis=1
        )
        self.uses_2 = pd.concat(
            [self.uses_2, self.uses_2.apply(method, axis=1, **params)], axis=1
        )

    @property
    def ids_to_uses(self) -> Dict[ID, Use]:
        if self._ids_to_uses is None:
            self._ids_to_uses = {
                use.identifier: Use(
                    target=self.name,
                    identifier=use.identifier,
                    context_preprocessed=use.context_preprocessed,
                    target_index_begin=use.begin_index_token_preprocessed,
                    target_index_end=use.end_index_token_preprocessed,
                )
                for uses in [self.uses_1, self.uses_2]
                for _, use in uses.iterrows()
            }
        return self._ids_to_uses
