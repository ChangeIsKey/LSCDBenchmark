from itertools import product, combinations
from typing import Callable, Tuple, Dict, List, Set

import numpy as np
import pandas as pd
import spacy
import sys
import importlib
import importlib.util
from pandas import DataFrame, Series

from src import preprocessing
from src.config import ID, Config, Uses
from src.use import Use


class Target:
    def __init__(self, name: str, uses_1: DataFrame, uses_2: DataFrame,
                 labels: DataFrame, judgments: DataFrame, config: Config, nlp: spacy.Language):
        self.name = name
        self.uses_1 = uses_1
        self.uses_2 = uses_2
        self._ids_to_uses = None
        self.labels = labels
        self.judgments = judgments
        self.grouping_combination = config.dataset.groupings
        self.config = config
        self._use_id_pairs, self._ids = None, None



        self.preprocess(how=self.config.dataset.preprocessing.method,
                        nlp=nlp,
                        **self.config.dataset.preprocessing.params)

    def preprocess(self, how: Callable[[Series], Tuple[str, int, int]] = None, nlp: spacy.Language = None, **params) -> None:
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
        assert how is None or callable(how), TypeError("preprocessing parameter type is invalid")

        if how is None:
            def keep_intact(s: Series, **kwargs) -> Tuple[str, int, int]:
                start, end = tuple(map(int, s.indices_token.split(":")))
                return s.context, start, end
            how = keep_intact

        def func(s: Series, **kwargs) -> Series:
            context, start, end = how(s, **kwargs)
            return Series({
                "context_preprocessed": context,
                "begin_index_token_preprocessed": start,
                "end_index_token_preprocessed": end,
            })

        if not params.get("cached"):
            params.update(dict(nlp=nlp))

        self.uses_1 = pd.concat([self.uses_1, self.uses_1.apply(func, axis=1, **params)], axis=1)
        self.uses_2 = pd.concat([self.uses_2, self.uses_2.apply(func, axis=1, **params)], axis=1)

        if callable(how):
            assert self.uses_1.context_preprocessed.apply(isinstance, args=[str]).all(), \
                TypeError(f"Invalid return type for preprocessing function {how.__name__}")
            assert self.uses_2.context_preprocessed.apply(isinstance, args=[str]).all(), \
                TypeError(f"Invalid return type for preprocessing function {how.__name__}")

    @property
    def ids_to_uses(self) -> Dict[ID, Use]:
        if self._ids_to_uses is None:
            self._ids_to_uses = dict()
            for uses in [self.uses_1, self.uses_2]:
                for _, use in uses.iterrows():
                    self._ids_to_uses[use.identifier] = Use(
                        identifier=use.identifier,
                        context_preprocessed=use.context_preprocessed,
                        target_index_begin=use.begin_index_token_preprocessed,
                        target_index_end=use.end_index_token_preprocessed
                    )
        return self._ids_to_uses