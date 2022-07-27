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
from src.config import Config, Uses
from src.use import Use


class Target:
    def __init__(self, name: str, uses_1: DataFrame, uses_2: DataFrame,
                 labels: DataFrame, judgments: DataFrame, config: Config, nlp: spacy.Language):
        self.name = name
        self.uses_1 = uses_1
        self.uses_2 = uses_2
        self.labels = labels
        self.judgments = judgments
        self.grouping_combination = config.dataset.groupings
        self.config = config
        self._use_id_pairs, self._ids = None, None



        self.preprocess(how=getattr(self.config.dataset.preprocessing.module, self.config.dataset.preprocessing.method) if self.config.dataset.preprocessing.method is not None else None,
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

    def get_uses(self) -> Tuple[Dict[str, Use], Dict[str, Use]]:
        ids2uses = (dict(), dict())
        for i, uses in enumerate([self.uses_1, self.uses_2]):
            for _, use in uses.iterrows():
                ids2uses[i][use.identifier] = Use(
                    identifier=use.identifier,
                    context_preprocessed=use.context_preprocessed,
                    target_index_begin=use.begin_index_token_preprocessed,
                    target_index_end=use.end_index_token_preprocessed
                )
        return ids2uses

    @property
    def use_id_pairs(self) -> List[Tuple[str, str]]:
        if self._ids is None and self._use_id_pairs is None:
            ids_1 = self.uses_1.identifier.tolist()
            ids_2 = self.uses_2.identifier.tolist()

            if self.config.dataset.uses.pairing == "COMPARE":
                pass
            elif self.config.dataset.uses.pairing == "EARLIER":
                ids_2 = ids_1
            elif self.config.dataset.uses.pairing == "LATER":
                ids_1 = ids_2

            if self.config.dataset.uses.type == "sampled":
                pairs = []
                for _ in range(self.config.dataset.uses.params.n):
                    id_1 = np.random.choice(ids_1, replace=self.config.dataset.uses.params.replacement)
                    id_2 = np.random.choice(ids_2, replace=self.config.dataset.uses.params.replacement)
                    pairs.append((id_1, id_2))
            elif self.config.dataset.uses.type == "all":
                if self.config.dataset.uses.pairing == "COMPARE":
                    pairs = [pair for pair in product(ids_1, ids_2)]
                elif self.config.dataset.uses.pairing in ["EARLIER", "LATER"]:
                    pairs = combinations(ids_1, r=2)
                else:
                    raise NotImplementedError
            elif self.config.dataset.uses.type == "annotated":
                ids_1 = self.judgments.identifier1
                ids_2 = self.judgments.identifier2
                # TODO implement pairing logic
                pairs = list(zip(ids_1, ids_2))
            else:
                raise NotImplementedError

            # sampled_ids = set()
            # for id1, id2 in pairs:
            #     sampled_ids.add(id1)
            #     sampled_ids.add(id2)

            self._use_id_pairs = pairs
        return self._use_id_pairs
