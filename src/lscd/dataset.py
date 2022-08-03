from pathlib import Path
from typing import Optional, List

import spacy
from pandas import DataFrame
from tqdm import tqdm

from src.config import Config
from src.lscd.target import Target


class Dataset:

    lang2model = {
        "en": "en_core_web_sm",
        "english": "en_core_web_sm",
        "es": "es_core_news_sm",
        "spanish": "es_core_news_sm",
        "swedish": "sv_core_news_sm",
        "sv": "sv_core_news_sm",
        "german": "de_core_news_sm",
        "de": "de_core_news_sm"
    }

    def __init__(self, config: Config, uses: DataFrame, labels: DataFrame, judgments: DataFrame, agreements: DataFrame, mask: bool = False):
        # TODO add test option

        self.config = config
        self.nlp: Optional[spacy.Language] = None

        self.uses = uses
        self.labels = labels
        self.judgments = judgments
        self.agreements = agreements
        self.mask = mask
        self.groupings = self.config.dataset.groupings
        self._targets = None

    @property
    def targets(self) -> List[Target]:
        if self._targets is None:
            self.agreements = self.agreements.iloc[1:, :].copy()  # remove "data=full" row
            print(self.agreements.head())
            
            if len(self.config.dataset.cleaning.fields) > 0:

                conditions = [f'{column} >= {cleaning_param.threshold}' if cleaning_param.above else f'{column} <= {cleaning_param.threshold}' 
                            for column, cleaning_param in self.config.dataset.cleaning.fields.items()]

                if self.config.dataset.cleaning.method == "all":
                    connector = '&'
                elif self.config.dataset.cleaning.method == "any":
                    connector = '|'
                else:
                    raise NotImplementedError

                self.agreements = self.agreements.query(connector.join(conditions))
                
            group_combination = "_".join(map(str, self.groupings))

            self._targets = [
                Target(
                    config=self.config,
                    name=target,
                    uses_1=self.uses[(self.uses.lemma == target) & (self.uses.grouping == self.groupings[0])].copy(),
                    uses_2=self.uses[(self.uses.lemma == target) & (self.uses.grouping == self.groupings[1])].copy(),
                    labels=self.labels[(self.labels.lemma == target) & (
                                self.labels.grouping == group_combination)] if not self.mask else None,
                    judgments=self.judgments[self.judgments.lemma == target],
                    nlp=self.nlp,
                )
                for target in tqdm(self.agreements.data.unique(), desc="Building targets")
            ]
        return self._targets