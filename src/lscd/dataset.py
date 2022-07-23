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

    def __init__(self, config: Config, uses: DataFrame, labels: DataFrame, judgments: DataFrame, mask: bool = False):
        # TODO add test option

        self.config = config
        self.nlp: Optional[spacy.Language] = None

        self.uses = uses
        self.labels = labels
        self.judgments = judgments
        self.mask = mask

        self.grouping_1 = self.config.dataset.grouping_1
        self.grouping_2 = self.config.dataset.grouping_2

        self._targets = None
        self.name2target = {target.name: target for target in self.targets}

    def get_spacy_model(self) -> spacy.Language:
        return spacy.load(self.lang2model[self.config.dataset.language])

    @property
    def targets(self) -> List[Target]:
        if self._targets is not None:
            return self._targets

        group_combination = f"{self.grouping_1}_{self.grouping_2}"
        target_names = [target.name for target in
                        Path(__file__).parent.parent.parent.joinpath("wug", self.config.dataset.name, "data")
                        .iterdir()]

        if not self.config.dataset.preprocessing.params.get("cached"):
            self.nlp = self.get_spacy_model()

        self._targets = [
            Target(
                config=self.config,
                name=target,
                grouping_1=self.grouping_1,
                grouping_2=self.grouping_2,
                uses_1=self.uses[(self.uses.lemma == target) & (self.uses.grouping == self.grouping_1)].copy(),
                uses_2=self.uses[(self.uses.lemma == target) & (self.uses.grouping == self.grouping_2)].copy(),
                labels=self.labels[(self.labels.lemma == target) & (
                            self.labels.grouping == group_combination)] if not self.mask else None,
                judgments=self.judgments[self.judgments.lemma == target],
                nlp=self.nlp,
            )
            for target in tqdm(target_names[:2], desc="Building targets")
        ]
        return self._targets

    @targets.setter
    def targets(self, targets: List[Target]) -> None:
        self._targets = targets

    def get_use_id_pairs(self):
        pairs = []
        for target in self.targets:
            pairs.extend(target.get_use_id_pairs(self.config.dataset.uses))

    def get_uses(self):
        uses_1 = dict()
        uses_2 = dict()
        for target in self.targets:
            target_uses_1, target_uses_2 = target.get_uses()
            uses_1.update(target_uses_1)
            uses_2.update(target_uses_2)

        return uses_1, uses_2
