from typing import List

from pandas import DataFrame
from tqdm import tqdm

from src.config import Config, ThresholdParam
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
        "de": "de_core_news_sm",
    }

    def __init__(
        self,
        config: Config,
        uses: DataFrame,
        labels: DataFrame,
        judgments: DataFrame,
        agreements: DataFrame,
    ):
        # TODO add test option

        self.config = config
        self.uses = uses
        self.labels = labels
        self.judgments = judgments
        self.agreements = agreements
        self.groupings = self.config.dataset.groupings
        self._targets = None

    @property
    def targets(self) -> List[Target]:
        if self._targets is None:
            self.agreements = self.agreements.iloc[
                1:, :
            ].copy()  # remove "data=full" row

            if len(self.config.dataset.cleaning.stats) > 0:

                conditions = [
                    f"{column} >= {cleaning_param.threshold}"
                    if cleaning_param.keep is ThresholdParam.ABOVE
                    else f"{column} <= {cleaning_param.threshold}"
                    for column, cleaning_param in self.config.dataset.cleaning.stats.items()
                ]

                if self.config.dataset.cleaning.method == "all":
                    connector = "&"
                elif self.config.dataset.cleaning.method == "any":
                    connector = "|"
                else:
                    raise NotImplementedError

                self.agreements = self.agreements.query(connector.join(conditions))

            group_combination = "_".join(map(str, self.groupings))

            self._targets = [
                Target(
                    config=self.config,
                    name=target,
                    uses=self.uses[self.uses.lemma == target].copy(),
                    labels=self.labels[
                        (self.labels.lemma == target)
                        & (self.labels.grouping == group_combination)
                    ],
                    judgments=self.judgments[self.judgments.lemma == target],
                )
                for target in tqdm(
                    self.uses.lemma.unique(), desc="Building targets", leave=False
                )
            ]
        return self._targets
