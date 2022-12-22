import numpy as np
from tqdm import tqdm
from functools import partial
from pydantic import Field

from src import wic
from src.lscd.model import GradedLSCDModel
from src.lemma import Lemma, UsePairOptions

class APD(GradedLSCDModel):
    wic: wic.WICModel
    use_pair_options: UsePairOptions = Field(alias="use_pairs")

    def predict(self, lemma: Lemma) -> float:
        use_pairs = lemma.use_pairs(
            group=self.use_pair_options.group, 
            sample=self.use_pair_options.sample
        )
        similarities = self.wic.predict(use_pairs)
        return np.mean(similarities).item()

    def predict_all(self, lemmas: list[Lemma]) -> list[float]:
        use_pairs_nested = [
            lemma.use_pairs(
                group=self.use_pair_options.group, 
                sample=self.use_pair_options.sample
            ) 
            for lemma in tqdm(lemmas, desc="Building use pairs", leave=False)
        ]
        use_pairs = [use_pair for sublist in use_pairs_nested for use_pair in sublist]
        id_pairs = [(use_0.identifier, use_1.identifier) for use_0, use_1 in use_pairs]
        self.wic.predictions = dict(zip(id_pairs, self.wic.predict_all(use_pairs=use_pairs)))
        return [self.predict(lemma) for lemma in tqdm(lemmas, desc="Processing lemmas")]


class DiaSense(GradedLSCDModel):
    wic: wic.WICModel

    def predict(self, lemma: Lemma) -> float:
        use_pairs_0 = lemma.use_pairs(group="COMPARE", sample="all")
        use_pairs_1 = lemma.use_pairs(group="ALL", sample="all")
        similarities_0 = self.wic.predict(use_pairs_0)
        similarities_1 = self.wic.predict(use_pairs_1)
        return np.mean(similarities_0).item() - np.mean(similarities_1).item()

    def predict_all(self, lemmas: list[Lemma]) -> list[float]:
        use_pairs_nested = [
            lemma.use_pairs(group="ALL", sample="all") 
            for lemma in tqdm(lemmas, desc="Building use pairs", leave=False)
        ]
        use_pairs = [use_pair for sublist in use_pairs_nested for use_pair in sublist]
        id_pairs = [(use_0.identifier, use_1.identifier) for use_0, use_1 in use_pairs]
        self.wic.predictions = dict(zip(id_pairs, self.wic.predict_all(use_pairs=use_pairs)))
        return [self.predict(lemma) for lemma in tqdm(lemmas, desc="Processing lemmas")]