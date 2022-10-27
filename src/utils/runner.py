from itertools import product
from typing import Any
from hydra import utils
from numpy import isin
from omegaconf import DictConfig
from src.dataset import Dataset
from src.evaluation import Evaluation
from src.wic.model import WICModel


def instantiate(config: DictConfig) -> tuple[Dataset, Any, Evaluation]:
    dataset = utils.instantiate(config.dataset, _convert_="all")
    model = utils.instantiate(config.task.model, _convert_="all")
    evaluation = utils.instantiate(config.task.evaluation, _convert_="all")
    return dataset, model, evaluation



def run(
    dataset: Dataset, model: Any, evaluation: Evaluation, write: bool = True
) -> float:
    predictions = {}
    if isinstance(model, WICModel):
        for lemma in dataset.lemmas:
            use_pairs = []
            for s, p in list(zip(dataset.sampling, dataset.pairing)):
                use_pairs += lemma.use_pairs(pairing=p, sampling=s)
            predictions.update(model.predict(use_pairs))
            # TODO: call thresholding for WIC models
    elif isinstance(model, LSCDModel):
        for lemma in dataset.lemmas:
            predictions.update({lemma: model.predict(lemma)})
    elif isinstance(model, WSIModel):
        for lemma in dataset.lemmas:
            uses = lemma.get_uses()
            use_pairs = list(product(uses))
            similarity_matrix = model.similarity_matrix(use_pairs)
            predictions.update(model.predict(uses))
        
    
    # predictions = model.predict(dataset.lemmas)
    labels = dataset.get_labels(evaluation.task)
    return evaluation(labels=labels, predictions=predictions, write=write)
