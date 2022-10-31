from itertools import product
from typing import Any, TypeAlias
from hydra import utils
from numpy import isin
from omegaconf import DictConfig
from src.dataset import Dataset
from src.evaluation import Evaluation
from src.wic.model import ThresholdedWicModel, WICModel
from src.lscd import GradedModel, BinaryThresholdModel
from src.wsi.model import WSIModel


Model: TypeAlias = WICModel | ThresholdedWicModel | GradedModel | BinaryThresholdModel | WSIModel

def instantiate(config: DictConfig) -> tuple[Dataset, Model, Evaluation]:
    dataset: Dataset = utils.instantiate(config.dataset, _convert_="all")
    model: Model = utils.instantiate(config.task.model, _convert_="all")
    evaluation: Evaluation = utils.instantiate(config.task.evaluation, _convert_="all")
    return dataset, model, evaluation



def run(
    dataset: Dataset, model: Model, evaluation: Evaluation, write: bool = True
) -> float:
    predictions = {}
    if isinstance(model, ThresholdedWicModel):
        assert dataset.sampling is not None
        assert dataset.pairing is not None
        for lemma in dataset.lemmas:
            use_pairs = []
            for s, p in list(zip(dataset.sampling, dataset.pairing)):
                use_pairs += lemma.use_pairs(pairing=p, sampling=s)
            predictions.update({lemma: model.predict(use_pairs)})
            # TODO: call thresholding for WIC models
    elif isinstance(model, GradedModel):
        for lemma in dataset.lemmas:
            predictions.update({lemma: model.predict(lemma)})
    elif isinstance(model, BinaryThresholdModel):
        graded_predictions = []
        for lemma in dataset.lemmas:
            graded_predictions.append(model.graded_model.predict(lemma))
        predictions = dict(zip(dataset.lemmas, model.predict(graded_predictions)))
    elif isinstance(model, WSIModel):
        for lemma in dataset.lemmas:
            uses = lemma.get_uses()
            predictions.update(dict(zip(uses, model.predict(uses))))
        
    
    # predictions = model.predict(dataset.lemmas)
    labels = dataset.get_labels(evaluation.task)
    return evaluation(labels=labels, predictions=predictions, write=write)
