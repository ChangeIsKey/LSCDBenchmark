from itertools import product
from typing import Any, TypeAlias
from hydra import utils
from numpy import isin
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
from src.dataset import Dataset
from src.evaluation import Evaluation
from src.wic.model import ThresholdedWicModel, WICModel
from src.lscd import GradedModel, BinaryThresholdModel
from src.wsi.model import WSIModel


Model: TypeAlias = (
    WICModel | ThresholdedWicModel | GradedModel | BinaryThresholdModel | WSIModel
)


def instantiate(config: DictConfig) -> tuple[Dataset, Model, Evaluation]:
    dataset: Dataset = utils.instantiate(config.dataset, _convert_="all")
    model: Model = utils.instantiate(config.task.model, _convert_="all")
    evaluation: Evaluation = utils.instantiate(config.task.evaluation, _convert_="all")
    return dataset, model, evaluation


def run(
    dataset: Dataset, model: Model, evaluation: Evaluation, write: bool = True
) -> float:
    predictions = {}
    lemma_pbar = tqdm(dataset.lemmas, desc="Processing lemmas")
    if isinstance(model, (ThresholdedWicModel, WICModel)):
        assert dataset.sampling is not None
        assert dataset.pairing is not None
        for lemma in lemma_pbar:
            use_pairs = []
            for s, p in list(zip(dataset.sampling, dataset.pairing)):
                use_pairs += lemma.use_pairs(pairing=p, sampling=s)
            id_pairs = [
                (use_0.identifier, use_1.identifier) for use_0, use_1 in use_pairs
            ]
            predictions.update(dict(zip(id_pairs, model.predict(use_pairs))))
            # TODO: call thresholding for WIC models
    elif isinstance(model, GradedModel):
        for lemma in lemma_pbar:
            predictions.update({lemma.name: model.predict(lemma)})
    elif isinstance(model, BinaryThresholdModel):
        graded_predictions = []
        lemma_names = [lemma.name for lemma in dataset.lemmas]
        for lemma in lemma_pbar:
            graded_predictions.append(model.graded_model.predict(lemma))
        predictions = dict(zip(lemma_names, model.predict(graded_predictions)))
    elif isinstance(model, WSIModel):
        for lemma in lemma_pbar:
            uses = lemma.get_uses()
            ids = [use.identifier for use in uses]
            predictions.update(dict(zip(ids, model.predict(uses))))

    labels = dataset.get_labels(evaluation.task)
    score = evaluation(labels=labels, predictions=predictions, write=write)
    print(predictions)
    return score
