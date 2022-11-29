from typing import Any, TypeAlias
from hydra import utils
from pathlib import Path
import os
import yaml
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from src.dataset import Dataset
from src.evaluation import Evaluation
from src.wic.model import ThresholdedWicModel, WICModel
from src.lscd import GradedLSCDModel, BinaryThresholdModel
from src.wsi.model import WSIModel


Model: TypeAlias = (
    WICModel | ThresholdedWicModel | GradedLSCDModel | BinaryThresholdModel | WSIModel
)


def instantiate(config: DictConfig) -> tuple[Dataset | None, Model | None, Evaluation | None]:
    dataset = None
    model = None
    evaluation = None

    if config.get("dataset") is not None:
        dataset = utils.instantiate(config.dataset, _convert_="all")
    if config.get("task") is not None and config.task.get("model") is not None:
        model = utils.instantiate(config.task.model, _convert_="all")
    if config.get("evaluation") is not None:
        evaluation = utils.instantiate(config.evaluation, _convert_="all")

    return dataset, model, evaluation


def run(
    dataset: Dataset | None, model: Model | None, evaluation: Evaluation | None, write: bool = True
) -> float | None:

    score = None
    if model is not None and dataset is not None:
        cwd = os.getcwd()
        predictions: Any = {}

        lemmas = dataset.filter_lemmas(dataset.lemmas)
        lemma_pbar = tqdm(lemmas, desc="Processing lemmas")
        if isinstance(model, WICModel):
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
        elif isinstance(model, GradedLSCDModel):
            for lemma in lemma_pbar:
                predictions.update({lemma.name: model.predict(lemma)})
        elif isinstance(model, BinaryThresholdModel):
            graded_predictions = []
            lemma_names = [lemma.name for lemma in dataset.lemmas]
            for lemma in lemma_pbar:
                graded_predictions.append(model.graded_model.predict(lemma))
            predictions.update(dict(zip(lemma_names, model.predict(graded_predictions))))
        elif isinstance(model, WSIModel):
            for lemma in lemma_pbar:
                uses = lemma.get_uses()
                ids = [use.identifier for use in uses]
                predictions.update(dict(zip(ids, model.predict(uses))))

        os.chdir(cwd)

        if evaluation is not None:
            labels = dataset.get_labels(evaluation_task=evaluation.task)
            score = evaluation(labels=labels, predictions=predictions)
    return score
