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

def populate_config(config: DictConfig) -> DictConfig:
    # Hydra cannot interpolate values from the final config in the defaults list
    # So we need a workaround
    with open(
        file=f"../../../splits/{config.dataset.name}_{config.dataset.version}.yaml",
        mode="r",
        encoding="utf8"
    ) as f:
        config.dataset.standard_split = yaml.safe_load(f)
    return config
    
def overwrite_config_file(config: DictConfig) -> None:
    """Hydra writes a config file to its working directory at .hydra/config.yaml
    However, this file only contains the values at launch time 
    (i.e., it does not contain fields added at runtime such as dataset.standard_split).
    It also doesn't write the interpolated config, which makes inspection of the resulting config
    more complicated
    """

    with open(file=f"{Path(os.getcwd()) / '.hydra' / 'config.yaml'}", mode="w", encoding="utf8") as f:
        config_copy = config.copy()
        OmegaConf.resolve(config_copy)
        f.write(OmegaConf.to_yaml(config_copy))

def instantiate(config: DictConfig) -> tuple[Dataset, Model, Evaluation]:
    config = populate_config(config)
    overwrite_config_file(config)

    dataset: Dataset = utils.instantiate(config.dataset, _convert_="all")
    model: Model = utils.instantiate(config.task.model, _convert_="all")
    evaluation: Evaluation = utils.instantiate(config.evaluation, _convert_="all")
    return dataset, model, evaluation


def run(
    dataset: Dataset, model: Model, evaluation: Evaluation, write: bool = True
) -> float:

    cwd = os.getcwd()
    labels = dataset.get_labels(evaluation_task=evaluation.task)
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
    score = evaluation(labels=labels, predictions=predictions, write=write)
    return score
