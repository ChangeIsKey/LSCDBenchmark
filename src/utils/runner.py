from typing import Any
from hydra import utils
from omegaconf import DictConfig
from src.dataset import Dataset
from src.evaluation import Evaluation

def instantiate(config: DictConfig) -> tuple[Dataset, Any, Evaluation]:
    dataset = utils.instantiate(config.dataset, _convert_="all")
    model = utils.instantiate(config.task.model, _convert_="all")
    evaluation = utils.instantiate(config.task.evaluation, _convert_="all")
    return dataset, model, evaluation
    

def run(dataset: Dataset, model: Any, evaluation: Evaluation, write: bool = True) -> float:
    predictions = model.predict(dataset.targets)
    labels = dataset.get_labels(evaluation.task)
    return evaluation(labels=labels, predictions=predictions, write=write)
