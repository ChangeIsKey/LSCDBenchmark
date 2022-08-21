import hydra
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
import pandas as pd
import json

from src.config import Config
from src.dataset import Dataset
from src.deepmistake import DeepMistake
from src.lscd.results import Results
from src.vector_model import VectorModel


@hydra.main(version_base=None, config_path="config", config_name="defaults")
def main(cfg: DictConfig):
    config = Config(**OmegaConf.to_object(cfg))

    dataset = Dataset(config)
    model = (
        DeepMistake() if config.model.lower() == "deep_mistake" 
        else VectorModel(config, dataset.targets)
    )

    predictions = {
        target.name: config.measure(target, model, **config.measure.method_params)
        for target in sorted(dataset.targets, key=lambda target: target.name)
    }

    labels = dict(zip(dataset.lscd_labels.lemma, dataset.lscd_labels[config.evaluation.task.value]))
    results = Results(config, predictions, labels)
    results.score()


if __name__ == "__main__":
    main()
