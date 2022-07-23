#!/usr/bin/env python
import logging

import hydra

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.config import Config
from src.dataloader import DataLoader
import src.lscd as lscd

log = logging.getLogger(f"{Path(__file__).name}:{__name__}")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    config = Config(**OmegaConf.to_object(config))
    dataloader = DataLoader(config)
    dataset = dataloader.load_dataset(task="lscd")

    for target in dataset.targets:
        uses_1, uses_2 = target.get_uses()

        model = lscd.VectorModel(config.model, list(uses_1.values()), list(uses_2.values()))
        for measure in model.config.measures:
            if measure == "apd":
                id_pairs = target.use_id_pairs
                print(model.apd(id_pairs))
            elif measure == "cos" or measure == "cosine":
                print(model.cosine())


if __name__ == "__main__":
    main()
