import logging
import os

import hydra
import pandas as pd
from pandas import DataFrame
from pathlib import Path
from tqdm import tqdm

from omegaconf import DictConfig, OmegaConf

from src.config import Config
from src.dataloader import DataLoader
from src.lscd.results import Results
import src.lscd as lscd
from src.vectorizer import Vectorizer



@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):

    config = Config(**OmegaConf.to_object(config))
    dataloader = DataLoader(config)
    vectorizer = Vectorizer(config.model)
    dataset = dataloader.load_dataset(task=config.dataset.task)

    predictions = dict()

    pbar = tqdm(dataset.targets)
    for target in pbar:
        pbar.set_description(f"Processing target '{target.name}'")
        uses_1, uses_2 = target.get_uses()
        model = lscd.VectorModel(config, list(uses_1.values()), list(uses_2.values()), vectorizer)
        
        grouping = "_".join(list(map(str, target.grouping_combination)))
        predictions[(target.name, grouping)] = dict()
        

        for measure in config.model.measures:
            if measure == "apd":
                id_pairs = target.use_id_pairs
                predictions[(target.name, grouping)]["apd"] = float(model.apd(target_name=target.name, pairs=id_pairs))
                
            elif measure == "cos":
                predictions[(target.name, grouping)]["apd"] = float(model.cos())
   
    
    labels = dataset.labels.loc[:, ["lemma", "grouping", "graded_jsd"]].set_index(["lemma", "grouping"]).to_dict("index")
    results = Results(config, predictions, labels)
    results.score("graded_change")


if __name__ == "__main__":
    main()
