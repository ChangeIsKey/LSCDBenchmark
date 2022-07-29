from pathlib import Path

import hydra
from omegaconf import OmegaConf
from tqdm import tqdm

from src.config import Config
from src.dataloader import DataLoader
from src.lscd.model import VectorModel
from src.lscd.results import Results
from src.vectorizer import Vectorizer


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: Config):
    config = Config(**OmegaConf.to_object(cfg))
    vectorizer = Vectorizer(config.model)
    dataset = DataLoader(config).load_dataset(task=config.dataset.task)

    predictions = dict()
    labels = (
        dataset.labels.loc[:, ["lemma", "graded_jsd"]]
        .set_index("lemma")
        .to_dict("index")
    )

    for target in tqdm(dataset.targets, desc="Processing targets"):
        uses = list(target.ids_to_uses.values())
        model = VectorModel(config, uses, vectorizer)
        predictions[target.name] = config.model.measure.method(target, model)

    results = Results(config, predictions, labels)
    results.score(task="graded_change")


if __name__ == "__main__":
    main()
