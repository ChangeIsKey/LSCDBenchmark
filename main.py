import logging

import hydra
from hydra.core.config_store import ConfigStore

from pathlib import Path

from config import DatasetConfig, Uses, Config, ModelConfig, Preprocessing, Task, Pairing, UsesType
from dataloader import DataLoader
from models.average import LSCDModel
from results import LSCDResults

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(name="data", node=DatasetConfig)
cs.store(name="model", node=ModelConfig)
cs.store(name="uses", node=Uses)
cs.store(name="preprocessing", node=Preprocessing)

log = logging.getLogger(f"{Path(__file__).name}:{__name__}")

# TODO make a default list of parameters for all datasets
# TODO make bert default baseline


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: Config):
    print(config)
    dataloader: DataLoader = hydra.utils.instantiate(config.dataset, _target_=DataLoader)
    dataset = dataloader.load_lscd_dataset()
    vectors_1, vectors_2 = dataset.vectorize(embedding=config.model.name, language=config.dataset.language,
                                             cased=config.model.cased)
    scores = dict()
    for target in dataset.targets:
        model = LSCDModel(vectors_1=[vector for vector in vectors_1[target.name].values()],
                          vectors_2=[vector for vector in vectors_2[target.name].values()],
                          id1_to_row={id_: i for id_, vectors in vectors_1[target.name].items() for i, _ in enumerate(vectors)},
                          id2_to_row={id_: i for id_, vectors in vectors_2[target.name].items() for i, _ in enumerate(vectors)},
                          config=config.model)

        match config.model.measure:
            case "apd":
                scores[target.name] = model.apd(target.use_id_pairs)
            case "cos":
                scores[target.name] = model.cos()

    results = LSCDResults(config=config, results=scores)


if __name__ == "__main__":
    main()
