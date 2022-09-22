import dotenv
dotenv.load_dotenv()

import hydra
from omegaconf import DictConfig

from src.config.config import Config
from src.dataset import Dataset
from src.results import Results


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config"
)
def main(config: DictConfig):
    config: Config = Config.from_dictconfig(config)
    model = config.model.instantiate(config)
    dataset = Dataset(config)
    predictions = model.predict(dataset.targets)
    results = Results(config=config, predictions=predictions, labels=dataset.labels)
    results.score()


if __name__ == "__main__":
    main()
