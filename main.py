import dotenv
dotenv.load_dotenv()

import hydra
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from src.config.config import Config
from src.dataset import Dataset
from src.wic.deepmistake import DeepMistake
from src.results import Results
from src.wic.vector_model import VectorModel


@hydra.main(version_base=None, config_path="config", config_name="defaults")
def main(cfg: DictConfig):
    config = Config(**OmegaConf.to_object(cfg))
    dataset = Dataset(config)
    model = (
        DeepMistake() if config.model.name.lower() == "deep_mistake" 
        else VectorModel(config, dataset.targets)
    )

    predictions = model.predict(dataset.targets)
    results = Results(config=config, predictions=predictions, labels=dataset.labels)
    results.score()


if __name__ == "__main__":
    main()
