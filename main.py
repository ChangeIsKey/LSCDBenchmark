import dotenv
dotenv.load_dotenv()

import hydra
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from src.config import Config
from src.dataset import Dataset
from src.deepmistake import DeepMistake
from src.results import Results
from src.vector_model import VectorModel


@hydra.main(version_base=None, config_path="config", config_name="defaults")
def main(cfg: DictConfig):
    config = Config(**OmegaConf.to_object(cfg))
    dataset = Dataset(config)
    model = (
        DeepMistake() if config.model.lower() == "deep_mistake" 
        else VectorModel(config, dataset.targets)
    )

    predictions = [config.measure(target, model) for target in tqdm(dataset.targets, desc="Computing predictions", leave=False)]
    predictions = {k: v for d in predictions for k, v in d.items()}
    results = Results(config=config, predictions=predictions, labels=dataset.labels)
    results.score()


if __name__ == "__main__":
    main()
