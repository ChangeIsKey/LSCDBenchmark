import hydra
from omegaconf import OmegaConf
from tqdm import tqdm

from src.config import Config
from src.dataloader import DataLoader
from src.deepmistake import DeepMistake
from src.lscd.results import Results
from src.vector_model import VectorModel


@hydra.main(version_base=None, config_path="config", config_name="defaults")
def main(cfg: Config):
    config = Config(**OmegaConf.to_object(cfg))

    dataset = DataLoader(config).load_dataset()
    if config.model.name.lower() == "deep_mistake":
        model = DeepMistake()
    else:
        model = VectorModel(config, dataset.targets)


    predictions = {
        target.name: config.model.measure(target, model, **config.model.measure.method_params)
        for target in sorted(dataset.targets, key=lambda target: target.name)
    }

    labels = dict(zip(dataset.labels.lemma, dataset.labels[config.evaluation.task.value]))
    results = Results(config, predictions, labels)
    results.score()


if __name__ == "__main__":
    main()
