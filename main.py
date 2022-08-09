import hydra
from omegaconf import OmegaConf
from tqdm import tqdm

from src.config import Config
from src.dataloader import DataLoader
from src.lscd.model import VectorModel
from src.lscd.results import Results
from src.vectorizer import Vectorizer


@hydra.main(version_base=None, config_path="config", config_name="defaults")
def main(cfg: Config):
    config = Config(**OmegaConf.to_object(cfg))
    vectorizer = Vectorizer(config)
    dataset = DataLoader(config).load_dataset()

    predictions = dict()
    labels = (
        dataset.labels.loc[:, ["lemma", "change_graded", "change_binary"]]
        .set_index("lemma")
        .to_dict("index")
    )

    for target in tqdm(dataset.targets, desc="Processing targets"):
        model = VectorModel(config, vectorizer, target)
        predictions[target.name] = config.model.measure.method(
            target, model, **config.model.measure.params
        )

    results = Results(config, predictions, labels)
    results.score(task="graded_change")


if __name__ == "__main__":
    main()
