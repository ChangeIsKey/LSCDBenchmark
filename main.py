import dotenv
import hydra
from hydra import utils
from omegaconf import DictConfig

dotenv.load_dotenv()


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config"
)
def main(config: DictConfig):
    dataset = utils.instantiate(config.dataset, _convert_="all")
    model = utils.instantiate(config.model, _convert_="all")
    evaluation = utils.instantiate(config.evaluation, _convert_="all")

    keys, predictions = model.predict(dataset.targets)
    labels = dataset.get_labels(evaluation.task, keys)
    score = evaluation(labels, predictions)
  
    print(score)


if __name__ == "__main__":
    main()
