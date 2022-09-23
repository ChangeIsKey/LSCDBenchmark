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
    dataset = utils.instantiate(config.dataset)
    model = utils.instantiate(config.model)
    evaluation = utils.instantiate(config.evaluation)

    keys, predictions = model.predict(dataset.targets)
    labels = dataset.get_labels(evaluation.task, keys)
    score = evaluation(labels, predictions)
    print(predictions)
    print(labels)
    print(score)


if __name__ == "__main__":
    main()
