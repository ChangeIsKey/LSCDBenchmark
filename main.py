import dotenv
import hydra
from hydra import utils
from omegaconf import DictConfig

dotenv.load_dotenv()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    dataset = utils.instantiate(config.dataset, _convert_="all")
    model = utils.instantiate(config.task.model, _convert_="all")
    evaluation = utils.instantiate(config.task.evaluation, _convert_="all")

    predictions = model.predict(dataset.targets)
    labels = dataset.get_labels(evaluation.task)
    score = evaluation(labels=labels, predictions=predictions)

    print(score)


if __name__ == "__main__":
    main()
