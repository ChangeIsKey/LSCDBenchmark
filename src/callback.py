from datetime import datetime
from pathlib import Path
from hydra.experimental.callback import Callback
from omegaconf import DictConfig
from typing import Any
from src.utils import utils
import pandas as pd

class ResultCollector(Callback):
    OUTPUTS_PATH = utils.path("outputs")
    MULTIRUN_PATH = utils.path("multirun")

    def on_run_end(self, config: DictConfig, **kwargs: Any) -> None:
        for date in self.OUTPUTS_PATH.iterdir():
            for experiment in date.iterdir():
                print(experiment)
                # timestamp = datetime.strptime(
                #     f"{date.name} {experiment.name}", "%Y-%m-%d %H-%M-%S"
                # )

        pass

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        pass

    @staticmethod
    def process(experiment: Path) -> pd.DataFrame:
        pass