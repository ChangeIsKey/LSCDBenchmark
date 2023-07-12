import datetime
from pathlib import Path
from hydra.experimental.callback import Callback
from omegaconf import DictConfig
from typing import Any
from src.utils import utils
import pandas as pd
import numpy as np
import yaml
import json
from abc import ABC, abstractproperty
from hydra.core.utils import JobReturn, JobStatus
import logging


class LogJobReturnCallback(Callback):
    def __init__(self) -> None:
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def on_job_end(
        self, config: DictConfig, job_return: JobReturn, **kwargs: Any
    ) -> None:
        """Check the status of job in hydra.

        :param config: DictConfig
        :type config: DictConfig
        :param job_return: JobReturn
        :type job_return: JobReturn
        """        
        if job_return.status == JobStatus.COMPLETED:
            self.log.info(f"Succeeded with return value: {job_return.return_value}")
        elif job_return.status == JobStatus.FAILED:
            self.log.error("", exc_info=job_return._return_value)
        else:
            self.log.error("Status unknown. This should never happen.")


class Experiment(ABC):
    def __init__(self, path: Path) -> None:
        self.path = path

    @abstractproperty
    def timestamp(self) -> datetime.datetime:
        raise NotImplementedError

    @staticmethod
    def parse_timestamp(date: str, time: str) -> datetime.datetime:
        return datetime.datetime.strptime(f"{date} {time}", "%Y-%m-%d %H-%M-%S")

    def get_score(self) -> float | None:
        path = self.path / "result.json"
        score = None
        if path.exists():
            with path.open(mode="r", encoding="utf8") as f:
                result = json.load(f)
                score = result["score"]
        return score

    def get_config(self) -> dict[str, Any]:
        stream = (self.path / ".hydra" / "config.yaml").read_text(encoding="utf8")
        return yaml.safe_load(stream)

    def get_n_targets(self) -> int | None:
        path = self.path / "predictions.csv"
        predictions = pd.DataFrame()
        if path.exists():
            predictions = pd.read_csv(
                filepath_or_buffer=path,
                sep="\t",
                engine="pyarrow",
            )
            return len(predictions)
        return None

    def process(self) -> dict:
        score = self.get_score()
        config = self.get_config()
        n_targets = self.get_n_targets()

        return {
            "time": str(self.timestamp),
            "score": score,
            "n_targets": n_targets,
            **pd.json_normalize(config).iloc[0].to_dict(),
        }


class RunExperiment(Experiment):
    @property
    def timestamp(self) -> datetime.datetime:
        """Retrive the time stamp from the name of directories.

        :return: date and time in string
        :rtype: datetime.datetime
        """
        date = self.path.parent.name
        time = self.path.name
        return self.parse_timestamp(date, time)


class MultirunExperiment(Experiment):
    @property
    def timestamp(self) -> datetime.datetime:
        """Retrive the time stamp from the name of directories.

        :return: date and time in string
        :rtype: datetime.datetime
        """        
        date = self.path.parent.parent.name
        time = self.path.parent.name
        return self.parse_timestamp(date, time)


class ResultCollector(Callback):
    OUTPUTS_PATH = utils.path("outputs")
    MULTIRUN_PATH = utils.path("multirun")

    def __init__(self) -> None:
        self.path = utils.path("results") / "results"
        self.results: list[dict[str, Any]] = []

    def write_results(self) -> None:
        """ Write results into csv and json files in the results directory.
        """        
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.results)
        df.to_csv(
            path_or_buf=self.path.with_suffix(".csv"),
            sep="\t",
            index=False,
        )
        df.to_json(
            path_or_buf=self.path.with_suffix(".json"), 
            force_ascii=False, 
            indent=4, 
            orient="records",
            date_format=None
        )

    def on_run_end(self, config: DictConfig, **kwargs: Any) -> None:
        """Load results from outputs directory and write them to the results directory.

        :param config: DictConfig
        :type config: DictConfig
        """        
        for date in self.OUTPUTS_PATH.iterdir():
            for path in date.iterdir():
                experiment = RunExperiment(path=path)
                self.results.append(experiment.process())

        self.write_results()

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        """Load results from multirun directory and write them to the results directory.

        :param config: DictConfig
        :type config: DictConfig
        """        
        for date in self.MULTIRUN_PATH.iterdir():
            for time in date.iterdir():
                for path in time.iterdir():
                    experiment = MultirunExperiment(path=path)
                    self.results.append(experiment.process())

        self.write_results()
