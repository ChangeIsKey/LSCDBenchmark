import datetime
from pathlib import Path
from hydra.experimental.callback import Callback
from omegaconf import DictConfig
from typing import Any
from src.utils import utils
import pandas as pd
import numpy as np
import yaml
from abc import ABC, abstractproperty
from hydra.core.utils import JobReturn, JobStatus
import logging


class LogJobReturnCallback(Callback):
    def __init__(self) -> None:
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def on_job_end(
        self, config: DictConfig, job_return: JobReturn, **kwargs: Any
    ) -> None:
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
        path = self.path / "score.txt"
        score = None
        if path.exists():
            score = np.loadtxt(path).item()
        return score

    def get_config(self) -> dict[str, Any]:
        stream = (self.path / ".hydra" / "config.yaml").read_text(encoding="utf8")
        return yaml.safe_load(stream)
    
    def get_n_targets(self) -> int | None:
        path = self.path / "predictions.tsv"
        predictions = pd.DataFrame()
        if path.exists():
            predictions = pd.read_csv(
                filepath_or_buffer=path, 
                sep="\t", 
                engine="pyarrow",
            )
            return len(predictions["target"].tolist())  # TODO: rename to instance
        return None
        
        
    def process(self) -> pd.DataFrame:
        score = self.get_score()
        config = self.get_config()
        n_targets = self.get_n_targets()

        return pd.concat(
            [
                pd.json_normalize(config),
                pd.DataFrame([{
                    "time": self.timestamp, 
                    "score": score,
                    "n_targets": n_targets,
                }]),
            ],
            axis=1,
        )

    
class RunExperiment(Experiment):
    @property
    def timestamp(self) -> datetime.datetime:
        date = self.path.parent.name
        time = self.path.name
        return self.parse_timestamp(date, time)
        

class MultirunExperiment(Experiment):
    @property
    def timestamp(self) -> datetime.datetime:
        date = self.path.parent.parent.name
        time = self.path.parent.name
        return self.parse_timestamp(date, time)
    

class ResultCollector(Callback):
    OUTPUTS_PATH = utils.path("outputs")
    MULTIRUN_PATH = utils.path("multirun")

    def __init__(self) -> None:
        self.path = utils.path("results.csv")
        self.results = self.load_results()
        
    def load_results(self) -> pd.DataFrame:
        results = pd.DataFrame()
        if self.path.exists():
            results = pd.read_csv(
                filepath_or_buffer=self.path,
                sep="\t",
                engine="pyarrow"
            )
        return results
    
    def write_results(self) -> None:
        self.results.to_csv(
            path_or_buf=self.path,
            sep="\t",
            index=False
        )
        
    def on_run_end(self, config: DictConfig, **kwargs: Any) -> None:
        if self.OUTPUTS_PATH.exists():
            for date in self.OUTPUTS_PATH.iterdir():
                for path in date.iterdir():
                    experiment = RunExperiment(path=path)
                    self.results = pd.concat([self.results, experiment.process()])

        self.write_results()

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        if self.MULTIRUN_PATH.exists():
            for date in self.OUTPUTS_PATH.iterdir():
                for time in date.iterdir():
                    for path in time.iterdir():
                        experiment = MultirunExperiment(path=path)
                        self.results = pd.concat([self.results, experiment.process()])

        self.write_results()