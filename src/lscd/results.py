from typing import Dict

from src.config import Config


class Results:
    def __init__(self, config: Config, results: Dict[str, float]):
        self.config = config

    def score(self, mapping_1: Dict[str, float], mapping_2: Dict[str, float], metric: str):
        pass

    def export(self, format: str):
        pass
