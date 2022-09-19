from src.models.lscd_model import LSCDModel
from src.config.config import Config
from target import Target
import numpy as np

class ApdCompareAll(LSCDModel):
    def __init__(self, config: Config):
        self.config = config
    
    def predict(self, targets: list[Target]) -> list[float]:
        raise NotImplementedError