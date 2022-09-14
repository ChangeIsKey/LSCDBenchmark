from typing import Optional, Any
from src.config.custom_method import CustomMethod
import numpy as np


class Threshold(CustomMethod):    
    def __init__(self, module: str, method: str | None, params: dict[str, Any]) -> None:
        super().__init__(module=module, method=method, default=None, params=params)

    def __call__(self, distances: np.ndarray):
        # there is no default threshold function, so we need to check if the user provided one
        return None if self.func is None else self.func(distances, **self.params)


