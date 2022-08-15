from typing import Callable, List, Tuple, Union
from src.config import ID, pairing, sampling
from src.distance_model import DistanceModel
from src.lscd.target import Target


class DeepMistake(DistanceModel):
    def distances(self, target: Target, sampling: sampling, pairing: pairing, method: Callable, return_pairs: bool = False, **kwargs) -> Union[Tuple[List[Tuple[ID, ID]], List[float]], List[float]]:
        pass
