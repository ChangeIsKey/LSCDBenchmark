from typing import Callable, List, Tuple, Union
from src.config.config import UseID
from src.distance_model import DistanceModel
from src.target import Target, Sampling, Pairing


class DeepMistake(DistanceModel):
    def distances(
        self,
        target: Target,
        sampling: Sampling,
        pairing: Pairing,
        method: Callable,
        return_pairs: bool = False,
        **kwargs
    ) -> Union[Tuple[List[Tuple[UseID, UseID]], List[float]], List[float]]:
        pass
