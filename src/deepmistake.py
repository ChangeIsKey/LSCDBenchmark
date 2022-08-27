from typing import Callable, List, Tuple, Union
from src.config import UseID, pairing, sampling
from src.distance_model import DistanceModel
from src.lscd.target import Target


class DeepMistake(DistanceModel):
    def distances(
        self,
        target: Target,
        sampling: sampling,
        pairing: pairing,
        method: Callable,
        return_pairs: bool = False,
        **kwargs
    ) -> Union[Tuple[List[Tuple[UseID, UseID]], List[float]], List[float]]:
        pass
