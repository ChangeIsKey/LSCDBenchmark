
from pydantic import Field
from src.config.custom_method import CustomMethod
from src.config.model.clustering import Clustering

from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.target import Target
    from src.distance_model import DistanceModel
    



class Measure(CustomMethod):
    clustering: Optional[Clustering] = Field(default=None)
    
    def __call__(self, target: "Target", model: "DistanceModel"):
        if self.clustering is not None and self.clustering.method is not None:
            return self.func(target, model, self.clustering)
        else:
            return self.func(target, model, **self.params)
