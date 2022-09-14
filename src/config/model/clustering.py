from typing import TYPE_CHECKING, Any, Optional

from src.config.custom_method import CustomMethod

if TYPE_CHECKING:
    from src.distance_model import DistanceModel
    from src.target import Target


class Clustering(CustomMethod):
    def __init__(self, module: str, method: Optional[str], **params) -> None:
        super().__init__(module, method, None, **params)

    def __call__(self, model: "DistanceModel", target: "Target") -> Any:
        return self.func(model, target, **self.params)


