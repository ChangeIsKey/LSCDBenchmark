from typing import Any, Optional
from pydantic import BaseModel, Field
from hydra import utils

from src.config.evaluation.task import EvaluationTask


class Method(BaseModel):
    target: str = Field()
    params: dict[str, Any] = Field(default_factory=dict)
    keep: Optional[int] = Field(default=None)

    def __call__(self, *args, **kwargs) -> float | int:
        asdict = self.dict(by_alias=True, include={"target"})
        kwargs = {**kwargs, **self.params}
        results = utils.instantiate(asdict, *args, **kwargs)
        if not isinstance(results, int) or not isinstance(results, float):
            return results[self.keep]
        return results


class EvaluationConfig(BaseModel):
    metric: Optional[Method] = Field(default=None)
    task: Optional[EvaluationTask] = Field(default=None)
