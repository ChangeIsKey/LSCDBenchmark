from dataclasses import dataclass
import functools
from typing import Any


@dataclass
class DistanceMetric:
    method: functools.partial
    params: dict[str, Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.method(*args, **kwargs, **self.params)
        

