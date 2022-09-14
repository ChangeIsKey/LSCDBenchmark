from ast import Call
from pydantic import BaseModel, Field
from typing import Optional, Any, Callable
import importlib
import src.utils as utils
import sys


class CustomMethod(BaseModel):
    module: str
    method: Optional[str]
    params: Optional[dict[str, Any]] = None
    default: Optional[Callable] = Field(exclude=True, default=None)
    func: Optional[Callable] = Field(exclude=True, default=None)
    
    def __init__(self, **data) -> None:
        if "params" not in data or data["params"] is None:
            data["params"] = {}

        module = utils.path(data["module"])
        spec = importlib.util.spec_from_file_location(
            name=module.stem,
            location=module,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        data["func"] = (
            data["default"] if data["method"] is None
            else getattr(module, data["method"])
        )
        super().__init__(**data)

        



    
