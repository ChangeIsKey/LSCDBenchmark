from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional, Any
from pandas import Series
from hydra import utils


class Preprocessing(BaseModel):
    target: str = Field(alias="_target_")
    spelling_normalization: Optional[dict[str, str]] = Field(default_factory=dict)
    params: Optional[dict[str, Any]] = Field(default_factory=dict)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.spelling_normalization is not None:
            self.spelling_normalization = {k.replace("_", " "): v for k, v in self.spelling_normalization.items()}
    
    def __call__(self, s: Series, translation_table: dict[str, str]) -> Series:
        asdict = {"_target_": self.target, "s": s, "translation_table": translation_table, **self.params}
        context, start, end = utils.instantiate(asdict)
        return Series(
            {
                "context_preprocessed": context,
                "target_index_begin": start,
                "target_index_end": end,
            }
        )