from pydantic import BaseModel
from pathlib import Path
from typing import Optional
import src.utils as utils


class Orthography(BaseModel):
    translation_table: Optional[Path]
    normalize: bool = True

    def __init__(self, **data) -> None:
        if "translation_table" in data and data["translation_table"] is not None:
            data["translation_table"] = utils.path(data["translation_table"])
        super().__init__(**data)
