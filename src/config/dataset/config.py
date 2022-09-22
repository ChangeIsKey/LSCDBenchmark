from typing import Optional, Any
from pathlib import Path
import json
from pydantic import BaseModel, Field

from src.config.dataset.cleaning import Cleaning
from src.config.dataset.preprocessing import Preprocessing
import src.utils as utils


class DatasetConfig(BaseModel):
    name: str
    cleaning: Optional[Cleaning] = Field(default_factory=lambda: None)
    preprocessing: Optional[Preprocessing] = Field(default_factory=lambda: None)
    targets: Optional[list[str] | int] = Field(default_factory=lambda: None)
    groupings: tuple[str, str] = Field(default_factory=lambda: [1, 2])
    version: Optional[str] = Field(default_factory=lambda: "latest", alias="version")
    path: Optional[Path] = Field(default_factory=lambda: None)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        wug_to_url = self.wug_to_url
        if self.version is None:
            self.version = "latest"
        
        if self.version == "latest" and self.name in wug_to_url:
            versions = sorted(wug_to_url[self.name].keys(), reverse=True)
            self.version = versions[0]
        

    @property
    def wug_to_url(self) -> dict[str, dict[str, str]]:
        path = utils.path("datasets.json")
        with path.open(mode="r") as f:
            return json.load(f)
    