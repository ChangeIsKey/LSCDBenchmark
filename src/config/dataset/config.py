from pydantic import BaseModel
from typing import Optional, TYPE_CHECKING
from pathlib import Path
from src.config.dataset.cleaning import Cleaning
from src.config.dataset.orthography import Orthography
from src.config.dataset.preprocessing import Preprocessing
import src.utils as utils
import json
import time


if TYPE_CHECKING:
    from src.config.config import str


class DatasetConfig(BaseModel):
    cleaning: Cleaning | None
    orthography: Orthography | None
    preprocessing: Preprocessing
    targets: list[str] | int | None
    groupings: tuple[str, str]
    version: str
    name: Optional[str] = None
    path: Optional[Path] = None

    @property
    def wug_to_url(self) -> dict[str, dict[str, str]]:
        path = utils.path("datasets.json")
        with path.open(mode="r") as f:
            return json.load(f)
    
    def __init__(self, **data) -> None:
        
        if data["version"] is None:
            data["version"] = "latest"

        if data["path"] is not None and data["name"] is None:
            data["path"] = utils.path(data["path"])
            data["name"] = data["path"].name
            data["version"] = time.ctime(data["path"].stat().st_mtime)

        if data["version"] == "latest" and data["name"] is not None:
            versions = sorted(self.wug_to_url[data["name"]].keys(), reverse=True)
            data["version"] = versions[0]
            data["path"] = utils.path("wug") / data["name"] / data["version"]
        
        super().__init__(**data)

    

