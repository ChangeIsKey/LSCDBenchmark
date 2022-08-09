import hydra
from pathlib import Path


def path(path: str) -> Path:
    return Path(hydra.utils.to_absolute_path(path))
