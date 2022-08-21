from pathlib import Path
from typing import Any, Callable

import hydra


def path(path: str) -> Path:
    return Path(hydra.utils.to_absolute_path(path))