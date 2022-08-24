from pathlib import Path
from typing import Any, Callable

import hydra


def path(path: str) -> Path:
    return Path(hydra.utils.to_absolute_path(path))

def xor(a: bool, b: bool):
    return (a and not b) or (not a and b)