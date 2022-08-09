from pathlib import Path

import hydra


def path(path: str) -> Path:
    return Path(hydra.utils.to_absolute_path(path))
