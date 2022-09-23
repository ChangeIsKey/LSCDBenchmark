from pathlib import Path
from typing import Any, TypeGuard

import hydra
import networkx as nx
import numpy as np

from src.lscd.lscd_model import LSCDModel


def path(path: str) -> Path:
    return Path(hydra.utils.to_absolute_path(path)).resolve()

def xor(a: bool, b: bool):
    return (a and not b) or (not a and b)

def _check_nan_weights_exits(graph: nx.Graph):
    """Check if there are NaN weights in the graph.
    Parameters
    ----------
    graph: networkx.Graph
        The graph to check NaN weights for
    Returns
    -------
    flag: bool
        True if there are NaN weights, False otherwise
    """
    return any(np.isnan(graph.get_edge_data(*edge)['weight']) for edge in graph.edges)


# Typeguards

def is_str_list(obj: list[Any] | None) -> TypeGuard[list[str]]:
    try:
        return obj is not None and all(isinstance(s, str) for s in obj)
    except TypeError:
        return False

def is_float(obj: object | None) -> TypeGuard[float]:
    try:
        return isinstance(obj, float)
    except TypeError:
        return False

def is_int(obj: object | None) -> TypeGuard[int]:
    try:
        return isinstance(obj, int)
    except TypeError:
        return False

def is_lscd_model(obj: object) -> TypeGuard[LSCDModel]:
    try:
        return isinstance(obj, LSCDModel)
    except TypeError:
        return False