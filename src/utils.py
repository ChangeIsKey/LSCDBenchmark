from pathlib import Path

import hydra
import networkx as nx
import numpy as np


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
