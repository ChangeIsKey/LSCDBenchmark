from pathlib import Path
from typing import Any, TypeGuard

import hydra
import networkx as nx
import numpy as np
import graph_tool
from graph_tool.inference import minimize_blockmodel_dl
from graph_tool.inference.blockmodel import BlockState


class ShouldNotHappen(Exception):
    ...


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


def _negative_weights_exist(graph: nx.Graph):
    """Check if there are negative edges in the graph.
    Parameters
    ----------
    graph: networkx.Graph
        The graph to check negative edges for
    Returns
    -------
    flag: bool
        True if there are negative edges, False otherwise
    """
    for i, j in graph.edges():
        if graph[i][j]["weight"] < 0:
            return True
    return False


def _nxgraph_to_graphtoolgraph(graph: nx.Graph):
    """Convert a networkx graph to a graphtool graph.
    Parameters
    ----------
    graph: networkx.Graph
        The graph to convert
    Returns
    -------
    gt_graph: graphtool.Graph
        The converted graph
    """
    graph_tool_graph = graph_tool.Graph(directed=False)

    nx2gt_vertex_id = dict()
    gt2nx_vertex_id = dict()
    for i, node in enumerate(graph.nodes()):
        nx2gt_vertex_id[node] = i
        gt2nx_vertex_id[i] = node

    new_weights = []
    for i, j in graph.edges():
        current_weight = graph[i][j]["weight"]
        if current_weight != 0 and not np.isnan(current_weight):
            graph_tool_graph.add_edge(nx2gt_vertex_id[i], nx2gt_vertex_id[j])
            new_weights.append(current_weight)

    original_edge_weights = graph_tool_graph.new_edge_property("double")
    original_edge_weights.a = new_weights
    graph_tool_graph.ep["weight"] = original_edge_weights

    new_vertex_id = graph_tool_graph.new_vertex_property("string")
    for k, v in nx2gt_vertex_id.items():
        new_vertex_id[v] = str(k)
    graph_tool_graph.vp.id = new_vertex_id

    return graph_tool_graph, nx2gt_vertex_id, gt2nx_vertex_id


def _minimize(graph: graph_tool.Graph, distribution: str) -> BlockState:
    """Minimize the graph using the given distribution as described by graph-tool.
    Parameters
    ----------
    graph: graphtool.Graph
        The graph to minimize
    distribution: str
        The distribution to use for the WSBM algorithm.
    Returns
    -------
    state: BlockState
        The minimized graph as BlockState object.
    """

    return minimize_blockmodel_dl(
        graph,
        state_args=dict(
            deg_corr=False, recs=[graph.ep.weight], rec_types=[distribution]
        ),
        multilevel_mcmc_args=dict(
            B_min=1,
            B_max=30,
            niter=100,
            entropy_args=dict(adjacency=False, degree_dl=False),
        ),
    )

# Typeguards

def is_list(obj: Any) -> TypeGuard[list[Any]]:
    return hasattr(obj, "__len__")
    

def is_str_list(obj: Any) -> TypeGuard[list[str]]:
    try:
        return obj is not None and all(isinstance(s, str) for s in obj)
    except TypeError:
        return False

def is_number(obj: Any) -> TypeGuard[float | int]:
    try:
        return isinstance(obj, float) or isinstance(obj, int)
    except TypeError:
        return False


def is_int(obj: Any) -> TypeGuard[int]:
    try:
        return isinstance(obj, int)
    except TypeError:
        return False