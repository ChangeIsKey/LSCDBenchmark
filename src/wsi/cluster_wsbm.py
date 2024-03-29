from collections import Counter
from itertools import combinations
from typing import Literal

import graph_tool
import networkx as nx
import numpy as np
from graph_tool.inference.blockmodel import BlockState
from src.use import Use

from src.utils import utils
from src.wsi.model import WSIModel


def _nxgraph_to_graphtoolgraph(graph: nx.Graph):
    """Convert a networkx graph to a graphtool graph.

    :param graph: The graph to convert
    :type graph: nx.Graph
    :return: The converted graph
    :rtype: graphtool.Graph
            
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

    :param graph: The graph to minimize
    :type graph: graph_tool.Graph
    :param distribution: The distribution to use for the WSBM algorithm.
    :type distribution: str
    :return: The minimized graph as BlockState object.
    :rtype: BlockState
            
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


class ClusterWSBM(WSIModel):
    """ """
    distribution: Literal[
        "real-exponential",
        "discrete-poisson",
        "discrete-geometric",
        "discrete-binomial",
        "real-normal",
    ]

    def predict(self, uses: list[Use]) -> list[int]:
        """ """        
        use_pairs = list(combinations(uses, r=2))
        similarity_matrix = self.similarity_matrix(use_pairs)
        ids = similarity_matrix.index
        graph = nx.Graph()
        for id1 in ids:
            for id2 in ids:
                graph.add_edge(id1, id2, weight=(similarity_matrix.loc[id1, id2]))

        if utils._negative_weights_exist(graph):
            raise ValueError(
                "Negative weights are not supported by the WSBM algorithm."
            )

        if utils._check_nan_weights_exits(graph):
            raise ValueError("NaN weights are not supported by the WSBM algorithm.")

        gt_graph, _, gt2nx = _nxgraph_to_graphtoolgraph(graph.copy())
        state: BlockState = _minimize(gt_graph, self.distribution)

        block2clusterid_map = {}
        for i, (k, _) in enumerate(
            dict(
                sorted(
                    Counter(state.get_blocks().get_array()).items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ).items()
        ):
            block2clusterid_map[k] = i

        communities = {}
        for i, block_id in enumerate(state.get_blocks().get_array()):
            nx_vertex_id = gt2nx[i]
            community_id = block2clusterid_map[block_id]
            if communities.get(community_id, None) is None:
                communities[community_id] = []
            communities[community_id].append(nx_vertex_id)

        classes = [set(v) for _, v in communities.items()]

        return [i for i, _ in enumerate(classes)]
