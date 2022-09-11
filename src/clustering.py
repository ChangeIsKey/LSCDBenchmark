from collections import Counter
from typing import Callable, Dict, List, Tuple

import chinese_whispers as cw
import networkx as nx
import numpy as np
import scipy
from sklearn.cluster import SpectralClustering

from src.config import Config, UseID, pairing, sampling
from src.target import Target
from src.vector_model import DistanceModel, VectorModel
from src._correlation import cluster_correlation_search
from src.utils import _check_nan_weights_exits


def split_clusters(clustering: Dict[UseID, int], target: Target) -> Tuple[np.ndarray, np.ndarray]:
    """
    splits clusters into two groups according to `groupings` parameter
    """

    groupings = target.grouping_combination
    grouping_to_uses = target.grouping_to_uses()

    groups = [
        [clustering[id] for id in grouping_to_uses[groupings[0]]],
        [clustering[id] for id in grouping_to_uses[groupings[1]]]
    ]

    for i, group in enumerate(groups):
        counts = Counter(group)
        n = len(group)
        for j, cluster in enumerate(group):
            groups[i][j] = counts[cluster] / n

    return tuple(groups)
 

def clustering_spectral(model: DistanceModel, target: Target) -> Dict[UseID, int]:
    n_clusters = len(target.clusters.cluster.unique())
    clustering = SpectralClustering(
        n_clusters=n_clusters, 
        assign_labels="kmeans",
        affinity="precomputed"
    )

    ids = target.uses.identifier.tolist()
    distance_matrix = model.distance_matrix(target).to_numpy()
    labels = clustering.fit_predict(distance_matrix)
    return dict(zip(ids, labels))


def clustering_chinese_whispers(model: DistanceModel, target: Target) -> Dict[UseID, int]:
    distance_matrix = model.distance_matrix(target)
    ids = distance_matrix.index
    G = nx.Graph()
    for id1 in ids:
        for id2 in ids:
            G.add_edge(id1, id2, weight=distance_matrix.loc[id1, id2])
    cw.chinese_whispers(G, weighting="top")
    new_ids = []
    new_labels = []
    for label, values in cw.aggregate_clusters(G).items():
        new_ids.extend(list(values))
        new_labels.extend([label for _ in range(len(list(values)))])
    result = dict(zip(new_ids, new_labels))
    return result


def correlation_clustering(
    model: DistanceModel, 
    target: Target,
    **params,
):

    # max_senses: int = 10,
    # max_attempts: int = 200,
    # max_iters: int = 5000,
    # initial: list = [],
    # split_flag=True

    """Clusters the graph using the correlation clustering algorithm.
    Parameters
    ----------
    graph: networkx.Graph
        The graph for which to calculate the cluster labels
    max_senses: int
        The maximum number of senses a word can have
    max_attempts: int
        Number of restarts for optimization
    max_iters: int
        Maximum number of iterations for optimization
    initial: list
        Initial cluster labels (optional)
    split_flag: bool
        If True, non evidence clusters will be splitted
    Returns
    -------
    classes : list[Set[int]]
        A list of sets of nodes, where each set is a cluster
    Raises
    ------
    ValueError
        If the graph contains non-value weights
    """

    distance_matrix = model.distance_matrix(target)
    ids = distance_matrix.index
    G = nx.Graph()
    for id1 in ids:
        for id2 in ids:
            G.add_edge(id1, id2, weight=distance_matrix.loc[id1, id2])

    if _check_nan_weights_exits(G):
        raise ValueError(
            "NaN weights are not supported by the correlation clustering method."
        )

    clusters, _ = cluster_correlation_search(G, **params)

    return {id_: i for i, cluster in enumerate(clusters) for id_ in cluster}