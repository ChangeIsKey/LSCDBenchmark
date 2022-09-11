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
    distance_matrix = model.distance_matrix(target)
    labels = clustering.fit_predict(distance_matrix)
    return dict(zip(ids, labels))


def clustering_chinese_whispers(_: DistanceModel, target: Target) -> Dict[UseID, int]:
    # create the edges with their weights
    edges = []
    judgments = target.judgments.fillna(0)
    for _, item in judgments.iterrows():
        id1, id2 = item["identifier1"], item["identifier2"]
        records = judgments[
            ((judgments.identifier1 == id1) & (judgments.identifier2 == id2)) | 
            ((judgments.identifier1 == id2) & (judgments.identifier2 == id1))
        ]
        mean = records["judgment"].mean()
        edges.append((item["identifier1"], item["identifier2"], mean))
    
    G = nx.Graph()
    for id1, id2, weight in edges:
        G.add_edge(id1, id2, weight=weight)

    cw.chinese_whispers(G, weighting="top")
    new_ids = []
    new_labels = []
    for label, values in cw.aggregate_clusters(G).items():
        new_ids.extend(list(values))
        new_labels.extend([label for i in range(len(list(values)))])
    result = dict(zip(new_ids, new_labels))
    return result

