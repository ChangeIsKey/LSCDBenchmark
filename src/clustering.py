from typing import Callable, Dict, List, Tuple
from collections import Counter

from sklearn.cluster import SpectralClustering
import numpy as np
import scipy
import networkx as nx
import chinese_whispers as cw

from src.config import UseID, Config, pairing, sampling
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


def clustering_chinese_whispers(model: DistanceModel, target: Target) -> Dict[UseID, int]:
    ids = []
    G = nx.Graph()
    # create the edges with their weights
    compare_ids = sampling.annotated(pairing.COMPARE)
    earlier_ids = sampling.annotated(pairing.EARLIER)
    later_ids = sampling.annotated(pairing.LATER)
    pair_ids = Counter()

    for r, item in target.judgments.iterrows():
        mean = target.judgments[
            ((target.judgments.identifier1 == item["identifier1"]) & 
            (target.judgments.identifier2 == item["identifier2"])) | 
            ((target.judgments.identifier1 == item["identifier2"]) & 
            (target.judgments.identifier2 == item["identifier1"]))]["judgment"].mean()
        pair_ids[(item["identifier1"], item["identifier2"])] = mean

    edges = [(id1, id2, value) for (id1, id2), value in pair_ids.items()]
    G.add_weighted_edges_from(edges)
    cw.chinese_whispers(G, weighting="top", seed=1337)
    new_ids = []
    new_labels = []
    for label, values in cw.aggregate_clusters(G).items():
        new_ids.extend(list(values))
        new_labels.extend([label for i in range(len(list(values)))])
    return dict(zip(new_ids, new_labels))

