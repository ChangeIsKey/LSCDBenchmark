from typing import Callable, Dict, List, Tuple
from collections import Counter

from sklearn.cluster import SpectralClustering
import numpy as np
import scipy

from src.config import UseID, Config
from src.target import Target
from src.vector_model import DistanceModel


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
