from typing import Callable, Dict, List, Tuple
from collections import Counter

from sklearn.cluster import SpectralClustering
import numpy as np

from src.config import UseID, Config
from src.lscd.target import Target
from src.vector_model import VectorModel


def split_clusters(clustering: Dict[UseID, int], target: Target) -> Tuple[np.ndarray, np.ndarray]:
    """
    splits clusters into two groups according to `groupings` parameter
    """

    groupings = target.grouping_combination
    grouping_to_uses = target.grouping_to_uses()


    clusters = list(clustering.values())
    
    counts = Counter(clusters)
    n_ids = len(target.uses.identifier.tolist())
    
    c1 = [clustering[id] for id in grouping_to_uses[groupings[0]]]
    for i, cluster in enumerate(c1):
        c1[i] = counts[cluster] / n_ids
        
    c2 = [clustering[id] for id in grouping_to_uses[groupings[1]]]
    for i, cluster in enumerate(c2):
        c2[i] = counts[cluster] / n_ids

    return np.array(c1), np.array(c2)        


def clustering_spectral(model: VectorModel, target: Target) -> Dict[UseID, int]:
    n_clusters = len(target.clusters.cluster.unique())
    clustering = SpectralClustering(n_clusters=n_clusters,
                                    assign_labels="kmeans", 
                                    random_state=0)

    ids = target.uses.identifier.tolist()
    vectors_usages = np.vstack([model.vectors[id] for id in ids])
    clustering.fit(vectors_usages)
    labels = clustering.labels_
    return dict(zip(ids, labels))
