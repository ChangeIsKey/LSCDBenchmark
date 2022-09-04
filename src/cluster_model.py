from typing import Callable, Dict, List, Tuple

from sklearn.cluster import SpectralClustering
import numpy as np

from src.config import UseID, Config
from src.lscd.target import Target


class ClusterModel:
    def __init__(
        self, use_pairs: List[Tuple[UseID, UseID]], distances: List[float]
    ) -> None:
        self.use_pairs = use_pairs
        self.distances = distances

    # def cluster(self, method: Callable, **kwargs) -> Dict[ID, int]:
    #     """
    #     method: a clustering algorithm
    #     returns: a mapping from context identifiers to cluster labels
    #     """
    #     # 1. construct distance matrix
    #     # 2. cluster distance matrix
    #     # 3. return clusters (mapping from use ids to clusters)
    #     pass

    def split(
        self,
        clustering: Dict[UseID, int],
        target: Target,
    ):
        """
        splits clusters into two groups according to `groupings` parameter
        """

        groupings = target.grouping_combination
        grouping_to_uses = target.grouping_to_uses()

        c1 = [clustering[id] for id in grouping_to_uses[groupings[0]] 
        c2 = [clustering[id] for id in grouping_to_uses[groupings[1]]

        return c1, c2        
       
   
def clustering_Spectral(modelVector, target, **params) -> Dict[UseID, int]:
    n_clusters = len(target.clusters.cluster.unique())
    clustering = SpectralClustering(n_clusters=n_clusters,
                                    assign_labels="discretize", 
                                    random_state=42)

    ids = target.uses.identifier.tolist()
    vectors_usages = np.array([modelVector.vectors[id] for id in ids])

    clustering.fit(vectors_usages)
    n_labels = clustering.labels_
    return dict(zip(ids, n_labels))
