

from typing import Callable, Dict, List, Tuple

from src.config import ID, Config


class ClusterModel:
    def __init__(self, use_pairs: List[Tuple[ID, ID]], distances: List[float]) -> None:
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
    
    def split(self, groupings: Tuple[int, int], clustering: Dict[ID, int], uses_to_groupings: Dict[ID, int]):
        """
        splits clusters into two groups according to `groupings` parameter
        """
        pass

    @staticmethod
    def clustering_method_1(self, n_clusters: int) -> Dict[ID, int]:
        pass

