from itertools import combinations

from sklearn.cluster import SpectralClustering

from src.use import Use
from src.wsi.model import WSIModel


class ClusterSpectral(WSIModel):
    n_clusters: int

    class Config:
        arbitrary_types_allowed = True

    def predict(self, uses: list[Use]) -> list[int]:
        clustering = SpectralClustering(
            n_clusters=self.n_clusters, assign_labels="kmeans", affinity="precomputed"
        )
        use_pairs = set(combinations(uses, r=2))
        similarity_matrix = self.similarity_matrix(use_pairs)
        labels = clustering.fit_predict(similarity_matrix)
        return labels
