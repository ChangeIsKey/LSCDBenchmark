from src.wsi.model import Model 
from src.target import Target
from sklearn.cluster import SpectralClustering

from src.use import UseID


class ClusterSpectral(Model):
    def predict(self, target: Target) -> dict[UseID, int]:
        n_clusters = len(target.clusters_df.cluster.unique())
        clustering = SpectralClustering(
            n_clusters=n_clusters, assign_labels="kmeans", affinity="precomputed"
        )
        ids = target.uses_df.identifier.tolist()
        distance_matrix = self.wic.distance_matrix(target).to_numpy()
        labels = clustering.fit_predict(distance_matrix)
        return dict(zip(ids, labels))
