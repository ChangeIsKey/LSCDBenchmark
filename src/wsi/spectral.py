from sklearn.cluster import SpectralClustering

from src.target import Target
from src.use import UseID
from src.wsi.model import Model


class ClusterSpectral(Model):
    n_clusters: int

    class Config:
        arbitrary_types_allowed = True

    def predict_target(self, target: Target) -> dict[UseID, int]:
        clustering = SpectralClustering(
            n_clusters=self.n_clusters,
            assign_labels="kmeans",
            affinity="precomputed"
        )
        similarity_matrix = self.wic.similarity_matrix(target).to_numpy()
        labels = clustering.fit_predict(similarity_matrix)
        ids = target.uses_df.identifier.tolist()
        return dict(zip(ids, labels))
