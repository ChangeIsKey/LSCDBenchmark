from typing import Callable
from src.lscd.model import Model
from src.target import Target
from src import wsi
import scipy

class ClusterJSD(Model):
    threshold_fn: Callable[[list[float]], list[int]] | None
    wsi: wsi.Model

    def predict(self, targets: list[Target]) -> list[int] | list[float]:
        predictions = []
        for target in targets:
            clusters = self.wsi.predict(target)
            c1, c2 = self.wsi.split_clusters(clusters, target.grouping_to_useid())
            normalized_1, normalized_2 = self.wsi.normalize_cluster(c1), self.wsi.normalize_cluster(c2)
            jsd = scipy.spatial.distance.jensenshannon(normalized_1, normalized_2, base=2.0)
            predictions.append(jsd)

        if self.threshold_fn is not None:
            return self.threshold_fn(predictions)

        return predictions


 