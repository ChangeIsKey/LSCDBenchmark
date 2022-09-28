from src.lscd.model import GradedModel
from src.target import Target
from src import wsi
import scipy


class ClusterJSD(GradedModel):
    wsi: wsi.Model

    def predict(self, targets: list[Target]) -> dict[str, float]:
        predictions = {}
        for target in targets:
            clusters = self.wsi.predict_target(target)
            c1, c2 = self.wsi.make_freq_dists(clusters, target.useid_to_grouping(), target.groupings)
            jsd = scipy.spatial.distance.jensenshannon(c1, c2, base=2.0)  # type: ignore
            predictions[target.name] = jsd

        return predictions
