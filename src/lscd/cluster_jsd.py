import scipy
from tqdm import tqdm

from src import wsi
from src.lscd.model import GradedModel
from src.target import Lemma


class ClusterJSD(GradedModel):
    wsi: wsi.Model

    def predict(self, lemma: Lemma) -> float:
        clusters = self.wsi.predict(lemma.get_uses())
        c1, c2 = self.wsi.make_freq_dists(
            clusters, lemma.useid_to_grouping(), lemma.groupings
        )
        return scipy.spatial.distance.jensenshannon(c1, c2, base=2.0)  # type: ignore
