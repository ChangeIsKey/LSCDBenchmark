import networkx as nx
import chinese_whispers as cw
from src.wsi.model import Model 
from src.target import Target

from src.use import UseID


class ClusterChineseWhispers(Model):
    n_iters: int

    def predict_target(self, target: Target) -> dict[UseID, int]:
        similarity_matrix = self.wic.similarity_matrix(target)
        ids = similarity_matrix.index
        G = nx.Graph()
        for id1 in ids:
            for id2 in ids:
                G.add_edge(id1, id2, weight=similarity_matrix.loc[id1, id2])
        cw.chinese_whispers(G, weighting="top", iterations=self.n_iters)
        new_ids = []
        new_labels = []
        for label, values in cw.aggregate_clusters(G).items():
            new_ids.extend(list(values))
            new_labels.extend([label for _ in range(len(list(values)))])
        return dict(zip(new_ids, new_labels))