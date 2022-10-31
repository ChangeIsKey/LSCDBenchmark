import chinese_whispers as cw
import networkx as nx
from itertools import combinations

from src.lemma import Lemma
from src.use import Use
from src.wsi.model import WSIModel


class ClusterChineseWhispers(WSIModel):
    n_iters: int

    def predict(self, uses: list[Use]) -> list[int]:
        use_pairs = list(combinations(uses, r=2))
        similarity_matrix = self.similarity_matrix(use_pairs)
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
        return new_labels
