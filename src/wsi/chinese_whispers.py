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
        graph = nx.Graph()
        for i in similarity_matrix:
            for j in similarity_matrix:
                graph.add_edge(i, j, weight=similarity_matrix[i, j])
        cw.chinese_whispers(graph, weighting="top", iterations=self.n_iters)
        new_ids = []
        new_labels = []
        for label, values in cw.aggregate_clusters(graph).items():
            new_ids.extend(list(values))
            new_labels.extend([label for _ in range(len(list(values)))])
        return new_labels
