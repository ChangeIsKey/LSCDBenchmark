from typing import Any
from itertools import combinations

import networkx as nx

from src.utils import utils
from src._correlation import cluster_correlation_search
from src.lemma import Lemma
from src.use import UseID, Use
from src.wsi.model import WSIModel


class ClusterCorrelation(WSIModel):
    max_senses: int
    max_attempts: int
    max_iters: int
    initial: list[Any]
    split_flag: bool

    def predict(self, uses: list[Use]) -> list[int]:
        use_pairs = list(combinations(uses, r=2))
        similarity_matrix = self.similarity_matrix(use_pairs)
        ids = similarity_matrix.index
        graph = nx.Graph()
        for id1 in ids:
            for id2 in ids:
                graph.add_edge(id1, id2, weight=similarity_matrix.loc[id1, id2])

        if utils._check_nan_weights_exits(graph):
            raise ValueError(
                "NaN weights are not supported by the correlation clustering method."
            )

        clusters, _ = cluster_correlation_search(
            graph=graph,
            max_senses=self.max_senses,
            max_attempts=self.max_attempts,
            max_iters=self.max_iters,
            initial=self.initial,
            split_flag=self.split_flag,
        )

        return [i for i, cluster in enumerate(clusters)]
