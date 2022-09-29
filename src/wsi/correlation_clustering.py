from typing import Any

import networkx as nx

from src import utils
from src._correlation import cluster_correlation_search
from src.target import Target
from src.use import UseID
from src.wsi.model import Model


class ClusterCorrelation(Model):
    max_senses: int
    max_attempts: int
    max_iters: int
    initial: list[Any]
    split_flag: bool

    def predict_target(self, target: Target) -> dict[UseID, int]:

        similarity_matrix = self.wic.similarity_matrix(target)
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
            G=graph,
            max_senses=self.max_senses,
            max_attempts=self.max_attempts,
            max_iters=self.max_iters,
            initial=self.initial,
            split_flag=self.split_flag
        )

        return {id_: i for i, cluster in enumerate(clusters) for id_ in cluster}
