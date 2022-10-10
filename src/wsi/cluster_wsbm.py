from collections import Counter
from typing import Literal

import networkx as nx
from graph_tool.inference.blockmodel import BlockState

from src import utils
from src.target import Target
from src.use import UseID
from src.wsi.model import Model


class ClusterWSBM(Model):
	distribution: Literal[
		"real-exponential", "discrete-poisson", "discrete-geometric", "discrete-binomial", "real-normal",]

	def predict_target(
		self,
		target: Target
	) -> dict[UseID, int]:
		similarity_matrix = self.wic.similarity_matrix(target)
		ids = similarity_matrix.index
		graph = nx.Graph()
		for id1 in ids:
			for id2 in ids:
				graph.add_edge(id1, id2, weight=(similarity_matrix.loc[id1, id2]))

		if utils._negative_weights_exist(graph):
			raise ValueError(
				"Negative weights are not supported by the WSBM algorithm."
			)

		if utils._check_nan_weights_exits(graph):
			raise ValueError("NaN weights are not supported by the WSBM algorithm.")

		gt_graph, _, gt2nx = utils._nxgraph_to_graphtoolgraph(graph.copy())
		state: BlockState = utils._minimize(gt_graph, self.distribution)

		block2clusterid_map = {}
		for i, (k, _) in enumerate(
				dict(
					sorted(
						Counter(state.get_blocks().get_array()).items(), key=lambda
							item: item[1], reverse=True, )
				).items()
		):
			block2clusterid_map[k] = i

		communities = {}
		for i, block_id in enumerate(state.get_blocks().get_array()):
			nx_vertex_id = gt2nx[i]
			community_id = block2clusterid_map[block_id]
			if communities.get(community_id, None) is None:
				communities[community_id] = []
			communities[community_id].append(nx_vertex_id)

		classes = [set(v) for _, v in communities.items()]

		return {use: i for i, set_ in enumerate(classes) for use in set_}
