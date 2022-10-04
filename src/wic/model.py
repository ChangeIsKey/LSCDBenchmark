from abc import (
	ABC,
	abstractmethod,
)

import numpy as np
from pandas import DataFrame
from pydantic import BaseModel

from src.target import Target
from src.use import (
	Use,
	UseID,
)


class Model(BaseModel, ABC):
	@abstractmethod
	def similarities(
		self,
		use_pairs: list[tuple[Use, Use]]
	) -> list[float]:
		...

	def predict(
		self,
		targets: list[Target]
	) -> dict[tuple[UseID, UseID], float]:
		predictions = {}
		for target in targets:
			use_pairs = (target.use_pairs(pairing="COMPARE", sampling="annotated") + target.use_pairs(
				pairing="EARLIER", sampling="annotated"
			) + target.use_pairs(pairing="LATER", sampling="annotated"))
			use_pairs_ids = [(use_1.identifier, use_2.identifier) for use_1, use_2 in use_pairs]
			predictions.update(dict(zip(use_pairs_ids, self.similarities(use_pairs))))
		return predictions

	def similarity_matrix(
		self,
		target: Target
	):
		compare_pairs = target.use_pairs(sampling="all", pairing="COMPARE")
		later_pairs = target.use_pairs(sampling="all", pairing="LATER")
		earlier_pairs = target.use_pairs(sampling="all", pairing="EARLIER")
		pairs = compare_pairs + later_pairs + earlier_pairs

		predictions = self.similarities(pairs)
		pairs_to_similarities = dict(zip(pairs, predictions))

		# get a sorted list of unique uses
		uses = sorted(
			{use for use_pair in list(pairs_to_similarities.keys()) for use in use_pair}
		)
		ids = [use.identifier for use in uses]
		n_ids = len(ids)

		similarity_matrix = np.zeros((n_ids, n_ids))
		for i, use_1 in enumerate(uses):
			for j, use_2 in enumerate(uses):
				try:
					similarity_matrix[i, j] = pairs_to_similarities[(use_1, use_2)]
				except KeyError:
					similarity_matrix[i, j] = pairs_to_similarities[(use_2, use_1)]

		return DataFrame(similarity_matrix, index=ids, columns=ids)
