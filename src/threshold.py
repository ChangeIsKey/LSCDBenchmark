import numpy as np


def mean_std(
	predictions: list[float],
	t: float
) -> list[int]:
	"""Find the predictions passing the threshold. The threshold is the mean of predictions add the product of t and std of predictions.

	:param predictions: a list of predictions
	:type predictions: list[float]
	:param t: _description_
	:type t: float
	:return: a list of 0 and 1 showing the posision of the non-passed and passed predictions correspondingly
	:rtype: list[int]
	"""	
	mean = np.mean(predictions, axis=0)
	std = np.std(predictions, axis=0)
	threshold = mean + t * std

	return [int(p >= threshold) for p in predictions]
