import numpy as np


def mean_std(
	predictions: list[float],
	t: float
) -> list[int]:
	"""Return a list of bool showing the positions of the predictions which are grater equal 
	than the threshold. The threshold equals the mean of predictions plus the product of parameter t 
	and std of predictions.

	:param predictions: a list of predictions
	:type predictions: list[float]
	:param t: input float
	:type t: float
	:return: a list of 0 and 1 showing the posision of the non-passed and passed predictions correspondingly
	:rtype: list[int]
	"""	
	mean = np.mean(predictions, axis=0)
	std = np.std(predictions, axis=0)
	threshold = mean + t * std

	return [int(p >= threshold) for p in predictions]
