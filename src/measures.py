from typing import Callable
import numpy as np
import scipy
import sklearn

from src.config import pairing, sampling
from src.distance_model import DistanceModel
from src.lscd.target import Target
from src.cluster_model import ClusterModel


def apd_compare_all(target: Target, model: DistanceModel) -> float:
    return np.mean(model.distances(target=target, sampling=sampling.all, pairing=pairing.COMPARE)).item()


def apd_merge_all(target: Target, model: DistanceModel) -> float:
    distances = model.distances(target=target, sampling=sampling.all, pairing=pairing.COMPARE) \
                + model.distances(target=target, sampling=sampling.all, pairing=pairing.LATER) \
                + model.distances(target=target, sampling=sampling.all, pairing=pairing.EARLIER)
    return np.mean(distances).item()


def apd_earlier_all(target: Target, model: DistanceModel) -> float:
    return np.mean(model.distances(target=target, sampling=sampling.all, pairing=pairing.EARLIER)).item()


def apd_later_all(target: Target, model: DistanceModel) -> float:
    return np.mean(model.distances(target=target, sampling=sampling.all, pairing=pairing.LATER)).item()


def apd_compare_annotated(target: Target, model: DistanceModel) -> float:
    return np.mean(model.distances(target=target, sampling=sampling.annotated, pairing=pairing.COMPARE)).item()

def apd_merge_annotated(target: Target, model: DistanceModel) -> float:
    distances = model.distances(target=target, sampling=sampling.annotated, pairing=pairing.COMPARE) \
                + model.distances(target=target, sampling=sampling.annotated, pairing=pairing.LATER) \
                + model.distances(target=target, sampling=sampling.annotated, pairing=pairing.EARLIER)
    return np.mean(distances).item()

def apd_later_annotated(target: Target, model: DistanceModel) -> float:
    return np.mean(model.distances(target=target, sampling=sampling.annotated, pairing=pairing.LATER))


def apd_earlier_annotated(target: Target, model: DistanceModel) -> float:
    return np.mean(model.distances(target=target, sampling=sampling.annotated, pairing=pairing.EARLIER))


def apd_compare_sampled(
        target: Target, model: DistanceModel, n: int, replace: bool
) -> float:
    return np.mean(
        model.distances(target=target, sampling=sampling.sampled, pairing=pairing.COMPARE, n=n, replace=replace)
    )


def apd_compare_sampled(
        target: Target, model: DistanceModel, n: int, replace: bool
) -> float:
    return np.mean(
        model.distances(sampling.sampled(pairing.COMPARE, target, n=n, replace=replace))
    )


def apd_compare_all_minus_all_annotated(target: Target, model: DistanceModel) -> float:
    return np.mean(
        model.distances(target=target, sampling=sampling.annotated, pairing=pairing.COMPARE)
    ) - np.mean(
                model.distances(target=target, sampling=sampling.annotated, pairing=pairing.COMPARE) \
                + model.distances(target=target, sampling=sampling.annotated, pairing=pairing.LATER) \
                + model.distances(target=target, sampling=sampling.annotated, pairing=pairing.EARLIER)
        )


diasense = apd_compare_all_minus_all_annotated


def cluster_jsd_merge_all(target: Target, model: DistanceModel, **clustering_params):
    compare_pairs, compare_distances = model.distances(target=target, sampling=sampling.all, pairing=pairing.COMPARE, return_pairs=True)
    later_pairs, later_distances = model.distances(target=target, sampling=sampling.all, pairing=pairing.LATER, return_pairs=True)
    earlier_pairs, earlier_distances = model.distances(target=target, sampling=sampling.all, pairing=pairing.EARLIER, return_pairs=True)
    
    cluster_model = ClusterModel(
        use_pairs=compare_pairs + later_pairs + earlier_pairs, 
        distances=compare_distances + later_distances + earlier_distances
    )

    # QUESTION how does the method passed to ClusterModel.cluster look like?
    # Is it an sklearn function? Is it a function that calls a predefined method (from sklearn or some other library)?
    
    clusters = cluster_model.cluster(
        # the method is an example
        method=sklearn.cluster.KMeans,
        **clustering_params
    )

    c1, c2 = cluster_model.split(target.grouping_combination, clusters, target.uses_to_grouping())
    return scipy.spatial.distance.jensenshannon(c1, c2, base=2.0)
