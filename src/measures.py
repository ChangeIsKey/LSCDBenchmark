# from typing import Dict, Tuple
# import numpy as np
# import scipy

# from src.use import UseID
# from src.wic.bert import Bert
# from src.target import Target, Sampling, Pairing
# from src.clustering import split_clusters
# import logging

# log = logging.getLogger(__name__)


# def cos(target: Target, model: Bert) -> Dict[str, float]:
#     log.info(f"Computing COS for target `{target.name}`")
#     earlier = target.uses[
#         target.uses.grouping == target.grouping_combination[0]
#     ].identifier.tolist()
#     later = target.uses[
#         target.uses.grouping == target.grouping_combination[1]
#     ].identifier.tolist()
#     earlier_vectors = np.mean([model.vectors[use] for use in earlier], axis=0)
#     later_vectors = np.mean([model.vectors[use] for use in later], axis=0)
#     return {target.name: scipy.spatial.distance.cosine(earlier_vectors, later_vectors)}


# def semantic_proximity(
#     target: Target, model: DistanceModel
# ) -> Dict[Tuple[UseID, UseID], float]:
#     compare_pairs, compare_distances = model.distances(
#         target=target,
#         sampling=Sampling.ANNOTATED,
#         pairing=Pairing.COMPARE,
#         return_pairs=True,
#     )
#     later_pairs, later_distances = model.distances(
#         target=target,
#         sampling=Sampling.ANNOTATED,
#         pairing=Pairing.LATER,
#         return_pairs=True,
#     )
#     earlier_pairs, earlier_distances = model.distances(
#         target=target,
#         sampling=Sampling.ANNOTATED,
#         pairing=Pairing.EARLIER,
#         return_pairs=True,
#     )

#     return dict(
#         zip(
#             compare_pairs + later_pairs + earlier_pairs,
#             compare_distances + later_distances + earlier_distances,
#         )
#     )


# def apd_compare_all(target: Target, model: DistanceModel) -> Dict[str, float]:
#     return {
#         target.name: np.mean(
#             model.distances(
#                 target=target, sampling=Sampling.ALL, pairing=Pairing.COMPARE
#             )
#         ).item()
#     }


# def apd_earlier_all(target: Target, model: DistanceModel) -> Dict[str, float]:
#     return {
#         target.name: np.mean(
#             model.distances(
#                 target=target, sampling=Sampling.ALL, pairing=Pairing.EARLIER
#             )
#         ).item()
#     }


# def apd_later_all(target: Target, model: DistanceModel) -> Dict[str, float]:
#     return {
#         target.name: np.mean(
#             model.distances(target=target, sampling=Sampling.ALL, pairing=Pairing.LATER)
#         ).item()
#     }


# def apd_compare_annotated(target: Target, model: DistanceModel) -> Dict[str, float]:
#     return {
#         target.name: np.mean(
#             model.distances(
#                 target=target, sampling=Sampling.ANNOTATED, pairing=Pairing.COMPARE
#             )
#         ).item()
#     }


# def apd_later_annotated(target: Target, model: DistanceModel) -> Dict[str, float]:
#     return {
#         target.name: np.mean(
#             model.distances(
#                 target=target, sampling=Sampling.ANNOTATED, pairing=Pairing.LATER
#             )
#         ).item()
#     }


# def apd_earlier_annotated(target: Target, model: DistanceModel) -> Dict[str, float]:
#     return {
#         target.name: np.mean(
#             model.distances(
#                 target=target, sampling=Sampling.ANNOTATED, pairing=Pairing.EARLIER
#             )
#         ).item()
#     }


# def apd_compare_sampled(
#     target: Target, model: DistanceModel, n: int, replace: bool
# ) -> Dict[str, float]:
#     return {
#         target.name: np.mean(
#             model.distances(
#                 target=target,
#                 sampling=Sampling.SAMPLED,
#                 pairing=Pairing.COMPARE,
#                 n=n,
#                 replace=replace,
#             )
#         ).item()
#     }


# def apd_earlier_sampled(
#     target: Target, model: DistanceModel, n: int, replace: bool
# ) -> Dict[str, float]:
#     return {
#         target.name: np.mean(
#             model.distances(
#                 sampling.sampled(pairing.EARLIER, target, n=n, replace=replace)
#             )
#         ).item()
#     }


# def apd_later_sampled(
#     target: Target, model: DistanceModel, n: int, replace: bool
# ) -> Dict[str, float]:
#     return {
#         target.name: np.mean(
#             model.distances(
#                 sampling.sampled(pairing.LATER, target, n=n, replace=replace)
#             )
#         ).item()
#     }


# def apd_compare_all_minus_all_annotated(
#     target: Target, model: DistanceModel
# ) -> Dict[str, float]:
#     return {
#         target.name: (
#             np.mean(
#                 model.distances(
#                     target=target, sampling=Sampling.ANNOTATED, pairing=Pairing.COMPARE
#                 )
#             )
#             - np.mean(
#                 model.distances(
#                     target=target, sampling=Sampling.ANNOTATED, pairing=Pairing.COMPARE
#                 )
#                 + model.distances(
#                     target=target, sampling=Sampling.ANNOTATED, pairing=Pairing.LATER
#                 )
#                 + model.distances(
#                     target=target, sampling=Sampling.ANNOTATED, pairing=Pairing.EARLIER
#                 )
#             )
#         )
#     }


# diasense = apd_compare_all_minus_all_annotated


# def cluster_jsd(
#     target: Target, model: DistanceModel, clustering: Clustering
# ) -> Dict[str, float]:
#     clusters = clustering(model, target)
#     c1, c2 = split_clusters(clusters, target)

#     return {target.name: scipy.spatial.distance.jensenshannon(c1, c2, base=2.0)}
