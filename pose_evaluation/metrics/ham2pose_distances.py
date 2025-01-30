import numpy as np
from numpy.ma.core import MaskedArray
from scipy.spatial.distance import euclidean
from pose_evaluation.metrics.distance_measure import (
    AggregatePointWiseThenAggregateTrajectorywiseDistance,
    DistanceAggregator,
)


class Ham2PoseAPEDistance(AggregatePointWiseThenAggregateTrajectorywiseDistance):
    def __init__(self) -> None:
        super().__init__(
            None,
            pointwise_aggregator=None,
            trajectorywise_aggregator=DistanceAggregator("sum"),
        )
        self.name = "ham2pose_ape"

    def get_trajectory_pair_distance(
        self, hyp_traj: MaskedArray, ref_traj: MaskedArray
    ) -> float:
        return ham2pose_ape(hyp_traj, ref_traj)


class Ham2PoseMSEDistance(AggregatePointWiseThenAggregateTrajectorywiseDistance):
    def __init__(self) -> None:
        super().__init__(
            None,
            pointwise_aggregator=None,
            trajectorywise_aggregator=DistanceAggregator("sum"),
        )
        self.name = "ham2pose_mse"

    def get_trajectory_pair_distance(
        self, hyp_traj: MaskedArray, ref_traj: MaskedArray
    ) -> float:
        return ham2pose_mse(hyp_traj, ref_traj)


class Ham2PoseMaskedEuclideanDistance(
    AggregatePointWiseThenAggregateTrajectorywiseDistance
):

    def __init__(self) -> None:
        super().__init__(
            pointwise_distance_function=ham2pose_masked_euclidean,
            pointwise_aggregator=DistanceAggregator("mean"),
            trajectorywise_aggregator=DistanceAggregator("mean"),
        )
        self.name = "ham2pose_masked_euclidean"
        self.mask_strategy = "ref_return_0,hyp_set_to_origin"
        self.pointwise_dist = "euclidean"

    # def get_signature(self) -> str:
    #     return "name:ham2pose_masked_euclidean|mask_strategy:ref_return_0,hyp_set_to_origin|dist:euclidean"


def ham2pose_masked_euclidean(hyp_point: MaskedArray, ref_point: MaskedArray) -> float:
    if np.ma.is_masked(ref_point):  # reference label keypoint is missing
        return 0
    elif np.ma.is_masked(
        hyp_point
    ):  # reference label keypoint is not missing, other label keypoint is missing
        return euclidean((0, 0, 0), ref_point) / 2
    d = euclidean(hyp_point, ref_point)
    return d


def ham2pose_ape(hyp_traj, ref_traj):
    sq_error = np.power(hyp_traj - ref_traj, 2).sum(-1)
    return np.sqrt(sq_error).mean()


def ham2pose_mse(hyp_traj, ref_traj):
    sq_error = np.power(hyp_traj - ref_traj, 2).sum(-1)
    return sq_error.mean()
