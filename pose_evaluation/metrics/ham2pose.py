# https://github.com/sign-language-processing/pose-evaluation/issues/31
import numpy as np
import numpy.ma as ma
from fastdtw import fastdtw  # type: ignore
from tqdm import tqdm

from pose_evaluation.metrics.distance_measure import AggregatedDistanceMeasure


# class Ham2Pose_MSE(AggregatedDistanceMetric):
#     pass


class Ham2PoseMSEDistanceMeasure(AggregatedDistanceMeasure):
    def __init__(self):
        super().__init__(
            name="Ham2PoseMSEDistanceMeasure",
            default_distance=0.0,
            aggregation_strategy="mean",
        )

    def get_distance(self, hyp_data: ma.MaskedArray, ref_data: ma.MaskedArray, progress=False) -> float:
        keypoint_count = hyp_data.shape[2]  # Assuming shape: (frames, person, keypoints, xyz)
        trajectory_distances = ma.empty(keypoint_count)  # Preallocate a NumPy array

        for i, (hyp_trajectory, ref_trajectory) in tqdm(
            enumerate(self._get_keypoint_trajectories(hyp_data, ref_data)),
            desc="getting dtw distances for trajectories",
            total=keypoint_count,
            disable=not progress,
        ):
            sq_error = np.power(hyp_trajectory - ref_trajectory, 2).sum(-1)
            trajectory_distances[i] = sq_error  # Store distance in the preallocated array
        trajectory_distances = ma.array(trajectory_distances)
        return self._aggregate(trajectory_distances)


class Ham2PoseAPEDistanceMeasure(AggregatedDistanceMeasure):
    def __init__(self):
        super().__init__(
            name="Ham2PoseMSEDistanceMeasure",
            default_distance=0.0,
            aggregation_strategy="mean",
        )

    def get_distance(self, hyp_data: ma.MaskedArray, ref_data: ma.MaskedArray, progress=False) -> float:
        keypoint_count = hyp_data.shape[2]  # Assuming shape: (frames, person, keypoints, xyz)
        trajectory_distances = ma.empty(keypoint_count)  # Preallocate a NumPy array

        for i, (hyp_trajectory, ref_trajectory) in tqdm(
            enumerate(self._get_keypoint_trajectories(hyp_data, ref_data)),
            desc="getting dtw distances for trajectories",
            total=keypoint_count,
            disable=not progress,
        ):
            sq_error = np.power(hyp_trajectory - ref_trajectory, 2).sum(-1)
            trajectory_distances[i] = np.sqrt(sq_error).mean()  # Store distance in the preallocated array
        trajectory_distances = ma.array(trajectory_distances)
        return self._aggregate(trajectory_distances)


# class Ham2Pose_APE(AggregatedDistanceMetric):
#     pass


# class Ham2PoseMetric(DistanceMetric):
#     def __init__(
#         self,
#         name: str,

#         **kwargs: Any,
#     ) -> None:
#         pose_preprocessors = [RemoveWorldLandmarksProcessor(), ReduceHolisticProcessor(), NormalizePosesProcessor()]
#         super().__init__(name=name, higher_is_better=False, pose_preprocessors, **kwargs)


# class Ham2PoseDTW(Ham2PoseMetric):
#     def __init__(
#         self,
#         name: str,
#         distance_measure: DistanceMeasure,
#         **kwargs: Any,
#     ) -> None:

# TODO: Masked Fill with zeros, then euclidean


# class Ham2PosenDTW(DistanceMetric):
#     pass


# class Ham2PoseMSEMetric(DistanceMetric):
#     def __init__(normalize=False):
#         name = "MSE"
#         if normalize:
#             name = f"n{name}"
#         super().__init__(
#             name=name,
#         )

#         # TODO: ZeroPad and MaskedFill and Reduce to Intersection
#         # TODO: zero-fill positions where EITHER is masked???
#         # TODO: Upper body and hands only
#         # https://github.com/J22Melody/iict-eval-private/blob/text2pose/metrics/ham2pose.py#L195
#         # https://github.com/J22Melody/iict-eval-private/blob/text2pose/metrics/metrics.py#L22 maybe this?
#         self.pose_preprocessors.append(processor)

#         if normalize:


# TODO: Distance Measure


def ham2pose_mse_trajectory_distance(trajectory1, trajectory2):
    sq_error = np.power(trajectory1 - trajectory2, 2).sum(-1)
    return sq_error


def ham2pose_ape_trajectory_distance(trajectory1, trajectory2):
    return np.sqrt(ham2pose_mse_trajectory_distance(trajectory1, trajectory2)).mean()


# No need for this if we just do FillMasked
# def ham2pose_unmasked_euclidean_point_distance(point1, point2):
#     if np.ma.is_masked(point2):  # reference label keypoint is missing
#         return euclidean((0, 0, 0), point1)
#     elif np.ma.is_masked(point1):  # reference label keypoint is not missing, other label keypoint is missing
#         return euclidean((0, 0, 0), point2)
#     d = euclidean(point1, point2)
#     return d
