from typing import Literal, List, TYPE_CHECKING, Optional
import numpy as np
from distance_metric import DistanceMetric, ValidPointDistanceKinds
from pose_evaluation.metrics.pose_processors import PoseProcessor


def mse(trajectory1, trajectory2):
    if len(trajectory1) < len(trajectory2):
        diff = len(trajectory2) - len(trajectory1)
        trajectory1 = np.concatenate((trajectory1, np.zeros((diff, 2))))
    elif len(trajectory2) < len(trajectory1):
        trajectory2 = np.concatenate((trajectory2, np.zeros((len(trajectory1) - len(trajectory2), 2))))
    pose1_mask = np.ma.getmask(trajectory1)
    pose2_mask = np.ma.getmask(trajectory2)
    trajectory1[pose1_mask] = 0
    trajectory1[pose2_mask] = 0
    trajectory2[pose1_mask] = 0
    trajectory2[pose2_mask] = 0
    sq_error = np.power(trajectory1 - trajectory2, 2).sum(-1)
    return sq_error.mean()

class MeanSquaredErrorMetric(DistanceMetric):
    def __init__(self, name: str = "MeanSquaredError", point_distance_calculation_kind: str = "euclidean", preprocessors: None | List[PoseProcessor] = None, normalize_poses: bool = True, reduce_poses_to_common_points: bool = True, zero_pad_shorter_sequence: bool = True, remove_legs: bool = True, remove_world_landmarks: bool = True, conf_threshold_to_drop_points: None | float = None):
        super().__init__(name, point_distance_calculation_kind, preprocessors, normalize_poses, reduce_poses_to_common_points, zero_pad_shorter_sequence, remove_legs, remove_world_landmarks, conf_threshold_to_drop_points)

    def trajectory_pair_distance_function(self, hyp_trajectory, ref_trajectory) -> float:
        return super().trajectory_pair_distance_function(hyp_trajectory, ref_trajectory)
    
