from typing import Literal, List, TYPE_CHECKING, Optional

import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


# from pose_evaluation.metrics.point_distance_calculator import masked_euclidean
from pose_evaluation.metrics.distance_metric import DistanceMetric, ValidPointDistanceKinds
from pose_evaluation.metrics.pose_processors import PoseProcessor



def masked_euclidean(point1, point2):
    if np.ma.is_masked(point2):  # reference label keypoint is missing
        return 0
    elif np.ma.is_masked(point1):  # reference label keypoint is not missing, other label keypoint is missing
        return euclidean((0, 0, 0), point2)/2
    d = euclidean(point1, point2)
    return d


class DynamicTimeWarpingMetric(DistanceMetric):
    def __init__(self, name: str = "DynamicTimeWarpingMetric", 
                 point_distance_calculation_kind: str = "euclidean", 
                 preprocessors: None | List[PoseProcessor] = None, 
                 normalize_poses: bool = True, reduce_poses_to_common_points: bool = True, zero_pad_shorter_sequence: bool = True, remove_legs: bool = True, remove_world_landmarks: bool = True, conf_threshold_to_drop_points: None | float = None):
        super().__init__(name, point_distance_calculation_kind, preprocessors, normalize_poses, reduce_poses_to_common_points, zero_pad_shorter_sequence, remove_legs, remove_world_landmarks, conf_threshold_to_drop_points)    

    def trajectory_pair_distance_function(self, hyp_trajectory, ref_trajectory) -> float:
        distance, indexes = fastdtw(hyp_trajectory, ref_trajectory, dist=self.coord_pair_distance_function)
        return distance
    
    # def point_pair_distance_function(self, hyp_coordinate, ref_coordinate) -> float:
    #     return masked_euclidean(hyp_coordinate, ref_coordinate)

    # def score(self, hypothesis:Pose, reference:Pose):
    #     hyp_points = hypothesis.body.points_perspective() # 560, 1, 93, 3 for example. joint-points, frames, xyz
    #     ref_points = reference.body.points_perspective()


    #     if hyp_points.shape[0] != ref_points.shape[0] or hyp_points.shape[-1] != ref_points.shape[-1]:
    #         raise ValueError(
    #             f"Shapes of hyp ({hyp_points.shape}) and ref ({ref_points.shape}) unequal. Not supported by {self.name}"
    #             )
        
    #     point_count = hyp_points.shape[0]
    #     total_error = 0.0  
    #     for hyp_point_data, ref_point_data in zip(hyp_points, ref_points):
    #         # shape is people, frames, xyz
    #         # NOTE: assumes only one person! # TODO: pytest test checking this.
    #         assert hyp_point_data.shape[0] == 1, f"{self} metric expects only one person. Hyp shape given: {hyp_point_data.shape}"
    #         assert ref_point_data.shape[0] == 1, f"{self} metric expects only one person. Reference shape given: {ref_point_data.shape}"
    #         hyp_point_trajectory = hyp_point_data[0]
    #         ref_point_trajectory = ref_point_data[0]
    #         total_error += fastdtw(hyp_point_trajectory, ref_point_trajectory, dist=masked_euclidean)

    #     return total_error/point_count
        
