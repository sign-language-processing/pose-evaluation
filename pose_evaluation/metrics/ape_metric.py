import numpy as np
from pose_format import Pose
from pose_evaluation.metrics.distance_metric import DistanceMetric, ValidDistanceKinds




class AveragePositionErrorMetric(DistanceMetric):
    def __init__(self, 
                 spatial_distance_kind: ValidDistanceKinds = "euclidean", 
                 normalize_poses: bool = True, 
                 reduce_poses_to_common_points: bool = True, 
                 zero_pad_shorter_sequence: bool = True, 
                 remove_legs: bool = True, 
                 remove_world_landmarks: bool = True, 
                 conf_threshold_to_drop_points: None | int = None):
        super().__init__(spatial_distance_kind, normalize_poses, reduce_poses_to_common_points, zero_pad_shorter_sequence, remove_legs, remove_world_landmarks, conf_threshold_to_drop_points)

    # def score(self, hypothesis: Pose, reference: Pose) -> float:

    #     hyp_points = hypothesis.body.points_perspective()
    #     ref_points = reference.body.points_perspective()


    #     if hyp_points.shape[0] != ref_points.shape[0] or hyp_points.shape[-1] != ref_points.shape[-1]:
    #         raise ValueError(
    #             f"Shapes of hyp ({hyp_points.shape}) and ref ({ref_points.shape}) unequal. Not supported by {self.name}"
    #             )
        
    #     point_count = hyp_points.shape[0]
    #     total_error = 0        
    #     for hyp_point_data, ref_point_data in zip(hyp_points, ref_points):
    #         # shape is people, frames, xyz
    #         # NOTE: assumes only one person! # TODO: pytest test checking this.
    #         assert hyp_point_data.shape[0] == 1, f"{self} metric expects only one person. Hyp shape given: {hyp_point_data.shape}"
    #         assert ref_point_data.shape[0] == 1, f"{self} metric expects only one person. Reference shape given: {ref_point_data.shape}"
    #         hyp_point_trajectory = hyp_point_data[0]
    #         ref_point_trajectory = ref_point_data[0]
    #         joint_trajectory_error = self.average_position_error(hyp_point_trajectory, ref_point_trajectory)
    #         total_error += joint_trajectory_error

    #     average_position_error = total_error/point_count
    #     return average_position_error

    def trajectory_pair_distance_function(self, hyp_trajectory, ref_trajectory) -> float:
        assert len(hyp_trajectory) == len(ref_trajectory)
        return self.average_position_error(hyp_trajectory, ref_trajectory)
    
    def average_position_error(self, trajectory1, trajectory2):
        # point_coordinate_count = trajectory1.shape[-1]
        # if len(trajectory1) < len(trajectory2):
        #     diff = len(trajectory2) - len(trajectory1)
        #     trajectory1 = np.concatenate((trajectory1, np.zeros((diff, point_coordinate_count))))
        # elif len(trajectory2) < len(trajectory1):
        #     trajectory2 = np.concatenate((trajectory2, np.zeros((len(trajectory1) - len(trajectory2), point_coordinate_count))))
        pose1_mask = np.ma.getmask(trajectory1)
        pose2_mask = np.ma.getmask(trajectory2)
        trajectory1[pose1_mask] = 0
        trajectory1[pose2_mask] = 0
        trajectory2[pose1_mask] = 0
        trajectory2[pose2_mask] = 0
        sq_error = np.power(trajectory1 - trajectory2, 2).sum(-1)
        return np.sqrt(sq_error).mean()        
