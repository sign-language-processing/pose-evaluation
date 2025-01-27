from typing import List, Optional
import numpy as np

from pose_evaluation.metrics.aggregate_distances_strategy import DistancesAggregator
from pose_evaluation.metrics.pose_processors import PoseProcessor
from pose_format import Pose
from pose_evaluation.metrics.distance_metric import DistanceMetric, DistanceMetricSignature, ValidPointDistanceKinds, MaskedKeypointPositionStrategy, TrajectoryAlignmentStrategy


class MeanJointDistanceSignature(DistanceMetricSignature):
    pass


class MeanJointDistanceMetric(DistanceMetric):
    def __init__(self, name: str = "MeanJointDistanceMetric", 
                 higher_is_better: bool = False, 
                 pose_preprocessors: None | List[PoseProcessor] = None, 
                 normalize_poses=True, 
                 reduce_poses_to_common_points=True, 
                 remove_legs=True, remove_world_landmarks=True, 
                 distance_measure: ValidPointDistanceKinds = 'euclidean', 
                 mask_strategy: MaskedKeypointPositionStrategy = 'masked_to_origin', 
                 alignment_strategy: TrajectoryAlignmentStrategy = 'zero_pad_shorter', 
                 distances_aggregator: DistancesAggregator | None = None):
        super().__init__(name, higher_is_better, pose_preprocessors, normalize_poses, reduce_poses_to_common_points, remove_legs, remove_world_landmarks, distance_measure, mask_strategy, alignment_strategy, distances_aggregator)