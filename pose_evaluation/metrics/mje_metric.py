from typing import List
from pose_evaluation.metrics.distance_measure import PowerDistance
from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.metrics.pose_processors import get_standard_pose_processors


class MeanJointErrorMetric(DistanceMetric):
    def __init__(self):
        pose_preprocessors = get_standard_pose_processors()
        super().__init__(distance_measure = PowerDistance(2), 
                         pose_preprocessors = pose_preprocessors, 
                         trajectory='keypoint')
        self.name = "MJE"