from typing import Literal

from numpy import ma
from pose_format import Pose

from pose_evaluation.metrics.base_pose_metric import PoseMetric
from pose_evaluation.metrics.distance_measure import DistanceMeasure


class DistanceMetric(PoseMetric):
    def __init__(self, name, distance_measure:DistanceMeasure):
        super().__init__(name=name, higher_is_better=False)
        self.distance_measure = distance_measure

    def score(self, hypothesis: Pose, reference: Pose) -> float:
        return self.distance_measure.get_distance(hypothesis.body.data, reference.body.data)
