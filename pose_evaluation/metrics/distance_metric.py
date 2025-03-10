from pose_format import Pose

from pose_evaluation.metrics.base_pose_metric import PoseMetric
from pose_evaluation.metrics.distance_measure import DistanceMeasure


class DistanceMetric(PoseMetric):
    """Computes the distance between two poses using the provided distance measure."""

    def __init__(self, name: str, distance_measure: DistanceMeasure) -> None:
        super().__init__(name=name, higher_is_better=False)
        self.distance_measure = distance_measure

    def score(self, hypothesis: Pose, reference: Pose) -> float:
        """Calculate the distance score between hypothesis and reference poses."""
        return self.distance_measure(hypothesis.body.data, reference.body.data)
