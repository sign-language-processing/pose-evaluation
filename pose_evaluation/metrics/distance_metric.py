from typing import List
from pose_format import Pose
from pose_evaluation.metrics.base_pose_metric import PoseMetric
from pose_evaluation.metrics.distance_measure import DistanceMeasure
from pose_evaluation.metrics.pose_processors import PoseProcessor


class DistanceMetric(PoseMetric):
    """Computes the distance between two poses using the provided distance measure."""

    def __init__(
        self,
        name: str,
        distance_measure: DistanceMeasure,
        pose_preprocessors: List[PoseProcessor] | None = None,
    ) -> None:
        super().__init__(
            name=name, higher_is_better=False, pose_preprocessors=pose_preprocessors
        )
        self.distance_measure = distance_measure

    def score(self, hypothesis: Pose, reference: Pose) -> float:
        """Calculate the distance score between hypothesis and reference poses."""
        hypothesis, reference = self.process_poses([hypothesis, reference])
        return self.distance_measure(hypothesis.body.data, reference.body.data)
