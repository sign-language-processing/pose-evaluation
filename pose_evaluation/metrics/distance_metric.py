from pose_format import Pose
from typing import Any

from pose_evaluation.metrics.base_pose_metric import PoseMetric
from pose_evaluation.metrics.distance_measure import DistanceMeasure


class DistanceMetric(PoseMetric):
    """Computes the distance between two poses using the provided distance measure."""

    def __init__(
            self,
            name: str,
            distance_measure: DistanceMeasure,
            **kwargs: Any,
    ) -> None:
        super().__init__(name=name, higher_is_better=False, **kwargs)

        self.distance_measure = distance_measure

    def _pose_score(self, processed_hypothesis: Pose, processed_reference: Pose) -> float:
        return self.distance_measure(processed_hypothesis.body.data, processed_reference.body.data)
