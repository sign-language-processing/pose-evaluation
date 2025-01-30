from typing import Literal, Tuple, List, Union, Optional
from pose_format import Pose
from pose_evaluation.metrics.base_pose_metric import PoseMetric, PoseMetricSignature
from pose_evaluation.metrics.distance_measure import DistanceMeasure, PowerDistance
from pose_evaluation.metrics.pose_processors import PoseProcessor

BuildTrajectoryStrategy = Literal["keypoint", "frame"]
TrajectoryAlignmentStrategy = Literal[
    "zero_pad_shorter", "truncate_longer", "by_reference"
]
MaskedKeypointPositionStrategy = Literal[
    "skip_masked",
    "return_zero",
    "masked_to_origin",
    "ref_return_zero_hyp_to_origin",
    "undefined",
]

KeypointPositionType = Union[
    Tuple[float, float, float], Tuple[float, float]
]  # XYZ or XY
ValidPointDistanceKinds = Literal["euclidean", "manhattan"]


class DistanceMetricSignature(PoseMetricSignature):
    def __init__(self, args: dict):
        super().__init__(args)
        self.update_signature_and_abbr("distance_measure", "dist", args)
        self.update_signature_and_abbr("trajectory", "trj", args)


class DistanceMetric(PoseMetric):
    """Metrics that compute some sort of distance"""

    _SIGNATURE_TYPE = DistanceMetricSignature

    def __init__(
        self,
        name="DistanceMetric",
        distance_measure: Optional[DistanceMeasure] = None,
        pose_preprocessors: None | List[PoseProcessor] = None,
        trajectory: BuildTrajectoryStrategy = "keypoint",
    ):
        super().__init__(
            name=name, higher_is_better=False, pose_preprocessors=pose_preprocessors
        )

        if distance_measure is None:
            self.distance_measure = PowerDistance()

        else:
            self.distance_measure = distance_measure

        self.trajectory = trajectory

    def score(self, hypothesis: Pose, reference: Pose) -> float:
        hypothesis, reference = self.process_poses([hypothesis, reference])
        if self.trajectory == "keypoint":
            return self.distance_measure(
                hypothesis.body.points_perspective(),
                reference.body.points_perspective(),
            )

        return self.distance_measure(hypothesis.body.data, reference.body.data)
