from typing import Iterable, List, cast, Union
from pose_format import Pose

from pose_evaluation.metrics.base import BaseMetric, Signature, Score
from pose_evaluation.metrics.pose_processors import PoseProcessor

class PoseMetricSignature(Signature):
    pass

class PoseMetricScore(Score):
    pass

class PoseMetric(BaseMetric[Pose]):
    _SIGNATURE_TYPE = PoseMetricSignature

    def __init__(
        self,
        name: str = "PoseMetric",
        higher_is_better: bool = False,
        pose_preprocessors: Union[None, List[PoseProcessor]] = None,
    ):

        super().__init__(name, higher_is_better)
        if pose_preprocessors is None:
            self.pose_preprocessors = []
        else:
            self.pose_preprocessors = pose_preprocessors

    def score(self, hypothesis: Pose, reference: Pose) -> float:
        hypothesis, reference = self.process_poses([hypothesis, reference])
        return self.score(hypothesis, reference)

    def process_poses(self, poses: Iterable[Pose]) -> List[Pose]:
        poses = list(poses)
        for preprocessor in self.pose_preprocessors:
            preprocessor = cast(PoseProcessor, preprocessor)
            poses = preprocessor.process_poses(poses)
        return poses

    def add_preprocessor(self, processor: PoseProcessor):
        self.pose_preprocessors.append(processor)
