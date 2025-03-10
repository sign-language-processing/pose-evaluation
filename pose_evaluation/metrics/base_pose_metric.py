from typing import Iterable, List, Sequence, cast, Union
from tqdm import tqdm

from pose_format import Pose

from pose_evaluation.metrics.base import BaseMetric, Signature, Score
from pose_evaluation.metrics.pose_processors import PoseProcessor
from pose_evaluation.metrics.pose_processors import get_standard_pose_processors


class PoseMetricSignature(Signature):
    def __init__(self, name: str, args: dict):
        super().__init__(name, args)
        self.update_abbr("pose_preprocessors", "pre")
        # self.update_signature_and_abbr("pose_preprocessors", "pre", args)


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
            self.pose_preprocessors = get_standard_pose_processors()
        else:
            self.pose_preprocessors = pose_preprocessors

    def score_with_signature(
        self, hypothesis: Pose, reference: Pose, short: bool = False
    ) -> PoseMetricScore:
        hypothesis, reference = self.process_poses([hypothesis, reference])
        return PoseMetricScore(
            name=self.name,
            score=self.score(hypothesis, reference),
            signature=self.get_signature().format(short=short),
        )

    def score_all_with_signature(
        self,
        hypotheses: Sequence[Pose],
        references: Sequence[Pose],
        progress_bar=False,
        short: bool = False,
    ) -> list[list[Score]]:
        hyp_len = len(hypotheses)
        ref_len = len(references)

        all_poses = self.process_poses(hypotheses + references)

        # Recover original lists if needed
        hypotheses = all_poses[:hyp_len]
        references = all_poses[hyp_len : hyp_len + ref_len]

        return [
            [self.score_with_signature(h, r , short=short) for r in references]
            for h in tqdm(hypotheses, desc="scoring:", disable=not progress_bar or len(hypotheses) == 1)
        ]

    def process_poses(self, poses: Iterable[Pose], progress=False) -> List[Pose]:
        poses = list(poses)
        for preprocessor in tqdm(
            self.pose_preprocessors, desc="Preprocessing Poses", disable=not progress
        ):
            preprocessor = cast(PoseProcessor, preprocessor)
            poses = preprocessor.process_poses(poses, progress=progress)
        return poses

    def add_preprocessor(self, processor: PoseProcessor):
        self.pose_preprocessors.append(processor)
