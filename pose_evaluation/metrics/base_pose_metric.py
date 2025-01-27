from pose_format import Pose
from typing import Literal, Iterable, Tuple, List, cast, get_args, Callable, Union, TYPE_CHECKING
from pose_evaluation.metrics.pose_processors import PoseProcessor, NormalizePosesProcessor, RemoveLegsPosesProcessor, HideLowConfProcessor, ZeroPadShorterPosesProcessor, ReducePosesToCommonComponentsProcessor, RemoveWorldLandmarksProcessor, RemoveComponentsProcessor

from pose_evaluation.metrics.base import BaseMetric, MetricSignature

MismatchedComponentsStrategy = Literal["reduce"]
class PoseMetricSignature(MetricSignature):

     def __init__(self, args: dict):
        super().__init__(args)

        self._abbreviated.update(
            {
                "pose_preprocessers":"pre"
            }
        )


        pose_preprocessors = args.get('pose_preprocessers', None)
        prep_string = ""
        if pose_preprocessors is not None:
            prep_string = "{" + "|".join([f"{prep}" for prep in pose_preprocessors]) + "}"


        self.signature_info.update(
            {
                'pose_preprocessers': prep_string if pose_preprocessors else None
            }
        )




class PoseMetric(BaseMetric[Pose]):

    _SIGNATURE_TYPE = PoseMetricSignature

    def __init__(self, name: str="PoseMetric", higher_is_better: bool = False,
                 pose_preprocessors: Union[None, List[PoseProcessor]] = None,
                 ):
                 
            super().__init__(name, higher_is_better)
            if pose_preprocessors is None:
                self.pose_preprocessers = [             
                ]
            else:
                self.pose_preprocessers = pose_preprocessors

    def score(self, hypothesis: Pose, reference: Pose) -> float:
        hypothesis, reference = self.process_poses([hypothesis, reference])
        return self.score(hypothesis, reference)

    def process_poses(self, poses:Iterable[Pose])->List[Pose]:
        poses = list(poses)
        for preprocessor in self.pose_preprocessers:
            preprocessor = cast(PoseProcessor, preprocessor)
            poses = preprocessor.process_poses(poses)
        return poses
        # self.set_coordinate_point_distance_function(coordinate_point_distance_kind)


