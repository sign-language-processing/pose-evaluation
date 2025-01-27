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

    def __init__(self, name: str="PoseMetric", higher_is_better: bool = True,
                 pose_preprocessors: Union[None, List[PoseProcessor]] = None,
                 normalize_poses = True,
                 reduce_poses_to_common_points = True,
                 remove_legs = True,
                 remove_world_landmarks = True,
                 ):
                 
            super().__init__(name, higher_is_better)
            if pose_preprocessors is None:
                self.pose_preprocessers = [             
                ]
            else:
                self.pose_preprocessers = pose_preprocessors
            

            
            if normalize_poses:
                self.pose_preprocessers.append(NormalizePosesProcessor())
            if reduce_poses_to_common_points:
                self.pose_preprocessers.append(ReducePosesToCommonComponentsProcessor())
            if remove_legs:
                self.pose_preprocessers.append(RemoveLegsPosesProcessor())
            if remove_world_landmarks:
                self.pose_preprocessers.append(RemoveWorldLandmarksProcessor())
            


    def score(self, hypothesis: Pose, reference: Pose) -> float:
        hypothesis, reference = self.preprocess_poses([hypothesis, reference])
        return self.score(hypothesis, reference)

    def preprocess_poses(self, poses:List[Pose])->List[Pose]:
        for preprocessor in self.pose_preprocessers:
            preprocessor = cast(PoseProcessor, preprocessor)
            poses = preprocessor.process_poses(poses)
        return poses
        # self.set_coordinate_point_distance_function(coordinate_point_distance_kind)


