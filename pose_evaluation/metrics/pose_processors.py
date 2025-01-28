from typing import Any, List, Union, Iterable, Callable
from pose_format import Pose

from pose_evaluation.metrics.base import SignatureMixin
from pose_evaluation.utils.pose_utils import remove_components, pose_remove_legs, get_face_and_hands_from_pose, reduce_pose_components_and_points_to_intersection, zero_pad_shorter_poses, copy_pose, pose_hide_legs, pose_hide_low_conf, set_masked_to_origin_position
PosesTransformerFunctionType = Callable[[Iterable[Pose]], List[Pose]]


class PoseProcessor(SignatureMixin):
    def __init__(self, name="PoseProcessor") -> None:
        self.name = name

    def __call__(self, pose_or_poses: Union[Iterable[Pose],Pose]) -> Any:
        if isinstance(pose_or_poses, Iterable):
            return self.process_poses(pose_or_poses)
        else:
            return self.process_pose(pose_or_poses)
    
    def __repr__(self) -> str:
        return self.name
    
    def __str__(self) -> str:
        return self.name
    
    def process_pose(self, pose : Pose) -> Pose:
        return pose
    
    def process_poses(self, poses: Iterable[Pose])-> List[Pose]:
        return [self.process_pose(pose) for pose in poses]
    
    
    
class RemoveComponentsProcessor(PoseProcessor):
    def __init__(self, landmarks:List[str]) -> None:
        super().__init__(f"remove_landmarks[landmarks{landmarks}]")
        self.landmarks = landmarks    
    
    def process_pose(self, pose: Pose) -> Pose:
        return remove_components(pose, self.landmarks)    
    
class RemoveWorldLandmarksProcessor(RemoveComponentsProcessor):
    def __init__(self) -> None:
        landmarks =  ["POSE_WORLD_LANDMARKS"]
        super().__init__(landmarks)

class RemoveLegsPosesProcessor(PoseProcessor):
    def __init__(self, name="remove_legs") -> None:
        super().__init__(name)
    
    def process_pose(self, pose: Pose) -> Pose:
        return pose_remove_legs(pose)                

class GetFaceAndHandsProcessor(PoseProcessor):
    def __init__(self, name="face_and_hands") -> None:
        super().__init__(name)

    def process_pose(self, pose: Pose) -> Pose:
        return get_face_and_hands_from_pose(pose)    

class ReducePosesToCommonComponentsProcessor(PoseProcessor):
    def __init__(self, name="reduce_pose_components") -> None:
        super().__init__(name)

    def process_pose(self, pose: Pose) -> Pose:
        return self.process_poses([pose])[0]
    
    def process_poses(self, poses: Iterable[Pose]) -> List[Pose]:
        return reduce_pose_components_and_points_to_intersection(poses)
    
class ZeroPadShorterPosesProcessor(PoseProcessor):
    def __init__(self) -> None:
        super().__init__(name="zero_pad_shorter_sequence")

    def process_poses(self, poses: Iterable[Pose]) -> List[Pose]:
        return zero_pad_shorter_poses(poses)


class PadOrTruncateByReferencePosesProcessor(PoseProcessor):
    def __init__(self) -> None:
        super().__init__(name="by_reference")

    def process_poses(self, poses: Iterable[Pose]) -> List[Pose]:
        raise NotImplementedError # TODO
        

class NormalizePosesProcessor(PoseProcessor):
    def __init__(self, info=None, scale_factor=1) -> None:
        super().__init__(f"normalize_poses[info:{info},scale_factor:{scale_factor}]")
        self.info = info
        self.scale_factor = scale_factor

    def process_pose(self, pose: Pose) -> Pose:
        return pose.normalize(self.info, self.scale_factor)


class HideLowConfProcessor(PoseProcessor):
    def __init__(self, conf_threshold:float = 0.2) -> None:

        super().__init__(f"hide_low_conf[{conf_threshold}]")
        self.conf_threshold = conf_threshold

    def process_pose(self, pose: Pose) -> Pose:
        pose = copy_pose(pose)
        pose_hide_low_conf(pose, self.conf_threshold)
        return pose
        


class SetMaskedValuesToOriginPositionProcessor(PoseProcessor):
    def __init__(self,) -> None:
        super().__init__(name="set_masked_to_origin")

    def process_pose(self, pose: Pose) -> Pose:
        return set_masked_to_origin_position(pose)
    

    
def get_standard_pose_processors(normalize_poses:bool=True, 
                                 reduce_poses_to_common_components:bool=True,
                                 remove_world_landmarks=True,
                                 remove_legs=True,
                                 zero_pad_shorter_poses=True,
                                 set_masked_values_to_origin=False,
                                 )-> List[PoseProcessor]:
    pose_processors = []

    if normalize_poses:
        pose_processors.append(NormalizePosesProcessor())

    if reduce_poses_to_common_components:
        pose_processors.append(ReducePosesToCommonComponentsProcessor())

    if remove_world_landmarks:
        pose_processors.append(RemoveWorldLandmarksProcessor())
    
    if remove_legs:
        pose_processors.append(RemoveLegsPosesProcessor())

    if zero_pad_shorter_poses:
        pose_processors.append(ZeroPadShorterPosesProcessor())

    if set_masked_values_to_origin:
        pose_processors.append(SetMaskedValuesToOriginPositionProcessor())

    return pose_processors
