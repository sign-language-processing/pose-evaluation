from typing import Any, List, Union, Iterable, Callable


from tqdm import tqdm
from numpy import ma

from pose_format import Pose
from pose_format.utils.generic import pose_hide_legs, reduce_holistic, detect_known_pose_format
from spoken_to_signed.gloss_to_pose.concatenate import trim_pose

from pose_evaluation.metrics.base import Signature
from pose_evaluation.utils.pose_utils import (
    zero_pad_shorter_poses,
    reduce_poses_to_intersection,
    pose_fill_masked_or_invalid,
    pose_mask_invalid_values,
    first_frame_pad_shorter_poses,
)


PosesTransformerFunctionType = Callable[[Iterable[Pose]], List[Pose]]


class PoseProcessor:
    _SIGNATURE_TYPE = Signature

    def __init__(self, name="PoseProcessor") -> None:
        self.name = name

    def __call__(self, pose_or_poses: Union[Iterable[Pose], Pose]) -> Any:
        if isinstance(pose_or_poses, Iterable):
            return self.process_poses(pose_or_poses)

        return self.process_pose(pose_or_poses)

    def __repr__(self) -> str:
        return str(self.get_signature())

    def __str__(self) -> str:
        return self.get_signature().format()

    def process_pose(self, pose: Pose) -> Pose:
        raise NotImplementedError(f"process_pose not implemented for {self.name}")

    def process_poses(self, poses: Iterable[Pose], progress=False) -> List[Pose]:
        return [self.process_pose(pose) for pose in tqdm(poses, desc=f"{self.name}", disable=not progress)]

    def get_signature(self) -> Signature:
        return self._SIGNATURE_TYPE(self.name, self.__dict__)


class NormalizePosesSignature(Signature):
    def __init__(self, name: str, args: dict):
        super().__init__(name, args)
        self.update_signature_and_abbr("scale_factor", "s", args)
        self.update_signature_and_abbr("info", "i", args)


class NormalizePosesProcessor(PoseProcessor):
    _SIGNATURE_TYPE = NormalizePosesSignature

    def __init__(self, info=None, scale_factor=1) -> None:
        super().__init__("normalize_poses")
        self.info = info
        self.scale_factor = scale_factor

    def process_pose(self, pose: Pose) -> Pose:
        return pose.normalize(self.info, self.scale_factor)


class RemoveWorldLandmarksProcessor(PoseProcessor):
    def __init__(self, name="remove_world_landmarks") -> None:
        super().__init__(name)

    def process_pose(self, pose: Pose) -> Pose:
        return pose.remove_components(["WORLD_LANDMARKS"])


class HideLegsPosesProcessor(PoseProcessor):
    def __init__(self, name="hide_legs", remove=True) -> None:
        super().__init__(name)
        self.remove = remove

    def process_pose(self, pose: Pose) -> Pose:
        return pose_hide_legs(pose, remove=self.remove)


class ReducePosesToCommonComponentsProcessor(PoseProcessor):
    def __init__(self, name="reduce_poses_to_intersection") -> None:
        super().__init__(name)

    def process_pose(self, pose: Pose) -> Pose:
        return self.process_poses([pose])[0]

    def process_poses(self, poses: Iterable[Pose], progress=False) -> List[Pose]:
        return reduce_poses_to_intersection(poses, progress=progress)


class ZeroPadShorterPosesProcessor(PoseProcessor):
    def __init__(self) -> None:
        super().__init__(name="zero_pad_shorter_sequence")

    def process_pose(self, pose: Pose) -> Pose:
        return pose

    def process_poses(self, poses: Iterable[Pose], progress=False) -> List[Pose]:
        return zero_pad_shorter_poses(poses)


class FirstFramePadShorterPosesProcessor(PoseProcessor):
    def __init__(self) -> None:
        super().__init__(name="first_frame_pad_shorter")

    def process_pose(self, pose: Pose) -> Pose:
        return pose

    def process_poses(self, poses: Iterable[Pose], progress=False) -> List[Pose]:
        return first_frame_pad_shorter_poses(poses)


class ReduceHolisticPoseProcessor(PoseProcessor):
    def __init__(self) -> None:
        super().__init__(name="reduce_holistic")

    def process_pose(self, pose: Pose) -> Pose:
        return reduce_holistic(pose)


class ZeroFillMaskedValuesPoseProcessor(PoseProcessor):
    def __init__(self) -> None:
        super().__init__(name="zero_fill_masked")

    def process_pose(self, pose: Pose) -> Pose:
        pose = pose.copy()
        pose.body = pose.body.zero_filled()
        return pose


class MaskInvalidValuesPoseProcessor(PoseProcessor):
    def __init__(self) -> None:
        super().__init__(name="mask_invalid_values")

    def process_pose(self, pose: Pose) -> Pose:
        return pose_mask_invalid_values(pose)


class FillMaskedOrInvalidValuesPoseProcessor(PoseProcessor):
    def __init__(self, masked_fill_value: float = 0.0) -> None:
        super().__init__(name="fill_masked_or_invalid")
        self.fill_val = masked_fill_value

    def process_pose(self, pose: Pose) -> Pose:
        return pose_fill_masked_or_invalid(pose, self.fill_val)


class TrimMeaninglessFramesPoseProcessor(PoseProcessor):
    def __init__(self, start=True, end=True) -> None:
        super().__init__(name="trim_pose")
        self.start = start
        self.end = end

    def process_pose(self, pose):
        if detect_known_pose_format(pose) == "holistic":

            return trim_pose(pose.copy(), start=self.start, end=self.end)
        # not supported
        return pose


class GetHandsOnlyHolisticPoseProcessor(PoseProcessor):
    def __init__(self) -> None:
        super().__init__(name="get_hands_only")

    def process_pose(self, pose: Pose) -> Pose:
        return pose.get_components(["LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"])


class InterpolateAllToSetFPSPoseProcessor(PoseProcessor):
    def __init__(self, fps=15, kind: str = "cubic") -> None:
        super().__init__(name="interpolate_to_fps")
        self.fps = fps
        self.kind = kind

    def process_pose(self, pose: Pose) -> Pose:
        pose = pose.copy()
        pose = pose.interpolate(new_fps=self.fps, kind=self.kind)
        return pose


def get_standard_pose_processors(
    trim_meaningless_frames: bool = True,
    normalize_poses: bool = True,
    reduce_poses_to_common_components: bool = True,
    remove_world_landmarks=True,
    remove_legs=True,
    reduce_holistic_to_face_and_upper_body=False,
    zero_fill_masked=False,
    zero_pad_shorter=True,
) -> List[PoseProcessor]:
    pose_processors = []

    # remove leading/trailing frames with no hands in frame.
    if trim_meaningless_frames:
        pose_processors.append(TrimMeaninglessFramesPoseProcessor())

    # Note: by default this uses the shoulder joints,
    # so it should be BEFORE anything that might remove those, such as reduce poses to common components
    if normalize_poses:
        pose_processors.append(NormalizePosesProcessor())

    if reduce_poses_to_common_components:
        pose_processors.append(ReducePosesToCommonComponentsProcessor())

    if remove_world_landmarks:
        pose_processors.append(RemoveWorldLandmarksProcessor())

    if remove_legs:
        pose_processors.append(HideLegsPosesProcessor())

    if reduce_holistic_to_face_and_upper_body:
        pose_processors.append(ReduceHolisticPoseProcessor())

    if zero_fill_masked:
        pose_processors.append(ZeroFillMaskedValuesPoseProcessor())

    if zero_pad_shorter:
        pose_processors.append(ZeroPadShorterPosesProcessor())

    # TODO: prune leading/trailing frames containing "almost all zeros, almost no face, or no hands"
    # TODO: Focus processor https://github.com/rotem-shalev/Ham2Pose/blob/main/metrics.py#L32-L62

    return pose_processors
