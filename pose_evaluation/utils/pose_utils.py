from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from pose_format import Pose
from pose_format.utils.openpose import OpenPose_Components
from pose_format.utils.openpose_135 import OpenPose_Components as OpenPose135_Components
# from pose_format.utils.holistic import holistic_components # creates an error: ImportError: Please install mediapipe with: pip install mediapipe
from collections import defaultdict
from pose_format.utils.generic import pose_normalization_info, pose_hide_legs, fake_pose


def pose_remove_world_landmarks(pose: Pose):
    return remove_components(pose, ["POSE_WORLD_LANDMARKS"])


def detect_format(pose: Pose) -> str:
    component_names = [c.name for c in pose.header.components]
    mediapipe_components = [
        "POSE_LANDMARKS",
        "FACE_LANDMARKS",
        "LEFT_HAND_LANDMARKS",
        "RIGHT_HAND_LANDMARKS",
        "POSE_WORLD_LANDMARKS",
    ]
    
    openpose_components = [c.name for c in OpenPose_Components]
    openpose_135_components = [c.name for c in OpenPose135_Components]
    for component_name in component_names:
        if component_name in mediapipe_components:
            return "mediapipe"
        if component_name in openpose_components:
            return "openpose"
        if component_name in openpose_135_components:
            return "openpose_135"

    raise ValueError(
        f"Unknown pose header schema with component names: {component_names}"
    )

def get_component_names_and_points_dict(pose:Pose)->Tuple[List[str], Dict[str, List[str]]]:
    component_names = []
    points_dict = defaultdict(list)
    for component in pose.header.components:
            component_names.append(component.name)
            
            for point in component.points:
                points_dict[component.name].append(point)

    return component_names, points_dict

def remove_components(
    pose: Pose, components_to_remove: List[str]|str, points_to_remove: List[str]|str|None=None
):
    if points_to_remove is None:
        points_to_remove = []
    if isinstance(components_to_remove, str):
        components_to_remove = [components_to_remove]
    if isinstance(points_to_remove, str):
        points_to_remove = [points_to_remove]
    components_to_keep = []
    points_dict = {}

    for component in pose.header.components:
        if component.name not in components_to_remove:
            components_to_keep.append(component.name)
            points_dict[component.name] = []
            for point in component.points:
                if point not in points_to_remove:
                    points_dict[component.name].append(point)

    return pose.get_components(components_to_keep, points_dict)


def pose_remove_legs(pose: Pose) -> Pose:
    detected_format = detect_format(pose)
    if detected_format == "mediapipe":
        mediapipe_point_names = ["KNEE", "ANKLE", "HEEL", "FOOT_INDEX"]
        mediapipe_sides = ["LEFT", "RIGHT"]
        point_names_to_remove = [
            side + "_" + name
            for name in mediapipe_point_names
            for side in mediapipe_sides
        ]
    else:
        raise NotImplementedError(
            f"Remove legs not implemented yet for pose header schema {detected_format}"
        )

    pose = remove_components(pose, [], point_names_to_remove)
    return pose


def copy_pose(pose: Pose) -> Pose:
    return pose.get_components([component.name for component in pose.header.components])



def get_face_and_hands_from_pose(pose: Pose) -> Pose:
    # based on MediaPipe Holistic format.
    components_to_keep = [
        "FACE_LANDMARKS",
        "LEFT_HAND_LANDMARKS",
        "RIGHT_HAND_LANDMARKS",
    ]
    return pose.get_components(components_to_keep)


def load_pose_file(pose_path: Path) -> Pose:
    pose_path = Path(pose_path).resolve()
    with pose_path.open("rb") as f:
        pose = Pose.read(f.read())
    return pose


def reduce_pose_components_to_intersection(poses: List[Pose]) -> List[Pose]:
    component_names_for_each_pose = []
    for pose in poses:
        names = set([c.name for c in pose.header.components])
        component_names_for_each_pose.append(names)

    set_of_common_components = list(set.intersection(*component_names_for_each_pose))
    poses = [pose.get_components(set_of_common_components) for pose in poses]
    return poses


def preprocess_poses(
    poses: List[Pose],
    normalize_poses: bool = True,
    reduce_poses_to_common_components: bool = False,
    remove_legs: bool = True,
    remove_world_landmarks: bool = False,
    conf_threshold_to_drop_points: None | float = None,
) -> List[Pose]:
    # NOTE: this is a lot of arguments. Perhaps a list may be better?
    if reduce_poses_to_common_components:
        poses = reduce_pose_components_to_intersection(poses)

    poses = [
        preprocess_pose(
            pose,
            normalize_poses=normalize_poses,
            remove_legs=remove_legs,
            remove_world_landmarks=remove_world_landmarks,
            conf_threshold_to_drop_points=conf_threshold_to_drop_points,
        )
        for pose in poses
    ]
    return poses


def preprocess_pose(
    pose: Pose,
    normalize_poses: bool = True,
    remove_legs: bool = True,
    remove_world_landmarks: bool = False,
    conf_threshold_to_drop_points: None | float = None,
) -> Pose:
    pose = copy_pose(pose)
    if normalize_poses:
        # note: latest version (not yet released) does it automatically
        pose = pose.normalize(pose_normalization_info(pose.header))

    # Drop legs
    if remove_legs:
        # pose_hide_legs(pose)
        pose = pose_remove_legs(pose)

    # not used, typically.
    if remove_world_landmarks:
        pose = pose_remove_world_landmarks(pose)

    # hide low conf
    if conf_threshold_to_drop_points is not None:
        pose_hide_low_conf(pose, confidence_threshold=conf_threshold_to_drop_points)

    return pose


def pose_hide_low_conf(pose: Pose, confidence_threshold: float = 0.2) -> None:
    mask = pose.body.confidence <= confidence_threshold
    pose.body.confidence[mask] = 0
    stacked_confidence = np.stack([mask, mask, mask], axis=3)
    masked_data = np.ma.masked_array(pose.body.data, mask=stacked_confidence)
    pose.body.data = masked_data
