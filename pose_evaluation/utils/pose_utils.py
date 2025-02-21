from pathlib import Path
from typing import List, Tuple, Dict, Iterable
from collections import defaultdict
import numpy as np
from numpy import ma
from pose_format import Pose


def pose_remove_world_landmarks(pose: Pose) -> Pose:
    return pose.remove_components(["POSE_WORLD_LANDMARKS"])


def get_component_names_and_points_dict(
    pose: Pose,
) -> Tuple[List[str], Dict[str, List[str]]]:
    component_names = []
    points_dict = defaultdict(list)
    for component in pose.header.components:
        component_names.append(component.name)

        for point in component.points:
            points_dict[component.name].append(point)

    return component_names, points_dict


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


def reduce_pose_components_and_points_to_intersection(
    poses: Iterable[Pose],
) -> List[Pose]:
    poses = [pose.copy() for pose in poses]
    component_names_for_each_pose = []
    point_dict_for_each_pose = []
    for pose in poses:
        names, points_dict = get_component_names_and_points_dict(pose)
        component_names_for_each_pose.append(set(names))
        point_dict_for_each_pose.append(points_dict)

    set_of_common_components = list(set.intersection(*component_names_for_each_pose))

    common_points = {}
    for component_name in set_of_common_components:
        max_length = 0
        min_length = np.inf
        points_for_each_pose = []
        for point_dict in point_dict_for_each_pose:
            points_list = point_dict.get(component_name)
            if points_list is None:
                min_length = 0
            max_length = max(max_length, len(points_list))
            min_length = min(min_length, len(points_list))
            points_for_each_pose.append(set(points_list))
        set_of_common_points = list(set.intersection(*points_for_each_pose))

        if 0 < min_length < max_length:
            common_points[component_name] = set_of_common_points

    poses = [
        pose.get_components(set_of_common_components, common_points) for pose in poses
    ]
    return poses


def zero_pad_shorter_poses(poses: Iterable[Pose]) -> List[Pose]:
    poses = [pose.copy() for pose in poses]
    # arrays = [pose.body.data for pose in poses]

    # first dimension is frames. Then People, joint-points, XYZ or XY
    max_frame_count = max(len(pose.body.data) for pose in poses)
    # Pad the shorter array with zeros
    for pose in poses:
        if len(pose.body.data) < max_frame_count:
            desired_shape = list(pose.body.data.shape)
            desired_shape[0] = max_frame_count - len(pose.body.data)
            padding_tensor = ma.zeros(desired_shape)
            padding_tensor_conf = ma.ones(desired_shape[:-1])
            pose.body.data = ma.concatenate([pose.body.data, padding_tensor], axis=0)
            pose.body.confidence = ma.concatenate(
                [pose.body.confidence, padding_tensor_conf]
            )
    return poses


def set_masked_to_origin_position(pose: Pose) -> Pose:
    pose = pose.copy()
    pose.body = pose.body.zero_filled()
    pose.body.data[pose.body.data.mask]=0

    return pose


def pose_hide_low_conf(pose: Pose, confidence_threshold: float = 0.2) -> None:
    mask = pose.body.confidence <= confidence_threshold
    pose.body.confidence[mask] = 0
    stacked_confidence = np.stack([mask, mask, mask], axis=3)
    masked_data = ma.masked_array(pose.body.data, mask=stacked_confidence)
    pose.body.data = masked_data
