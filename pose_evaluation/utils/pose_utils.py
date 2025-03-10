from pathlib import Path
from typing import List, Tuple, Dict, Iterable
from collections import defaultdict
import numpy as np
from numpy import ma
from pose_format import Pose
from tqdm import tqdm


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


def reduce_poses_to_intersection(
    poses: Iterable[Pose],
    progress=False,
) -> List[Pose]:
    poses = list(poses)  # get a list, no need to copy

    # look at the first pose
    component_names = {c.name for c in poses[0].header.components}
    points = {c.name: set(c.points) for c in poses[0].header.components}

    # remove anything that other poses don't have
    for pose in tqdm(poses[1:], desc="reduce poses to intersection", disable=not progress):
        component_names.intersection_update({c.name for c in pose.header.components})
        for component in pose.header.components:
            points[component.name].intersection_update(set(component.points))

    # change datatypes to match get_components, then update the poses
    points_dict = {}
    for c_name in points.keys():
        points_dict[c_name] = list(points[c_name])
    poses = [pose.get_components(list(component_names), points_dict) for pose in poses]
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


def pose_hide_low_conf(pose: Pose, confidence_threshold: float = 0.2) -> None:
    mask = pose.body.confidence <= confidence_threshold
    pose.body.confidence[mask] = 0
    stacked_confidence = np.stack([mask, mask, mask], axis=3)
    masked_data = ma.masked_array(pose.body.data, mask=stacked_confidence)
    pose.body.data = masked_data
