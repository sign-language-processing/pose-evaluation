from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Iterable, Union, Set

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
    poses: Iterable["Pose"],
    progress: bool = False,
    debug: bool = False,
) -> List["Pose"]:
    poses = list(poses)  # Ensure it's a list so we can iterate multiple times

    if not poses:
        if debug:
            print("No poses provided, returning empty list.")
        return []

    # === Stage 1: Reduce to common components ===
    common_components: Set[str] = {comp.name for comp in poses[0].header.components}
    if debug:
        print(f"Initial components from first pose: {sorted(common_components)}")

    for i, pose in enumerate(tqdm(poses[1:], desc="Intersecting components", disable=not progress)):
        current_components = {comp.name for comp in pose.header.components}
        if debug:
            print(f"Pose {i+1} components: {sorted(current_components)}")
        common_components.intersection_update(current_components)
        if debug:
            print(f"Updated common components: {sorted(common_components)}")

    common_components_list = sorted(common_components)

    if debug:
        print(f"Final list of common components: {common_components_list}")
        print("Applying get_components to all poses...")

    # Apply component filtering
    poses = [pose.get_components(common_components_list) for pose in poses]

    # === Stage 2: Reduce to common points within each component ===
    common_points: Dict[str, Set[str]] = {comp.name: set(comp.points) for comp in poses[0].header.components}

    if debug:
        print("Initial points per component:")
        for name, pts in common_points.items():
            print(f"  {name}: {sorted(pts)}")

    for i, pose in enumerate(tqdm(poses[1:], desc="Intersecting points", disable=not progress)):
        current_points = {comp.name: set(comp.points) for comp in pose.header.components}
        for name in common_points:
            if debug:
                before = common_points[name]
                current = current_points.get(name, set())
                print(f"Pose {i+1}, component '{name}': intersecting {sorted(before)} with {sorted(current)}")
            common_points[name].intersection_update(current_points.get(name, set()))
            if debug:
                print(f"Updated points for '{name}': {sorted(common_points[name])}")

    # Final dictionary of intersected points
    final_points_dict: Dict[str, List[str]] = {name: sorted(list(pts)) for name, pts in common_points.items()}

    if debug:
        print("Final points per component to apply:")
        for name, pts in final_points_dict.items():
            print(f"  {name}: {pts}")

    # Apply final component + point reduction
    reduced_poses = [pose.get_components(common_components_list, points=final_points_dict) for pose in poses]

    return reduced_poses


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
            pose.body.confidence = ma.concatenate([pose.body.confidence, padding_tensor_conf])
    return poses


def pose_hide_low_conf(pose: Pose, confidence_threshold: float = 0.2) -> None:
    mask = pose.body.confidence <= confidence_threshold
    pose.body.confidence[mask] = 0
    stacked_confidence = np.stack([mask, mask, mask], axis=3)
    masked_data = ma.masked_array(pose.body.data, mask=stacked_confidence)
    pose.body.data = masked_data


def pose_fill_masked_or_invalid(pose: Pose, fill_val=0.0) -> Pose:
    pose = pose.copy()
    pose.body.data = ma.masked_invalid(pose.body.data)  # replace all unmasked invalid values
    pose.body.data = ma.array(
        pose.body.data.filled(fill_val), mask=False
    )  # Replace masked values. Still a MaskedArray for compatibility
    return pose
