from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Iterable, Union, Set

import numpy as np
from numpy import ma
from pose_format import Pose
from pose_format.utils.generic import detect_known_pose_format
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


def first_frame_pad_shorter_poses(poses: Iterable[Pose]) -> List[Pose]:
    poses = [pose.copy() for pose in poses]
    max_frame_count = max(len(pose.body.data) for pose in poses)

    for pose in poses:
        current_len = len(pose.body.data)
        if current_len < max_frame_count:
            pad_count = max_frame_count - current_len
            first_frame = pose.body.data[0:1]  # Keep dims
            first_conf = pose.body.confidence[0:1]

            padding_tensor = ma.concatenate([first_frame] * pad_count, axis=0)
            padding_tensor_conf = ma.concatenate([first_conf] * pad_count, axis=0)

            # PAD AT BEGINNING
            pose.body.data = ma.concatenate([padding_tensor, pose.body.data], axis=0)
            pose.body.confidence = ma.concatenate([padding_tensor_conf, pose.body.confidence], axis=0)

    return poses


def get_youtube_asl_mediapipe_keypoints(pose: Pose):
    if detect_known_pose_format(pose) != "holistic":
        return pose

    # https://arxiv.org/pdf/2306.15162
    # For each hand, we use all 21 landmark points.
    # Colin: So that's
    # For the pose, we use 6 landmark points, for the shoulders, elbows and hips
    # These are indices 11, 12, 13, 14, 23, 24
    # For the face, we use 37 landmark points, from the eyes, eyebrows, lips, and face outline.
    # These are indices 0, 4, 13, 14, 17, 33, 37, 39, 46, 52, 55, 61, 64, 81, 82, 93, 133, 151, 152, 159, 172, 178,
    # 181, 263, 269, 276, 282, 285, 291, 294, 311, 323, 362, 386, 397, 468, 473.
    # Colin: note that these are with refine_face_landmarks on, and are relative to the component itself. Working it all out the result is:
    chosen_component_names = ["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"]
    points_dict = {
        "POSE_LANDMARKS": ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP", "LEFT_ELBOW", "RIGHT_ELBOW"],
        "FACE_LANDMARKS": [
            "0",
            "4",
            "13",
            "14",
            "17",
            "33",
            "37",
            "39",
            "46",
            "52",
            "55",
            "61",
            "64",
            "81",
            "82",
            "93",
            "133",
            "151",
            "152",
            "159",
            "172",
            "178",
            "181",
            "263",
            "269",
            "276",
            "282",
            "285",
            "291",
            "294",
            "311",
            "323",
            "362",
            "386",
            "397",
        ],
    }

    # check if we have the extra points from refine_face_landmarks
    additional_face_points = ["468", "473"]
    for additional_point in additional_face_points:
        try:
            point_index = pose.header.get_point_index("FACE_LANDMARKS", additional_point)
            points_dict["FACE_LANDMARKS"].append(additional_point)
        except ValueError:
            # not in the list
            pass
    pose = pose.get_components(components=chosen_component_names, points=points_dict)
    return pose


def pose_hide_low_conf(pose: Pose, confidence_threshold: float = 0.2) -> None:
    mask = pose.body.confidence <= confidence_threshold
    pose.body.confidence[mask] = 0
    stacked_confidence = np.stack([mask, mask, mask], axis=3)
    masked_data = ma.masked_array(pose.body.data, mask=stacked_confidence)
    pose.body.data = masked_data


def pose_mask_invalid_values(pose: Pose, overwrite_confidence=True) -> Pose:
    pose = pose.copy()
    pose.body.data = ma.masked_invalid(pose.body.data)  # mask all invalid values
    if overwrite_confidence:
        # Zero confidence wherever any coordinate is masked
        invalid_mask = pose.body.data.mask
        if invalid_mask is not ma.nomask:
            pose.body.confidence[invalid_mask.any(axis=-1)] = 0
    return pose


def pose_fill_masked_or_invalid(pose: Pose, fill_val=0.0, overwrite_confidence=True) -> Pose:
    pose = pose_mask_invalid_values(pose, overwrite_confidence=overwrite_confidence)

    # Fill it...
    pose.body.data = ma.masked_array(
        pose.body.data.filled(fill_val), mask=False
    )  # Replace masked values. Still a MaskedArray for compatibility

    # ...and overwrite the confidence as well.
    if overwrite_confidence:
        # update the confidence to all ones. We are now fully "confident" that these are this value
        pose.body.confidence = np.ones_like(pose.body.confidence, dtype=pose.body.confidence.dtype)

    return pose


def add_z_offsets_to_pose(pose: Pose, speed: float = 1.0) -> Pose:

    offset = speed / pose.body.fps
    # Assuming pose.data is a numpy masked array
    pose_data = pose.body.data  # Shape: (frames, persons, keypoints, xyz)

    # Create an offset array that only modifies the Z-dimension (index 2)
    offsets = ma.arange(pose_data.shape[0]).reshape(-1, 1, 1, 1) * offset

    # Apply the offsets only to the Z-axis (index 2), preserving masks
    pose_data[:, :, :, 2] += offsets[:, :, :, 0]

    pose.body.data = pose_data
    return pose
