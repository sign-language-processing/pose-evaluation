import json
import numpy as np
from typing import List, Dict, Tuple
import pytest
from pathlib import Path
from pose_format import Pose
from pose_format.utils.generic import pose_hide_legs
from pose_evaluation.utils.pose_utils import (
    load_pose_file,
    pose_remove_world_landmarks,
    remove_components,
    pose_remove_legs,
    pose_hide_low_conf,
    copy_pose,
    get_face_and_hands_from_pose,
    reduce_pose_components_to_intersection,
    preprocess_pose,
    get_component_names_and_points_dict,
    preprocess_poses,
    detect_format,
)


def test_load_poses_mediapipe(
    test_mediapipe_poses_paths: List[Path],
    standard_mediapipe_components_dict: Dict[str, List[str]],
):

    poses = [load_pose_file(pose_path) for pose_path in test_mediapipe_poses_paths]

    assert len(poses) == 3

    for pose in poses:
        # do they all have headers?
        assert pose.header is not None

        # check if the expected components are there.
        for component in pose.header.components:
            # should have specific expected components
            assert component.name in standard_mediapipe_components_dict

            # should have specific expected points
            assert sorted(component.points) == sorted(
                standard_mediapipe_components_dict[component.name]
            )

        # checking the data:
        # Frames, People, Points, Dims
        assert pose.body.data.ndim == 4

        # all frames have the standard shape?
        assert all(frame.shape == (1, 576, 3) for frame in pose.body.data)


def test_remove_specific_landmarks_mediapipe(
    test_mediapipe_poses: List[Pose],
    standard_mediapipe_components_dict: Dict[str, List[str]],
):
    for pose in test_mediapipe_poses:
        component_count = len(pose.header.components)
        assert component_count == len(standard_mediapipe_components_dict.keys())
        for component_name in standard_mediapipe_components_dict.keys():
            pose_with_component_removed = remove_components(pose, [str(component_name)])
            assert component_name not in pose_with_component_removed.header.components
            assert (
                len(pose_with_component_removed.header.components)
                == component_count - 1
            )


def test_pose_copy(test_mediapipe_poses: List[Pose]):
    for pose in test_mediapipe_poses:
        copy = copy_pose(pose)

        assert copy != pose  # Not the same object
        assert (
            pose.header.components != copy.header.components
        )  # header is also not the same object
        assert pose.body != copy.body  # also not the same
        assert np.array_equal(
            copy.body.data, pose.body.data
        )  # the data should have the same values

        assert sorted([c.name for c in pose.header.components]) == sorted(
            [c.name for c in copy.header.components]
        )  # same components
        assert (
            copy.header.total_points() == pose.header.total_points()
        )  # same number of points


def test_pose_remove_legs(test_mediapipe_poses: List[Pose]):
    points_that_should_be_hidden = ["KNEE", "HEEL", "FOOT", "TOE"]
    for pose in test_mediapipe_poses:
        # pose_hide_legs(pose)
        pose = pose_remove_legs(pose)

        for component in pose.header.components:
            point_names = [point.upper() for point in component.points]
            for point_name in point_names:
                for point_that_should_be_hidden in points_that_should_be_hidden:
                    assert point_that_should_be_hidden not in point_name


def test_pose_remove_legs_openpose(fake_openpose_poses):
    for pose in fake_openpose_poses:
        with pytest.raises(NotImplementedError):
            pose_remove_legs(pose)


def test_reduce_pose_components_to_intersection(
    test_mediapipe_poses: List[Pose],
    standard_mediapipe_components_dict: Dict[str, List[str]],
):

    test_poses_with_one_reduced = [copy_pose(pose) for pose in test_mediapipe_poses]
    pose_with_only_face_and_hands = get_face_and_hands_from_pose(
        test_poses_with_one_reduced.pop()
    )
    test_poses_with_one_reduced.append(pose_with_only_face_and_hands)
    assert len(test_mediapipe_poses) == len(test_poses_with_one_reduced)

    original_component_count = len(
        standard_mediapipe_components_dict.keys()
    )  # 5, at time of writing

    target_component_count = 3  # face, left hand, right hand
    assert (
        len(pose_with_only_face_and_hands.header.components) == target_component_count
    )

    reduced_poses = reduce_pose_components_to_intersection(test_poses_with_one_reduced)
    for reduced_pose in reduced_poses:
        assert len(reduced_pose.header.components) == target_component_count

    # check if the originals are unaffected
    assert all(
        [
            len(pose.header.components) == original_component_count
            for pose in test_mediapipe_poses
        ]
    )


def test_remove_world_landmarks(test_mediapipe_poses: List[Pose]):
    for pose in test_mediapipe_poses:
        component_names = [c.name for c in pose.header.components]
        starting_component_count = len(pose.header.components)
        assert "POSE_WORLD_LANDMARKS" in component_names

        pose = pose_remove_world_landmarks(pose)
        component_names = [c.name for c in pose.header.components]
        assert "POSE_WORLD_LANDMARKS" not in component_names
        ending_component_count = len(pose.header.components)

        assert ending_component_count == starting_component_count - 1


def test_remove_one_point_and_one_component(test_mediapipe_poses: List[Pose]):
    component_to_drop = "POSE_WORLD_LANDMARKS"
    point_to_drop = "LEFT_KNEE"
    for pose in test_mediapipe_poses:
        original_component_names, original_points_dict = (
            get_component_names_and_points_dict(pose)
        )

        assert component_to_drop in original_component_names
        assert point_to_drop in original_points_dict["POSE_LANDMARKS"]
        reduced_pose = remove_components(pose, component_to_drop, point_to_drop)
        new_component_names, new_points_dict = get_component_names_and_points_dict(
            reduced_pose
        )
        assert component_to_drop not in new_component_names
        assert point_to_drop not in new_points_dict["POSE_LANDMARKS"]


def test_detect_format(
    fake_openpose_poses, fake_openpose_135_poses, test_mediapipe_poses
):
    for pose in fake_openpose_poses:
        assert detect_format(pose) == "openpose"

    for pose in fake_openpose_135_poses:
        assert detect_format(pose) == "openpose_135"

    for pose in test_mediapipe_poses:
        assert detect_format(pose) == "mediapipe"

    for pose in test_mediapipe_poses:
        unsupported_component_name = "UNSUPPORTED"
        pose.header.components[0].name = unsupported_component_name
        pose = pose.get_components(["UNSUPPORTED"])
        component_names, _ = get_component_names_and_points_dict(pose)
        assert len(pose.header.components) == 1

        # pose.header.components[0]
        # assert pose.header.components[0] == changing_component
        with pytest.raises(
            ValueError, match="Unknown pose header schema with component names"
        ):
            detect_format(pose)


def test_preprocess_pose(test_mediapipe_poses_paths: List[Path]):
    poses = [load_pose_file(pose_path) for pose_path in test_mediapipe_poses_paths]
    preprocessed_poses = []

    for pose in poses:
        processed_pose = preprocess_pose(pose,
                        normalize_poses=True,
                        remove_legs=True,
                        remove_world_landmarks=True,
                        conf_threshold_to_drop_points=0.2)
        #TODO: check expected result


def test_preprocess_poses(test_mediapipe_poses: List[Pose]):

    preprocessed_poses = preprocess_poses(
        test_mediapipe_poses,
        normalize_poses=True,
        reduce_poses_to_common_components=True,
        remove_world_landmarks=True,
        remove_legs=True,
        conf_threshold_to_drop_points=0.2,
    )

    for pose in preprocessed_poses:
        component_names, points_dict = get_component_names_and_points_dict(pose)
        assert "LEFT_KNEE" not in points_dict["POSE_LANDMARKS"]
        assert "POSE_WORLD_LANDMARKS" not in component_names



def test_hide_low_conf(test_mediapipe_poses: List[Pose]):
    copies = [copy_pose(pose) for pose in test_mediapipe_poses]
    for pose, copy in zip(test_mediapipe_poses, copies):
        pose_hide_low_conf(pose, 1.0)

        assert np.array_equal(pose.body.confidence, copy.body.confidence) == False
