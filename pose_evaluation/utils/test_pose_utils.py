from pathlib import Path
from typing import List, Dict

import numpy as np
import numpy.ma as ma  # pylint: disable=consider-using-from-import

import pytest
from pose_format import Pose
from pose_format.utils.generic import detect_known_pose_format, pose_hide_legs

from pose_evaluation.utils.pose_utils import (
    load_pose_file,
    pose_remove_world_landmarks,
    pose_hide_low_conf,
    get_face_and_hands_from_pose,
    reduce_poses_to_intersection,
    get_component_names_and_points_dict,
    zero_pad_shorter_poses,
)


def test_load_poses_mediapipe(
    mediapipe_poses_test_data_paths: List[Path],
    standard_mediapipe_components_dict: Dict[str, List[str]],
):
    poses = [load_pose_file(pose_path) for pose_path in mediapipe_poses_test_data_paths]

    assert len(poses) == 3

    for pose in poses:
        # do they all have headers?
        assert pose.header is not None

        # check if the expected components are there.
        for component in pose.header.components:
            # should have specific expected components
            assert component.name in standard_mediapipe_components_dict

            # should have specific expected points
            assert sorted(component.points) == sorted(standard_mediapipe_components_dict[component.name])

        # checking the data:
        # Frames, People, Points, Dims
        assert pose.body.data.ndim == 4

        # all frames have the standard shape?
        assert all(frame.shape == (1, 576, 3) for frame in pose.body.data)


def test_remove_specific_landmarks_mediapipe(
    mediapipe_poses_test_data: List[Pose],
    standard_mediapipe_components_dict: Dict[str, List[str]],
):
    for pose in mediapipe_poses_test_data:
        component_count = len(pose.header.components)
        assert component_count == len(standard_mediapipe_components_dict.keys())
        for component_name in standard_mediapipe_components_dict.keys():
            pose_with_component_removed = pose.remove_components([str(component_name)])
            assert component_name not in pose_with_component_removed.header.components
            assert len(pose_with_component_removed.header.components) == component_count - 1


def test_pose_copy(mediapipe_poses_test_data: List[Pose]):
    for pose in mediapipe_poses_test_data:
        copy = pose.copy()

        assert copy != pose  # Not the same object
        assert pose.body != copy.body  # also not the same
        assert np.array_equal(copy.body.data, pose.body.data)  # the data should have the same values

        #
        assert sorted([c.name for c in pose.header.components]) == sorted(
            [c.name for c in copy.header.components]
        ), "should have the same components"
        assert copy.header.total_points() == pose.header.total_points(), "should have the same number of points"


def test_pose_remove_legs(mediapipe_poses_test_data: List[Pose]):
    points_that_should_be_removed = [
        "LEFT_KNEE",
        "LEFT_HEEL",
        "LEFT_FOOT",
        "LEFT_TOE",
        "LEFT_FOOT_INDEX",
        "RIGHT_KNEE",
        "RIGHT_HEEL",
        "RIGHT_FOOT",
        "RIGHT_TOE",
        "RIGHT_FOOT_INDEX",
    ]
    for pose in mediapipe_poses_test_data:
        c_names = [c.name for c in pose.header.components]
        assert "POSE_LANDMARKS" in c_names
        pose_landmarks_index = c_names.index("POSE_LANDMARKS")
        assert "LEFT_KNEE" in pose.header.components[pose_landmarks_index].points

        pose_with_legs_removed = pose_hide_legs(pose, remove=True)
        assert pose_with_legs_removed != pose
        assert pose_with_legs_removed.header != pose.header
        assert pose_with_legs_removed.header.components != pose.header.components
        new_c_names = [c.name for c in pose_with_legs_removed.header.components]
        assert "POSE_LANDMARKS" in new_c_names

        for component in pose_with_legs_removed.header.components:
            point_names = [point.upper() for point in component.points]
            for point_name in point_names:
                for point_that_should_be_hidden in points_that_should_be_removed:
                    assert (
                        point_that_should_be_hidden not in point_name
                    ), f"{component.name}: {point_names}"


def test_pose_remove_legs_openpose(fake_openpose_poses):
    points_that_should_be_removed = [
        "Hip",
        "Knee",
        "Ankle",
        "BigToe",
        "SmallToe",
        "Heel",
    ]
    for pose in fake_openpose_poses:
        pose_with_legs_removed = pose_hide_legs(pose, remove=True)

        for component in pose_with_legs_removed.header.components:
            point_names = list(point for point in component.points)
            for point_name in point_names:
                for point_that_should_be_hidden in points_that_should_be_removed:
                    assert (
                        point_that_should_be_hidden not in point_name
                    ), f"{component.name}: {point_names}"


def test_reduce_pose_components_to_intersection(
    mediapipe_poses_test_data: List[Pose],
    standard_mediapipe_components_dict: Dict[str, List[str]],
):
    test_poses_with_one_reduced = [pose.copy() for pose in mediapipe_poses_test_data]

    pose_with_only_face_and_hands_and_no_wrist = get_face_and_hands_from_pose(test_poses_with_one_reduced.pop())

    c_names, p_dict = get_component_names_and_points_dict(pose_with_only_face_and_hands_and_no_wrist)

    new_p_dict = {}
    for c_name, p_list in p_dict.items():
        new_p_dict[c_name] = [point_name for point_name in p_list if "WRIST" not in point_name]

    pose_with_only_face_and_hands_and_no_wrist = pose_with_only_face_and_hands_and_no_wrist.get_components(
        c_names, new_p_dict
    )
    test_poses_with_one_reduced.append(pose_with_only_face_and_hands_and_no_wrist)
    assert len(mediapipe_poses_test_data) == len(test_poses_with_one_reduced)

    # 5, at time of writing
    original_component_count = len(standard_mediapipe_components_dict.keys())

    num_components = len(pose_with_only_face_and_hands_and_no_wrist.header.components)
    assert num_components == 3, "should only have face, left hand and right hand"

    target_point_count = pose_with_only_face_and_hands_and_no_wrist.header.total_points()

    reduced_poses = reduce_poses_to_intersection(test_poses_with_one_reduced)
    for reduced_pose in reduced_poses:
        assert len(reduced_pose.header.components) == 3, "should only have face, left hand and right hand"
        assert reduced_pose.header.total_points() == target_point_count

    # check if the originals are unaffected
    assert all(len(pose.header.components) == original_component_count for pose in mediapipe_poses_test_data)


def test_remove_world_landmarks(mediapipe_poses_test_data: List[Pose]):
    for pose in mediapipe_poses_test_data:
        component_names = [c.name for c in pose.header.components]
        starting_component_count = len(pose.header.components)
        assert "POSE_WORLD_LANDMARKS" in component_names

        pose = pose_remove_world_landmarks(pose)
        component_names = [c.name for c in pose.header.components]
        assert "POSE_WORLD_LANDMARKS" not in component_names
        ending_component_count = len(pose.header.components)

        assert ending_component_count == starting_component_count - 1


def test_remove_one_point_and_one_component(mediapipe_poses_test_data: List[Pose]):
    component_to_drop = "POSE_WORLD_LANDMARKS"
    point_to_drop = "LEFT_KNEE"
    for pose in mediapipe_poses_test_data:
        original_component_names, original_points_dict = get_component_names_and_points_dict(pose)

        assert component_to_drop in original_component_names
        assert point_to_drop in original_points_dict["POSE_LANDMARKS"]
        reduced_pose = pose.remove_components(
            component_to_drop, {"POSE_LANDMARKS": [point_to_drop]}
        )
        new_component_names, new_points_dict = get_component_names_and_points_dict(
            reduced_pose
        )
        assert component_to_drop not in new_component_names
        assert point_to_drop not in new_points_dict["POSE_LANDMARKS"]


def test_detect_format(fake_openpose_poses, fake_openpose_135_poses, mediapipe_poses_test_data):
    for pose in fake_openpose_poses:
        assert detect_known_pose_format(pose) == "openpose"

    for pose in fake_openpose_135_poses:
        assert detect_known_pose_format(pose) == "openpose_135"

    for pose in mediapipe_poses_test_data:
        assert detect_known_pose_format(pose) == "holistic"

    for pose in mediapipe_poses_test_data:
        unsupported_component_name = "UNSUPPORTED"
        pose.header.components[0].name = unsupported_component_name
        pose = pose.get_components(["UNSUPPORTED"])
        assert len(pose.header.components) == 1

        with pytest.raises(
            ValueError,
            match="Could not detect pose format, unknown pose header schema with component names",
        ):
            detect_known_pose_format(pose)


def test_hide_low_conf(mediapipe_poses_test_data: List[Pose]):
    copies = [pose.copy() for pose in mediapipe_poses_test_data]
    for pose, copy in zip(mediapipe_poses_test_data, copies):
        pose_hide_low_conf(pose, 1.0)

        assert np.array_equal(pose.body.confidence, copy.body.confidence) is False


def test_zero_pad_shorter_poses(mediapipe_poses_test_data: List[Pose]):
    copies = [pose.copy() for pose in mediapipe_poses_test_data]

    max_len = max(len(pose.body.data) for pose in mediapipe_poses_test_data)
    padded_poses = zero_pad_shorter_poses(mediapipe_poses_test_data)

    for i, padded_pose in enumerate(padded_poses):
        assert mediapipe_poses_test_data[i] != padded_poses[i], "shouldn't be the same object"
        old_length = len(copies[i].body.data)
        new_length = len(padded_pose.body.data)
        assert new_length == max_len
        if old_length == new_length:
            assert old_length == max_len

        # does the confidence match?
        assert padded_pose.body.confidence.shape == padded_pose.body.data.shape[:-1]
