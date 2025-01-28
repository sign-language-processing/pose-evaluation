import os
import json
import copy
import itertools
from pathlib import Path
from typing import List, Dict, Tuple
from pose_format import Pose
import pytest
from pose_evaluation.utils.pose_utils import load_pose_file
from pose_format.utils.generic import fake_pose
from pose_format.utils.openpose_135 import OpenPose_Components as openpose_135_components


utils_test_data_dir = Path(os.path.dirname(os.path.realpath(__file__))) / 'test'/'test_data'

@pytest.fixture(scope="function")
def test_mediapipe_poses_paths()->List[Path]:
    pose_file_paths = list(utils_test_data_dir.glob("*.pose"))
    return pose_file_paths

@pytest.fixture(scope="function")
def test_mediapipe_poses(test_mediapipe_poses_paths)->List[Pose]:
    original_poses = [load_pose_file(pose_path) for pose_path in test_mediapipe_poses_paths]
    # I ran into issues where if one test would modify a Pose, it would affect other tests. 
    # specifically, pose.header.components[0].name = unsupported_component_name in test_detect_format
    # this ensures we get a fresh object each time. 
    return copy.deepcopy(original_poses) 

# @pytest.fixture
# def pairs_of_identical_test_mediapipe_poses(test_mediapipe_poses)->List[Tuple[Pose, Pose]]:
#     poses =[]
#     for pose in test_mediapipe_poses:
#         poses.append(pose, pose)


@pytest.fixture
def standard_mediapipe_components_dict()->Dict[str, List[str]]:
    format_json = utils_test_data_dir/"mediapipe_components_and_points.json"
    with open(format_json, "r") as f:
        return json.load(f)

@pytest.fixture
def fake_openpose_poses(count:int=3)->List[Pose]:
    return [fake_pose(30) for _ in range(count)]

@pytest.fixture
def fake_openpose_135_poses(count:int=3)->List[Pose]:
    return [fake_pose(30, components=openpose_135_components) for _ in range(count)]
