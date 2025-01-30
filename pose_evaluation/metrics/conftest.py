import shutil
from pathlib import Path
from typing import Callable, Union, Tuple, List
import torch
import numpy as np
import numpy.ma as ma
import pytest
import os
import copy

from pose_evaluation.metrics.distance_measure import PowerDistance
from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.metrics.dynamic_time_warping_metric import DTWMetric
from pose_evaluation.metrics.ham2pose_distances import Ham2PoseAPEDistance, Ham2PoseMSEDistance, Ham2PoseMaskedEuclideanDistance
from pose_evaluation.metrics.mje_metric import MeanJointErrorMetric
from pose_evaluation.metrics.pose_processors import get_standard_pose_processors
from pose_format import Pose
from pose_format.utils.generic import fake_pose
from pose_format.numpy import NumPyPoseBody

from pose_evaluation.utils.pose_utils import load_pose_file, copy_pose


@pytest.fixture(scope="session", autouse=True)
def clean_test_artifacts():
    """Fixture to clean up test artifacts before each test session."""
    test_artifacts_dir = Path(__file__).parent / "tests"  # Using Path
    if test_artifacts_dir.exists():
        shutil.rmtree(test_artifacts_dir)  # shutil.rmtree still works with Path
    test_artifacts_dir.mkdir(parents=True, exist_ok=True)  # Using Path.mkdir
    yield  # This allows the test session to run
    # (Optional) You can add cleanup logic here to run after the session if needed


@pytest.fixture(name="distance_matrix_shape_checker")
def fixture_distance_matrix_shape_checker() -> Callable[[int, int, torch.Tensor], None]:
    def _check_shape(hyp_count: int, ref_count: int, distance_matrix: torch.Tensor):

        expected_shape = torch.Size([hyp_count, ref_count])
        assert (
            distance_matrix.shape == expected_shape
        ), f"For M={hyp_count} hypotheses, N={ref_count} references,  Distance Matrix should be MxN={expected_shape}. Instead, received {distance_matrix.shape}"

    return _check_shape


@pytest.fixture(name="distance_range_checker")
def fixture_distance_range_checker() -> Callable[[Union[torch.Tensor, np.ndarray], float, float], None]:
    def _check_range(
        distances: Union[torch.Tensor, np.ndarray],
        min_val: float = 0,
        max_val: float = 2,
    ) -> None:
        max_distance = distances.max().item()
        min_distance = distances.min().item()

        # Use np.isclose for comparisons with tolerance
        assert (
            np.isclose(min_distance, min_val, atol=1e-6) or min_val <= min_distance <= max_val
        ), f"Minimum distance ({min_distance}) is outside the expected range [{min_val}, {max_val}]"
        assert (
            np.isclose(max_distance, max_val, atol=1e-6) or min_val <= max_distance <= max_val
        ), f"Maximum distance ({max_distance}) is outside the expected range [{min_val}, {max_val}]"

    return _check_range



utils_test_data_dir = Path(os.path.dirname(os.path.realpath(__file__))).parent/'utils' / 'test'/'test_data'

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




@pytest.fixture(scope="function")
def test_mediapipe_poses_zeros_and_ones_different_length(test_mediapipe_poses)->List[Pose]:
    hypothesis = copy_pose(test_mediapipe_poses[0])

    reference = copy_pose(test_mediapipe_poses[1])

    
    zeros_data = ma.array(np.zeros_like(hypothesis.body.data), mask=hypothesis.body.data.mask)
    hypothesis_body = NumPyPoseBody(fps=hypothesis.body.fps, data=zeros_data, confidence= hypothesis.body.confidence)

    ones_data = ma.array(np.ones_like(reference.body.data), mask=reference.body.data.mask)
    reference_body = NumPyPoseBody(fps= reference.body.fps, data=ones_data, confidence=reference.body.confidence)


    hypothesis = Pose(hypothesis.header, hypothesis_body)

    reference = Pose(reference.header, reference_body)


    return copy.deepcopy([hypothesis, reference]) 


@pytest.fixture(scope="function")
def test_mediapipe_poses_zeros_and_ones_same_length(test_mediapipe_poses)->List[Pose]:
    hypothesis = copy_pose(test_mediapipe_poses[0])
    reference = copy_pose(test_mediapipe_poses[0])

    zeros_data = ma.array(np.zeros_like(hypothesis.body.data), mask=hypothesis.body.data.mask)
    hypothesis_body = NumPyPoseBody(fps=hypothesis.body.fps, data=zeros_data, confidence= hypothesis.body.confidence)

    ones_data = ma.array(np.ones_like(reference.body.data), mask=reference.body.data.mask)
    reference_body = NumPyPoseBody(fps= reference.body.fps, data=ones_data, confidence=reference.body.confidence)


    hypothesis = Pose(hypothesis.header, hypothesis_body)
    reference = Pose(reference.header, reference_body)

    return copy.deepcopy([hypothesis, reference]) 



@pytest.fixture
def ham2pose_metrics_for_testing()->List[DistanceMetric]:
    metrics =[]
    MJEMetric = (
        MeanJointErrorMetric()
    )  # automatically sets distance measure, zero-padding.
    metrics.append(MJEMetric)

    Ham2Pose_DTW_MJE_Metric = DTWMetric(
        name="DTW_MJE",
        distance_measure=PowerDistance(2, 0),
        pose_preprocessors=get_standard_pose_processors(zero_pad_shorter=False),
    )

    Ham2Pose_nDTW_MJE_Metric = DTWMetric(
        name="nDTW_MJE",
        distance_measure=Ham2PoseMaskedEuclideanDistance(),
        pose_preprocessors=get_standard_pose_processors(zero_pad_shorter=False),
    )

    metrics.append(Ham2Pose_DTW_MJE_Metric)
    metrics.append(Ham2Pose_nDTW_MJE_Metric)

    # Ham2Pose APE is a PowerDistance. But with a few preprocessors.
    # 1. standard preprocessors
    # 2. then these: basically this is "zero_pad_shorter", and also setting masked values to zero.
    # if len(trajectory1) < len(trajectory2):
    #     diff = len(trajectory2) - len(trajectory1)
    #     trajectory1 = np.concatenate((trajectory1, np.zeros((diff, 3))))
    # elif len(trajectory2) < len(trajectory1):
    #     trajectory2 = np.concatenate((trajectory2, np.zeros((len(trajectory1) - len(trajectory2), 3))))
    # pose1_mask = np.ma.getmask(trajectory1)
    # pose2_mask = np.ma.getmask(trajectory2)
    # trajectory1[pose1_mask] = 0
    # trajectory1[pose2_mask] = 0
    # trajectory2[pose1_mask] = 0
    # trajectory2[pose2_mask] = 0
    # 3. pointwise aggregate by SUM
    # sq_error = np.power(trajectory1 - trajectory2, 2).sum(-1)
    # 4. trajectorywise aggregate by MEAN
    # np.sqrt(sq_error).mean()
    Ham2PoseAPEMetric = DistanceMetric(
        name="Ham2PoseAPEMetric",
        distance_measure=Ham2PoseAPEDistance(),
        pose_preprocessors=get_standard_pose_processors(
            zero_pad_shorter=True, set_masked_values_to_origin=True
        ),
    )
    metrics.append(Ham2PoseAPEMetric)

    # MSE from Ham2Pose is zero-padding, plus set to origin, and then squared error.
    # if len(trajectory1) < len(trajectory2):
    #     diff = len(trajectory2) - len(trajectory1)
    #     trajectory1 = np.concatenate((trajectory1, np.zeros((diff, 3))))
    # elif len(trajectory2) < len(trajectory1):
    #     trajectory2 = np.concatenate((trajectory2, np.zeros((len(trajectory1) - len(trajectory2), 3))))
    # pose1_mask = np.ma.getmask(trajectory1)
    # pose2_mask = np.ma.getmask(trajectory2)
    # trajectory1[pose1_mask] = 0
    # trajectory1[pose2_mask] = 0
    # trajectory2[pose1_mask] = 0
    # trajectory2[pose2_mask] = 0
    # sq_error = np.power(trajectory1 - trajectory2, 2).sum(-1)
    # return sq_error.mean()
    Ham2PoseMSEMetric = DistanceMetric(
        name="Ham2PoseMSEMetric",
        distance_measure=Ham2PoseMSEDistance(),
        pose_preprocessors=get_standard_pose_processors(
            zero_pad_shorter=True, set_masked_values_to_origin=True
        ),
    )
    metrics.append(Ham2PoseMSEMetric)
    return metrics