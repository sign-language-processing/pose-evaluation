import math
import unittest
from typing import Tuple, List, get_args

import numpy as np
import pytest
import random

from pose_format import Pose
from pose_format.utils.generic import fake_pose, detect_known_pose_format
from pose_format.numpy import NumPyPoseBody

from pose_evaluation.metrics.distance_metric import DistanceMetric, ValidPointDistanceKinds
from pose_evaluation.metrics import dynamic_time_warping_metric
# from pose_evaluation.metrics import ape_metric, mse_metric, ndtw_mje_metric

# pytest --cov --cov-report lcov .

DISTANCE_KINDS_TO_CHECK = get_args(ValidPointDistanceKinds)


# def get_test_poses(frame_count1: int, frame_count2: int, people_count:int=3, point_count:int=5, fill_value=2, fps=25)-> Tuple[Pose, Pose]:
#     # point_coordinate_count = 3 # x, y, z 

#     hypothesis = fake_pose(num_frames=frame_count1, fps=fps)
#     reference = fake_pose(num_frames=frame_count2, fps=fps)

#     data_tensor = np.full_like(hypothesis.body.data, fill_value=fill_value)
#     zeros_tensor = np.zeros_like(reference.body.data)
#     data_confidence = np.ones(data_tensor.shape[:-1])
#     zeros_confidence = np.ones(zeros_tensor.shape[:-1])

#     hypothesis.body.data = data_tensor
#     reference.body.data = zeros_tensor

#     # fake_pose_header = fake_pose(1).header


#     # hypothesis = Pose(header=fake_pose_1.header, body=NumPyPoseBody(fps=fps, data=data_tensor, confidence=data_confidence))
#     # reference = Pose(header=fake_pose_2.header, body=NumPyPoseBody(fps=fps, data=zeros_tensor, confidence=zeros_confidence))
#     return hypothesis, reference
    

# def test_get_test_poses():
#     # hyp, ref = test_pose_pair
#     frame_count1 = 20
#     frame_count2 = 30
#     people_count = 1
#     point_count = 137
#     coordinate_count_per_point = 3
#     fill_value = 2
#     hyp, ref = get_test_poses(frame_count1=frame_count1,frame_count2=frame_count2, point_count=point_count, fill_value=fill_value)

#     assert hyp.body.data.shape == (frame_count1, people_count, point_count, coordinate_count_per_point)
#     assert ref.body.data.shape == (frame_count2, people_count, point_count, coordinate_count_per_point)
#     # assert np.sum(hyp.body.data) == frame_count1*people_count*point_count*coordinate_count_per_point*fill_value
#     assert np.sum(ref.body.data) == 0

def test_data_validity(test_mediapipe_poses:List[Pose], test_mediapipe_poses_zeros_and_ones_different_length, test_mediapipe_poses_zeros_and_ones_same_length):
    assert len(test_mediapipe_poses) == 3
    assert len(test_mediapipe_poses_zeros_and_ones_different_length) == 2
    assert len(test_mediapipe_poses_zeros_and_ones_same_length) == 2
    test_mediapipe_poses.extend(test_mediapipe_poses_zeros_and_ones_same_length)
    test_mediapipe_poses.extend(test_mediapipe_poses_zeros_and_ones_different_length)
    
    for pose in test_mediapipe_poses:
        assert np.count_nonzero(np.isnan(pose.body.data)) == 0
        assert pose.header.num_dims() == pose.body.data.shape[-1]
        assert pose.body.confidence.shape == pose.body.data.shape[:-1]
        
        assert detect_known_pose_format(pose) == "holistic"

def test_data_zeros_and_ones(test_mediapipe_poses_zeros_and_ones_different_length):
    hyp, ref = test_mediapipe_poses_zeros_and_ones_different_length[0], test_mediapipe_poses_zeros_and_ones_different_length[1]
    assert np.ma.sum(hyp.body.data) == 0
    
    for frame_data in ref.body.data:
        for person_data in frame_data:
            for keypoint_data in person_data:
                if np.ma.count_masked(keypoint_data) < len(keypoint_data):

                    assert np.ma.sum(keypoint_data) == len(keypoint_data) - np.ma.count_masked(keypoint_data)


@pytest.mark.parametrize("metric_name", DISTANCE_KINDS_TO_CHECK)
def test_preprocessing(metric_name: ValidPointDistanceKinds, test_mediapipe_poses:List[Pose]):
    metric = DistanceMetric(point_distance_calculation_kind=metric_name)
    for pose in test_mediapipe_poses:
        assert np.count_nonzero(np.isnan(pose.body.data)) == 0
    poses = metric.preprocess_poses(test_mediapipe_poses)

    
    for pose in poses:
        data = pose.body.data
        assert np.count_nonzero(np.isnan(data)) ==0
        
        assert isinstance(data, np.ma.MaskedArray)
        assert len(data) == len(poses[0].body.data)

@pytest.mark.parametrize("metric_name", DISTANCE_KINDS_TO_CHECK)
def test_preprocessing_with_zeros_and_ones_different_length(metric_name: ValidPointDistanceKinds, test_mediapipe_poses_zeros_and_ones_different_length:List[Pose]):
    metric = DistanceMetric(point_distance_calculation_kind=metric_name, normalize_poses=False) # normalizing when they're all zeros gives an error
    
    for pose in test_mediapipe_poses_zeros_and_ones_different_length:
        assert np.count_nonzero(np.isnan(pose.body.data)) == 0
    poses = metric.preprocess_poses(test_mediapipe_poses_zeros_and_ones_different_length)

    
    for pose in poses:
        data = pose.body.data
        assert np.count_nonzero(np.isnan(data)) ==0
        
        assert isinstance(data, np.ma.MaskedArray)
        assert len(data) == len(poses[0].body.data)

        

        



@pytest.mark.parametrize("metric_name", DISTANCE_KINDS_TO_CHECK)
def test_base_distance_metric_scores_equal_length(metric_name:ValidPointDistanceKinds, test_mediapipe_poses_zeros_and_ones_same_length):

    # hypothesis, reference = get_test_poses(2, 3)
    # hypothesis, reference = test_mediapipe_poses[0], test_mediapipe_poses[1]
    metric = DistanceMetric(point_distance_calculation_kind=metric_name, normalize_poses=False) # gives me nans in this case
    fill_value =1

    hyp, ref = test_mediapipe_poses_zeros_and_ones_same_length[0], test_mediapipe_poses_zeros_and_ones_same_length[1]
    coordinates_per_point = hyp.body.data.shape[-1]
    point_count = np.prod(hyp.body.confidence.shape)

    assert np.count_nonzero(np.isnan(hyp.body.data)) == 0
    assert np.count_nonzero(np.isnan(ref.body.data)) == 0
    
    if metric_name == 'euclidean':
        # euclidean difference per point: 
        # between (2,2,2) and (0,0,0) is 3.4641, 
        # aka sqrt((2-0)^2 +(2-0)^2 +(2-0^2)
        expected_difference_per_point = np.sqrt(fill_value*fill_value*coordinates_per_point)
        expected_distance = expected_difference_per_point # it's a mean value, they should all be the same
        score = metric.score(hyp, ref)

        
        

    elif metric_name == 'manhattan':
        
        expected_difference_per_point = fill_value * coordinates_per_point 
        expected_distance = expected_difference_per_point

        score = metric.score(hyp, ref)
        # point_count = np.prod(hyp.body.confidence.shape)
        assert score == expected_difference_per_point # mean error for every pair of spatial points is the same
    assert np.isclose(score, expected_distance)        
    assert isinstance(score, float)  # Check if the score is a float


def get_all_subclasses(base_class):
    """Recursively discover all subclasses of a given base class."""
    subclasses = set(base_class.__subclasses__())
    for subclass in base_class.__subclasses__():
        subclasses.update(get_all_subclasses(subclass))
    return subclasses

# @pytest.mark.parametrize('distance_metric_to_test',get_all_subclasses(DistanceMetric))
# def test_all_distance_metrics_and_kinds(distance_metric_to_test):
#     for kind in DISTANCE_KINDS_TO_CHECK:
#         metric = distance_metric_to_test(kind)
#         assert isinstance(metric, DistanceMetric)



def test_get_subclasses_for_distance_metric():
    distance_metrics= get_all_subclasses(DistanceMetric)
    assert len(distance_metrics) > 0

def generate_test_cases(base_class, kinds):
    """Generate tuples of (metric_class, kind) for parameterization."""
    subclasses = get_all_subclasses(base_class)
    return [(subclass, kind) for subclass in subclasses for kind in kinds]

# Parameterize with (metric_class, kind)
@pytest.mark.parametrize("metric_class,kind", generate_test_cases(DistanceMetric, DISTANCE_KINDS_TO_CHECK))
def test_distance_metric_calculations(metric_class, kind):
    """Test all DistanceMetric subclasses with various 'kinds'."""
    metric = metric_class(kind)

    # if kind == "default":
    #     result = metric.calculate(3, 7)
    # elif kind == "weighted" and hasattr(metric, "calculate"):  # Check for additional arguments
    #     result = metric.calculate(3, 7)  # Modify if weighted args are supported
    # else:
    #     pytest.skip(f"{metric_class} does not support kind '{kind}'")
    # assert result is not None  # Example assertion


def test_distance_metric_invalid_kind(test_mediapipe_poses_zeros_and_ones_same_length):
    normalize_poses = False # our test data makes normalize divide by zero
    

    with pytest.raises(NotImplementedError, match="invalid distance function"):
        metric = DistanceMetric(point_distance_calculation_kind="invalid", normalize_poses=normalize_poses) # type: ignore
        metric.score(test_mediapipe_poses_zeros_and_ones_same_length[0], test_mediapipe_poses_zeros_and_ones_same_length[1])


    # what if we do one that is in scipy?
    metric = DistanceMetric(point_distance_calculation_kind="chebyshev", normalize_poses=normalize_poses) # type: ignore 
    metric.score(test_mediapipe_poses_zeros_and_ones_same_length[0], test_mediapipe_poses_zeros_and_ones_same_length[1])


@pytest.mark.parametrize("metric_class,kind", generate_test_cases(DistanceMetric, DISTANCE_KINDS_TO_CHECK))
def test_scores_are_symmetric(metric_class, kind, test_mediapipe_poses:List[Pose]):
    metric = metric_class(spatial_distance_function_kind=kind)

    # hypothesis, reference = get_test_poses(2, 3)
    hypothesis, reference = test_mediapipe_poses[0], test_mediapipe_poses[1]


    score1 = metric.score(hypothesis, reference)
    # pylint: disable=arguments-out-of-order
    score2 = metric.score(reference, hypothesis)
    assert np.isclose(score1, score2)