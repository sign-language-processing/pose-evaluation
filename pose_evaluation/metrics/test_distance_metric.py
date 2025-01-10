import math
import unittest
from typing import Tuple, List, get_args

import numpy as np
import pytest
import random

from pose_format import Pose
from pose_format.utils.generic import fake_pose
from pose_format.numpy import NumPyPoseBody

from pose_evaluation.metrics.distance_metric import DistanceMetric, ValidDistanceKinds

DISTANCE_KINDS_TO_CHECK = get_args(ValidDistanceKinds)

def get_test_poses(frame_count1: int, frame_count2: int, people_count:int=3, point_count:int=4, fill_value=2)-> Tuple[Pose, Pose]:
    point_coordinate_count = 3 # x, y, z 

    data_tensor = np.full((frame_count1, people_count, point_count, point_coordinate_count), fill_value=fill_value)
    zeros_tensor = np.zeros((frame_count2, people_count, point_count, point_coordinate_count))
    data_confidence = np.ones(data_tensor.shape[:-1])
    zeros_confidence = np.ones(zeros_tensor.shape[:-1])

    fake_pose_header = fake_pose(1).header

    hypothesis = Pose(header=fake_pose_header, body=NumPyPoseBody(fps=1, data=data_tensor, confidence=data_confidence))
    reference = Pose(header=fake_pose_header, body=NumPyPoseBody(fps=1, data=zeros_tensor, confidence=zeros_confidence))
    return hypothesis, reference
    

def test_get_test_poses():
    # hyp, ref = test_pose_pair
    frame_count1 = 20
    frame_count2 = 30
    people_count = 3
    point_count = 4
    coordinate_count_per_point = 3
    fill_value = 2
    hyp, ref = get_test_poses(frame_count1=frame_count1,frame_count2=frame_count2, point_count=point_count, fill_value=fill_value)
    assert hyp.body.data.shape == (frame_count1, people_count, point_count, coordinate_count_per_point)
    assert ref.body.data.shape == (frame_count2, people_count, point_count, coordinate_count_per_point)
    assert np.sum(hyp.body.data) == frame_count1*people_count*point_count*coordinate_count_per_point*fill_value
    assert np.sum(ref.body.data) == 0






@pytest.mark.parametrize("metric_name", DISTANCE_KINDS_TO_CHECK)
def test_scores_are_symmetric(metric_name: ValidDistanceKinds):

    hypothesis, reference = get_test_poses(2, 3)

    metric = DistanceMetric(metric_name)
    score1 = metric.score(hypothesis, reference)
    # pylint: disable=arguments-out-of-order
    score2 = metric.score(reference, hypothesis)
    assert np.isclose(score1, score2)


@pytest.mark.parametrize("metric_name", DISTANCE_KINDS_TO_CHECK)
def test_scores_equal_length(metric_name:ValidDistanceKinds):
    metric = DistanceMetric(metric_name)
    fill_values = [1, 2, 3.5]
    for fill_value in fill_values:

        hyp, ref = get_test_poses(20, 20, fill_value=fill_value)
        coordinates_per_point = hyp.body.data.shape[-1]
        point_count = np.prod(hyp.body.confidence.shape)
        
        if metric_name == 'euclidean':
            # euclidean difference per point: 
            # between (2,2,2) and (0,0,0) is 3.4641, 
            # aka sqrt((2-0)^2 +(2-0)^2 +(2-0^2)
            expected_difference_per_point = np.sqrt(fill_value*fill_value*coordinates_per_point)
            if fill_value == 3:
                expected_difference_per_point = 3.461
            expected_distance = expected_difference_per_point * point_count
            
            

        elif metric_name == 'manhattan':
            
            expected_difference_per_point = fill_value * coordinates_per_point 
            expected_distance = expected_difference_per_point * point_count

        score = metric.score(hyp, ref)
        point_count = np.prod(hyp.body.confidence.shape)
        assert np.isclose(score, expected_distance)        
        assert isinstance(score, float)  # Check if the score is a float

@pytest.mark.parametrize('kind', DISTANCE_KINDS_TO_CHECK)
def test_all_distance_metrics_and_kinds(DistanceMetricToTest, kind):
    metric = DistanceMetricToTest(kind)
    assert isinstance(metric, DistanceMetric)


def get_all_subclasses(base_class):
    """Recursively discover all subclasses of a given base class."""
    subclasses = set(base_class.__subclasses__())
    for subclass in base_class.__subclasses__():
        subclasses.update(get_all_subclasses(subclass))
    return subclasses

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