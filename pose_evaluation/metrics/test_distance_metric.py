import unittest
from typing import Optional

import numpy as np
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody

from pose_evaluation.metrics.distance_measure import AggregatedPowerDistance
from pose_evaluation.metrics.distance_metric import DistanceMetric

from pose_evaluation.metrics.pose_processors import get_standard_pose_processors


def get_poses(
    length1: int,
    length2: int,
    conf1: Optional[float] = None,
    conf2: Optional[float] = None,
    people1: int = 3,
    people2: int = 3,
    keypoints1: int = 4,
    keypoints2: int = 4,
    coordinate_dimensions1: int = 3,
    coordinate_dimensions2: int = 3,
    fill_value1: float = 1.0,
    fill_value2: float = 0.0,
):
    """
    Utility function to generate hypothesis and reference Pose objects for testing.

    Args:
        length1 (int): Number of frames in the hypothesis pose.
        length2 (int): Number of frames in the reference pose.
        conf1 (float, optional): Confidence multiplier for the hypothesis.
        conf2 (float, optional): Confidence multiplier for the reference.

    Returns:
        tuple: A tuple containing (hypothesis, reference) Pose objects.
    """

    data_tensor = np.full([length1, people1, keypoints1, coordinate_dimensions1], fill_value=fill_value1)
    zeros_tensor = np.full((length2, people2, keypoints2, coordinate_dimensions2), fill_value=fill_value2)
    data_confidence = np.ones(data_tensor.shape[:-1])
    zeros_confidence = np.ones(zeros_tensor.shape[:-1])

    if conf1 is not None:
        data_confidence = conf1 * data_confidence
    if conf2 is not None:
        zeros_confidence = conf2 * zeros_confidence

    # TODO: add an actual header, something like
    # header = PoseHeader(1.0, PoseHeaderDimensions(10, 20, 5), [PoseHeaderComponent(...)], is_bbox=True)
    hypothesis = Pose(
        header=None,  # type: ignore
        body=NumPyPoseBody(fps=1, data=data_tensor, confidence=data_confidence),
    )
    reference = Pose(
        header=None,  # type: ignore
        body=NumPyPoseBody(fps=1, data=zeros_tensor, confidence=zeros_confidence),
    )
    return hypothesis, reference


class TestDistanceMetricMeanL1(unittest.TestCase):
    """Tests for the L1 (Manhattan) distance metric using mean aggregation."""

    def setUp(self):
        self.metric = DistanceMetric(
            "mean_l1_metric",
            distance_measure=AggregatedPowerDistance(order=1, default_distance=0),
            # preprocessors that won't crash
            pose_preprocessors=get_standard_pose_processors(
                normalize_poses=False,
                remove_world_landmarks=False,
                remove_legs=False,
                reduce_poses_to_common_components=False,
            ),
        )

    def test_score_equal_length(self):
        hypothesis, reference = get_poses(3, 3)

        # each pointwise distance is 3 (1+1+1)
        # Since they're all the same, the mean is also 3
        expected_distance = 3

        score = self.metric.score(hypothesis, reference)
        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, expected_distance)


class TestDistanceMetricL2(unittest.TestCase):
    """Tests for the L2 (Euclidean) distance metric with default distance substitution."""

    def setUp(self):
        self.default_distance = 17
        self.metric = DistanceMetric(
            "l2_metric",
            distance_measure=AggregatedPowerDistance(order=2, default_distance=self.default_distance),
            # preprocessors that won't crash
            pose_preprocessors=get_standard_pose_processors(
                normalize_poses=False,
                remove_world_landmarks=False,
                remove_legs=False,
                reduce_poses_to_common_components=False,
            ),
        )

    def _check_against_expected(self, hypothesis, reference, expected):
        score = self.metric.score(hypothesis, reference)
        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, expected)

    def test_score_equal_length(self):
        hypothesis, reference = get_poses(3, 3)
        # pointwise distance is sqrt((1-0)**2 + (1-0)**2 + (1-0)**2) = sqrt(3)
        # Every pointwise distance is the same, so the mean is also 3.
        expected_distance = np.sqrt(3)
        self._check_against_expected(hypothesis, reference, expected=expected_distance)

    def test_score_equal_length_one_masked(self):
        hypothesis, reference = get_poses(2, 2, conf1=0.0)
        self._check_against_expected(hypothesis, reference, expected=self.default_distance)

    # TODO: Add tests for other aggregation strategies and power values


if __name__ == "__main__":
    unittest.main()
