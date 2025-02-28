import unittest
from typing import Optional

import numpy as np
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody

from pose_evaluation.metrics.distance_measure import AggregatedPowerDistance
from pose_evaluation.metrics.distance_metric import DistanceMetric


def get_poses(
    length1: int,
    length2: int,
    conf1: Optional[float] = None,
    conf2: Optional[float] = None,
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
    data_tensor = np.full((length1, 3, 4, 3), fill_value=2)
    zeros_tensor = np.zeros((length2, 3, 4, 3))
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
        )

    def test_score_equal_length(self):
        hypothesis, reference = get_poses(2, 2)
        expected_distance = 6  # Sum of absolute differences: 2 + 2 + 2

        score = self.metric.score(hypothesis, reference)
        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, expected_distance)


class TestDistanceMetricL2(unittest.TestCase):
    """Tests for the L2 (Euclidean) distance metric with default distance substitution."""

    def setUp(self):
        self.default_distance = 17
        self.metric = DistanceMetric(
            "l2_metric",
            distance_measure=AggregatedPowerDistance(
                order=2, default_distance=self.default_distance
            ),
        )

    def _check_against_expected(self, hypothesis, reference, expected):
        score = self.metric.score(hypothesis, reference)
        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, expected)

    def test_score_equal_length(self):
        hypothesis, reference = get_poses(2, 2)
        expected_distance = np.sqrt(2**2 + 2**2 + 2**2)  # sqrt(12)
        self._check_against_expected(hypothesis, reference, expected=expected_distance)

    def test_score_equal_length_one_masked(self):
        hypothesis, reference = get_poses(2, 2, conf1=0.0)
        self._check_against_expected(
            hypothesis, reference, expected=self.default_distance
        )

    # TODO: Add tests for other aggregation strategies and power values


if __name__ == "__main__":
    unittest.main()
