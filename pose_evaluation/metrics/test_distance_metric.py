import math
import unittest

import numpy as np
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody

from pose_evaluation.metrics.distance_metric import DistanceMetric


def get_test_poses(length1: int, length2: int):
    data_tensor = np.full((length1, 3, 4, 3), fill_value=2)
    zeros_tensor = np.zeros((length2, 3, 4, 3))
    data_confidence = np.ones(data_tensor.shape[:-1])
    zeros_confidence = np.ones(zeros_tensor.shape[:-1])

    hypothesis = Pose(header=None, body=NumPyPoseBody(fps=1, data=data_tensor, confidence=data_confidence))
    reference = Pose(header=None, body=NumPyPoseBody(fps=1, data=zeros_tensor, confidence=zeros_confidence))
    return hypothesis, reference

class TestDistanceMetricGeneric(unittest.TestCase):
    def setUp(self):
        self.metric = DistanceMetric("euclidean")

    def test_scores_are_symmetric(self):
        hypothesis, reference = get_test_poses(2, 2)

        score1 = self.metric.score(hypothesis, reference)
        # pylint: disable=arguments-out-of-order
        score2 = self.metric.score(reference, hypothesis)
        self.assertAlmostEqual(score1, score2)

    def test_score_different_length(self):
        hypothesis, reference = get_test_poses(3, 2)

        difference = 6 * np.prod(hypothesis.body.confidence.shape)

        score = self.metric.score(hypothesis, reference)
        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, difference)

class TestDistanceMetricL1(unittest.TestCase):
    def setUp(self):
        self.metric = DistanceMetric("manhattan")

    def test_score_equal_length(self):
        hypothesis, reference = get_test_poses(2, 2)

        # calculate what the difference should be
        difference = 6 * np.prod(hypothesis.body.confidence.shape)

        score = self.metric.score(hypothesis, reference)
        self.assertIsInstance(score, float)  # Check if the score is a float
        self.assertAlmostEqual(score, difference)

class TestDistanceMetricL2(unittest.TestCase):
    def setUp(self):
        self.metric = DistanceMetric("euclidean")

    def test_score_equal_length(self):
        hypothesis, reference = get_test_poses(2, 2)

        # calculate what the difference should be
        difference = math.sqrt(12) * np.prod(hypothesis.body.confidence.shape)

        score = self.metric.score(hypothesis, reference)
        self.assertIsInstance(score, float)  # Check if the score is a float
        self.assertAlmostEqual(score, difference)


if __name__ == '__main__':
    unittest.main()
