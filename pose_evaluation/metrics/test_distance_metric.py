import math
import unittest

import numpy as np
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody

from pose_evaluation.metrics.distance_measure import AggregatedPowerDistance
from pose_evaluation.metrics.distance_metric import DistanceMetric


def get_poses(length1: int, length2: int, conf1= None, conf2=None):
    data_tensor = np.full((length1, 3, 4, 3), fill_value=2)
    zeros_tensor = np.zeros((length2, 3, 4, 3))
    data_confidence = np.ones(data_tensor.shape[:-1])
    zeros_confidence = np.ones(zeros_tensor.shape[:-1])
    
    if conf1 is not None:
        data_confidence = conf1 * data_confidence
    if conf2 is not None: 
        zeros_confidence = conf2 * zeros_confidence

    hypothesis = Pose(header=None, body=NumPyPoseBody(fps=1, data=data_tensor, confidence=data_confidence))
    reference = Pose(header=None, body=NumPyPoseBody(fps=1, data=zeros_tensor, confidence=zeros_confidence))
    return hypothesis, reference

# class TestDistanceMetricGeneric(unittest.TestCase):
#     def setUp(self):
#         self.metric = DistanceMetric("l2")

#     def test_scores_are_symmetric(self):
#         hypothesis, reference = get_poses(2, 2)

#         score1 = self.metric.score(hypothesis, reference)
#         # pylint: disable=arguments-out-of-order
#         score2 = self.metric.score(reference, hypothesis)
#         self.assertAlmostEqual(score1, score2)

#     def test_score_different_length(self):
#         hypothesis, reference = get_poses(3, 2)

#         difference = 6 * np.prod(hypothesis.body.confidence.shape)

#         score = self.metric.score(hypothesis, reference)
#         self.assertIsInstance(score, float)
#         self.assertAlmostEqual(score, difference)

class TestDistanceMetricMeanL1(unittest.TestCase):
    def setUp(self):
        self.metric = DistanceMetric("mean_l1_metric", distance_measure=AggregatedPowerDistance(1, 0))

    def test_score_equal_length(self):
        hypothesis, reference = get_poses(2, 2)

        expected_mean = 6 # absolute distance between (2, 2, 2) and (0, 0, 0)
        

        score = self.metric.score(hypothesis, reference)
        self.assertIsInstance(score, float)  # Check if the score is a float
        self.assertAlmostEqual(score, expected_mean)

class TestDistanceMetricL2(unittest.TestCase):
    def setUp(self):
        self.default_distance = 17
        self.metric = DistanceMetric("l2_metric", distance_measure=AggregatedPowerDistance(2, self.default_distance))

    def _check_against_expected(self, hypothesis, reference, expected):
        score = self.metric.score(hypothesis, reference)
        self.assertIsInstance(score, float)  # Check if the score is a float
        self.assertAlmostEqual(score, expected)

    def test_score_equal_length(self):
        hypothesis, reference = get_poses(2, 2)

        expected_mean = np.sqrt(2**2+2**2+2**2) # all pairs are (2,2,2) and (0,0,0), so the mean is the same: sqrt(12)
        self._check_against_expected(hypothesis, reference, expected=expected_mean)

    def test_score_equal_length_one_masked(self):
        hypothesis, reference = get_poses(2, 2, conf1=0.0)
        self._check_against_expected(hypothesis, reference, expected=self.default_distance)

    # TODO: mean, max, sum, min, other powers

if __name__ == '__main__':
    unittest.main()
