import unittest
from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.metrics.pose_processors import get_standard_pose_processors
from pose_evaluation.metrics.test_distance_metric import get_poses
from pose_evaluation.metrics.dtw_metric import DTWAggregatedPowerDistanceMeasure


class TestDTWMetricL1(unittest.TestCase):
    def setUp(self):
        distance_measure = DTWAggregatedPowerDistanceMeasure(order=1, aggregation_strategy="mean", default_distance=0.0)
        self.metric = DistanceMetric(
            name="DTWPowerDistance",
            distance_measure=distance_measure,
            pose_preprocessors=get_standard_pose_processors(
                trim_meaningless_frames=False,  # fake poses have no components, this crashes.
                normalize_poses=False,  # no shoulders, will crash
                remove_world_landmarks=False,  # there are none, will crash
                reduce_poses_to_common_components=False,  # removes all components, there are none in common
                remove_legs=False,  # there are none, it will crash
                zero_pad_shorter=False,  # defeats the point of dtw
                reduce_holistic_to_face_and_upper_body=False,
            ),
        )
        self.poses_supplier = get_poses

    def test_score_equal_length(self):
        hypothesis, reference = self.poses_supplier(3, 3)
        expected_distance = 9

        score = self.metric.score(hypothesis, reference)
        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, expected_distance)

    def test_score_unequal_length(self):
        hypothesis, reference = self.poses_supplier(2, 3, people1=1, people2=1)

        # fastdtw creates mappings such that they are effectively both length 3.
        # Each of the point distances are (0,0,0) to (1,1,1), so 1+1+1=3
        # 3 pointwise distances, so 3*3 = 9
        # Those are then aggregated by mean aggregation across keypoints, but they're all 9.
        # so mean of all 9s is 9
        expected_distance = 9

        score = self.metric.score(hypothesis, reference)
        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, expected_distance)
