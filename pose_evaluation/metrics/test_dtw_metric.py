import itertools
import unittest

import numpy as np
import numpy.ma as ma
import pytest
from pose_format import Pose

from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.metrics.dtw_metric import DTWAggregatedPowerDistanceMeasure, DTWDTAIImplementationDistanceMeasure
from pose_evaluation.metrics.pose_processors import (
    FillMaskedOrInvalidValuesPoseProcessor,
    ReducePosesToCommonComponentsProcessor,
    get_standard_pose_processors,
)
from pose_evaluation.metrics.test_distance_metric import get_poses


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


def test_dtai_distance_with_masked_poses(real_mixed_shape_files: list[Pose]):
    default_distance = 10.0
    metric = DistanceMetric(
        name="testmetric_with_no_masked_preprocessing",
        distance_measure=DTWDTAIImplementationDistanceMeasure(
            name="dtaiDTWAggregatedDistanceMeasureFast",
            use_fast=True,
            default_distance=default_distance,
        ),
        pose_preprocessors=[],
    )
    with pytest.warns(
        RuntimeWarning, match=f"Invalid distance calculated, setting to default value {default_distance}"
    ):
        for hyp, ref in itertools.combinations(real_mixed_shape_files, 2):
            score = metric.score_with_signature(hyp, ref)
            assert score is not None
            assert score.score is not None
            assert not np.isinf(score.score)
            assert not np.isnan(score.score)
            assert score.score == default_distance

    metric = DistanceMetric(
        name="testmetric_with_masked_preprocessing",
        distance_measure=DTWDTAIImplementationDistanceMeasure(
            name="dtaiDTWAggregatedDistanceMeasureFast", use_fast=True
        ),
        pose_preprocessors=[FillMaskedOrInvalidValuesPoseProcessor(), ReducePosesToCommonComponentsProcessor()],
    )
    for hyp, ref in itertools.combinations(real_mixed_shape_files, 2):
        processed_poses = metric.process_poses([hyp, ref], progress=True)
        for pose in processed_poses:
            assert ma.count_masked(pose.body.data) == 0
        score = metric.score_with_signature(hyp, ref)
        assert score is not None
        assert score.score is not None
        assert not np.isinf(score.score)
        assert not np.isnan(score.score)
        assert score.score != default_distance
