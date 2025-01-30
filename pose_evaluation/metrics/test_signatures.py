from typing import List

import pytest

from pose_evaluation.metrics.base_pose_metric import PoseMetric
from pose_evaluation.metrics.distance_measure import PowerDistance
from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.metrics.dynamic_time_warping_metric import DTWMetric
from pose_evaluation.metrics.ham2pose_distances import Ham2PoseAPEDistance, Ham2PoseMSEDistance, Ham2PoseMaskedEuclideanDistance
from pose_evaluation.metrics.mje_metric import MeanJointErrorMetric
from pose_evaluation.metrics.pose_processors import (
    NormalizePosesProcessor,
    get_standard_pose_processors,
)

 

def test_pose_metric_signature_has_preprocessor_information():
    metric = PoseMetric("PoseMetric", pose_preprocessors=[NormalizePosesProcessor()])

    assert "pose_preprocessers" in metric.get_signature().format()
    assert "pre" in metric.get_signature().format(short=True)

    metric = PoseMetric("PoseMetric")
    assert "pose_preprocessers" not in metric.get_signature().format()
    assert "pre" not in metric.get_signature().format(short=True)

def test_pose_metric_signature_has_distance_measure_information(ham2pose_metrics_for_testing:List[DistanceMetric]):
    for metric in ham2pose_metrics_for_testing:
        assert "distance_measure:{" in metric.get_signature().format(short=False)
        assert "dist:{" in metric.get_signature().format(short=True)