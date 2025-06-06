from unittest.mock import MagicMock, patch

import pytest
from pose_format import Pose

from pose_evaluation.metrics.base import Score
from pose_evaluation.metrics.base_pose_metric import PoseMetric
from pose_evaluation.metrics.pose_processors import PoseProcessor

# --- Dummy setup ---


def make_fake_pose():
    """Returns a mocked Pose object."""
    return MagicMock(spec=Pose)


class DummyPoseMetric(PoseMetric):
    def _pose_score(self, processed_hypothesis: Pose, processed_reference: Pose):
        return 1.23  # constant for testing


@pytest.fixture
def dummy_metric():
    return DummyPoseMetric(name="Dummy", higher_is_better=True, pose_preprocessors=[])


# --- Tests ---


def test_score_calls_internal_pose_score(dummy_metric):
    h = make_fake_pose()
    r = make_fake_pose()

    dummy_metric.process_poses = MagicMock(return_value=[h, r])
    result = dummy_metric.score(h, r)

    assert result == 1.23


def test_score_all_returns_matrix(dummy_metric):
    h_list = [make_fake_pose() for _ in range(2)]
    r_list = [make_fake_pose() for _ in range(3)]

    matrix = dummy_metric.score_all(h_list, r_list)

    assert len(matrix) == 2
    assert all(len(row) == 3 for row in matrix)
    assert all(score == 1.23 for row in matrix for score in row)


def test_score_with_signature_returns_score(dummy_metric):
    h = make_fake_pose()
    r = make_fake_pose()

    score_obj = dummy_metric.score_with_signature(h, r, short=True)

    assert isinstance(score_obj, Score)
    assert score_obj.score == 1.23
    assert "Dummy" in score_obj.name


def test_score_all_with_signature(dummy_metric):
    h_list = [make_fake_pose()]
    r_list = [make_fake_pose(), make_fake_pose()]

    results = dummy_metric.score_all_with_signature(h_list, r_list)

    assert len(results) == 1
    assert all(isinstance(score, Score) for score in results[0])


def test_add_preprocessor_appends_to_list(dummy_metric):
    proc = MagicMock(spec=PoseProcessor)
    dummy_metric.add_preprocessor(proc)

    assert proc in dummy_metric.pose_preprocessors


def test_process_poses_applies_all_preprocessors(dummy_metric):
    poses = [make_fake_pose(), make_fake_pose()]

    proc1 = MagicMock(spec=PoseProcessor)
    proc2 = MagicMock(spec=PoseProcessor)

    # Each processor returns the input poses unchanged
    proc1.process_poses.side_effect = lambda poses, progress=False: poses
    proc2.process_poses.side_effect = lambda poses, progress=False: poses

    dummy_metric.pose_preprocessors = [proc1, proc2]
    result = dummy_metric.process_poses(poses)

    assert result == poses
    proc1.process_poses.assert_called_once()
    proc2.process_poses.assert_called_once()


def test_pose_metric_pose_score_not_implemented():
    class IncompletePoseMetric(PoseMetric):
        def _pose_score(self, processed_hypothesis: Pose, processed_reference: Pose):
            # Intentionally call super() to trigger the abstract method error
            return super()._pose_score(processed_hypothesis, processed_reference)

    metric = IncompletePoseMetric(pose_preprocessors=[])
    h = make_fake_pose()
    r = make_fake_pose()

    with pytest.raises(NotImplementedError, match="Subclasses must implement _pose_score"):
        metric.score(h, r)


def test_constructor_defaults_to_standard_pose_processors():
    with patch("pose_evaluation.metrics.base_pose_metric.get_standard_pose_processors") as mock_get:
        mock_get.return_value = ["fake_processor"]

        metric = DummyPoseMetric(pose_preprocessors=None)

        assert metric.pose_preprocessors == ["fake_processor"]
        mock_get.assert_called_once()
