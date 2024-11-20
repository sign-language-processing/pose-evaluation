import pytest
import numpy as np
from pose_format import Pose
from pose_evaluation.metrics.signclip_distance_metric import SignCLIPEmbeddingDistanceMetric

# Mock a simple Pose object for compatibility (if not already available)
class MockPose:
    def __init__(self, data):
        self.data = data

@pytest.fixture
def metric():
    """Fixture to create a SignCLIPEmbeddingDistanceMetric instance."""
    return SignCLIPEmbeddingDistanceMetric(kind="cosine")

@pytest.fixture
def embeddings():
    """Fixture to create dummy embeddings for testing."""
    # Generate 5 random 768-dimensional embeddings
    return [np.random.rand(768) for _ in range(5)]

def test_score_symmetric(metric):
    """Test that the metric is symmetric for cosine distance."""
    emb1 = np.random.rand(768)
    emb2 = np.random.rand(768)

    score1 = metric.score(emb1, emb2)
    score2 = metric.score(emb2, emb1)

    assert pytest.approx(score1) == score2, "Score should be symmetric."

def test_score_with_path(metric, tmp_path):
    """Test that score works with embeddings loaded from paths."""
    emb1 = np.random.rand(768)
    emb2 = np.random.rand(768)

    # Save embeddings to temporary files
    file1 = tmp_path / "emb1.npy"
    file2 = tmp_path / "emb2.npy"
    np.save(file1, emb1)
    np.save(file2, emb2)

    score = metric.score(file1, file2)
    expected_score = metric.score(emb1, emb2)

    assert pytest.approx(score) == expected_score, "Score with paths should match direct computation."

def test_score_all(metric, embeddings):
    """Test the score_all function."""
    scores = metric.score_all(embeddings, embeddings)
    assert scores.shape == (len(embeddings), len(embeddings)), "Output shape mismatch for score_all."
    assert np.allclose(scores.diagonal(), 0), "Self-comparison scores should be zero for cosine distance."

def test_score_all_with_different_sizes(metric):
    """Test score_all with different sizes for hypotheses and references."""
    hyps = [np.random.rand(768) for _ in range(3)]
    refs = [np.random.rand(768) for _ in range(5)]

    scores = metric.score_all(hyps, refs)
    assert scores.shape == (len(hyps), len(refs)), "Output shape mismatch for score_all with different sizes."

def test_score_all_edge_case(metric):
    """Test score_all with empty inputs."""
    scores = metric.score_all([], [])
    assert scores.size == 0, "Score_all should return an empty array for empty inputs."
