import pytest
import numpy as np
import torch
from pose_evaluation.metrics.embedding_distance_metric import EmbeddingDistanceMetric
from pose_evaluation.metrics.conftest import distance_range_checker
import matplotlib.pyplot as plt
import logging
from typing import List
from pathlib import Path

# TODO: many fixes. Including the fact that we test cosine but not Euclidean,


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration for PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cosine_metric():
    """Fixture to create an EmbeddingDistanceMetric instance."""
    return EmbeddingDistanceMetric(kind="cosine")


@pytest.fixture
def embeddings() -> List[torch.Tensor]:
    """Fixture to create dummy embeddings for testing."""
    return [random_tensor(768) for _ in range(5)]


def save_and_plot_distances(distances, matrix_name, num_points, dim):
    """Helper function to save distance matrix and plot distances."""
    test_artifacts_dir = Path(__file__).parent / "tests"
    output_path = test_artifacts_dir / f"distance_matrix_{matrix_name}_{num_points}_{dim}D.csv"
    np.savetxt(output_path, distances.numpy(), delimiter=",", fmt="%.4f")
    print(f"Distance matrix saved to {output_path}")

    # Generate plot
    plt.figure(figsize=(10, 6))
    for i, row in enumerate(distances.numpy()):
        plt.plot(row, label=f"Point {i}")
    plt.title(f"Distance Matrix Rows ({matrix_name})")
    plt.xlabel("Point Index")
    plt.ylabel("Distance")
    plt.legend()
    plot_path = output_path.with_suffix(".png")
    plt.savefig(plot_path)
    print(f"Distances plot saved to {plot_path}")
    plt.close()


def random_tensor(size: int) -> torch.Tensor:
    """Generate a random tensor on the appropriate device."""
    return torch.rand(size, dtype=torch.float32, device=DEVICE)


def generate_unit_circle_points(num_points: int, dim: int = 2) -> torch.Tensor:
    angles = torch.linspace(0, 2 * np.pi, num_points + 1)[:-1]
    x_coords = torch.cos(angles)
    y_coords = torch.sin(angles)
    points = torch.stack([x_coords, y_coords], dim=1)
    if dim > 2:
        padding = torch.zeros((num_points, dim - 2))
        points = torch.cat([points, padding], dim=1)
    return points


def generate_orthogonal_rows_with_repeats(num_rows: int, dim: int) -> torch.Tensor:
    orthogonal_rows = torch.empty(0, dim)
    for _ in range(min(num_rows, dim)):
        random_vector = torch.randn(1, dim)
        if orthogonal_rows.shape[0] > 0:
            random_vector -= (
                torch.matmul(random_vector, orthogonal_rows.T)
                @ orthogonal_rows
                / torch.norm(orthogonal_rows, dim=1, keepdim=True) ** 2
            )
        orthogonal_rows = torch.cat([orthogonal_rows, random_vector / torch.norm(random_vector)])
    if num_rows > dim:
        orthogonal_rows = orthogonal_rows.repeat(num_rows // dim + 1, 1)[:num_rows]
    return orthogonal_rows


def generate_orthogonal_rows_in_pairs(num_pairs: int, dim: int) -> torch.Tensor:
    """
    Generates a tensor with orthogonal rows in pairs.
    The first row of each pair is orthogonal to the second row of the same pair.

    Args:
        num_pairs: The number of orthogonal pairs to generate.
        dim: The dimensionality of the vectors.

    Returns:
        A PyTorch tensor with orthogonal rows in pairs.
    """

    orthogonal_rows = torch.empty(0, dim)
    for _ in range(num_pairs):
        # Generate the first vector of the pair
        first_vector = torch.randn(1, dim)
        first_vector = first_vector / torch.norm(first_vector)  # Normalize

        # Generate the second vector orthogonal to the first
        second_vector = torch.randn(1, dim)
        second_vector = second_vector - (second_vector @ first_vector.T) * first_vector
        second_vector = second_vector / torch.norm(second_vector)  # Normalize

        # Concatenate the pair to the result
        orthogonal_rows = torch.cat([orthogonal_rows, first_vector, second_vector], dim=0)

    return orthogonal_rows


def generate_ones_tensor(rows: int, dims: int) -> torch.Tensor:
    """Generates a tensor with all elements equal to 1.0 (float)."""
    return torch.ones(rows, dims, dtype=torch.float32)


def generate_identity_matrix_rows(rows, cols):
    """
    Returns an identity matrix with the specified number of rows and columns.
    """
    identity = torch.eye(max(rows, cols))
    return identity[:rows, :cols]


def create_increasing_rows_tensor(num_rows: int, num_cols: int) -> torch.Tensor:
    """
    Creates a tensor where every row has identical values all the way across,
    but increasing row by row.

    Args:
        num_rows: The number of rows in the tensor.
        num_cols: The number of columns in the tensor.

    Returns:
        A PyTorch tensor with the specified properties.
    """

    tensor = torch.arange(1.0, num_rows + 1).unsqueeze(1).repeat(1, num_cols)
    return tensor


def test_score_symmetric(cosine_metric: EmbeddingDistanceMetric) -> None:
    """Test that the metric is symmetric for cosine distance."""
    emb1 = random_tensor(768)
    emb2 = random_tensor(768)

    score1 = cosine_metric.score(emb1, emb2)
    score2 = cosine_metric.score(emb2, emb1)

    logger.info(f"Score 1: {score1}, Score 2: {score2}")
    assert pytest.approx(score1) == score2, "Score should be symmetric."


def test_score_with_path(metric: EmbeddingDistanceMetric, tmp_path: Path) -> None:
    """Test that score works with embeddings loaded from file paths."""
    emb1 = random_tensor(768).cpu().numpy()  # Save as NumPy for file storage
    emb2 = random_tensor(768).cpu().numpy()

    # Save embeddings to temporary files
    file1 = tmp_path / "emb1.npy"
    file2 = tmp_path / "emb2.npy"
    np.save(file1, emb1)
    np.save(file2, emb2)

    # Load files as PyTorch tensors
    emb1_loaded = torch.tensor(np.load(file1), dtype=torch.float32, device=DEVICE)
    emb2_loaded = torch.tensor(np.load(file2), dtype=torch.float32, device=DEVICE)

    score = cosine_metric.score(emb1_loaded, emb2_loaded)
    expected_score = cosine_metric.score(torch.tensor(emb1, device=DEVICE), torch.tensor(emb2, device=DEVICE))

    logger.info(f"Score from file: {score}, Direct score: {expected_score}")
    assert pytest.approx(score) == expected_score, "Score with paths should match direct computation."


def test_score_all_against_self(
    metric: EmbeddingDistanceMetric, embeddings: List[torch.Tensor], distance_range_checker
) -> None:
    """Test the score_all function."""
    scores = cosine_metric.score_all(embeddings, embeddings)
    assert scores.shape == (len(embeddings), len(embeddings)), "Output shape mismatch for score_all."
    assert torch.allclose(
        torch.diagonal(scores), torch.zeros(len(embeddings), device=DEVICE), atol=1e-6
    ), "Self-comparison scores should be zero for cosine distance."
    distance_range_checker(scores, min_val=0, max_val=2)
    logger.info(f"Score matrix shape: {scores.shape}, Diagonal values: {torch.diagonal(scores)}")


def test_score_all_with_different_sizes(cosine_metric, distance_range_checker):
    """Test score_all with different sizes for hypotheses and references."""
    hyps = [np.random.rand(768) for _ in range(3)]
    refs = [np.random.rand(768) for _ in range(5)]

    scores = cosine_metric.score_all(hyps, refs)
    assert scores.shape == (
        len(hyps),
        len(refs),
    ), f"Output shape mismatch ({scores.shape}) vs {(len(hyps), len(refs))} for score_all with different sizes. "
    distance_range_checker(scores, min_val=0, max_val=2)


# def test_score_all_with_empty_inputs(metric):
#     """Test score_all with empty inputs."""
#     scores = metric.score_all([], [])
#     assert scores.shape == (0,), f"Score_all should return an empty array for empty inputs. Output: {scores.shape}"


def test_invalid_input(cosine_metric: EmbeddingDistanceMetric) -> None:
    """Test the metric with invalid inputs."""
    emb1 = random_tensor(768)
    invalid_inputs = ["invalid_input", None, -1, 1]

    for invalid_input in invalid_inputs:
        with pytest.raises((TypeError, AttributeError)):
            cosine_metric.score(emb1, invalid_input)

    logger.info("Invalid input test passed.")


def test_score_tensor_input(cosine_metric):
    """Test score function with torch.Tensor inputs."""
    emb1 = torch.rand(768)
    emb2 = torch.rand(768)

    score = cosine_metric.score(emb1, emb2)
    assert isinstance(score, float), "Output should be a float."


def test_score_ndarray_input(cosine_metric):
    """Test score function with np.ndarray inputs."""
    emb1 = np.random.rand(768)
    emb2 = np.random.rand(768)

    score = cosine_metric.score(emb1, emb2)
    assert isinstance(score, float), "Output should be a float."


def test_score_all_tensor_input(cosine_metric):
    """Test score_all function with torch.Tensor inputs."""
    hyps = [torch.rand(768) for _ in range(5)]
    refs = [torch.rand(768) for _ in range(5)]

    scores = cosine_metric.score_all(hyps, refs)
    assert len(scores) == len(hyps), f"Output row count mismatch for torch.Tensor input. Shape:{scores.shape}"
    assert len(scores[0]) == len(refs), f"Output column count mismatch for torch.Tensor input. Shape:{scores.shape}"


def test_device_handling(cosine_metric):
    """Test device handling for the metric."""
    assert cosine_metric.device.type in ["cuda", "cpu"], "Device should be either 'cuda' or 'cpu'."
    if torch.cuda.is_available():
        assert cosine_metric.device.type == "cuda", "Should use 'cuda' when available."
    else:
        assert cosine_metric.device.type == "cpu", "Should use 'cpu' when CUDA is unavailable."


def test_mixed_input(cosine_metric):
    """Test score function with mixed input types."""
    emb1 = np.random.rand(768)
    emb2 = torch.rand(768)

    score = cosine_metric.score(emb1, emb2)
    assert isinstance(score, float), "Output should be a float."


@pytest.mark.parametrize("num_points, dim", [(16, 2)])
def test_unit_circle_points(cosine_metric, num_points, dim):
    embeddings = generate_unit_circle_points(num_points, dim)
    distances = cosine_metric.score_all(embeddings, embeddings)
    save_and_plot_distances(distances=distances, matrix_name="Unit Circle", num_points=num_points, dim=dim)


@pytest.mark.parametrize("num_points, dim", [(20, 2)])
def test_orthogonal_rows_with_repeats_2d(cosine_metric, num_points, dim):
    embeddings = generate_orthogonal_rows_with_repeats(num_points, dim)
    distances = cosine_metric.score_all(embeddings, embeddings)
    save_and_plot_distances(
        distances=distances, matrix_name="Orthogonal Rows (with repeats)", num_points=num_points, dim=dim
    )

    # Create expected pattern directly within the test function
    expected_pattern = torch.zeros(num_points, num_points, dtype=torch.float32)
    for i in range(num_points):
        for j in range(num_points):
            if (i + j) % 2 != 0:
                expected_pattern[i, j] = 1

    # We expect 0 1 0  across and down
    assert torch.allclose(
        distances, expected_pattern, atol=1e-6
    ), "Output does not match the expected alternating pattern"


@pytest.mark.parametrize("num_points, dim", [(20, 2)])
def test_orthogonal_rows_in_pairs(cosine_metric, num_points, dim, distance_range_checker):
    embeddings = generate_orthogonal_rows_in_pairs(num_points, dim)
    distances = cosine_metric.score_all(embeddings, embeddings)
    save_and_plot_distances(distances, "orthogonal_rows_in_pairs", num_points, dim)
    distance_range_checker(distances, min_val=0, max_val=2)  # Check distance range


@pytest.mark.parametrize("num_points, dim", [(10, 5)])
def test_ones_tensor(cosine_metric, num_points, dim, distance_range_checker):
    embeddings = generate_ones_tensor(num_points, dim)
    distances = cosine_metric.score_all(embeddings, embeddings)
    save_and_plot_distances(distances, "ones_tensor", num_points, dim)
    distance_range_checker(distances, min_val=0, max_val=0)  # Expect all distances to be 0


@pytest.mark.parametrize("num_points, dim", [(15, 15)])  # dim should be equal to num_points for identity matrix
def test_identity_matrix_rows(cosine_metric, num_points, dim, distance_range_checker):
    embeddings = generate_identity_matrix_rows(num_points, dim)
    distances = cosine_metric.score_all(embeddings, embeddings)
    save_and_plot_distances(distances, "identity_matrix_rows", num_points, dim)
    distance_range_checker(distances, min_val=0, max_val=2)  # Check distance range


# def test_progress_bar(cosine_metric):
#     """Test score_all with progress_bar argument."""
#     hyps = [np.random.rand(768) for _ in range(5)]
#     refs = [np.random.rand(768) for _ in range(5)]

#     # Disable progress bar
#     scores = cosine_metric.score_all(hyps, refs, progress_bar=False)
#     assert len(scores) == len(hyps), "Output row count mismatch with progress_bar=False."
#     assert len(scores[0]) == len(refs), "Output column count mismatch with progress_bar=False."
