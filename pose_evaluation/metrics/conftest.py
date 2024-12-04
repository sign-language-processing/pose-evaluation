import shutil
from pathlib import Path
from typing import Callable, Union
import torch
import numpy as np
import pytest


@pytest.fixture(scope="session", autouse=True)
def clean_test_artifacts():
    """Fixture to clean up test artifacts before each test session."""
    test_artifacts_dir = Path(__file__).parent / "tests"  # Using Path
    if test_artifacts_dir.exists():
        shutil.rmtree(test_artifacts_dir)  # shutil.rmtree still works with Path
    test_artifacts_dir.mkdir(parents=True, exist_ok=True)  # Using Path.mkdir
    yield  # This allows the test session to run
    # (Optional) You can add cleanup logic here to run after the session if needed


@pytest.fixture(name="distance_range_checker")
def fixture_distance_range_checker() -> Callable[[Union[torch.Tensor, np.ndarray], float, float], None]:
    def _check_range(distances: Union[torch.Tensor, np.ndarray], min_val: float = 0, max_val: float = 2) -> None:
        max_distance = distances.max().item()
        min_distance = distances.min().item()

        # Use np.isclose for comparisons with tolerance
        assert (
            np.isclose(min_distance, min_val, atol=1e-6) or min_val <= min_distance <= max_val
        ), f"Minimum distance ({min_distance}) is outside the expected range [{min_val}, {max_val}]"
        assert (
            np.isclose(max_distance, max_val, atol=1e-6) or min_val <= max_distance <= max_val
        ), f"Maximum distance ({max_distance}) is outside the expected range [{min_val}, {max_val}]"

    return _check_range
