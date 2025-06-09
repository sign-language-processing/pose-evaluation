from unittest.mock import MagicMock

import pytest

from pose_evaluation.evaluation.create_metrics import (
    construct_metric,
    extract_metric_name_dist,
    extract_signature_distance,
    get_metrics,
)

# ---- Tests for extract_signature_distance ----


@pytest.mark.parametrize(
    "signature,expected",
    [
        ("abc default_distance:0.75 xyz", 0.75),
        ("default_distance:10.5", 10.5),
        ("nothing here", None),
    ],
)
def test_extract_signature_distance(signature, expected):
    result = extract_signature_distance(signature)
    assert result == expected


# ---- Tests for extract_metric_name_dist ----


@pytest.mark.parametrize(
    "name,expected",
    [
        ("metric_defaultdist0.25", 0.25),
        ("defaultdist10", 10.0),
        ("no match here", None),
    ],
)
def test_extract_metric_name_dist(name, expected):
    result = extract_metric_name_dist(name)
    assert result == expected


# ---- Minimal test for construct_metric ----


def test_construct_metric_basic():
    # Mock distance_measure with required interface
    mock_measure = MagicMock()
    mock_measure.name = "MockMeasure"
    mock_measure.get_signature.return_value.format.return_value = "default_distance:0.0"

    metric = construct_metric(distance_measure=mock_measure)

    assert isinstance(metric.name, str)
    assert "defaultdist0.0" in metric.name
    assert metric.distance_measure == mock_measure
    assert hasattr(metric, "pose_preprocessors")
    assert isinstance(metric.pose_preprocessors, list)
    # Ensure some common processors are present
    assert any("trim" in proc.__class__.__name__.lower() for proc in metric.pose_preprocessors)


@pytest.mark.parametrize("include_return4", [True, False])
@pytest.mark.parametrize("include_masked", [False, True])
def test_get_metrics_uniqueness_and_consistency(include_return4, include_masked):
    """
    Runs get_metrics() with various config flags and checks:

    - All DistanceMetric names are unique
    - All DistanceMetric signatures are unique
    - No two metrics share the same DistanceMeasure object
    - defaultdist in name matches default_distance in signature
    """
    metrics = get_metrics(include_return4=include_return4, include_masked=include_masked)

    metric_names = [m.name for m in metrics]
    metric_sigs = [m.get_signature().format() for m in metrics]
    distance_measure_ids = [id(m.distance_measure) for m in metrics]

    # 1. All names are unique
    assert len(metric_names) == len(
        set(metric_names)
    ), f"Duplicate metric names found for include_return4={include_return4}, include_masked={include_masked}"

    # 2. All signatures are unique
    assert len(metric_sigs) == len(
        set(metric_sigs)
    ), f"Duplicate metric signatures found for include_return4={include_return4}, include_masked={include_masked}"

    # 3. No reused DistanceMeasure objects
    assert len(distance_measure_ids) == len(set(distance_measure_ids)), (
        "DistanceMeasures are shared across metrics,",
        f" include_return4={include_return4}, include_masked={include_masked}",
    )

    # 4. Signature default_distance matches name defaultdist
    for name, sig in zip(metric_names, metric_sigs, strict=False):
        dist_from_name = extract_metric_name_dist(name)
        dist_from_sig = extract_signature_distance(sig)
        assert dist_from_name == dist_from_sig, (
            f"Default distance mismatch for include_return4={include_return4}, include_masked={include_masked}:\n"
            f"  From name: {name} → {dist_from_name}\n"
            f"  From sig:  {sig} → {dist_from_sig}"
        )
