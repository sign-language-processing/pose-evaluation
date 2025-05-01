from abc import ABC, abstractmethod
from typing import Literal, Dict, Any

import numpy.ma as ma  # pylint: disable=consider-using-from-import

from pose_evaluation.metrics.base import Signature

AggregationStrategy = Literal["max", "min", "mean", "sum"]


class DistanceMeasureSignature(Signature):
    """Signature for distance measure metrics."""

    def __init__(self, name: str, args: Dict[str, Any]) -> None:
        super().__init__(name=name, args=args)
        self.update_abbr("distance", "dist")
        self.update_abbr("power", "pow")


class DistanceMeasure(ABC):
    """Abstract base class for distance measures."""

    _SIGNATURE_TYPE = DistanceMeasureSignature

    def __init__(self, name: str, default_distance=0.0) -> None:
        self.name = name
        self.default_distance = default_distance

    @abstractmethod
    def get_distance(self, hyp_data: ma.MaskedArray, ref_data: ma.MaskedArray) -> float:
        """
        Compute the distance between hypothesis and reference data.

        This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def _get_keypoint_trajectories(self, hyp_data: ma.MaskedArray, ref_data: ma.MaskedArray):
        # frames, persons, keypoint
        for keypoint_idx in range(hyp_data.shape[2]):
            hyp_trajectory, ref_trajectory = (
                hyp_data[:, 0, keypoint_idx, :],
                ref_data[:, 0, keypoint_idx, :],
            )
            yield hyp_trajectory, ref_trajectory

    def __call__(self, hyp_data: ma.MaskedArray, ref_data: ma.MaskedArray) -> float:
        return self.get_distance(hyp_data, ref_data)

    def _calculate_pointwise_distances(self, hyp_data: ma.MaskedArray, ref_data: ma.MaskedArray) -> ma.MaskedArray:
        raise NotImplementedError

    def get_signature(self) -> Signature:
        """Return the signature of the distance measure."""
        return self._SIGNATURE_TYPE(self.name, self.__dict__)

    def set_default_distance(self, default_distance:float):
        self.default_distance = default_distance


class PowerDistanceSignature(DistanceMeasureSignature):
    """Signature for power distance measures."""

    def __init__(self, name: str, args: Dict[str, Any]) -> None:
        super().__init__(name=name, args=args)
        self.update_signature_and_abbr("order", "ord", args)
        self.update_signature_and_abbr("default_distance", "dflt", args)
        self.update_signature_and_abbr("aggregation_strategy", "agg", args)


class AggregatedDistanceMeasure(DistanceMeasure):
    def __init__(
        self,
        name: str,
        default_distance: float = 0.0,
        aggregation_strategy: AggregationStrategy = "mean",
    ) -> None:
        super().__init__(name, default_distance=default_distance)
        self.aggregation_strategy = aggregation_strategy

    def _aggregate(self, distances: ma.MaskedArray) -> float:
        """
        Aggregate computed distances using the specified strategy.

        :param distances: A masked array of computed distances.
        :return: A single aggregated distance value.
        """
        aggregation_funcs = {
            "mean": distances.mean,
            "max": distances.max,
            "min": distances.min,
            "sum": distances.sum,
        }
        if self.aggregation_strategy in aggregation_funcs:
            return aggregation_funcs[self.aggregation_strategy]()

        raise NotImplementedError(f"Aggregation Strategy {self.aggregation_strategy} not implemented")

    def get_distance(self, hyp_data: ma.MaskedArray, ref_data: ma.MaskedArray) -> float:
        """Compute and aggregate the distance between hypothesis and reference data."""
        calculated = self._calculate_pointwise_distances(hyp_data, ref_data)
        return self._aggregate(calculated)


class AggregatedPowerDistance(AggregatedDistanceMeasure):
    """Aggregated power distance metric using a specified aggregation strategy."""

    _SIGNATURE_TYPE = PowerDistanceSignature

    def __init__(
        self,
        name="AggregatedPowerDistance",
        order: int = 2,
        default_distance: float = 0.0,
        aggregation_strategy: AggregationStrategy = "mean",
    ) -> None:
        """
        Initialize the aggregated power distance metric.

        :param order: The exponent to which differences are raised.
        :param default_distance: The value to fill in for masked entries.
        :param aggregation_strategy: Strategy to aggregate computed distances.
        """
        super().__init__(
            name=name,
            aggregation_strategy=aggregation_strategy,
            default_distance=default_distance,
        )
        self.power = float(order)

    def _calculate_pointwise_distances(self, hyp_data: ma.MaskedArray, ref_data: ma.MaskedArray) -> ma.MaskedArray:
        return masked_array_power_distance(
            hyp_data=hyp_data, ref_data=ref_data, power=self.power, default_distance=self.default_distance
        )


def masked_array_power_distance(
    hyp_data: ma.MaskedArray, ref_data: ma.MaskedArray, power: float, default_distance: float
) -> ma.MaskedArray:
    """
    Compute element-wise distances between hypothesis and reference data.

    Steps:
      1. Compute the absolute differences.
      2. Raise the differences to the specified power.
      3. Sum the powered differences along the last axis.
      4. Extract the root corresponding to the power.
      5. Fill masked values with the default distance.

    :param hyp_data: Hypothesis data as a masked array.
    :param ref_data: Reference data as a masked array.
    :return: A masked array of computed distances.
    """
    diffs = ma.abs(hyp_data - ref_data)
    raised_to_power = ma.power(diffs, power)
    summed_results = ma.sum(raised_to_power, axis=-1, keepdims=True)
    roots = ma.power(summed_results, 1 / power)
    return ma.filled(roots, default_distance)
