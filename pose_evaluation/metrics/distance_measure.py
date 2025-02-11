import numpy as np
import numpy.ma as ma
from pose_evaluation.metrics.base import Signature


class DistanceMeasureSignature(Signature):
    def __init__(self, name: str, args: dict):
        super().__init__(name=name, args=args)
        self.update_abbr("distance", "dist")


class DistanceMeasure:
    _SIGNATURE_TYPE = DistanceMeasureSignature

    def __init__(self, name: str) -> None:
        self.name = name

    def get_distance(
        self, hyp_data: np.ma.MaskedArray, ref_data: np.ma.MaskedArray
    ) -> float:
        raise NotImplementedError

    def __call__(
        self, hyp_data: np.ma.MaskedArray, ref_data: np.ma.MaskedArray
    ) -> float:
        return self.get_distance(hyp_data, ref_data)

    def get_signature(self) -> Signature:
        return self._SIGNATURE_TYPE(self.name, self.__dict__)


class PowerDistanceSignature(DistanceMeasureSignature):
    def __init__(self, name, args: dict):
        super().__init__(name=name, args=args)
        self.update_signature_and_abbr("order", "ord", args)
        self.update_signature_and_abbr("default_distance", "dflt", args)
        self.update_signature_and_abbr("aggregation_strategy", "agg", args)


class AggregatedPowerDistance(DistanceMeasure):
    _SIGNATURE_TYPE = PowerDistanceSignature

    def __init__(
        self,
        order: int = 2,
        default_distance=0,
        aggregation_strategy = "mean"
    ) -> None:
        super().__init__(name="power_distance")
        self.power = order
        self.default_distance = default_distance
        self.aggregation_strategy = aggregation_strategy

    def aggregate(self, distances: ma.MaskedArray)->float:
        if self.aggregation_strategy == "mean":
            return distances.mean()
        if self.aggregation_strategy == "max":
            return distances.max()
        if self.aggregation_strategy == "min":
            return distances.min()
        if self.aggregation_strategy == "sum":
            return distances.sum()
        
        raise NotImplementedError(f"Aggregation Strategy {self.aggregation_strategy} not implemented")
    
    def _calculate_distances(self, hyp_data: ma.MaskedArray, ref_data: ma.MaskedArray):

        diffs = ma.abs(hyp_data - ref_data) # elementwise, for example if 3D the last dim is still 3, e.g. (2, 2, 2)
        raised_to_power = ma.power(diffs, self.power)  # (2, 2, 2) becomes (2**power, 2**power, 2**power), for example (4, 4, 4)
        summed_results = ma.sum(raised_to_power, axis=-1, keepdims=True) # (4, 4, 4) becomes (12). If we had (30 frames, 1 person, 137 keypoints, xyz), now we have just (30, 1, 137, 1)
        roots = ma.power(summed_results, 1/self.power)


        
        filled_with_defaults = ma.filled(roots, self.default_distance)
        # distances = ma.linalg.norm(diffs, ord=self.power, axis=-1)
        return filled_with_defaults


    def get_distance(
        self, hyp_data: ma.MaskedArray, ref_data: ma.MaskedArray
    ) -> float:        
        return self.aggregate(self._calculate_distances(hyp_data, ref_data))
