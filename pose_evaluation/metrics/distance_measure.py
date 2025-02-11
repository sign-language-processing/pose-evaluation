import numpy as np
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
        self.update_signature_and_abbr("power", "pow", args)
        self.update_signature_and_abbr("default_distance", "dflt", args)


class PowerDistance(DistanceMeasure):
    _SIGNATURE_TYPE = PowerDistanceSignature

    def __init__(
        self, name: str = "power_distance", power: int = 2, default_distance=0
    ) -> None:
        super().__init__(name)
        self.power = power
        self.default_distance = default_distance

    def get_distance(
        self, hyp_data: np.ma.MaskedArray, ref_data: np.ma.MaskedArray
    ) -> float:
        return (
            (hyp_data - ref_data)
            .pow(self.power)
            .abs()
            .filled(self.default_distance)
            .mean()
        )
