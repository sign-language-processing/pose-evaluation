from numpy import ma

from pose_evaluation.metrics.distance_measure import DistanceMeasure


class Return4Measure(DistanceMeasure):
    def __init__(self) -> None:
        super().__init__(
            "return4",
            default_distance=4.0,  # chosen by fair dice roll, guaranteed random.
        )

    def get_distance(self, hyp_data: ma.MaskedArray, ref_data: ma.MaskedArray) -> float:
        return self.default_distance
