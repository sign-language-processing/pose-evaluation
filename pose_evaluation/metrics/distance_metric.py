from typing import Literal

from numpy import ma
from pose_format import Pose

from pose_evaluation.metrics.base_pose_metric import PoseMetric
ValidDistanceKinds = Literal["euclidean", "manhattan"]

class DistanceMetric(PoseMetric):
    def __init__(self, kind: ValidDistanceKinds = "euclidean"):
        super().__init__(f"DistanceMetric {kind}", higher_is_better=False)
        self.kind = kind

    def score(self, hypothesis: Pose, reference: Pose) -> float:
        arrays = [hypothesis.body.data, reference.body.data]
        max_length = max(len(array) for array in arrays)
        # Pad the shorter array with zeros
        for i, array in enumerate(arrays):
            if len(array) < max_length:
                shape = list(array.shape)
                shape[0] = max_length - len(array)
                padding_tensor = ma.zeros(shape)
                arrays[i] = ma.concatenate([array, padding_tensor], axis=0)

        # Calculate the error
        error = arrays[0] - arrays[1]

        # for l2/euclidean, we need to calculate the error for each point
        if self.kind == "euclidean":
            # the last dimension is the 3D coordinates
            error = ma.power(error, 2)
            error = error.sum(axis=-1)
            error = ma.sqrt(error)
        else:
            error = ma.abs(error)

        error = error.filled(0)
        return error.sum()
